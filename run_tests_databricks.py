"""
Run insurance-distill tests on Databricks serverless compute.
"""
import os
import base64
import time

env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_DIR = "/Workspace/insurance-distill-ci"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_dir(remote_path: str):
    try:
        w.workspace.mkdirs(path=remote_path)
    except Exception:
        pass


def upload_file(local_path: str, remote_path: str):
    with open(local_path, "rb") as f:
        content = f.read()
    encoded = base64.b64encode(content).decode()
    parent = "/".join(remote_path.split("/")[:-1])
    ensure_dir(parent)
    try:
        w.workspace.delete(remote_path)
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        overwrite=True,
        format=ImportFormat.AUTO,
    )


def upload_dir(local_dir: str, remote_dir: str, extensions=(".py", ".toml")):
    ensure_dir(remote_dir)
    for root, dirs, files in os.walk(local_dir):
        dirs[:] = [d for d in dirs if d not in (
            "__pycache__", ".git", "dist", ".eggs", "notebooks",
        ) and not d.endswith(".egg-info") and d != ".venv"]
        for fname in files:
            if any(fname.endswith(ext) for ext in extensions):
                local_path = os.path.join(root, fname)
                rel = os.path.relpath(local_path, local_dir)
                remote_path = f"{remote_dir}/{rel.replace(os.sep, '/')}"
                print(f"  Uploading {rel} -> {remote_path}")
                upload_file(local_path, remote_path)


print("Uploading project to Databricks workspace...")
upload_dir(PROJECT_ROOT, WORKSPACE_DIR)


# Notebook: use dbutils.notebook.exit() to surface test output
NOTEBOOK_CONTENT = r'''
import subprocess, sys, os, shutil

WORKSPACE_SRC = "/Workspace/insurance-distill-ci"
TMP_DIR = "/tmp/insurance-distill-ci"

# Copy from read-only workspace to writable /tmp
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
shutil.copytree(WORKSPACE_SRC, TMP_DIR)
os.chdir(TMP_DIR)

# Add src/ to sys.path
src_path = os.path.join(TMP_DIR, "src")
env = os.environ.copy()
env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")

# Install runtime dependencies
deps = ["pytest>=7", "scikit-learn>=1.3", "glum>=2.0", "polars>=0.20", "numpy>=1.24"]
r_deps = subprocess.run(
    [sys.executable, "-m", "pip", "install"] + deps + ["--quiet"],
    capture_output=True, text=True, cwd=TMP_DIR
)
if r_deps.returncode != 0:
    dbutils.notebook.exit("INSTALL_FAILED\n" + r_deps.stderr[-2000:])

# Run pytest, write output to a file to capture it
log_file = "/tmp/pytest_output.txt"
r3 = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short",
     "-p", "no:cacheprovider"],
    stdout=open(log_file, "w"),
    stderr=subprocess.STDOUT,
    cwd=TMP_DIR,
    env=env
)

with open(log_file) as f:
    output = f.read()

# Surface the first 5000 chars via notebook exit value
summary = output[-5000:]  # tail of pytest output
status = "PASSED" if r3.returncode == 0 else "FAILED"
dbutils.notebook.exit(f"{status}\n{summary}")
'''

encoded_nb = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
nb_path = f"{WORKSPACE_DIR}/_run_tests"
ensure_dir(WORKSPACE_DIR)
try:
    w.workspace.delete(nb_path)
except Exception:
    pass
w.workspace.import_(
    path=nb_path,
    content=encoded_nb,
    overwrite=True,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
)
print(f"Test notebook created at {nb_path}")

print("Submitting serverless test run...")

ENV_KEY = "distill-test-env"

run_waiter = w.jobs.submit(
    run_name="insurance-distill-tests",
    environments=[
        jobs.JobEnvironment(
            environment_key=ENV_KEY,
            spec=compute.Environment(client="2"),
        )
    ],
    tasks=[
        jobs.SubmitTask(
            task_key="run-tests",
            environment_key=ENV_KEY,
            notebook_task=jobs.NotebookTask(
                notebook_path=nb_path,
            ),
        )
    ],
)

run_id = run_waiter.run_id
print(f"Run ID: {run_id}")
print("Waiting for run...")

while True:
    state = w.jobs.get_run(run_id=run_id)
    life = state.state.life_cycle_state
    life_str = life.value if hasattr(life, "value") else str(life)
    print(f"  State: {life_str}")
    if any(s in life_str for s in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR")):
        break
    time.sleep(15)

result_state = state.state.result_state
print(f"\nResult: {result_state}")

# Fetch per-task output
output_text = ""
if state.tasks:
    for task in state.tasks:
        if task.attempt_number == 0:
            try:
                out = w.jobs.get_run_output(run_id=task.run_id)
                if out.notebook_output and out.notebook_output.result:
                    output_text = out.notebook_output.result
                    print("\n--- Test output ---")
                    print(output_text)
                if out.error:
                    print("\n--- Error ---")
                    print(out.error)
                if out.error_trace:
                    print("\n--- Error trace ---")
                    print(out.error_trace[-3000:])
            except Exception as e:
                print(f"Could not fetch task output: {e}")

result_str = str(result_state)
if "SUCCESS" not in result_str or "FAILED" in output_text:
    raise SystemExit(f"Test run failed: {result_state}")

print("\nTests completed successfully on Databricks.")

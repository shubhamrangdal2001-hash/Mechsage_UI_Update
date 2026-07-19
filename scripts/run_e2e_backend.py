"""Start an isolated MechSage backend for Playwright."""

from pathlib import Path
import os
import sys
import warnings

import uvicorn


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    (repo_root / ".pytest_tmp").mkdir(exist_ok=True)
    port = int(os.getenv("E2E_BACKEND_PORT", "8010"))
    uvicorn.run("backend.main:app", host="127.0.0.1", port=port, log_level="warning")

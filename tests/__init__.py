import sys
from pathlib import Path

# Ensure repository root has precedence on sys.path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

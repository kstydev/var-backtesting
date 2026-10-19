import sys
from pathlib import Path

# Make `config` and `src` importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

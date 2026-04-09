import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import scripts.czech_das_to_dasnet` works
sys.path.insert(0, str(Path(__file__).parent.parent))

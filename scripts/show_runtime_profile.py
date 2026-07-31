from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime_profile import apply_runtime_environment


def main() -> None:
    profile = apply_runtime_environment()
    print(json.dumps(profile.to_dict(), indent=2))


if __name__ == '__main__':
    main()
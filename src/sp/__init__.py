

from pathlib import Path
import sys


src_base = Path(__file__).parent  # Ensure the current directory is set correctly
root_base = Path(__file__).parent.parent.parent  # Adjust as needed for the root directory
# sys.path.append(str(src_base))


__all__ = ["src_base", "root_base"]

# Created by j.sha on 27.01.2023

import os
from pathlib import Path

root = Path(__file__).parents[1]


buffer_path = root / "src" / "runs" / "buffer"
if __name__ == "__main__":
    print("root path: " + str(root))
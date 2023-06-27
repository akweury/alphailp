# Created by jing at 26.06.23

import os
from pathlib import Path

root = Path(__file__).parents[2]
buffer_path = root / ".." / "storage"
if not os.path.exists(buffer_path):
    os.mkdir(buffer_path)
data_path = buffer_path / "featureMapExplainer"
if not os.path.exists(data_path):
    os.mkdir(data_path)
exp_path = root / "data" / "featureMapExplainer" / "patterns"

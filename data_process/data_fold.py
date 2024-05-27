import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import datasets
from sklearn import model_selection

root = Path("/mnt/mountA/cwy/pointcloud/superpoint_transformer-master/data/grss/raw")
path = root / "data.txt"
cloud = np.loadtxt(path)
CLASS_NUM = 20
kf = model_selection.StratifiedKFold(n_splits=CLASS_NUM)

import laspy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from block import Pc_Img_Map,block_points
import tifffile

pc_root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\pc")
pc_path = [x for x in pc_root.glob("*.las")]
hs_root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\hs")
hs_path = hs_root/"HSI3.tif"

data = laspy.read(pc_path[0]).xyz
for i in pc_path[1:]:
    temp = laspy.read(i)
    data = np.vstack((data, temp[temp.return_number == 1].xyz))

points = np.array(data)

img = tifffile.imread(hs_path)[:,:,:48]
img.shape[1]/4
cuted_img = img[:,:int(img.shape[1]/4),:]
cuted_img_flatten = cuted_img.reshape((-1, cuted_img.shape[-1]))

map = Pc_Img_Map(points)
bp = block_points(4, 1)
block_indices = bp.block_split(map)

hs_feat = np.zeros(block_indices[0].__len__())
points_0=points[block_indices[0]]

map = Pc_Img_Map(points_0)
bp = block_points(cuted_img.shape[1], cuted_img.shape[0])
block_indices = bp.block_split(map)

for i,block in enumerate(block_indices):
    hs_feat[block] = cuted_img_flatten[i,-1]

plt.figure(figsize=(10, 8))
plt.scatter(points_0[:, 0], points_0[:, 1], c=hs_feat, cmap='gray', s=1)  # 用灰度颜色图绘制点云
plt.colorbar(label='elevation (m)')
plt.title('Point Cloud Data')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.show()
pass
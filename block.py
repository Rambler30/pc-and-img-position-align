from pathlib import Path
import numpy as np
import tifffile
import laspy
import torch
import open3d as o3d
import cv2
RGB_lay = [20, 13, 8]
CIR_lay = [34, 20, 13]
MS_lay = [13, 32, 47]
R1_pix = [27,75]
R2_pix = [10,75]
__all__=['block_points','Pc_Img_Map']

class Pc_Img_Map:
    def __init__(self, points) -> None:
        self.points = points
    
    def pc_img_indices(self, img):
        x_min, y_min = self.points[:, 0].min(), self.points[:, 1].min()
        x_max, y_max= self.points[:, 0].max(), self.points[:, 1].max()
        points = torch.tensor(self.points[:,:3], dtype=torch.float32).cuda()
        origin = torch.tensor([x_min, y_min], dtype=points.dtype, device=points.device)
        img_range, img_size = self.count_indices(img, x_min, y_min, x_max, y_max)

        img_coord = ((points[:, :2] - origin) /  img_range * img_size).to(torch.int)
        img_coord[:,0] = torch.clamp(img_coord[:,0], min=0, max=img.shape[1]-1)
        # 图像原点是左下角
        img_coord[:,1] = torch.clamp(img_coord[:,1], min=0, max=img.shape[0]-1)
        img_coord[:,1] = img_coord[:,1].max() - img_coord[:,1]

        assert img_coord[:,0].max() < img.shape[1] and img_coord[:,1].max() < img.shape[0]
        return np.array(img_coord.cpu())

    def count_indices(self, img, x_min, y_min, x_max, y_max):
        img_hight, img_width = img.shape[0], img.shape[1]
        x_range, y_range = x_max - x_min, y_max - y_min
        range = torch.tensor([x_range, y_range], dtype=torch.float, device="cuda:0")
        size = torch.tensor([img_width, img_hight], dtype=torch.float, device="cuda:0")
        return range, size   

class block_points:
    def __init__(self, block_width, block_hight) -> None:
        super().__init__()
        self.block_width = block_width
        self.block_hight = block_hight
        self.img = np.zeros((block_hight, block_width))
    
    def block_split(self, map: Pc_Img_Map):
        data_coord = map.pc_img_indices(self.img)
        data_indices = data_coord[:,0] + data_coord[:,1]*self.block_width
        block_indices = [[] for _ in range(self.block_width*self.block_hight)]
        for i in range(len(data_indices)):
            block_indices[data_indices[i]].append(i)
        return block_indices
    
def vis_pc(points, color =None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color[:,:3])
    o3d.visualization.draw_geometries([pcd])

def hs_point_vis(points, hs_feat):
    max_values = np.max(hs_feat[:,RGB_lay], axis=(0, 1))
    normal_hs = hs_feat[:,RGB_lay] / max_values
    vis_pc(points, normal_hs)

def label(points):
    root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\label")
    label_path = root / "2018_IEEE_GRSS_DFC_GT_TR.tif"
    label_img = tifffile.imread(label_path)
    label_img_flatten = label_img.reshape((-1, 1))

    # 赋予label
    map = Pc_Img_Map(points)
    bp = block_points(label_img.shape[1], label_img.shape[0])
    block_indices = bp.block_split(map)
    label = np.zeros((points.shape[0], 1))
    for i, block in enumerate(block_indices):
        label[block] = label_img_flatten[i]
    return label

def get_each_class_random_sets(img, label, mini_class_size):
    img_flatten = img.reshape((-1, img.shape[-1]))
    label_flatten = label.reshape((-1, 1))
    random_sets = []
    for i in range(np.unique(label_flatten).shape[0]):
        label_indices = np.argwhere(label_flatten==i)[:, 0]
        random_selection = np.random.choice(label_indices, size=mini_class_size, replace=False)
        temp_img = img_flatten[random_selection]
        temp_label = label_flatten[random_selection]
        random_sets.append(np.concatenate((temp_img, temp_label), axis=1))
    return random_sets

def selected_band(img, label):
    img = img[~np.where(label==0)[0]]
    label = label[~np.where(label==0)[0]]
    mini_class_size = minisize(label)
    random_sets = get_each_class_random_sets(img, label, mini_class_size)
    roi_sets = [roi_set[:, :-1] for roi_set in random_sets]
    roi_label = [roi_set[:, -1] for roi_set in random_sets]
    random_mean_set = np.mean(roi_sets, axis=1)
    full = len(roi_sets)
    USBs = compute_diff(random_mean_set, full).astype(int)
    num_bands = USBs.shape[0]
    selected_img = img[:,USBs]
    return USBs, selected_img


def minisize(label):
    label_flatten = label.reshape((-1,1))
    class_set = np.unique(label_flatten)
    min_size = label_flatten.shape[0]
    for i, classes in enumerate(class_set):
        class_size = np.where(label_flatten == classes)[0].shape[0]
        if min_size > class_size:
            min_size = class_size
    return min_size

def compute_diff(data, full):
    class_diff = np.zeros((full, full, 3))
    for i in range(full):
        for j in range(full):
            if i==j:
                class_diff[i, j, :] = [-1,-1,-1]
            if i != j:
                d = np.abs(data[i] - data[j])
                class_diff[i, j, :] = np.argsort(d)[-3:]

    uniqueBands = []
    for i in range(full):
        if i==0:
            continue
        temp = class_diff[i, :, :].flatten()
        uniqueBands.append(np.unique(temp))

    USBs = np.unique(np.concatenate(uniqueBands))
    USBs = USBs[USBs!=-1]
    return USBs

if __name__ == '__main__':
    #  path
    pc_root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\pc")
    pc_path = [x for x in pc_root.glob("*.las")]
    hs_root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\hs")
    hs_path = hs_root / "hs_label.tif"

    # read
    data = []
    for i, path in enumerate(pc_path):
        las = laspy.read(path)
        data.append(las[las.points['return_num']==1].xyz)
    points = np.concatenate(data, axis=0)

    hs = tifffile.imread(hs_path)
    s_img = hs[:,:,:48]
    label = hs[:,:,-1]
    s_img_flatten = s_img.reshape((-1, s_img.shape[-1]))
    label_flatten = label.reshape((-1, 1))
    usbs, selected_img = selected_band(s_img_flatten, label_flatten)
    s_img_flatten = s_img_flatten[:,usbs]
    
    # process
    # 赋色
    map = Pc_Img_Map(points)
    bp = block_points(hs.shape[1], hs.shape[0])
    block_indices = bp.block_split(map)
    hs_feat = np.zeros((points.shape[0], s_img_flatten.shape[-1]))
    label_ = np.zeros((points.shape[0]))
    for i, block in enumerate(block_indices):
        hs_feat[block] = s_img_flatten[i,:]
        label_[block] = label_flatten[i]

    # label = label(points)
    assert points.shape[0] == label_.shape[0] == hs_feat.shape[0]

    cloud = np.concatenate((points, hs_feat, label_.reshape(-1,1)), axis=1)

    # 分块
    hs_map = Pc_Img_Map(points)
    hs_bp = block_points(1, 5)
    hs_block_indices = hs_bp.block_split(hs_map)
    data_list = [[] for _ in range(1*5)]
    for i, block in enumerate(hs_block_indices):
        data_list[i].append(cloud[block])
    
    test = data_list[0][0]
    train = data_list[1][0]
    for i in range(2,5):
        train = np.concatenate((train, data_list[i][0]), axis=0)

    root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\data")
    path = root / "train.txt"
    np.savetxt(path, train)

    root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\data")
    path = root / "test.txt"
    np.savetxt(path, test)
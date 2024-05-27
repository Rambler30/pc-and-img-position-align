import numpy as np
import torch
from typing import Any
from data_utils import *
from data_import import DataImporter

__all__=['Pc_Img_Map','Img_Label_Pc']

class Pc_Img_Map:
    def __init__(self, points) -> None:
        self.points = points
    
    def pc_img_indices(self, img):
        x_min, y_min = self.points[:, 0].min(), self.points[:, 1].min()
        x_max, y_max= self.points[:, 0].max(), self.points[:, 1].max()
        points = torch.tensor(self.points[:,:3], dtype=torch.float32).cuda()
        origin = torch.tensor([x_min, y_min], dtype=points.dtype, device=points.device)
        img_range, img_size = self.count_indices(img, x_min, y_min, x_max, y_max)

        img_indices = ((points[:, :2] - origin) /  img_range * img_size).to(torch.int)
        img_indices[:,0] = torch.clamp(img_indices[:,0], min=0, max=img.shape[1]-1)
        img_indices[:,1] = torch.clamp(img_indices[:,1], min=0, max=img.shape[0]-1)
        assert img_indices[:,0].max() < img.shape[1] and img_indices[:,1].max() < img.shape[0]
        return np.array(img_indices.cpu())

    def count_indices(self, img, x_min, y_min, x_max, y_max):
        img_hight, img_width = img.shape[0], img.shape[1]
        x_range, y_range = x_max - x_min, y_max - y_min
        range = torch.tensor([x_range, y_range], dtype=torch.float, device="cuda:0")
        size = torch.tensor([img_width, img_hight], dtype=torch.float, device="cuda:0")
        return range, size   

class Img_Label_Pc(Pc_Img_Map):
    def __init__(self, labels) -> None:
        super().__init__()
        self.labels = labels

    def fusion(self):
        img_indices = self.pc_img_indices(self.img)
        label_indices = self.pc_img_indices(self.labels)
        feature, labels = self.img_label_pc_map(img_indices, label_indices)
        fusion_points = self.concat(self.points, feature, labels)
        return fusion_points

    def img_label_pc_map(self, img_indices, label_indices):
        feature_arr = np.zeros((self.points.shape[0], self.img.shape[2]))
        labels_arr = np.zeros(self.points.shape[0])
        feature_arr[:] = self.img[img_indices[:,1],img_indices[:,0]]
        labels_arr = self.labels[label_indices[:,1],label_indices[:,0]].reshape((-1,1))
        return feature_arr, labels_arr

class block_points:
    def __init__(self, block_width, block_hight) -> None:
        super().__init__()
        self.block_width = block_width
        self.block_hight = block_hight
        self.img = np.zeros((block_hight, block_width))
    
    def block_split(self, map: Pc_Img_Map):
        block_indices = map.pc_img_indices(self.img)
        block_index = block_indices[:,0] + block_indices[:,1]*self.block_width
        block_list = [[] for _ in range(self.block_width*self.block_hight)]
        for i in range(len(block_indices)):
            block_list[block_index[i]].append(i)
        return block_list

def block_split():
    root = Path("/mnt/mountA/cwy/pointcloud/scene_seg/dataset/GDUT/lable")
    data_path = root/"university.txt"
    label_path = root/"label.txt"
    cloud = np.loadtxt(data_path)
    label = np.loadtxt(label_path)
    points = np.concate((cloud, label), 1)
    map = Pc_Img_Map(points)
    bp = block_points(1, 10)
    block_list = bp.block_split(map)
    for i, block_index in enumerate(block_list):
        point = points[block_index]
        path = Path("/mnt/mountA/cwy/pointcloud/superpoint_transformer-master/data/gdut/raw/split_{}.txt".format(i+6))
        np.savetxt(path, point)

def block_module(points, width, hight, path):
    img = np.zeros((width,hight))
    map = Pc_Img_Map(points, img)
    block = block_points(1,5)
    block_list = block.block_split(map)
    List_to_txt.write(path, block_list)

def main():
    data_loader = DataImporter()
    root = Path("/mnt/mountA/cwy/pointcloud/scene_seg/dataset/GRSS/2018IEEE_Contest/Phase2/TrainingGT")
    data_path = root / "data" / "pc_hs_label.txt"
    path = root / "idxs" / "hs_idxs.txt"
    data_loader.import_data([data_path], ["int"])
    data = data_loader.get_data()
    points = np.array(data[0])
    block_module(points, 32, 8, path)


def minidata():
    path = Path("/mnt/mountA/cwy/pointcloud/superpoint_transformer-master/data/grss/raw/data.txt")
    points = np.loadtxt(path)
    lenth = points.shape[0]
    sample_len = int(lenth*0.5)
    point = points[:sample_len, :]
    save_path = Path("/mnt/mountA/cwy/pointcloud/superpoint_transformer-master/data/grss/raw/mini.txt")
    np.savetxt(save_path, point)


if __name__ == '__main__':
    block_split()
    main()
    root = Path("/mnt/mountA/cwy/pointcloud/scene_seg/dataset/GRSS/2018IEEE_Contest/Phase2/TrainingGT")
    img_root, labels_root, points_root = root / "hs", root / "label", root / "pc"
    img_path, labels_path = img_root /"hs.tif", labels_root / "2018_IEEE_GRSS_DFC_GT_TR.tif"
    save_root = root / "data"
    save_path = save_root / "pc_hs_label.txt"

    path_list = []
    path_list = [x for x in points_root.glob("*.txt")]
    path_list.append(img_path)
    path_list.append(labels_path)

    dtype = ["int" for x in range(len(path_list))]
    dtype.append("float")
    dtype.append("float")

    data_import = DataImporter()
    data_import.import_data(path_list, dtype)
    data_list = data_import.get_data()

    cloud = np.concatenate(data_list[:4], 0)
    img = np.array(data_list[4])
    labels = np.array(data_list[5])
    img_label_pc = Img_Label_Pc(cloud, img[:,:,:10], labels)
    points = img_label_pc.fusion()
    Np_to_txt.write(path, points)
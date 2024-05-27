import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
import tifffile
import laspy

class Data_Vis:
    @staticmethod
    def colors_pc_vis(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(points[:,3:6]/255.)
        o3d.visualization.draw_geometries([pcd])

    @staticmethod
    def gray_pc_vis(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])

    @staticmethod
    def total_vis(pcd_list):
        o3d.visualization.draw_geometries(pcd_list)
    
    @staticmethod
    def img_vis(img):
        plt.imshow(img, cmap='gray')  # 使用灰度颜色映射
        plt.colorbar()  # 显示颜色条
        plt.show()

class Img_Load:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load(path):
        return tifffile.imread(path)
    
class Points_Load:
    def __init__(self) -> None:
        pass

    def load(path) -> np:
        raise NotImplementedError

class Load_Points_From_txt(Points_Load): 
    def __init__(self) -> None:
        super().__init__()

    def load(self, path):
        return Np_to_txt.read(path)
    
    def load_from_pathlist(self, list: list):
        data_list = []
        for path in list:
            data_list.append(self.load(path))
        points = np.concatenate(data_list, 0)
        return points

class Load_Points_From_Las(Points_Load): 
    def __init__(self) -> None:
        super().__init__()

    def load(path):
        cloud = laspy.read(path)
        points = cloud.xyz
        return points

class vis_total_block(Data_Vis):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def vis_total_points(block_list, points):
        pcd_combined = []
        for i in range(len(block_list)):
            exec(f"pcd{i + 1} = o3d.geometry.PointCloud()")
            exec(f"pcd{i + 1}.points = o3d.utility.Vector3dVector(points[block_list[i]][:,:3])")
            exec(f"pcd_combined.append(pcd{i + 1})")
        Data_Vis.total_vis(pcd_combined)

class File_Read_Write:
    def read(self):
        raise NotImplementedError
    
    def write(self):
        raise NotImplementedError

class Np_to_txt(File_Read_Write):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def read(path):
        return np.loadtxt(path)
    
    @staticmethod
    def write(path, data):
        np.savetxt(path, data)

class List_to_txt(File_Read_Write):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def read(path):
        with open(path, "r") as file:
            lines = file.readlines()
        data_list = []
        for line in lines:
            values = line.strip().split()  # 使用适当的分隔符拆分数据
            data = [int(value) for value in values]
            data_list.append(data)
        return data_list
    
    @staticmethod
    def write(path, list):
        with open(path, "w") as file:
            for row in list:
                line = " ".join(map(str, row))
                file.write(line + "\n")

class Get_File_Path:
    def root2path(self):
        raise NotImplementedError
    def path_list(self):
        raise NotImplementedError

class Grss_Get_Path(Get_File_Path):
    def __init__(self, root, target, is_mult=False) -> None:
        super().__init__()
        self.is_mult = is_mult
        self.root = Path(root)
        self.target = Path(target)

    def root2path(self):
        return self.root / self.target
    
    def path_list(self):
        return [x for x in self.root.glob("*.{}".format(self.target)) if x.is_file()]

def random_block(lenth, k, choiced_list):
    list = []
    while len(list) < k :
        choice_num = random.choice(range(lenth))
        if choice_num not in choiced_list:
            list.append(int(choice_num))
    return list

def block_index(points_index, list):
    return [points_index[i] for i in list]

if __name__ == '__main__':
    root = Path("/mnt/mountA/cwy/pointcloud/scene_seg/dataset/GRSS/2018IEEE_Contest/Phase2/TrainingGT")
    index_root = root
    get_path = Grss_Get_Path(index_root, "test2.txt")
    path = get_path.root2path()

    points_index = List_to_txt.read(path)
    lenth = len(points_index)
    sample_rate = 0.1
    sample_num = int(lenth*sample_rate)
    val_list = random_block(lenth, sample_num, [])
    test_list = random_block(lenth, sample_num, val_list)
    train_list = random_block(lenth, lenth-sample_num*2, val_list+test_list)
    test_block_index = block_index(points_index, test_list)
    val_block_index = block_index(points_index, val_list)
    train_block_index = block_index(points_index, train_list)

    idxs_root = root / "idxs"
    test_list_path = idxs_root / "test_index.txt"
    val_list_path = idxs_root / "val_index.txt"
    train_list_path = idxs_root / "train_index.txt"
    
    points = Np_to_txt.read("/mnt/mountA/cwy/pointcloud/scene_seg/dataset/GRSS/2018IEEE_Contest/Phase2/TrainingGT/data/data.txt")
    vis_total_block.vis_total_points(test_block_index , points)
    List_to_txt.write(test_list_path, test_block_index)
    List_to_txt.write(val_list_path, val_block_index)
    List_to_txt.write(train_list_path, train_block_index)

    print("breakpoint")

    """
    1. 读数据，从txt里面读取list、np，从img里面读数据
    2. 写数据，将数据写到txt
    """
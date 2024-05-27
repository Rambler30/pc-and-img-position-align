import numpy as np
import tifffile
import laspy
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import multiprocessing
from itertools import repeat
import os
import re
import subprocess

# 路径

root = Path("/mnt/mountA/cwy/pointcloud/scene_seg/dataset/GRSS/2018IEEE_Contest/Phase2/TrainingGT")
cloud_root = root / "pc"
cloud_path_list = [x for x in cloud_root.glob("*.las") if x.is_file()]
img_root = root / "rgb"
img_path_list = [x for x in img_root.glob("*.tif") if x.is_file()]
label_root = root / "label"
label_path_list = list(label_root.glob("*.tif"))
test_root = root / "raw"
test_path_list = [x for x in test_root.glob("*.txt")]
import open3d as o3d
# label = tifffile.imread(label_path_list[0])

def test_module(path):
    
    points = np.loadtxt(path)

    color = np.zeros((points.shape[0], 3))
    for i in range(points.shape[0]):
        if points[i, -1] == 0:
            continue
        else:
            color[i] = [255,255,255]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(color[:]/255)
    o3d.visualization.draw_geometries([pcd])

def test_module2(data):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(data[:,3:6]/255.0)
    o3d.visualization.draw_geometries([pcd])


# test_module(test_path_list[0])
def read_data_single():
    # 读取数据
    assert len(cloud_path_list) == len(img_path_list)
    start_time = time.time()
    data_list = []
    # label
    label = tifffile.imread(label_path_list[0])
    label_row_step, label_col_step = 0.5, 0.5
    label_width, label_hight = label.shape[0], label.shape[1]
    label_origin_world_row, label_origin_world_col = 272056.0, 3289689.0
    label_max_x, label_max_y = label_origin_world_row + label_width * label_row_step, label_origin_world_col + label_hight * label_col_step

    save_to_root = root / "raw"
    import cv2
    import matplotlib.pyplot as plt
    row_step, col_step = 0.05, 0.05
    for i in range(len(img_path_list)):
        points = laspy.read(cloud_path_list[i])
        feature = np.zeros((points.xyz.shape[0], 4))
        img = tifffile.imread(img_path_list[i])
        # img[112:172, 110:170] = [0,0,0]
        # plt.imshow(img)
        # plt.axis('off')  # 可以关闭坐标轴
        # plt.show()
        origin_world_row, origin_world_col = map(float, cloud_path_list[i].stem.split("_"))
        img_width, img_hight = img.shape[0], img.shape[1]
        max_x, max_y = origin_world_row + img_width * row_step, origin_world_col + img_hight * col_step
        for j, point in enumerate(points.xyz):
            x_indices, y_indices = map(int, (np.array((point[:2] - np.array([origin_world_row, origin_world_col])) / 
                                np.array([max_x - origin_world_row, max_y - origin_world_col])) * np.array([img_width, img_hight])))
            x_label_indices, y_label_indices = map(int, (np.array((point[:2] - np.array([label_origin_world_row, label_origin_world_col]))) / 
                                                np.array([label_max_x - label_origin_world_row, label_max_y - label_origin_world_col])) *
                                                np.array([label_width, label_hight]))
            feature[j, :3] = img[img_hight - y_indices, x_indices]
            feature[j, -1] = label[label_hight - y_label_indices, x_label_indices]
        data = np.concatenate([points.xyz, feature], 1)
        # test_module2(data)
        np.savetxt(save_to_root / "Block_{}.txt".format(i), data)
        data_list.append([data])
        # data_list.append(data)


    runtime = time.time() - start_time

    print(f"程序运行时间：{runtime:.2f} 秒")


# 加一个多线程的尝试
# def starmap_with_kwargs(fn, args_iter,kwargs_iter, processes=4):
#     # Prepare kwargs
#     # if kwargs_iter is None:
#     #     kwargs_iter = repeat({})
#     # if isinstance(kwargs_iter, dict):
#     #     kwargs_iter = repeat(kwargs_iter)

#     # Apply fn in multiple processes
#     with multiprocessing.get_context("spawn").Pool(processes=processes) as pool:
#         args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
#         out = pool.starmap(apply_args_and_kwargs, args_for_starmap)

#     return out

def starmap_with_kwargs(fn, *args, processes=4):
    # Prepare kwargs
    # if kwargs_iter is None:
    #     kwargs_iter = repeat({})
    # if isinstance(kwargs_iter, dict):
    #     kwargs_iter = repeat(kwargs_iter)

    # Apply fn in multiple processes

    with multiprocessing.get_context("spawn").Pool(processes=processes) as pool:
        args_for_starmap = zip(repeat(fn), args[0], args[1])
        # apply_args_and_kwargs(fn, args[0], args[1])
        out = pool.starmap(apply_args_and_kwargs, args_for_starmap)
    return out

# def apply_args_and_kwargs(fn, args, kwargs):
#     return fn(*args, **kwargs)
def apply_args_and_kwargs(fn, *args):
    return fn(*args)

# def args_test(*args):
#     # args_for_starmap = zip(repeat(my_add), args[0], args[1])
#     out = apply_args_and_kwargs(my_add, args[0], args[1])
#     return out

def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')

import torch
from torch import tensor
row_step, col_step = 0.05, 0.05
label_row_step, label_col_step = 0.5, 0.5

def torch_process_test(points, cloud_path, img, label_origin, label_range, label_size):
    points = torch.tensor(points, dtype=torch.float32).cuda()
    origin_world_row, origin_world_col = map(float, cloud_path.stem.split("_"))
    img_hight, img_width, _= img.shape
    x_max, y_max = origin_world_row + row_step * img_width, \
                    origin_world_col + col_step * img_hight
    x_range, y_range = x_max - origin_world_row, y_max - origin_world_col
    origin = torch.tensor([origin_world_row, origin_world_col], dtype=points.dtype, device=points.device)
    img_range = torch.tensor([x_range, y_range], dtype=points.dtype, device=points.device)
    img_size = torch.tensor([img_width, img_hight], dtype=points.dtype, device=points.device)
    indices = ((points[:, :2] - origin) /  img_range * img_size).to(torch.int)

    label_indices = ((points[:, :2] - label_origin) / label_range * label_size).to(torch.int)
    indices = indices.cpu()
    label_indices = label_indices.cpu()
    return indices, label_indices

def label_indices_process():
    label_origin_world_row, label_origin_world_col=272056.0, 3289689.0
    label_hight, label_width = label.shape[:2]
    label_x_max, label_y_max = label_origin_world_row + label_row_step * label_width, \
                                label_origin_world_col + label_col_step * label_hight
    label_x_range, label_y_range = label_x_max - label_origin_world_row, label_y_max - label_origin_world_col
    label_origin = torch.tensor([label_origin_world_row, label_origin_world_col], dtype=torch.float).cuda()
    label_range = torch.tensor([label_x_range, label_y_range], dtype=torch.float).cuda()
    label_size = torch.tensor([label_width, label_hight], dtype=torch.float).cuda()
    return label_origin, label_range, label_size

def read_data(cloud_path=None, img_path=None):
    points = laspy.read(cloud_path)
    img = tifffile.imread(img_path)
    return points.xyz, img

def color_points(points,cloud_path, img, label_origin, label_range, label_size):
    indices, label_indices = torch_process_test(points,cloud_path, img, label_origin, label_range, label_size)
    colors = np.zeros((points.shape[0], 3))
    labels = np.zeros(points.shape[0])
    y = np.array([img.shape[0] - x if x > 0 else -1 for x in indices[:, 1]])
    x = np.array([x if x < img.shape[1] else -1 for x in indices[:,0]])
    # colors[:] = img[indices[:,1], indices[:,0]]
    label_x = np.array([x if x <label.shape[1] else -1 for x in label_indices[:, 0]])
    label_y = np.array([label.shape[0] - x if x > 0 else -1 for x in label_indices[:, 1]])
    colors[:] = img[y,x]
    labels[:] = label[label_y,label_x]
    colors_points = np.concatenate((points, colors, labels.reshape((labels.shape[0],1))), 1)
    return colors_points

def my_add(a, b):
    return a+b
    
def test():
    a = [1,1,1]
    b = [2,2,2]
    args_iter = [(a[i], b[i]) for i in range(len(a))]
    # freeze_support()
    processes = available_cpu_count() if processes < 1 else processes
    out = starmap_with_kwargs(
            my_add, a, b, processes=processes)
    # out = starmap_with_kwargs(
    #         my_add, args_iter, kwargs, processes=processes)

def out_process(out):
    points = []
    img = []
    for i in range(len(out)):
        points.append(out[i][0])
        img.append(out[i][1])
    points = np.concatenate(points, 0)
    img = np.concatenate(img, 1)
    return points, img

def load_txt(path):
    return np.loadtxt(path, dtype=float)

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
        # 图像原点是左下角
        img_indices[:,1] = torch.clamp(img_indices[:,1], min=0, max=img.shape[0]-1)
        img_indices[:,1] = img_indices[:,1].max() - img_indices[:,1]
        assert img_indices[:,0].max() < img.shape[1] and img_indices[:,1].max() < img.shape[0]
        return np.array(img_indices.cpu())

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
        block_indices = map.pc_img_indices(self.img)
        block_index = block_indices[:,0] + block_indices[:,1]*self.block_width
        block_list = [[] for _ in range(self.block_width*self.block_hight)]
        for i in range(len(block_indices)):
            block_list[block_index[i]].append(i)
        return block_list

def block_process(points):
    origin_row, origin_col = points[:,0].min(), points[:,1].min() 
    hight, width = 8, 8*4
    x_max, y_max = points[:,0].max(), points[:,1].max()
    x_range, y_range = x_max - origin_row, y_max - origin_col
    origin = torch.tensor([origin_row, origin_col], dtype=torch.float, device="cuda:0")
    range = torch.tensor([x_range, y_range], dtype=torch.float, device="cuda:0")
    size = torch.tensor([width, hight], dtype=torch.float, device="cuda:0")
    points = torch.tensor(points[:,:2], dtype=torch.float, device="cuda:0")

    # 得到每个点属于哪个块 (x,y)-(min_x, min_y) / range * size
    indices = torch.floor((points[:, :2] - origin) /  (range+0.00001) * size)

    indices[:,0] = torch.clamp(indices[:,0], min=0, max=32-1)
    indices[:,1] = torch.clamp(indices[:,1], min=0, max=8-1)
    return np.array(indices.cpu())

def vis(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    colors = np.zeros((points.shape[0], 3))
    for i in range(points.shape[0]):
        if points[i,-1]!=0:
            colors[i] = [255.0,255.0,255.0]
    pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
    o3d.visualization.draw_geometries([pcd])
    
def read_list_from_txt(path):
    data_list = [] 
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            cells = line.split("\t") 
            data_list.append(cells)
    return data_list

def write_list2txt(path, list):
    with open(path, "w") as file:
        for row in list:
            line = " ".join(map(str, row))
            file.write(line + "\n")

import time
if __name__ == '__main__':
    # strat_time = time.time()
    # points = torch_process_test(cloud_path_list[0], img_path_list[0])
    # np.savetxt(root/"test.txt", points)
    # print(time.time()-strat_time)
    # processes = -1
    # # read_data_single()
    # # args_iter = [[cloud_path_list[i]] for i in range(len(cloud_path_list))]
    # # kwargs = [[img_path_list[i]] for i in range(len(cloud_path_list))]
    # start_time = time.time()
    # processes = available_cpu_count() if processes < 1 else processes
    # out = starmap_with_kwargs(
    #         read_data, cloud_path_list, img_path_list, processes=processes)
    # points, img = out_process(out)
    # # points, img = out[0][0], out[0][1]
    # label_origin, label_range, label_size = label_indices_process()
    # for i in range(len(out)):
    #     points, img = out[i][0], out[i][1]
    #     points = color_points(points,cloud_path_list[i], img, label_origin, label_range, label_size)
    #     np.savetxt(root/"pc"/"Block_{}.txt".format(i), points)
    # # points = color_points(points, img)
    # # vis(points)
    # print(time.time()-start_time)
    # print("???")
    
    # sorted(np.array([1,1,2, 1]))
    cloud_root = root / "pc"
    cloud_path_list = [x for x in cloud_root.glob("*.txt")]
    clouds = []
    # index_root = root / "index"

    block_index_list = [[] for i in range(8*8*4)]
    for i, path in enumerate(cloud_path_list):
        cloud = load_txt(path)
        clouds.append(cloud)
    points = np.concatenate(clouds, 0)
    indices = block_process(points[:,:3])
    index = np.array(indices[:, 0] + indices[:, 1]*32, dtype=int)
    for i in range(32*8):
        sq_block = np.array(np.where(index==i)).reshape((-1,))
        block_index_list[i] = sq_block
    
    #     indices = block_process(np.array(clouds[i]))
        
    #     index = np.array(indices[:, 0] + indices[:, 1]*8, dtype=int)
    #     for j in range(64):
    #         sq_block = np.array(np.where(index==i)).reshape((-1,))
    #         block_index_list[i+ 64*i] = sq_block

    train_index_path = root / "test.txt"
    index = read_list_from_txt(train_index_path)

    # train_index , test_index, val_index = split_block(block_index_list)



    # for i in range(len(cloud)):
    #     block_index_list[index[i]].append(map_index[i])
    # vis(cloud[block_index_list[0]])
    write_list2txt(train_index_path, block_index_list)

    points_index_root = root / "test"
    points_index_path = points_index_root / "test.txt"
    points_index = np.loadtxt(points_index_path)
    # clouds[]
    
    print("???")

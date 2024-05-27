import numpy as np
import cv2
import laspy
from pathlib import Path
import tifffile
from sklearn.decomposition import PCA

# from scipy.stats import norm

# 自定义概率密度函数
def custom_pdf(x, mean, std):
    exponent = -0.5 * ((x - mean) / std) ** 2
    prob_density = np.exp(exponent) / (std * np.sqrt(2 * np.pi))
    return prob_density

def joint_entropy(gradients):
    H = np.zeros(gradients.shape)
    pdf = np.zeros(gradients.shape)
    for i in range(gradients.shape[0]):
        temp = 0
        for j in range(gradients.shape[1]):
            gradients_mean = np.mean(gradients[i][j])
            # gradients_std = np.std(gradients[i][j])
            pdf[i][j] = norm.pdf(gradients[i][j], gradients_mean)
            # pdf[i][j] = custom_pdf(gradients[i][j], gradients_mean, gradients_std)
        for b in range(gradients.shape[0]):
            H[i,:, b] = pdf[i,:, b]*np.log2(pdf[i,:, b])
    return -H

def SSI(h_mul, h_mean_mul, h_sigma, h_mean_sigma, all_mean_sigma, b):
    k1, k2, k3 = 0,0,0
    ssi = np.zeros((b, 1))
    for i in range(b):
        up = (2*h_mul[i]*h_mean_mul+k1)*(2*all_mean_sigma[i]+k2)
        down = (np.square(h_mul[i])+np.square(h_mean_mul)+k1
                )*(np.square(h_sigma[i])+np.square(h_mean_sigma)+k2)
        ssi[i] = up / down
    return ssi

def structuralsimilarity(h):
    band_mean = np.zeros((h.shape[0], h.shape[1]))
    # for i in range(h.shape[0]):
    #     for j in range(h.shape[1]):
    #         if i==j:
    #             continue
    #         band_mean[i][j] = np.mean(h[i][j])
    band_mean = np.mean(h, axis=2)
    row, col, b = h.shape[0], h.shape[1], h.shape[2]
    h_mul = mul(h, row, col, b)
    h_mean_mul = mean_mul(band_mean, row, col)
    h_sigma = sigma(h, row, col, h_mul, b)
    h_mean_sigma = mean_sigma(band_mean, row, col, h_mean_mul)
    all_sigma = all_mean_sigma(row, col, h, h_mul, band_mean, h_mean_mul, b)

    # l = l(h_mutal, h_mean_mutal)
    # c = c(h_sita, h_mean_sita)
    # s = s(h_sita, h_mean_sita, all_mean_sita)

    ssi = SSI(h_mul, h_mean_mul, h_sigma, h_mean_sigma, all_sigma, b)
    return ssi

def mul(data, row, col, b):
    d = np.zeros((b,1))
    for i in range(b):
        d[i] = np.sum(data[:,:,i])
    return 1/(row*col)*d

def mean_mul(data, row, col):
    d = np.sum(data)
    return 1/(row*col)*d

def sigma(data, row, col, mutal, b):
    d = np.zeros((b,1))
    for i in range(b):
        d[i] = np.sum(np.square(data[:,:,i] - mutal[i]))
    return np.sqrt(1/(row*col-1)*d)

def mean_sigma(data, row, col, mutal):
    d = np.sum(np.square(data - mutal))
    return np.sqrt(1/(row*col-1)*d)

def all_mean_sigma(row, col, data, mutal, band_mean, mean_mutal, b):
    d = np.zeros((b, 1))
    for i in range(b):
        d[i] = np.sum((data[:,:,i] - mutal[i])*(band_mean - mean_mutal))
    return np.sqrt(1/(row*col-1)*d)



def get_point_each_class_random_sets(points):
    label = points[:,-1]
    class_set = np.unique(label)
    mini_class_size = points.shape[0]
    for i, classes in enumerate(class_set):
        if i==0:
            continue
        class_size = np.where(label == classes)[0].shape[0]
        if mini_class_size > class_size:
            mini_class_size = class_size
    random_sets = []
    for i in range(class_set.shape[0]):
        if i==0:
            continue
        label_indices = np.argwhere(label==i)[:, 0]
        random_selection = np.random.choice(label_indices, size=mini_class_size, replace=False)
        class_points = points[random_selection, :]
        random_sets.append(class_points)
    return random_sets
    

# from skimage.metrics import structural_similarity as ssim
# from scipy.stats import entropy
import matplotlib.pyplot as plt

def plot_pca_scatter(x, y):
    estimator = PCA(n_components=2)
    X_pca = estimator.fit_transform(x)
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 
          'red', 'lime', 'cyan', 'orange', 'gray',
          'darkred', 'darkblue', 'darkgreen', 'darkorange', 
          'darkviolet', 'darkcyan', 'darkgray', 'darkmagenta',
          'dodgerblue', 'tomato']
    mins = np.argwhere(y==0).shape[0]

    reduce_data = []
    for i in range(5):
        test_band = x[:,i*6:(i+1)*6]
        pca = PCA(n_components=5)
        pca.fit(test_band)
        reduced_group_data = pca.transform(test_band)
        reduce_data.append(reduced_group_data)
    reduce_data = np.hstack(reduce_data)

    for i in range(len(colors)):
        class_num = np.argwhere(y==i+1).shape[0]
        if class_num < mins:
            mins = class_num
    for i in range(len(colors)):
        px = reduced_group_data[:, 0][np.argwhere(y==i+1)[:class_num]]
        py = reduced_group_data[:, 1][np.argwhere(y==i+1)[:class_num]]
        plt.scatter(px, py, c=colors[i])
    
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

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

# def pac_rebuild(o_data, data):
#     pca = PCA(n_components=48)
#     pca.fit(img)
#     reduced_group_data = pca.transform(img)

def classify(data, label):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def vis_feature(data, y):
    from mpl_toolkits.mplot3d import Axes3D
    # 三维散点图
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 
          'red', 'lime', 'cyan', 'orange', 'gray',
          'darkred', 'darkblue', 'darkgreen', 'darkorange', 
          'darkviolet', 'darkcyan', 'darkgray', 'darkmagenta',
          'dodgerblue', 'tomato']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')   
    pca = PCA(n_components=3)
    pca.fit(data)
    reduced_group_data = pca.transform(data)
    mins = data.shape[0]
    for i in range(len(colors)):
        class_num = np.argwhere(y==i+1).shape[0]
        if class_num < mins:
            mins = class_num
    class_num=100
    for i in range(len(colors)):
        if i==9 or i== 0 or i==8:
            continue
        px = reduced_group_data[:, 0][np.argwhere(y==i+1)[:class_num]].ravel()
        py = reduced_group_data[:, 1][np.argwhere(y==i+1)[:class_num]].ravel()
        pz = reduced_group_data[:, 2][np.argwhere(y==i+1)[:class_num]].ravel()
        ax.scatter(px, py, pz, label='Group_{}'.format(i+1))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()

def reduce_band():
    root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\hs")
    path = root / "hs_label.tif"
    data = tifffile.imread(path)
    data = data.reshape((-1,data.shape[-1]))
    img, label = data[:,:-3], data[:,-1]
    selected_img1 = img[np.where(label!=0)[0], :]
    label1 = label[np.where(label!=0)[0]]
    accuracy = classify(selected_img1, label1)
    num_classes = np.unique(label).shape[0]
    USBs, selected_img = selected_band(img, label)
    selected_img1 = selected_img[np.where(label!=0)[0], :]
    label1 = label[np.where(label!=0)[0]]
    accuracy = classify(selected_img1, label1)
    
    plot_pca_scatter(selected_img,label)
    spectra = [selected_img[np.argwhere(label==i)[:,0]] for i in range(num_classes) if i != 0]

    linestyle = ['solid','dashed', 'dotted', 'dashdot']
    plt.figure(figsize=(10, 6))
    for i in range(num_classes-1):
        if i+1==9 or i==0:
            continue
        plt.plot(range(1, selected_img.shape[-1]+1), spectra[i][0], label=f'Class {i+1}', linestyle=linestyle[i%4])
    plt.legend()
    plt.show()

    selected_data = data[:,USBs]
    ssim_matix = np.zeros((USBs.shape[0], USBs.shape[0]))
    for i in range(USBs.shape[0]):
        for j in range(USBs.shape[0]):
            if i == j:
                continue
            ssim_matix[i][j] = ssim(selected_data[i], selected_data[j], multichannel=True)


    ssim_matrix = np.corrcoef(selected_img.T)  # 使用相关系数作为相似性度量

    # 使用谱聚类算法将波段分组
    num_clusters = 5  # 假设分成 3 组
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
    spectral_clustering.fit(ssim_matrix)
    band_groups = spectral_clustering.labels_

    # vbgmm
    from sklearn.mixture import BayesianGaussianMixture
    vbgmm = BayesianGaussianMixture(n_components=5, covariance_type='full', max_iter=100, random_state=0)
    vbgmm.fit(ssim_matrix)
    cluster_labels = vbgmm.predict(ssim_matrix)


    # 获取每个波段所属的组别
    test_band = selected_img[:,np.argwhere(band_groups==0)[:,0]]
    pca = PCA(n_components=5)
    pca.fit(img)
    reduced_group_data = pca.transform(img)

    reduce_data = []
    for i in range(5):
        if i==3 or i==4:
            reduce_data.append(selected_img[:,i].reshape(-1,1))
            continue 
        test_band = selected_img[:,np.argwhere(band_groups==i)[:,0]]
        pca = PCA(n_components=5)
        pca.fit(test_band)
        reduced_group_data = pca.transform(test_band)
        reduce_data.append(reduced_group_data)
    reduce_data = np.hstack(reduce_data)
    spectra = [reduce_data[np.argwhere(label==i)[:,0]] for i in range(num_classes) if i != 0]
    
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
# 贪心降维
def RFE_reduce(data,y):
    model = LinearRegression()
    rfe = RFE(
    estimator=model,
    n_features_to_select=20)
    rfe.fit(data, y)
    # 使用 RFE 选择的特征进行数据转换
    X_transformed = rfe.transform(data)


def main():
    # data = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    # reduce_band()
    hs_root = Path(r"C:\data\GRSS\2018IEEE_Contest\Phase2\TrainingGT\hs")
    hs_path = hs_root / "hs_label.tif"
    hs = tifffile.imread(hs_path)
    s_img = hs[:,:,:48]
    label = hs[:,:,-1]

    s_img_flatten = s_img.reshape((-1, s_img.shape[-1]))
    label_flatten = label.reshape((-1, 1))

    usbs, selected_img = selected_band(s_img_flatten, label_flatten)

    o_data2 = np.loadtxt(path)
    data2 = np.hstack(o_data2[:,3:-1], o_data2[-1])
    data = np.vstack((data1, data2))
    random_sets = get_point_each_class_random_sets(data)
    roi_sets = [roi_set[:, 3:-1] for roi_set in random_sets]
    random_mean_set = np.mean(roi_sets, axis=1)
    full = len(roi_sets)
    USBs = compute_diff(random_mean_set, full).astype(int)
    
    
    img, label = data[:,:-3], data[:,-1]
    mini_class_size = minisize(label)
    random_sets = get_each_class_random_sets(img, label, mini_class_size)
    roi_sets = [roi_set[:, :-1] for roi_set in random_sets]
    roi_label = [roi_set[:, -1] for roi_set in random_sets]
    random_mean_set = np.mean(roi_sets, axis=1)
    full = len(roi_sets)
    USBs = compute_diff(random_mean_set, full)
    hs_data = img_label[USBs]

    gradients = np.gradient(hs_data)[-1]
    H = joint_entropy(gradients)
    SSI = structuralsimilarity(hs_data)



if __name__ == '__main__':
    main()
import numpy as np
from scipy.io import loadmat
import lattice_regression as mlr
import matplotlib.pyplot as plt


def load_mat(m_path, m_key):
    m_data = loadmat(m_path)
    return m_data[m_key]

# 同色异谱黑和晶格回归的光谱反射率重建实现
def Mine(m_train_refls, m_train_rgb, m_test_rgb, m_A):
    # 训练样本的同色异谱黑
    R = np.dot(np.dot(m_A, np.linalg.pinv(np.dot(m_A.T, m_A))), m_A.T)  # 生成R矩阵
    train_black = np.transpose(m_train_refls.T - np.dot(R, m_train_refls.T))
    # 测试样本的基本刺激光谱
    test_fun_spectrum = np.dot(np.dot(m_A, np.linalg.pinv(np.dot(m_A.T, m_A))), m_test_rgb.T).T

    # 晶格回归重建同色异谱黑
    test_black = mlr.lattice_reg(train_black, m_train_rgb, m_test_rgb, 2)

    m_rec_refls = test_black + test_fun_spectrum  # 组合同色异谱黑和基本刺激光谱

    return m_rec_refls


if __name__ == '__main__':

    # 载入数据
    train_refls = load_mat('Data/train_refls.mat', 'pixels')
    test_refls = load_mat('Data/test_refls.mat', 'test_refls')
    spd = load_mat('Data/d65_spd.mat', 'spd')
    SSFs = load_mat('Data/canon_60D.mat', 'canon_60d')
    A = np.multiply(spd[:, 1:2], SSFs[:, 1:4])
    train_rgb = np.dot(train_refls, A)
    test_rgb = np.dot(test_refls, A)

    # 光谱重建
    rec_refls = Mine(train_refls, train_rgb, test_rgb, A)
    # 计算平均RMSE
    mean_RMSE = np.mean(np.sqrt(np.mean((rec_refls - test_refls) ** 2, 1)))
    print("平均RMSE：",mean_RMSE)

    plt.plot(np.linspace(400, 700, 31, endpoint=True), rec_refls.T)
    plt.show()



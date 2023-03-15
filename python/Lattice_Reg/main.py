# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/25 21:21   CHEN      1.0         None
"""

import numpy as np
from scipy.io import loadmat
import LatticeRegression
import matplotlib.pyplot as plt


def load_mat(m_path, m_key):
    # 根据路径和键 加载mat文件
    m_data = loadmat(m_path)
    return m_data[m_key]

# 同色异谱黑和晶格回归的光谱反射率重建实现
def Mine(m_train_refls, m_train_rgb, m_test_rgb, m_A):

    # 生成矩阵R
    R = np.dot(np.dot(m_A, np.linalg.pinv(np.dot(m_A.T, m_A))), m_A.T)

    # 训练样本的同色异谱黑
    train_black = np.transpose(m_train_refls.T - np.dot(R, m_train_refls.T))

    # 测试样本的基本刺激光谱
    test_fun_spectrum = np.dot(np.dot(m_A, np.linalg.pinv(np.dot(m_A.T, m_A))), m_test_rgb.T).T

    # 重建测试样本的同色异谱黑
    lr = LatticeRegression.LatticeRegression(train_black, m_train_rgb, m_test_rgb, smoothness=0.3, grid_size=(20, 20, 20))
    test_black = lr.lattice_reg()

    # lr.fit()
    # test_black = lr.predict(test_rgb)

    # 组合测试样本的基本刺激光谱和同色异谱黑，生成光谱反射率
    m_rec_refls = test_black + test_fun_spectrum

    return m_rec_refls


if __name__ == '__main__':

    # 载入数据
    train_refls = load_mat('Data/train_refls.mat', 'pixels')  # 训练样本光谱数据
    test_refls = load_mat('Data/test_refls.mat', 'test_refls')  # 测试样本光谱数据
    spd = load_mat('Data/d65_spd.mat', 'spd')  # d65光源光谱数据
    SSFs = load_mat('Data/canon_60D.mat', 'canon_60d')  # 相机光谱敏感度函数
    A = np.multiply(spd[:, 1:2], SSFs[:, 1:4])  # 光谱响应矩阵
    train_rgb = np.dot(train_refls, A)  # 生成训练样本的rgb值
    test_rgb = np.dot(test_refls, A)  # 生成测试样本的rgb值

    # 光谱反射率重建
    rec_refls = Mine(train_refls, train_rgb, test_rgb, A)

    # 计算平均RMSE
    mean_RMSE = np.mean(np.sqrt(np.mean((rec_refls - test_refls) ** 2, 1)))
    print("平均RMSE：",mean_RMSE)

    #重建光谱反射率曲线绘制
    bands = np.linspace(400, 700, 31, endpoint=True)  # 光谱范围和光谱分辨率
    plt.plot(bands, rec_refls.T)
    plt.show()



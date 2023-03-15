# -*- encoding: utf-8 -*-
"""
@File    :   LatticeRegression.py
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/25 22:20   CHEN      1.0         None
"""
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from scipy.interpolate import interpn
import time


class LatticeRegression:

    """
    :param train_spectrum: 2维数组，shape=(n, bdim), n为训练样本数量， bdim为光谱样本的维度
    :param train_rgb: 2维数组，shape=(n, rdim), n为训练样本数量， rdim为rgb样本的维度,一般为3维
    :param test_rgb: 2维数组，shape=(m, rdim), m为测试样本数量， rdim为rgb样本的维度，,一般为3维
    :param smoothness: 1维数组 or 标量， 数组时shape=(rdim, ), rdim为样本的维度
    :param grid_size: 元组， shape=(rdim, )
    :return: reData: 数组， shape=(m, bdim), m为测试样本数量， bdim为光谱样本的维度

    """
    def __init__(self, train_data, train_rgb, test_rgb, smoothness, grid_size=(17, 17, 17)):
        self.train_spectrum = train_data
        self.train_rgb = train_rgb
        self.test_rgb = test_rgb
        self.smoothness = smoothness
        self.grid_size = grid_size
        self.nDim = self.train_spectrum.shape[1]  # 数据的维度
        self.LUT = []
        self.gridPoints = []

    def lattice_reg(self):

        start_time = time.time()
        self.fit()
        end_time = time.time()
        print("建立LUT耗时: {:.2f}秒".format(end_time - start_time))

        sec_time = time.time()
        recData = self.predict(self.test_rgb)
        print("重建耗时: {:.2f}秒".format(sec_time - end_time))
        return recData

    def fit(self):
        ###根据数据的最大最小值确定网格范围，生成网格数据
        xGrid = np.linspace(np.min(self.train_rgb[:, 0]), np.max(self.train_rgb[:, 0]), self.grid_size[0])
        yGrid = np.linspace(np.min(self.train_rgb[:, 1]), np.max(self.train_rgb[:, 1]), self.grid_size[1])
        zGrid = np.linspace(np.min(self.train_rgb[:, 2]), np.max(self.train_rgb[:, 2]), self.grid_size[2])
        self.gridPoints = [xGrid, yGrid, zGrid]

        ###对每个维度进行建立查找表
        for i in range(self.nDim):
            lut = self.__gridInterpNd(self.train_rgb, self.train_spectrum[:, i], self.gridPoints, self.smoothness)
            self.LUT.append(lut)

    def predict(self, m_test_rgb):
        reData = []  # 记录插值生成的数据

        for i in range(self.nDim):
            # 网格线性插值
            d = interpn(self.gridPoints, self.LUT[i], m_test_rgb, method='linear')
            reData.append(d)

        reData = np.array(reData).T
        return reData

    def __gridInterpNd(self, x, y, xGrid, m_smoothness=0.3):
        n, nDimension = x.shape

        if np.isscalar(m_smoothness):  # 按照维数扩展平滑权重
            m_smoothness = [m_smoothness] * nDimension

        nGrid = np.array([len(item) for item in xGrid])  # 网格的尺寸
        nTotalGridPoints = np.prod(nGrid)  # 总的网格点数量
        nScatterPoints = x.shape[0]  # 训练数据点的数量

        dx = list(map(lambda item: np.diff(item), xGrid))  # 计算每个维度的一阶差分

        ###三线性插值确定权重
        bins = list(map(lambda item: list(bin(item)[2:].zfill(nDimension)),
                        list(np.linspace(0, 2 ** nDimension - 1, 2 ** nDimension, dtype=np.int))))

        # 8个点的局部坐标
        localCellIndex = np.array(list(map(lambda item: list(map(int, item)), bins))).T
        weight = np.ones((nScatterPoints, 2 ** nDimension), np.float64)
        xWeigthIndex = []

        #求局部样本的三维坐标
        for iDim in range(nDimension):
            xIndex = np.digitize(x[:, iDim], xGrid[iDim], True) - 1
            xIndex[xIndex == -1] = 0

            cellFraction = np.minimum(1, np.maximum(0, (x[:, iDim] - xGrid[iDim][xIndex]) / dx[iDim][xIndex]))

            weightsCurrentDimension = np.array([1 - cellFraction, cellFraction])
            weight = weight * weightsCurrentDimension[localCellIndex[iDim, :], :].T
            xWeigthIndex.append(np.repeat(np.expand_dims(xIndex, axis=1), 8, 1) + localCellIndex[iDim, :])

        #将三维坐标转换为线性索引，以便构造二维矩阵
        xWeigthIndex = self.__sub2index(xWeigthIndex, nGrid)

        # 为构造稀疏矩阵做准备
        rows = np.repeat(
            np.expand_dims(np.arange(0, nScatterPoints, 1, dtype=int), axis=1),
            2 ** nDimension, 1)
        rows = rows.flatten()
        cols = xWeigthIndex.flatten()
        weight_spare = weight.flatten()
        # 构建权重的稀疏矩阵
        W = sparse.coo_matrix((weight_spare, (rows, cols)), shape=(nScatterPoints, nTotalGridPoints))

        ###平滑度约束
        nEquationsPerDimension = np.repeat(np.expand_dims(nGrid, 1), nDimension, 1) - np.eye(N=nDimension) * 2
        nSmoothEquations = np.prod(nEquationsPerDimension, 1, dtype=int)
        # nTotalSmoothEquations = np.sum(nSmoothEquations)  # 总的平滑方程数
        multiIndex = np.cumprod(nGrid)

        ###每个维度循环计算二阶差分
        S = []
        for iDim in range(0, nDimension):
            if iDim == 0:
                indexL = np.arange(0, nGrid[0] - 2)[:, None]
                indexM = np.arange(1, nGrid[0] - 1)[:, None]
                indexR = np.arange(2, nGrid[0])[:, None]
            else:
                indexL = np.arange(0, nGrid[0])[:, None]
                indexM = np.arange(0, nGrid[0])[:, None]
                indexR = np.arange(0, nGrid[0])[:, None]

            # 按照列优先的方式将三维坐标转换成线性索引
            for iCell in range(1, nDimension):
                if iCell == iDim:
                    indexL = np.reshape(
                        indexL + np.arange(0, nGrid[0] - 2) * multiIndex[iCell - 1],
                        (len(indexL) * (nGrid[0] - 2), 1), order='F')
                    indexM = np.reshape(
                        indexM + np.arange(1, nGrid[0] - 1) * multiIndex[iCell - 1],
                        (len(indexM) * (nGrid[0] - 2), 1), order='F')
                    indexR = np.reshape(
                        indexR + np.arange(2, nGrid[0]) * multiIndex[iCell - 1],
                        (len(indexR) * (nGrid[0] - 2), 1), order='F')
                else:
                    curDimIndex = np.arange(0, nGrid[1])
                    indexL = np.reshape(indexL + curDimIndex * multiIndex[iCell - 1], (len(indexL) * len(curDimIndex), 1),
                                        order='F')
                    indexM = np.reshape(indexM + curDimIndex * multiIndex[iCell - 1], (len(indexM) * len(curDimIndex), 1),
                                        order='F')
                    indexR = np.reshape(indexR + curDimIndex * multiIndex[iCell - 1], (len(indexR) * len(curDimIndex), 1),
                                        order='F')
            index = np.hstack((indexL, indexM, indexR)).flatten()

            smoothnessMat = [1, -2, 1] * nSmoothEquations[iDim]
            rows = np.repeat(
                np.expand_dims(np.arange(0, nSmoothEquations[iDim], 1, dtype=int), axis=1),
                nDimension, 1).flatten()

            # 构建平滑权重的稀疏矩阵
            smoothnessSpareMatrix = sparse.coo_matrix((smoothnessMat, (rows, index)),
                                                      shape=(nSmoothEquations[iDim], nTotalGridPoints), dtype=np.float64)
            # 将三个平滑权重矩阵垂直拼接
            if iDim == 0:
                S = smoothnessSpareMatrix * m_smoothness[iDim]
            else:
                S = sparse.vstack((S, smoothnessSpareMatrix * m_smoothness[iDim]))

        ###求解yGrids = （W.T * W + Ks) * W.T * y
        Ks = S.T * S
        A = W.T * W + Ks
        b = W.T * y[:, None]

        # 使用稀疏矩阵进行求解，速度更快
        yGrids = sparse.linalg.lsmr(A, b, damp=1e-6, atol=1e-10, btol=1e-10)[0]  # 最小二乘解

        return np.reshape(yGrids, nGrid, order='F')

    def __sub2index(self, multi_index, grid_size):
        # 按照列方向优先将三维坐标转为线性索引
        k = np.cumprod(grid_size)
        ndx = multi_index[0]
        for i in range(1, len(multi_index)):
            ndx = ndx + multi_index[i] * k[i - 1]
        return ndx

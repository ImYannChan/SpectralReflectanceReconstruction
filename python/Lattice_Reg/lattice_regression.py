# -*- encoding: utf-8 -*-
"""
@File    :   lattice_regression.py    
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/25 22:20   CHEN      1.0         None
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.interpolate import griddata
import time


def lattice_reg(train_black, train_rgb, test_rgb, smoothness):
    xLen, yLen, zLen = 20, 20, 20  # 晶格尺寸
    xGrid = np.linspace(np.min(train_rgb[:, 0]), np.max(train_rgb[:, 0]), xLen)
    yGrid = np.linspace(np.min(train_rgb[:, 1]), np.max(train_rgb[:, 1]), yLen)
    zGrid = np.linspace(np.min(train_rgb[:, 2]), np.max(train_rgb[:, 2]), zLen)
    gridPoints = [xGrid, yGrid, zGrid]
    nDim = train_black.shape[1]

    LUT = []
    # 对每个维度进行建立查找表
    start_time = time.time()
    for i in range(nDim):
        lut = regularizedNd(train_rgb, train_black[:, i], gridPoints, smoothness)  # 建立查找表
        LUT.append(lut)
    sec_time = time.time()
    print("建立LUT耗时: {:.2f}秒".format(sec_time - start_time))

    # 对每个维度通过LUT插值
    R, G, B = np.meshgrid(xGrid, yGrid, zGrid)
    LUT_index = np.hstack((R.ravel(order='F')[:, None], G.ravel(order='F')[:, None], B.ravel(order='F')[:, None]))
    np.savez('LUTs\luts.npz', LUT_index, np.array(LUT))

    reData = []
    for i in range(nDim):
        d = griddata(LUT_index, LUT[i], test_rgb, method='linear')
        reData.append(d)
    end_time = time.time()
    print("重建耗时: {:.2f}秒".format(end_time - sec_time))
    return np.array(reData).T


def test(nDim=31, mtest_rgb=[]):
    luts = np.load('LUTs\luts.npz')
    LUT_index = luts['LUT_index']
    LUT = luts['LUT']
    reData = []
    for i in range(nDim):
        d = griddata(LUT_index, LUT[i,:], mtest_rgb, method='linear')
        reData.append(d)
    return np.array(reData).T


def regularizedNd(x, y, xGrid, m_smoothness=0.3):
    n, nDimension = x.shape

    if np.isscalar(m_smoothness):  # 按照维数扩展平滑权重
        m_smoothness = [m_smoothness] * nDimension

    nGrid = np.array([len(item) for item in xGrid])  # 网格的尺寸
    nTotalGridPoints = np.prod(nGrid)  # 总的网格点数量
    nScatterPoints = x.shape[0]  # 训练数据点的数量

    # xGridMin = np.array([np.min(item) for item in xGrid])
    # xGridMax = np.array([np.max(item) for item in xGrid])

    # 计算每个维度的一阶差分
    dx = list(map(lambda item: np.diff(item), xGrid))

    # 三线性插值确定权重
    bins = list(map(lambda item: list(bin(item)[2:].zfill(nDimension)),
                    list(np.linspace(0, 2 ** nDimension - 1, 2 ** nDimension, dtype=np.int))))
    localCellIndex = np.array(list(map(lambda item: list(map(int, item)), bins))).T
    weight = np.ones((nScatterPoints, 2 ** nDimension), np.float64)
    xWeigthIndex = []

    for iDim in range(nDimension):
        xIndex = np.digitize(x[:, iDim], xGrid[iDim], True) - 1
        xIndex[xIndex == -1] = 0

        cellFraction = np.minimum(1, np.maximum(0, (x[:, iDim] - xGrid[iDim][xIndex]) / dx[iDim][xIndex]))

        weightsCurrentDimension = np.array([1 - cellFraction, cellFraction])
        weight = weight * weightsCurrentDimension[localCellIndex[iDim, :], :].T
        xWeigthIndex.append(np.repeat(np.expand_dims(xIndex, axis=1), 8, 1) + localCellIndex[iDim, :])

    xWeigthIndex = sub2index(xWeigthIndex, nGrid)
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
    nTotalSmoothEquations = np.sum(nSmoothEquations)
    multiIndex = np.cumprod(nGrid)

    # 每个维度循环计算二阶差分
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

        #     smoothnessMat = np.repeat(np.array([[1, -2, 1]]), nSmoothEquations[iDim], 0).flatten()
        smoothnessMat = [1, -2, 1] * nSmoothEquations[iDim]
        rows = np.repeat(
            np.expand_dims(np.arange(0, nSmoothEquations[iDim], 1, dtype=int), axis=1),
            nDimension, 1).flatten()

        #构建平滑权重的稀疏矩阵
        smoothnessSpareMatrix = sparse.coo_matrix((smoothnessMat, (rows, index)),
                                                  shape=(nSmoothEquations[iDim], nTotalGridPoints), dtype=np.float64)
        # 将三个平滑权重矩阵垂直拼接
        if iDim == 0:
            S = smoothnessSpareMatrix * m_smoothness[iDim]
        else:
            S = sparse.vstack((S, smoothnessSpareMatrix * m_smoothness[iDim]))

    ###求解yGrids = （W.T * W + Ks) * W.T * y
    Ks = S.T * S
    # Ki = sparse.coo_matrix(([10e-5] * nTotalGridPoints, (np.arange(0, nTotalGridPoints), np.arange(0, nTotalGridPoints))), shape=(nTotalGridPoints, nTotalGridPoints))
    A = W.T * W + Ks
    b = W.T * y[:, None]

    # 使用稀疏矩阵进行求解，速度更快
    yGrids = sparse.linalg.lsmr(A, b, damp=1e-6, atol=1e-10, btol=1e-10)[0]  # 最小二乘解

    return yGrids


def sub2index(multi_index, grid_size):
    # 按照列方向优先将三维坐标转为线性索引
    k = np.cumprod(grid_size)
    ndx = multi_index[0]
    for i in range(1, len(multi_index)):
        ndx = ndx + multi_index[i] * k[i - 1]
    return ndx


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4]) - 1
    b = np.array([1, 2, 3, 4]) - 1
    c = np.array([1, 2, 3, 4]) - 1
    value = np.arange(0, 64)
    x, y, z = np.meshgrid(a, b, c)
    abc = np.array([x, y, z])

    xyz = np.hstack((x.ravel(order='F')[:, None], y.ravel(order='F')[:, None], z.ravel(order='F')[:, None]))
    aaa = (x.ravel(order='F'), y.ravel(order='F'), z.ravel(order='F'))

    ndxs = sub2index(aaa, [4, 4, 4])
    nnn = np.array(np.ravel_multi_index(aaa, (4, 4, 4), order='F'))


参考论文：
Chen Y, Zhang S, Xu L. Spectral Reflectance Reconstruction of Organ Tissue Based on Metameric Black and Lattice Regression[J]. SENSORS, 2022, 22(23):9405.

Data文件夹包含光源、相机、训练样本和测试样本的光谱数据，运行时需要加入程序路径；

latte_reg.m 文件是晶格回归函数；

Mine.m 文件是同色异谱黑和晶格回归方法的结合函数；

regularizeNd.p 为LUT建立函数；

rmse_mine.m 用以计算重建光谱和实际光谱间的均方根误差rmse；

main.m文件是程序入口。

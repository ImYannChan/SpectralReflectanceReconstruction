%载入光源和相机响应函数
load 'canon_60D.mat';
load 'd65_spd.mat';

%光源和观察者的权重矩阵
A = spd(:,2) .* canon_60d(:,2:4);
%载入训练反射率的光谱反射率
load 'train_refls_4900.mat';

%根据相机响应函数获取相机的rgb响应值
train_rgb = pixels * A;

%光谱分辨率及范围
bands = 400:10:700;

%载入测试样本的光谱反射率
load 'test_refls_1629.mat';

% 将测试样本的光谱反射率映射到相机rgb空间
test_rgb = test_refls * A;

% 光谱反射率重建
tic;
rec_refl = Mine(pixels, train_rgb, test_rgb, A);
toc;

% 计算rmse
[mean_rmse, min_rmse, max_rmse, median_rmse,per_rmse] = rmse_mine(rec_refl, test_refls);

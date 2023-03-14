function [rec_refls] = Mine(train_refl,train_rgb, test_rgb, A)
%MINE 此处显示有关此函数的摘要
%   此处显示详细说明


    %计算训练样本的同色异谱黑
    R = A * pinv(A'*A) * A';
    train_black = (train_refl' - R * train_refl')';
    
    %计算测试样本的基本刺激
    test_s = (A * pinv(A'*A) * test_rgb')';
    
    %% 晶格回归重建
    rec_black=latte_reg(train_black, train_rgb, test_rgb, 0.0013, 31);
    %测试样本的基本刺激加上其插值出来的同色异谱黑
    rec_refls = test_s + rec_black;
end


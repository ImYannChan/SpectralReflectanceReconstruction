function [rec_refl] = latte_reg(train_refl,train_rgb,test_rgb, smoothness,n)
%LATTE_REG 此处显示有关此函数的摘要
%   此处显示详细说明
    xGrid = linspace(min(train_rgb(:,1)), max(train_rgb(:,1)), 15); 
    yGrid = linspace(min(train_rgb(:,2)), max(train_rgb(:,2)), 15);
    zGrid = linspace(min(train_rgb(:,3)), max(train_rgb(:,3)), 15);
    gridPoints = {xGrid, yGrid, zGrid,};
%     tic; %启动秒表计时器
    for i = 1:n
        eval(['u3x_meas_Grid_',num2str(i),' = regularizeNd(train_rgb, train_refl(:,i), gridPoints, smoothness, ''linear'');']);
    end
%     toc;
%     tic;
    for i = 1:n
        eval(['Function_',num2str(i),' = griddedInterpolant(gridPoints, u3x_meas_Grid_', num2str(i), ',''linear'');']);
    end
    %验证插值效果
    for i=1:n
        eval(['Plot_',num2str(i),' = Function_', num2str(i), '(test_rgb);']);
    end

    for i=1:n
        eval(['rec_refl(:, i)=Plot_', num2str(i),';']);
    end
%     toc;
end


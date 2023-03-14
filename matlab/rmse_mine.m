function [mean_rmse,min_rmse, max_rmse, median_rmse, per] = rmse_mine(rec_data,test_data)
    RMSE_ALL = sqrt(mean((rec_data - test_data).^2, 2));
    max_rmse = max(RMSE_ALL);
    min_rmse = min(RMSE_ALL);
    mean_rmse = mean(RMSE_ALL);
    median_rmse = median(RMSE_ALL, 'all');
    per = prctile(RMSE_ALL,80);
end


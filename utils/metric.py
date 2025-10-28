import numpy as np

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


def cal_metric(ground_truth, prediction, args):
    args.logger.info("[*] year {}, testing".format(args.year))
    mae_list, rmse_list, mape_list = [], [], []
    for i in range(1, 13):
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        if i==3 or i==6 or i==12:
            args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
            args.result[str(i)][" MAE"][args.year] = mae
            args.result[str(i)]["MAPE"][args.year] = mape
            args.result[str(i)]["RMSE"][args.year] = rmse
    args.result["Avg"][" MAE"][args.year] = np.mean(mae_list)
    args.result["Avg"]["RMSE"][args.year] = np.mean(rmse_list)
    args.result["Avg"]["MAPE"][args.year] = np.mean(mape_list)
    args.logger.info("T:Avg\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))
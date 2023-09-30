# @Time     : 2022/4/14 20:41
# @Author   : Chen nengzhen
# @FileName : utils.py
# @Software : PyCharm
import errno
import math
import os
import sys
from collections import Counter

import numpy as np
import torch

lg_e_10 = math.log(10)


def log10(x):
    """
    Convert a new tensor with the base-10 logarithm of the elements of x
    :param x:
    :return:
    """
    return torch.log(x) / lg_e_10


class Metrics(object):
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0  # Scale invariant logarithmic error [log(m)*100]
        self.set_to_worst()

    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, delta1, delta2, delta3, silog):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.silog = silog

    def evaluate(self, output, target):
        valid_mask = target > 0.1

        # convert from meters to mm
        # 注意：模型预测的是绝对深度值，单位是米（m）
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        abs_diff = (output_mm - target_mm).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float(((abs_diff / target_mm) ** 2).mean())

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

        # silog uses meters
        err_log = torch.log(target[valid_mask]) - torch.log(output[valid_mask])
        normalized_squared_log = (err_log ** 2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask]) ** (-1)
        inv_target_km = (1e-3 * target[valid_mask]) ** (-1)

        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

    def print_metrics(self):
        print(
            "RMSE={:.3f}  MAE={:.3f}  iRMSE={:.3f}  iMAE={:.3f}  "
            "sqrel={:.3f}  silog={:3f}  "
            "Delta1={:.3f}  Delta2={:.3f}  Delta3={:.3f}  "
            "absrel={:.3f}  lg10={:.3f}\n"
            .format(self.rmse, self.mae, self.irmse, self.imae,
                    self.squared_rel, self.silog,
                    self.delta1, self.delta2, self.delta3,
                    self.absrel, self.lg10))


class AverageMetrics(object):
    def __init__(self):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_silog = 0

    def update(self, metric, n=1):
        self.count += n
        self.sum_irmse += n * metric.irmse
        self.sum_imae += n * metric.imae
        self.sum_mse += n * metric.mse
        self.sum_rmse += n * metric.rmse
        self.sum_mae += n * metric.mae
        self.sum_absrel += n * metric.absrel
        self.sum_squared_rel += n * metric.squared_rel
        self.sum_lg10 += n * metric.lg10
        self.sum_delta1 += n * metric.delta1
        self.sum_delta2 += n * metric.delta2
        self.sum_delta3 += n * metric.delta3
        self.sum_silog += n * metric.silog

    def average(self):
        avg = Metrics()
        if self.count > 0:
            avg.update(
                self.sum_irmse / self.count, self.sum_imae / self.count,
                self.sum_mse / self.count, self.sum_rmse / self.count,
                self.sum_mae / self.count, self.sum_absrel / self.count,
                self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
                self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                self.sum_delta3 / self.count, self.sum_silog / self.count,
            )
        return avg


def first_run(save_path):
    txt_file = os.path.join(save_path, 'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return ''
        return saved_epoch
    return ''


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as os_error:
            if os_error.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_file(content, location):
    file = open(location, 'w')
    file.write(str(content))
    file.close()


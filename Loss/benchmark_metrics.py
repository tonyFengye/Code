# @Time     : 2022/4/14 19:28
# @Author   : Chen nengzhen
# @FileName : benchmark_metrics.py
# @Software : PyCharm
import math

import torch


class Metrics(object):
    def __init__(self, max_depth=85):
        self.rmse, self.mae, self.irmse, self.imae = 0.0, 0.0, 0.0, 0.0
        self.num = 0
        self.max_depth = max_depth

    def calculate(self, prediction, gt):

        prediction[prediction <= 0.9] = 0.9
        prediction[prediction > 85] = 85
        valid_mask = (gt > 0).detach()
        # valid_mask = (gt > 0 & gt <= 100).detach()  # copy from ACMNet val.py

        self.num = valid_mask.sum().item()  # 计算所有valid_mask的有笑点个数

        # convert from meters to mm
        # 注意：模型预测的是绝对深度值，单位是米（m）
        prediction_mm = prediction[valid_mask]
        gt_mm = gt[valid_mask]

        prediction_mm = torch.clamp(prediction_mm, min=0, max=self.max_depth)

        abs_diff = (prediction_mm - gt_mm).abs()

        self.rmse = torch.sqrt(torch.mean(torch.pow(abs_diff, 2))).item()
        self.mae = abs_diff.mean().item()

        # convert from meters to km
        # 计算irmse, imae
        inv_output_km = (1e-3 * prediction[valid_mask]) ** (-1)
        inv_target_km = (1e-3 * gt[valid_mask]) ** (-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

    def get_metrics(self, metric_name):
        return self.__dict__[metric_name]


def allowed_metrics():
    return Metrics().__dict__.keys()

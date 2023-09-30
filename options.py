# @Time     : 2022/4/14 20:41
# @Author   : Chen nengzhen
# @FileName : options.py
# @Software : PyCharm

import argparse
import os

import criteria
from Loss.benchmark_metrics import allowed_metrics

parser = argparse.ArgumentParser(description="two-branch sparse to dense depth completion")

parser.add_argument("--workers", default=10, type=int, metavar="N", help="number of data loading workers")
parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number(useful or restart")
parser.add_argument("--start-epoch-bias", default=0, type=int, metavar="N",
                    help="manual epoch number bias(useful on restarts)")
parser.add_argument("-c", "--criterion", metavar="LOSS", default="L2", choices=criteria.loss_names)
parser.add_argument("-b", "--batch-size", default=16, type=int, help="mini-batch size (default 1)")
parser.add_argument("-lr", "--lr", default=1e-3, type=float, metavar="LR", help="initial learning rate (default le-5)")
parser.add_argument("--optimizer", type=str, default="adam", help="adam or sgd")

parser.add_argument("--weight-decay", default=1e-6, type=float, help="weight decay (default: 0)")
parser.add_argument("--print_freq", "-p", default=10, type=int, help="print frequency (default 10)")
parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint (default: none)")
parser.add_argument("--data-folder", default="G:\\Dataset\\KITTI\\rawdata", help="data folder path")
parser.add_argument("--data-folder-save", default="./data/kitti_depth_test", type=str, help="data folder test results_resnet34_no_attention")

# Loss setting
parser.add_argument("--metric_1", type=str, default="rmse", choices=allowed_metrics(), help="metric to use during evaluation")
parser.add_argument("--metric_2", type=str, default="mae", choices=allowed_metrics(), help="metric to use during evaluation")
parser.add_argument("--metric_3", type=str, default="irmse", choices=allowed_metrics(), help="metric to use during evaluation")
parser.add_argument("--metric_4", type=str, default="imae", choices=allowed_metrics(), help="metric to use during evaluation")

parser.add_argument("-i", "--input", type=str, default="rgbd", choices=["rgbd", "gd", "rgb", "d"])
parser.add_argument("--val", type=str, default="select", choices=["select", "full"],
                    help="full or select validation set")
parser.add_argument("--jitter", type=float, default=0.1, help="color jitter for image")
parser.add_argument("--rank-metric", type=str, default="rmse", help="metrics for which best results_resnet34_no_attention is saved")
parser.add_argument('--evaluate', action='store_true', help='only evaluate')
parser.add_argument('--test_mode', action='store_true', help='Do not use resume')
parser.add_argument("-f", "--freeze-backbone", action="store_true", default=False, help="freeze parameters in backbone")
parser.add_argument("--test", action="store_true", default=False, help="save results_resnet34_no_attention kitti test dataset for submission")
parser.add_argument("--cpu", action="store_true", default=False, help="run on cpu")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--log-path", type=str, default="./log")

# 模型参数设置
parser.add_argument("--layers", type=list, default=[3, 4, 6, 3], help="number of convs in each block, default resnet18")

# 设置分布式训练
parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")

# 设置对比损失温度参数
parser.add_argument("--temperature", default=0.07, type=float, help="对比损失温度系数")

# random cropping
parser.add_argument("--not-random-crop", action="store_true", default=False, help="prohibit random cropping")
parser.add_argument("-he", "--random-crop-height", default=320, type=int, metavar="N", help="random crop height")
parser.add_argument("-w", '--random-crop-width', default=1216, type=int, metavar="N", help="random crop width")

args = parser.parse_args()

args.use_rgb = ("rgb" in args.input)
args.use_d = ("d" in args.input)
args.use_g = ("g" in args.input)  # for gray image
args.val_h = 352
args.val_w = 1216

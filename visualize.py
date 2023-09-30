# @FileName : visualize.py
# @Software : PyCharm
import argparse
import os

import PIL.Image
import matplotlib
import numpy as np
import torch
from matplotlib import cm
from torch.utils.data import DataLoader
from networks.model import IDNet
from dataloader.kitti_loader import KittiDepth
import options
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Simple testing function for depthCompletionNet")
parser.add_argument("--ckpt_path", default="./log/mode-train_optim-Adam_criterion-L2_lr-0.001_bs-20_epochs-35/checkpoints", help="path to load model checkpoints")
parser.add_argument("--ckpt", default="checkpoint_model_epoch_15.pth", help="which model checkpoint to load")
parser.add_argument("--ext", type=str, default="png", help="image extension to search for in folder")
parser.add_argument("--save_path", type=str, default="outputs-e6", help="path to save model output depth maps")

args = parser.parse_args()


device = torch.device("cuda")
model_path = os.path.join(args.ckpt_path, args.ckpt)
if not os.path.exists(model_path):
    print("ERROR! cannot find model pth from: {}".format(model_path))
    exit(0)
print("-> Loading IDNet checkpoint from ", model_path)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
rgb_out_path = os.path.join(args.save_path, "rgb")
depth_out_path = os.path.join(args.save_path, "depth")
fuse_out_path = os.path.join(args.save_path, "fuse")
if not os.path.exists(rgb_out_path):
    os.makedirs(rgb_out_path)
if not os.path.exists(depth_out_path):
    os.makedirs(depth_out_path)
if not os.path.exists(fuse_out_path):
    os.makedirs(fuse_out_path)

# define model
model = IDNet()
# load model pth
model_dict = torch.load(model_path, map_location=device)["state_dict"]
# model.module.load_state_dict(model_dict)
# new_state_dict = OrderedDict()
# for k, v in model_dict.items():
#     name = k[7:]
#     new_state_dict[name]=v
# model.state_dict = new_state_dict
# model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"].module)
model = nn.DataParallel(model).cuda()
model.load_state_dict(model_dict)

model.to(device)
model.eval()

dataset = KittiDepth(split="test_completion", args=options.args)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)

print(len(dataloader))
for index, item in enumerate(dataloader):
    for key, value in item.items():
        item[key] = value.to(device)
        # if key == "rgb":
        #     item[key] = item[key] * 0.0
    fuse_branch_out, rgb_branch_out, depth_branch_out, _ = model(item)
    # rgb_branch_out = F.interpolate(rgb_branch_out, [352, 1216], mode="bilinear", align_corners=False)
    # depth_branch_out = F.interpolate(depth_branch_out,  [352, 1216], mode="bilinear", align_corners=False)
    fuse_branch_out, rgb_branch_out, depth_branch_out = fuse_branch_out[0], rgb_branch_out[0], depth_branch_out[0]

    # saving colormapped depth images
    rgb_branch_out = np.squeeze(rgb_branch_out.data.cpu().numpy())
    depth_branch_out = np.squeeze(depth_branch_out.data.cpu().numpy())
    fuse_out = np.squeeze(fuse_branch_out.data.cpu().numpy())

    # save rgb branch output depth map
    vmax_rgb = np.percentile(rgb_branch_out, 95)
    normalizer = matplotlib.colors.Normalize(vmin=rgb_branch_out.min(), vmax=vmax_rgb)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    colormapped_rgb_out_depthmap = (mapper.to_rgba(rgb_branch_out)[:, :, :3] * 255).astype(np.uint8)
    rgb_im = PIL.Image.fromarray(colormapped_rgb_out_depthmap)
    rgb_im.save(os.path.join(rgb_out_path, "rgb_" + str(index) + ".jpeg"))

    print("-> finished saving " + "rgb_" + str(index) + ".jpeg")

    # save depth branch output depth map
    vmax_depth = np.percentile(depth_branch_out, 95)
    normalizer = matplotlib.colors.Normalize(vmin=depth_branch_out.min(), vmax=vmax_depth)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    colormapped_depth_out_depthmap = (mapper.to_rgba(depth_branch_out)[:, :, :3] * 255).astype(np.uint8)
    depth_im = PIL.Image.fromarray(colormapped_depth_out_depthmap)
    depth_im.save(os.path.join(depth_out_path, "depth_" + str(index) + ".jpeg"))

    print("-> finished saving " + "depth_" + str(index) + ".jpeg")

    # save fuse branch output depth map
    vmax_fuse = np.percentile(fuse_out, 95)
    normalizer = matplotlib.colors.Normalize(vmin=fuse_branch_out.min(), vmax=vmax_fuse)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    colormapped_fuse_out_depthmap = (mapper.to_rgba(fuse_out)[:, :, :3] * 255).astype(np.uint8)
    fuse_im = PIL.Image.fromarray(colormapped_fuse_out_depthmap)
    fuse_im.save(os.path.join(fuse_out_path, "fuse_" + str(index) + ".jpeg"))

    print("-> finished saving " + "fuse_" + str(index) + ".jpeg")

    if index == 10:
        break




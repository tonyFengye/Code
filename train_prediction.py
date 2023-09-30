# @Time     : 2022/3/22 14:50
# @Author   : Chen nengzhen
# @FileName : train_prediction.py
# @Software : PyCharm
import glob
import os
import random
import shutil
import sys
import time
from datetime import datetime

from torch import nn
from tqdm import tqdm
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from Loss.benchmark_metrics import Metrics
from criteria import MaskedMSELoss, ReconstructionLoss, SILogLoss
from dataloader.kitti_loader import KittiDepth, KittiPredictionDepth, KittiPrediction
from networks.model import IDNet
from options import args
from utils import first_run, mkdir_if_missing, Logger, AverageMeter


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # warnings.warn("You have chosen to seed training. This will turn on the CUDNN deterministic setting, which
    # can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.")


# 设置随机数种子
setup_seed(True)


def adjust_learning_rate(lr_init, optimizer, epoch):
    """ Sets the learning rate to the initial LR decacy by 10 every 5 epochs"""
    lr = lr_init
    if epoch >= 15:
        lr = lr_init * 0.5
    if epoch >= 20:
        lr = lr_init * 0.2
    if epoch >= 25:
        lr = lr_init * 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(state, save_path, to_copy, epoch):
    checkpoint_path = os.path.join(save_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    filepath = os.path.join(checkpoint_path, "checkpoint_model_epoch_{}.pth".format(epoch))
    torch.save(state, filepath)
    if to_copy:
        if epoch > 0:
            lst = glob.glob(os.path.join(checkpoint_path, "model_best*"))
            if len(lst) != 0:
                os.remove(lst[0])
            shutil.copyfile(filepath, os.path.join(checkpoint_path, "model_best_epoch_{}.pth".format(epoch)))
            print("Best model copied")
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(filepath, "checkpoint_model_epoch_{}.pth".format(epoch - 1))
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.start_epoch = args.start_epoch
        self.epochs = args.epochs

        self.device = args.device

        self.train_dataset = KittiPrediction(eval_split="eigen", mode="train", args=args)
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)

        self.val_select_dataset = KittiPrediction(eval_split="eigen", mode="val", args=args)
        self.val_loader = DataLoader(self.val_select_dataset, 4, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

        # define model
        self.model = IDNet()
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)

        # 定义损失函数：L2损失silog loss，
        self.l2_fuse_loss = MaskedMSELoss()
        self.l2_depth_loss = MaskedMSELoss()
        self.reconstruction_loss = ReconstructionLoss()
        self.silog_loss = SILogLoss()

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

        self.save_id = "mode-{}_optim-{}_criterion-{}_lr-{}_bs-{}_epochs-{}".format("train", "Adam", args.criterion, self.lr, self.batch_size, self.epochs)

        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        self.summary_writer = SummaryWriter(logdir=os.path.join(args.log_path, self.save_id))

    def train(self):
        # Resume training
        best_epoch = 0
        lowest_loss = np.inf
        save_path = os.path.join(args.log_path, self.save_id)
        mkdir_if_missing(save_path)

        log_file_name = "log_train_start_0.txt"
        resume = first_run(save_path)

        if resume:
            path = os.path.join(save_path, "checkpoints/checkpoint_model_epoch_{}.pth".format(int(resume)))
            if os.path.isfile(path):
                log_file_name = "log_train_start_{}.txt".format(resume)
                # stdout
                sys.stdout = Logger(os.path.join(save_path, log_file_name))
                print("==> Loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(path)
                args.start_epoch = checkpoint["epoch"]
                lowest_loss = checkpoint["loss"]
                best_epoch = checkpoint["best epoch"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("==> Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint["epoch"]))
            else:
                log_file_name = "log_train_start_0.txt"
                # stdout
                sys.stdout = Logger(os.path.join(save_path, log_file_name))
                print("==> No checkpoint found at '{}'".format(path))

        else:
            sys.stdout = Logger(os.path.join(save_path, log_file_name))

        # Init Model
        print(40 * "=" + "\nArgs:{}\n".format(args) + 40 * "=")
        print("Number of parameters in model is {:.3f}M".format(
            sum(tensor.numel() for tensor in self.model.parameters()) / 1e6))

        # start training
        for epoch in range(args.start_epoch, self.epochs):
            print("\n => Start Epoch {}".format(epoch))
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(save_path)
            # 调整学习率
            lr = adjust_learning_rate(args.lr, self.optimizer, epoch)
            print("lr is set to {}".format(lr))

            # Define container objects
            data_time = AverageMeter()

            # train model
            self.model.train()

            # compute timing
            end = time.time()

            # load dataset
            for idx, data in tqdm(enumerate(self.train_loader)):
                # Time dataloader
                data_time.update(time.time() - end)

                # put inputs on gpu if possible
                data = {key: value.to(self.device) for key, value in data.items() if value is not None}
                gt = data["d"]  # sparse depth map
                gray = data["g"]  # gray image corresponding to rgb

                fused_map, rgb_out, depth_out, reconstructed_gray_image = self.model(data)

                # fuse loss
                loss_fuse = self.silog_loss(fused_map[0], gt) + 0.2 * self.silog_loss(fused_map[1], gt) + 0.2 * self.silog_loss(fused_map[2], gt)
                loss_rgb = self.silog_loss(rgb_out[0], gt) + 0.2 * self.silog_loss(rgb_out[1], gt) + 0.2 * self.silog_loss(rgb_out[2], gt)
                loss_depth = self.silog_loss(depth_out[0], gt) + 0.2 * self.l2_depth_loss(depth_out[1], gt) + 0.2 * self.l2_depth_loss(depth_out[2], gt)
                loss_reconstruction = self.reconstruction_loss(reconstructed_gray_image, gray)

                sum_loss = loss_fuse + loss_depth + loss_rgb + 1e-4 * loss_reconstruction

                train_step = int(epoch) * len(self.train_loader) + idx + 1
                self.summary_writer.add_scalar("Loss/total_loss", sum_loss.item(), train_step)
                self.summary_writer.add_scalar("Loss/loss_fuse", loss_fuse.item(), train_step)
                self.summary_writer.add_scalar("Loss/loss_rgb", loss_rgb.item(), train_step)
                self.summary_writer.add_scalar("Loss/loss_depth", loss_depth.item(), train_step)
                self.summary_writer.add_scalar("Loss/图像重构损失", loss_reconstruction.item(), train_step)

                # setup backward pass
                self.optimizer.zero_grad()
                sum_loss.backward()
                self.optimizer.step()

            # Evaluate model on Eigen split
            print("=> Start selection validation set")
            mean_errors = evaluate(self.model, self.device, self.val_loader, "eigen")
            abs_rel = mean_errors.tolist()[0]
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

            # File to keep latest epoch
            with open(os.path.join(save_path, "first_run.txt"), "w") as f:
                f.write(str(epoch))

            # Save model
            to_save = False
            if abs_rel < lowest_loss:
                to_save = True
                best_epoch = epoch
                lowest_loss = abs_rel

            print("===> Last best score was RMSE of {:.4f} in epoch {}".format(lowest_loss, best_epoch))

            save_checkpoint({
                "epoch": epoch,
                "best epoch": best_epoch,
                "state_dict": self.model.state_dict(),
                "loss": lowest_loss,
                "optimizer": self.optimizer.state_dict()
            }, save_path, to_save, epoch)


def compute_errors(gt, pred):
    """computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(model, device, val_loader, split):
    """ Evaluates metrics using a specified test set"""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # 将模型调整为测试模式
    model.eval()

    errors = []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_loader)):
            data = {key: value.to(device) for key, value in data.items() if value is not None}
            d1, d2, d4, reconstruction = model(data)
            pred = d1.cpu()[:, 0].numpy()
            gt = data["gt"].cpu()[:, 0].numpy()
            gt_height, gt_width = 352, 1216

            if split == "eigen":
                mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)  # crop.shape 4, 1242
                crop_mask = np.zeros(mask.shape)  # crop_mask.shape: 375, 1242

                # 问题出在这一句，问题代码为crop_mask[0crop[0]:crop[1], crop[2]:crop[3]] = 1, 由于crop_mask的shape为[bs, h, w],导致赋值1失败，crop_mask全0.
                crop_mask[:, crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            else:
                mask = gt > 0
            pred = pred[mask]
            gt = gt[mask]

            pred[pred < MIN_DEPTH] = MIN_DEPTH
            pred[pred > MAX_DEPTH] = MAX_DEPTH

            errors.append(compute_errors(gt, pred))

    mean_errors = np.array(errors).mean(0)

    return mean_errors


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

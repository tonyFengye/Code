# @Time     : 2022/3/20 14:32
# @Author   : Chen nengzhen
# @FileName : kitti_loader.py
# @Software : PyCharm

import glob
import os

import numpy as np
import skimage
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import options
from dataloader.dense_to_sparse import UniformSampling
from options import args
from dataloader import transforms
from kitti_utils import generate_depth_map

ori_height, ori_width = 352, 1216


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_paths_and_transform(split, args):
    global get_rgb_paths, glob_gt, glob_d, glob_rgb
    # assert (args.use_d or args.use_rgb or args.use_g), "no proper input selected"

    if split == "train":
        transform = train_transform
        # sparse depth, about 5% density
        glob_d = os.path.join(
            args.data_folder,
            "data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2, 3]/*.png"
        )
        # annotated depth 30% density
        glob_gt = os.path.join(
            args.data_folder,
            "data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2, 3]/*.png"
        )

        def get_rgb_paths(p):
            ps = p.split("/")
            # pnew = "/".join([args.data_folder] + ["data_rgb"] + ps[-6:-4] + ps[-2:-1] + ["data"] + ps[-1:])
            # 修改RGB图像构造路径以适应数据集组织结构
            date = ps[-5:-4][0].split("_drive_")[0]
            pnew = "/".join([args.data_folder] + [date] + ps[-5:-4] + ps[-2:-1] + ["data"] + ps[-1:])
            pnew = pnew.replace("\\", "/")
            return pnew
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                "data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2, 3]/*.png"
            )
            glob_gt = os.path.join(
                args.data_folder,
                "data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2, 3]/*.png"
            )

            def get_rgb_paths(p):
                ps = p.split("/")
                date = ps[-5:-4][0].split("_drive_")[0]
                pnew = "/".join(ps[:-7] + [date] + ps[-5:-4] + ps[-2:-1] + ["data"] + ps[-1:])
                return pnew
        elif args.val == "select":
            transform = no_transform
            glob_d = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/velodyne_raw/*.png"
            )
            glob_gt = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            )

            def get_rgb_paths(p):
                return p.replace("groundtruth_depth", "image")
    elif split == "test_completion":  # 对于depth completion任务，测试时输入是稀疏深度图+RGB图像，输出是稠密深度图
        transform = no_transform
        glob_d = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/image/*.png"
        )
    elif split == "test_prediction":  # 对于depth prediction任务，测试时输入是RGB图像，输出是深度图
        transform = no_transform
        glob_d = None
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_prediction_anonymous/image/*.png"
        )
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))  # 获得sparse depth paths
        paths_d = [path_d.replace("\\", "/") for path_d in paths_d]
        paths_gt = sorted(glob.glob(glob_gt))  # 获得annotated depth paths
        paths_gt = [path_gt.replace("\\", "/") for path_gt in paths_gt]
        paths_rgb = sorted(get_rgb_paths(p) for p in paths_gt)
        paths_rgb = [path_rgb.replace("\\", "/") for path_rgb in paths_rgb]
    else:
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def train_transform(rgb, sparse, target, args):
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    transform_geometric = transforms.Compose([
        transforms.BottomCrop((ori_height, ori_width)),
        transforms.HorizontalFlip(do_flip),

    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=0),
            transform_geometric,

        ])
        rgb = transform_rgb(rgb)
    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target


def val_transform(rgb, sparse, target, args):
    transform = transforms.Compose([
        transforms.BottomCrop((ori_height, ori_width))
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)

    return rgb, sparse, target


def no_transform(rgb, sparse, target, args):
    return rgb, sparse, target


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file, dtype="uint8")  # in the range of [0, 255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    # load depth map D from png file and return it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)  # 使用Image.open()打开稀疏深度图，每个像素点的值都变成255*depth_value。
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit
    assert np.max(depth_png) > 255, "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256
    depth = np.expand_dims(depth, -1)
    return depth


class KittiDepth(Dataset):
    """
        A dataloader for kitti datast
    """

    def __init__(self, split, args):
        super(KittiDepth, self).__init__()
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        # self.K = load_calib()
        self.thresh_translation = 0.1
        self.to_tensor = transforms.ToTensor()
        # self.to_float_tensor = self.to_tensor()

    def __getraw__(self, index):
        rgb = rgb_read(self.paths["rgb"][index]) if (
                    self.paths["rgb"][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        sparse = depth_read(self.paths["d"][index]) if (
                    self.paths["d"][index] is not None and self.args.use_d) else None
        target = depth_read(self.paths["gt"][index]) if self.paths["gt"][index] is not None else None
        return rgb, sparse, target

    def __getitem__(self, index):
        rgb, sparse, target = self.__getraw__(index)
        rgb, sparse, target = self.transform(rgb=rgb, sparse=sparse, target=target, args=self.args)
        rgb, gray = handle_gray(rgb, self.args)

        candidates = {
            "rgb": rgb, "d": sparse, "gt": target, "g": gray
        }
        items = {
            key: self.to_tensor(val).float() for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths["gt"])


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class KittiPrediction(Dataset):
    def __init__(self, eval_split="eigen", mode="train", args=None):
        super(KittiPrediction, self).__init__()
        if mode is None:
            mode = ["train", "val"]
        self.data_path = options.args.data_folder
        self.full_res_shape = (1216, 352)
        self.mode = mode
        if mode == "train":
            self.filenames = readlines("../data_splits/eigen_train_files_with_gt.txt")
        elif mode == "val":
            self.filenames = readlines("../data_splits/eigen_test_files_with_gt.txt")
        self.loader = pil_loader
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.to_tensor = transforms.ToTensor()

        self.sparsifier = UniformSampling(num_samples=10000, max_depth=80)

    def get_color(self, rgb_path):
        color = self.loader(rgb_path)
        color = np.asarray(color)  # 将PIL Image转为numpy array
        return color

    def get_depth(self, depth_path):

        assert os.path.exists(depth_path), "file not found: {}".format(depth_path)
        img_file = Image.open(depth_path)  # 使用Image.open()打开稀疏深度图，每个像素点的值都变成255*depth_value。
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit
        assert np.max(depth_png) > 255, "np.max(depth_png)={}, path={}".format(np.max(depth_png), depth_path)

        depth = depth_png.astype(np.float) / 256
        depth = np.expand_dims(depth, -1)
        return depth

    def __getitem__(self, index):
        line = self.filenames[index].split()
        rgb_file = line[0]
        depth_gt = os.path.join(rgb_file.split('/')[0], line[1])
        depth_sparse = depth_gt.replace("groundtruth", "velodyne_raw")

        image_path = os.path.join(self.data_path, rgb_file)
        depth_gt_path = os.path.join(self.data_path, depth_gt)
        depth_sparse_path = os.path.join(self.data_path, depth_sparse)

        if self.mode == "train":
            rgb, sparse, groundtruth = self.train_transform(self.get_color(image_path), self.get_depth(depth_sparse_path), self.get_depth(depth_gt_path), args)
            sparse100 = self.create_sparse_depth(sparse)
            rgb, gray = handle_gray(rgb, args)
        if self.mode == "val":
            rgb, sparse, groundtruth = self.val_transform(self.get_color(image_path), self.get_depth(depth_sparse_path), self.get_depth(depth_gt_path), args)
            sparse100 = self.create_sparse_depth(sparse)
            rgb, gray = handle_gray(rgb, args)
        inputs = {"rgb": self.to_tensor(rgb).float(), "gt": self.to_tensor(groundtruth).float(), "d": self.to_tensor(sparse).float(), "g": self.to_tensor(gray), "s100": self.to_tensor(sparse100).float()}

        return inputs

    def __len__(self):
        return len(self.filenames)

    def create_sparse_depth(self, depth):
        mask_keep = self.sparsifier.dense_to_sparse(depth)
        sparse_depth = np.zeros(depth.shape)
        sparse_depth[mask_keep] = depth[mask_keep]
        return sparse_depth


class KittiPredictionDepth(Dataset):
    def __init__(self, eval_split):
        super(KittiPredictionDepth, self).__init__()
        self.data_path = options.args.data_folder
        self.side_map = {"l": 2, "r": 3}
        self.full_res_shape = (1216, 352)
        self.filenames = readlines(os.path.join("split", eval_split, "test_files.txt"))
        self.loader = pil_loader
        self.transform = val_transform
        self.to_tensor = transforms.ToTensor()

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_idx = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_idx))
        )
        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_idx, side):
        color = self.loader(self.get_image_path(folder, frame_idx, side))
        color = np.asarray(color)  # 将PIL Image转为numpy array
        return color

    def get_depth(self, folder, frame_idx, side):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_idx))
        )
        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode="constant"
        )
        depth_gt = depth_gt.astype(np.float)
        depth_gt = np.expand_dims(depth_gt, -1)
        return depth_gt

    def get_image_path(self, folder, frame_idx, side):
        f_str = "{:010d}{}".format(frame_idx, ".png")
        image_path = os.path.join(self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def __getitem__(self, item):
        line = self.filenames[item].split()
        folder = line[0]
        frame_idx = int(line[1])
        side = line[2]
        rgb, gt, _ = self.transform(self.get_color(folder, frame_idx, side), self.get_depth(folder, frame_idx, side), None, None)
        inputs = {"rgb": self.to_tensor(rgb).float(), "gt": self.to_tensor(gt).float(), "d": self.to_tensor(gt).float()}
        return inputs

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    from options import args
    import matplotlib.pyplot as plt

    # args.val = "full"
    # dataset_val_full = KittiDepth(split="val", args=args)
    # args.val = "select"
    # dataset_val_select = KittiDepth(split="val", args=args)
    # print(len(dataset_val_full))
    # print(len(dataset_val_select))
    # loader = DataLoader(dataset=dataset_val_full, batch_size=1, shuffle=False)
    # # print(len(loader))
    # sum_valid_mask = 0
    # sum_total = 0
    # for idx, data in enumerate(loader):
    #     gt = data["gt"]
    #     valid_mask = gt > 0
    #     sum_valid_mask += valid_mask.sum().item()
    #     sum_total += gt.sum().item()
    #     print(data["rgb"].shape, data["d"].shape, data["gt"].shape, data["g"].shape)
    #     print(data["rgb"].min(), data["d"].min(), data["gt"].min(), data["g"].min())
    #     print(data["rgb"].max(), data["d"].max(), data["gt"].max(), data["g"].max())
    #     break
    #     print(data["gt"].size()[0])
    #     rgb = data["rgb"].numpy()
    #     rgb = rgb[0].transpose([1, 2, 0])
    #     depth = data["d"].numpy()
    #     depth = depth[0].transpose([1, 2, 0])
    #     gt = data["gt"].numpy()
    #     gt = gt[0].transpose([1, 2, 0])
    #     gray = data["g"].numpy()
    #     gray = gray[0].transpose([1, 2, 0])
    #     plt.figure(figsize=(4, 4))
    #
    #     plt.subplot(2, 2, 1)
    #     plt.title("rgb")
    #     plt.imshow(data["rgb"][0].permute(1, 2, 0) / 255)
    #     plt.subplot(2, 2, 2)
    #     plt.title("depth")
    #     plt.imshow(data["d"][0].permute(1, 2, 0))
    #     plt.subplot(2, 2, 3)
    #     plt.title("gt")
    #     plt.imshow(data["gt"][0].permute(1, 2, 0))
    #     plt.subplot(2, 2, 4)
    #     plt.title("gray")
    #     plt.imshow(data["g"][0].permute(1, 2, 0), cmap="gray")  # 这里如果不加cmap参数，就会以热量图的形式显示。
    #     plt.axis("off")
    #     plt.show()

    # print(sum_valid_mask)
    # print(sum_total)
    # 497492214
    # 7811417114.28125
    """ test KittiPredictionDepth dataset """
    pred = KittiPrediction(eval_split="eigen", mode="train", args=args)
    print(len(pred))
    loader = DataLoader(pred, batch_size=1, shuffle=True)
    for index, data in enumerate(loader):
        print(index, data["rgb"].shape, data["gt"].shape, data["d"].shape, data["g"].shape, data["s100"].shape)
        print(data["rgb"].max(), data["rgb"].min(), data["rgb"].mean())
        print("gt     ", data["gt"].max(), data["gt"].min(), data["gt"].mean(), np.count_nonzero(data["gt"].numpy()))
        print("64线： ", data["d"].max(), data["d"].min(), data["d"].mean(), np.count_nonzero(data["d"].numpy()))
        print("gray   ", data["g"].max(), data["g"].min(), np.count_nonzero(data["g"].numpy()))
        print("sparse 100 ", data["s100"].max(), data["s100"].min(), data["s100"].mean(), np.count_nonzero(data["s100"].numpy()))

        rgb = data["rgb"].numpy()
        rgb = rgb[0].transpose([1, 2, 0])
        gt = data["gt"].numpy()
        gt = gt[0].transpose([1, 2, 0])
        d = data["d"].numpy()
        d = d[0].transpose([1, 2, 0])
        g = data["g"].numpy()
        g = g[0].transpose([1, 2, 0])
        s100 = data["s100"].numpy()
        s100 = s100[0].transpose([1, 2, 0])

        plt.figure(figsize=(2, 2))

        plt.subplot(4, 4, 1)
        plt.title("rgb")
        plt.imshow(rgb / 255)

        plt.subplot(2, 2, 1)
        plt.title("depth")
        plt.imshow(d)

        plt.subplot(2, 2, 2)
        plt.title("gt")
        plt.imshow(gt)
        #
        # plt.subplot(4, 4, 4)
        # plt.title("g")
        # plt.imshow(g, cmap="gray")

        plt.subplot(2, 2, 3)
        plt.title("s100")
        plt.imshow(s100)

        plt.axis("off")
        plt.show()
        break

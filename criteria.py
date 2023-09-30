# @FileName : criteria.py
# @Software : PyCharm
import torch
from torch import nn
import torch.nn.functional as F

loss_names = ['l1', 'l2']


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = 0.0

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        pred = torch.clamp(F.interpolate(pred, [352, 1216], mode="bilinear", align_corners=False), min=1e-3, max=80)

        diff = target - pred
        diff1 = diff[valid_mask]
        self.loss = (diff1 ** 2).mean()

        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.loss = 0.0

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.loss = 0.0
        self.variance_focus = 0.85

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        pred = torch.clamp(F.interpolate(pred, [352, 1216], mode="bilinear", align_corners=False), min=1e-3, max=80)
        d = torch.log(pred[valid_mask]) - torch.log(target[valid_mask])
        self.loss = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

        return self.loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = 0.0

    def forward(self, reconstructed_gray_img, gt_gray_img):
        assert reconstructed_gray_img.dim() == gt_gray_img.dim(), "inconsistent dimensions"
        valid_mask = gt_gray_img >= 0
        diff = reconstructed_gray_img[valid_mask] - gt_gray_img[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device="cuda", temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool).to(device)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss


if __name__ == '__main__':
    contra = ContrastiveLoss(4, "cpu")
    # rgb_e9 = torch.randint(0, 100, (16, 128)).float()
    # depth = torch.randint(0, 100, (16, 128)).float()
    a, b = torch.rand(4, 512), torch.rand(4, 512)
    l = contra(a, b)
    print(l)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import uuid


class PointNetEncoder(nn.Module):
    def __init__(self, channel=3, d_model=512):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, d_model, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        """
        :param x: torch tensor [B, D, N]
        :return: torch tensor [B, d_model, N]
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


class PointNetDecoder(nn.Module):
    def __init__(self, channel=3, d_model=512):
        super(PointNetDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(64, channel, 1)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.conv3 = torch.nn.Conv1d(d_model, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(channel)

    def forward(self, x):
        """
        :param x: torch tensor [B, D, N]
        :return: torch tensor [B, d_model, N]
        """
        x = F.leaky_relu(self.bn1(self.conv3(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.conv1(x)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def chamfer_loss(x, y, ps=91):
    """
    compute chamfer_loss between two point cloud
    :param x: tensor. [B,C,N]
    :param y: tensor. [B,C,N]
    :param ps:
    :return: torch.float
    """

    A = x.permute(0, 2, 1)
    B = y.permute(0, 2, 1)
    r = torch.sum(A * A, dim=2)
    r = r.unsqueeze(-1)
    r1 = torch.sum(B * B, dim=2)
    r1 = r1.unsqueeze(-1)

    temp1 = r.repeat(1, 1, ps)
    temp2 = -2 * torch.bmm(A, B.permute(0, 2, 1))
    temp3 = r1.permute(0, 2, 1).repeat(1, ps, 1)
    t = temp1 + temp2 + temp3
    d1, _ = t.min(dim=1)
    d2, _ = t.min(dim=2)
    ls = (d1 + d2) / 2
    return ls.mean()


def gaussian_mix_loss(x, y, var=1, ps=91, w=0, sigma=20):
    """

    :param x: tensor. [B,C,N]
    :param y: tensor. [B,C,N]
    :param var:
    :param ps:
    :param w:
    :param sigma:
    :return: torch.float
    """
    # center is B
    A = x.permute(0, 2, 1)
    B = y.permute(0, 2, 1)
    bs = A.shape[0]
    ps = A.shape[1]
    A = (A.unsqueeze(2)).repeat(1, 1, ps, 1)
    B = (B.unsqueeze(1)).repeat(1, ps, 1, 1)
    sigma_inverse = ((torch.eye(2) * (1.0 / var)).unsqueeze(0).unsqueeze(0).unsqueeze(0)).repeat(
        [bs, ps, ps, 1, 1]).cuda()
    sigma_inverse = sigma * sigma_inverse
    sigma_inverse = sigma_inverse.view(-1, 2, 2)
    tmp1 = (A - B).unsqueeze(-2).view(-1, 1, 2)
    tmp = torch.bmm(tmp1, sigma_inverse)
    tmp = torch.bmm(tmp, tmp1.permute(0, 2, 1))
    tmp = tmp.view(bs, ps, ps)
    tmp = torch.exp(-0.5 * tmp)
    tmp = tmp / (2 * np.pi * var)
    tmp1 = tmp.sum(dim=-1)
    tmp2 = tmp.sum(dim=1)
    tmp1 = torch.clamp(tmp1, min=0.01)
    return (-torch.log(tmp1 / 90.0)).mean()


def PC_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    compute euclidean distance between two point cloud
    :param x: point cloud x [B,C,N]
    :param y:point cloud y [B,C,N]
    :return: [B,1]
    """
    return torch.mean((x - y) ** 2, [1, 2])

def save_pc2visual(save_dir,epoch,point_set1,point_set2,warped):
    id=str(uuid.uuid4())
    save_path=os.path.join(save_dir,str(epoch)+'epoch_'+ id)
    os.mkdir(save_path)
    l=point_set1.size()[0]
    point_set1,point_set2,warped=point_set1.cpu().numpy(),point_set2.cpu().numpy(),warped.cpu().numpy()
    for i in range(l):
        np.savetxt(os.path.join(save_path,str(i)+"pc1.txt"),point_set1[i])
        np.savetxt(os.path.join(save_path,str(i)+"pc2.txt"),point_set2[i])
        np.savetxt(os.path.join(save_path,str(i)+"warped.txt"),warped[i])

import ctypes
from torch.autograd import Function
lib=ctypes.cdll.LoadLibrary("libmorton/encode.so")
lib.encode.restype=ctypes.c_uint64

def z_order_encode(inputs):
    shape=list(inputs.shape)
    shape[-1]=1
    code=np.ndarray(shape,dtype=np.uint64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x,y,z=inputs[i,j].tolist()
            code[i,j]=lib.encode(x,y,z)
    return code.astype(np.float64)

class Z_order_sorting(Function):
    @staticmethod
    def forward(ctx,xyz,normal):
        data=((xyz+2)*4096).cpu().numpy()
        data = data.astype(dtype=np.uint32)
        assert data.shape[-1] == 3
        z_order_code=torch.from_numpy(z_order_encode(data)).cuda()
        _,idx=torch.sort(z_order_code,dim=1)
        batch_idx=torch.arange(xyz.shape[0]).reshape(xyz.shape[0],1,1)
        return xyz[batch_idx,idx].squeeze(2),normal[batch_idx,idx].squeeze(2)

    @staticmethod
    def backward(ctx,grad_out):
        return ()

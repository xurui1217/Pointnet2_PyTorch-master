import torch
import torch.nn as nn
from pointnet2.utils import pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import (
    PointnetSAModuleMSG, PointnetSAModule,PointnetSAModule_att,PointnetSAModuleMSG_att
)
from collections import namedtuple
import sys


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to('cuda')
            # inputs = inputs.to('cuda', non_blocking=True)
            labels = labels.to('cuda')
            # print('FFFUUU')
            preds = model(inputs)  # (16,40)
            # print(preds.size())
            # sys.exit()
            labels = labels.view(-1)  # [16]
            loss = criterion(preds, labels)

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

            return ModelReturn(
                preds, loss, {
                    "acc": acc.item(),
                    'loss': loss.item()
                }
            )

    return model_fn


class Pointnet2MSG_att(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=3, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG_att(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[15, 32, 128],
                mlps=[[input_channels, 32, 32,
                       64], [input_channels, 64, 64, 128],
                      [input_channels, 64, 96, 128]],
                use_xyz=use_xyz
            )
        )
        # xyz=torch.Size([16, 2048, 3])
        # new_features=torch.Size([16, 64+128+128, 512])
        # 先512个点来一轮特征提取
        # 输出new_xyz(16,512,3),new_features(16, 64+128+128=320, 512)
        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG_att(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[[input_channels, 64, 64,
                       128], [input_channels, 128, 128, 256],
                      [input_channels, 128, 128, 256]],
                use_xyz=use_xyz
            )
        )
        # 在用128个点再来一轮特征提取,这里的输入是new_xyz和new_features
        # 输出new_xyz(16,128,3)
        # new_features=torch.Size([16, 128+256+256, 128])
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024], use_xyz=use_xyz
            )
        )
        # 用mlp提取这128个点的特征，包括之前的特征和128个点的坐标
        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 256, bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(256, num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        # print(xyz.size()) #16,2048,3

        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # print('pointcloud size:',pointcloud.size()) #16,2048,3
        xyz, features = self._break_up_pc(pointcloud)
        # print(features.size())
        # exit()
        for i, module in enumerate(self.SA_modules):
            xyz, features = module(xyz, features)
            # print('layer name is:{},xyz is:{},features is:{}.'.format(i,xyz.size(),features.size()))
        # final feature is torch.Size([16, 1024, 1])->FC_layer(1024,512,256,40)特征提取出来直接进FC层
        return self.FC_layer(features.squeeze(-1))

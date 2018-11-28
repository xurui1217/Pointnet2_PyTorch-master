import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/media/jcc/xr/xrhh/3D/program/Pointnet2_PyTorch-master/')
from pointnet2.utils import pointnet2_utils
from pointnet2.utils import pytorch_utils as pt_utils
from typing import List

class FC_ATT(nn.Module):
    def __init__(self,k=320):
        super(FC_ATT, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512,k)
        #self.fc3 = nn.Linear(256, 9)
        self.relu = nn.PReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        #self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):#16,3,512
        batchsize = x.size()[0]#16
        x = self.relu(self.bn1(self.conv1(x)))#16,64,512
        #print(x.size())
        x = self.relu(self.bn2(self.conv2(x)))#16,128,512
        x = self.relu(self.bn3(self.conv3(x)))#16,1024,512
        x = torch.max(x, 2, keepdim=True)[0]#16,1024,1
        x = x.view(-1, 1024)#16,1024
        x = self.relu(self.bn4(self.fc1(x)))#16,512
        #print(x.size())
        #x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc2(x)#16,k=320
        return x



class _PointnetSAModuleBase_att(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.fc_att=None



    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()  #cls_torch.Size([16, 3, 2048])
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped,
            pointnet2_utils.furthest_point_sample(xyz, self.npoint)  #存放最大搜索idx（2,2)
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  #把最大搜索idx的坐标放在new_xyz里（2,2,3),其他的点都不要了.
        #seg-torch.Size([32, 1024, 3])
        #test_cls(16,512,3)
        #attention提取过程+++++++++++++++++++++++++++++++++++++++++++
        '''
        new_xyz_att=new_xyz.transpose(1,2).contiguous().unsqueeze(3)
        #torch.Size([16, 3, 512, 1])
        att_feature=self.mlps_att[0](new_xyz_att).squeeze(3).transpose(1,2).contiguous()
        #1torch.Size([16,512,1024])
        #2torch.Size([16, 128, 1024])
        att_feature=self.FC_att(att_feature).transpose(1,2).contiguous()
        #(16,128+128+64=320,512)
        '''#失败，把模型写成class模式继续尝试
        xyz_att=new_xyz.transpose(1, 2).contiguous()
        #16,3,512  #2:16,3,128
        att_feature=self.fc_att(xyz_att)
        #16,320
        att_feature= F.sigmoid(att_feature)
        att_feature = att_feature.unsqueeze(2).repeat(1, 1, self.npoint)  # 32,1024,2500
        #16,320
        #torch.Size([16, 320, 512])把特征放在第二个维度才能实现sigmoid
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
            #groupall() torch.Size([16, 643, 1, 128])
            #test_cls_torch.Size([16, 3, 512, 15])
            new_features = self.mlps[i](
                new_features
            )
            # (B, mlp[-1], npoint, nsample)
            #16,,1024,512,15
            # #layer2 torch.Size([16, 1024, 1, 128])
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)  #(2,3,2,1) 选6个点中的max feature作为这几个点的特征
             #layer2 torch.Size([16, 1024, 1, 1])
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint) #（2,3,2）

            new_features_list.append(new_features)
        features_cat=torch.cat(new_features_list, dim=1)
        features_end=features_cat.mul(att_feature)
        #torch.Size([16, 320, 512])
        #用sigmoid函数来进行注意力机制的提取
        return new_xyz, features_end


class PointnetSAModuleMSG_att(_PointnetSAModuleBase_att):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            npoint: int,
            radii: List[float], #radii=[0.1, 0.2, 0.4],
            nsamples: List[int], #nsamples=[15, 32, 128],
            mlps: List[List[int]],
            bn: bool = True,
            use_xyz: bool = True
    ):
        #print('mlps is:',mlps)
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint #512
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        mlps_out=0
        for i in range(len(radii)):
            mlps_out += mlps[i][-1]
        #print(mlps_out)
        self.fc_att = FC_ATT(k=mlps_out)

        '''这里写的有点问题，没法前向传递
        self.FC_att = nn.Sequential(
            FC_ATT(1024, 512, bn=True),
            nn.Dropout(p=0.5),
            FC_ATT(512, mlps_out, activation=None)
        )
        '''
        '''
        self.FC_att = nn.Sequential(
            ('fc_att1',nn.Linear(1024,512,bias=False)),
            nn.BatchNorm1d(512),
            nn.PReLU(inplace=True),
            nn.Dropout(p=0.5),
            ('fc_att2',nn.Linear(512,mlps_out)),
        )
        
        #这里的MLP层数是一个可以调节的参数，可以后期调参试试看！！！
        
        mlp_att = [3, 256, 512, 1024]
        self.mlps_att.append(pt_utils.SharedMLP(mlp_att, bn=bn))
        '''


        for i in range(len(radii)): #i=0,1,2
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3  #input_channel=3+3=6

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

class PointnetSAModule_att(PointnetSAModuleMSG_att):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True
    ):
        super().__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz
        )



class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()  #torch.Size([32, 3, 4096])
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped,
            pointnet2_utils.furthest_point_sample(xyz, self.npoint)  #存放最大搜索idx（2,2)
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  #把最大搜索idx的坐标放在new_xyz里（2,2,3),其他的点都不要了.
        #(16,512,3)
        #seg-torch.Size([32, 1024, 3])
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)  #test1(2,6+3,2,6) #test2(2,6+3,2,3)
            #mlp[0](16,3,512,15)  #groupall() torch.Size([16, 643, 1, 128])
            new_features = self.mlps[i](
                new_features
            )  # (B, mlp[-1], npoint, nsample) #(2,3,2,6)  #layer2 torch.Size([16, 1024, 1, 128])
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)  #(2,3,2,1) 选6个点中的max feature作为这几个点的特征
             #layer2 torch.Size([16, 1024, 1, 1])
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint) #（2,3,2）

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            npoint: int,
            radii: List[float], #radii=[0.1, 0.2, 0.4],
            nsamples: List[int], #nsamples=[15, 32, 128],
            mlps: List[List[int]],
            bn: bool = True,
            use_xyz: bool = True
    ):
        #print('mlps is:',mlps)
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint #512
        self.groupers = nn.ModuleList() 
        self.mlps = nn.ModuleList()
        for i in range(len(radii)): #i=0,1,2
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3  #input_channel=3+3=6 

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True
    ):
        super().__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor,
            unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known ftest_moduleeatures
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *known_feats.size()[0:2], unknown.size(1)
            )

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats],
                                   dim=1)  #(B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(16, 512, 3).cuda(), requires_grad=True)
    #xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)
    #xyz_feats=xyz_feats.transpose(1,2).contiguous()

    '''
    xyz_feats=None
    test_module = PointnetSAModuleMSG_att(
        npoint=512, 
        radii=[0.1, 0.2, 0.4],
        nsamples=[15,32,128],
        mlps=[[0, 32, 32,
        64], [0, 64, 64, 128],
        [0, 64, 96, 128]],
        use_xyz=True,
    )
    '''


    xyz_feats=torch.randn(16,320,512).cuda()
    input_channels=320
    test_module = PointnetSAModuleMSG_att(
        npoint = 128,
        radii = [0.2, 0.4, 0.8],
        nsamples = [32, 64, 128],
        mlps = [[input_channels, 64, 64,
                 128], [input_channels, 128, 128, 256],
                [input_channels, 128, 128, 256]],
        use_xyz = True,
    )


    print('PointnetSAModuleMSG_att is:',test_module)
    test_module.cuda()
    #print(test_module(xyz, xyz_feats))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        new_xyz, new_features = test_module(xyz, xyz_feats)  #xyz=torch.Size([16, 2048, 3])
        print(new_features.size()) #new_features=torch.Size([16, 320, 512])
        new_features.backward(
            torch.cuda.FloatTensor(*new_features.size()).fill_(1)
        )
        print(new_features.size())
        print(xyz.grad.size())

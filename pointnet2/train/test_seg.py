import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import numpy as np
import os
import sys
sys.path.append('/media/jcc/xr/xrhh/3D/program/Pointnet2_PyTorch-master/')
from pointnet2.models import Pointnet2SemMSG as Pointnet
from pointnet2.models.pointnet2_msg_sem import model_fn_decorator
from pointnet2.data import Indoor3DSemSeg
import pointnet2.utils.pytorch_utils as pt_utils

import argparse

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-batch_size", type=int, default=32, help="Batch size [default: 32]"
)
parser.add_argument(
    "-num_points",
    type=int,
    default=4096,
    help="Number of points to train with [default: 2048]"
)
parser.add_argument(
    "-weight_decay",
    type=float,
    default=0,
    help="L2 regularization coeff [default: 0.0]"
)
parser.add_argument(
    "-lr",
    type=float,
    default=1e-2,
    help="Initial learning rate [default: 1e-2]"
)
parser.add_argument(
    "-lr_decay",
    type=float,
    default=0.5,
    help="Learning rate decay gamma [default: 0.5]"
)
parser.add_argument(
    "-decay_step",
    type=float,
    default=2e5,
    help="Learning rate decay step [default: 20]"
)
parser.add_argument(
    "-bn_momentum",
    type=float,
    default=0.9,
    help="Initial batch norm momentum [default: 0.9]"
)
parser.add_argument(
    "-bn_decay",
    type=float,
    default=0.5,
    help="Batch norm momentum decay gamma [default: 0.5]"
)
parser.add_argument(
    "-checkpoint", type=str, default='/media/jcc/xr/xrhh/3D/program/Pointnet2_PyTorch-master/pointnet2/train/checkpoints/poitnet2_semseg_best.pth.tar', help="Checkpoint to start from"
)
parser.add_argument(
    "-epochs", type=int, default=200, help="Number of epochs to train for"
)
parser.add_argument(
    "-run_name",
    type=str,
    default="sem_seg_run_1",
    help="Name for run in tensorboard_logger"
)
parser.add_argument('--visdom-port', type=int, default=8097)


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parser.parse_args()

    test_set = Indoor3DSemSeg(args.num_points, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    train_set = Indoor3DSemSeg(args.num_points)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
        shuffle=True
    )

    model = Pointnet(num_classes=13, input_channels=6, use_xyz=True)
    #model = nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    lr_lbmd = lambda it: max(args.lr_decay**(int(it * args.batch_size / args.decay_step)), lr_clip / args.lr)
    bnm_lmbd = lambda it: max(args.bn_momentum * args.bn_decay**(int(it * args.batch_size / args.decay_step)), bnm_clip)

    if args.checkpoint is None:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
        bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
        start_epoch = 1
        best_prec = 0
        best_loss = 1e10
    else:
        print(args.checkpoint.split(".")[0])
        _,start_epoch, best_prec, best_loss = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0]
        )
        print('start from epoch:',start_epoch)
        print('best loss is:',best_loss)
        lr_scheduler = lr_sched.LambdaLR(
            optimizer, lr_lbmd, last_epoch=start_epoch
        )
        bnm_scheduler = pt_utils.BNMomentumScheduler(
            model, bnm_lmbd, last_epoch=start_epoch
        )

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    viz = pt_utils.VisdomViz(port=args.visdom_port)
    viz.text(str(vars(args)))

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name="/media/jcc/xr/xrhh/3D/program/Pointnet2_PyTorch-master/pointnet2/train/checkpoints/pointnet2_smeseg",
        best_name="/media/jcc/xr/xrhh/3D/program/Pointnet2_PyTorch-master/pointnet2/train/checkpoints/poitnet2_semseg_best",
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        viz=viz
    )


    '''
    trainer.train(
        0,
        start_epoch,
        args.epochs,
        train_loader,
        test_loader,
        best_loss=best_loss
    )
    '''

    val_loss,eval_dict_test,acc_all= trainer.eval_epoch(test_loader)
    print('best_prec is:', best_prec)
    print('best_loss is:', best_loss)
    print('val_loss is:',val_loss)
    print('eval_dict_test is:',eval_dict_test)
    print('acc_all is:',acc_all)
    #viz.update('val_test', it, res)
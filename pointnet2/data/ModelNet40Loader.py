import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py, subprocess, shlex
from visdom import Visdom
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR='/media/jcc/xr/xrhh/3D/data'

viz = Visdom(port=8097, server="http://localhost")

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f] 


def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


class ModelNet40Cls(data.Dataset):

    def __init__(self, num_points, transforms=None, train=True, download=False):
        super().__init__()

        self.transforms = transforms

        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        #self.data_dir="/media/jcc/xr/xrhh/3D/data/modelnet40_ply_hdf5_2048"
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train, self.num_points = train, num_points
        if self.train:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))#['modelnet40_ply_hdf5_2048/ply_data_train0.h5','modelnet40_ply_hdf5_2048/ply_data_train1.h5',....]

        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))#['modelnet40_ply_hdf5_2048/ply_data_test0.h5','...']

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        #print(self.points.shape)
        self.labels = np.concatenate(label_list, 0)
        #print(self.labels.shape)
        self.randomize()

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.actual_number_of_points)#2048
        np.random.shuffle(pt_idxs)
        #print(pt_idxs)
        current_points = self.points[idx, pt_idxs].copy() #the idx's 2048 points belongs to one label
        #print(idx)
        #print(pt_idxs)
        #print(current_points)
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = pts
        self.actual_number_of_points = pts

    def randomize(self):
        self.actual_number_of_points = min(
            max(
                np.random.randint(self.num_points * 0.8, self.num_points * 1.2),
                1
            ), self.points.shape[1]
        )


if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter()
    ])
    dset = ModelNet40Cls(4096, train=True, transforms=transforms)
    print(dset[0][0].size()) #16,3
    print(dset[0][1]) #n
    print(len(dset)) #9840
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    print('FFF')
    train_set = ModelNet40Cls(4096, transforms=transforms)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    viz = Visdom(port=8097, server="http://localhost",env='data')

    i=0
    for batch in train_loader:
        i+=1
        #print('i is:',i)
        point,label=batch
        #print(point.size()) #16,2048,3
        viz.scatter(
            X=point[7],
            Y=np.ones(2048),
            opts=dict(
                legend=[str(label[7])],
                markersize=2,
                xtickmin=0,
                xtickmax=2,
                xlabel='Arbitrary',
                xtickvals=[0, 0.75, 1.6, 2],
                ytickmin=0,
                ytickmax=2,
                ytickstep=0.5,
                ztickmin=0,
                ztickmax=1,
                ztickstep=0.5,
            )
        )
        #print(label.size()) #16,1
        #print(label[7])
        if i==10:
            sys.exit()





import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py, subprocess, shlex
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR='/media/jcc/xr/xrhh/3D/data'

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]  #<class 'tuple'>: (1000, 4096, 9)
    label = f['label'][:]  #<class 'tuple'>: (1000, 4096)
    return data, label


class Indoor3DSemSeg(data.Dataset):

    def __init__(self, num_points, train=True, download=False, data_precent=1.0):
        super().__init__()
        self.data_precent = data_precent
        self.folder = "indoor3d_sem_seg_hdf5_data"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"

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

        all_files = _get_data_files(
            os.path.join(self.data_dir, "all_files.txt")
        )
        #list['indoor3d_sem_seg_hdf5_data/ply_data_all_0.h5','indoor3d_sem_seg_hdf5_data/ply_data_all_1.h5'.....]
        room_filelist = _get_data_files(
            os.path.join(self.data_dir, "room_filelist.txt")
        )
        #list['Area_1_conferenceRoom_1','Area_1_conferenceRoom_1'.....]
        data_batchlist, label_batchlist = [], []
        for f in all_files:
            d, l = _load_data_file(os.path.join(BASE_DIR, f))
            data_batchlist.append(d)
            label_batchlist.append(l)

        data_batches = np.concatenate(data_batchlist, 0)  #<class 'tuple'>: (23585, 4096, 9)
        labels_batches = np.concatenate(label_batchlist, 0)  #<class 'tuple'>: (23585, 4096)

        test_area = 'Area_5'
        train_idxs, test_idxs = [], []
        for i, room_name in enumerate(room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        if self.train:
            self.points = data_batches[train_idxs, ...]
            self.labels = labels_batches[train_idxs, ...]
        else:
            self.points = data_batches[test_idxs, ...]
            self.labels = labels_batches[test_idxs, ...]

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()
                                         ).type(torch.FloatTensor)
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()
                                         ).type(torch.LongTensor)

        return current_points, current_labels

    def __len__(self):
        return int(self.points.shape[0] * self.data_precent)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = Indoor3DSemSeg(16, train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())

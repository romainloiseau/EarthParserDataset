import copy
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_scatter
from numpy.lib.recfunctions import structured_to_unstructured
from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm

from .base import BaseDataModule
from .transforms import (CenterCrop, GridSampling, MaxPoints,
                         RandomCropSubTileTrain, RandomCropSubTileVal,
                         RandomFlip, RandomRotate, RandomScale, ZCrop)
from .utils import color as color
from .utils.labels import apply_learning_map, from_sem_to_color


class LidarHDSplit(InMemoryDataset):
    def __init__(self, options, mode):
        self.options = copy.deepcopy(options)
        self.options.data_dir = osp.join(self.options.data_dir, self.options.name)
        self.mode = mode

        self.feature_normalizer = torch.tensor(
            [[self.options.max_xy, self.options.max_xy, self.options.max_z]])

        super().__init__(
            self.options.data_dir,
            transform=self.get_transform(),
            pre_transform=self.get_pre_transform(),
            pre_filter=None
        )

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.point_y = apply_learning_map(
            self.data.point_y, self.options.learning_map)

        if mode in ["train", "val"]:
            self.items_per_epoch = int(self.options.items_per_epoch[mode] / (len(self.slices["pos"]) - 1 if self.slices is not None else 1))
        else:
            self.prepare_test_dataset()

        if mode == "val":
            self.load_all_items()

    def prepare_test_dataset(self):
        self.items_per_epoch = []
        self.tiles_unique_selection = {}
        self.tiles_min_z = {}
        self.from_idx_to_tile = {}
        idx = 0
        for i in range(self.__superlen__()):
            unique_i, inverse_i = torch.unique((self.__getsuperitem__(
                    i).pos[:, :2] / self.options.max_xy).int(), dim=0, return_inverse=True)
            self.items_per_epoch.append(unique_i.shape[0])
            self.tiles_unique_selection[i] = unique_i
            self.tiles_min_z[i] = torch_scatter.scatter_min(
                    self.__getsuperitem__(i).pos[:, 2], inverse_i, dim=0)[0]
            for j in range(unique_i.shape[0]):
                self.from_idx_to_tile[idx] = (i, unique_i[j])
                idx += 1

    def load_all_items(self):
        self.items = [super(LidarHDSplit, self).__getitem__(int(i / self.items_per_epoch)) for i in range(len(self))]
        del self.data

    def get_pre_transform(self):
        pre_transform = []

        if self.mode == "test":
            pre_transform.append(
                GridSampling(
                    self.options.pre_transform_grid_sample /
                    4. if not "windturbine" in self.options.name else self.options.pre_transform_grid_sample
                )
            )
        else:
            pre_transform.append(GridSampling(
                self.options.pre_transform_grid_sample))
        pre_transform = T.Compose(pre_transform)
        return pre_transform

    def get_transform(self):
        if self.mode == "train":
            transform = [
                RandomScale(tuple(self.options.random_scales)),
                RandomCropSubTileTrain(2.**.5 * self.options.max_xy),
                RandomRotate(degrees=180., axis=2),
                RandomFlip(0, self.options.max_xy),
                CenterCrop(self.options.max_xy),
                ZCrop(self.options.max_z),
                T.RandomTranslate(self.options.random_jitter)
            ]
        elif self.mode == "val":
            transform = [
                RandomCropSubTileVal(self.options.max_xy),
                ZCrop(self.options.max_z)
            ]
        elif self.mode == "test":
            transform = []
        else:
            raise NotImplementedError(
                f"Mode {self.mode} not implemented. Should be in ['train', 'val', 'test']")
        
        if self.mode in ["train", "val"]:
            if self.options.N_scene != 0:
                transform.append(T.FixedPoints(self.options.N_scene))
            else:
                transform.append(MaxPoints(self.options.N_max))

        transform = T.Compose(transform)
        return transform

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.mode})"

    @property
    def raw_file_names(self):
        laz_dirs = []
        for tile in os.listdir(osp.join(self.options.data_dir, "tiles")):
            tile_dir = osp.join(self.options.data_dir, "tiles", tile)
            if osp.isdir(tile_dir):
                for subtile in os.listdir(tile_dir):
                    if subtile.split(".")[-1] in ["las", "laz"]:
                        laz_dirs.append(osp.join(tile_dir, subtile))
            if tile_dir.split(".")[-1] in ["las", "laz"]:
                laz_dirs.append(tile_dir)

        return laz_dirs

    @property
    def processed_file_names(self):
        return [f'grid{1000*self.options.pre_transform_grid_sample:.0f}mm_data.pt']

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.mode if self.mode != "val" else "train")

    def process(self):
        data_list = []

        import pdal
        for laz in tqdm(self.raw_file_names):
            list_pipeline = [pdal.Reader(laz)]
            pipeline = pdal.Pipeline()
            for p in list_pipeline:
                pipeline |= p
            count = pipeline.execute()
            arrays = pipeline.arrays

            label = structured_to_unstructured(arrays[0][["Classification"]])
            for l in np.unique(label):
                if l not in self.options.raw_class_names.keys():
                    label[label == l] = 1

            xyz = structured_to_unstructured(arrays[0][["X", "Y", "Z"]])
            intensity = structured_to_unstructured(arrays[0][["Intensity"]])

            if "Red" in arrays[0].dtype.names:
                rgb = structured_to_unstructured(
                    arrays[0][["Red", "Green", "Blue"]])

                if rgb.max() >= 256:
                    rgb = (rgb / 2**8).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
            else:
                rgb = np.zeros_like(xyz, dtype=np.uint8)

            intensity = intensity.clip(np.percentile(intensity.flatten(), .1), np.percentile(intensity.flatten(), 99.9))
            intensity = (intensity / intensity.max()).astype(np.float32)

            label = label.astype(np.int64)

            xyz -= xyz.min(0)
            xyz = xyz.astype(np.float32)
            maxi = xyz.max(0)

            if self.options.subtile_max_xy > 0:
                n_split = 1 + \
                        (maxi / ((1 + (self.mode == "test")) * self.options.subtile_max_xy)).astype(np.int32)
            else:
                n_split = [1, 1]

            for i in range(n_split[0]):
                for j in range(n_split[1]):
                    keep = np.logical_and(
                        np.logical_and(
                            xyz[:, 0] >= i * maxi[0] / n_split[0],
                            xyz[:, 0] < (i + 1) * maxi[0] / n_split[0]
                        ),
                        np.logical_and(
                            xyz[:, 1] >= j * maxi[1] / n_split[1],
                            xyz[:, 1] < (j + 1) * maxi[1] / n_split[1]
                        )
                    )

                    data_list.append(Data(
                        pos=torch.from_numpy(xyz[keep] - xyz[keep].min(0)),
                        intensity=torch.from_numpy(intensity[keep]),
                        rgb=torch.from_numpy(rgb[keep]),
                        point_y=torch.from_numpy(label[keep]),
                    ))

                    if self.mode in ["train", "val"]:
                        data_list[-1].pos_maxi = data_list[-1].pos[:, :2].max(0)[0]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

    def __superlen__(self) -> int:
        return super().__len__()

    def __len__(self) -> int:
        if self.mode != "test":
            return self.items_per_epoch * self.__superlen__()
        else:
            return sum(self.items_per_epoch)

    def __getsuperitem__(self, idx):
        return super().__getitem__(idx)

    def __getitem__(self, idx):
        if self.mode == "train":
            item = self.__getsuperitem__(int(idx / self.items_per_epoch))
        elif self.mode == "val":
            item = self.items[idx]
        else:
            this_idx, tile = self.from_idx_to_tile[idx]
            item = self.__getsuperitem__(this_idx)
            keep = (item.pos[:, :2] / self.options.max_xy).int()
            keep = torch.logical_and(
                keep[:, 0] == tile[0], keep[:, 1] == tile[1])
            for k in item.keys:
                setattr(item, k, getattr(item, k)[keep])

            item.pos[:, :2] -= self.options.max_xy * tile
            item.pos[:, -1] -= item.pos[:, -1].min()

            keep = item.pos[:, -1] < self.options.max_z
            for k in item.keys:
                setattr(item, k, getattr(item, k)[keep])

        item.pos_lenght = torch.tensor(item.pos.size(0))

        if self.options.modality == "3D":
            pad = self.options.N_max - item.pos.size(0) if self.mode != "test" else 0
            if self.options.distance == "xyz":
                item.pos_padded = F.pad(item.pos, (0, 0, 0, pad), mode="constant", value=0).unsqueeze(0)
            elif self.options.distance == "xyzk":
                item.pos_padded = F.pad(torch.cat([item.pos, item.intensity], -1), (0, 0, 0, pad), mode="constant", value=0).unsqueeze(0)
            else:
                raise NotImplementedError(
                    f"LiDAR-HD can't produce {self.options.distance}")

        item.features = 2 * \
            torch.cat([item.rgb / 255., item.pos /
                      self.feature_normalizer, item.intensity], -1) - 1
        #del item.intensity

        if self.options.modality == "2D":
            item = self.from3Dto2Ditem(item)

        return item

    def from3Dto2Ditem(self, item):
        del item.pos_lenght
        res, n_dim = self.options.image.res, self.options.image.n_dim
        intensity = item.intensity.squeeze()
        rgb = item.rgb.float()
        labels = item.point_y.float().squeeze()
        xy, z = item.pos[:, :2],  item.pos[:, 2]
        xy = torch.clamp(torch.floor(
                xy / (self.options.max_xy / (res-0.001))), 0, res - 1)
        xy = (xy[:, 0] * res + xy[:, 1]).long()
        features = []
        for values, init in zip([z, intensity, rgb, labels], [0, 0, 0, -1]):
            for mode in ['min', 'max']:
                features.append(gather_values(
                        xy, z, values, mode=mode, res=res, init=init))
        item = torch.cat(features, dim=-1)
        item = item.reshape(res, res, n_dim)
        return item

def gather_values(xy, z, values, mode='max', res=32, init=0):
    img = (torch.ones(
        res*res, values.shape[1] if 1 < len(values.shape) else 1) * init).squeeze()
    z = z.sort(descending=mode == 'max')
    xy, values = xy[z[1]], values[z[1]]
    unique, inverse = torch.unique(xy, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(
        0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    img.index_put_((xy[perm],), values[perm], accumulate=False)
    img = img.reshape(res, res, -1)
    return img


class LidarHDDataModule(BaseDataModule):
    _DatasetSplit_ = {
        "train": LidarHDSplit,
        "val": LidarHDSplit,
        "test": LidarHDSplit
    }

    def from_labels_to_color(self, labels):
        return from_sem_to_color(apply_learning_map(labels, self.myhparams.learning_map_inv), self.myhparams.color_map)

    def get_feature_names(self):
        return ["red", "green", "blue", "x", "y", "z", "intensity"]

    def describe(self):
        print(self)

        for split in ["train", "val", "test"]:
            if hasattr(self, f"{split}_dataset") and hasattr(getattr(self, f"{split}_dataset"), "data"):
                print(f"{split} data\t", getattr(
                    self, f"{split}_dataset").data)
                if hasattr(getattr(self, f"{split}_dataset").data, "point_y"):
                    for c, n in zip(*np.unique(getattr(self, f"{split}_dataset").data.point_y.flatten().numpy(), return_counts=True)):
                        print(
                            f"class {self.myhparams.raw_class_names[self.myhparams.learning_map_inv[int(c)]]} ({c}) {(20 - len(self.myhparams.raw_class_names[self.myhparams.learning_map_inv[int(c)]]))*' '}  \thas {n} \tpoints")

        if hasattr(self, "val_dataset"):
            lens = [item.pos.size(0) for item in self.val_dataset.items]
            plt.hist(lens)
            plt.title(
                f"size of val dataset items between {min(lens)} and {max(lens)}")
            plt.show()

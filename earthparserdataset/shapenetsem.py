import copy
import os.path as osp

from torch_geometric.io import read_txt_array
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from .utils.labels import from_sem_to_color
from .base import BaseDataModule
from tqdm.auto import tqdm
import json

class MeanRemove(T.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.pos -= data.pos.mean(0)
        return data

class ShapeNetSemDataset(InMemoryDataset):
    CAT2ID = {
        "Airplane" : "02691156",
        "Bag" : "02773838",
        "Cap" : "02954340",
        "Car" : "02958343",
        "Chair" : "03001627",
        "Earphone" : "03261776",
        "Guitar" : "03467517",
        "Knife" : "03624134",
        "Lamp" : "03636649",
        "Laptop" : "03642806",
        "Motorbike" : "03790512",
        "Mug" : "03797390",
        "Pistol" : "03948459",
        "Rocket" : "04099429",
        "Skateboard" : "04225987",
        "Table" : "04379243",
    }

    def __init__(self, options, mode):
        self.options = copy.deepcopy(options)
        self.mode = mode

        transform = [
            MeanRemove()
        ]

        if mode == "train":
            transform += [
                T.RandomTranslate(0.01),
                T.RandomFlip(axis=1),
            ]

        if self.options.rotate_z:
            transform.append(T.RandomRotate(180, axis=2))
            
        super().__init__(options.data_dir, T.Compose(transform), None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.data.point_y -= self.data.point_y.min()

        self.N_point_max = int((self.slices["pos"][1:] - self.slices["pos"][:-1]).max().item())

    @property
    def raw_file_names(self):
        
        files = []
        modes = ["train", "val", "test"] if self.mode in ["train", "test"] else ["val"]
        for mode in modes:
            with open(osp.join(self.root, 'train_test_split', f'shuffled_{mode}_file_list.json'), 'r') as f:
                files += json.load(f)

        select_ids = [self.CAT2ID[cat] for cat in self.options.classes]
        file_names = []
        for file in files:
            if file.split('/')[1] in select_ids:
                file_names.append(f"{osp.join(*file.split('/')[1:])}.txt")
        
        return file_names

    @property
    def processed_file_names(self):
        return [f"{'_'.join(self.options.classes)}.pt"]
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.mode)

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for file in tqdm(self.raw_file_names):
            data = read_txt_array(osp.join(self.root, file))

            data_list.append(Data(pos=data[:, :3], point_y=data[:, -1]))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        data.pos = data.pos[:, [0, 2, 1]]
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.raw_file_names)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.features = data.pos.clone()

        data.pos *= .5 * self.options.max_xy / ((data.pos**2).sum(-1)**.5).max()
        data.pos += .5 * self.options.max_xy

        data.pos_lenght = torch.tensor(data.pos.size(0))

        data.pos_padded = F.pad(data.pos, (0, 0, 0, self.N_point_max - data.pos.shape[0])).unsqueeze(0)

        return data

class ShapeNetSemDataModule(BaseDataModule):
    _DatasetSplit_ = {
        "train": ShapeNetSemDataset,
        "val": ShapeNetSemDataset,
        "test": ShapeNetSemDataset
    }                

    def get_feature_names(self):
        return ["x", "y", "z"]

    def from_labels_to_color(self, labels):
        return from_sem_to_color(
            labels,
            self.myhparams.color_map
        )
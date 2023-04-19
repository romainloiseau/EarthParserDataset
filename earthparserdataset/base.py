from types import SimpleNamespace

import copy
import torch
import plotly.graph_objects as go
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from matplotlib import cm
import torch_scatter
import torch.nn.functional as F
import matplotlib.pyplot as plt

from hydra.utils import to_absolute_path


class BaseDataModule(pl.LightningDataModule):
    FEATURE2NAME = {"y": "Ground truth", "y_pred": "Prediction", "inst": "Ground truth instance",
                    "inst_pred": "Predicted instance", "i": "Intensity", "rgb": "RGB", "infrared": "Infrared", "xyz": "Position"}

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.myhparams = SimpleNamespace(**kwargs)
        self.myhparams.data_dir = to_absolute_path(self.myhparams.data_dir)

    def __repr__(self) -> str:

        out = f"DataModule:\t{self.__class__.__name__}"

        for split in ["train", "val", "test"]:
            if hasattr(self, f"{split}_dataset"):
                out += f"\n\t{getattr(self, f'{split}_dataset')}"

        return out

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            self.train_dataset = self._DatasetSplit_["train"](self.myhparams, "train")
            self.val_dataset = self._DatasetSplit_["val"](self.myhparams, "val")
        elif stage in (None, 'validate'):
            self.val_dataset = self._DatasetSplit_["val"](self.myhparams, "val")
        elif stage in (None, 'train'):
            self.train_dataset = self._DatasetSplit_["train"](self.myhparams, "train")

        if stage in (None, 'test'):
            self.train_dataset = self._DatasetSplit_["train"](self.myhparams, "train")
            self.test_dataset = self._DatasetSplit_["test"](self.myhparams, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.myhparams.batch_size,
            shuffle=True,
            num_workers=self.myhparams.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.myhparams.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.myhparams.num_workers
        )

    def get_label_from_raw_feature(self, c):
        return self.FEATURE2NAME[c]

    def get_color_from_item(self, item, c):
        if c == "y" and hasattr(item, "point_y"):
            color = self.from_labels_to_color(item.point_y.squeeze()).numpy()
        elif c == "y_pred" and hasattr(item, "point_y_pred"):
            try:
                color = self.from_labels_to_color(
                    item.point_y_pred.squeeze()).numpy()
            except:
                color = cm.get_cmap("tab20")(item.point_y_pred.squeeze(
                ).cpu().numpy() / item.point_y_pred.max().item())[:, :-1]
                color = (255*color).astype(np.uint8)
        elif c in ["inst_pred", "inst"] and hasattr(item, f"point_{c}"):
            color = cm.get_cmap("tab20")(np.random.permutation(getattr(item, f"point_{c}").max().item(
            )+1)[getattr(item, f"point_{c}").squeeze().cpu().numpy()] / getattr(item, f"point_{c}").max().item())[:, :-1]
            color = (255*color).astype(np.uint8)
        elif c == "i" and hasattr(item, "intensity"):
            color = item.intensity.squeeze().numpy()
            color = 0.01 + 0.98*(color - color.min()) / \
                (color.max() - color.min())
            color = cm.get_cmap("viridis")(color)[:, :-1]
            color = (255*color).astype(np.uint8)
        elif c == "rgb" and hasattr(item, "rgb"):
            color = 0.01 + 0.98*item.rgb / 255.
        else:
            color = item.pos.squeeze().numpy()
            color = 0.01 + 0.98*(color - color.min()) / \
                (color.max() - color.min())
            color = (255*color).astype(np.uint8)
        return color
    
    def show(self, item, voxelize=0, color="y;xyz;rgb;i", ps=5):
        if self.myhparams.modality == "3D":
            self.show3D(item, voxelize, color, ps)
        elif self.myhparams.modality == "2D":
            self.show2D(item)
        else:
            raise NotImplementedError(f"Modality {self.hparams.modality} not implemented")
        
    def show2D(self, item):
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        
        ax[0].imshow(item[..., 1] / 255.)
        ax[0].set_title("z max projection")
        ax[0].set_aspect("equal")
        ax[0].axis("off")

        ax[1].imshow(item[..., 3] / 255.)
        ax[1].set_title("intensity max projection")
        ax[1].set_aspect("equal")
        ax[1].axis("off")

        ax[2].imshow(item[..., [4,5,6]] / 255.)
        ax[2].set_title("rgb")
        ax[2].set_aspect("equal")
        ax[2].axis("off")

        ax[3].imshow(self.from_labels_to_color(item[..., -1]))
        ax[3].set_title("labels")
        ax[3].set_aspect("equal")
        ax[3].axis("off")

        plt.show()

    def show3D(self, item, voxelize=0, color="y;xyz;rgb;i", ps=5):

        thisitem = copy.deepcopy(item)

        if voxelize:
            choice = torch.unique(
                (thisitem.pos / voxelize).int(), return_inverse=True, dim=0)[1]
            for attr in ["point_y", "point_y_pred", "point_inst", "point_inst_pred"]:
                if hasattr(thisitem, attr):
                    setattr(thisitem, attr, torch_scatter.scatter_sum(F.one_hot(
                        getattr(thisitem, attr).squeeze().long()), choice, 0).argmax(-1).unsqueeze(0))
            for attr in ["pos", "features"]:
                if hasattr(thisitem, attr):
                    setattr(thisitem, attr, torch_scatter.scatter_mean(
                        getattr(thisitem, attr), choice, 0))
            for attr in ["intensity", "rgbi", "rgb"]:
                if hasattr(thisitem, attr):
                    setattr(thisitem, attr, torch_scatter.scatter_max(
                        getattr(thisitem, attr), choice, 0)[0])

        datadtype = torch.float16
        margin = int(0.02 * 600)
        layout = go.Layout(
            width=1000,
            height=600,
            margin=dict(l=margin, r=margin, b=margin, t=4*margin),
            uirevision=True,
            showlegend=False
        )
        fig = go.Figure(
            layout=layout,
            data=go.Scatter3d(
                x=thisitem.pos[:, 0].to(datadtype), y=thisitem.pos[:, 1].to(datadtype), z=thisitem.pos[:, 2].to(datadtype),
                mode='markers',
                marker=dict(size=ps, color=self.get_color_from_item(
                    thisitem, color.split(";")[0])),
            )
        )
        updatemenus = [
            dict(
                buttons=list([
                    dict(
                        args=[{"marker": dict(size=ps, color=self.get_color_from_item(thisitem, c))}, [
                            0]],
                        label=self.get_label_from_raw_feature(c),
                        method="restyle"
                    ) for c in color.split(";")
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=0.88,
                yanchor="top"
            ),
        ]

        fig.update_layout(updatemenus=updatemenus)
        fig.update_layout(
            scene_aspectmode='data',
        )

        fig.show()

        del thisitem
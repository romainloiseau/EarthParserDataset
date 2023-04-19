"""
Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/grid_sampling.html
"""

import re
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean, scatter_max

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class GridSampling(BaseTransform):
    r"""Clusters points into voxels with size :attr:`size`.
    Each cluster returned is a new point based on the mean of all points
    inside the given cluster.

    Args:
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
    """
    def __init__(self, size: Union[float, List[float], Tensor],
                 start: Optional[Union[float, List[float], Tensor]] = None,
                 end: Optional[Union[float, List[float], Tensor]] = None):
        self.size = size
        self.start = start
        self.end = end

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes

        batch = data.get('batch', None)

        c = torch_geometric.nn.voxel_grid(data.pos, self.size, batch,
                                          self.start, self.end)
        c, perm = torch_geometric.nn.pool.consecutive.consecutive_cluster(c)

        for key, item in data:
            if bool(re.search('edge', key)):
                raise ValueError(
                    'GridSampling does not support coarsening of edges')

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key in ['y', 'point_y', 'label', 'point_inst']:
                    item = F.one_hot(item)
                    item = scatter_add(item, c, dim=0)
                    data[key] = item.argmax(dim=-1)
                elif key == 'batch':
                    data[key] = item[perm]
                elif data[key].dtype == torch.uint8:
                    data[key] = (255. * scatter_mean(item / 255., c, dim=0)).to(torch.uint8)
                else:
                    data[key] = scatter_mean(item, c, dim=0)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'

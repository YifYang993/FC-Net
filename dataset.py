# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch.utils.data import DataLoader
import torch

from ret_benchmark.data.datasets import BaseDataSet
from ret_benchmark.data.samplers import RandomIdentitySampler
from ret_benchmark.data.transforms import build_transforms

def collate_fn(batch):
    imgs, labels = zip(*batch)
    labels = [int(k) for k in labels]
    labels = torch.tensor(labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels

def build_data(cfg, is_train=True):
    transforms = build_transforms(cfg, is_train=is_train)
    if is_train:
        dataset = BaseDataSet(cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        sampler = RandomIdentitySampler(dataset=dataset,
                                        batch_size=cfg.DATA.TRAIN_BATCHSIZE,
                                        num_instances=cfg.DATA.NUM_INSTANCES,
                                        max_iters=cfg.SOLVER.MAX_ITERS
                                        )
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_fn,
                                 batch_sampler=sampler,
                                 num_workers=cfg.DATA.NUM_WORKERS,
                                 pin_memory=True
                                 )
    else:
        dataset = BaseDataSet(cfg.DATA.TEST_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_fn,
                                 shuffle=False,
                                 batch_size=cfg.DATA.TEST_BATCHSIZE,
                                 num_workers=cfg.DATA.NUM_WORKERS
                                 )
    return data_loader


def build_datav1( is_train=True):
    transforms = build_transforms(cfg, is_train=is_train)
    if is_train:
        dataset = BaseDataSet("/home/yangyifan/code/research-ms-loss/scripts/resource/datasets/CUB_200_2011/train.txt", transforms=transforms, mode=cfg.INPUT.MODE)
        sampler = RandomIdentitySampler(dataset=dataset,
                                        batch_size=32,
                                        num_instances=5,
                                        max_iters=3000
                                        )
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_fn,
                                 batch_sampler=sampler,
                                 num_workers=8,
                                 pin_memory=True
                                 )
    else:
        dataset = BaseDataSet("/home/yangyifan/code/research-ms-loss/scripts/resource/datasets/CUB_200_2011/test.txt", transforms=transforms, mode=cfg.INPUT.MODE)
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_fn,
                                 shuffle=False,
                                 batch_size=16,
                                 num_workers=8
                                 )
    return data_loader


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a retrieval network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='config file',
        default=None,
        type=str)
    return parser.parse_args()

if __name__ == '__main__':
    import argparse
    args = parse_args()
    # cfg.merge_from_file(args.cfg_file)
    train_loader = build_datav1 (is_train=True)
    for i,j in enumerate(train_loader):
        print(i)
        print(j[0])
        print("##########")
        print(j[1])

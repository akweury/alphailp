# Created by jing at 26.06.23

import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FMEDataset(Dataset):

    def __init__(self, data_path, setname='train'):
        if setname in ['train', 'test']:
            self.data = np.array(sorted(glob.glob(str(data_path / f"*_{setname}_*.pth.tar"), recursive=True)))
        else:
            raise ValueError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        data = torch.load(self.data[item])
        x = data['x']
        y = data['y']

        return x, y, item


def create_dataloader(args):
    dataset_path = args.data_path
    train_on = args.train_on

    train_dataset = FMEDataset(dataset_path, setname='train')
    # test_dataset = SyntheticDepthDataset(dataset_path, setname='selval')
    test_dataset = FMEDataset(dataset_path, setname='test')
    # Select the desired number of images from the training set
    if train_on != 'full':
        import random
        training_idxs = np.array(random.sample(range(0, len(train_dataset)), int(train_on)))
        train_dataset.data = train_dataset.data[training_idxs]
        # test_dataset.training_case = test_dataset.training_case[np.array([0, 1, 2, 3])]
    print("train data size: " + str(train_dataset.data.shape))
    print("test data size: " + str(test_dataset.data.shape))
    # test_dataset.training_case = test_dataset.training_case[:3]
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=1)
    print('- Found {} images in "{}" folder.'.format(train_data_loader.dataset.__len__(), 'train'))
    print('- Found {} images in "{}" folder.'.format(test_data_loader.dataset.__len__(), 'test'))

    return train_data_loader, test_data_loader

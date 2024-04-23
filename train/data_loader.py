import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import torch
import pickle
torch.set_default_dtype(torch.float64)

class PCADataModule(pl.LightningDataModule):

    def __init__(self, data_path, nb_train=.6, nb_val=.2, nb_test=.2,
                 batch_size=1024, num_workers=8, **kwargs):
        """ Loads paired data samples of AIA EUV images and EVE irradiance measures.

        Input data needs to be paired.
        Parameters
        ----------
        stacks_csv_path: path to the matches
        eve_npy_path: path to the EVE data file
        """
        super().__init__()
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.data_path = data_path
        self.batch_size = batch_size
        self.nb_train = nb_train
        self.nb_val = nb_val
        self.nb_test = nb_test


    # TODO: Extract SDO dats for train/valid/test sets

    def normalization_function(self, data, props_dict):
        num_vars = data.shape[0]

        norm_data = np.zeros_like(data)
        # print(props_dict.keys())

        for el in range(num_vars):
            norm_data[el, :] = ((data[el, :] - props_dict.item()["var_" + format(el, '02d') + "_mean"])
                                  / props_dict.item()["var_" + format(el, '02d') + "_std"])

        return norm_data

    def normalize_data(self, input_data, output_data):
        input_props = np.load(self.data_path + "dbase_input_properties.npy", allow_pickle=True)
        output_props = np.load(self.data_path + "dbase_output_properties.npy", allow_pickle=True)

        input_data_norm = self.normalization_function(input_data, input_props)
        output_data_norm = self.normalization_function(output_data, output_props)

        return input_data_norm, output_data_norm

    def setup(self, stage=None):
        # load data stacks (paired samples)

        with open(self.data_path + 'MgIIhk_PCAdatabase.pickle', "rb") as f:
            tdict = pickle.load(f)

        PCA_dbase_coeff_all = tdict['PCA_dbase_coeff_all']
        PCA_dbase_model_all = tdict['PCA_dbase_model_all']

        input_flat = PCA_dbase_coeff_all.reshape(PCA_dbase_coeff_all.shape[0] * PCA_dbase_coeff_all.shape[1],
                                                 PCA_dbase_coeff_all.shape[2])

        output_flat = PCA_dbase_model_all.reshape(PCA_dbase_model_all.shape[0] * PCA_dbase_model_all.shape[1],
                                                  PCA_dbase_model_all.shape[2])

        output_indices = [0, 1, 2, 5, 10, 11, 12, 15, 16]
        output_flat = output_flat[output_indices, :]

        input_data_norm, output_data_norm = self.normalize_data(input_flat, output_flat)

        generator1 = torch.Generator().manual_seed(42)
        # ind = random_split(indices, [self.nb_train, self.nb_val, self.nb_test],
        #                    generator=generator1)

        self.train_sampler, self.val_sampler, self.test_sampler = random_split(range(input_flat.shape[1]),
                                                                               [self.nb_train, self.nb_val, self.nb_test],
                                                                               generator=generator1)

        #train_ds_input = input_data_norm[:, ind[0]]
        #train_ds_output = output_data_norm[:, ind[0]]
        #valid_ds_input = input_data_norm[:, ind[1]]
        #valid_ds_output = output_data_norm[:, ind[1]]
        #test_ds_input = input_data_norm[:, ind[2]]
        #test_ds_output = output_data_norm[:, ind[2]]

        # self.train_ds = self.datafn(train_ds_input, train_ds_output)
        # self.valid_ds = self.datafn(valid_ds_input, valid_ds_output)
        # self.test_ds  = self.datafn(test_ds_input, test_ds_output)
        self.dataset = dataFn(input_data_norm.T, output_data_norm.T)
        # print(input_data_norm.shape(), output_data_norm.shape())

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=self.test_sampler)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=self.val_sampler)


class dataFn(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1[:, 0])

    def __getitem__(self, idx):
        return self.data1[idx, :], self.data2[idx, :]

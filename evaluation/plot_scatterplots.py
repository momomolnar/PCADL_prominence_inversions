import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
from tqdm import tqdm
import pandas as pd
import datetime
import matplotlib.dates as mdates
import dateutil.parser as dt
from infer import ipredict
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


def load_test_data(work_path, test_file,
                   props_dict_input, props_dict_output):
    ''' Loader function for the test input
    :param work_path: path to the test_data
    :param test_file: name of the test_file
    :param props_dict_input: dictionary of properties of the input data
    :param props_dict_output: dictionary of properties of the output data
    :return: test_input_data , test_output_data -- data to be inverted
    :return: test_PCA_inv_model, test_PCA_Stokes_inv -- comparison data from the PCA inversion
    '''

    with open(work_path + test_file, "rb") as f:
        tdict = pickle.load(f)

    test_obs_model_all = tdict['obs_model_all']
    test_PCA_inv_model = tdict['inv_model_all']

    test_obs_C_IQUV_all = tdict['obs_C_IQUV_all']
    test_PCA_Stokes_inv = tdict['inv_C_IQUV_all']

    test_input_data_norm = normalize(test_obs_C_IQUV_all, props_dict_input)
    test_output_data_norm = normalize(test_obs_model_all, props_dict_output)

    test_input_data_norm = test_input_data_norm.reshape(test_input_data_norm.shape[0] * test_input_data_norm.shape[1],
                                                        test_input_data_norm.shape[2])

    test_output_data_norm = test_output_data_norm.reshape(test_output_data_norm.shape[0] * test_output_data_norm.shape[1],
                                                          test_output_data_norm.shape[2])

    print(f"Test input data norm: {test_input_data_norm.shape}")
    print(f"Test output data norm: {test_output_data_norm.shape}")

    output_indices = [10, 11, 12, 15, 16]

    test_output_data = torch.tensor(test_output_data_norm[output_indices, :].T)
    test_input_data  = np.zeros((test_input_data_norm.shape[0]+2,
                                 test_input_data_norm.shape[1]))

    test_input_data[0, :] = test_output_data_norm[0, :]
    test_input_data[1, :] = test_output_data_norm[1, :]
    test_input_data[2:, :] = test_input_data_norm
    test_input_data = torch.tensor(test_input_data.T)

    return test_input_data, test_obs_model_all, test_PCA_inv_model, test_PCA_Stokes_inv

# Absolute errors
def absolute_errors_scalar(v1, v2):
    return np.sqrt((v1 - v2) ** 2)


# Relative errors
def relative_errors_scalar(v1, v2):
    return np.sqrt((v1 - v2) ** 2) / np.sqrt(v1 ** 2 + 1.e-12)


# Absolute errors
def absolute_errors_vector(v1_x, v1_y, v2_x, v2_y):
    return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2)


# Relative errors
def relative_errors_vector(v1_x, v1_y, v2_x, v2_y):
    return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2) / np.sqrt(v1_x ** 2 + v1_y ** 2)


# Cosine similatiry
def cosine_similarity_vector(v1_x, v1_y, v2_x, v2_y):
    return (v2_x * v1_x + v2_y * v1_y) / (np.sqrt(v2_x ** 2 + v2_y ** 2) * np.sqrt(v1_x ** 2 + v1_y ** 2))

def normalize(data, props_dict):
    norm_data = np.zeros_like(data)

    for el in range(data.shape[1]):
        norm_data[:, el] = ((data[:, el] - props_dict.item()["var_" + format(el, '02d') + "_mean"])
                            / props_dict.item()["var_" + format(el, '02d') + "_std"])

    return norm_data

def unnormalize(data, props_dict):
    unnorm_data = torch.zeros_like(data)

    print(f"data shape: {data.shape}")
    for el in range(data.shape[1]):
        unnorm_data[:, el] = ((data[:, el] * props_dict.item()["var_" + format(el, '02d') + "_std"])
                              + props_dict.item()["var_" + format(el, '02d') + "_mean"])

    return unnorm_data

def load_valid_eve_dates(eve, line_indices=None):
    """ load all valid dates from EVE

    Parameters
    ----------
    eve: NETCDF dataset of EVE.
    line_indices: wavelengths used for data check (optional).

    Returns
    -------
    numpy array of all valid EVE dates and corresponding indices
    """
    # load and parse eve dates
    eve_date_str = eve.variables['isoDate'][:]
    # convert to naive datetime object
    eve_dates = np.array([dt.isoparse(d).replace(tzinfo=None) for d in eve_date_str])
    # get all indices
    eve_indices = np.indices(eve_dates.shape)[0]
    # find invalid eve data points
    eve_data = eve.variables['irradiance'][:]
    if line_indices is not None:
        eve_data = eve_data[:, line_indices]
    # set -1 entries to nan
    eve_data[eve_data < 0] = np.nan
    # set outliers to nan
    outlier_threshold = np.nanmedian(eve_data, 0) + 3 * np.nanstd(eve_data, 0)
    eve_data[eve_data > outlier_threshold[None]] = np.nan
    # filter eve dates and indices
    # eve_dates = eve_dates[~np.any(np.isnan(eve_data), 1)]
    # eve_indices = eve_indices[~np.any(np.isnan(eve_data), 1)]
    # eve_data = eve_data[~np.any(np.isnan(eve_data), 1)]

    return eve_data, eve_dates, eve_indices


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-checkpoint', type=str,
                   required=True, help='Path to checkpoint.')
    p.add_argument('-test_file', type=str,
                   default="allstats.pickle")
    p.add_argument('-data_dir', type=str,
                   default='/glade/work/mmolnar/PCA_inversions/',
                   help='Path to input data directory.')
    p.add_argument('-output_path', type=str,
                   default='/glade/u/home/mmolnar/Projects/PROPCA/results/',
                   help='path to save the results.')
    p.add_argument('-input_props_dict', type=str,
                   default='dbase_input_properties.npy')
    p.add_argument('-output_props_dict', type=str,
                   default='dbase_output_properties.npy')
    args = p.parse_args()

    # Selection of plot colors
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    checkpoint = args.checkpoint
    data_path = args.data_dir

    # Output path
    result_path = args.output_path
    # os.makedirs(result_path, exist_ok=True)

    # Initalize model
    state = torch.load(args.checkpoint)
    model = state['model']

    # load data properties
    output_props_dict = args.output_props_dict
    input_dict        = args.input_props_dict
    test_data_file    = args.test_file

    input_props  = np.load(data_path + output_props_dict, allow_pickle=True)
    output_props = np.load(data_path + output_props_dict, allow_pickle=True)

    test_input_data, output_test_data, test_PCA_inv_model, test_PCA_Stokes_inv = load_test_data(data_path,
                                                                                                test_data_file,
                                                                                                input_props,
                                                                                                output_props)

    # normalize the data

    # load eve data

    # STEREO-B
    input_data = (torch.stack([torch.tensor(test_input_data)]))[0, :, :]
    print(f"shape of input_data: {input_data.shape}")
    test_inv_result = [irr for irr in tqdm(ipredict(model, input_data),
                                           total=input_data.shape[1])]
    test_NN_inversions = torch.stack(test_inv_result).numpy()

    #print(f"shape of test_NN_inversions: {output_test_data.shape}")
    #print(f"shape of  output_test_data: {test_PCA_inv_model.shape}")
    plt.scatter(test_NN_inversions[:, 0], output_test_data[2, 0, :])
    plt.scatter(test_PCA_inv_model[2, 0, :], output_test_data[2, 0, :])
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.savefig(result_path + "test_1.png")
    plt.show()
    plt.clf()

    plt.hist(test_NN_inversions[:, 0], bins=100, density=True,
             range=(0, 100))
    plt.xlim(0, 100)
    plt.savefig(result_path + "histogram_NN_inversion.png")

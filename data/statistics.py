# Load the data, compute stats, save it to dictionary in .npy
import pickle
import numpy as np

pwd = "/glade/work/mmolnar/PCA_inversions/"

# database file
with open(pwd + 'MgIIhk_PCAdatabase.pickle', "rb") as f:
    tdict = pickle.load(f)
PCA_dbase_coeff_all = tdict['PCA_dbase_coeff_all']
PCA_dbase_model_all = tdict['PCA_dbase_model_all']


# eigenbasis file
with open(pwd + 'MgIIhk_PCA_eigenbasis.pickle', "rb") as f:
    tdict = pickle.load(f)
PCA_eigenbasis_IQUV    = tdict['PCA_eigenbasis_IQUV']
PCA_IQUV_mean0         = tdict['PCA_mean_IQUV']
PCA_ebasis_lambda_arr  = tdict['lambda_arr']

dbase_input_properties = {}
dbase_output_properties = {}

# Flatten the input data

PCA_dbase_coeff_flat = PCA_dbase_coeff_all.reshape(PCA_dbase_coeff_all.shape[0] * PCA_dbase_coeff_all.shape[1],
                                                   PCA_dbase_coeff_all.shape[2])

PCA_dbase_model_flat = PCA_dbase_model_all.reshape(PCA_dbase_model_all.shape[0] * PCA_dbase_model_all.shape[1],
                                                   PCA_dbase_model_all.shape[2])

# Compute the

for el in range(PCA_dbase_coeff_flat.shape[0]):
    # Compute min, max, mean, std, median
    dbase_input_properties["var_" + format(el, '02d') + "_min"] = np.amin(PCA_dbase_coeff_flat[el, :])
    dbase_input_properties["var_" + format(el, '02d') + "_max"] = np.amax(PCA_dbase_coeff_flat[el, :])
    dbase_input_properties["var_" + format(el, '02d') + "_std"] = np.std(PCA_dbase_coeff_flat[el, :])
    dbase_input_properties["var_" + format(el, '02d') + "_median"] = np.median(PCA_dbase_coeff_flat[el, :])
    dbase_input_properties["var_" + format(el, '02d') + "_mean"] = np.mean(PCA_dbase_coeff_flat[el, :])


output_indices = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
PCA_dbase_model_flat = PCA_dbase_model_flat[output_indices, :]


for el in range(PCA_dbase_model_flat.shape[0]):
    # min, max, mean, std, median
    dbase_output_properties["var_" + format(el, '02d') + "_min"] = np.amin(PCA_dbase_model_flat[el, :])
    dbase_output_properties["var_" + format(el, '02d') + "_max"] = np.amax(PCA_dbase_model_flat[el, :])
    dbase_output_properties["var_" + format(el, '02d') + "_std"] = np.std(PCA_dbase_model_flat[el, :])
    dbase_output_properties["var_" + format(el, '02d') + "_median"] = np.median(PCA_dbase_model_flat[el, :])
    dbase_output_properties["var_" + format(el, '02d') + "_mean"] = np.mean(PCA_dbase_model_flat[el, :])

# Save npy file
output_path = '/glade/u/home/btremblay/PROPCA/laughing-happiness/'
np.save(f"{output_path}dbase_input_properties.npy",
        dbase_input_properties)

np.save(f"{output_path}dbase_output_properties.npy",
        dbase_output_properties)


# To load the dictionaries use:
# dbase_input_properties = np.load(pwd + "dbase_input_properties.npy")
# dbase_output_properties = np.load(pwd + "dbase_output_properties.npy")




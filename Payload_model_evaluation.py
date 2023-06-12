# import libraries
import time
import random

import tensorflow as tf
from keras.metrics import accuracy
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU, Layer, \
    Masking
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from tqdm import tqdm
import sklearn
import scipy.stats as ss
import math
from scipy import spatial

from road_attacks import ID_speedometer, ID_max_speedometer_mas, ID_corr_sig, ID_corr_sig_mas, ID_reverse_light_on, \
    ID_reverse_light_on_mas, ID_reverse_light_off, ID_reverse_light_off_mas

dirname = os.getcwd()

from Payload_model import threshold_estimation, encoder_model, decoder_model

warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

pd.options.display.float_format = '{:.6f}'.format
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import sklearn
from sklearn.preprocessing import LabelEncoder

# %%
# data creation for threshold estimation using a benign dataset
# use a benign dataset

def data_creation_eval(df_benign):
    cols = df_benign.columns[3:11]
    for i in cols:
        df_benign[i] = df_benign[i].astype('int')

    # convert ID to columns
    for x in df_benign["id"].unique():
        # loop through select columns
        for p in list(df_benign.columns[3:11]):
            # create new column, only use matched id value data, and forward fill the rest
            df_benign[x + "_" + p] = df_benign[p].where(df_benign["id"] == x, np.nan).ffill()

    df_benign = df_benign.dropna()
    df_benign = df_benign.reset_index(drop=True)

    # copy_benign = df_benign.copy()
    # drop columns which was not in train dataset (col_list_0)
    # load col_list_0
    with open('column_list_dlc_0', 'rb') as fp:
        col_list_0 = pickle.load(fp)
    df_benign = df_benign.drop(col_list_0, axis=1)

    # keep only train dataset columns in same order
    with open('column_list', 'rb') as fp:
        df_col_list = pickle.load(fp)
    print('number of columns :', len(df_col_list))
    df_benign = df_benign[df_col_list]
    # df_benign = df_benign[combined_df.columns[:]]

    # select only selected ID (0D0 for this)
    # df_benign = df_benign[df_benign.id == '125']
    # df_benign = df_benign[df_benign.id.isin(['125', '0D0', '033', '6E0', '00E', '0A7', '354', '5E1'])]

    # dataset for autoencoder
    autoencoder_benign = df_benign.drop(df_benign.columns[:11], axis=1)

    # feature scaling
    fit_scaling = pickle.load(open('minmax_scale.sav', 'rb'))
    fit_apply_ben = fit_scaling.transform(autoencoder_benign)
    autoencoder_df = pd.DataFrame(fit_apply_ben, columns=autoencoder_benign.columns, index=autoencoder_benign.index)

    # replace values with Zero for unassociated varaiable
    autoencoder_df['id'] = df_benign['id']
    all_vars = autoencoder_df.columns[:-1]
    with open('corr_list_id', 'rb') as fp:
        corr_list_id = pickle.load(fp)
    print('value replacement')
    for k, v in corr_list_id.items():
        remove_var = list(set(all_vars) - set(v))
        autoencoder_df.loc[autoencoder_df['id'] == k, remove_var] = 0

    autoencoder_df_test_id = autoencoder_df.copy()
    autoencoder_df = autoencoder_df.drop('id', axis=1)

    # updated autoencoder_df with only selected features for the ID
    # load saved column list
    # with open('corr_list_id', 'rb') as fp:
    #     corr_list_id = pickle.load(fp)
    # select columns
    # autoencoder_df = autoencoder_df[corr_list_id]

    X_test = autoencoder_df.to_numpy()

    return X_test, autoencoder_df, autoencoder_df_test_id, df_benign


# threshold_df = df_basic_short.copy()
threshold_df = pd.concat(threshold_estimation, ignore_index=True)
X_benign, autoencoder_df_test, combined_df_test, df_benign = data_creation_eval(threshold_df)

#%%
# load encoder and decoder
# encoder predictions to get Z
X_benign_Z = encoder_model.predict(X_benign)

test_id_col = combined_df_test[['id']] # identify the ID
test_id_col = test_id_col.reset_index(drop=True)

#%%
# benign distance limits calculations
latent_filepath = os.path.join(dirname, 'saved_files/Latent_autoencoder_model')
AE_latent = load_model(latent_filepath)
new_X_ben_Z_arr = AE_latent.predict(X_benign_Z)

#%%
# to match the predictions to IDs
id_list_ben = list(test_id_col['id'])
# find indexes of each ID
id_index_ben = {req_word: [idx for idx, word in enumerate(id_list_ben) if word == req_word] for req_word in
                set(id_list_ben)}

#%%
new_X_benign_Z = []
# dist_ori = []
latent_ID_TH = {}
for i in range(len(list(test_id_col['id']))):
    new_z_ben = new_X_ben_Z_arr[i]
    X_benign_Z_i = X_benign_Z[i]  # get same raw from Z
    dist = np.mean(np.abs(new_z_ben-X_benign_Z_i))
    dic_value = test_id_col['id'][i]
    # print(dic_value)
    if dic_value not in latent_ID_TH:
        print(latent_ID_TH)
        latent_ID_TH[dic_value] = [dist]
        print(latent_ID_TH)
    else:
        latent_ID_TH[dic_value].append(dist)

#%%
x_pred_benign = decoder_model.predict(X_benign_Z) # original model

MAE_pred_df = pd.DataFrame(x_pred_benign, columns=autoencoder_df_test.columns)
MAE_pred_df = MAE_pred_df.add_suffix('_pred')
MAE_pred_df_list = MAE_pred_df.to_numpy()

MAE_true_df = pd.DataFrame(X_benign, columns=autoencoder_df_test.columns)
MAE_true_df = MAE_true_df.add_suffix('_true')
MAE_true_df_list = MAE_true_df.to_numpy()

combined_df_test = combined_df_test.reset_index(drop=True)
benign_join = pd.concat([MAE_true_df, MAE_pred_df], axis=1)
benign_join['id'] = combined_df_test['id']

# %%
# threshold estimation for benign dataset
test = benign_join
# ID_thresholds = {}
ID_thresholds_max = {}
ID_thresholds_99 = {}
signal_thresholds_max = {}
signal_thresholds_99 = {}

for id in test['id'].unique():
    # filter ID
    id_df = test[test.id == id]
    signals = id_df[id_df.columns[id_df.columns.str.contains(id)]]  # select columns relevant to current ID
    # for ID threshold
    # signal_true = signals[signals.columns[signals.columns.str.contains('true')]].values  # get actual columns
    # signal_pred = signals[signals.columns[signals.columns.str.contains('pred')]].values  # get predicted columns

    # for All variable threshold
    signal_true = id_df[id_df.columns[id_df.columns.str.contains('true')]].values  # get actual columns
    signal_pred = id_df[id_df.columns[id_df.columns.str.contains('pred')]].values  # get predicted columns


    # calculate MAE for IDs
    # axis=0 for signal wise and 1 for ID wise
    ID_MAE = np.mean(np.abs(signal_true - signal_pred), axis=1)
    ID_MAE_max = ID_MAE.max()
    ID_MAE_99 = np.quantile(ID_MAE, 0.99)
    ID_thresholds_max[id] = ID_MAE_max
    ID_thresholds_99[id] = ID_MAE_99

    # calculate errors for each signals
    max_error = np.max(np.abs(signal_true - signal_pred), axis=0)
    quantile_99 = np.quantile(np.abs(signal_true - signal_pred), 0.99, axis=0)
    signal_thresholds_max[id] = max_error
    signal_thresholds_99[id] = quantile_99

# %%
# Attack Evaluation
def data_creation_attack(df_attack):
    cols = df_attack.columns[5:13]
    for i in cols:
        df_attack[i] = df_attack[i].astype('int')
    print('converted to int')

    # convert ID to columns
    print('feature creation...')
    for x in df_attack["id"].unique():
        # loop through select columns
        for p in list(df_attack.columns[5:13]):
            # create new column, only use matched id value data, and forward fill the rest
            df_attack[x + "_" + p] = df_attack[p].where(df_attack["id"] == x, np.nan).ffill()
    print('feature created')

    df_attack = df_attack.dropna()
    df_attack = df_attack.reset_index(drop=True)
    print('reset index')
    # copy_benign = df_benign.copy()
    # drop columns which was not in train dataset (col_list_0)
    # load col_list_0
    with open('column_list_dlc_0', 'rb') as fp:
        col_list_0 = pickle.load(fp)
    df_attack = df_attack.drop(col_list_0, axis=1)
    # Ori_lable = df_attack[['label']]

    # select only selected ID (0D0 for this)
    # df_attack = df_attack[df_attack.id.isin(['125', '0D0', '033', '6E0', '00E', '0A7', '354', '5E1'])]
    # df_attack = df_attack[df_attack.id == '125']
    Ori_lable = df_attack[['label']]

    # keep only train dataset columns in same order
    with open('column_list', 'rb') as fp:
        df_col_list = pickle.load(fp)
    print('number of columns :', len(df_col_list))
    df_attack = df_attack[df_col_list]
    print('applied keep only df_col_list ')

    # dataset for autoencoder
    autoencoder_attack = df_attack.drop(df_attack.columns[:11], axis=1)

    # feature scaling
    fit_scaling = pickle.load(open('minmax_scale.sav', 'rb'))
    fit_apply_ben = fit_scaling.transform(autoencoder_attack)
    autoencoder_df = pd.DataFrame(fit_apply_ben, columns=autoencoder_attack.columns, index=autoencoder_attack.index)

    # replace values with Zero for unassociated varaiable
    autoencoder_df['id'] = df_attack['id']
    all_vars = autoencoder_df.columns[:-1]
    with open('corr_list_id', 'rb') as fp:
        corr_list_id = pickle.load(fp)
    print('value replacement')
    for k, v in corr_list_id.items():
        remove_var = list(set(all_vars) - set(v))
        autoencoder_df.loc[autoencoder_df['id'] == k, remove_var] = 0
    autoencoder_df = autoencoder_df.drop('id', axis=1)

    X_test = autoencoder_df.to_numpy()

    return X_test, autoencoder_df, df_attack, Ori_lable


# %%
attacks = [ID_speedometer, ID_max_speedometer_mas, ID_corr_sig, ID_corr_sig_mas, ID_reverse_light_on,
           ID_reverse_light_on_mas, ID_reverse_light_off, ID_reverse_light_off_mas]

for i in attacks:
    print(i.name)
    Attack, autoencoder_df_attack, combined_df_attack, Ori_lable = data_creation_attack(i)

    # encoder predictions to get Z
    X_attack_Z = encoder_model.predict(Attack)

    att_id_col = combined_df_attack[['id']] # identify the ID
    att_id_col = att_id_col.reset_index(drop=True)

    new_X_attack_Z_arr = AE_latent.predict(X_attack_Z)
    ID_MAE_AE_ori = np.mean(np.abs(new_X_attack_Z_arr - X_attack_Z), axis=1)

    x_pred_attack = decoder_model.predict(X_attack_Z) # original model

    combined_df_attack = combined_df_attack.reset_index(drop=True)

    MAE_pred_df = pd.DataFrame(x_pred_attack, columns=autoencoder_df_attack.columns)
    MAE_pred_df = MAE_pred_df.add_suffix('_pred')
    MAE_pred_df_list = MAE_pred_df.to_numpy()

    MAE_true_df = pd.DataFrame(Attack, columns=autoencoder_df_attack.columns)
    MAE_true_df = MAE_true_df.add_suffix('_true')
    MAE_true_df_list = MAE_true_df.to_numpy()

    combined_df_attack.reset_index(drop=True, inplace=True)
    Ori_lable.reset_index(drop=True, inplace=True)
    attack_join = pd.concat([MAE_true_df, MAE_pred_df], axis=1)
    attack_join['Label'] = Ori_lable
    attack_join['ID'] = combined_df_attack['id']
    attack_join['Time'] = combined_df_attack['time_abs']
    attack_join['Latent_MAE_ori'] = ID_MAE_AE_ori

    # for All variable threshold
    ID_MAE = np.mean(np.abs(MAE_true_df_list - MAE_pred_df_list), axis=1)
    attack_join['MAE'] = ID_MAE

    # attack_join['MAE'] = mae_list
    attack_join['threshold'] = attack_join['ID'].map(ID_thresholds_99)
    attack_join['pred_class'] = np.where(attack_join['MAE'] >= attack_join['threshold'], 1, 0)

    print(classification_report(attack_join['Label'], attack_join['pred_class']))
    cm = confusion_matrix(attack_join['Label'], attack_join['pred_class'])
    print(cm)

    print('Point based evaluation........')
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    print('TP rate', TP * 100 / (TP + FN))
    print('TN rate', TN * 100 / (TN + FP))
    print('FP rate', FP * 100 / (FP + TN))
    print('FN rate', FN * 100 / (FN + TP))

    loss_df = attack_join.copy()
    loss_df['Time'] = loss_df['Time'] - loss_df['Time'].min()


    def winRatio(tempData):
        true_label_1 = len(tempData[tempData['Label'] == 1])
        true_label_all = len(tempData)

        pred_label_1 = len(tempData[tempData['pred_class'] == 1])
        pred_label_all = len(tempData)

        label_ratio = 0.01  # change for low frequent attacks
        if true_label_1 / true_label_all > label_ratio:
            true_label = 1
        else:
            true_label = 0

        label_ratio = 0.01
        if pred_label_1 / pred_label_all > label_ratio:
            pred_label = 1
        else:
            pred_label = 0

        return true_label, pred_label


    eval_df = loss_df.copy()
    windowSize = 0.025
    numWindows = (eval_df.Time.max() - eval_df.Time.min()) / windowSize
    # benWinLimit = round((testSet.time[len(BenTestSet)]-testSet.time.min())/wndowSize)

    startValue = eval_df.Time.min()
    stopValue = startValue + windowSize

    k_list = []
    ratio_list = []
    for k in range(1, int(numWindows - 1)):
        smallerWindow = eval_df[(eval_df.Time >= startValue) & (eval_df.Time < stopValue)]

        ratio = winRatio(smallerWindow)
        k_list.append(k)
        ratio_list.append(ratio)

        startValue = stopValue
        stopValue = startValue + windowSize

    # ratio return 2 lists, true window at 0 and pred_window at 1
    true_window = []
    for i in range(len(ratio_list)):
        l = ratio_list[i][0]
        true_window.append(l)

    pred_window = []
    for i in range(len(ratio_list)):
        l = ratio_list[i][1]
        pred_window.append(l)

    print('Window based evaluation....')
    print(classification_report(true_window, pred_window))
    cm = confusion_matrix(true_window, pred_window)
    print(cm)

    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    print('TP rate', TP * 100 / (TP + FN))
    print('TN rate', TN * 100 / (TN + FP))
    print('FP rate', FP * 100 / (FP + TN))
    print('FN rate', FN * 100 / (FN + TP))
    print('\n')


# attacks for road dataset                         *
# pre-processed for id and payloads  (int)         *
#***************************************************

# import libraries
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU, Layer
from tensorflow.keras.models import Sequential
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

warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

pd.options.display.float_format = '{:.6f}'.format
from sklearn.preprocessing import MinMaxScaler
dirname = os.getcwd()


# function to preprocess data
def data_preprocessing(df):
    df['time'], df['can'], df['ID'] = df['CAN_frame'].str.split('\s+', 2).str
    df['id'], df['payload'] = df['ID'].str.split('#').str
    df = df[['id', 'payload', 'time']]
    df['time'] = df['time'].str.replace(r"\(", "")
    df['time'] = df['time'].str.replace(r"\)", "")
    df['label'] = 0

    # change datatypes
    cols = df.columns
    for i in cols:
        df[i] = df[i].astype('category')

    df['time'] = pd.to_numeric(df['time'])
    df['time_abs'] = df.time - min(df.time)

    # create features for time difference and time diff for each id
    df['time_dif'] = df['time_abs'].diff()
    # df['time_dif'] = df['time_dif'].fillna(df['time_dif'].mean())
    # df[['ID_time_diff']] = df.groupby('id')['time_abs'].diff()
    # df['ID_time_diff'] = df['ID_time_diff'].fillna(df.groupby('id')['ID_time_diff'].transform('mean'))

    df['d1'] = df['payload'].str[:2].astype('category')
    df['d2'] = df['payload'].str[2:4].astype('category')
    df['d3'] = df['payload'].str[4:6].astype('category')
    df['d4'] = df['payload'].str[6:8].astype('category')
    df['d5'] = df['payload'].str[8:10].astype('category')
    df['d6'] = df['payload'].str[10:12].astype('category')
    df['d7'] = df['payload'].str[12:14].astype('category')
    df['d8'] = df['payload'].str[14:16].astype('category')

    # function to convert hex to int
    def hex_to_int(hex):
        val = int(hex, 16)
        return val

    def hex_to_bin(hex):
        val = bin(int(hex, 16))[2:]
        return val

    # convert hex signal to int signal
    df['id_int'] = df['id'].apply(hex_to_int).astype('int')
    df['d1_int'] = df['d1'].apply(hex_to_int).astype('int')
    df['d2_int'] = df['d2'].apply(hex_to_int).astype('int')
    df['d3_int'] = df['d3'].apply(hex_to_int).astype('int')
    df['d4_int'] = df['d4'].apply(hex_to_int).astype('int')
    df['d5_int'] = df['d5'].apply(hex_to_int).astype('int')
    df['d6_int'] = df['d6'].apply(hex_to_int).astype('int')
    df['d7_int'] = df['d7'].apply(hex_to_int).astype('int')
    df['d8_int'] = df['d8'].apply(hex_to_int).astype('int')

    sorted_df = df.sort_values(by=['time'])

    return sorted_df

cols = ['id', 'id_int', 'time_abs','time', 'label', 'd1_int', 'd2_int', 'd3_int', 'd4_int', 'd5_int', 'd6_int', 'd7_int', 'd8_int']
#%%
# testing dataset pre-processing
print('fuzzing attack...')
fuzzing = os.path.join(dirname, 'Data/Attacks/fuzzing_attack_1.log')
fuzzing = pd.read_csv(fuzzing, engine='python',header=None)
fuzzing.columns =['CAN_frame']
fuzzing = data_preprocessing(fuzzing)

ID_fuzzing = fuzzing
A_start = 1000000004.622975
A_end = 1000000007.958234
ID_fuzzing['label'] = np.where((ID_fuzzing.time>=A_start)
                            &(ID_fuzzing.time<=A_end)
                            &(ID_fuzzing.payload =='FFFFFFFFFFFFFFFF'),1,0)

ID_fuzzing.reset_index(drop=True, inplace=True)
ID_fuzzing = ID_fuzzing[cols]

ID_fuzzing.name = 'Fuzzing attack'

# **********************************************************************************************************************
# testing dataset pre-processing
print('max_speedometer...')
max_speedometer = os.path.join(dirname, 'Data/Attacks/max_speedometer_attack_1.log')
max_speedometer = pd.read_csv(max_speedometer, engine='python',header=None)
max_speedometer.columns =['CAN_frame']
max_speedometer = data_preprocessing(max_speedometer)

ID_speedometer = max_speedometer
A_start = 1110000042.009204
A_end = 1110000066.449010
# A_start = 42.009204
# A_end = 66.449010
ID_speedometer['label'] = np.where((ID_speedometer.time>=A_start)
                            &(ID_speedometer.time<=A_end)
                            &(ID_speedometer.id=='0D0')
                            &(ID_speedometer.d6=='FF'),1,0)
ID_speedometer.reset_index(drop=True, inplace=True)
ID_speedometer = ID_speedometer[cols]

# cols = ['id','payload','time','time_abs','ID_time_diff','label']
# ID_speedometer = ID_speedometer[cols]
ID_speedometer.name = 'Speedometer attack'

# **********************************************************************************************************************
print('max_speedometer_mas...')
max_speedometer_mas = os.path.join(dirname, 'Data/Attacks/max_speedometer_attack_1_masquerade.log')
max_speedometer_mas = pd.read_csv(max_speedometer_mas, engine='python',header=None)
max_speedometer_mas.columns =['CAN_frame']
max_speedometer_mas = data_preprocessing(max_speedometer_mas)

ID_max_speedometer_mas = max_speedometer_mas
A_start = 1140000042.009204
A_end = 1140000066.449010
ID_max_speedometer_mas['label'] = np.where((ID_max_speedometer_mas.time>=A_start)
                                   &(ID_max_speedometer_mas.time<=A_end)
                                   &(ID_max_speedometer_mas.id=='0D0')
                                   &(ID_max_speedometer_mas.d6=='FF'),1,0)
ID_max_speedometer_mas.reset_index(drop=True, inplace=True)
ID_max_speedometer_mas = ID_max_speedometer_mas[cols]
ID_max_speedometer_mas.name = 'Max speedometer mas attack'

# **********************************************************************************************************************
print('corr_sig...')
corr_sig = os.path.join(dirname, 'Data/Attacks/correlated_signal_attack_1.log')
corr_sig = pd.read_csv(corr_sig, engine='python',header=None)
corr_sig.columns =['CAN_frame']
corr_sig = data_preprocessing(corr_sig)

ID_corr_sig = corr_sig
A_start = 1030000009.191851
A_end = 1030000030.050109
# A_start = 9.191851
# A_end = 30.050109
ID_corr_sig['label'] = np.where((ID_corr_sig.time>=A_start)
                       &(ID_corr_sig.time<=A_end)
                       &(ID_corr_sig.id=='6E0')
                       &(ID_corr_sig.payload=='595945450000FFFF'),1,0)
ID_corr_sig.reset_index(drop=True, inplace=True)
ID_corr_sig = ID_corr_sig[cols]
ID_corr_sig.name = 'corr_sig attack'

# **********************************************************************************************************************
print('corr_sig_mas...')
corr_sig_mas = os.path.join(dirname, 'Data/Attacks/correlated_signal_attack_1_masquerade.log')
corr_sig_mas = pd.read_csv(corr_sig_mas, engine='python',header=None)
corr_sig_mas.columns =['CAN_frame']
corr_sig_mas = data_preprocessing(corr_sig_mas)


ID_corr_sig_mas = corr_sig_mas
A_start = 1060000009.191851
A_end = 1060000030.050109
ID_corr_sig_mas['label'] = np.where((ID_corr_sig_mas.time>=A_start)
                                &(ID_corr_sig_mas.time<=A_end)
                                &(ID_corr_sig_mas.id=='6E0')
                                &(ID_corr_sig_mas.payload=='595945450000FFFF'),1,0)
ID_corr_sig_mas.reset_index(drop=True, inplace=True)
ID_corr_sig_mas = ID_corr_sig_mas[cols]
ID_corr_sig_mas.name = 'corr_sig_mas attack'

# **********************************************************************************************************************
print('reverse_light_on...')
reverse_light_on = os.path.join(dirname, 'Data/Attacks/reverse_light_on_attack_1.log')
reverse_light_on = pd.read_csv(reverse_light_on, engine='python',header=None)
reverse_light_on.columns =['CAN_frame']
reverse_light_on = data_preprocessing(reverse_light_on)

ID_reverse_light_on = reverse_light_on
A_start = 1230000018.929177
A_end = 1230000038.836015
# A_start = 18.929177
# A_end = 38.836015
ID_reverse_light_on['label'] = np.where((ID_reverse_light_on.time>=A_start)
                       &(ID_reverse_light_on.time<=A_end)
                       &(ID_reverse_light_on.id=='0D0')
                       &(ID_reverse_light_on.d3=='0C'),1,0)
ID_reverse_light_on.reset_index(drop=True, inplace=True)
ID_reverse_light_on = ID_reverse_light_on[cols]
ID_reverse_light_on.name = 'reverse_light_on attack'

# **********************************************************************************************************************
print('reverse_light_on_mas...')
reverse_light_on_mas = os.path.join(dirname, 'Data/Attacks/reverse_light_on_attack_1_masquerade.log')
reverse_light_on_mas = pd.read_csv(reverse_light_on_mas, engine='python',header=None)
reverse_light_on_mas.columns =['CAN_frame']
reverse_light_on_mas = data_preprocessing(reverse_light_on_mas)

ID_reverse_light_on_mas = reverse_light_on_mas
A_start = 1260000018.929177
A_end = 1260000038.836015
ID_reverse_light_on_mas['label'] = np.where((ID_reverse_light_on_mas.time>=A_start)
                                        &(ID_reverse_light_on_mas.time<=A_end)
                                        &(ID_reverse_light_on_mas.id=='0D0')
                                        &(ID_reverse_light_on_mas.d3=='0C'),1,0)
ID_reverse_light_on_mas.reset_index(drop=True, inplace=True)
ID_reverse_light_on_mas = ID_reverse_light_on_mas[cols]
ID_reverse_light_on_mas.name = 'reverse_light_on_mas attack'

# **********************************************************************************************************************
print('reverse_light_off...')
reverse_light_off = os.path.join(dirname, 'Data/Attacks/reverse_light_off_attack_1.log')
reverse_light_off = pd.read_csv(reverse_light_off, engine='python',header=None)
reverse_light_off.columns =['CAN_frame']
reverse_light_off = data_preprocessing(reverse_light_off)

ID_reverse_light_off = reverse_light_off
A_start = 1170000016.627923
A_end = 1170000023.347311
# A_start = 16.627923
# A_end = 23.347311
ID_reverse_light_off['label'] = np.where((ID_reverse_light_off.time>=A_start)
                                        &(ID_reverse_light_off.time<=A_end)
                                        &(ID_reverse_light_off.id=='0D0')
                                        &(ID_reverse_light_off.d3=='04'),1,0)
ID_reverse_light_off.reset_index(drop=True, inplace=True)
ID_reverse_light_off = ID_reverse_light_off[cols]
ID_reverse_light_off.name = 'reverse_light_off attack'

# **********************************************************************************************************************
print('reverse_light_off_mas...')
reverse_light_off_mas = os.path.join(dirname, 'Data/Attacks/reverse_light_off_attack_1_masquerade.log')
reverse_light_off_mas = pd.read_csv(reverse_light_off_mas, engine='python',header=None)
reverse_light_off_mas.columns =['CAN_frame']
reverse_light_off_mas = data_preprocessing(reverse_light_off_mas)

ID_reverse_light_off_mas = reverse_light_off_mas
A_start = 1200000016.627923
A_end = 1200000023.347311
ID_reverse_light_off_mas['label'] = np.where((ID_reverse_light_off_mas.time>=A_start)
                                         &(ID_reverse_light_off_mas.time<=A_end)
                                         &(ID_reverse_light_off_mas.id=='0D0')
                                         &(ID_reverse_light_off_mas.d3=='04'),1,0)
ID_reverse_light_off_mas.reset_index(drop=True, inplace=True)
ID_reverse_light_off_mas = ID_reverse_light_off_mas[cols]
ID_reverse_light_off_mas.name = 'reverse_light_off_mas'

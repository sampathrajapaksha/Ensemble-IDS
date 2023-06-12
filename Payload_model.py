# ********************************************************#
# feature based Autoencoder model for CAN payload

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
# from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import sklearn
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.6f}'.format
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

    # remove IDs which have all 0 payload values
    ids_to_remove = ['FFF', '671']
    df = df[~df['id'].isin(ids_to_remove)]

    df['time'] = pd.to_numeric(df['time'])
    # calculate absolute time as seconds
    df['time_abs'] = df.time - min(df.time)

    # identify payload fields
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

    # convert hex signal to int signal
    df['id_int'] = df['id'].apply(hex_to_int)
    df['d1_int'] = df['d1'].apply(hex_to_int)
    df['d2_int'] = df['d2'].apply(hex_to_int)
    df['d3_int'] = df['d3'].apply(hex_to_int)
    df['d4_int'] = df['d4'].apply(hex_to_int)
    df['d5_int'] = df['d5'].apply(hex_to_int)
    df['d6_int'] = df['d6'].apply(hex_to_int)
    df['d7_int'] = df['d7'].apply(hex_to_int)
    df['d8_int'] = df['d8'].apply(hex_to_int)

    # sort the dataframe by time
    sorted_df = df.sort_values(by=['time'])
    sorted_df.reset_index(drop=True, inplace=True)

    # filter columns
    sorted_df = sorted_df[
        ['id', 'id_int', 'time_abs', 'd1_int', 'd2_int', 'd3_int', 'd4_int', 'd5_int', 'd6_int', 'd7_int', 'd8_int']]

    return sorted_df

#%%
# import benign datasets

start = time.time()
dirname = os.getcwd()

basic_long = os.path.join(dirname, 'Data/Benign/ambient_dyno_drive_basic_long.log')
basic_short = os.path.join(dirname, 'Data/Benign/ambient_dyno_drive_basic_short.log')
benign_anomaly = os.path.join(dirname, 'Data/Benign/ambient_dyno_drive_benign_anomaly.log')
extended_long = os.path.join(dirname, 'Data/Benign/ambient_dyno_drive_extended_long.log')
extended_short = os.path.join(dirname, 'Data/Benign/ambient_dyno_drive_extended_short.log')
radio_infotainment = os.path.join(dirname, 'Data/Benign/ambient_dyno_drive_radio_infotainment.log')
drive_winter = os.path.join(dirname, 'Data/Benign/ambient_dyno_drive_winter.log')
exercise_all_bits = os.path.join(dirname, 'Data/Benign/ambient_dyno_exercise_all_bits.log')
idle_radio_infotainment = os.path.join(dirname, 'Data/Benign/ambient_dyno_idle_radio_infotainment.log')
reverse = os.path.join(dirname, 'Data/Benign/ambient_dyno_reverse.log')
highway_street_driving = os.path.join(dirname, 'Data/Benign/ambient_highway_street_driving_diagnostics.log')
highway_street_driving_long = os.path.join(dirname, 'Data/Benign/ambient_highway_street_driving_long.log')

# read csv as pandas
print('read .log files...')
df_basic_long = pd.read_csv(basic_long, engine='python', header=None)
df_basic_long.columns = ['CAN_frame']
df_basic_short = pd.read_csv(basic_short, engine='python', header=None)
df_basic_short.columns = ['CAN_frame']
df_extended_long = pd.read_csv(extended_long, engine='python', header=None)
df_extended_long.columns = ['CAN_frame']
df_extended_short = pd.read_csv(extended_short, engine='python', header=None)
df_extended_short.columns = ['CAN_frame']
df_radio_infotainment = pd.read_csv(radio_infotainment, engine='python', header=None)
df_radio_infotainment.columns = ['CAN_frame']
df_drive_winter = pd.read_csv(drive_winter, engine='python', header=None)
df_drive_winter.columns = ['CAN_frame']
df_exercise_all_bits = pd.read_csv(exercise_all_bits, engine='python', header=None)
df_exercise_all_bits.columns = ['CAN_frame']
df_idle_radio_infotainment = pd.read_csv(idle_radio_infotainment, engine='python', header=None)
df_idle_radio_infotainment.columns = ['CAN_frame']
df_reverse = pd.read_csv(reverse, engine='python', header=None)
df_reverse.columns = ['CAN_frame']
df_highway_street_driving = pd.read_csv(highway_street_driving, engine='python', header=None)
df_highway_street_driving.columns = ['CAN_frame']
df_highway_street_driving_long = pd.read_csv(highway_street_driving_long, engine='python', header=None)
df_highway_street_driving_long.columns = ['CAN_frame']
df_benign_anomaly = pd.read_csv(benign_anomaly, engine='python', header=None)
df_benign_anomaly.columns = ['CAN_frame']
end = time.time()
print('file reading completed')
print('time : ', end - start)

# %%
# data preprocessing
start = time.time()
df_basic_long = data_preprocessing(df_basic_long)
df_basic_short = data_preprocessing(df_basic_short)
df_extended_long = data_preprocessing(df_extended_long)
df_extended_short = data_preprocessing(df_extended_short)
df_radio_infotainment = data_preprocessing(df_radio_infotainment)
df_drive_winter = data_preprocessing(df_drive_winter)
df_exercise_all_bits = data_preprocessing(df_exercise_all_bits)
df_idle_radio_infotainment = data_preprocessing(df_idle_radio_infotainment)
df_reverse = data_preprocessing(df_reverse)
df_highway_street_driving = data_preprocessing(df_highway_street_driving)
df_highway_street_driving_long = data_preprocessing(df_highway_street_driving_long)
df_benign_anomaly = data_preprocessing(df_benign_anomaly)
end = time.time()
print('time : ', end - start)

# %%
# create a function to fit scaler for all variables
""" need to use a large dataset which represents the benign driving population.
scaler_fit function transform scale each variables using min max normalization """

def scaler_fit(combined_df):
    print(combined_df.columns)
    print('scaling started...')
    # convert categorical columns to int
    cols = combined_df.columns[3:10]
    for i in cols:
        combined_df[i] = combined_df[i].astype('int')

    # dataframe transformation - IDs as variables
    print('column transformation...')
    for x in combined_df["id"].unique():
        # loop through select columns
        for p in list(combined_df.columns[3:11]):
            # create new column, only use matched id value data, and forward fill the rest
            combined_df[x + "_" + p] = combined_df[p].where(combined_df["id"] == x, np.nan).ffill()

    # remove empty rows
    combined_df = combined_df.dropna()
    combined_df = combined_df.reset_index(drop=True)
    print('Combined_df size for scaling:', combined_df.shape)

    # identify columns with constant values (DLC < 8 IDS)
    col_list_0 = []
    for cols in combined_df.columns:
        uni_val = len(combined_df[cols].unique())
        if uni_val == 1 and combined_df[cols].values[0] == 0:
            col_list_0.append(cols)
    print('DLC<8 varaiables')
    print(len(col_list_0))

    # save col_list_0
    constant_variables = os.path.join(dirname, 'saved_files/column_list_dlc_0')
    print('saving DLC <8 feature list')
    with open(constant_variables, 'wb') as fp:
        pickle.dump(col_list_0, fp)
    # load col_list_0
    with open(constant_variables, 'rb') as fp:
        col_list_0 = pickle.load(fp)

    # remove identified constant variables from the dataframe
    combined_df = combined_df.drop(col_list_0, axis=1)
    # save remaining column list
    print('saving combined df column list')
    column_list = os.path.join(dirname, 'saved_files/column_list')
    with open(column_list, 'wb') as fp:
        pickle.dump(combined_df.columns, fp)

    # load saved column list
    with open(column_list, 'rb') as fp:
        df_col_list = pickle.load(fp)
    print('number of columns :', len(df_col_list))

    # Dataframe to train AutoEncoder model (This selects only relevant variables)
    autoencoder_df = combined_df.drop(combined_df.columns[:11], axis=1)

    # feature scaling - Apply min max scaling to each variable
    scaler = MinMaxScaler()
    fit_scaling = scaler.fit(autoencoder_df)
    print('scaling saving...')
    minmax_scale = os.path.join(dirname, 'saved_files/minmax_scale.sav')
    pickle.dump(fit_scaling, open(minmax_scale, 'wb'))  # save scaler
    fit_scaling = pickle.load(open(minmax_scale, 'rb'))  # load scaler
    print('scaling saved')

    return fit_scaling, autoencoder_df

# training dataset for model update
"""
data_creation function creates a dataset for autoencoder model training 
"""

def data_creation(combined_df):
    print('data creation started...')
    # convert categorical columns to int
    cols = combined_df.columns[3:10]
    for i in cols:
        combined_df[i] = combined_df[i].astype('int')

    # dataframe transformation - IDs as variables
    print('column transformation...')
    for x in combined_df["id"].unique():
        # loop through select columns
        for p in list(combined_df.columns[3:11]):
            # create new column, only use matched id value data, and forward fill the rest
            combined_df[x + "_" + p] = combined_df[p].where(combined_df["id"] == x, np.nan).ffill()

    combined_df = combined_df.dropna()
    combined_df = combined_df.reset_index(drop=True)
    print('Combined_df size for training:', combined_df.shape)

    # load col_list_0
    constant_variables = os.path.join(dirname, 'saved_files/column_list_dlc_0')
    with open(constant_variables, 'rb') as fp:
        col_list_0 = pickle.load(fp)

    # remove identified constant variables from the dataframe
    time_df = combined_df.copy()
    combined_df = combined_df.drop(col_list_0, axis=1)
    # load saved column list
    column_list = os.path.join(dirname, 'saved_files/column_list')
    with open(column_list, 'rb') as fp:
        df_col_list = pickle.load(fp)
    print('number of columns :', len(df_col_list))
    # order the combined_df into the same order of df_col_list
    combined_df = combined_df[df_col_list]
    # extract ids of combined_df
    combined_df_id = combined_df[['id']]

    # select only selected ID (0D0 for this)
    # keep for association calculations, remove for model training
    # combined_df = combined_df[combined_df.id.isin(['125','0D0','033','6E0','00E','0A7','354','5E1'])]

    # Dataframe to train AutoEncoder model (This selects only relevant variables)
    autoencoder_df = combined_df.drop(combined_df.columns[:11], axis=1)
    print('autoencoder_df size')
    print(autoencoder_df.shape)

    # apply feature scaling
    print('apply scaling')
    minmax_scale = os.path.join(dirname, 'saved_files/minmax_scale.sav')
    fit_scaling = pickle.load(open(minmax_scale, 'rb'))
    fit_apply = fit_scaling.transform(autoencoder_df)
    print('applied scalling')
    # create a new df using scaled variables
    autoencoder_df = pd.DataFrame(fit_apply, columns=autoencoder_df.columns, index=autoencoder_df.index)

    print('replace values for unassociated variables')
    # replace values with Zero for unassociated varaiable
    autoencoder_df['id'] = combined_df['id']
    all_vars = autoencoder_df.columns[:-1]

    # use this only after identifying associated variables
    # comment this code for create a df for associated variable identification
    # with open('corr_list_id', 'rb') as fp:
    #     corr_list_id = pickle.load(fp)
    # print('value replacement')
    # for k,v in corr_list_id.items():
    #      remove_var = list(set(all_vars)-set(v))
    #      autoencoder_df.loc[autoencoder_df['id'] == k, remove_var] = 0

    # shuffle before ID drop
    autoencoder_df = sklearn.utils.shuffle(autoencoder_df)
    autoencoder_df_id = autoencoder_df.copy()
    autoencoder_df = autoencoder_df.drop('id', axis=1)

    df_shuffle = autoencoder_df.copy()
    X_train = df_shuffle.to_numpy() # convert to numpy array
    print('X_train created')

    return X_train, autoencoder_df, autoencoder_df_id

#%%
# training and threshold dataset selection
dataframes = [df_extended_long, df_extended_short, df_radio_infotainment, df_drive_winter, df_exercise_all_bits,
              df_idle_radio_infotainment, df_reverse, df_highway_street_driving, df_highway_street_driving_long,
              df_benign_anomaly]

training = []
threshold_estimation = []

for df in dataframes:
    total_rows = len(df)
    train_rows = int(0.7 * total_rows)

    train_df = df.iloc[:train_rows]
    test_df = df.iloc[train_rows:]

    training.append(train_df)
    threshold_estimation.append(test_df)

# dataframe to fit scaler
scale_dataframe = pd.concat(training, ignore_index=True)
fit_scaling, autoencoder_df_scaling = scaler_fit(scale_dataframe)

#%%
# dataset creation for model training
train_dataset = pd.concat(training, ignore_index=True)
X_train, autoencoder_df, autoencoder_df_train_id = data_creation(train_dataset)

#%%
# model fitting
train_loss = []
val_loss = []
def model_fit(X_train):
    # This is the dimension of the original space
    input_dim = X_train.shape[1]

    # This is the dimension of the latent space (encoding space)
    #latent_dim = input_dim-608
    latent_dim = 20 # over 99 PCA variance
    nodes = [128]
    for nodes in nodes:
        encoder = Sequential([
            Dense(nodes, activation='relu', input_shape=(input_dim,), name='input_layer'),
            # kernel_regularizer=regularizers.l2(1e-8)
            Dropout(0.0),
            Dense(latent_dim, activation='relu', name='latent_layer')
        ])

        decoder = Sequential([
            Dense(nodes, activation='relu', input_shape=(latent_dim,), name='decoder_input_layer'),
            Dropout(0.0),
            Dense(input_dim, activation=None, name='output_layer')
        ])

        autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,
            decay_rate=0.9)
        print(lr_schedule)
        l_rate = 0.0001
        autoencoder.compile(loss='mse', optimizer=Adam(lr=l_rate))
        autoencoder.summary()

        es = EarlyStopping('val_loss', mode='min', verbose=1, patience=5)
        filepath = os.path.join(dirname, 'saved_files/autoencoder_model_checkpoint')
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')
        model_history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=128, verbose=1, validation_split=0.1,
                                        callbacks=[es, checkpoint]).history


        print('encoder saving...')
        encoder_model = Model(inputs=encoder.input, outputs=encoder.output)
        encoder_path = os.path.join(dirname, 'saved_files/encoder_model')
        encoder_model.save(encoder_path)

        print('decoder saving...')
        decoder_model = Model(inputs=encoder.output, outputs=decoder(encoder.output))
        decoder_path = os.path.join(dirname, 'saved_files/encoder_model')
        decoder_model.save(decoder_path)

        train_loss.append(model_history["loss"])
        val_loss.append(model_history["val_loss"])

        plt.plot(model_history["loss"][1:], 'b', label='Train')
        plt.plot(model_history["val_loss"][1:], 'r', label='Validation')
        plt.title("Loss vs. Epoch for: Nodes=%i" %nodes)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.show()

    return model_history, l_rate


model_history, lr_schedule = model_fit(X_train)

#%%
# load encoder and decoder
encoder_path = os.path.join(dirname, 'saved_files/encoder_model')
decoder_path = os.path.join(dirname, 'saved_files/encoder_model')
encoder_model = load_model(encoder_path)
decoder_model = load_model(decoder_path)

# latent (Z) vector for train dataset
print('encoder output')
Z = encoder_model.predict(X_train)

# output for latent layer (should be equal to autoencoder x_pred)
print('decoder output')
deco_outputs = decoder_model.predict(Z)

# to match the encoder predictions to IDs
id_list = list(autoencoder_df_train_id['id'])
# find indexes of each ID
id_index = {req_word: [idx for idx, word in enumerate(id_list) if word == req_word] for req_word in set(id_list)}

# Identify IDs for predicted train latent space
id_pred_dic = {}  # define a dic for {ID:[[][]....]...}
Z_list = Z.tolist() # convert the Z array to lists of lists
for k,v in id_index.items():
    T = [Z_list[i] for i in v]
    id_pred_dic[k] = T

Z_id_list = os.path.join(dirname, 'saved_files/id_pred_dic')
with open(Z_id_list, 'wb') as fp:
    pickle.dump(id_pred_dic, fp)

#%% Estimate atent size
from sklearn.decomposition import PCA
pca = PCA(n_components=0.90)
X_pca = pca.fit(Z) # this will fit and reduce dimensions
print(pca.n_components_)
#%% AE model for latent space
input_dim = Z.shape[1]
latent_dim = 18

# functional API model
inputD = Input(shape=(input_dim,))
# encoder
encoded_1 = Dense(latent_dim, activation='relu', name='first_layer')(inputD)
# encoded_2 = Dense(latent_dim, activation='tanh', name='latent_layer')(encoded_1)

# decoder
# decoded_1 = Dense(128, activation='tanh', name='decoder_first_layer')(encoded_2)
decoded = Dense(input_dim, activation=None, name='decoder_last_layer')(encoded_1)  # one layer - encoded_1

AE_latent = Model(inputD, decoded)

AE_latent.compile(loss='mse', optimizer=Adam(lr=0.0001))
AE_latent.summary()

es = EarlyStopping('val_loss', mode='min', verbose=1, patience=5)
latent_filepath = os.path.join(dirname, 'saved_files/Latent_autoencoder_model')
checkpoint = ModelCheckpoint(filepath=latent_filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
AE_model_history = AE_latent.fit(Z, Z, epochs=100, batch_size=256, verbose=1, validation_split=0.1,
                                 callbacks=[es, checkpoint]).history

plt.plot(AE_model_history["loss"][1:], 'b', label='Train')
plt.plot(AE_model_history["val_loss"][1:], 'r', label='Validation')
plt.title("Loss for latent AE model")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()




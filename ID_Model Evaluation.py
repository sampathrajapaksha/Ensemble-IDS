
# import libraries
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU, Layer, SimpleRNN
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
import time
import seaborn as sns
from sklearn.svm import OneClassSVM

from road_attacks import ID_speedometer, ID_max_speedometer_mas, ID_corr_sig, ID_corr_sig_mas, ID_reverse_light_on, \
    ID_reverse_light_on_mas, ID_reverse_light_off, ID_reverse_light_off_mas

warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping
dirname = os.getcwd()


# data preprocessing
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
    sorted_df = df.sort_values(by=['time'])

    return sorted_df


# %%
# import benign datasets
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
print('file reading completed')

# data preprocessing
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

BenTrainSet = pd.concat(training, ignore_index=True)
# convert payload values to int
payload_columns = ['d1_int', 'd2_int', 'd3_int', 'd4_int', 'd5_int', 'd6_int', 'd7_int', 'd8_int']
for i in payload_columns:
    BenTrainSet[i] = BenTrainSet[i].astype('int')
print('dataset imported')

train = BenTrainSet[:]
train = train[['id', 'time_abs', 'time_dif']]
train.reset_index(drop=True, inplace=True)

# %%
benign_df = train.copy()

# convert df data into a series
tempId = pd.Series(benign_df['id'])
tempId = tempId.str.cat(sep=' ')

tokenizer = Tokenizer(oov_token=True)
tokenizer.fit_on_texts([tempId])
#
# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('id_tokens.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([tempId])[0]

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

#%%
# tokenizer non-keras version
id_tokens = tokenizer.index_word
try:
    del id_tokens[1]
except:
    pass
id_tokens_upper = {k:v.upper() for k,v in id_tokens.items()}
id_tokens_dic = {value:key for key, value in id_tokens_upper.items()}
# testing
test_sequence = '69E'
start = time.time()
try:
    val = id_tokens_dic[test_sequence]
except:
    val = 1
end = time.time()
print(end-start)

print('saving tokenize dictionary')
with open('id_tokens_dic', 'wb') as fp:
    pickle.dump(id_tokens_dic, fp)

# load col_list_0
with open('id_tokens_dic', 'rb') as fp:
    id_tokens_dic = pickle.load(fp)

# %%
# create sequences
def seq(sequence_data):
    sequences = []
    # for 2 context ids beside center id
    # j to defined the context window, 2 for one context word from one side, 4 for 2 context words
    j = 10    # use 10 for dgx model
    for i in range(j, len(sequence_data)):
        words = sequence_data[i - j:i + 1]
        sequences.append(words)

    # print("The Length of sequences are: ", len(sequences))
    sequences = np.array(sequences)

    # create X and y
    X = []
    y = []

    # for j context ids beside center id
    k = int(j / 2)
    for i in range(len(sequences)):
        X.append(list((np.delete(sequences[i], k, 0).flatten())))

    for i in sequences:
        y.append(i[k])  # select center id as y

    X = np.array(X)
    y = np.array(y)

    return X, y

#%%
X, y = seq(sequence_data)
y = to_categorical(y, num_classes=vocab_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)

# %%
# model training
j = 10
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=j))
model.add((LSTM(32, return_sequences=False)))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=0.5, callbacks=[es]).history

#%%
model.save('saved_files/load_lstm_id_model_2.h5')
pickle.dump(history, open("saved_files/load_lstm_id_history_2.pkl", "wb"))

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
# prediction
def winRatio(tempData):
    true_label_1 = len(tempData[tempData['label'] == 1])
    true_label_all = len(tempData)

    pred_label_1 = len(tempData[tempData['pred_class'] == 1])
    pred_label_all = len(tempData)

    label_ratio = 0.01 # change for low frequent attacks
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


# prediction function
def results_evaluation(pred_RPM, eval_df, y, winRatio, id_threshold_dic, threshold_all):
    # best prediction
    pred_prob = []
    for i in range(len(pred_RPM)):
        id_prob = pred_RPM[i][y[i]]  # tokenizer.word_index to get the mapping of id and index
        pred_prob.append(id_prob)

    eval_df['true_id'] = y
    eval_df['id_pred_prob'] = pred_prob
    eval_df['threshold'] = eval_df['id'].map(id_threshold_dic)
    eval_df['threshold'] = eval_df['threshold'].astype('float')
    eval_df['threshold'] = eval_df['threshold'].fillna(1)

    # pred class
    eval_df['pred_class'] = np.where(eval_df['id_pred_prob'] <= eval_df['threshold'], 1, 0)

    windowSize = 0.025
    numWindows = (eval_df.time.max() - eval_df.time.min()) / windowSize
    # benWinLimit = round((testSet.time[len(BenTestSet)]-testSet.time.min())/wndowSize)

    startValue = eval_df.time.min()
    stopValue = startValue + windowSize

    k_list = []
    ratio_list = []
    for k in range(1, int(numWindows - 1)):
        smallerWindow = eval_df[(eval_df.time >= startValue) & (eval_df.time < stopValue)]

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

    print(accuracy_score(eval_df['label'], eval_df['pred_class']))

    return true_window, pred_window


# %%
model = load_model('saved_files/load_lstm_id_model_2.h5')
BenTestSet = pd.concat(threshold_estimation, ignore_index=True)

start = time.time()
cols = ['id', 'payload', 'time', 'time_abs', 'label']
df = BenTestSet[cols]

# convert df data into a series
BenId = pd.Series(df['id'])
BenId = BenId.str.cat(sep=' ')

sequence_data = tokenizer.texts_to_sequences([BenId])[0]
X, y = seq(sequence_data)
eval_df_benign = df[5:-5]  # 3 - no of context words

start = time.time()

for i in X:
    # print(i)
    x_input = np.expand_dims(i, axis=0)
    pred_RPM = model.predict_on_batch(x_input)
end = time.time()
print((end-start)/len(X))
eval_df_benign['true_id'] = y

#%% Threshold calculation
# find the predicted probability for each ID
pred_prob = []
for i in range(len(pred_RPM)):
    id_prob = pred_RPM[i][y[i]]  # tokenizer.word_index to get the mapping of id and index
    pred_prob.append(id_prob)

eval_df_benign['id_pred_prob'] = pred_prob
# threshold considering all IDs
threshold_all = eval_df_benign['id_pred_prob'].quantile(0.001)

# threshold calculation for each id
threshold_df = eval_df_benign.groupby('id')['id_pred_prob'].min()  # defalut - min()
id_threshold_dic = dict(threshold_df)
end = time.time()
tot_time = end - start
print('groupby time : ', tot_time)

# %%
# density plot visualization
df_id = eval_df_benign[eval_df_benign.id == '4E7']
#df_id = eval_df_benign
plt.figure(figsize=(10, 6), dpi=80)
# plt.title('density plot 580', fontsize=16)
sns.distplot(df_id['id_pred_prob'], bins=20, kde=True, color='blue');
plt.xlabel('softmax probability', fontsize = 16)
plt.ylabel('frequency', fontsize = 16)
plt.show()
plt.figure(figsize=(10, 6), dpi=80)
sns.distplot(df_id['ID_time_diff'], bins=20, kde=True, color='blue');
plt.xlabel('ID inter arrival time', fontsize = 16)
plt.ylabel('frequency', fontsize = 16)
plt.show()


# %%
model = load_model('saved_files/load_lstm_id_model_2.h5')
attacks = [ID_speedometer, ID_max_speedometer_mas, ID_corr_sig, ID_corr_sig_mas, ID_reverse_light_on,
           ID_reverse_light_on_mas, ID_reverse_light_off, ID_reverse_light_off_mas]
#
for i in attacks:
    # convert df data into a series
    print('Attack : ', i.name)
    df = i
    BenId = pd.Series(df['id'])
    BenId = BenId.str.cat(sep=' ')

    start = time.time()
    sequence_data = tokenizer.texts_to_sequences([BenId])[0]
    X, y = seq(sequence_data)
    eval_df = df[5:-5] # [5:-5] for dgx model


    start = time.time()
    pred_RPM = model.predict(X)
    end = time.time()
    print((end-start)/len(X))

    true_window, pred_window = results_evaluation(pred_RPM, eval_df, y, winRatio, id_threshold_dic, threshold_all)

    gru_list = pred_window

    # latency calculation
    end = time.time()
    tot_time = end - start
    print('Total time :', tot_time)
    print('Time for one frame :', tot_time / len(eval_df))

    print(classification_report(true_window, pred_window))
    cm = confusion_matrix(true_window, pred_window)
    print(cm)

    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    print('FP rate', FP*100/(FP+TN))
    print('FN rate', FN*100/(FN+TP))

    print('New attack data loading....')
    print('**************************************************')
    print()

# %%
# Ensemble model predictions
or_list = [a or b for a,b in zip(gru_list, payload_list)]

print(classification_report(true_window, or_list))
cm = confusion_matrix(true_window, or_list)
print(cm)

TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print('FP rate', FP*100/(FP+TN))
print('FN rate', FN*100/(FN+TP))


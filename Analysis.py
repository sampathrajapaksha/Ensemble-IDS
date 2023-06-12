# %%
# *********************** Data Analysis *****************************

# import libraries
import pickle
from datetime import time
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import random
import time
import plotly.express as px
import plotly.io as pio
import os
dirname = os.getcwd()
# %%
from Payload_model import autoencoder_df, df_basic_short, df_basic_long, X_train, df_extended_short, \
    autoencoder_df_train_id, df_reverse

#%%
''' cramers_corrected_stat calculate the association between each pair of variables'''

def cramers_corrected_stat(confusion_matrix):

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

combined_df = autoencoder_df.copy()
combined_df['id'] = autoencoder_df_train_id['id']
selected_IDs = list(combined_df['id'])

#%%
start = time.time()
id_dict ={}
for k in selected_IDs:
    filtered_DF = combined_df[combined_df.id == k]
    id_vars = filtered_DF[filtered_DF.columns[filtered_DF.columns.str.contains(k)]].columns
    print(k)
    print(id_vars)
    var_columns = filtered_DF.columns[:-1]
    var_1 = var_columns
    var_2 = id_vars
    id_list = []
    corr_list = []
    var_dic = {}
    # start = time()
    for i in var_2:
        print(i)
        for j in var_1:
            confusion_matrix = pd.crosstab(filtered_DF[i], filtered_DF[j])
            corr_value = cramers_corrected_stat(confusion_matrix.to_numpy())
            id_list.append(j)
            corr_list.append(corr_value)
            # create a dic with variables and correlations
            cat_corr = dict(zip(id_list,corr_list))
            # select keys(variables) with values greater than n (parameter)
            d = dict((k,v) for k,v in cat_corr.items() if v>0.80)
            # find top 5 max associations
            max_correlations = sorted(d, key=d.get, reverse=True)[:5]
            var_dic[i] = max_correlations

    # extract lists in dictionary into one list  (set is to remove duplicates if any)
    # new_var = dict((k,v) for k, v in var_dic.items() if len(v)>1) # to remove vars which does not have any correlations
    check_lists = list(set(list(np.hstack(var_dic.values()))))
    id_dict[k] = check_lists

end = time.time()
Total_time = end-start
print('Total time :', Total_time)

with open('corr_list_id', 'wb') as fp:
    pickle.dump(id_dict, fp)

#%%
# for one variable
combined_df = autoencoder_df.copy()
var_columns = combined_df.columns[:]
var_1 = var_columns
var_2 = '0D0_d3_int'
corr_list = []
id_list = []
for i in var_1:
    confusion_matrix = pd.crosstab(combined_df[i], combined_df[var_2])
    corr_value = cramers_corrected_stat(confusion_matrix.to_numpy())
    id_list.append(i)
    corr_list.append(corr_value)
    # print(i, corr_value)

# create a dic with variables and correlations
cat_corr = dict(zip(id_list,corr_list))

# select keys(variables) with values greater than n (parameter)
d = dict((k,v) for k,v in cat_corr.items() if v>0.80)

#%%
print('pearson correlation benign:')
pear_id_list = []
pear_corr_list = []
var_2 = '0D0_d3_int'
for i in var_1:
    perason_corr = combined_df[var_2].corr(combined_df[i])
    pear_id_list.append(i)
    pear_corr_list.append(perason_corr)

# create a dic with variables and correlations
pearson_corralation = dict(zip(pear_id_list, pear_corr_list))

# select keys(variables) with values greater than n (parameter)
d_pear = dict((k,v) for k,v in pearson_corralation.items() if v>0.8)

#%%
# to visualize associations
def data_creation_vis(combined_df):
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

    time_df = combined_df.copy()
    combined_df = combined_df.drop(col_list_0, axis=1)  # remove constant value variables
    # load saved column list
    column_list = os.path.join(dirname, 'saved_files/column_list')
    with open(column_list, 'rb') as fp:
        df_col_list = pickle.load(fp)
    print('number of columns :', len(df_col_list))
    combined_df = combined_df[df_col_list]
    combined_df_id = combined_df[['id']]

    # AutoEncoder model
    autoencoder_df = combined_df.drop(combined_df.columns[:11], axis=1) # select only relevent variables
    print('autoencoder_df size')
    print(autoencoder_df.shape)

    # feature scaling
    print('apply scaling')
    minmax_scale = os.path.join(dirname, 'saved_files/minmax_scale.sav')
    fit_scaling = pickle.load(open(minmax_scale, 'rb'))
    fit_apply = fit_scaling.transform(autoencoder_df)
    print('applied scalling')
    autoencoder_df = pd.DataFrame(fit_apply, columns=autoencoder_df.columns, index=autoencoder_df.index)

    print('replace values for unassociated variables')
    # replace values with Zero for unassociated varaiable
    autoencoder_df['id'] = combined_df['id']
    all_vars = autoencoder_df.columns[:-1]

    # shuffle before ID drop
    # autoencoder_df = sklearn.utils.shuffle(autoencoder_df)
    autoencoder_df_id = autoencoder_df.copy()
    autoencoder_df_id['time_abs'] = time_df['time_abs']
    autoencoder_df = autoencoder_df.drop('id', axis=1)

    df_shuffle = autoencoder_df.copy()
    X_train = df_shuffle.to_numpy() # convert to numpy array
    print('X_train created')

    return X_train, autoencoder_df, autoencoder_df_id

train_dataset = pd.concat([df_basic_short], ignore_index=True)
X_train_vis, autoencoder_df_vis, autoencoder_df_train_id_vis = data_creation_vis(train_dataset)

#%%
# for one variable
combined_df = autoencoder_df_vis.copy()
var_columns = combined_df.columns[:]
var_1 = var_columns
var_2 = '0D0_d3_int'
corr_list = []
id_list = []
for i in var_1:
    confusion_matrix = pd.crosstab(combined_df[i], combined_df[var_2])
    corr_value = cramers_corrected_stat(confusion_matrix.to_numpy())
    id_list.append(i)
    corr_list.append(corr_value)

# create a dic with variables and correlations
cat_corr = dict(zip(id_list,corr_list))

# # select keys(variables) with values greater than n (parameter)
d = dict((k,v) for k,v in cat_corr.items() if v>0.80)

#%%
print('pearson correlation benign:')
pear_id_list = []
pear_corr_list = []
var_2 = '0D0_d6_int'
for i in var_1:
    perason_corr = combined_df[var_2].corr(combined_df[i])
    pear_id_list.append(i)
    pear_corr_list.append(perason_corr)

# create a dic with variables and correlations
pearson_corralation = dict(zip(pear_id_list, pear_corr_list))

# select keys(variables) with values greater than n (parameter)
d_pear = dict((k,v) for k,v in pearson_corralation.items() if v>0.8)


#%%
# Association visualization
import plotly.express as px
import plotly.io as pio
id_0d0 = autoencoder_df_train_id_vis[autoencoder_df_train_id_vis['id']=='0D0']
id_0d0 = id_0d0.reset_index(drop=True)

pio.renderers.default = "browser"
def custom_legend_name(new_names):
    for i, new_name in enumerate(new_names):
        fig.data[i].name = new_name

fig = px.line(id_0d0, x =id_0d0.time_abs, y = ['0D0_d3_int', '274_d7_int'], labels={'time_abs':'Time (s)', 'value':'Value'})
fig.update_layout(
    autosize=False,
    width=1600,
    height=500,
    showlegend=True,
    legend_title="Variables",
    font=dict(size=19),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1)
)
custom_legend_name(['0D0_D3','498_D1'])
fig.show()

#%%
# unique payload value identification
dataframe = df_basic_long
id_groups = dataframe.groupby('id')
id_groups_n = id_groups.agg(
    {'d1_int': 'nunique', 'd2_int': 'nunique', 'd3_int': 'nunique', 'd4_int': 'nunique', 'd5_int': 'nunique',
     'd6_int': 'nunique', 'd7_int': 'nunique', 'd8_int': 'nunique'})
id_groups_sample = id_groups_n.reset_index()

# find unique number of variables
val_df_basic_sample = id_groups_n[['d1_int', 'd2_int', 'd3_int', 'd4_int', 'd5_int', 'd6_int', 'd7_int', 'd8_int']]
val_df_basic_sample = pd.value_counts(val_df_basic_sample.values.ravel())

#%%
# ID frequency Analysis
df = df_basic_short.copy()
df['ID_time_diff'] = df.groupby('id')['time_abs'].diff()
df['ID_time_diff'] = df['ID_time_diff'].fillna(df.groupby('id')['ID_time_diff'].transform('mean'))

freq = df.groupby(['id'])['ID_time_diff'].mean()
freq = freq.sort_values(ascending=True)


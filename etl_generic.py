# import necessary libraries

import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from functools import lru_cache
import gc
import shutil
import nvtabular as nvt

# read data: edit as required
train_data_dir = '.'
test_data_dir = '.'
task = 'task1'
PREDS_PER_SESSION = 100


# Necessary functions
def read_product_data():
    return pd.read_csv(os.path.join(train_data_dir, 'products_train.csv'))


def read_train_data():
    return pd.read_csv(os.path.join(train_data_dir, 'sessions_train.csv'))


def read_test_data(task):
    return pd.read_csv(os.path.join(test_data_dir, f'sessions_test_{task}.csv'))

# Function for data cleaning
def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l


# read data

products = read_product_data()
train_sessions = read_train_data()
test_sessions = read_test_data(task)

train_sessions_UK = pd.DataFrame()
test_sessions_UK = pd.DataFrame()
products_UK = pd.DataFrame()

# locale filtering
# Filter train_sessions for rows with locale='DE'
train_sessions_UK = train_sessions[train_sessions['locale'] == 'DE']
test_sessions_UK = test_sessions[test_sessions['locale'] == 'DE']
products_UK = products[products['locale'] == 'DE']
train_sessions_UK.head()

# convert items to list

dat_train = train_sessions_UK['prev_items'].tolist()
dat_test = test_sessions_UK['prev_items'].tolist()

cleaned_train = []
for i in range(len(dat_train)):
    cleaned_train.append(str2list(dat_train[i]))

# cleaned_train
cleaned_test = []
for i in range(len(dat_test)):
    cleaned_test.append(str2list(dat_test[i]))

# cleaned_test
train_sessions_UK['prev_items'] = cleaned_train
test_sessions_UK['prev_items'] = cleaned_test

# Add next item within item array
train_sessions_UK['prev_items'] = train_sessions_UK.apply(lambda row: row['prev_items'] + [row['next_item']], axis=1)

# Adding session_id based on index
train_sessions_UK.reset_index(level=0, inplace=True)
train_sessions_UK.rename(columns={'index': 'session_id'}, inplace=True)
# Split 'prev_items' into separate rows
train_sessions_UK = train_sessions_UK.explode('prev_items')

# Adding session_id based on index
test_sessions_UK.reset_index(level=0, inplace=True)
test_sessions_UK.rename(columns={'index': 'session_id'}, inplace=True)
# Split 'prev_items' into separate rows
test_sessions_UK = test_sessions_UK.explode('prev_items')

# Add value to make train and test distinct (not necessary for other locales )
train_sessions_UK['session_id'] = train_sessions_UK['session_id'] + 1000000

# merge product attribute to item list
session_attribute_train = pd.merge(train_sessions_UK, products_UK, left_on='prev_items', right_on='id',
                                   how='left').drop(['id', 'locale_x', 'locale_y'], axis=1)

session_attribute_test = pd.merge(test_sessions_UK, products_UK, left_on='prev_items', right_on='id', how='left').drop(
    ['id', 'locale_x', 'locale_y'], axis=1)

# Now drop next item column
raw_df1 = session_attribute_train.drop('next_item', axis=1)

#join train and test data
join_data = pd.concat([raw_df1, session_attribute_test])
#finalize raw_df
raw_df = join_data

# categorify session id
cols = list(raw_df.columns)
cols.remove('session_id')
print(cols)

# load data
df_event = nvt.Dataset(raw_df)

# categorify user_session
cat_feats = ['session_id'] >> nvt.ops.Categorify()

workflow = nvt.Workflow(cols + cat_feats)
workflow.fit(df_event)
df = workflow.transform(df_event).to_ddf().compute()

# categorify features
item_id = ['prev_items'] >> nvt.ops.TagAsItemID()
cat_feats = item_id + ['title', 'price', 'brand', 'color', 'size', 'model', 'material', 'author',
                       'desc'] >> nvt.ops.Categorify(start_index=1)

# Smoothing price long-tailed distribution and applying standardization
price_log = ['price'] >> nvt.ops.LogOp() >> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(
    name='price_log_norm')

from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags


# Relative price to the average price for the category_id
def relative_price_to_avg_categ(col, gdf):
    epsilon = 1e-5
    col = ((gdf['price'] - col) / (col + epsilon)) * (col > 0).astype(int)
    return col


avg_category_id_pr = ['brand'] >> nvt.ops.JoinGroupby(cont_cols=['price'], stats=["mean"]) >> nvt.ops.Rename(
    name='avg_category_id_price')
relative_price_to_avg_category = (
        avg_category_id_pr >>
        nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=['price']) >>
        nvt.ops.Rename(name="relative_price_to_avg_categ_id") >>
        nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
)

groupby_feats = ['session_id'] + cat_feats + price_log + relative_price_to_avg_category

price_log + relative_price_to_avg_category

# Define Groupby Workflow
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["session_id"],
    aggs={
        'prev_items': ["list", "count"],
        'title': ["list"],
        'price_log_norm': ["list"],
        'relative_price_to_avg_categ_id': ["list"],
        'brand': ["list"],
        'color': ["list"],
        'size': ["list"],
        'model': ["list"],
        'material': ["list"],
        'author': ["list"],
        'desc': ["list"]
    },
    name_sep="-")

groupby_features_list = groupby_features[
    'prev_items-list',
    'title-list',
    'price_log_norm-list',
    'relative_price_to_avg_categ_id-list',
    'brand-list',
    'color-list',
    'size-list',
    'model-list',
    'material-list',
    'author-list',
    'desc-list'
]
#define max_sessions length and min_sessions length
SESSIONS_MAX_LENGTH = 10
MINIMUM_SESSION_LENGTH = 2

#trim to max_sessions
groupby_features_trim = groupby_features_list >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH)

sess_id = groupby_features['session_id'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
selected_features = sess_id + groupby_features['prev_items-count'] + groupby_features_trim

filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["prev_items-count"] >= MINIMUM_SESSION_LENGTH)

# create workflow
workflow = nvt.Workflow(filtered_sessions)
dataset = nvt.Dataset(df)
# Learn features statistics necessary of the preprocessing workflow
# The following will generate schema.pbtxt file in the provided folder and export the parquet files.

workflow.fit_transform(dataset).to_parquet(os.path.join('/kaggle/working/', "processed_nvt"))

workflow.output_schema
# read data
INPUT_DATA_DIR = '/kaggle/working/'
# read in the processed train dataset
sessions_gdf = pd.read_parquet(os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet"))
# read unique session id and item_id
session_ids = pd.read_parquet('/kaggle/working/categories/unique.session_id.parquet')
item_ids = pd.read_parquet('/kaggle/working/categories/unique.prev_items.parquet')

# First, make sure the index of `session_ids` is integer type, this ensures the matching works correctly
session_ids.index = session_ids.index.astype(int)

# Then map `session_id` in `sessions_df` to index in `session_ids`
sessions_gdf['session_org'] = sessions_gdf['session_id'].map(session_ids['session_id'])

sessions_df_test = sessions_gdf[sessions_gdf['session_org'] < 999999]
sessions_gdf = sessions_gdf.drop('session_org', axis=1)
sessions_df_test = sessions_df_test.drop('session_org', axis=1)

sessions_df_test.to_parquet('/kaggle/working/processed_nvt/test_0.parquet')

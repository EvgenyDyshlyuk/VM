# %%
import pickle

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score, accuracy_score, recall_score
from sklearn.dummy import DummyClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2 / 0.8

# %%
# LOAD DATA
# df = pd.read_csv('gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv')
# df.to_csv(r'input/task_1.csv', index=False)
df = pd.read_csv(r'input/task_1.csv')
df
# ----------------------------------------------------------------------------------------------------------------------
# %%
# EDA
df.info()

# %%
df.isnull().sum().sum()  # no NaN values

# %%
# decode all object columns to category
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

# for each category column, count the number of unique values and print
for col in df.select_dtypes(include='category').columns:
    print(f'{col}: {df[col].nunique()}')
# looks good

# %%
df.describe()

# %%
# plot all int features using seaborn displot, decode category columns to int for plotting
df_tmp = df.copy()
for col in df_tmp.columns:
    sns.displot(df_tmp[col])

# age is unusal - probably in months

# %%
target = 'Adopted'
# change tartget to 0/1
df[target] = df[target].map({'Yes': 1, 'No': 0})
df[target].value_counts()
# a bit unbalanced - use sample weight
# ----------------------------------------------------------------------------------------------------------------------
# %%
# TRAIN/VAL/TEST SPLIT
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[target])
train, val = train_test_split(train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=train[target])

X_train = train.drop(target, axis=1)
y_train = train[target]
X_val = val.drop(target, axis=1)
y_val = val[target]
X_test = test.drop(target, axis=1)
y_test = test[target]

print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print()
# map sample weight from 0 and 1 values
samples_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]
print(f'samples_ratio: {samples_ratio}')
sw_train = y_train.map({0: 1, 1: samples_ratio}).to_numpy()
sw_val = y_val.map({0: 1, 1: samples_ratio}).to_numpy()
sw_test = y_test.map({0: 1, 1: samples_ratio}).to_numpy()

# first 10 samples and weights:
dict = {'y_train': y_train[:10], 'sw_train': sw_train[:10]}
pd.DataFrame(dict)

# ----------------------------------------------------------------------------------------------------------------------
# %%
# TRAIN BASELINE MODEL
model = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(f'Dummy val accuracy: {model.score(X_val, y_val)}')
print(f'Dummy val f1: {f1_score(y_val, y_pred)}')
print(f'Dummy val recall: {recall_score(y_val, y_pred)}')
# %%
# TRAIN LGBM MODEL
early_stopping_callback = lgb.early_stopping(5)

model = LGBMClassifier(random_state=RANDOM_STATE, objective='binary', metric='binary_logloss')

model.fit(X_train, y_train,
          sample_weight=sw_train,
          eval_set=[(X_val, y_val)],
          eval_sample_weight=[sw_val],
          eval_names=['val'],
          eval_metric=['binary_logloss'],
          callbacks=[early_stopping_callback],
          verbose=1)

y_pred_proba_val = model.predict_proba(X_val, num_iteration=model.best_iteration_)
logloss_val = log_loss(y_val, y_pred_proba_val[:, 1], sample_weight=sw_val)
print("Binary Log Loss Val:", logloss_val)

y_pred_proba_test = model.predict_proba(X_test)
logloss_test = log_loss(y_test, y_pred_proba_test[:, 1], sample_weight=sw_test)
print("Binary Log Loss Test:", logloss_test)

# %%
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# print binary logloss
print(f'Train accuracy: {accuracy_score(y_train, y_pred_train)}')
print(f'Val accuracy: {accuracy_score(y_val, y_pred_val)}')
print(f'Test accuracy: {accuracy_score(y_test, y_pred_test)}')
print()
print(f'Train f1: {f1_score(y_train, y_pred_train)}')
print(f'Val f1: {f1_score(y_val, y_pred_val)}')
print(f'Test f1: {f1_score(y_test, y_pred_test)}')
print()
print(f'Train recall: {recall_score(y_train, y_pred_train)}')
print(f'Val recall: {recall_score(y_val, y_pred_val)}')
print(f'Test recall: {recall_score(y_test, y_pred_test)}')

# print first 10 predictions and true values
print()
print(f'First 10 predictions: {y_pred_test[:10]}')
print(f'First 10 true values: {y_test[:10].to_numpy()}')

# %%
# plot feature importance
lgb.plot_importance(model, figsize=(15, 5))
# ----------------------------------------------------------------------------------------------------------------------

# %%
# Pickle model to artifacts folder
with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# load model and check performance
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)
print(f'Train accuracy: {accuracy_score(y_train, y_pred_train)}')
print(f'Val accuracy: {accuracy_score(y_val, y_pred_val)}')
print(f'Test accuracy: {accuracy_score(y_test, y_pred_test)}')

# %%

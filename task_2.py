# %%
import pandas as pd
import seaborn as sns

from sklearn.metrics import f1_score, accuracy_score, recall_score

import pickle


# %%
# LOAD DATA
# df = pd.read_csv('gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv')
# df.to_csv(r'input/task_2.csv', index=False)
df = pd.read_csv(r'input/task_2.csv')
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

y = df[target]
X = df.drop(target, axis=1)

# load model and check performance
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

# %%
y_pred = model.predict(X)

print(f'Train accuracy: {accuracy_score(y, y_pred)}')
print(f'Train f1: {f1_score(y, y_pred)}')
print(f'Train recall: {recall_score(y, y_pred)}')

# %%
# map y_pred back to Yes/No
y_pred = pd.Series(y_pred).map({1: 'Yes', 0: 'No'})
y_pred.to_csv('output/predicts.csv', index=False, header=False)

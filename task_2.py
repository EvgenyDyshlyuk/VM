# %%
from typing import Union
from pathlib import Path

import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, recall_score

import pickle

from IPython.display import display


def predict(input_file_path: Union[str, Path],
            target: str,
            model_path: Union[str, Path],
            output_results_path: Union[str, Path]) -> None:
    """
    Predicts the target column for the input file using the saved model and saves the results to the output file.

    Args:
        input_file_path (Union[str, Path]): Path to the input file.
        target (str): Name of the target column.
        model_path (Union[str, Path]): Path to the saved model.
        output_results_path (Union[str, Path]): Path to the output file.
    """

    df = pd.read_csv(input_file_path)

    with open(Path(model_path), 'rb') as f:
        model = pickle.load(f)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    df[target] = df[target].map({'Yes': 1, 'No': 0})
    X = df.drop(target, axis=1)
    y = df[target]

    y_pred = model.predict(X)

    print(f'Accuracy: {round(accuracy_score(y, y_pred), 3)}')
    print(f'F1: {round(f1_score(y, y_pred), 3)}')
    print(f'Recall: {round(recall_score(y, y_pred), 3)}')

    y = pd.Series(y).map({1: 'Yes', 0: 'No'})
    y_pred = pd.Series(y_pred).map({1: 'Yes', 0: 'No'})
    df = pd.concat([X, y, y_pred], axis=1)
    df.columns = list(X.columns) + [target, target + '_prediction']

    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].astype('object')

    df.to_csv(output_results_path, index=False, header=True)
    print('Results saved to: ', output_results_path)

    # print dtype of each column
    print('Data types:')
    display(df.dtypes)
    display(df.head(2), df.tail(2))


if __name__ == '__main__':
    predict(
        input_file_path=Path('input', 'task_2.csv'),
        target='Adopted',
        model_path=Path('artifacts', 'model.pkl'),
        output_results_path=Path('output', 'results.csv'),
    )
# %%

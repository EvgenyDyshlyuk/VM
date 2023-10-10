import unittest
import pickle
import pandas as pd
from pathlib import Path
from task_2 import predict


class TestPredict(unittest.TestCase):

    def setUp(self):
        self.input_file_path = Path('tmp_test.csv')
        self.output_results_path = Path('tmp_results.csv')
        self.model_path = Path('artifacts', 'model.pkl')

        # create test data (3 first rows of the original data)
        self.test_data = pd.DataFrame({
            'Type': ['Cat', 'Cat', 'Dog'],
            'Age': [3, 1, 1],
            'Breed1': ['Tabby', 'Domestic Medium Hair', 'Mixed Breed'],
            'Gender': ['Male', 'Male', 'Male'],
            'Color1': ['Black', 'Black', 'Brown'],
            'Color2': ['White', 'Brown', 'White'],
            'MaturitySize': ['Small', 'Medium', 'Medium'],
            'FurLength': ['Short', 'Medium', 'Medium'],
            'Vaccinated': ['No', 'Not Sure', 'Yes'],
            'Sterilized': ['No', 'Not Sure', 'No'],
            'Health': ['Healthy', 'Healthy', 'Healthy'],
            'Fee': [100, 0, 0],
            'PhotoAmt': [1, 2, 7],
            'Adopted': ['Yes', 'Yes', 'Yes']
        })

        self.test_data.to_csv(self.input_file_path, index=False)

    def test_predict(self):
        # test predict function
        predict(
            input_file_path=self.input_file_path,
            target='Adopted',
            model_path=self.model_path,
            output_results_path=self.output_results_path
        )

        # check that model file exists
        self.assertTrue(self.model_path.exists())
        self.model = pickle.load(open(self.model_path, 'rb'))

        # check if output file exists
        self.assertTrue(self.output_results_path.exists())

        # check if output file has correct columns
        expected_columns = ['Type', 'Age', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize',
                            'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Fee', 'PhotoAmt', 'Adopted',
                            'Adopted_prediction']
        output_data = pd.read_csv(self.output_results_path)
        self.assertListEqual(list(output_data.columns), expected_columns)

        # check if output file has correct values in the 'Adopted_prediction' column
        expected_predictions = set(['Yes', 'No'])
        predicted_values = list(output_data['Adopted_prediction'])
        self.assertTrue(set(predicted_values).issubset(expected_predictions))

        # check if output file has correct number of rows: 3
        self.assertEqual(len(output_data), 3)

        # check data types in the output file
        expected_dtypes = {
            'Type': 'O',
            'Age': 'int64',
            'Breed1': 'O',
            'Gender': 'O',
            'Color1': 'O',
            'Color2': 'O',
            'MaturitySize': 'O',
            'FurLength': 'O',
            'Vaccinated': 'O',
            'Sterilized': 'O',
            'Health': 'O',
            'Fee': 'int64',
            'PhotoAmt': 'int64',
            'Adopted': 'O',
            'Adopted_prediction': 'O',
        }
        self.assertDictEqual(output_data.dtypes.to_dict(), expected_dtypes)

    def tearDown(self):
        # remove test data and model
        self.input_file_path.unlink()
        self.output_results_path.unlink()

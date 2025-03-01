import unittest
import pandas as pd
from MEDS_preprocess.preprocessors.preMEDS import MEDSPreprocessor
from MEDS_preprocess.preprocessors.constants import SUBJECT_ID

class TestMEDSPreprocessor(unittest.TestCase):
    def test_factorize_subject_id_basic(self):
        # Test with simple string IDs
        input_df = pd.DataFrame({
            SUBJECT_ID: ['A', 'B', 'A', 'C', 'B'],
            'other_col': [1, 2, 3, 4, 5]
        })
        
        result_df, mapping = MEDSPreprocessor._factorize_subject_id(input_df)
        
        # Check mapping properties
        self.assertEqual(len(mapping), 3)  # Three unique values
        self.assertTrue(all(isinstance(v, int) for v in mapping.values()))  # All values are integers
        self.assertTrue(all(v > 0 for v in mapping.values()))  # All values are positive
        
        # Check DataFrame properties
        self.assertEqual(result_df[SUBJECT_ID].nunique(), 3)  # Three unique values
        self.assertEqual(result_df[SUBJECT_ID].dtype, 'int64')  # Integer type
        self.assertNotIn('integer_id', result_df.columns)  # Helper column was dropped
        self.assertEqual(len(result_df), len(input_df))  # No rows were lost

    def test_factorize_subject_id_empty(self):
        # Test with empty DataFrame
        input_df = pd.DataFrame({SUBJECT_ID: [], 'other_col': []})
        
        result_df, mapping = MEDSPreprocessor._factorize_subject_id(input_df)
        
        self.assertEqual(len(mapping), 0)
        self.assertEqual(len(result_df), 0)

    def test_factorize_subject_id_mixed_types(self):
        # Test with mixed types (strings and numbers)
        input_df = pd.DataFrame({
            SUBJECT_ID: ['A', '123', 'B', '456', 'A'],
            'other_col': [1, 2, 3, 4, 5]
        })
        
        result_df, mapping = MEDSPreprocessor._factorize_subject_id(input_df)
        
        self.assertEqual(len(mapping), 4)  # Four unique values
        self.assertTrue(all(isinstance(v, int) for v in mapping.values()))
        self.assertEqual(result_df[SUBJECT_ID].dtype, 'int64')

    def test_factorize_subject_id_preserves_other_columns(self):
        # Test that other columns remain unchanged
        input_df = pd.DataFrame({
            SUBJECT_ID: ['A', 'B', 'A'],
            'col1': [1, 2, 3],
            'col2': ['x', 'y', 'z']
        })
        
        result_df, mapping = MEDSPreprocessor._factorize_subject_id(input_df)
        
        self.assertEqual(list(result_df.columns), [SUBJECT_ID, 'col1', 'col2'])
        self.assertEqual(result_df['col1'].tolist(), [1, 2, 3])
        self.assertEqual(result_df['col2'].tolist(), ['x', 'y', 'z'])

if __name__ == '__main__':
    unittest.main()

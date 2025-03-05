import unittest
from unittest.mock import patch

import pandas as pd

from MEDS_preprocess.preprocessors.constants import (ADMISSION, CODE,
                                                     DISCHARGE, SUBJECT_ID,
                                                     TIMESTAMP)
from MEDS_preprocess.preprocessors.preMEDS import ConceptProcessor


class TestConceptProcessor(unittest.TestCase):
    def test_select_and_rename_columns(self):
        # Create a DataFrame with three columns; only two are defined in the mapping.
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": [True, False, True]
        })
        columns_map = {"A": "alpha", "B": "beta"}
        result = ConceptProcessor._select_and_rename_columns(df, columns_map)
        self.assertListEqual(list(result.columns), ["alpha", "beta"])
        # Column "C" should be dropped.
        self.assertNotIn("C", result.columns)

    def test_fill_missing_values_with_regex(self):
        # Create a DataFrame with a target column that has missing values.
        df = pd.DataFrame({
            "target": [None, "val2", None],
            "fillcol": ["(D123)", "(D456)", "(D789)"]
        })
        fillna_cfg = {
            "target": {"column": "fillcol", "regex": r"\((D\d+)\)"}
        }
        result = ConceptProcessor._fill_missing_values(df, fillna_cfg)
        # Expected: first row filled with "D123", second row unchanged, third row with "D789".
        expected = ["D123", "val2", "D789"]
        self.assertEqual(result["target"].tolist(), expected)
        # The filler column should be dropped.
        self.assertNotIn("fillcol", result.columns)

    def test_process_codes_prefix_and_fill(self):
        # Test that _process_codes fills missing values from the filler column and then adds the prefix.
        df = pd.DataFrame({
            CODE: [None, "ABC"],
            "fill_code": ["X", "Y"]
        })
        concept_config = {
            "fillna": {
                CODE: {"column": "fill_code", "regex": None}
            },
            "code_prefix": "PRE_"
        }
        result = ConceptProcessor._process_codes(df, concept_config)
        # Row1: missing value filled with "X" then prefixed → "PRE_X"
        # Row2: "ABC" becomes "PRE_ABC"
        expected = ["PRE_X", "PRE_ABC"]
        self.assertEqual(result[CODE].tolist(), expected)
        self.assertNotIn("fill_code", result.columns)

    def test_convert_and_clean_data(self):
        # Create a DataFrame with a numeric column and subject_id that needs mapping.
        df = pd.DataFrame({
            SUBJECT_ID: ["A", "B", "A", "C"],
            CODE: ["x", "y", "x", "z"],
            TIMESTAMP: ["2022-01-01", "2022-01-02", None, "2022-01-03"],
            "num": ["1", "2", "3", "not_a_number"]
        })
        concept_config = {"numeric_columns": ["num"]}
        subject_id_mapping = {"A": 10, "B": 20, "C": 30}
        result = ConceptProcessor._convert_and_clean_data(df, concept_config, subject_id_mapping)
        # Row with non-numeric "num" and missing TIMESTAMP should be dropped.
        self.assertTrue(result["num"].apply(lambda x: isinstance(x, (int, float))).all())
        self.assertFalse(result[TIMESTAMP].isnull().any())
        self.assertTrue(set(result[SUBJECT_ID].unique()).issubset({10, 20, 30}))

    def test_merge_admissions(self):
        # Create overlapping admission intervals for subject 1 and a separate interval for subject 2.
        data = {
            SUBJECT_ID: [1, 1, 2],
            ADMISSION: ["2022-01-01 00:00", "2022-01-01 11:00", "2022-01-02 00:00"],
            DISCHARGE: ["2022-01-01 12:00", "2022-01-01 14:00", "2022-01-02 12:00"]
        }
        df = pd.DataFrame(data)
        result = ConceptProcessor._merge_admissions(df)
        # For subject 1, the two intervals should merge into one row.
        merged_1 = result[result[SUBJECT_ID] == 1]
        self.assertEqual(len(merged_1), 1)
        self.assertEqual(merged_1.iloc[0][ADMISSION], pd.to_datetime("2022-01-01 00:00"))
        self.assertEqual(merged_1.iloc[0][DISCHARGE], pd.to_datetime("2022-01-01 14:00"))
        # For subject 2, the row remains unchanged.
        merged_2 = result[result[SUBJECT_ID] == 2]
        self.assertEqual(len(merged_2), 1)
        self.assertEqual(merged_2.iloc[0][ADMISSION], pd.to_datetime("2022-01-02 00:00"))

    def test_check_columns_error(self):
        # Test that check_columns raises an error when required columns are missing.
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        columns_map = {"A": "alpha", "C": "gamma"}
        with self.assertRaises(ValueError) as context:
            ConceptProcessor.check_columns(df, columns_map)
        self.assertIn("Missing columns", str(context.exception))

    def test_postprocess_switch_unknown(self):
        df = pd.DataFrame()
        with self.assertRaises(ValueError):
            ConceptProcessor._postprocess_switch(df, "unknown_postprocess")

    def test_combine_datetime_columns(self):
        # Test data
        df = pd.DataFrame({
            'date_col': ['2023-01-01', '2023-01-02', '2023-01-03', None],
            'time_col': ['10:30:00', '11:45:00', '09:15:00', '08:00:00'],
            'other_col': [1, 2, 3, 4]
        })
        
        # Test config with drop_original=True (default)
        config_drop = {
            'combine_datetime': {
                'timestamp': {
                    'date_col': 'date_col',
                    'time_col': 'time_col'
                }
            }
        }
        
        result_drop = ConceptProcessor._combine_datetime_columns(df.copy(), config_drop)
        
        # Check that combined column exists
        self.assertIn('timestamp', result_drop.columns)
        # Check first row's datetime
        expected_dt = pd.to_datetime('2023-01-01 10:30:00')
        self.assertEqual(result_drop.loc[0, 'timestamp'], expected_dt)
        # Check that original columns are dropped
        self.assertNotIn('date_col', result_drop.columns)
        self.assertNotIn('time_col', result_drop.columns)
        
        # Test config with drop_original=False
        config_keep = {
            'combine_datetime': {
                'timestamp': {
                    'date_col': 'date_col',
                    'time_col': 'time_col',
                    'drop_original': False
                }
            }
        }
        
        result_keep = ConceptProcessor._combine_datetime_columns(df.copy(), config_keep)
        
        # Check that original columns are kept
        self.assertIn('date_col', result_keep.columns)
        self.assertIn('time_col', result_keep.columns)
        
        # Test with missing column
        config_missing = {
            'combine_datetime': {
                'timestamp': {
                    'date_col': 'missing_col',
                    'time_col': 'time_col'
                }
            }
        }
        
        result_missing = ConceptProcessor._combine_datetime_columns(df.copy(), config_missing)
        # Should not create the timestamp column
        self.assertNotIn('timestamp', result_missing.columns)
        
        # Test with multiple datetime combinations
        df_multi = pd.DataFrame({
            'date_start': ['2023-01-01', '2023-01-02'],
            'time_start': ['10:30:00', '11:45:00'],
            'date_end': ['2023-01-05', '2023-01-07'],
            'time_end': ['15:30:00', '16:45:00']
        })
        
        config_multi = {
            'combine_datetime': {
                'admission': {
                    'date_col': 'date_start',
                    'time_col': 'time_start'
                },
                'discharge': {
                    'date_col': 'date_end',
                    'time_col': 'time_end'
                }
            }
        }
        
        result_multi = ConceptProcessor._combine_datetime_columns(df_multi.copy(), config_multi)
        self.assertIn('admission', result_multi.columns)
        self.assertIn('discharge', result_multi.columns)

    def test_apply_secondary_mapping(self):
        # Test data
        df = pd.DataFrame({
            'drug_code': ['A01', 'B02', 'C03'],
            'value': [100, 200, 300]
        })
        
        mapping_df = pd.DataFrame({
            'VNR': ['A01', 'B02', 'C03', 'D04'],
            'PNAME': ['Drug A', 'Drug B', 'Drug C', 'Drug D']
        })
        
        config = {
            'secondary_mapping': {
                'left_on': 'drug_code',
                'right_on': 'VNR'
            }
        }
        
        # Mock the _load_mapping_file method
        with patch.object(ConceptProcessor, '_load_mapping_file', return_value=mapping_df):
            result = ConceptProcessor._apply_secondary_mapping(df.copy(), config)
            
            # Check that mapping was successful
            self.assertIn('PNAME', result.columns)
            self.assertNotIn('drug_code', result.columns)  # Should be dropped
            self.assertEqual(len(result), 3)  # All rows should match
            self.assertEqual(result.loc[0, 'PNAME'], 'Drug A')
        
        # Test with non-matching values
        df_non_match = pd.DataFrame({
            'drug_code': ['X99', 'Y88', 'Z77'],
            'value': [100, 200, 300]
        })
        
        with patch.object(ConceptProcessor, '_load_mapping_file', return_value=mapping_df):
            result_non_match = ConceptProcessor._apply_secondary_mapping(df_non_match.copy(), config)
            
            # With inner join, no rows should match
            self.assertEqual(len(result_non_match), 0)

    def test_convert_numeric_columns(self):
        # Test data with mix of numeric and non-numeric values
        df = pd.DataFrame({
            'col1': ['1', '2', '3', 'a'],
            'col2': ['10.5', '20.7', 'error', '40.2'],
            'col3': ['text1', 'text2', 'text3', 'text4']
        })
        
        # Config with some numeric columns
        config = {
            'numeric_columns': ['col1', 'col2']
        }
        
        result = ConceptProcessor._convert_numeric_columns(df.copy(), config)
        
        # Check that col1 and col2 are converted to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(result['col1']))
        self.assertTrue(pd.api.types.is_numeric_dtype(result['col2']))
        
        # Check that col3 remains as object/string
        self.assertTrue(pd.api.types.is_object_dtype(result['col3']))
        
        # Check that invalid values become NaN
        self.assertTrue(pd.isna(result.loc[3, 'col1']))
        self.assertTrue(pd.isna(result.loc[2, 'col2']))
        
        # Test with empty numeric_columns
        config_empty = {'numeric_columns': []}
        result_empty = ConceptProcessor._convert_numeric_columns(df.copy(), config_empty)
        
        # No columns should be converted
        self.assertTrue(pd.api.types.is_object_dtype(result_empty['col1']))
        self.assertTrue(pd.api.types.is_object_dtype(result_empty['col2']))
        
        # Test with non-existent columns
        config_missing = {'numeric_columns': ['col1', 'missing_col']}
        result_missing = ConceptProcessor._convert_numeric_columns(df.copy(), config_missing)
        
        # Only existing columns should be converted
        self.assertTrue(pd.api.types.is_numeric_dtype(result_missing['col1']))
        self.assertTrue(pd.api.types.is_object_dtype(result_missing['col2']))

    def test_unroll_columns(self):
        # Test data
        df = pd.DataFrame({
            'subject_id': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'specialty': ['cardio', 'neuro', 'ortho'],
            'unit_type': ['emergency', 'regular', 'emergency'],
            'other_col': ['a', 'b', 'c']
        })
        
        # Config with multiple unroll columns
        config = {
            'columns_map': {
                'original_timestamp': 'timestamp'
            },
            'unroll_columns': [
                {
                    'column': 'specialty',
                    'prefix': 'SPEC_'
                },
                {
                    'column': 'unit_type',
                    'prefix': 'UNIT_'
                }
            ]
        }
        
        result_dfs = ConceptProcessor._unroll_columns(df.copy(), config)
        
        # Should return 2 dataframes (one for each unrolled column)
        self.assertEqual(len(result_dfs), 2)
        
        # Check first dataframe (specialty)
        specialty_df = result_dfs[0]
        self.assertIn(CODE, specialty_df.columns)
        self.assertEqual(specialty_df.loc[0, CODE], 'SPEC_cardio')
        
        # Check second dataframe (unit_type)
        unit_df = result_dfs[1]
        self.assertIn(CODE, unit_df.columns)
        self.assertEqual(unit_df.loc[0, CODE], 'UNIT_emergency')
        
        # Test with no prefix
        config_no_prefix = {
            'columns_map': {
                'original_timestamp': 'timestamp'
            },
            'unroll_columns': [
                {
                    'column': 'specialty'
                }
            ]
        }
        
        result_no_prefix = ConceptProcessor._unroll_columns(df.copy(), config_no_prefix)
        self.assertEqual(len(result_no_prefix), 1)
        self.assertEqual(result_no_prefix[0].loc[0, CODE], 'cardio')  # No prefix
        
        # Test with no unroll_columns (just prefix)
        config_no_unroll = {
            'code_prefix': 'TEST_'
        }
        
        # Create df with CODE column
        df_with_code = df.copy()
        df_with_code[CODE] = ['code1', 'code2', 'code3']
        
        result_no_unroll = ConceptProcessor._unroll_columns(df_with_code, config_no_unroll)
        self.assertEqual(len(result_no_unroll), 1)
        self.assertEqual(result_no_unroll[0].loc[0, CODE], 'TEST_code1')

    def test_apply_main_mapping(self):
        # Test data
        df = pd.DataFrame({
            'dw_ek_kontakt': ['K001', 'K002', 'K003'],
            'code': ['D123', 'D456', 'D789']
        })
        
        mapping_df = pd.DataFrame({
            'dw_ek_kontakt': ['K001', 'K002', 'K003', 'K004'],
            SUBJECT_ID: ['P001', 'P002', 'P003', 'P004']
        })
        
        config = {
            'main_mapping': {
                'left_on': 'dw_ek_kontakt',
                'right_on': 'dw_ek_kontakt'
            }
        }
        
        # Mock the mapping file load more explicitly
        with patch.object(
            ConceptProcessor, 
            '_load_mapping_file',
            return_value=mapping_df,
            autospec=True  # Ensures the mock respects the method's signature
        ) as mock_load:
            result = ConceptProcessor._apply_main_mapping(df.copy(), config)
            
            # Verify the mock was called correctly
            mock_load.assert_called_once()
            
            # Check mapping results
            self.assertIn(SUBJECT_ID, result.columns)
            self.assertNotIn('dw_ek_kontakt', result.columns)
            self.assertEqual(len(result), 3)
        
        # Test with different column names
        df_diff = pd.DataFrame({
            'kontakt_id': ['K001', 'K002', 'K003'],
            'code': ['D123', 'D456', 'D789']
        })
        
        config_diff = {
            'main_mapping': {
                'left_on': 'kontakt_id',
                'right_on': 'dw_ek_kontakt'
            }
        }
        
        with patch.object(
            ConceptProcessor, 
            '_load_mapping_file',
            return_value=mapping_df,
            autospec=True  # Ensures the mock respects the method's signature
        ) as mock_load:
            result_diff = ConceptProcessor._apply_main_mapping(df_diff.copy(), config_diff)
            
            # Verify the mock was called correctly
            mock_load.assert_called_once()
            
            # Check mapping results
            self.assertIn(SUBJECT_ID, result_diff.columns)
            self.assertNotIn('kontakt_id', result_diff.columns)
            self.assertEqual(len(result_diff), 3)

    def test_map_and_clean_data(self):
        # Test data
        df = pd.DataFrame({
            SUBJECT_ID: ['P001', 'P002', 'P003', 'P004', 'P005'],
            CODE: ['C1', 'C2', None, 'C4', 'C5'],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', None, '2023-01-05'],
            'other_col': [1, 2, 3, 4, 5]
        })
        
        # Subject ID mapping
        subject_mapping = {
            'P001': 1,
            'P002': 2,
            'P003': 3,
            'P004': 4,
            'P005': 5,
            'P006': 6
        }
        
        result = ConceptProcessor._map_and_clean_data(df.copy(), subject_mapping)
        
        # Check that subject_id was mapped to integers
        self.assertEqual(result.loc[0, SUBJECT_ID], 1)
        self.assertEqual(result.loc[1, SUBJECT_ID], 2)
        
        # Check that rows with missing mandatory columns were dropped
        # Row 2 (index 2) has missing CODE
        # Row 3 (index 3) has missing timestamp
        # Only 3 rows should remain
        self.assertEqual(len(result), 3)
        
        # Test with missing subject_id column
        df_no_subject = pd.DataFrame({
            'other_id': ['P001', 'P002'],
            CODE: ['C1', 'C2'],
            'timestamp': ['2023-01-01', '2023-01-02']
        })
        
        result_no_subject = ConceptProcessor._map_and_clean_data(df_no_subject.copy(), subject_mapping)
        
        # No mapping should occur, but no error should be raised
        self.assertNotIn(SUBJECT_ID, result_no_subject.columns)
        
        # Test with duplicate rows
        df_dupes = pd.DataFrame({
            SUBJECT_ID: ['P001', 'P001', 'P002'],
            CODE: ['C1', 'C1', 'C2'],
            'timestamp': ['2023-01-01', '2023-01-01', '2023-01-02']
        })
        
        result_dupes = ConceptProcessor._map_and_clean_data(df_dupes.copy(), subject_mapping)
        
        # Duplicates should be removed
        self.assertEqual(len(result_dupes), 2)
        
        # Test with unknown subject IDs
        df_unknown = pd.DataFrame({
            SUBJECT_ID: ['P001', 'P999'],
            CODE: ['C1', 'C2'],
            'timestamp': ['2023-01-01', '2023-01-02']
        })
        
        result_unknown = ConceptProcessor._map_and_clean_data(df_unknown.copy(), subject_mapping)
        
        # Row with unknown subject ID will have NaN after mapping
        # This should result in the row being dropped
        self.assertEqual(len(result_unknown), 1)
        self.assertEqual(result_unknown.loc[0, SUBJECT_ID], 1)


if __name__ == "__main__":
    unittest.main()

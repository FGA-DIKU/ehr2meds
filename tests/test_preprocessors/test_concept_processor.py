import unittest
import pandas as pd
from datetime import timedelta
from MEDS_preprocess.preprocessors.constants import (
    ADMISSION, CODE, DISCHARGE, FILENAME,
    MANDATORY_COLUMNS, SUBJECT_ID, TIMESTAMP
)
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


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import (
    apply_mapping,
    check_columns,
    clean_data,
    convert_numeric_columns,
    factorize_subject_id,
    fill_missing_values,
    map_pids_to_ints,
    prefix_codes,
    select_and_rename_columns,
    unroll_columns,
)


class TestSelectAndRenameColumns(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"patient_id": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]}
        )

    def test_simple(self):
        # Create a DataFrame with three columns; only two are defined in the mapping.
        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [True, False, True]}
        )
        columns_map = {"A": "alpha", "B": "beta"}
        result = select_and_rename_columns(df, columns_map)
        self.assertListEqual(list(result.columns), ["alpha", "beta"])
        # Column "C" should be dropped.
        self.assertNotIn("C", result.columns)


class TestPrefixCodes(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"code": ["123", "456", "789"]})

    def test_prefix_codes(self):
        result = prefix_codes(self.df, "PRE_")
        expected = ["PRE_123", "PRE_456", "PRE_789"]
        self.assertEqual(result["code"].tolist(), expected)


class TestFillMissingValues(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"target": [None, "val2", None], "fillcol": ["(D123)", "(D456)", "(D789)"]}
        )

    def test_fill_missing_values_with_regex(self):
        # Create a DataFrame with a target column that has missing values.
        fillna_cfg = {"target": {"column": "fillcol", "regex": r"\((D\d+)\)"}}
        result = fill_missing_values(self.df, fillna_cfg)
        # Expected: first row filled with "D123", second row unchanged, third row with "D789".
        expected = ["D123", "val2", "D789"]
        self.assertEqual(result["target"].tolist(), expected)
        # The filler column should be dropped.
        self.assertNotIn("fillcol", result.columns)


class TestCheckColumns(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"patient_id": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]}
        )

    def test_raise_error(self):
        columns_map = {"A": "alpha", "B": "beta"}
        with self.assertRaises(ValueError):
            check_columns(self.df, columns_map)

    def test_no_error(self):
        columns_map = {"patient_id": "id", "value": "val"}
        check_columns(self.df, columns_map)


class TestFactorizeSubjectId(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "subject_id": ["A", "B", "A", "C", "D"],
                "value": ["a", "b", "c", "d", "e"],
            }
        )

    def test_factorize_subject_id(self):
        result, mapping = factorize_subject_id(self.df)
        expected = pd.DataFrame(
            {"subject_id": [1, 2, 1, 3, 4], "value": ["a", "b", "c", "d", "e"]}
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
        self.assertEqual(mapping, {"A": 1, "B": 2, "C": 3, "D": 4})


class TestApplyMapping(unittest.TestCase):
    def setUp(self):
        # Create a base dataframe for tests
        self.df = pd.DataFrame(
            {"patient_id": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]}
        )
        # Create a mapping table
        self.map_table = pd.DataFrame(
            {"old_id": [1, 2, 3, 5], "new_id": ["A", "B", "C", "E"]}
        )

    def test_inner_join_mapping(self):
        # With an inner join, rows without a matching mapping should be dropped.
        result = apply_mapping(
            self.df,
            self.map_table,
            join_col="old_id",
            source_col="patient_id",
            target_col="new_id",
            rename_to=None,
            how="inner",
            drop_source=False,
        )
        expected = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "value": ["a", "b", "c"],
                "new_id": ["A", "B", "C"],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_left_join_mapping(self):
        # With a left join, all rows in df should be preserved and unmatched mapping entries should be NaN.
        result = apply_mapping(
            self.df,
            self.map_table,
            join_col="old_id",
            source_col="patient_id",
            target_col="new_id",
            rename_to=None,
            how="left",
            drop_source=False,
        )
        expected = pd.DataFrame(
            {
                "patient_id": [1, 2, 3, 4],
                "value": ["a", "b", "c", "d"],
                "new_id": ["A", "B", "C", pd.NA],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_drop_source_column(self):
        # When drop_source is True, the original source column should be removed.
        result = apply_mapping(
            self.df,
            self.map_table,
            join_col="old_id",
            source_col="patient_id",
            target_col="new_id",
            rename_to=None,
            how="inner",
            drop_source=True,
        )
        expected = pd.DataFrame({"value": ["a", "b", "c"], "new_id": ["A", "B", "C"]})
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_rename_column(self):
        # Test that the target column is correctly renamed.
        result = apply_mapping(
            self.df,
            self.map_table,
            join_col="old_id",
            source_col="patient_id",
            target_col="new_id",
            rename_to="mapped_id",
            how="inner",
            drop_source=False,
        )
        expected = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "value": ["a", "b", "c"],
                "mapped_id": ["A", "B", "C"],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_join_col_same_as_source_col(self):
        # When join_col and source_col are identical, the source column should never be dropped,
        # even if drop_source is True.
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        map_table = pd.DataFrame({"id": [1, 2, 3], "new_val": ["A", "B", "C"]})
        result = apply_mapping(
            df,
            map_table,
            join_col="id",
            source_col="id",
            target_col="new_val",
            rename_to="mapped",
            how="inner",
            drop_source=True,
        )
        expected = pd.DataFrame({"value": ["a", "b", "c"], "mapped": ["A", "B", "C"]})
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_multiple_matches(self):
        # Test the behavior when there are duplicate join keys in the mapping table.
        df = pd.DataFrame({"patient_id": [1, 1], "value": ["a", "b"]})
        map_table = pd.DataFrame({"old_id": [1, 1], "new_id": ["A", "AA"]})
        result = apply_mapping(
            df,
            map_table,
            join_col="old_id",
            source_col="patient_id",
            target_col="new_id",
            rename_to=None,
            how="inner",
            drop_source=False,
        )
        # Expect a Cartesian product of the matches.
        expected = pd.DataFrame(
            {
                "patient_id": [1, 1, 1, 1],
                "value": ["a", "a", "b", "b"],
                "new_id": ["A", "AA", "A", "AA"],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_no_mapping_matches(self):
        # Test the case where no rows in df match the mapping table in an inner join.
        df = pd.DataFrame({"patient_id": [10, 20], "value": ["a", "b"]})
        result = apply_mapping(
            df,
            self.map_table,
            join_col="old_id",
            source_col="patient_id",
            target_col="new_id",
            rename_to=None,
            how="inner",
            drop_source=False,
        )
        expected = pd.DataFrame(
            {
                "patient_id": pd.Series([], dtype="int64"),
                "value": pd.Series([], dtype="object"),
                "new_id": pd.Series([], dtype="object"),
            }
        )

        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_empty_df(self):
        # Test applying the mapping on an empty dataframe.
        empty_df = pd.DataFrame(columns=["patient_id", "value"])
        result = apply_mapping(
            empty_df,
            self.map_table,
            join_col="old_id",
            source_col="patient_id",
            target_col="new_id",
            rename_to=None,
            how="inner",
            drop_source=False,
        )
        expected = pd.DataFrame(columns=["patient_id", "value", "new_id"])
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_map_table(self):
        # Test the scenario where the mapping table is empty.
        empty_map = pd.DataFrame(columns=["old_id", "new_id"])
        result = apply_mapping(
            self.df,
            empty_map,
            join_col="old_id",
            source_col="patient_id",
            target_col="new_id",
            rename_to=None,
            how="left",
            drop_source=False,
        )
        # For a left join, the new_id column should be NaN for all rows.
        expected = self.df.copy()
        expected["new_id"] = pd.NA
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


class TestConvertNumericColumns(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3, 4],
                "non_numeric_col": ["a", "b", "c", "d"],
                "mixed_col": ["1", "2.5", "3", "4.0"],
            }
        )

    def test_basic_conversion(self):
        result = convert_numeric_columns(self.df, {"numeric_columns": ["numeric_col"]})
        expected = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3, 4],
                "non_numeric_col": ["a", "b", "c", "d"],
                "mixed_col": ["1", "2.5", "3", "4.0"],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_mixed_numeric_conversion(self):
        """Test converting mixed integers and floats."""
        config = {"numeric_columns": ["mixed_col"]}
        result = convert_numeric_columns(self.df.copy(), config)
        self.assertEqual(result["mixed_col"].dtype, np.float64)
        np.testing.assert_array_almost_equal(
            result["mixed_col"], np.array([1.0, 2.5, 3.0, 4.0])
        )


class TestMapPidsToInts(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"subject_id": ["A", "B", "A", "C", "B"], "other_col": [1, 2, 3, 4, 5]}
        )

    def test_map_pids_to_ints(self):
        mapping = {"A": 1, "B": 2, "C": 3, "D": 4}
        result = map_pids_to_ints(self.df, mapping)
        expected = pd.DataFrame(
            {"subject_id": [1, 2, 1, 3, 2], "other_col": [1, 2, 3, 4, 5]}
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


class TestCleanData(unittest.TestCase):
    def test_dropna_when_mandatory_present(self):
        """Test that rows missing values in mandatory columns (defined by MANDATORY_COLUMNS) are dropped."""
        # Example DataFrame based on MANDATORY_COLUMNS ['A', 'B']
        df = pd.DataFrame(
            {
                "subject_id": [1, 2, None, 4],
                "code": [None, 2, 3, 4],
                "timestamp": [10, 20, 30, 40],
            }
        )
        # Expected: only rows where both 'A' and 'B' are non-null (rows with index 1 and 3)
        expected = pd.DataFrame(
            {"subject_id": [2.0, 4.0], "code": [2.0, 4.0], "timestamp": [20, 40]}
        )
        result = clean_data(df)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_no_dropna_when_mandatory_missing(self):
        """Test that dropna is not applied when not all mandatory columns are present."""
        # Example DataFrame missing the 'B' column.
        df = pd.DataFrame({"subject_id": [1, None, 3], "code": [10, 20, 30]})
        # Expected: no rows dropped based on NaN in 'B', only duplicates (if any) are removed.
        expected = df.drop_duplicates()
        result = clean_data(df)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_remove_duplicates(self):
        """Test that duplicate rows are removed."""
        # Example DataFrame using MANDATORY_COLUMNS with duplicate rows.
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2],
                "code": [3, 3, 4, 4],
                "timestamp": [10, 10, 20, 20],
            }
        )
        # Expected: one instance per duplicate row.
        expected = pd.DataFrame(
            {"subject_id": [1, 2], "code": [3, 4], "timestamp": [10, 20]}
        )
        result = clean_data(df)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_combined_effects(self):
        """Test combined effect of dropping NaNs in mandatory columns and duplicate removal."""
        # Example DataFrame with some missing mandatory values and duplicate rows.
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, None, 2, 2, 2],
                "code": [5, 5, 6, None, 7, 7],
                "timestamp": [100, 100, 200, 300, 300, 300],
            }
        )
        # Expected: rows missing any mandatory value are removed and duplicates eliminated.
        # Only one row with (1,5) and one row with (2,7) should remain.
        expected = pd.DataFrame(
            {"subject_id": [1.0, 2.0], "code": [5.0, 7.0], "timestamp": [100, 300]}
        )
        result = clean_data(df)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )


class TestUnrollColumns(unittest.TestCase):
    def test_single_unroll_with_prefix(self):
        """Test a single unroll column with a prefix and both required columns present."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 2],
                "timestamp": ["2021-01-01", "2021-01-02"],
                "col1": ["A", "B"],
            }
        )
        concept_config = {"unroll_columns": [{"column": "col1", "prefix": "P_"}]}
        result = unroll_columns(df, concept_config)
        self.assertEqual(len(result), 1)
        unrolled_df = result[0]
        expected_columns = ["subject_id", "timestamp", "code"]
        self.assertEqual(list(unrolled_df.columns), expected_columns)
        self.assertListEqual(list(unrolled_df["code"]), ["P_A", "P_B"])

    def test_single_unroll_without_prefix(self):
        """Test a single unroll column without a prefix."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 2],
                "timestamp": ["2021-01-01", "2021-01-02"],
                "col1": ["A", "B"],
            }
        )
        concept_config = {"unroll_columns": [{"column": "col1"}]}  # No prefix specified
        result = unroll_columns(df, concept_config)
        self.assertEqual(len(result), 1)
        unrolled_df = result[0]
        expected_columns = ["subject_id", "timestamp", "code"]
        self.assertEqual(list(unrolled_df.columns), expected_columns)
        self.assertListEqual(list(unrolled_df["code"]), ["A", "B"])

    def test_unroll_without_timestamp(self):
        """Test that when the input DataFrame lacks a timestamp, only the subject_id and code columns are returned."""
        df = pd.DataFrame({"subject_id": [1, 2], "col1": ["X", "Y"]})
        concept_config = {"unroll_columns": [{"column": "col1", "prefix": "Z_"}]}
        result = unroll_columns(df, concept_config)
        self.assertEqual(len(result), 1)
        unrolled_df = result[0]
        expected_columns = ["subject_id", "code"]
        self.assertEqual(list(unrolled_df.columns), expected_columns)
        self.assertListEqual(list(unrolled_df["code"]), ["Z_X", "Z_Y"])

    def test_multiple_unroll_columns(self):
        """Test unrolling multiple columns simultaneously."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "timestamp": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "col1": ["A", "B", "C"],
                "col2": ["D", "E", "F"],
            }
        )
        concept_config = {
            "unroll_columns": [
                {"column": "col1", "prefix": "P1_"},
                {"column": "col2"},  # No prefix provided
            ]
        }
        result = unroll_columns(df, concept_config)
        self.assertEqual(len(result), 2)
        # Validate first unrolled DataFrame (col1)
        unrolled_df1 = result[0]
        expected_columns = ["subject_id", "timestamp", "code"]
        self.assertEqual(list(unrolled_df1.columns), expected_columns)
        self.assertListEqual(list(unrolled_df1["code"]), ["P1_A", "P1_B", "P1_C"])
        # Validate second unrolled DataFrame (col2)
        unrolled_df2 = result[1]
        self.assertEqual(list(unrolled_df2.columns), expected_columns)
        self.assertListEqual(list(unrolled_df2["code"]), ["D", "E", "F"])

    def test_unroll_column_not_in_df(self):
        """Test that if the specified column does not exist in the DataFrame, it is ignored."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 2],
                "timestamp": ["2021-01-01", "2021-01-02"],
                "col1": ["A", "B"],
            }
        )
        concept_config = {"unroll_columns": [{"column": "nonexistent", "prefix": "P_"}]}
        result = unroll_columns(df, concept_config)
        self.assertEqual(len(result), 0)

    def test_empty_unroll_columns(self):
        """Test that an empty unroll_columns list returns an empty list."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 2],
                "timestamp": ["2021-01-01", "2021-01-02"],
                "col1": ["A", "B"],
            }
        )
        concept_config = {"unroll_columns": []}
        result = unroll_columns(df, concept_config)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()

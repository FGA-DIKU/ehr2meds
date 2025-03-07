import unittest

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import apply_mapping


class TestApplyMapping(unittest.TestCase):
    def setUp(self):
        # Create a base dataframe for tests
        self.df = pd.DataFrame({
            'patient_id': [1, 2, 3, 4],
            'value': ['a', 'b', 'c', 'd']
        })
        # Create a mapping table
        self.map_table = pd.DataFrame({
            'old_id': [1, 2, 3, 5],
            'new_id': ['A', 'B', 'C', 'E']
        })

    def test_inner_join_mapping(self):
        # With an inner join, rows without a matching mapping should be dropped.
        result = apply_mapping(
            self.df, self.map_table,
            join_col='old_id', source_col='patient_id',
            target_col='new_id', rename_to=None, how='inner', drop_source=False
        )
        expected = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'value': ['a', 'b', 'c'],
            'new_id': ['A', 'B', 'C']
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_left_join_mapping(self):
        # With a left join, all rows in df should be preserved and unmatched mapping entries should be NaN.
        result = apply_mapping(
            self.df, self.map_table,
            join_col='old_id', source_col='patient_id',
            target_col='new_id', rename_to=None, how='left', drop_source=False
        )
        expected = pd.DataFrame({
            'patient_id': [1, 2, 3, 4],
            'value': ['a', 'b', 'c', 'd'],
            'new_id': ['A', 'B', 'C', pd.NA]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_drop_source_column(self):
        # When drop_source is True, the original source column should be removed.
        result = apply_mapping(
            self.df, self.map_table,
            join_col='old_id', source_col='patient_id',
            target_col='new_id', rename_to=None, how='inner', drop_source=True
        )
        expected = pd.DataFrame({
            'value': ['a', 'b', 'c'],
            'new_id': ['A', 'B', 'C']
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_rename_column(self):
        # Test that the target column is correctly renamed.
        result = apply_mapping(
            self.df, self.map_table,
            join_col='old_id', source_col='patient_id',
            target_col='new_id', rename_to='mapped_id', how='inner', drop_source=False
        )
        expected = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'value': ['a', 'b', 'c'],
            'mapped_id': ['A', 'B', 'C']
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_join_col_same_as_source_col(self):
        # When join_col and source_col are identical, the source column should never be dropped,
        # even if drop_source is True.
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': ['a', 'b', 'c']
        })
        map_table = pd.DataFrame({
            'id': [1, 2, 3],
            'new_val': ['A', 'B', 'C']
        })
        result = apply_mapping(
            df, map_table,
            join_col='id', source_col='id',
            target_col='new_val', rename_to='mapped', how='inner', drop_source=True
        )
        expected = pd.DataFrame({
            'value': ['a', 'b', 'c'],
            'mapped': ['A', 'B', 'C']
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_multiple_matches(self):
        # Test the behavior when there are duplicate join keys in the mapping table.
        df = pd.DataFrame({
            'patient_id': [1, 1],
            'value': ['a', 'b']
        })
        map_table = pd.DataFrame({
            'old_id': [1, 1],
            'new_id': ['A', 'AA']
        })
        result = apply_mapping(
            df, map_table,
            join_col='old_id', source_col='patient_id',
            target_col='new_id', rename_to=None, how='inner', drop_source=False
        )
        # Expect a Cartesian product of the matches.
        expected = pd.DataFrame({
            'patient_id': [1, 1, 1, 1],
            'value': ['a', 'a', 'b', 'b'],
            'new_id': ['A', 'AA', 'A', 'AA']
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_no_mapping_matches(self):
        # Test the case where no rows in df match the mapping table in an inner join.
        df = pd.DataFrame({
            'patient_id': [10, 20],
            'value': ['a', 'b']
        })
        result = apply_mapping(
            df, self.map_table,
            join_col='old_id', source_col='patient_id',
            target_col='new_id', rename_to=None, how='inner', drop_source=False
        )
        expected = pd.DataFrame({
            'patient_id': pd.Series([], dtype='int64'),
            'value': pd.Series([], dtype='object'),
            'new_id': pd.Series([], dtype='object')
        })
        
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_empty_df(self):
        # Test applying the mapping on an empty dataframe.
        empty_df = pd.DataFrame(columns=['patient_id', 'value'])
        result = apply_mapping(
            empty_df, self.map_table,
            join_col='old_id', source_col='patient_id',
            target_col='new_id', rename_to=None, how='inner', drop_source=False
        )
        expected = pd.DataFrame(columns=['patient_id', 'value', 'new_id'])
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_map_table(self):
        # Test the scenario where the mapping table is empty.
        empty_map = pd.DataFrame(columns=['old_id', 'new_id'])
        result = apply_mapping(
            self.df, empty_map,
            join_col='old_id', source_col='patient_id',
            target_col='new_id', rename_to=None, how='left', drop_source=False
        )
        # For a left join, the new_id column should be NaN for all rows.
        expected = self.df.copy()
        expected['new_id'] = pd.NA
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

if __name__ == "__main__":
    unittest.main()
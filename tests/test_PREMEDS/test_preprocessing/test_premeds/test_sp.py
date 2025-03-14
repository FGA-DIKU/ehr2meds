import unittest

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import CODE, SUBJECT_ID, ADMISSION_IND
from ehr2meds.PREMEDS.preprocessing.premeds.sp import ConceptProcessor


class TestServiceProvider(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        # Initialize the ServiceProvider

        # Mock subject ID mapping
        self.subject_id_mapping = {"patient1": 1, "patient2": 2}

        # Sample admissions config
        self.admissions_config = {
            "rename_columns": {
                "original_col1": "type",
                "original_col2": "section",
                "original_col3": "timestamp_in",
                "original_col4": "timestamp_out",
                "original_col5": SUBJECT_ID,
            },
            "save_adm_move": True,
        }

        # Sample dataframe with two patients
        self.sample_df = pd.DataFrame(
            {
                "original_col1": [ADMISSION_IND, "flyt ind", ADMISSION_IND],
                "original_col2": ["dept1", "dept2", "dept3"],
                "original_col3": ["2023-01-01", "2023-01-02", "2023-01-01"],
                "original_col4": ["2023-01-02", "2023-01-03", "2023-01-02"],
                "original_col5": ["patient1", "patient1", "patient2"],
            }
        )

    def test_process_adt_admissions_basic(self):
        """Test basic processing of ADT admissions without chunk spanning."""
        # Process the sample dataframe
        result_df, _ = ConceptProcessor.process_adt_admissions(
            self.sample_df, self.admissions_config, self.subject_id_mapping
        )

        # Check that result is a DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)

        # We expect 7 events total:
        # Patient 1: ADMISSION_ADT + AFSNIT_ADT_dept1 + ADM_move + AFSNIT_ADT_dept2 + DISCHARGE_ADT
        # Patient 2: ADMISSION_ADT + AFSNIT_ADT_dept3
        self.assertEqual(result_df.shape[0], 7)

        # Check events for patient 1
        patient1_events = result_df[result_df[SUBJECT_ID] == 1]
        self.assertEqual(len(patient1_events), 5)

        # Check events for patient 2
        patient2_events = result_df[result_df[SUBJECT_ID] == 2]
        self.assertEqual(len(patient2_events), 2)

    def test_process_adt_admissions_with_last_patient(self):
        """Test processing with data from a previous chunk."""
        # Create last patient data
        last_patient_data = {
            SUBJECT_ID: 1,
            "admission_start": pd.Series(
                {
                    "type": ADMISSION_IND,
                    "section": "prev_dept",
                    "timestamp_in": "2023-01-01",
                    "timestamp_out": "2023-01-02",
                }
            ),
            "last_transfer": None,
            "events": [],
        }

        # Create a continuation dataframe
        cont_df = pd.DataFrame(
            {
                "original_col1": ["flyt ind"],
                "original_col2": ["next_dept"],
                "original_col3": ["2023-01-02"],
                "original_col4": ["2023-01-03"],
                "original_col5": ["patient1"],
            }
        )

        # Process with last patient data
        result_df, _ = ConceptProcessor.process_adt_admissions(
            cont_df, self.admissions_config, self.subject_id_mapping, last_patient_data
        )

        # Check that events were generated correctly
        self.assertEqual(result_df.shape[0], 2)  # ADM_move + AFSNIT_ADT/next_dept

        # Verify event types
        codes = list(result_df[CODE])
        self.assertEqual(codes[0], "MOVE_ADT")
        self.assertEqual(codes[1], "AFSNIT_ADT/next_dept")

    def test_process_adt_admissions_empty_df(self):
        """Test processing with an empty DataFrame."""
        empty_df = pd.DataFrame(columns=self.sample_df.columns)

        result_df, last_patient_info = ConceptProcessor.process_adt_admissions(
            empty_df, self.admissions_config, self.subject_id_mapping
        )

        # Check that result is an empty DataFrame
        self.assertTrue(result_df.empty)
        # Check that no last patient info is returned
        self.assertIsNone(last_patient_info)

    def test_process_adt_admissions_single_patient(self):
        """Test processing with a single patient's data."""
        # Create single patient dataframe
        single_patient_df = pd.DataFrame(
            {
                "original_col1": [ADMISSION_IND, "flyt ind"],
                "original_col2": ["dept1", "dept2"],
                "original_col3": ["2023-01-01", "2023-01-02"],
                "original_col4": ["2023-01-02", "2023-01-03"],
                "original_col5": ["patient1", "patient1"],
            }
        )

        result_df, _ = ConceptProcessor.process_adt_admissions(
            single_patient_df, self.admissions_config, self.subject_id_mapping
        )

        # Check number of events (ADMISSION + dept1 + ADM_move + dept2 + DISCHARGE)
        self.assertEqual(result_df.shape[0], 4)

        # Verify all events belong to the same patient
        self.assertTrue((result_df[SUBJECT_ID] == 1).all())

        # Check event sequence
        codes = list(result_df[CODE])
        self.assertEqual(codes[0], "ADMISSION_ADT")
        self.assertEqual(codes[1], "AFSNIT_ADT/dept1")
        self.assertEqual(codes[2], "MOVE_ADT")
        self.assertEqual(codes[3], "AFSNIT_ADT/dept2")

    def test_process_adt_admissions_discharge_handling(self):
        """Test that discharge information is correctly handled across chunks."""
        # First chunk: Patient admission and transfer
        first_chunk_df = pd.DataFrame(
            {
                "original_col1": [ADMISSION_IND, "flyt ind"],
                "original_col2": ["dept1", "dept2"],
                "original_col3": ["2023-01-01", "2023-01-02"],
                "original_col4": ["2023-01-02", "2023-01-03"],
                "original_col5": ["patient1", "patient1"],
            }
        )

        # Process first chunk
        result_df1, last_patient_info = ConceptProcessor.process_adt_admissions(
            first_chunk_df, self.admissions_config, self.subject_id_mapping
        )

        # Verify first chunk results
        self.assertEqual(result_df1.shape[0], 4)  # ADMISSION + dept1 + ADM_move + dept2
        self.assertIsNotNone(last_patient_info)
        self.assertEqual(last_patient_info[SUBJECT_ID], 1)

        # Second chunk: New patient starts, forcing discharge of previous patient
        second_chunk_df = pd.DataFrame(
            {
                "original_col1": [ADMISSION_IND],
                "original_col2": ["dept3"],
                "original_col3": ["2023-01-04"],
                "original_col4": ["2023-01-05"],
                "original_col5": ["patient2"],
            }
        )

        # Process second chunk with last_patient_info
        result_df2, _ = ConceptProcessor.process_adt_admissions(
            second_chunk_df,
            self.admissions_config,
            self.subject_id_mapping,
            last_patient_info,
        )

        # Verify second chunk results
        self.assertEqual(
            result_df2.shape[0], 3
        )  # DISCHARGE (from prev) + ADMISSION + dept3

        # Check specific events in second chunk
        codes = list(result_df2[CODE])
        self.assertEqual(
            codes[0], "DISCHARGE_ADT"
        )  # First event should be discharge of previous patient
        self.assertEqual(
            result_df2[SUBJECT_ID].iloc[0], 1
        )  # Discharge event should be for patient1
        self.assertEqual(codes[1], "ADMISSION_ADT")  # Then admission of new patient
        self.assertEqual(
            codes[2], f"AFSNIT_ADT/dept3"
        )  # Then department for new patient
        self.assertEqual(
            result_df2[SUBJECT_ID].iloc[1], 2
        )  # New events should be for patient2


if __name__ == "__main__":
    unittest.main()

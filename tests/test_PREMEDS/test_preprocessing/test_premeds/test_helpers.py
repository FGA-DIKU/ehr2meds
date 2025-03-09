import unittest

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import CODE, SUBJECT_ID
from ehr2meds.PREMEDS.preprocessing.premeds.helpers import (
    create_events_dataframe,
    finalize_previous_patient,
    handle_admission_event,
    handle_transfer_event,
    initialize_patient_state,
    prepare_last_patient_info,
    preprocess_admissions_df,
)


class TestAdtProcessingFunctions(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        # Mock subject ID mapping
        self.subject_id_mapping = {"patient1": 1, "patient2": 2, "patient3": 3}

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

        # Sample dataframe
        self.sample_df = pd.DataFrame(
            {
                "original_col1": ["indlaeggelse", "flyt ind"],
                "original_col2": ["dept1", "dept2"],
                "original_col3": ["2023-01-01", "2023-01-02"],
                "original_col4": ["2023-01-02", "2023-01-03"],
                "original_col5": ["patient1", "patient1"],
            }
        )

        # Sample patient state
        self.patient_state = {
            "current_patient_id": 1,
            "admission_start": pd.Series(
                {
                    "type": "indlaeggelse",
                    "section": "dept1",
                    "timestamp_in": "2023-01-01",
                    "timestamp_out": "2023-01-02",
                }
            ),
            "last_transfer": pd.Series(
                {
                    "type": "flyt ind",
                    "section": "dept2",
                    "timestamp_in": "2023-01-02",
                    "timestamp_out": "2023-01-03",
                }
            ),
        }

    def test_preprocess_admissions_df(self):
        """Test preprocessing of admissions dataframe."""
        # Test with valid input
        result = preprocess_admissions_df(
            self.sample_df, self.admissions_config, self.subject_id_mapping
        )

        # Check that columns were renamed correctly
        self.assertIn("type", result.columns)
        self.assertIn("section", result.columns)
        self.assertIn("timestamp_in", result.columns)
        self.assertIn(SUBJECT_ID, result.columns)

        # Check that subject_id was mapped correctly
        self.assertEqual(result[SUBJECT_ID].iloc[0], 1)

        # Check sorting
        self.assertTrue(result.equals(result.sort_values([SUBJECT_ID, "timestamp_in"])))

        # Test with missing SUBJECT_ID column: Expect KeyError since "subject_id" is required for sorting
        df_without_subject = self.sample_df.copy()
        df_without_subject.drop("original_col5", axis=1, inplace=True)
        config_without_subject = {
            "rename_columns": {
                "original_col1": "type",
                "original_col2": "section",
                "original_col3": "timestamp_in",
                "original_col4": "timestamp_out",
            }
        }
        with self.assertRaises(KeyError) as context:
            preprocess_admissions_df(
                df_without_subject, config_without_subject, self.subject_id_mapping
            )
        self.assertEqual(str(context.exception), "'subject_id'")

    def test_initialize_patient_state(self):
        """Test initialization of patient state."""
        # Test with last_patient_data provided
        last_patient = {
            "subject_id": 1,
            "admission_start": "test_admission",
            "last_transfer": "test_transfer",
        }
        state = initialize_patient_state(last_patient)
        self.assertEqual(state["current_patient_id"], 1)
        self.assertEqual(state["admission_start"], "test_admission")
        self.assertEqual(state["last_transfer"], "test_transfer")

        # Test with no last_patient_data
        state = initialize_patient_state(None)
        self.assertIsNone(state["current_patient_id"])
        self.assertIsNone(state["admission_start"])
        self.assertIsNone(state["last_transfer"])

    def test_finalize_previous_patient(self):
        """Test finalizing previous patient."""
        events = []

        # Test with complete admission data
        finalize_previous_patient(events, self.patient_state)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][CODE], "DISCHARGE_ADT")
        self.assertEqual(events[0][SUBJECT_ID], 1)
        self.assertEqual(events[0]["timestamp"], "2023-01-03")

        # Test patient state is reset
        self.assertIsNone(self.patient_state["admission_start"])
        self.assertIsNone(self.patient_state["last_transfer"])

        # Test with incomplete admission data
        incomplete_state = {
            "current_patient_id": 2,
            "admission_start": None,
            "last_transfer": None,
        }
        events = []
        finalize_previous_patient(events, incomplete_state)
        self.assertEqual(len(events), 0)  # No events should be added

    def test_handle_admission_event(self):
        """Test handling an admission event."""
        events = []
        subject_id = 1
        timestamp_in = "2023-01-01"
        dept = "dept1"
        row = pd.Series(
            {
                "type": "indlaeggelse",
                "section": dept,
                "timestamp_in": timestamp_in,
                "timestamp_out": "2023-01-02",
            }
        )

        # Test with no previous admission
        initial_state = {
            "current_patient_id": subject_id,
            "admission_start": None,
            "last_transfer": None,
        }
        handle_admission_event(
            subject_id, timestamp_in, dept, row, initial_state, events
        )

        # Should create 2 events (ADMISSION_ADT and department)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][CODE], "ADMISSION_ADT")
        self.assertEqual(events[1][CODE], f"AFSNIT_ADT_{dept}")

        # Test with existing admission
        events = []
        existing_state = self.patient_state.copy()
        new_dept = "dept3"
        new_row = pd.Series(
            {
                "type": "indlaeggelse",
                "section": new_dept,
                "timestamp_in": "2023-01-03",
                "timestamp_out": "2023-01-04",
            }
        )

        handle_admission_event(
            subject_id, "2023-01-03", new_dept, new_row, existing_state, events
        )

        # Should create 3 events (DISCHARGE_ADT, ADMISSION_ADT and department)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0][CODE], "DISCHARGE_ADT")
        self.assertEqual(events[1][CODE], "ADMISSION_ADT")
        self.assertEqual(events[2][CODE], f"AFSNIT_ADT_{new_dept}")

    def test_handle_transfer_event(self):
        """Test handling a transfer event."""
        events = []
        subject_id = 1
        timestamp_in = "2023-01-02"
        dept = "dept2"
        row = pd.Series(
            {
                "type": "flyt ind",
                "section": dept,
                "timestamp_in": timestamp_in,
                "timestamp_out": "2023-01-03",
            }
        )

        # Test with save_adm_move=True
        handle_transfer_event(
            subject_id,
            timestamp_in,
            dept,
            row,
            self.patient_state,
            events,
            self.admissions_config,
        )

        # Should create 2 events (ADM_move and department)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][CODE], "ADM_move")
        self.assertEqual(events[1][CODE], f"AFSNIT_ADT_{dept}")

        # Test with save_adm_move=False
        events = []
        config_no_save = self.admissions_config.copy()
        config_no_save["save_adm_move"] = False

        handle_transfer_event(
            subject_id,
            timestamp_in,
            dept,
            row,
            self.patient_state,
            events,
            config_no_save,
        )

        # Should create only 1 event (department only)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][CODE], f"AFSNIT_ADT_{dept}")

        # Verify last_transfer was updated
        self.assertEqual(self.patient_state["last_transfer"].equals(row), True)

    def test_create_events_dataframe(self):
        """Test creating events dataframe."""
        # Test with non-empty events list
        events = [
            {SUBJECT_ID: 1, "timestamp": "2023-01-02", CODE: "ADMISSION_ADT"},
            {SUBJECT_ID: 1, "timestamp": "2023-01-01", CODE: "DISCHARGE_ADT"},
            {SUBJECT_ID: 2, "timestamp": "2023-01-03", CODE: "ADMISSION_ADT"},
        ]

        result = create_events_dataframe(events)

        # Check result is a DataFrame with correct shape
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 3))

        # Check sorting
        self.assertEqual(result.iloc[0][SUBJECT_ID], 1)
        self.assertEqual(result.iloc[0]["timestamp"], "2023-01-01")

        # Test with empty events list
        empty_result = create_events_dataframe([])
        self.assertTrue(empty_result.empty)

    def test_prepare_last_patient_info(self):
        """Test preparing last patient info."""
        # Test with complete patient state
        result = prepare_last_patient_info(self.patient_state)
        self.assertIsNotNone(result)
        self.assertEqual(result["subject_id"], 1)
        self.assertEqual(
            result["admission_start"].equals(self.patient_state["admission_start"]),
            True,
        )
        self.assertEqual(
            result["last_transfer"].equals(self.patient_state["last_transfer"]), True
        )
        self.assertEqual(result["events"], [])

        # Test with incomplete patient state
        incomplete_state = {
            "current_patient_id": 1,
            "admission_start": None,
            "last_transfer": None,
        }
        result = prepare_last_patient_info(incomplete_state)
        self.assertIsNone(result)

        # Test with no patient ID
        no_patient_state = {
            "current_patient_id": None,
            "admission_start": self.patient_state["admission_start"],
            "last_transfer": self.patient_state["last_transfer"],
        }
        result = prepare_last_patient_info(no_patient_state)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

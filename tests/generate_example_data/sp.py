import pandas as pd
import numpy as np
import random
import os
from tests.generate_example_data.helpers import (
    generate_cpr_hash,
    generate_medical_code,
    generate_timestamps,
)

from tests.generate_example_data.constants import (
    DESCRIPTIONS,
    BODY_PARTS,
    MEDICATION_NAMES,
    MED_TYPES,
    MED_UNITS,
    MED_ADMINISTRATIONS,
    MED_INFUSION_SPEED,
    MED_INFUSION_DOSE,
    MED_ACTIONS,
    PROCEDURE_NAMES,
    LAB_TESTS,
    LAB_ANTIBIOTICS,
    LAB_SENSITIVITIES,
    LAB_ORGANISMS,
    LAB_RESULTS,
)
from ehr2meds.PREMEDS.preprocessing.constants import ADMISSION_IND


def generate_diagnosis(save_dir, hashes, birthdates, deathdates, seed=0):
    def generate_diagnosis_description(n=1000, diag_codes=None):
        diagnoses = []
        if diag_codes is not None:
            n = len(diag_codes)
        else:
            diag_codes = generate_medical_code(n, prefix="D")
        for i in range(n):
            phrase = f"{random.choice(DESCRIPTIONS)} of the {random.choice(BODY_PARTS)}"
            if random.random() > 0.3:  # 70% chance to include (Dxxx)
                diagnoses.append(f"{phrase} ({diag_codes[i]})")
            else:
                diagnoses.append(phrase)
        return diagnoses

    total_concepts = len(hashes)
    diag_codes = generate_medical_code(total_concepts, prefix="D")
    df = pd.DataFrame(
        {
            "CPR_hash": hashes,
            "Diagnosekode": diag_codes,
            "Diagnose": generate_diagnosis_description(total_concepts, diag_codes),
            "Noteret_dato": generate_timestamps(birthdates, deathdates, total_concepts),
            "Løst_dato": generate_timestamps(birthdates, deathdates, total_concepts),
        }
    )
    os.makedirs(save_dir, exist_ok=True)
    df.to_parquet(f"{save_dir}/CPMI_Diagnoser.parquet", index=False)


def generate_medication(save_dir, hashes, birthdates, deathdates, seed=0):
    def generate_medication_description(n=1000):
        dose = [random.randint(10, 1000) for _ in range(n)]
        unit = [random.choice(MED_UNITS) for _ in range(n)]
        generic_names = [random.choice(MEDICATION_NAMES) for _ in range(n)]
        med_type = [random.choice(MED_TYPES) for _ in range(n)]
        descs = [
            f"{name} {med_type} {dose} {unit}".upper()
            for name, med_type, dose, unit in zip(generic_names, med_type, dose, unit)
        ]
        return descs, generic_names, dose, unit

    total_concepts = len(hashes)
    description, generic_names, dose, unit = generate_medication_description(
        total_concepts
    )
    med_codes = generate_medical_code(
        total_concepts, start=1000, end=9999, mix_letters=True
    )
    df = pd.DataFrame(
        {
            "CPR_hash": hashes,
            "BestOrd_ID": np.random.randint(1e6, 1e7, total_concepts),
            "Ordineret_lægemiddel": description,
            "Generisk_navn": generic_names,
            "ATC": med_codes,
            "Bestillingsdato": generate_timestamps(
                birthdates, deathdates, total_concepts
            ),
            "Administrationstidspunkt": generate_timestamps(
                birthdates, deathdates, total_concepts
            ),
            "Administrationsdosis": dose,
            "Dosisenhed": unit,
            "Administrationsvej": [
                random.choice(MED_ADMINISTRATIONS) for _ in range(total_concepts)
            ],
            "Seponeringstidspunkt": generate_timestamps(
                birthdates, deathdates, total_concepts
            ),
            "Ordineret_dosis": [random.randint(1, 500) for _ in range(total_concepts)],
            "Infusionshastighed": [
                random.choice(MED_INFUSION_SPEED) for _ in range(total_concepts)
            ],
            "Infusionsdosis": [
                random.choice(MED_INFUSION_DOSE) for _ in range(total_concepts)
            ],
            "Handling": [random.choice(MED_ACTIONS) for _ in range(total_concepts)],
        }
    )
    df.to_parquet(f"{save_dir}/CPMI_Medicin.parquet", index=False)


def generate_procedures(save_dir, hashes, birthdates, deathdates, seed=0):
    total_concepts = len(hashes)
    dates = pd.Series(generate_timestamps(birthdates, deathdates, total_concepts))
    df = pd.DataFrame(
        {
            "CPR_hash": hashes,
            "ProcedureCode": generate_medical_code(
                len(hashes), start=100, end=999, mix_letters=True
            ),
            "ProcedureName": [
                random.choice(PROCEDURE_NAMES) for _ in range(total_concepts)
            ],
            "ServiceDate": dates.dt.date,
            "ServiceTime": dates.apply(
                lambda x: pd.Timestamp("1970-01-01")
                + pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
            ),
            "ServiceDatetime": dates,
        }
    )
    df.to_parquet(f"{save_dir}/CPMI_Procedurer.parquet", index=False)


def generate_labtests(save_dir, hashes, birthdates, deathdates, seed=0):
    total_concepts = len(hashes)
    dates = pd.Series(generate_timestamps(birthdates, deathdates, total_concepts))
    time_for_results = np.random.randint(0, 4, total_concepts)
    results_date = dates + pd.to_timedelta(time_for_results, unit="d")
    df = pd.DataFrame(
        {
            "CPR_hash": hashes,
            "BestOrd_ID": np.random.randint(1e6, 1e7, total_concepts),
            "BestOrd": [random.choice(LAB_TESTS) for _ in range(total_concepts)],
            "Bestillingsdato": dates.dt.date,
            "Prøvetagningstidspunkt": dates,
            "Resultatdato": results_date,
            "Resultatværdi": [
                random.choice(LAB_RESULTS) for _ in range(total_concepts)
            ],
            "Antibiotika": [
                random.choice(LAB_ANTIBIOTICS) for _ in range(total_concepts)
            ],
            "Følsomhed": [
                random.choice(LAB_SENSITIVITIES) for _ in range(total_concepts)
            ],
            "Organisme": [random.choice(LAB_ORGANISMS) for _ in range(total_concepts)],
        }
    )
    df.to_parquet(f"{save_dir}/CPMI_Labsvar.parquet", index=False)


def generate_adt_events(save_dir, hashes, birthdates, deathdates, seed=0):
    all_events = []

    # Convert inputs to pandas Series for easier filtering
    hash_series = pd.Series(hashes)
    birth_series = pd.Series(birthdates)
    death_series = pd.Series(deathdates)

    # For each patient, generate 1-3 admission episodes
    for patient_hash in set(hashes):
        n_admissions = random.randint(1, 3)

        # Find this patient's birth and death dates
        patient_indices = hash_series[hash_series == patient_hash].index
        patient_birth = birth_series.iloc[patient_indices[0]]
        patient_death = death_series.iloc[patient_indices[0]]

        for _ in range(n_admissions):
            # Start with admission event (Indlæggelse)
            admission_start = generate_timestamps(
                np.array([patient_birth]), np.array([patient_death]), 1
            )[0]
            initial_dept = f"Afdeling {random.choice(['A', 'B', 'C', 'D', 'E'])}{random.randint(1, 5)}"

            # Length of entire admission episode (in hours)
            total_stay_hours = random.randint(24, 240)  # 1-10 days

            # Initial admission event
            current_time = admission_start
            next_time = current_time + pd.Timedelta(hours=random.randint(2, 8))
            events = [
                {
                    "CPR_hash": patient_hash,
                    "Flyt_ind": current_time,
                    "Flyt_ud": next_time,
                    "Afsnit": initial_dept,
                    "ADT_haendelse": ADMISSION_IND,
                }
            ]

            # Generate chain of Flyt Ind events
            current_dept = initial_dept
            current_time = next_time

            # Number of transfers during this admission
            n_transfers = random.randint(2, 5)

            for _ in range(n_transfers):
                # Ensure we don't exceed total stay duration
                if (
                    current_time - admission_start
                ).total_seconds() / 3600 >= total_stay_hours:
                    break

                # Generate new department different from current
                new_dept = current_dept
                while new_dept == current_dept:
                    new_dept = f"Afdeling {random.choice(['A', 'B', 'C', 'D', 'E'])}{random.randint(1, 5)}"

                # Calculate next movement time
                next_time = min(
                    current_time + pd.Timedelta(hours=random.randint(4, 48)),
                    admission_start + pd.Timedelta(hours=total_stay_hours),
                )

                # Add Flyt Ind event
                events.append(
                    {
                        "CPR_hash": patient_hash,
                        "Flyt_ind": current_time,
                        "Flyt_ud": next_time,
                        "Afsnit": new_dept,
                        "ADT_haendelse": "Flyt Ind",
                    }
                )

                # Randomly insert "Tilbage fra orlov" events (10% chance)
                if random.random() < 0.1:
                    leave_return_time = current_time + pd.Timedelta(
                        hours=random.randint(1, 4)
                    )
                    if leave_return_time < next_time:
                        events.append(
                            {
                                "CPR_hash": patient_hash,
                                "Flyt_ind": leave_return_time,
                                "Flyt_ud": leave_return_time,  # Same timestamp for leave returns
                                "Afsnit": new_dept,
                                "ADT_haendelse": "Tilbage fra orlov",
                            }
                        )

                current_dept = new_dept
                current_time = next_time

            all_events.extend(events)

    # Convert to DataFrame and sort by patient and timestamp
    df = pd.DataFrame(all_events)
    df = df.sort_values(["CPR_hash", "Flyt_ind"])

    df.to_parquet(f"{save_dir}/CPMI_ADTHaendelser.parquet", index=False)


def generate_patients_info(n_patients):
    # Set a random seed for reproducibility (optional)
    np.random.seed(42)

    # Define the range of birthdates (e.g., between 1940 and 2020)
    start_birthdate = np.datetime64("1970-01-01")
    end_birthdate = np.datetime64("2020-01-01")

    # Generate random birthdates between start and end dates
    birthdates = np.random.choice(
        np.arange(start_birthdate, end_birthdate, dtype="datetime64[D]"), n_patients
    )

    # Generate deathdates where some people are still alive (i.e., deathdate is NaT)
    death_prob = np.random.rand(n_patients)

    # For those with death_prob > 0.8, generate a deathdate between their birthdate and a future date (e.g., 2025)
    deathdates = np.where(
        death_prob > 0.8,
        np.array(
            [
                np.random.choice(
                    np.arange(
                        birthdate + np.timedelta64(10),
                        np.datetime64("2024-01-01"),
                        dtype="datetime64[D]",
                    )
                )
                for birthdate in birthdates
            ]
        ),
        pd.NaT,
    )
    genders = np.random.choice(["Mand", "Kvinde"], size=n_patients)

    # Generate random PIDs
    hashes = generate_cpr_hash(n_patients)
    return pd.DataFrame(
        {
            "CPR_hash": hashes,
            "Fødselsdato": birthdates,
            "Dødsdato": deathdates,
            "Køn": genders,
        }
    )

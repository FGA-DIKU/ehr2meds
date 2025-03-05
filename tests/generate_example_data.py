import pandas as pd
import numpy as np
import random
import hashlib
import argparse
import os
from os.path import join
import string


from tests.constants import (
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
    FORL_TYPE,
    FORL_SPECIALTY,
    FORL_REGIONS,
    FORL_SHAK,
    MAPPING_DIR,
    SP_DIR,
    REGISTER_DIR,
)

DEFAULT_N_CONCEPTS = 5  # concepts per patient
DEFAULT_N = 100  # number of patients
DEFAULT_WRITE_DIR = "example_data"


def generate_cpr_hash(n):
    hashes = []
    for _ in range(n):
        random_str = str(random.randint(1000000000, 9999999999))
        hash_object = hashlib.sha256(random_str.encode())
        hashes.append(hash_object.hexdigest())
    return hashes


def generate_medical_code(n, start=100, end=999, mix_letters=False, prefix=None):
    if prefix is None:
        prefix = ""

    codes = []
    for _ in range(n):
        number_part = str(random.randint(start, end))
        letter_part = (
            "".join(random.choices(string.ascii_uppercase, k=2)) if mix_letters else ""
        )
        codes.append(f"{prefix}{letter_part}{number_part}")

    return codes


def generate_timestamps(birthdates, deathdates, n=1000):
    # Convert inputs to numpy arrays if they aren't already
    birthdates = np.array(birthdates)
    deathdates = np.array(deathdates)

    # Convert to timestamps if they aren't already
    if not isinstance(birthdates[0], (np.datetime64, pd.Timestamp)):
        birthdates = pd.to_datetime(birthdates)
    if not isinstance(deathdates[0], (np.datetime64, pd.Timestamp)):
        deathdates = pd.to_datetime(deathdates)

    # Convert to unix timestamps (nanoseconds since epoch)
    birth_timestamps = pd.Series(birthdates).apply(lambda x: x.timestamp())
    death_timestamps = pd.Series(deathdates).apply(
        lambda x: (
            x.timestamp() if pd.notna(x) else pd.Timestamp("2025-01-01").timestamp()
        )
    )

    random_timestamps = [
        np.random.randint(int(birth), int(death))
        for birth, death in zip(birth_timestamps, death_timestamps)
    ]
    timestamps = pd.to_datetime(random_timestamps, unit="s")
    return timestamps


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
    df.to_csv(f"{save_dir}/CPMI_Diagnoser.csv", index=False)


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
            "Resultatværdi": [random.choice(LAB_TESTS) for _ in range(total_concepts)],
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
                    "ADT_haendelse": "Indlæggelse",
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


def generate_register_diagnosis(save_dir, mapping, kont, seed=0, n_concepts=3):
    pids = kont["PID"].tolist()
    pids_lst = np.tile(pids, n_concepts)
    dw_eks_kontakt = np.tile(kont["dw_ek_kontakt"].tolist(), n_concepts)
    n_total = len(pids_lst)
    df = pd.DataFrame(
        {
            "dw_ek_kontakt": dw_eks_kontakt,
            "diagnosekode": generate_medical_code(n_total, prefix="D"),
            "diagnosetype": [
                random.choices(["A", "B", "+"], weights=[15, 80, 5])[0]
                for _ in range(n_total)
            ],
            "senere_afkraeftet": [
                random.choices(["Nej", "Ja"], weights=[99, 1])[0]
                for _ in range(n_total)
            ],
            "diagnosetype_parent": [
                random.choices(["", "A", "B"], weights=[80, 15, 5])[0]
                for _ in range(n_total)
            ],
            "lpindberetningssystem": ["LPR3" for _ in range(n_total)],
        }
    )
    df.to_csv(f"{save_dir}/diagnoser.asc", index=False)


def generate_forloeb(mapping_merged):
    filtered_pids = mapping_merged[
        (mapping_merged["forloeb"] == True) & (mapping_merged["CPR_hash"].isna())
    ]["PID"].values
    subset_size = int(0.1 * len(filtered_pids))
    random_subset = np.random.choice(filtered_pids, subset_size, replace=False)
    pids_to_nan = random_subset.tolist()
    forloeb_pids = mapping_merged[mapping_merged["forloeb"] == True][
        ["PID", "Fødselsdato", "Dødsdato"]
    ].copy()
    forloeb_pids["PID"] = forloeb_pids["PID"].apply(
        lambda x: x if x not in pids_to_nan else "nan"
    )  # there are weird nan strings in this column

    n_pids = len(forloeb_pids)
    dates_start = generate_timestamps(
        forloeb_pids["Fødselsdato"], forloeb_pids["Dødsdato"], n_pids
    )
    durations = np.random.randint(1, 365, n_pids)
    dates_end = dates_start + pd.to_timedelta(durations, unit="d")

    # Calculate ages at start and end
    birth_dates = pd.to_datetime(forloeb_pids["Fødselsdato"])
    ages_start = ((dates_start - birth_dates).dt.days / 365.25).astype(int)
    ages_days_start = (dates_start - birth_dates).dt.days
    ages_end = ((dates_end - birth_dates).dt.days / 365.25).astype(int)
    ages_days_end = (dates_end - birth_dates).dt.days

    # Generate gender information
    genders = np.random.choice(["M", "K"], n_pids)

    # Generate referral information
    referral_dates = dates_start - pd.to_timedelta(
        np.random.randint(1, 30, n_pids), unit="d"
    )

    forloeb = pd.DataFrame(
        {
            "dw_ek_forloeb": np.arange(1, n_pids + 1),
            "dw_ek_helbredsforloeb": np.arange(10001, 10001 + n_pids),
            "sorenhed_ans": np.arange(20001, 20001 + n_pids),
            "enhedstype_ans": np.random.choice(FORL_TYPE, n_pids),
            "hovedspeciale_ans": np.random.choice(FORL_SPECIALTY, n_pids),
            "region_ans": np.random.choice(FORL_REGIONS, n_pids),
            "shak_sgh_ans": np.arange(30001, 30001 + n_pids),
            "shak_afd_ans": np.arange(40001, 40001 + n_pids),
            "shak_afs_ans": np.random.choice(FORL_SHAK, n_pids),
            "dato_start": dates_start.date,
            "tidspunkt_start": dates_start.time,
            "dato_slut": dates_end.date,
            "tidspunkt_slut": dates_end.time,
            "alder_start": ages_start,
            "alder_dage_start": ages_days_start,
            "alder_slut": ages_end,
            "alder_dage_slut": ages_days_end,
            "koen": genders,
            "sorenhed_hen": np.arange(50001, 50001 + n_pids),
            "enhedstype_hen": np.random.choice(FORL_TYPE, n_pids),
            "hovedspeciale_hen": np.random.choice(FORL_SPECIALTY, n_pids),
            "region_hen": np.random.choice(FORL_REGIONS, n_pids),
            "shak_sgh_hen": np.arange(60001, 60001 + n_pids),
            "shak_afd_hen": np.arange(70001, 70001 + n_pids),
            "shak_afs_hen": np.random.choice(FORL_SHAK, n_pids),
            "dato_henvisning": referral_dates.date,
            "tidspunkt_henvisning": referral_dates.time,
            "henvisningsaarsag": generate_medical_code(n_pids, prefix="D"),
            "henvisningsmaade": np.random.choice(["A", "B", "C"], n_pids),
            "henvisning_fritvalg": np.random.choice(["Ja", "Nej"], n_pids),
            "sorenhed_ind": np.arange(80001, 80001 + n_pids),
            "enhedstype_ind": np.random.choice(FORL_TYPE, n_pids),
            "hovedspeciale_ind": np.random.choice(FORL_SPECIALTY, n_pids),
            "region_ind": np.random.choice(FORL_REGIONS, n_pids),
            "shak_sgh_ind": np.arange(90001, 90001 + n_pids),
            "shak_afd_ind": np.arange(100001, 100001 + n_pids),
            "shak_afs_ind": np.random.choice(FORL_SHAK, n_pids),
            "lprindberetningssystem": ["LPR3" for _ in range(n_pids)],
            "PID": forloeb_pids["PID"].tolist(),
        }
    )
    return forloeb


def generate_kontakter(mapping_merged, forloeb, n_visits=3):
    filtered_pids = mapping_merged[
        (mapping_merged["kontakter"] == True) & (mapping_merged["CPR_hash"].isna())
    ]["PID"].values
    subset_size = int(0.1 * len(filtered_pids))
    random_subset = np.random.choice(filtered_pids, subset_size, replace=False)
    pids_to_nan = random_subset.tolist()
    kont_pids = mapping_merged[mapping_merged["forloeb"] == True][
        ["PID", "Fødselsdato", "Dødsdato"]
    ].copy()
    kont_pids["PID"] = kont_pids["PID"].apply(
        lambda x: x if x not in pids_to_nan else "nan"
    )  # there are weird nan strings in this column

    n_total_visits = len(kont_pids) * n_visits

    # Expand to ensure multiple visits
    pids = kont_pids["PID"].tolist()
    pids_lst = np.tile(pids, n_visits)
    birthdates = np.tile(kont_pids["Fødselsdato"], n_visits)
    deathdates = np.tile(kont_pids["Dødsdato"], n_visits)
    dw_eks_forloeb = np.tile(forloeb["dw_ek_forloeb"].tolist(), n_visits)
    dates_start = generate_timestamps(birthdates, deathdates, n_total_visits)
    durations = np.random.randint(15, 480, n_total_visits)
    dates_end = dates_start + pd.to_timedelta(durations, unit="m")

    # Calculate ages at start and end - fix for TimedeltaIndex error
    birth_dates = pd.to_datetime(birthdates)
    # Convert to Series to use .dt accessor
    date_diff_start = pd.Series(dates_start - birth_dates)
    date_diff_end = pd.Series(dates_end - birth_dates)

    ages_start = (date_diff_start.dt.days / 365.25).astype(int)
    ages_days_start = date_diff_start.dt.days
    ages_end = (date_diff_end.dt.days / 365.25).astype(int)
    ages_days_end = date_diff_end.dt.days

    # Generate gender information
    genders = np.random.choice(["M", "K"], n_total_visits)

    # Generate referral information
    referral_dates = dates_start - pd.to_timedelta(
        np.random.randint(1, 30, n_total_visits), unit="d"
    )

    kontakter = pd.DataFrame(
        {
            "dw_ek_kontakt": np.arange(1, n_total_visits + 1),
            "dw_ek_forloeb": dw_eks_forloeb,
            "sorenhed_ans": np.arange(20001, 20001 + n_total_visits),
            "enhedstype_ans": np.random.choice(FORL_TYPE, n_total_visits),
            "hovedspeciale_ans": np.random.choice(FORL_SPECIALTY, n_total_visits),
            "region_ans": np.random.choice(FORL_REGIONS, n_total_visits),
            "shak_sgh_ans": np.arange(30001, 30001 + n_total_visits),
            "shak_afd_ans": np.arange(40001, 40001 + n_total_visits),
            "shak_afs_ans": np.random.choice(FORL_SHAK, n_total_visits),
            "dato_start": dates_start.date,
            "tidspunkt_start": dates_start.time,
            "dato_slut": dates_end.date,
            "tidspunkt_slut": dates_end.time,
            "alder_start": ages_start,
            "alder_dage_start": ages_days_start,
            "alder_slut": ages_end,
            "alder_dage_slut": ages_days_end,
            "koen": genders,
            "aktionsdiagnose": generate_medical_code(n_total_visits, prefix="D"),
            "kontaktaarsag": np.random.choice(["A", "B", "C"], n_total_visits),
            "sorenhed_hen": np.arange(50001, 50001 + n_total_visits),
            "enhedstype_hen": np.random.choice(FORL_TYPE, n_total_visits),
            "hovedspeciale_hen": np.random.choice(FORL_SPECIALTY, n_total_visits),
            "region_hen": np.random.choice(FORL_REGIONS, n_total_visits),
            "shak_sgh_hen": np.arange(60001, 60001 + n_total_visits),
            "shak_afd_hen": np.arange(70001, 70001 + n_total_visits),
            "shak_afs_hen": np.random.choice(FORL_SHAK, n_total_visits),
            "dato_henvisning": referral_dates.date,
            "tidspunkt_henvisning": referral_dates.time,
            "henvisningsaarsag": generate_medical_code(n_total_visits, prefix="D"),
            "henvisningsmaade": np.random.choice(["A", "B", "C"], n_total_visits),
            "henvisning_fritvalg": np.random.choice(["Ja", "Nej"], n_total_visits),
            "dato_behandling_start": dates_start.date,
            "tidspunkt_behandling_start": dates_start.time,
            "sorenhed_ind": np.arange(80001, 80001 + n_total_visits),
            "enhedstype_ind": np.random.choice(FORL_TYPE, n_total_visits),
            "hovedspeciale_ind": np.random.choice(FORL_SPECIALTY, n_total_visits),
            "region_ind": np.random.choice(FORL_REGIONS, n_total_visits),
            "shak_sgh_ind": np.arange(90001, 90001 + n_total_visits),
            "shak_afd_ind": np.arange(100001, 100001 + n_total_visits),
            "shak_afs_ind": np.random.choice(FORL_SHAK, n_total_visits),
            "lprindberetningssystem": ["LPR3" for _ in range(n_total_visits)],
            "PID": pids_lst,
        }
    )
    return kontakter


def generate_mapping(pids, patients_info):
    # Generate new unique PIDs different from CPR_hash values
    new_pids = generate_cpr_hash(len(pids))

    # Ensure at least some patients get data even with small counts
    n_register = max(len(new_pids) // 2, 1)  # At least 1 patient
    n_epikur = max(n_register // 2, 1)  # At least 1 patient
    n_forl = max((n_epikur * 4) // 5, 1)  # At least 1 patient

    pts_with_register_data = np.random.choice(new_pids, size=n_register, replace=False)
    pts_with_epikur = np.random.choice(
        pts_with_register_data, size=n_epikur, replace=False
    )
    pts_with_forl = np.random.choice(pts_with_register_data, size=n_forl, replace=False)

    # Create mapping between PIDs and CPR_hash
    pids_mapping = {pid: cpr_hash for pid, cpr_hash in zip(new_pids, pids)}
    mapping = pd.DataFrame({"PID": new_pids})
    mapping["CPR_hash"] = mapping["PID"].map(lambda x: pids_mapping.get(x, None))
    mapping["epikur"] = mapping["PID"].apply(lambda x: x in pts_with_epikur)
    mapping["kontakter"] = mapping["PID"].apply(lambda x: x in pts_with_forl)
    mapping["forloeb"] = mapping["PID"].apply(lambda x: x in pts_with_forl)

    for col in ["epikur", "kontakter", "forloeb"]:
        mask = (mapping[col] == False) & (mapping["CPR_hash"].isna())
        mapping.loc[mask, col] = np.random.choice([True, False], size=mask.sum())

    mapping["t_adm"] = [np.random.choice([True, False]) for _ in range(len(mapping))]
    mapping["t_tumor"] = [np.random.choice([True, False]) for _ in range(len(mapping))]

    # Add birthday to merged
    start_birthdate = np.datetime64("1940-01-01")
    end_birthdate = np.datetime64("2020-01-01")
    mapping_merged = pd.merge(mapping, patients_info, on="CPR_hash", how="left")
    mask = mapping_merged["Fødselsdato"].isna()
    mapping_merged.loc[mask, "Fødselsdato"] = np.random.choice(
        np.arange(start_birthdate, end_birthdate, dtype="datetime64[D]"),
        size=mask.sum(),
    )
    mapping_merged["Dødsdato"] = mapping_merged["Dødsdato"].fillna(
        np.datetime64("2025-01-01")
    )
    mapping_merged["Dødsdato"] = pd.to_datetime(mapping_merged["Dødsdato"])
    mapping_merged["Fødselsdato"] = pd.to_datetime(mapping_merged["Fødselsdato"])

    return mapping_merged, mapping


def generate_register_medication(save_dir, kont, n_concepts=3):
    # Get PIDs from kontakter
    pids = kont["PID"].unique()
    n_pids = len(pids)

    # Generate dates within a reasonable range
    dates = pd.date_range(
        start="2020-01-01", end="2023-12-31", periods=n_pids * n_concepts
    )
    # Convert to list before shuffling
    dates_list = dates.tolist()
    random.shuffle(dates_list)

    # Create dataframe
    df = pd.DataFrame(
        {
            "PID": np.repeat(pids, n_concepts),
            "eksd": dates_list,
            "aar": [d.year for d in dates_list],
            "ekst": np.random.choice(
                ["EI", "HA", "LS", "EM", "DD", "LI", "HF"], n_pids * n_concepts
            ),
            "atc": generate_medical_code(
                n_pids * n_concepts, start=1000, end=9999, mix_letters=True
            ),
            "vnr": np.random.randint(100000, 999999, n_pids * n_concepts),
            "apk": np.random.choice([1.0, 2.0, 3.0], n_pids * n_concepts),
            "packsize": np.random.choice(
                [30.000, 60.000, 100.000, 300.000], n_pids * n_concepts
            ),
            "volapk": np.random.choice(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], n_pids * n_concepts
            ),
            "indo": [np.nan for _ in range(n_pids * n_concepts)],
            "doso": [np.nan for _ in range(n_pids * n_concepts)],
        }
    )

    os.makedirs(save_dir, exist_ok=True)
    df.to_parquet(f"{save_dir}/epikur.parquet", index=False)


def generate_laegemidler(save_dir):
    # Generate a reasonable number of unique medications
    n_medications = 1000

    df = pd.DataFrame(
        {
            "ATC": generate_medical_code(
                n_medications, start=1000, end=9999, mix_letters=True
            ),
            "ATC1": [
                code[:1]
                for code in generate_medical_code(
                    n_medications, start=1000, end=9999, mix_letters=True
                )
            ],
            "VNR": np.random.randint(100000, 999999, n_medications),
            "PNAME": [
                f"{random.choice(MEDICATION_NAMES)} {random.choice(MED_TYPES)}".upper()
                for _ in range(n_medications)
            ],
            "PACKTEXT": [
                f"{random.randint(1, 100)} {random.choice(['tabletter', 'kapsler', 'ml', 'mg'])}".upper()
                for _ in range(n_medications)
            ],
            "VOLUME": [
                random.choice([1, 5, 10, 20, 30, 50, 100]) for _ in range(n_medications)
            ],
            "VOLTYPETXT": [
                random.choice(["TABLET", "KAPSEL", "MILLILITER", "MILLIGRAM"])
                for _ in range(n_medications)
            ],
            "VOLTYPECODE": [
                random.choice(["TAB", "KAP", "ML", "MG"]) for _ in range(n_medications)
            ],
            "STRENG": [
                f"{random.randint(1, 1000)} {random.choice(['mg', 'mcg', 'g', 'ml'])}".upper()
                for _ in range(n_medications)
            ],
            "STRUNIT": [
                random.choice(["MG", "MCG", "G", "ML"]) for _ in range(n_medications)
            ],
            "STRNUM": [random.randint(1, 1000) for _ in range(n_medications)],
            "DRUGID": [f"DRUG{i:05d}" for i in range(n_medications)],
        }
    )

    # Add some special rows for telephone prescriptions and adjustments
    special_rows = pd.DataFrame(
        {
            "ATC": ["", ""],
            "ATC1": ["", ""],
            "VNR": [100000, 100015],
            "PNAME": [
                "Telefonreceptgebyr",
                "Udligning af for meget eller for lidt udbetalt...",
            ],
            "PACKTEXT": ["", ""],
            "VOLUME": [".", "."],
            "VOLTYPETXT": ["", ""],
            "VOLTYPECODE": ["", ""],
            "STRENG": ["", ""],
            "STRUNIT": ["", ""],
            "STRNUM": ["", ""],
            "DRUGID": [".", "."],
        }
    )

    df = pd.concat([special_rows, df], ignore_index=True)

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/laegemiddeloplysninger.asc", index=False)


def generate_register_procedures(save_dir, kont, forloeb, n_concepts=3):
    # Get unique kontakt IDs and forloeb IDs
    dw_eks_kontakt = kont["dw_ek_kontakt"].tolist()
    dw_eks_forloeb = kont["dw_ek_forloeb"].tolist()

    # Make sure we have the same number of entries for each procedure
    n_total = len(dw_eks_kontakt) * n_concepts

    # Repeat the IDs to match the number of concepts
    dw_eks_kontakt = np.repeat(dw_eks_kontakt, n_concepts)
    dw_eks_forloeb = np.repeat(dw_eks_forloeb, n_concepts)

    # Generate dates
    dates_start = pd.to_datetime(np.random.choice(kont["dato_start"], n_total))
    durations = np.random.randint(15, 240, n_total)
    dates_end = dates_start + pd.to_timedelta(durations, unit="m")

    # Generate reporting dates (usually after procedure)
    report_delay = np.random.randint(1, 48, n_total)
    report_dates = dates_end + pd.to_timedelta(report_delay, unit="h")

    # Surgical procedures
    kirurgi_df = pd.DataFrame(
        {
            "dw_ek_forloeb": dw_eks_forloeb,
            "dw_ek_kontakt": dw_eks_kontakt,
            "dw_ek_procedureregistrering": np.random.randint(1000, 9999, n_total),
            "procedurekode": generate_medical_code(n_total, mix_letters=True),
            "proceduretype": np.random.choice(["P", "+"], n_total, p=[0.9, 0.1]),
            "procedurekode_parent": [
                (
                    ""
                    if random.random() > 0.2
                    else generate_medical_code(1, mix_letters=True)[0]
                )
                for _ in range(n_total)
            ],
            "proceduretype_parent": [
                "" if random.random() > 0.2 else "P" for _ in range(n_total)
            ],
            "sorenhed_pro": np.arange(10001, 10001 + n_total),
            "dato_start": dates_start.date,
            "tidspunkt_start": dates_start.time,
            "dato_slut": [
                date if random.random() > 0.3 else None for date in dates_end.date
            ],
            "tidspunkt_slut": [
                time if random.random() > 0.3 else None for time in dates_end.time
            ],
            "dato_indberetning": report_dates.date,
            "tidspunkt_indberetning": report_dates.time,
            "lprindberetningssystem": ["LPR3" for _ in range(n_total)],
            "procedureregistrering_id": [f"PROC{i:06d}" for i in range(n_total)],
        }
    )

    # Other procedures
    andre_df = pd.DataFrame(
        {
            "dw_ek_forloeb": dw_eks_forloeb,
            "dw_ek_kontakt": dw_eks_kontakt,
            "procedurekode": generate_medical_code(n_total, mix_letters=True),
            "proceduretype": np.random.choice(
                ["P", "+", "I"], n_total, p=[0.8, 0.1, 0.1]
            ),
            "procedurekode_parent": [
                (
                    ""
                    if random.random() > 0.2
                    else generate_medical_code(1, mix_letters=True)[0]
                )
                for _ in range(n_total)
            ],
            "sorenhed_pro": np.arange(20001, 20001 + n_total),
            "enhedstype_pro": np.random.choice(FORL_TYPE, n_total),
            "hovedspeciale_pro": np.random.choice(FORL_SPECIALTY, n_total),
            "region_pro": np.random.choice(FORL_REGIONS, n_total),
            "dato_start": dates_start.date,
            "tidspunkt_start": dates_start.time,
            "dato_slut": [
                date if random.random() > 0.3 else None for date in dates_end.date
            ],
            "tidspunkt_slut": [
                time if random.random() > 0.3 else None for time in dates_end.time
            ],
            "dato_indberetning": report_dates.date,
            "tidspunkt_indberetning": report_dates.time,
            "lprindberetningssystem": ["LPR3" for _ in range(n_total)],
            "procedureregistrering_id": [
                f"PROC{i+n_total:06d}" for i in range(n_total)
            ],
        }
    )

    os.makedirs(save_dir, exist_ok=True)
    kirurgi_df.to_csv(f"{save_dir}/procedurer_kirurgi.asc", index=False)
    andre_df.to_csv(f"{save_dir}/procedurer_andre.asc", index=False)


def main_write(
    n_patients=DEFAULT_N, n_concepts=DEFAULT_N_CONCEPTS, write_dir=DEFAULT_WRITE_DIR
):
    sp_dir = join(write_dir, SP_DIR)
    register_dir = join(write_dir, REGISTER_DIR)
    mapping_dir = join(write_dir, MAPPING_DIR)
    os.makedirs(sp_dir, exist_ok=True)
    os.makedirs(register_dir, exist_ok=True)
    os.makedirs(mapping_dir, exist_ok=True)
    np.random.seed(0)
    patients_info = generate_patients_info(n_patients)
    patients_info.to_parquet(f"{sp_dir}/CPMI_PatientInfo.parquet", index=False)

    # Getting lists for the CPR_hash, birthdate, and deathdate
    pids = patients_info["CPR_hash"]
    hashes = np.tile(pids, n_concepts)
    birthdates = np.tile(patients_info["Fødselsdato"], n_concepts)
    deathdates = np.tile(patients_info["Dødsdato"], n_concepts)
    birthdates = pd.to_datetime(birthdates)
    deathdates = pd.to_datetime(deathdates).fillna(
        pd.Timestamp(year=2025, month=1, day=1)
    )

    for birth, death in zip(birthdates, deathdates):
        assert birth < death, f"Birthdate {birth} is not before deathdate {death}"

    generate_diagnosis(sp_dir, hashes, birthdates, deathdates)
    generate_medication(sp_dir, hashes, birthdates, deathdates)
    generate_procedures(sp_dir, hashes, birthdates, deathdates)
    generate_labtests(sp_dir, hashes, birthdates, deathdates)
    generate_adt_events(sp_dir, hashes, birthdates, deathdates)
    mapping_merged, mapping = generate_mapping(pids, patients_info)
    forl = generate_forloeb(mapping_merged)
    kont = generate_kontakter(mapping_merged, forl, n_visits=3)

    # Save forloeb to parquet
    forl.to_parquet(f"{register_dir}/forloeb.parquet", index=False)

    # Generate and save register data
    generate_register_diagnosis(register_dir, mapping, kont, n_concepts=n_concepts)
    generate_register_medication(register_dir, kont, n_concepts=n_concepts)
    generate_laegemidler(register_dir)
    generate_register_procedures(register_dir, kont, forl, n_concepts=n_concepts)

    # Save kontakter to parquet
    kont.to_parquet(f"{register_dir}/kontakter.parquet", index=False)

    # Save mapping
    mapping.to_csv(f"{mapping_dir}/mapping.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate data for performance testing and profiling."
    )
    parser.add_argument(
        "n_patients",
        type=int,
        help=f"Number of patients to generate (default: {DEFAULT_N})",
        nargs="?",
        default=DEFAULT_N,
    )
    parser.add_argument(
        "n_concepts",
        type=int,
        help=f"Number of concepts to generate per patient (default: {DEFAULT_N_CONCEPTS})",
        nargs="?",
        default=DEFAULT_N_CONCEPTS,
    )
    parser.add_argument(
        "write_dir",
        type=str,
        help=f"Directory to write output files (default: {DEFAULT_WRITE_DIR})",
        nargs="?",
        default=DEFAULT_WRITE_DIR,
    )

    args = parser.parse_args()
    main_write(args.n_patients, args.n_concepts, args.write_dir)

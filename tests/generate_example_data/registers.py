import os
import random

import numpy as np
import pandas as pd

from tests.generate_example_data.constants import (
    FORL_REGIONS,
    FORL_SHAK,
    FORL_SPECIALTY,
    FORL_TYPE,
    MED_TYPES,
    MEDICATION_NAMES,
)
from tests.generate_example_data.helpers import (
    generate_cpr_hash,
    generate_medical_code,
    generate_timestamps,
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

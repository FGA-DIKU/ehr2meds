import argparse
import os
from os.path import join

import numpy as np
import pandas as pd

from tests.generate_example_data.constants import MAPPING_DIR, REGISTER_DIR, SP_DIR
from tests.generate_example_data.registers import (
    generate_forloeb,
    generate_kontakter,
    generate_mapping,
    generate_register_diagnosis,
    generate_register_medication,
    generate_register_procedures,
    generate_laegemidler,
)
from tests.generate_example_data.sp import (
    generate_adt_events,
    generate_diagnosis,
    generate_labtests,
    generate_medication,
    generate_patients_info,
    generate_procedures,
)

DEFAULT_N_CONCEPTS = 5  # concepts per patient
DEFAULT_N = 100  # number of patients
DEFAULT_WRITE_DIR = "example_data"


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

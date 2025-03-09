import hashlib
import random
import string

import numpy as np
import pandas as pd


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

import hashlib
import numpy as np
import random
import string
from datetime import date, datetime, time, timedelta


def medical_code(prefix="", min=100, max=999):
    letter = random.choice(string.ascii_uppercase)
    number = random.randint(min, max)
    return f"{prefix}{letter}{number}"


def person_id(min=1000000000, max=9999999999):
    random_str = str(random.randint(min, max))
    hash_object = hashlib.sha256(random_str.encode())
    return hash_object.hexdigest()


def rand_int(min_val=0, max_val=100):
    return random.randint(min_val, max_val)


def rand_float(min_val=0.0, max_val=100.0):
    return random.uniform(min_val, max_val)


def rand_date(start=1970, end=2020, format=None):
    """
    Random date generator with optional formatting.

    Examples:
    format=None -> date object
    format="%Y-%m-%d" -> "1999-02-01"
    """
    delta = date(end, 12, 31) - date(start, 1, 1)
    random_days = random.randint(0, delta.days)
    d = date(start, 1, 1) + timedelta(days=random_days)

    if format:
        return d.strftime(format).upper()

    return d


def rand_datetime(start=1970, end=2020):
    delta = datetime(end, 12, 31) - datetime(start, 1, 1)
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return datetime(start, 1, 1) + timedelta(seconds=random_seconds)


def rand_datetimeseconds(start=1970, end=2020):
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    delta = datetime(end, 12, 31) - datetime(start, 1, 1)
    random_seconds = random.randint(0, int(delta.total_seconds()))
    random_microseconds = random.randint(0, 999_999)
    dt = datetime(start, 1, 1) + timedelta(seconds=random_seconds, microseconds=random_microseconds)
    return dt.strftime(fmt)


def rand_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return time(hour, minute, second)


def rand_timemicroseconds():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    microsecond = random.randint(0, 999_999)
    return time(hour, minute, second, microsecond)


def rand_string(min_length=10, max_length=100, include_digits=True):
    if include_digits:
        return "".join(
            random.choices(
                string.ascii_letters + string.digits,
                k=random.randint(min_length, max_length),
            )
        )
    else:
        return "".join(random.choices(string.ascii_letters, k=random.randint(min_length, max_length)))


def choice(options):
    return random.choice(options)


def greater_than_date(min_date, end=2020):
    delta = date(end, 12, 31) - min_date
    random_days = random.randint(0, delta.days)
    return min_date + timedelta(days=random_days)


def greater_than_datetime(min_date, end=2020):
    delta = datetime(end, 12, 31) - min_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return min_date + timedelta(seconds=random_seconds)


def greater_than_datetimeseconds(min_date, end=2020):
    min_date = datetime.strptime(min_date, "%Y-%m-%d %H:%M:%S.%f")
    delta = datetime(end, 12, 31) - min_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return min_date + timedelta(seconds=random_seconds)


def date_between(start, end, start_offset_days=0, end_offset_days=0, format="%Y-%m-%d"):
    """Random date from start + start_offset_days to end + end_offset_days. Missing if start or end is."""
    if start is None or end is None:
        return None
    first = datetime.fromisoformat(str(start)) + timedelta(days=start_offset_days)
    last = datetime.fromisoformat(str(end)) + timedelta(days=end_offset_days)
    seconds = random.randint(0, max(int((last - first).total_seconds()), 0))
    return (first + timedelta(seconds=seconds)).strftime(format)


def date_offset(anchor, min_days=0, max_days=0, format="%Y-%m-%d"):
    """Random date min_days to max_days days after anchor (negative for before)."""
    return date_between(anchor, anchor, min_days, max_days, format)


def days_before(anchor, days, format="%Y-%m-%d"):
    """The date the given number of days before anchor."""
    return date_offset(anchor, -days, -days, format)


def age_at(birthdate, date):
    """Age in whole years on date. Missing if either date is."""
    if birthdate is None or date is None:
        return None
    birthdate = datetime.fromisoformat(str(birthdate))
    date = datetime.fromisoformat(str(date))
    age = date.year - birthdate.year
    if (date.month, date.day) < (birthdate.month, birthdate.day):
        age -= 1
    return age


# Specialized functions for SP
def gestationsalder_with_fraction(ga_days):
    """SP format of a GA in days, e.g. "38 2/7", or "38" on whole weeks."""
    weeks, days = divmod(int(ga_days), 7)
    if days == 0:
        return f"{weeks}"
    return f"{weeks} {days}/7"


# Specialized functions for DST dataset
def honuge(start_year, end_year):
    week = random.randint(1, 52)
    year = str(random.randint(start_year, end_year))[-2:]
    return f"{year}{week:02d}"


# Function to mix multiple functions
def mix_function(functions, probabilities):
    if not np.isclose(sum(probabilities), 1.0):
        raise ValueError(f"Probabilities must sum to 1.0, got {sum(probabilities)}")

    selected_idx = random.choices(range(len(functions)), weights=probabilities, k=1)[0]
    selected_func = functions[selected_idx]
    return selected_func["func"](**selected_func["args"])

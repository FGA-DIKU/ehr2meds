import random
import string
import hashlib
from datetime import date, datetime, timedelta, time


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


def rand_date(start=1970, end=2020):
    delta = date(end, 12, 31) - date(start, 1, 1)
    random_days = random.randint(0, delta.days)
    return date(start, 1, 1) + timedelta(days=random_days)


def rand_datetime(start=1970, end=2020):
    delta = datetime(end, 12, 31) - datetime(start, 1, 1)
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return datetime(start, 1, 1) + timedelta(seconds=random_seconds)


def rand_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return time(hour, minute, second)


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


# Specialized functions for DST dataset
def honuge(start_year, end_year):
    week = random.randint(1, 52)
    year = str(random.randint(start_year, end_year))[-2:]
    return f"{year}{week:02d}"

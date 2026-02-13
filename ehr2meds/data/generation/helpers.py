import random
import string
import hashlib
from datetime import date, datetime, timedelta, time
import inspect
import ehr2meds.data.generation.helpers as ghelpers


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

def rand_string(min_length=10, max_length=100, include_digits=True):
    if include_digits:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(min_length, max_length)))
    else:
        return ''.join(random.choices(string.ascii_letters, k=random.randint(min_length, max_length)))


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

# Function to mix multiple functions
def mix_function(functions, probabilities=None):
    gfunc_dict = {
        name: obj
        for name, obj in inspect.getmembers(ghelpers)
        if inspect.isfunction(obj) and name != "mix_function"
    }
    
    if not isinstance(functions, list) or len(functions) < 2:
        raise ValueError("'functions' must be a list with at least 2 functions")
    
    # Parse function configurations
    func_configs = []
    for func_cfg in functions:
        func_name = func_cfg.get("name")
        if func_name is None:
            raise ValueError("Each function configuration must have a 'name' key")
        
        func_args = func_cfg.get("args", {})
        
        if func_name not in gfunc_dict:
            raise ValueError(f"Function '{func_name}' not found in helpers")
        
        func_configs.append({
            "func": gfunc_dict[func_name],
            "args": func_args
        })
    
    # Handle probabilities
    if probabilities is None:
        # Equal probability for all functions
        probabilities = [1.0 / len(func_configs)] * len(func_configs)
    else:
        if len(probabilities) != len(func_configs):
            raise ValueError(f"Number of probabilities ({len(probabilities)}) must match number of functions ({len(func_configs)})")
                    
        # Enforce that probabilities sum to 1
        prob_sum = sum(probabilities)
        if abs(prob_sum - 1.0) > 1e-10:  # Allow small floating point errors
            raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")
    
    # Select function 
    selected_idx = random.choices(range(len(func_configs)), weights=probabilities, k=1)[0]
    return func_configs[selected_idx]["func"](**func_configs[selected_idx]["args"])        

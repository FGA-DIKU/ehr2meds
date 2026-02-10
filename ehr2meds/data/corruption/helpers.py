import random

def row_dropout(row, probability):
    return {col: (None if random.random() < probability else val) for col, val in row.items()}

def truncation(value, max_length, probability):
    assert isinstance(value, str), "Truncation can only be applied to string values."
    if random.random() < probability:
        return value[:max_length]
    return value
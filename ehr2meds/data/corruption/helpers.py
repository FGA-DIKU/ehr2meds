import random

def row_dropout(row, row_index, probability, forced_idx=[]):
    if row_index in forced_idx:
        return {col: None for col in row}

    return {col: (None if random.random() < probability else val) for col, val in row.items()}

def truncation(value, row_index, max_length, probability, forced_idx=[]):
    assert isinstance(value, str), "Truncation can only be applied to string values."
    if row_index in forced_idx:
        return value[:max_length]
    if random.random() < probability:
        return value[:max_length]
    return value
import polars as pl
from datetime import date, datetime
from dftly import Parser

CONTACT_TIME_EXPRESSION = "set_time($date_start, $time_start as '%H:%M:%S%.f')"


def test_contact_time_parses_mixed_string_precision():
    contacts = pl.DataFrame(
        {
            "date_start": [date(2019, 11, 27), date(2026, 3, 8), date(2026, 3, 6)],
            "time_start": ["08:43:00", "8:51:00.0000000", "20:14:00.0000000"],
        }
    )

    expression = Parser()(CONTACT_TIME_EXPRESSION).polars_expr
    timestamps = contacts.select(expression.alias("time"))["time"].to_list()

    assert timestamps == [
        datetime(2019, 11, 27, 8, 43),
        datetime(2026, 3, 8, 8, 51),
        datetime(2026, 3, 6, 20, 14),
    ]


def test_unparsed_string_time_reproduces_null_timestamp_bug():
    contact = pl.DataFrame(
        {
            "date_start": [date(2019, 11, 27)],
            "time_start": ["08:43:00"],
        }
    )

    old_expression = Parser()("set_time($date_start, $time_start)").polars_expr

    assert contact.select(old_expression.alias("time"))["time"].item() is None

"""Raw inputs that pin intended preMEDS behaviour on the edge cases a polars port could change.


Each value is chosen to hit one behaviour; the comment says which. Files are written at test
time so the inputs stay reviewable here instead of living as binary fixtures.
"""

import datetime as dt
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from pathlib import Path

# Subject ID mapping source: string IDs, a duplicate and a null.
POPULATION_CSV = """\
cpr,other
P01,a
P02,b
P03,c
P01,dup
,missing
P04,d
"""

# CSV read with engine="python": mixed-type columns, untrimmed strings, '.' placeholders.
# Duplicate pairs sit inside one 4-row chunk so the chunksize test isolates per-chunk dtype inference.
CONTACTS_CSV = """\
pid,contact_id,diag,priority,date_start,time_start,date_end,time_end,sor_id
P01,1,  dj45 ,1,2020-01-05,12:34:56,2020-01-06,08:00:00,="21000016006"
P01,2,DJ45,2,2020-01-05,12:34:00,.,.,21000016006.0
P02,3,da01,1.0,2020-01-05, 08:15:00 ,2020-01-07,,31000016008
P02,3,da01,1.0,2020-01-05, 08:15:00 ,2020-01-07,,31000016008
P03,4,,X,2020-01-05,25:00:00,2020-01-08,23:59:59,
P99,5,dz99,2,2020-01-05,00:00:00,2020-01-09,00:00:00,99999
,6,dk20,1,2020-01-05,10:00:00,2020-01-10,10:00:00,21000016006
P04,9,dk21,2,2018-02-10,,2018-02-11,,21000016006
P04,8,dk20,ALCC01,2019-01-05,10:00:00,2020-01-10,10:00:00,21000016006
P04,8,dk20,ALCC01,2019-01-05,10:00:00,2020-01-10,10:00:00,21000016006
"""

# Join target for a mapping where the source key is a string column and the map key is numeric.
CONTACT_REGION_CSV = """\
contact_id,region_code
1,1084
2,1085
3,1084.0
4,
8,1086
"""


def t_adm_table() -> pa.Table:
    """Typed parquet input: hour/minute components, int-like floats, timezone-aware datetimes."""
    return pa.table(
        {
            "v_cpr": pa.array(["P01", "P02", "P03", "P04", "P04"]),
            "c_adiag": pa.array(["dk20 ", "dk21", "dk22", None, "dk24"]),
            "d_inddto": pa.array([dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3), dt.date(2020, 1, 4), None]),
            # 24 is out of range: the hour and everything finer must be dropped.
            "v_indtime": pa.array([10, 24, None, 0, 5], type=pa.int64()),
            # 60 is out of range; minute is ignored once hour is unresolved.
            "v_indminut": pa.array([30.0, 15.0, 45.0, 60.0, 1.0], type=pa.float64()),
            # int-like floats normalize to Int64; 2.5 becomes null.
            "c_indm": pa.array([1.0, 2.0, 2.5, None, 1.0], type=pa.float64()),
            "c_kontaars": pa.array(["2", "3.0", "9", "5", None]),
            "loaded_at": pa.array(
                [dt.datetime(2020, 1, 1, 12, tzinfo=dt.timezone.utc)] * 5, type=pa.timestamp("us", tz="Europe/Copenhagen")
            ),
        }
    )


def write_inputs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "population.csv").write_text(POPULATION_CSV)
    (raw_dir / "contacts.csv").write_text(CONTACTS_CSV)
    (raw_dir / "contact_region.csv").write_text(CONTACT_REGION_CSV)
    pq.write_table(t_adm_table(), raw_dir / "t_adm.parquet")


# Same shape as the YAML tables in configs/preMEDS; $RAW is the raw input directory.
TABLES_YAML = """
contacts:
  filename: $RAW/contacts.csv
  columns: {pid: subject_id, contact_id: null, diag: code, priority: null, date_start: null,
            time_start: null, date_end: null, time_end: null, sor_id: null}
  mappings:
    - {source_column: contact_id, via_file: $RAW/contact_region.csv, join_on: contact_id,
       target_columns: {region_code: null}, how: left}
    - {function: sor, source_column: sor_id, source_date_column: date_start,
       via_file: sor2_contact_mapping.parquet, join_on: sor_id}
  normalize_integer_columns: [region, region_code]
  value_map:
    date_end: {".": null}
    time_end: {".": null}
    priority: {1: ATA1, 2: ATA3}
  timestamp_columns:
    timestamp: {date: date_start, time: time_start}
    end_timestamp: {date: date_end, time: time_end}
t_adm:
  filename: $RAW/t_adm.parquet
  columns: {v_cpr: subject_id, c_adiag: code, d_inddto: date_start, v_indtime: hour_start,
            v_indminut: minute_start, c_indm: priority, c_kontaars: null, loaded_at: null}
  normalize_integer_columns: [priority, c_kontaars]
  value_map:
    priority: {1: ATA1, 2: ATA3}
    c_kontaars: {2: ALCC02, 3: ALCC03, 9: ALCC90}
  timestamp_columns:
    timestamp: {date: date_start, hour: hour_start, minute: minute_start}
"""


def table_configs(raw_dir: Path):
    return OmegaConf.create(TABLES_YAML.replace("$RAW", str(raw_dir)))

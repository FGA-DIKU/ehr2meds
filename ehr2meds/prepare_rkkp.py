"""Shape the merged RKQP quality-register table into longitudinal events."""

from __future__ import annotations

import argparse
import json
import logging
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "patientid",
    "timestamp",
    "code",
    "numeric_value",
    "text_value",
    "supertype",
    "source_column",
    "time_source",
    "temporal_confidence",
]


@dataclass(frozen=True)
class EventSpec:
    column: str
    namespace: str
    concept: str | None = None
    numeric: bool = False
    unit: str | None = None
    time_column: str = "date_diagnosis"
    fallback_time_columns: tuple[str, ...] = ()
    temporal_confidence: str = "diagnosis_assumed"


BASELINE_SPECS = [
    EventSpec("supertype", "RKQP_REGISTRY", temporal_confidence="exact"),
    EventSpec("subtype", "RKQP_SUBTYPE"),
    EventSpec("PS_diagnosis", "RKQP_PS"),
    EventSpec("AA_stage_diagnosis", "RKQP_STAGE", "ANN_ARBOR"),
    EventSpec("binet", "RKQP_STAGE", "BINET"),
    EventSpec("ISS_diagnosis", "RKQP_STAGE", "ISS"),
    EventSpec("RISS_diagnosis", "RKQP_STAGE", "RISS"),
    EventSpec("IPI_score_diagnosis", "RKQP_SCORE", "IPI"),
    EventSpec("aaIPI_score_diagnosis", "RKQP_SCORE", "AAIPI"),
    EventSpec("IPS_score_diagnosis", "RKQP_SCORE", "IPS"),
    EventSpec("FLIPI_diagnosis", "RKQP_SCORE", "FLIPI"),
    EventSpec("FLIPI2_diagnosis", "RKQP_SCORE", "FLIPI2"),
    EventSpec("n_regions_diagnosis", "RKQP_FEATURE", "N_REGIONS"),
    EventSpec(
        "n_extranodal_regions_diagnosis",
        "RKQP_FEATURE",
        "N_EXTRANODAL_REGIONS",
    ),
    EventSpec("max_tumor_diameter_diagnosis", "RKQP_FEATURE", "MAX_TUMOR_DIAMETER", numeric=True),
    EventSpec("discordant_lymphoma_diagnosis", "RKQP_FEATURE", "DISCORDANT_LYMPHOMA"),
    EventSpec("b_symptoms_diagnosis", "RKQP_FEATURE", "B_SYMPTOMS"),
    EventSpec("bulky_disease_diagnosis", "RKQP_FEATURE", "BULKY_DISEASE"),
    EventSpec("other_malignancy_diagnosis", "RKQP_FEATURE", "OTHER_MALIGNANCY"),
    EventSpec("nodal_disease_diagnosis", "RKQP_FEATURE", "NODAL_DISEASE"),
    EventSpec("extranodal_disease_diagnosis", "RKQP_FEATURE", "EXTRANODAL_DISEASE"),
    EventSpec("CNS_involvement_diagnosis", "RKQP_FEATURE", "CNS_INVOLVEMENT"),
    EventSpec("progression_from_prior_MGUS", "RKQP_FEATURE", "PROGRESSION_FROM_PRIOR_MGUS"),
    EventSpec("bone_lesion_diagnosis", "RKQP_FEATURE", "BONE_LESION"),
    EventSpec("extramedullary_myeloma_diagnosis", "RKQP_FEATURE", "EXTRAMEDULLARY_MYELOMA"),
    EventSpec("amyloidosis_diagnosis", "RKQP_FEATURE", "AMYLOIDOSIS"),
    EventSpec("renal_insufficiency_diagnosis", "RKQP_FEATURE", "RENAL_INSUFFICIENCY"),
    EventSpec("IGHV", "RKQP_BIOMARKER", "IGHV"),
    EventSpec("del13q", "RKQP_BIOMARKER", "DEL13Q"),
    EventSpec("tri12", "RKQP_BIOMARKER", "TRISOMY12"),
    EventSpec("del11q", "RKQP_BIOMARKER", "DEL11Q"),
    EventSpec("del17p", "RKQP_BIOMARKER", "DEL17P"),
    EventSpec("TP53_mut", "RKQP_BIOMARKER", "TP53_MUTATION"),
    EventSpec("CD38", "RKQP_BIOMARKER", "CD38"),
    EventSpec("ZAP70", "RKQP_BIOMARKER", "ZAP70"),
    EventSpec("FISH_no_aberrations", "RKQP_BIOMARKER", "FISH_NO_ABERRATIONS"),
    EventSpec("FISH_t4_14", "RKQP_BIOMARKER", "FISH_T4_14"),
    EventSpec("FISH_t11_14", "RKQP_BIOMARKER", "FISH_T11_14"),
    EventSpec("FISH_t14_16", "RKQP_BIOMARKER", "FISH_T14_16"),
    EventSpec("FISH_t14_20", "RKQP_BIOMARKER", "FISH_T14_20"),
    EventSpec("FISH_amp1q", "RKQP_BIOMARKER", "FISH_AMP1Q"),
    EventSpec("FISH_amp11q", "RKQP_BIOMARKER", "FISH_AMP11Q"),
    EventSpec("HB_diagnosis", "RKQP_LABTEST", "HB", numeric=True),
    EventSpec("WBC_diagnosis", "RKQP_LABTEST", "WBC", numeric=True),
    EventSpec("TRC_diagnosis", "RKQP_LABTEST", "PLATELETS", numeric=True),
    EventSpec("lymphocyte_percentage_diagnosis", "RKQP_LABTEST", "LYMPHOCYTES_PERCENT", numeric=True),
    EventSpec("ALC_diagnosis", "RKQP_LABTEST", "ALC", numeric=True),
    EventSpec("ALB_diagnosis", "RKQP_LABTEST", "ALBUMIN", numeric=True),
    EventSpec("CA2_diagnosis", "RKQP_LABTEST", "CALCIUM", numeric=True),
    EventSpec("CA_albumin_corrected_diagnosis", "RKQP_LABTEST", "CORRECTED_CALCIUM", numeric=True),
    EventSpec("CREA_diagnosis", "RKQP_LABTEST", "CREATININE", numeric=True),
    EventSpec("B2M_diagnosis", "RKQP_LABTEST", "BETA2_MICROGLOBULIN", numeric=True),
    EventSpec("LDH_diagnosis", "RKQP_LABTEST", "LDH", numeric=True),
    EventSpec("bilirubin_diagnosis", "RKQP_LABTEST", "BILIRUBIN", numeric=True),
    EventSpec("ALAT_diagnosis", "RKQP_LABTEST", "ALAT", numeric=True),
    EventSpec("BASP_diagnosis", "RKQP_LABTEST", "BASP", numeric=True),
    EventSpec("IgA_diagnosis", "RKQP_LABTEST", "IGA", numeric=True),
    EventSpec("IgG_diagnosis", "RKQP_LABTEST", "IGG", numeric=True),
    EventSpec("IgM_diagnosis", "RKQP_LABTEST", "IGM", numeric=True),
]

RESPONSE_SPECS = [
    EventSpec(
        "response_1st_line",
        "RKQP_RESPONSE",
        "LINE_1",
        time_column="date_response_1st_line",
        temporal_confidence="recorded_event_date",
    ),
    EventSpec(
        "response_2nd_line",
        "RKQP_RESPONSE",
        "LINE_2",
        time_column="date_response_2nd_line",
        temporal_confidence="recorded_event_date",
    ),
]

TREATMENT_EXCLUSIONS = re.compile(
    r"(^date_|response|complication|_fu$|death|within_\d+_days|^time_|^PS_|^HB_|^B2M_|^ALB_|^CREA_)"
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".asc"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output type: {path.suffix}")


def format_category(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    return value.replace("//", "/")


def parse_numeric(series: pd.Series) -> pd.Series:
    """Extract strict numeric values while leaving inequalities as text."""
    extracted = series.astype("string").str.extract(
        r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
        expand=False,
    )
    return pd.to_numeric(extracted, errors="coerce")


def make_events(df: pd.DataFrame, spec: EventSpec) -> pd.DataFrame:
    if spec.column not in df or spec.time_column not in df:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    values = df[spec.column]
    timestamp = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    time_source = pd.Series(pd.NA, index=df.index, dtype="string")
    for time_column in (spec.time_column, *spec.fallback_time_columns):
        if time_column not in df:
            continue
        candidate = pd.to_datetime(df[time_column], errors="coerce")
        fill = timestamp.isna() & candidate.notna()
        timestamp.loc[fill] = candidate.loc[fill]
        time_source.loc[fill] = time_column
    keep = values.notna() & timestamp.notna()
    if not keep.any():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    source = df.loc[keep]
    values = values.loc[keep]
    concept = spec.concept or ""
    base_code = spec.namespace + (f"//{concept}" if concept else "")

    if spec.numeric:
        numeric = parse_numeric(values)
        text = values.astype("string").where(numeric.isna())
        code = base_code + (f"//{spec.unit}" if spec.unit else "//UNIT_UNKNOWN")
    else:
        categories = values.map(format_category)
        valid = categories.notna()
        source = source.loc[valid]
        timestamp = timestamp.loc[keep].loc[valid]
        categories = categories.loc[valid]
        numeric = pd.Series(np.nan, index=source.index, dtype="float64")
        text = pd.Series(pd.NA, index=source.index, dtype="string")
        code = categories.map(lambda value: f"{base_code}//{value}")

    return pd.DataFrame(
        {
            "patientid": source["patientid"],
            "timestamp": timestamp if not spec.numeric else timestamp.loc[keep],
            "code": code,
            "numeric_value": numeric,
            "text_value": text,
            "supertype": source["supertype"],
            "source_column": spec.column,
            "time_source": time_source.loc[source.index],
            "temporal_confidence": spec.temporal_confidence,
        }
    )[OUTPUT_COLUMNS]


def treatment_time_columns(column: str, line: int, available: set[str]) -> tuple[str, ...]:
    suffix = f"_{line}{'st' if line == 1 else 'nd'}_line"
    if not column.endswith(suffix):
        return ()

    if "chemo" in column:
        candidates = [f"date_chemo_start{suffix}", f"date_treatment{suffix}"]
    elif "immun" in column:
        candidates = [f"date_immuno_start{suffix}", f"date_treatment{suffix}"]
    elif column.startswith("RT_") or "radiation" in column:
        candidates = [f"date_RT{suffix}", f"date_treatment{suffix}"]
    elif "surgery" in column:
        candidates = [f"date_surgery{suffix}"]
    elif "stem_cell" in column or "transplant" in column or "ASCT" in column or "HDT" in column:
        candidates = [f"date_stem_cell_infusion{suffix}", f"date_treatment{suffix}"]
    else:
        candidates = [f"date_treatment{suffix}"]
        if line == 2:
            candidates.insert(0, "date_treatment_2nd_line_start")

    return tuple(candidate for candidate in candidates if candidate in available)


def treatment_numeric_details(column: str) -> tuple[bool, str | None]:
    if "dosis_Gy" in column:
        return True, "GY"
    if "dosis_mCkg" in column:
        return True, "MCGY"
    return False, None


def treatment_specs(columns: list[str]) -> list[EventSpec]:
    """Build conservative treatment specs; response/outcome fields stay separate."""
    available = set(columns)
    specs = []
    for column in columns:
        if TREATMENT_EXCLUSIONS.search(column):
            continue
        line = 1 if column.endswith("_1st_line") else 2 if column.endswith("_2nd_line") else None
        if line is None:
            continue
        time_columns = treatment_time_columns(column, line, available)
        if not time_columns:
            continue
        concept = re.sub(r"_(1st|2nd)_line$", "", column).upper()
        numeric, unit = treatment_numeric_details(column)
        specs.append(
            EventSpec(
                column,
                "RKQP_TREATMENT",
                f"LINE_{line}//{concept}",
                numeric=numeric,
                unit=unit,
                time_column=time_columns[0],
                fallback_time_columns=time_columns[1:],
                temporal_confidence="recorded_or_line_start",
            )
        )
    return specs


def summarize_numeric_distributions(
    df: pd.DataFrame,
    specs: list[EventSpec],
    *,
    wbc_max: float,
    minimum_group_size: int = 20,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize numeric fields by registry and flag clearly different scales."""
    rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    for spec in specs:
        if not spec.numeric or spec.column not in df:
            continue
        values = parse_numeric(df[spec.column])
        if spec.column == "WBC_diagnosis":
            values = values.where((values > 0) & (values <= wbc_max))

        field_rows = []
        for registry, indices in df.groupby("supertype", dropna=False).groups.items():
            group = values.loc[indices].dropna()
            if group.empty:
                continue
            quantiles = group.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            row = {
                "source_column": spec.column,
                "code": spec.namespace + (f"//{spec.concept}" if spec.concept else ""),
                "supertype": format_category(registry),
                "count": int(len(group)),
                "q01": float(quantiles.loc[0.01]),
                "q25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "q75": float(quantiles.loc[0.75]),
                "q99": float(quantiles.loc[0.99]),
            }
            rows.append(row)
            if len(group) >= minimum_group_size:
                field_rows.append(row)

        positive_medians = [row for row in field_rows if row["median"] > 0]
        if len(positive_medians) >= 2:
            smallest = min(positive_medians, key=lambda row: row["median"])
            largest = max(positive_medians, key=lambda row: row["median"])
            ratio = largest["median"] / smallest["median"]
            if ratio >= 5:
                warnings.append(
                    {
                        "source_column": spec.column,
                        "reason": "registry medians differ by at least five-fold; verify units before sharing one code",
                        "median_ratio": float(ratio),
                        "smallest_median_supertype": smallest["supertype"],
                        "largest_median_supertype": largest["supertype"],
                    }
                )

    return rows, warnings


def shape_rkkp(
    df: pd.DataFrame,
    *,
    include_treatments: bool = False,
    include_responses: bool = False,
    wbc_max: float = 1_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"patientid", "date_diagnosis", "supertype"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required RKQP columns: {sorted(missing)}")

    specs = list(BASELINE_SPECS)
    if include_treatments:
        specs.extend(treatment_specs(list(df.columns)))
    if include_responses:
        specs.extend(RESPONSE_SPECS)

    frames = [make_events(df, spec) for spec in specs]
    events = pd.concat(frames, ignore_index=True)

    wbc = events["source_column"].eq("WBC_diagnosis")
    invalid_wbc = wbc & ((events["numeric_value"] <= 0) | (events["numeric_value"] > wbc_max))
    n_invalid_wbc = int(invalid_wbc.sum())
    events = events.loc[~invalid_wbc]

    events = events.dropna(subset=["patientid", "timestamp", "code"])
    events = events.drop_duplicates().sort_values(["patientid", "timestamp", "code"], kind="stable")
    events = events.reset_index(drop=True)

    distributions, distribution_warnings = summarize_numeric_distributions(
        df,
        specs,
        wbc_max=wbc_max,
    )

    report = {
        "input_rows": int(len(df)),
        "output_events": int(len(events)),
        "patients": int(events["patientid"].nunique()),
        "include_treatments": include_treatments,
        "include_responses": include_responses,
        "wbc_valid_range": {"exclusive_min": 0, "inclusive_max": wbc_max},
        "wbc_events_removed": n_invalid_wbc,
        "numeric_distributions_by_supertype": distributions,
        "numeric_distribution_warnings": distribution_warnings,
        "events_by_namespace": events["code"].str.split("//").str[0].value_counts().sort_index().to_dict(),
        "events_by_supertype": events["supertype"].value_counts().sort_index().to_dict(),
    }
    return events, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Merged RKQP CSV or Parquet file")
    parser.add_argument("output", type=Path, help="Long event output (.csv or .parquet)")
    parser.add_argument(
        "--include-treatments",
        action="store_true",
        help="Include first- and second-line treatment fields at their recorded treatment anchors",
    )
    parser.add_argument(
        "--include-responses",
        action="store_true",
        help="Include response events; disabled by default because retrospective recording may leak future information",
    )
    parser.add_argument("--wbc-max", type=float, default=1_000, help="Maximum accepted WBC value (default: 1000)")
    parser.add_argument("--report", type=Path, help="QC report path (default: <output>.summary.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Reading %s", args.input)
    source = read_table(args.input)
    events, report = shape_rkkp(
        source,
        include_treatments=args.include_treatments,
        include_responses=args.include_responses,
        wbc_max=args.wbc_max,
    )
    write_table(events, args.output)
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".summary.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %s events and QC report %s", f"{len(events):,}", report_path)


if __name__ == "__main__":
    main()

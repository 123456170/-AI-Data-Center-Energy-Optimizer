import os
import io
import json
import math
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import plotly.graph_objects as go
import plotly.express as px
import psutil

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


APP_TITLE = "AI Data-Center Energy Optimizer"
SCHEMA_VERSION = "1.0.0"
SEED = 42
DB_PATH = "dc_energy_optimizer.db"
VIDEO_DIR = Path("demo_videos")
VIDEO_PATH = VIDEO_DIR / "ai_dc_energy_optimizer_demo.mp4"
VIDEO_FALLBACK_PATH = VIDEO_DIR / "ai_dc_energy_optimizer_demo.avi"

SITE_ID = "dc-synthetic-east-1"

RACKS = [
    {"rack_id": "rack-a", "role": "critical", "base_it": 8.2},
    {"rack_id": "rack-b", "role": "critical", "base_it": 7.5},
    {"rack_id": "rack-c", "role": "critical", "base_it": 9.1},
    {"rack_id": "rack-d", "role": "batch", "base_it": 6.4},
    {"rack_id": "rack-e", "role": "batch", "base_it": 5.8},
    {"rack_id": "rack-f", "role": "batch", "base_it": 7.0},
]
RACK_INDEX = {r["rack_id"]: i for i, r in enumerate(RACKS)}

REQUIRED_COLUMNS = [
    "ts_utc",
    "site_id",
    "rack_id",
    "it_load_kw",
    "cooling_kw",
    "total_power_kw",
    "pue",
    "outdoor_temp_c",
    "economizer_available",
    "carbon_intensity_gco2_per_kwh",
    "inlet_temp_c",
    "relative_humidity_pct",
    "workload_priority",
    "deferred_load_kw",
    "sensor_status",
    "quality_score",
    "provenance",
    "schema_version",
]

NUMERIC_COLUMNS = [
    "it_load_kw",
    "cooling_kw",
    "total_power_kw",
    "pue",
    "outdoor_temp_c",
    "carbon_intensity_gco2_per_kwh",
    "inlet_temp_c",
    "relative_humidity_pct",
    "deferred_load_kw",
    "quality_score",
]

FEATURE_COLS = [
    "it_load_kw",
    "outdoor_temp_c",
    "hour",
    "dayofweek",
    "is_weekend",
    "economizer_int",
    "priority_batch",
    "relative_humidity_pct",
]

TARGET = "total_power_kw"

LIVE_EXTRA_COLUMNS = [
    "baseline_total_power_kw",
    "pred_total_power_kw",
    "pred_std",
    "pred_lower",
    "pred_upper",
    "model_confidence",
    "linear_pred_total_power_kw",
    "fallback_used",
    "validation_errors",
    "ingested_at",
]

DEFAULT_CONTROLS = {
    "electricity_cost_per_kwh": 0.085,
    "carbon_price_per_tco2": 35.0,
    "max_inlet_temp_c": 27.0,
    "cooling_setpoint_c": 12.0,
    "fan_speed_pct": 75.0,
    "economizer_auto": False,
    "batch_shift_pct": 0.0,
    "batch_shift_cap": 20.0,
    "critical_sla_pct": 99.9,
    "human_approval_required": True,
}

CONTROL_KEY_MAP = {k: f"ctrl_{k}" for k in DEFAULT_CONTROLS}

DATA_CONTRACT = [
    {"field": "ts_utc", "type": "datetime", "unit": "UTC ISO-8601", "required": True, "description": "Measurement timestamp."},
    {"field": "site_id", "type": "string", "unit": "identifier", "required": True, "description": "Data-center site identifier."},
    {"field": "rack_id", "type": "string", "unit": "identifier", "required": True, "description": "Rack or row identifier."},
    {"field": "it_load_kw", "type": "float", "unit": "kW", "required": True, "description": "Measured IT power draw."},
    {"field": "cooling_kw", "type": "float", "unit": "kW", "required": True, "description": "Cooling subsystem power."},
    {"field": "total_power_kw", "type": "float", "unit": "kW", "required": True, "description": "IT + cooling + auxiliary power."},
    {"field": "pue", "type": "float", "unit": "ratio", "required": True, "description": "Total power divided by IT power."},
    {"field": "outdoor_temp_c", "type": "float", "unit": "degC", "required": True, "description": "Outdoor dry-bulb temperature."},
    {"field": "economizer_available", "type": "bool", "unit": "flag", "required": True, "description": "Free-cooling conditions are available."},
    {"field": "carbon_intensity_gco2_per_kwh", "type": "float", "unit": "gCO2e/kWh", "required": True, "description": "Grid carbon intensity."},
    {"field": "inlet_temp_c", "type": "float", "unit": "degC", "required": True, "description": "Server rack inlet temperature."},
    {"field": "relative_humidity_pct", "type": "float", "unit": "%", "required": True, "description": "Relative humidity."},
    {"field": "workload_priority", "type": "string", "unit": "enum", "required": True, "description": "critical or batch."},
    {"field": "deferred_load_kw", "type": "float", "unit": "kW", "required": True, "description": "Load deferred by shifting."},
    {"field": "sensor_status", "type": "string", "unit": "enum", "required": True, "description": "ok, missing, stale, outlier, fallback."},
    {"field": "quality_score", "type": "float", "unit": "0-1", "required": True, "description": "Row-level data quality score."},
    {"field": "provenance", "type": "string", "unit": "enum", "required": True, "description": "historical, simulated_live, simulated_fallback."},
    {"field": "schema_version", "type": "string", "unit": "semver-like", "required": True, "description": "Canonical schema version."},
]

RISK_REGISTER = [
    {
        "ID": "R1",
        "Risk": "Sensor outage or stale telemetry",
        "Likelihood": "Medium",
        "Impact": "High",
        "Early warning": "quality_score < 0.75, sensor_status missing/stale",
        "Mitigation": "Quarantine row, use deterministic fallback, alert operator",
        "Human approval": "Yes for corrective setpoint/fan changes",
        "Acceptance check": "Fault injection produces fallback and no unsafe recommendation",
    },
    {
        "ID": "R2",
        "Risk": "Model prediction error",
        "Likelihood": "Medium",
        "Impact": "Medium",
        "Early warning": "low model_confidence, wide uncertainty interval",
        "Mitigation": "Compare with transparent baseline, suppress savings actions, hold",
        "Human approval": "Yes for energy-saving setpoint changes",
        "Acceptance check": "Low-confidence rows generate Hold or protective action only",
    },
    {
        "ID": "R3",
        "Risk": "Thermal guardrail violation",
        "Likelihood": "Low",
        "Impact": "Critical",
        "Early warning": "inlet_temp_c near max_inlet_temp_c",
        "Mitigation": "Block energy-saving actions, recommend protective cooling",
        "Human approval": "Yes, but safe failure mode is hold and alert",
        "Acceptance check": "No recommendation proposes raising setpoint when margin < 2 degC",
    },
    {
        "ID": "R4",
        "Risk": "Workload shift harms SLA",
        "Likelihood": "Medium",
        "Impact": "High",
        "Early warning": "critical_sla_pct, batch deferral queue depth",
        "Mitigation": "Only shift batch load, cap shift percent, require approval",
        "Human approval": "Yes",
        "Acceptance check": "Critical racks are never shifted in demo logic",
    },
    {
        "ID": "R5",
        "Risk": "Privacy or operational confidentiality",
        "Likelihood": "Low",
        "Impact": "High",
        "Early warning": "real customer data introduced",
        "Mitigation": "Initial demo uses synthetic data only; no personal data",
        "Human approval": "Data onboarding requires governance review",
        "Acceptance check": "All demo records have provenance synthetic/simulated",
    },
    {
        "ID": "R6",
        "Risk": "Fairness across business units",
        "Likelihood": "Medium",
        "Impact": "Medium",
        "Early warning": "one rack/group consistently deferred or throttled",
        "Mitigation": "Segment metrics, quota batch shift, review audit history",
        "Human approval": "Yes for recurring shift policies",
        "Acceptance check": "Dashboard shows per-rack and per-priority outcomes",
    },
    {
        "ID": "R7",
        "Risk": "Misuse as automatic control system",
        "Likelihood": "Medium",
        "Impact": "Critical",
        "Early warning": "auto-apply enabled without approval",
        "Mitigation": "Consequential actions require human approval in prototype",
        "Human approval": "Yes",
        "Acceptance check": "Every actionable recommendation has requires_approval=True",
    },
    {
        "ID": "R8",
        "Risk": "Cost of computing exceeds savings",
        "Likelihood": "Low",
        "Impact": "Medium",
        "Early warning": "CPU/memory high, latency high",
        "Mitigation": "CPU-first models, batching, caching, resource monitor",
        "Human approval": "No",
        "Acceptance check": "Demo runs on CPU-only Windows laptop",
    },
]


class Reading(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ts_utc: datetime
    site_id: str
    rack_id: str
    it_load_kw: float
    cooling_kw: float
    total_power_kw: float
    pue: float
    outdoor_temp_c: float
    economizer_available: bool
    carbon_intensity_gco2_per_kwh: float
    inlet_temp_c: float
    relative_humidity_pct: float
    workload_priority: str
    deferred_load_kw: float = 0.0
    sensor_status: str
    quality_score: float
    provenance: str
    schema_version: str = SCHEMA_VERSION
    ingested_at: Optional[datetime] = None

    @field_validator("workload_priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        if v not in {"critical", "batch"}:
            raise ValueError("workload_priority must be critical or batch")
        return v

    @field_validator("sensor_status")
    @classmethod
    def validate_sensor_status(cls, v: str) -> str:
        allowed = {"ok", "missing", "stale", "outlier", "fallback"}
        if v not in allowed:
            raise ValueError(f"sensor_status must be one of {allowed}")
        return v

    @field_validator(
        "it_load_kw",
        "cooling_kw",
        "total_power_kw",
        "pue",
        "outdoor_temp_c",
        "carbon_intensity_gco2_per_kwh",
        "inlet_temp_c",
        "relative_humidity_pct",
        "quality_score",
    )
    @classmethod
    def validate_finite(cls, v: float) -> float:
        vf = float(v)
        if math.isnan(vf) or math.isinf(vf):
            raise ValueError("non-finite numeric value")
        return vf


def utc_now_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def empty_live_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS + LIVE_EXTRA_COLUMNS)


def generate_historical_raw() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    end = utc_now_ts().floor("h")
    start = end - pd.Timedelta(days=7)
    times = pd.date_range(start, end, freq="h", inclusive="left")

    rows: List[Dict[str, Any]] = []

    for ts in times:
        hour = ts.hour
        dow = ts.dayofweek

        outdoor = 12 + 8 * math.sin(2 * math.pi * (hour - 14) / 24)
        outdoor += 3 * math.sin(2 * math.pi * dow / 7)
        outdoor += float(rng.normal(0, 0.8))
        outdoor = float(np.clip(outdoor, -5, 35))

        humidity = 55 + 10 * math.sin(2 * math.pi * (hour - 9) / 24)
        humidity += float(rng.normal(0, 3))
        humidity = float(np.clip(humidity, 20, 95))

        carbon = 260 + 130 * math.sin(2 * math.pi * (hour - 8) / 24)
        carbon += float(rng.normal(0, 18))
        carbon = float(np.clip(carbon, 70, 750))

        economizer = bool(outdoor < 18 and humidity < 85)

        for rack in RACKS:
            daily = 1 + 0.18 * math.sin(2 * math.pi * (hour - 13) / 24)
            weekly = 0.72 if dow >= 5 else 1.0
            it = rack["base_it"] * daily * weekly * (1 + float(rng.normal(0, 0.06)))
            it = float(np.clip(it, 0.8, 12.0))

            setpoint = 12.0
            fan = 75.0
            econ_bonus = 0.06 if economizer else 0.0

            cooling = it * (
                0.30
                + 0.013 * max(0.0, outdoor - 16.0)
                - econ_bonus
                - 0.007 * (setpoint - 12.0)
                - 0.0015 * (fan - 70.0)
            )
            cooling += float(rng.normal(0, 0.12))
            cooling = float(max(0.07 * it, cooling))

            aux = 0.05 * it
            total = it + cooling + aux
            pue = total / it

            inlet = (
                setpoint
                + 3.7
                + 0.20 * (it / 10.0)
                - 1.4 * (1.0 if economizer else 0.0)
                + 0.018 * (75.0 - fan)
                + float(rng.normal(0, 0.25))
            )
            inlet = float(np.clip(inlet, 10.0, 32.0))

            row = {
                "ts_utc": ts,
                "site_id": SITE_ID,
                "rack_id": rack["rack_id"],
                "it_load_kw": it,
                "cooling_kw": cooling,
                "total_power_kw": total,
                "pue": pue,
                "outdoor_temp_c": outdoor,
                "economizer_available": economizer,
                "carbon_intensity_gco2_per_kwh": carbon,
                "inlet_temp_c": inlet,
                "relative_humidity_pct": humidity,
                "workload_priority": rack["role"],
                "deferred_load_kw": 0.0,
                "sensor_status": "ok",
                "quality_score": 0.95,
                "provenance": "historical_synthetic",
                "schema_version": SCHEMA_VERSION,
                "ingested_at": ts + pd.Timedelta(minutes=2),
            }

            u = float(rng.random())
            if u < 0.012:
                row.update(
                    {
                        "it_load_kw": np.nan,
                        "cooling_kw": np.nan,
                        "total_power_kw": np.nan,
                        "pue": np.nan,
                        "sensor_status": "missing",
                        "quality_score": 0.15,
                    }
                )
            elif u < 0.020:
                row["total_power_kw"] = float(row["total_power_kw"]) * 1.9
                row["pue"] = float(row["total_power_kw"]) / float(row["it_load_kw"])
                row["sensor_status"] = "outlier"
                row["quality_score"] = 0.55
            elif u < 0.024:
                row["sensor_status"] = "stale"
                row["total_power_kw"] = float(row["total_power_kw"]) * 0.98
                row["quality_score"] = 0.65

            rows.append(row)

    if rows:
        dup_rows = []
        for r in rows[-12:]:
            d = dict(r)
            d["ingested_at"] = pd.Timestamp(d["ingested_at"]) + pd.Timedelta(minutes=7)
            dup_rows.append(d)
        rows.extend(dup_rows)

    return pd.DataFrame(rows)


def validate_dataframe(df: pd.DataFrame, source: str = "raw") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    report = {
        "source": source,
        "input_rows": int(len(df)),
        "valid_rows": 0,
        "quarantined_rows": 0,
        "duplicate_rows": 0,
        "missing_critical_rows": 0,
        "outlier_rows": 0,
        "late_rows": 0,
        "schema_version": SCHEMA_VERSION,
    }

    if df.empty:
        return df.copy(), df.copy(), report

    data = df.copy()

    if "ingested_at" not in data.columns:
        data["ingested_at"] = utc_now_ts()

    data["ts_utc"] = pd.to_datetime(data["ts_utc"], utc=True, errors="coerce")
    data["ingested_at"] = pd.to_datetime(data["ingested_at"], utc=True, errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "economizer_available" in data.columns:
        data["economizer_available"] = data["economizer_available"].astype(bool)

    data["sensor_status"] = data.get("sensor_status", pd.Series(["missing"] * len(data), index=data.index)).fillna("missing").astype(str)
    data["workload_priority"] = data.get("workload_priority", pd.Series(["critical"] * len(data), index=data.index)).fillna("critical").astype(str)
    data["provenance"] = data.get("provenance", pd.Series([source] * len(data), index=data.index)).fillna(source).astype(str)
    data["schema_version"] = data.get("schema_version", pd.Series([SCHEMA_VERSION] * len(data), index=data.index)).fillna(SCHEMA_VERSION).astype(str)
    data["quality_score"] = data.get("quality_score", pd.Series([0.0] * len(data), index=data.index)).fillna(0.0).clip(0, 1)
    data["deferred_load_kw"] = data.get("deferred_load_kw", pd.Series([0.0] * len(data), index=data.index)).fillna(0.0)

    data["validation_errors"] = [[] for _ in range(len(data))]

    def add_error(mask: pd.Series, msg: str) -> None:
        for idx in data.index[mask]:
            data.at[idx, "validation_errors"].append(msg)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_cols:
        add_error(pd.Series(True, index=data.index), f"missing_columns:{','.join(missing_cols)}")
    else:
        add_error(data["ts_utc"].isna(), "invalid_timestamp")

        critical_numeric = [
            "it_load_kw",
            "cooling_kw",
            "total_power_kw",
            "pue",
            "outdoor_temp_c",
            "carbon_intensity_gco2_per_kwh",
            "inlet_temp_c",
            "relative_humidity_pct",
        ]
        miss_mask = data[critical_numeric].isna().any(axis=1)
        add_error(miss_mask, "missing_critical_value")
        report["missing_critical_rows"] = int(miss_mask.sum())

        add_error(data["it_load_kw"] < 0, "negative_it_load")
        add_error(data["cooling_kw"] < 0, "negative_cooling")
        add_error(data["total_power_kw"] < 0, "negative_total_power")
        add_error((data["pue"] < 1.0) | (data["pue"] > 3.0), "pue_out_of_range")
        add_error((data["inlet_temp_c"] < 0) | (data["inlet_temp_c"] > 60), "inlet_temp_out_of_range")
        add_error((data["relative_humidity_pct"] < 0) | (data["relative_humidity_pct"] > 100), "humidity_out_of_range")
        add_error(~data["workload_priority"].isin(["critical", "batch"]), "invalid_priority")
        add_error(~data["sensor_status"].isin(["ok", "missing", "stale", "outlier", "fallback"]), "invalid_sensor_status")

        outlier_mask = (data["total_power_kw"] > 35.0) | (data["cooling_kw"] > 25.0)
        add_error(outlier_mask, "outlier_power")
        report["outlier_rows"] = int(outlier_mask.sum())

        dup_mask = data.duplicated(subset=["site_id", "rack_id", "ts_utc"], keep="last")
        add_error(dup_mask, "duplicate_record")
        report["duplicate_rows"] = int(dup_mask.sum())

        late_mask = (data["ingested_at"] - data["ts_utc"]) > pd.Timedelta(hours=2)
        data.loc[late_mask, "quality_score"] = (data.loc[late_mask, "quality_score"] - 0.1).clip(0, 1)
        report["late_rows"] = int(late_mask.sum())

    valid_mask = data["validation_errors"].map(len) == 0
    valid = data[valid_mask].copy()
    quarantine = data[~valid_mask].copy()

    valid = valid.sort_values(["ts_utc", "rack_id"]).reset_index(drop=True)
    quarantine = quarantine.sort_values(["ts_utc", "rack_id"]).reset_index(drop=True)

    report["valid_rows"] = int(len(valid))
    report["quarantined_rows"] = int(len(quarantine))

    return valid, quarantine, report


@st.cache_data(show_spinner="Loading synthetic data contract...")
def prepare_historical() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    raw = generate_historical_raw()
    valid, quarantine, report = validate_dataframe(raw, "historical_synthetic")
    return valid, quarantine, report


def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


@st.cache_resource(show_spinner="Initializing local audit database...")
def cached_db_conn() -> sqlite3.Connection:
    return get_db_conn()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS readings (
            ts_utc TEXT NOT NULL,
            site_id TEXT NOT NULL,
            rack_id TEXT NOT NULL,
            it_load_kw REAL,
            cooling_kw REAL,
            total_power_kw REAL,
            pue REAL,
            outdoor_temp_c REAL,
            economizer_available INTEGER,
            carbon_intensity_gco2_per_kwh REAL,
            inlet_temp_c REAL,
            relative_humidity_pct REAL,
            workload_priority TEXT,
            deferred_load_kw REAL,
            sensor_status TEXT,
            quality_score REAL,
            provenance TEXT,
            schema_version TEXT,
            ingested_at TEXT,
            PRIMARY KEY (site_id, rack_id, ts_utc)
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            scope TEXT,
            action TEXT,
            status TEXT,
            requires_approval INTEGER,
            current_value REAL,
            proposed_value REAL,
            predicted_savings_kw REAL,
            predicted_co2_kg_per_hour REAL,
            score REAL,
            confidence REAL,
            evidence_json TEXT,
            approver TEXT,
            approved_at TEXT
        );

        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT,
            event_type TEXT,
            actor TEXT,
            rec_id TEXT,
            details TEXT
        );

        CREATE TABLE IF NOT EXISTS model_runs (
            id TEXT PRIMARY KEY,
            run_at TEXT,
            algorithm TEXT,
            train_rows INTEGER,
            test_rows INTEGER,
            mae REAL,
            rmse REAL,
            r2 REAL,
            coverage REAL,
            notes TEXT
        );
        """
    )
    conn.commit()


def save_readings_to_db(conn: sqlite3.Connection, df: pd.DataFrame, ignore: bool = True) -> None:
    if df.empty:
        return

    cols = [
        "ts_utc",
        "site_id",
        "rack_id",
        "it_load_kw",
        "cooling_kw",
        "total_power_kw",
        "pue",
        "outdoor_temp_c",
        "economizer_available",
        "carbon_intensity_gco2_per_kwh",
        "inlet_temp_c",
        "relative_humidity_pct",
        "workload_priority",
        "deferred_load_kw",
        "sensor_status",
        "quality_score",
        "provenance",
        "schema_version",
        "ingested_at",
    ]

    rows = []
    for _, r in df.iterrows():
        try:
            ing = r.get("ingested_at")
            if pd.isna(ing):
                ing = utc_now_ts()
            rows.append(
                (
                    pd.Timestamp(r["ts_utc"]).isoformat(),
                    str(r["site_id"]),
                    str(r["rack_id"]),
                    float(r["it_load_kw"]),
                    float(r["cooling_kw"]),
                    float(r["total_power_kw"]),
                    float(r["pue"]),
                    float(r["outdoor_temp_c"]),
                    int(bool(r["economizer_available"])),
                    float(r["carbon_intensity_gco2_per_kwh"]),
                    float(r["inlet_temp_c"]),
                    float(r["relative_humidity_pct"]),
                    str(r["workload_priority"]),
                    float(r.get("deferred_load_kw", 0.0)),
                    str(r["sensor_status"]),
                    float(r["quality_score"]),
                    str(r["provenance"]),
                    str(r["schema_version"]),
                    pd.Timestamp(ing).isoformat(),
                )
            )
        except Exception:
            continue

    if not rows:
        return

    marks = ",".join(["?"] * len(cols))
    action = "OR IGNORE" if ignore else ""
    sql = f"INSERT {action} INTO readings ({','.join(cols)}) VALUES ({marks})"
    conn.executemany(sql, rows)
    conn.commit()


def save_recommendation(conn: sqlite3.Connection, rec: Dict[str, Any]) -> None:
    sql = """
    INSERT OR REPLACE INTO recommendations (
        id, created_at, scope, action, status, requires_approval,
        current_value, proposed_value, predicted_savings_kw,
        predicted_co2_kg_per_hour, score, confidence, evidence_json,
        approver, approved_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn.execute(
        sql,
        (
            rec["id"],
            rec["created_at"],
            rec["scope"],
            rec["action"],
            rec["status"],
            int(bool(rec["requires_approval"])),
            float(rec["current_value"]),
            float(rec["proposed_value"]),
            float(rec["predicted_savings_kw"]),
            float(rec["predicted_co2_kg_per_hour"]),
            float(rec["score"]),
            float(rec["confidence"]),
            rec["evidence_json"],
            rec.get("approver"),
            rec.get("approved_at"),
        ),
    )
    conn.commit()


def save_audit_event(conn: sqlite3.Connection, event: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO audit_events (ts_utc, event_type, actor, rec_id, details)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            event["ts_utc"],
            event["event_type"],
            event["actor"],
            event.get("rec_id"),
            event["details"],
        ),
    )
    conn.commit()


def load_recent_audit(conn: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT ts_utc, event_type, actor, rec_id, details
        FROM audit_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [dict(row) for row in cur.fetchall()]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ts_utc"] = pd.to_datetime(d["ts_utc"], utc=True, errors="coerce")
    d["hour"] = d["ts_utc"].dt.hour.fillna(0).astype(int)
    d["dayofweek"] = d["ts_utc"].dt.dayofweek.fillna(0).astype(int)
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["economizer_int"] = d["economizer_available"].astype(int)
    d["priority_batch"] = (d["workload_priority"] == "batch").astype(int)
    return d


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


@st.cache_resource(show_spinner="Training transparent baseline and ML models...")
def get_models() -> Dict[str, Any]:
    hist, _, _ = prepare_historical()
    train_df = hist[hist["quality_score"] >= 0.8].copy().sort_values("ts_utc")

    if len(train_df) < 30:
        raise RuntimeError("Not enough clean synthetic rows to train models.")

    n = len(train_df)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)

    tr = train_df.iloc[:i1]
    va = train_df.iloc[i1:i2]
    te = train_df.iloc[i2:]

    Xtr = add_features(tr)[FEATURE_COLS].values.astype(float)
    Xva = add_features(va)[FEATURE_COLS].values.astype(float)
    Xte = add_features(te)[FEATURE_COLS].values.astype(float)

    ytr = tr[TARGET].values.astype(float)
    yva = va[TARGET].values.astype(float)
    yte = te[TARGET].values.astype(float)

    lin = LinearRegression()
    lin.fit(Xtr, ytr)

    rf = RandomForestRegressor(
        n_estimators=40,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=1,
    )
    rf.fit(Xtr, ytr)

    lin_pred_te = lin.predict(Xte)
    tree_preds = np.vstack([est.predict(Xte) for est in rf.estimators_])
    rf_mean_te = tree_preds.mean(axis=0)
    rf_std_te = tree_preds.std(axis=0)
    conf_te = 1.0 - np.clip(rf_std_te / (np.abs(rf_mean_te) + 1e-6), 0.0, 1.0)

    past_for_baseline = pd.concat([tr, va], ignore_index=True)
    baseline_pred_te = []
    global_mean = float(np.mean(ytr)) if len(ytr) else 10.0

    for _, row in te.iterrows():
        prev = past_for_baseline[
            (past_for_baseline["rack_id"] == row["rack_id"])
            & (past_for_baseline["ts_utc"] < row["ts_utc"])
        ].tail(6)
        if len(prev) > 0:
            baseline_pred_te.append(float(prev["total_power_kw"].mean()))
        else:
            baseline_pred_te.append(global_mean)

    baseline_pred_te = np.asarray(baseline_pred_te, dtype=float)

    metrics = {
        "naive_baseline": compute_metrics(yte, baseline_pred_te),
        "linear": compute_metrics(yte, lin_pred_te),
        "random_forest": compute_metrics(yte, rf_mean_te),
        "coverage_80": float(
            np.mean(
                (yte >= rf_mean_te - 1.28 * rf_std_te)
                & (yte <= rf_mean_te + 1.28 * rf_std_te)
            )
        ),
        "fallback_rate": float(np.mean((te["quality_score"].values < 0.75) | (conf_te < 0.55))),
    }

    test_eval = te[["ts_utc", "rack_id", "total_power_kw", "quality_score"]].copy()
    test_eval["baseline_total_power_kw"] = baseline_pred_te
    test_eval["linear_pred_total_power_kw"] = lin_pred_te
    test_eval["pred_total_power_kw"] = rf_mean_te
    test_eval["pred_std"] = rf_std_te
    test_eval["model_confidence"] = conf_te

    feature_importance = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    return {
        "linear": lin,
        "rf": rf,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "train_rows": int(len(tr)),
        "val_rows": int(len(va)),
        "test_rows": int(len(te)),
        "feature_importance": feature_importance,
        "test_eval": test_eval,
        "global_mean": global_mean,
    }


def predict_rows(models: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [
        "pred_total_power_kw",
        "pred_std",
        "pred_lower",
        "pred_upper",
        "model_confidence",
        "linear_pred_total_power_kw",
    ]

    if out.empty:
        for c in cols:
            out[c] = pd.Series(dtype=float)
        return out

    feat = add_features(out)[models["feature_cols"]]
    feat = feat.fillna(feat.median())
    X = feat.values.astype(float)

    tree_preds = np.vstack([est.predict(X) for est in models["rf"].estimators_])
    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0)
    lin = models["linear"].predict(X)
    conf = 1.0 - np.clip(std / (np.abs(mean) + 1e-6), 0.0, 1.0)

    out["pred_total_power_kw"] = mean
    out["pred_std"] = std
    out["pred_lower"] = mean - 1.28 * std
    out["pred_upper"] = mean + 1.28 * std
    out["model_confidence"] = conf
    out["linear_pred_total_power_kw"] = lin
    return out


def get_baseline_for_rows(models: Dict[str, Any], current_df: pd.DataFrame, past_df: pd.DataFrame) -> pd.DataFrame:
    out = current_df.copy()
    if out.empty:
        out["baseline_total_power_kw"] = pd.Series(dtype=float)
        return out

    baselines = []
    for _, row in out.iterrows():
        prev = past_df[
            (past_df["rack_id"] == row["rack_id"])
            & (past_df["ts_utc"] < row["ts_utc"])
        ].tail(6)
        if len(prev) > 0 and prev["total_power_kw"].notna().any():
            baselines.append(float(prev["total_power_kw"].mean()))
        else:
            baselines.append(float(models["global_mean"]))

    out["baseline_total_power_kw"] = baselines
    return out


def simulate_row(prev: pd.Series, ts: pd.Timestamp, controls: Dict[str, Any], scenario: str, tick: int, fault_until: int) -> Dict[str, Any]:
    rack_idx = RACK_INDEX.get(str(prev["rack_id"]), 0)
    rng = np.random.default_rng(SEED + tick * 131 + rack_idx * 17)

    hour = ts.hour
    dow = ts.dayofweek

    outdoor = float(prev["outdoor_temp_c"]) + float(rng.normal(0, 0.4))
    outdoor += 0.3 * math.sin(2 * math.pi * (hour - 14) / 24)
    outdoor = float(np.clip(outdoor, -5, 35))

    humidity = float(prev["relative_humidity_pct"]) + float(rng.normal(0, 1.5))
    humidity = float(np.clip(humidity, 20, 95))

    carbon = float(prev["carbon_intensity_gco2_per_kwh"]) + float(rng.normal(0, 10))
    carbon += 20 * math.sin(2 * math.pi * (hour - 8) / 24)
    if scenario == "high_carbon":
        carbon += 120
    carbon = float(np.clip(carbon, 50, 800))

    economizer_available = bool(outdoor < 18 and humidity < 85)

    base_it = float(prev["it_load_kw"]) if not pd.isna(prev["it_load_kw"]) else 6.0
    daily = 1 + 0.15 * math.sin(2 * math.pi * (hour - 13) / 24)
    weekly = 0.78 if dow >= 5 else 1.0
    noise = float(rng.normal(0, 0.12))

    it = base_it * daily * weekly * (1 + noise)

    if scenario == "thermal_stress":
        it *= 1.05

    deferred = 0.0
    if str(prev["workload_priority"]) == "batch":
        shift_pct = float(controls.get("batch_shift_pct", 0.0))
        if scenario == "high_carbon":
            shift_pct = max(shift_pct, 10.0)
        if shift_pct > 0:
            shifted = it * shift_pct / 100.0
            it -= shifted
            deferred = shifted

    it = float(np.clip(it, 0.8, 12.0))

    setpoint = float(controls.get("cooling_setpoint_c", 12.0))
    fan = float(controls.get("fan_speed_pct", 75.0))
    econ_bonus = 0.06 if (controls.get("economizer_auto", False) and economizer_available) else 0.0

    cooling = it * (
        0.30
        + 0.013 * max(0.0, outdoor - 16.0)
        - econ_bonus
        - 0.007 * (setpoint - 12.0)
        - 0.0015 * (fan - 70.0)
    )
    cooling += float(rng.normal(0, 0.15))
    cooling = float(max(0.07 * it, cooling))

    aux = 0.05 * it
    total = it + cooling + aux

    inlet = (
        setpoint
        + 3.6
        + 0.22 * (it / 10.0)
        - 1.4 * (1.0 if econ_bonus > 0 else 0.0)
        + 0.02 * (70.0 - fan)
        + float(rng.normal(0, 0.25))
    )
    if scenario == "thermal_stress":
        inlet += 1.8
    inlet = float(np.clip(inlet, 10.0, 35.0))

    pue = total / it if it > 0 else np.nan

    sensor_status = "ok"
    quality = 0.95

    fault_active = bool(fault_until and tick < fault_until)
    if fault_active and float(rng.random()) < 0.55:
        choice = float(rng.random())
        if choice < 0.40:
            sensor_status = "missing"
            it = np.nan
            cooling = np.nan
            total = np.nan
            pue = np.nan
            quality = 0.15
        elif choice < 0.70:
            sensor_status = "stale"
            total = float(prev["total_power_kw"])
            pue = float(prev["pue"])
            quality = 0.45
        else:
            sensor_status = "outlier"
            total = float(total) * 1.8
            pue = float(total) / float(it) if it and not pd.isna(it) else np.nan
            quality = 0.55

    return {
        "ts_utc": ts,
        "site_id": SITE_ID,
        "rack_id": str(prev["rack_id"]),
        "it_load_kw": it,
        "cooling_kw": cooling,
        "total_power_kw": total,
        "pue": pue,
        "outdoor_temp_c": outdoor,
        "economizer_available": economizer_available,
        "carbon_intensity_gco2_per_kwh": carbon,
        "inlet_temp_c": inlet,
        "relative_humidity_pct": humidity,
        "workload_priority": str(prev["workload_priority"]),
        "deferred_load_kw": deferred,
        "sensor_status": sensor_status,
        "quality_score": quality,
        "provenance": "simulated_live",
        "schema_version": SCHEMA_VERSION,
        "ingested_at": utc_now_ts(),
    }


def make_recommendations(state: Dict[str, Any], current_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if current_df.empty:
        return []

    df = current_df[current_df["total_power_kw"].notna()].copy()
    if df.empty:
        return []

    controls = state["controls"]
    elec = float(controls["electricity_cost_per_kwh"])
    carbon_price = float(controls["carbon_price_per_tco2"])
    max_inlet_limit = float(controls["max_inlet_temp_c"])

    total_power = float(df["total_power_kw"].sum())
    baseline_power = float(df["baseline_total_power_kw"].sum()) if "baseline_total_power_kw" in df else total_power
    pred_power = float(df["pred_total_power_kw"].sum()) if "pred_total_power_kw" in df else total_power
    total_cooling = float(df["cooling_kw"].sum())
    total_it = float(df["it_load_kw"].sum())
    avg_conf = float(df["model_confidence"].mean()) if "model_confidence" in df else 0.5
    avg_quality = float(df["quality_score"].mean())
    max_inlet = float(df["inlet_temp_c"].max())
    avg_inlet = float(df["inlet_temp_c"].mean())
    outdoor = float(df["outdoor_temp_c"].mean())
    carbon = float(df["carbon_intensity_gco2_per_kwh"].mean())
    batch_load = float(df.loc[df["workload_priority"] == "batch", "it_load_kw"].sum())
    fallback_any = bool(df["fallback_used"].any()) if "fallback_used" in df else False

    created = utc_now_iso()
    recs: List[Dict[str, Any]] = []

    def add_rec(
        action: str,
        scope: str,
        current_value: float,
        proposed_value: float,
        savings_kw: float,
        co2_kg: float,
        confidence: float,
        risk: float,
        evidence: Dict[str, Any],
        requires_approval: bool = True,
        status: str = "pending",
    ) -> None:
        cost_value = savings_kw * elec
        carbon_value = (co2_kg / 1000.0) * carbon_price
        score = cost_value + carbon_value - risk

        if action.startswith("Protect") or action.startswith("Hold"):
            score = max(score, 1.0)

        if score <= 0.05 and not action.startswith("Protect") and not action.startswith("Hold"):
            return

        rec = {
            "id": uuid.uuid4().hex[:10],
            "created_at": created,
            "scope": scope,
            "action": action,
            "status": status,
            "requires_approval": bool(requires_approval),
            "current_value": float(current_value),
            "proposed_value": float(proposed_value),
            "predicted_savings_kw": float(savings_kw),
            "predicted_co2_kg_per_hour": float(co2_kg),
            "score": float(score),
            "confidence": float(confidence),
            "evidence_json": json.dumps(to_jsonable(evidence)),
            "approver": None,
            "approved_at": None,
        }
        recs.append(rec)

    base_evidence = {
        "site_id": SITE_ID,
        "total_power_kw": total_power,
        "baseline_power_kw": baseline_power,
        "predicted_power_kw": pred_power,
        "total_cooling_kw": total_cooling,
        "total_it_kw": total_it,
        "avg_model_confidence": avg_conf,
        "avg_quality_score": avg_quality,
        "max_inlet_temp_c": max_inlet,
        "avg_inlet_temp_c": avg_inlet,
        "outdoor_temp_c": outdoor,
        "carbon_intensity_gco2_per_kwh": carbon,
        "fallback_active": fallback_any,
        "controls": controls,
    }

    if max_inlet >= max_inlet_limit - 0.2:
        add_rec(
            action="Protect: increase cooling airflow / lower setpoint",
            scope="site",
            current_value=max_inlet,
            proposed_value=max(10.0, float(controls["cooling_setpoint_c"]) - 1.0),
            savings_kw=0.0,
            co2_kg=0.0,
            confidence=avg_conf,
            risk=0.0,
            evidence={**base_evidence, "reason": "Thermal margin is too small for energy-saving actions."},
            requires_approval=True,
            status="pending",
        )
        return sorted(recs, key=lambda x: x["score"], reverse=True)[:5]

    if fallback_any or avg_quality < 0.75 or avg_conf < 0.55:
        add_rec(
            action="Hold: deterministic fallback",
            scope="site",
            current_value=total_power,
            proposed_value=baseline_power,
            savings_kw=0.0,
            co2_kg=0.0,
            confidence=avg_conf,
            risk=0.0,
            evidence={**base_evidence, "reason": "Input quality or model confidence is inadequate."},
            requires_approval=False,
            status="informational",
        )
        return sorted(recs, key=lambda x: x["score"], reverse=True)[:5]

    margin = max_inlet_limit - max_inlet

    if outdoor < 16.0 and not controls["economizer_auto"]:
        savings = 0.06 * total_cooling
        co2 = savings * carbon / 1000.0
        risk = (1.0 - avg_conf) * 5.0
        add_rec(
            action="Enable economizer/free cooling",
            scope="site",
            current_value=float(controls["economizer_auto"]),
            proposed_value=1.0,
            savings_kw=savings,
            co2_kg=co2,
            confidence=avg_conf,
            risk=risk,
            evidence={**base_evidence, "constraint": "Outdoor conditions support free cooling."},
        )

    if margin >= 2.0:
        proposed_setpoint = float(controls["cooling_setpoint_c"]) + 1.0
        predicted_inlet = max_inlet + 1.2
        if predicted_inlet <= max_inlet_limit - 0.5:
            savings = 0.035 * total_cooling
            co2 = savings * carbon / 1000.0
            risk = (1.0 - avg_conf) * 6.0
            add_rec(
                action="Raise chilled-water setpoint by 1 degC",
                scope="site",
                current_value=float(controls["cooling_setpoint_c"]),
                proposed_value=proposed_setpoint,
                savings_kw=savings,
                co2_kg=co2,
                confidence=avg_conf,
                risk=risk,
                evidence={
                    **base_evidence,
                    "predicted_max_inlet_temp_c": predicted_inlet,
                    "thermal_margin_degC": margin,
                },
            )

    if controls["fan_speed_pct"] > 60 and margin >= 1.5:
        proposed_fan = max(40.0, float(controls["fan_speed_pct"]) - 5.0)
        savings = 0.02 * total_cooling
        co2 = savings * carbon / 1000.0
        risk = (1.0 - avg_conf) * 5.0 + 1.0
        add_rec(
            action="Reduce fan speed by 5%",
            scope="site",
            current_value=float(controls["fan_speed_pct"]),
            proposed_value=proposed_fan,
            savings_kw=savings,
            co2_kg=co2,
            confidence=avg_conf,
            risk=risk,
            evidence={**base_evidence, "predicted_inlet_increase_degC": 0.8},
        )

    if carbon > 300 and batch_load > 20 and controls["batch_shift_cap"] > 0:
        shift_kw = min(batch_load * 0.10, float(controls["batch_shift_cap"]))
        future_carbon = max(120.0, carbon - 90.0)
        co2 = shift_kw * (carbon - future_carbon) / 1000.0
        risk = (1.0 - avg_conf) * 8.0 + 2.0
        add_rec(
            action="Shift batch workload to low-carbon window",
            scope="batch_racks",
            current_value=carbon,
            proposed_value=future_carbon,
            savings_kw=0.0,
            co2_kg=co2,
            confidence=avg_conf,
            risk=risk,
            evidence={
                **base_evidence,
                "shift_kw": shift_kw,
                "critical_load_protected": True,
                "sla_guardrail": controls["critical_sla_pct"],
            },
        )

    return sorted(recs, key=lambda x: x["score"], reverse=True)[:5]


def add_audit(state: Dict[str, Any], event_type: str, actor: str, details: str, rec_id: Optional[str] = None) -> None:
    event = {
        "ts_utc": utc_now_iso(),
        "event_type": event_type,
        "actor": actor,
        "rec_id": rec_id,
        "details": details,
    }
    state["audit"].insert(0, event)
    state["audit"] = state["audit"][:300]
    try:
        save_audit_event(state["db_conn"], event)
    except Exception:
        pass


def ensure_control_session_defaults() -> None:
    for name, key in CONTROL_KEY_MAP.items():
        if key not in st.session_state:
            st.session_state[key] = DEFAULT_CONTROLS[name]


def sync_controls() -> None:
    st.session_state["controls"] = {
        name: st.session_state[key] for name, key in CONTROL_KEY_MAP.items()
    }


def initialize_session_state() -> None:
    if "initialized" in st.session_state:
        return

    ensure_control_session_defaults()
    sync_controls()

    hist, quar, report = prepare_historical()
    models = get_models()
    conn = cached_db_conn()

    save_readings_to_db(conn, hist, ignore=True)

    st.session_state["initialized"] = True
    st.session_state["historical"] = hist
    st.session_state["quarantine"] = quar
    st.session_state["quality_report"] = report
    st.session_state["models"] = models
    st.session_state["live"] = empty_live_frame()
    st.session_state["current_ts"] = hist["ts_utc"].max() + pd.Timedelta(minutes=15)
    st.session_state["tick_count"] = 0
    st.session_state["running"] = True
    st.session_state["scenario"] = "normal"
    st.session_state["fault_until"] = 0
    st.session_state["recommendations"] = []
    st.session_state["audit"] = load_recent_audit(conn, 100)
    st.session_state["last_rec_tick"] = -99
    st.session_state["db_conn"] = conn

    add_audit(st.session_state, "session_initialized", "system", "Synthetic demo data loaded and simulation armed")


def advance_simulation(steps: int = 1) -> None:
    state = st.session_state
    if "initialized" not in state:
        return

    for _ in range(max(1, int(steps))):
        past = pd.concat([state["historical"], state["live"]], ignore_index=True)
        last_by_rack = past.sort_values("ts_utc").groupby("rack_id").tail(1)

        ts = state["current_ts"]
        rows = []
        for _, prev in last_by_rack.iterrows():
            rows.append(
                simulate_row(
                    prev=prev,
                    ts=ts,
                    controls=state["controls"],
                    scenario=state["scenario"],
                    tick=state["tick_count"],
                    fault_until=state["fault_until"],
                )
            )

        new_df = pd.DataFrame(rows)
        valid_new, quar_new, rep_new = validate_dataframe(new_df, "simulated_live")

        available_racks = set(valid_new["rack_id"]) if not valid_new.empty else set()
        fallback_rows = []
        for rack in set(new_df["rack_id"]) - available_racks:
            prev = last_by_rack[last_by_rack["rack_id"] == rack]
            if prev.empty:
                continue
            prev = prev.iloc[0]
            fb = {col: prev[col] for col in REQUIRED_COLUMNS}
            fb.update(
                {
                    "ts_utc": ts,
                    "provenance": "simulated_fallback",
                    "sensor_status": "fallback",
                    "quality_score": 0.35,
                    "schema_version": SCHEMA_VERSION,
                    "ingested_at": utc_now_ts(),
                }
            )
            fallback_rows.append(fb)

        if fallback_rows:
            valid_new = pd.concat([valid_new, pd.DataFrame(fallback_rows)], ignore_index=True)

        valid_new = get_baseline_for_rows(state["models"], valid_new, past)
        valid_new = predict_rows(state["models"], valid_new)
        valid_new["fallback_used"] = (
            (valid_new["quality_score"] < 0.75)
            | (valid_new["model_confidence"] < 0.55)
            | valid_new["sensor_status"].isin(["missing", "stale", "outlier", "fallback"])
        )

        state["live"] = pd.concat([state["live"], valid_new], ignore_index=True)
        if len(state["live"]) > 1800:
            state["live"] = (
                state["live"]
                .sort_values("ts_utc")
                .groupby("rack_id", group_keys=False)
                .tail(300)
                .reset_index(drop=True)
            )

        state["current_ts"] = ts + pd.Timedelta(minutes=15)
        state["tick_count"] += 1

        if state["tick_count"] % 4 == 0:
            save_readings_to_db(state["db_conn"], valid_new, ignore=True)

        if rep_new["quarantined_rows"] > 0:
            add_audit(state, "data_quality_alert", "system", f"{rep_new['quarantined_rows']} live rows quarantined")

        if bool(valid_new["fallback_used"].any()):
            add_audit(state, "fallback_triggered", "optimizer", "Deterministic fallback active for low-quality or low-confidence inputs")

        if state["tick_count"] - state["last_rec_tick"] >= 6:
            recs = make_recommendations(state, valid_new)
            if recs:
                state["recommendations"].extend(recs)
                for r in recs:
                    save_recommendation(state["db_conn"], r)
                state["last_rec_tick"] = state["tick_count"]
                add_audit(state, "recommendation_generated", "optimizer", f"{len(recs)} recommendations generated")

        if state["fault_until"] and state["tick_count"] >= state["fault_until"]:
            state["fault_until"] = 0
            add_audit(state, "fault_recovered", "system", "Sensor fault window ended")


def reset_simulation() -> None:
    state = st.session_state
    state["live"] = empty_live_frame()
    state["current_ts"] = state["historical"]["ts_utc"].max() + pd.Timedelta(minutes=15)
    state["tick_count"] = 0
    state["fault_until"] = 0
    state["recommendations"] = []
    state["last_rec_tick"] = -99
    state["running"] = True
    add_audit(state, "simulation_reset", "operator", "Live simulation reset to initial synthetic state")


def find_recommendation(rec_id: str) -> Optional[Dict[str, Any]]:
    for r in st.session_state["recommendations"]:
        if r["id"] == rec_id:
            return r
    return None


def approve_recommendation(rec_id: str) -> None:
    state = st.session_state
    rec = find_recommendation(rec_id)
    if rec is None or rec["status"] != "pending":
        return

    rec["status"] = "approved"
    rec["approver"] = "local-operator"
    rec["approved_at"] = utc_now_iso()

    action = rec["action"]
    if action.startswith("Raise"):
        st.session_state[CONTROL_KEY_MAP["cooling_setpoint_c"]] = min(18.0, float(st.session_state[CONTROL_KEY_MAP["cooling_setpoint_c"]]) + 1.0)
    elif action.startswith("Reduce fan"):
        st.session_state[CONTROL_KEY_MAP["fan_speed_pct"]] = max(40.0, float(st.session_state[CONTROL_KEY_MAP["fan_speed_pct"]]) - 5.0)
    elif action.startswith("Enable economizer"):
        st.session_state[CONTROL_KEY_MAP["economizer_auto"]] = True
    elif action.startswith("Shift batch"):
        cap = float(st.session_state[CONTROL_KEY_MAP["batch_shift_cap"]])
        st.session_state[CONTROL_KEY_MAP["batch_shift_pct"]] = min(cap, float(st.session_state[CONTROL_KEY_MAP["batch_shift_pct"]]) + 10.0)
    elif action.startswith("Protect"):
        st.session_state[CONTROL_KEY_MAP["cooling_setpoint_c"]] = max(8.0, float(st.session_state[CONTROL_KEY_MAP["cooling_setpoint_c"]]) - 1.0)
        st.session_state[CONTROL_KEY_MAP["fan_speed_pct"]] = min(100.0, float(st.session_state[CONTROL_KEY_MAP["fan_speed_pct"]]) + 5.0)

    sync_controls()
    save_recommendation(state["db_conn"], rec)
    add_audit(state, "recommendation_approved", "local-operator", f"Approved: {action}", rec_id=rec_id)


def reject_recommendation(rec_id: str) -> None:
    state = st.session_state
    rec = find_recommendation(rec_id)
    if rec is None or rec["status"] != "pending":
        return

    rec["status"] = "rejected"
    rec["approver"] = "local-operator"
    rec["approved_at"] = utc_now_iso()
    save_recommendation(state["db_conn"], rec)
    add_audit(state, "recommendation_rejected", "local-operator", f"Rejected: {rec['action']}", rec_id=rec_id)


@st.cache_data(show_spinner="Generating synthetic live demo video (MP4)...")
def ensure_demo_video() -> str:
    VIDEO_DIR.mkdir(exist_ok=True)

    if VIDEO_PATH.exists() and VIDEO_PATH.stat().st_size > 1000:
        return str(VIDEO_PATH)
    if VIDEO_FALLBACK_PATH.exists() and VIDEO_FALLBACK_PATH.stat().st_size > 1000:
        return str(VIDEO_FALLBACK_PATH)

    width, height, fps, frames = 1280, 720, 10, 150

    writer = None
    output_path = VIDEO_PATH
    for codec in ("avc1", "mp4v"):
        trial = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if trial.isOpened():
            writer = trial
            output_path = VIDEO_PATH
            break
        trial.release()

    if writer is None:
        output_path = VIDEO_FALLBACK_PATH
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    if writer is None or not writer.isOpened():
        raise RuntimeError("OpenCV could not open a video writer. Install opencv-python and retry.")

    rng = np.random.default_rng(7)
    t = np.arange(frames)
    hours = t * 0.25

    it = 800 + 120 * np.sin(2 * np.pi * ((hours % 24) - 13) / 24) + rng.normal(0, 8, frames)
    outdoor = 14 + 7 * np.sin(2 * np.pi * (((hours + 8) % 24) - 14) / 24) + rng.normal(0, 0.5, frames)
    carbon = 280 + 130 * np.sin(2 * np.pi * (((hours + 6) % 24) - 9) / 24) + rng.normal(0, 12, frames)

    fault_start, fault_end = 55, 80
    high_start, high_end = 90, 125

    carbon[high_start:high_end] += 160

    econ = (outdoor < 17).astype(float)
    cooling = it * (0.30 + 0.013 * np.maximum(0, outdoor - 16) - 0.05 * econ) + rng.normal(0, 5, frames)
    cooling = np.maximum(0.08 * it, cooling)
    total = it + cooling + 0.05 * it
    pue = total / it
    inlet = 16 + 0.004 * it + 0.08 * np.maximum(0, outdoor - 16) - 1.5 * econ + rng.normal(0, 0.2, frames)

    baseline = pd.Series(total).rolling(6, min_periods=1).mean().shift(1).fillna(float(total.mean())).to_numpy()
    pred = total + rng.normal(0, 6, frames)
    std = np.full(frames, 5.0)
    std[fault_start:fault_end] = 25.0
    conf = 1.0 - np.clip(std / (np.abs(pred) + 1e-6), 0.0, 1.0)

    observed = total.copy()
    observed[fault_start:fault_end:3] = np.nan

    quality = np.full(frames, 0.95)
    quality[fault_start:fault_end] = 0.45

    rec_text = ["Monitoring normal operation"] * frames
    for i in range(fault_start, fault_end):
        rec_text[i] = "Fallback: hold setpoints; sensor quality low"
    for i in range(fault_end, high_start):
        rec_text[i] = "Recovered: resume optimization"
    for i in range(high_start, high_end):
        rec_text[i] = "Proposed: shift batch workload to low-carbon window"
    for i in range(high_end, frames):
        rec_text[i] = "Protected: thermal margin verified; economizer enabled"

    for i in range(frames):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.35, wspace=0.2)

        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(i + 1)
        ax1.plot(x, observed[: i + 1], label="Observed total power (simulated)", color="#1f77b4")
        ax1.plot(x, baseline[: i + 1], label="Baseline estimate", color="#ff7f0e", linestyle="--")
        ax1.plot(x, pred[: i + 1], label="Model prediction", color="#2ca02c", alpha=0.85)
        ax1.fill_between(
            x,
            pred[: i + 1] - 1.28 * std[: i + 1],
            pred[: i + 1] + 1.28 * std[: i + 1],
            color="#2ca02c",
            alpha=0.12,
            label="80% uncertainty",
        )
        ax1.set_ylabel("Site power (kW)")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.2)
        ax1.set_title("AI Data-Center Energy Optimizer — Synthetic Live Demo", fontsize=14)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(x, pue[: i + 1], color="#9467bd")
        ax2.set_ylabel("PUE (simulated)")
        ax2.grid(alpha=0.2)

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(x, inlet[: i + 1], color="#8c564b")
        ax3.axhline(27, color="red", linestyle=":", label="Thermal guardrail")
        ax3.set_ylabel("Avg inlet temp (degC)")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.2)

        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(x, carbon[: i + 1], color="#17becf")
        ax4.set_ylabel("Carbon intensity (gCO2e/kWh)")
        ax4.grid(alpha=0.2)

        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        if fault_start <= i < fault_end:
            status = "FAULT / FALLBACK"
        elif high_start <= i < high_end:
            status = "HIGH CARBON"
        elif fault_end <= i < high_start:
            status = "RECOVERY"
        else:
            status = "NORMAL"

        ax5.text(0.05, 0.85, f"Scenario: {status}", fontsize=12, weight="bold")
        ax5.text(0.05, 0.65, f"Action: {rec_text[i]}", fontsize=10)
        ax5.text(0.05, 0.45, f"Data quality: {quality[i]:.2f} | Model confidence: {conf[i]:.2f}", fontsize=10)
        ax5.text(0.05, 0.25, f"Time: {(i * 15) // 60}h {(i * 15) % 60:02d}m | Tick {i + 1}/{frames}", fontsize=10)
        ax5.text(0.05, 0.05, "All values are synthetic/simulated for demo; no customer data.", fontsize=8, color="gray")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        img = np.frombuffer(buf.read(), dtype=np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    return str(output_path)


def run_self_checks() -> pd.DataFrame:
    results = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        results.append({"Check": name, "Pass": bool(passed), "Detail": detail})

    try:
        hist, quar, rep = prepare_historical()
        add("Historical data loaded", len(hist) > 0, f"{len(hist)} valid rows, {len(quar)} quarantined")
        add("No NaN in valid critical fields", not hist[NUMERIC_COLUMNS].isna().any().any(), "")
        add(
            "Duplicates removed from valid set",
            not hist.duplicated(subset=["site_id", "rack_id", "ts_utc"]).any(),
            f"{rep['duplicate_rows']} duplicate rows quarantined",
        )

        models = get_models()
        rf_m = models["metrics"]["random_forest"]
        add("Model metrics finite", all(math.isfinite(v) for v in [rf_m["MAE"], rf_m["RMSE"], rf_m["R2"]]), "")
        add("Test evaluation available", len(models["test_eval"]) > 0, f"{len(models['test_eval'])} test rows")

        temp_state = {"controls": st.session_state["controls"]}
        tail = hist.tail(12).copy()
        tail = get_baseline_for_rows(models, tail, hist)
        tail = predict_rows(models, tail)
        tail["fallback_used"] = False
        recs = make_recommendations(temp_state, tail)
        actionable = [r for r in recs if r["requires_approval"]]
        add("Actionable recommendations require approval", all(r["requires_approval"] for r in actionable), f"{len(actionable)} actionable")

        thermal_ok = True
        max_limit = float(st.session_state["controls"]["max_inlet_temp_c"])
        for r in recs:
            ev = json.loads(r["evidence_json"])
            if r["action"].startswith("Raise") and ev.get("predicted_max_inlet_temp_c", 99) > max_limit - 0.5:
                thermal_ok = False
        add("No recommendation violates thermal guardrail", thermal_ok, "")

        vp = ensure_demo_video()
        add("Demo video generated", Path(vp).exists() and Path(vp).stat().st_size > 1000, vp)

    except Exception as exc:
        add("Self-check exception", False, str(exc))

    return pd.DataFrame(results)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()

with st.sidebar:
    st.title("⚡ AI Data-Center Energy Optimizer")
    st.caption("CPU-first, synthetic-data prototype. No API keys.")

    ensure_control_session_defaults()

    st.subheader("Simulation")
    c1, c2 = st.columns(2)
    if c1.button("Pause" if st.session_state["running"] else "Resume", use_container_width=True):
        st.session_state["running"] = not st.session_state["running"]
    if c2.button("Step", use_container_width=True):
        try:
            advance_simulation(1)
        except Exception as exc:
            st.error(f"Step failed: {exc}")

    if st.button("Advance 10 ticks", use_container_width=True):
        try:
            advance_simulation(10)
        except Exception as exc:
            st.error(f"Batch step failed: {exc}")

    if st.button("Reset simulation", use_container_width=True):
        reset_simulation()

    scenario_options = ["normal", "sensor_fault", "high_carbon", "thermal_stress", "grid_price_peak"]
    scenario_index = scenario_options.index(st.session_state["scenario"]) if st.session_state["scenario"] in scenario_options else 0
    st.session_state["scenario"] = st.selectbox("Scenario", scenario_options, index=scenario_index, key="scenario_select")

    if st.button("Inject sensor fault for 6 ticks", use_container_width=True):
        st.session_state["fault_until"] = st.session_state["tick_count"] + 6
        st.session_state["scenario"] = "sensor_fault"

    st.subheader("Operational assumptions")

    st.number_input(
        "Electricity cost ($/kWh)",
        min_value=0.02,
        max_value=0.50,
        step=0.005,
        key=CONTROL_KEY_MAP["electricity_cost_per_kwh"],
    )
    st.number_input(
        "Carbon price ($/tCO2e)",
        min_value=0.0,
        max_value=250.0,
        step=5.0,
        key=CONTROL_KEY_MAP["carbon_price_per_tco2"],
    )
    st.slider(
        "Max inlet temp guardrail (degC)",
        min_value=24.0,
        max_value=32.0,
        step=0.5,
        key=CONTROL_KEY_MAP["max_inlet_temp_c"],
    )
    st.slider(
        "Cooling setpoint (degC)",
        min_value=8.0,
        max_value=18.0,
        step=0.5,
        key=CONTROL_KEY_MAP["cooling_setpoint_c"],
    )
    st.slider(
        "Fan speed (%)",
        min_value=40.0,
        max_value=100.0,
        step=1.0,
        key=CONTROL_KEY_MAP["fan_speed_pct"],
    )
    st.checkbox("Economizer auto", key=CONTROL_KEY_MAP["economizer_auto"])
    st.slider(
        "Batch shift (%)",
        min_value=0.0,
        max_value=30.0,
        step=1.0,
        key=CONTROL_KEY_MAP["batch_shift_pct"],
    )
    st.slider(
        "Batch shift cap (%)",
        min_value=0.0,
        max_value=30.0,
        step=1.0,
        key=CONTROL_KEY_MAP["batch_shift_cap"],
    )
    st.checkbox(
        "Require human approval for consequential actions",
        key=CONTROL_KEY_MAP["human_approval_required"],
        disabled=True,
        help="Production guardrail: setpoint, fan, economizer, and workload-shift actions require explicit approval.",
    )

    sync_controls()

    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    st.caption(f"Local resources: CPU {cpu:.0f}% | RAM {mem:.0f}% | DB {DB_PATH}")

if hasattr(st, "fragment"):
    try:
        @st.fragment(run_every=2.0)
        def _live_fragment():
            try:
                if st.session_state.get("running", False):
                    advance_simulation(1)
                st.caption(
                    f"Auto simulation tick: {st.session_state['tick_count']} | "
                    f"Next UTC: {st.session_state['current_ts']:%Y-%m-%d %H:%M} | "
                    f"Running: {st.session_state['running']}"
                )
            except Exception as exc:
                st.session_state["running"] = False
                st.caption(f"Simulation paused after error: {exc}")

        _live_fragment()
    except Exception:
        st.caption("Auto-refresh unavailable. Use Step buttons.")
else:
    st.caption(
        f"Simulation tick: {st.session_state['tick_count']} | "
        f"Next UTC: {st.session_state['current_ts']:%Y-%m-%d %H:%M} | "
        f"Running: {st.session_state['running']}. Use Step buttons."
    )

tabs = st.tabs(
    [
        "🎬 Live Demo Video",
        "📊 Operations",
        "🧪 Simulator",
        "📘 Product Spec",
        "🗂️ Data Contract",
        "🤖 Models & Evaluation",
        "✅ Recommendations & Audit",
        "💼 Business Value",
        "🛡️ Risk & Governance",
        "🏗️ Architecture",
    ]
)

with tabs[0]:
    st.header("Synthetic live demo video")
    st.caption("This video is generated locally from synthetic data. It demonstrates normal operation, sensor fault/fallback, high-carbon workload shifting, and recovery.")

    try:
        video_path = ensure_demo_video()
        st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button(
                "Download demo video",
                data=f.read(),
                file_name=Path(video_path).name,
                mime="video/mp4" if Path(video_path).suffix.lower() == ".mp4" else "video/x-msvideo",
            )
        st.success(f"Video file ready: {video_path}")
    except Exception as exc:
        st.error(f"Video generation failed: {exc}")
        st.info("On Windows, ensure opencv-python is installed and Microsoft Visual C++ redistributable is present.")

    st.markdown(
        """
        **Labeling convention**
        - **Observed/Simulated**: synthetic sensor-like values generated by the simulator.
        - **Baseline estimate**: transparent rolling historical baseline.
        - **Model prediction**: random-forest prediction with uncertainty band.
        - **Estimated/Proposed**: optimizer recommendation, not applied until human approval.
        """
    )

with tabs[1]:
    st.header("Operations dashboard")

    models = st.session_state["models"]
    hist_tail = st.session_state["historical"].tail(168).copy()
    hist_tail = get_baseline_for_rows(models, hist_tail, st.session_state["historical"])
    hist_tail = predict_rows(models, hist_tail)
    hist_tail["fallback_used"] = False

    live = st.session_state["live"]
    chart_df = pd.concat([hist_tail, live], ignore_index=True).sort_values("ts_utc").tail(500)

    if not live.empty:
        latest_ts = live["ts_utc"].max()
        latest_df = live[live["ts_utc"] == latest_ts]
    else:
        latest_ts = hist_tail["ts_utc"].max()
        latest_df = hist_tail[hist_tail["ts_utc"] == latest_ts]

    total_power = float(latest_df["total_power_kw"].sum())
    baseline_power = float(latest_df["baseline_total_power_kw"].sum())
    pred_power = float(latest_df["pred_total_power_kw"].sum())
    total_it = max(float(latest_df["it_load_kw"].sum()), 1e-6)
    pue = total_power / total_it
    carbon = float(latest_df["carbon_intensity_gco2_per_kwh"].mean())
    conf = float(latest_df["model_confidence"].mean())
    quality = float(latest_df["quality_score"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Site power", f"{total_power:.1f} kW", f"{total_power - baseline_power:+.1f} kW vs baseline")
    c2.metric("PUE", f"{pue:.3f}")
    c3.metric("Carbon intensity", f"{carbon:.0f} gCO2e/kWh")
    c4.metric("Model confidence", f"{conf:.2f}")
    c5.metric("Data quality", f"{quality:.2f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["ts_utc"],
            y=chart_df["total_power_kw"],
            name="Observed/Simulated",
            mode="lines",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["ts_utc"],
            y=chart_df["baseline_total_power_kw"],
            name="Baseline estimate",
            mode="lines",
            line=dict(color="#ff7f0e", dash="dash", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["ts_utc"],
            y=chart_df["pred_total_power_kw"],
            name="ML prediction",
            mode="lines",
            line=dict(color="#2ca02c", width=2),
        )
    )

    unc = chart_df.dropna(subset=["pred_lower", "pred_upper"])
    if not unc.empty:
        fig.add_trace(
            go.Scatter(
                x=unc["ts_utc"],
                y=unc["pred_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=unc["ts_utc"],
                y=unc["pred_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(44,160,44,0.14)",
                name="80% uncertainty",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Site power: observed, baseline, and predicted",
        xaxis_title="UTC time",
        yaxis_title="kW",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest rack telemetry")
    table_cols = [
        "ts_utc",
        "rack_id",
        "workload_priority",
        "it_load_kw",
        "cooling_kw",
        "total_power_kw",
        "baseline_total_power_kw",
        "pred_total_power_kw",
        "model_confidence",
        "inlet_temp_c",
        "sensor_status",
        "quality_score",
        "fallback_used",
    ]
    available_cols = [c for c in table_cols if c in latest_df.columns]
    st.dataframe(latest_df[available_cols], use_container_width=True, hide_index=True)

    st.subheader("Recent alerts and fallback events")
    audit_df = pd.DataFrame(st.session_state["audit"][:30])
    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No audit events yet.")

with tabs[2]:
    st.header("Simulator")

    st.markdown(
        """
        This simulator generates rack-level energy, thermal, and carbon telemetry.
        It is intentionally simple, deterministic where possible, and labeled as synthetic.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Tick", st.session_state["tick_count"])
    c2.metric("Next timestamp", f"{st.session_state['current_ts']:%Y-%m-%d %H:%M} UTC")
    c3.metric("Scenario", st.session_state["scenario"])

    st.subheader("Live rows")
    if st.session_state["live"].empty:
        st.info("No live rows yet. The simulation is arming; use Step if auto-refresh is unavailable.")
    else:
        live_tail = st.session_state["live"].tail(36)
        st.dataframe(live_tail, use_container_width=True, hide_index=True)

    st.subheader("Fault and recovery cases")
    st.markdown(
        """
        - **Sensor fault**: inject missing, stale, or outlier rows. Expected: quarantine, fallback, Hold recommendation.
        - **High carbon**: carbon intensity rises. Expected: batch-shift recommendation, critical load protected.
        - **Thermal stress**: inlet temperature approaches guardrail. Expected: protective action, no setpoint raise.
        """
    )

with tabs[3]:
    st.header("Product specification")

    st.markdown(
        """
        ### Executive summary
        **AI Data-Center Energy Optimizer** helps data-center energy and operations teams reduce electricity and carbon consumption without violating thermal or SLA guardrails. It ingests rack-level power, cooling, environmental, and grid-carbon telemetry; establishes a transparent baseline; predicts total power; and proposes reversible, explainable actions such as raising chilled-water setpoint, enabling economizer, reducing fan speed, or shifting batch workload.

        A large technology company would pay for this because data-center energy is a major operating expense and sustainability constraint. Even a 2-5% reduction in cooling-related power at fleet scale can represent millions of dollars and thousands of tonnes of CO2e avoided, while improving capacity headroom and grid-resilience reporting.

        ### Primary users
        - **Data-center sustainability engineer**: monitors PUE, carbon intensity, and savings opportunities.
        - **DC operations manager**: responds to thermal alarms, sensor faults, and equipment constraints.
        - **Energy procurement analyst**: evaluates time-of-use and carbon-aware workload shifting.
        - **AI/ML platform engineer**: validates model quality, fallback behavior, and auditability.

        ### Decisions improved
        - Whether to raise chilled-water setpoint by 1 degC.
        - Whether to enable or verify economizer/free cooling.
        - Whether to reduce fan speed within thermal margin.
        - Whether to shift batch workload to a lower-carbon or lower-price window.
        - Whether to hold due to poor data quality or model uncertainty.

        ### Realistic user journeys
        1. **Morning sustainability review**
           - Legacy workflow: engineer opens spreadsheets, DCIM exports, and utility invoices; manually compares yesterday’s PUE and carbon.
           - Improved workflow: dashboard shows baseline vs predicted power, uncertainty, data quality, and ranked recommendations.
           - Decision: approve a safe setpoint raise or schedule batch shift.

        2. **Sensor outage during peak cooling**
           - Legacy workflow: operator chases missing telemetry, manually checks rack temperatures, and delays action.
           - Improved workflow: ingestion quarantines bad rows, fallback activates, dashboard alerts, and protective recommendation is pending approval.
           - Decision: approve protective cooling adjustment or hold until sensors recover.

        3. **Carbon-aware batch scheduling**
           - Legacy workflow: analyst reviews grid carbon forecast in a separate tool and emails operations.
           - Improved workflow: optimizer detects high carbon intensity, estimates shifted batch carbon reduction, protects critical SLA, and requests approval.
           - Decision: approve 10% batch shift for the next low-carbon window.

        ### Functional requirements
        - FR1: Load synthetic historical telemetry with canonical schema and provenance.
        - FR2: Validate rows for missing values, duplicates, outliers, late arrivals, and schema mismatches.
        - FR3: Maintain a transparent rolling baseline before ML.
        - FR4: Train and serve a CPU-friendly prediction model for total power.
        - FR5: Report prediction uncertainty and model confidence.
        - FR6: Generate ranked, explainable recommendations with guardrails.
        - FR7: Require human approval for consequential control actions.
        - FR8: Persist recommendations and audit events locally.
        - FR9: Provide a dashboard with observed, estimated, simulated, and predicted labels.
        - FR10: Generate a local demo video from synthetic data on first run.

        ### Non-functional requirements
        - NFR1: Runs on Windows 10/11, CPU-only, no paid API keys.
        - NFR2: Core demo startup under a few minutes on a typical developer laptop.
        - NFR3: Reproducible synthetic data using fixed seed and schema version.
        - NFR4: Deterministic fallback when data quality or model confidence is inadequate.
        - NFR5: Audit trail for recommendations, approvals, rejections, and fallbacks.
        - NFR6: Least-privilege local storage; no secrets in code.
        - NFR7: Modular code boundaries suitable later for service extraction.

        ### Assumptions
        - Initial demo uses synthetic data only.
        - Rack-level power and inlet temperature are available at 15-minute or hourly resolution.
        - Carbon intensity can be approximated from grid signals or synthetic forecasts.
        - Consequential actions are not automatically applied in the prototype.

        ### Dependencies
        - Python packages listed in requirements.txt.
        - Local SQLite database for audit and recommendation persistence.
        - OpenCV for local MP4/AVI demo video generation.

        ### Out of scope for prototype
        - Direct BMS/DCIM write access.
        - Multi-tenant SaaS billing.
        - Real-time hardware control loops.
        - Water usage optimization beyond energy/carbon reporting.
        - Regulatory filing automation.

        ### Release milestones
        - M1: Synthetic data + schema validation + baseline.
        - M2: ML prediction + uncertainty + fallback.
        - M3: Recommendation engine + approval workflow + audit.
        - M4: Dashboard + demo video + business-value calculator.
        - M5: Packaging, tests, docs, and production-readiness gap analysis.

        ### Prototype vs production enterprise service
        - **Prototype**: single-file Streamlit app, SQLite, synthetic data, CPU models, local approval buttons, demo video.
        - **Production**: authenticated multi-site ingestion, stream/batch pipelines, model registry, policy engine, BMS integration with safety certification, RBAC, SSO, HA database, observability, change management, and regulated audit retention.

        ### Prioritized backlog
        | Priority | Item | Why |
        |---:|---|---|
        | P0 | Data contract and validation | Prevents garbage-in recommendations |
        | P0 | Transparent baseline | Provides trustworthy comparison |
        | P0 | Thermal guardrails | Avoids operational harm |
        | P1 | Prediction uncertainty | Enables fallback |
        | P1 | Human approval workflow | Governance |
        | P1 | Audit history | Traceability |
        | P2 | Carbon-aware shifting | Sustainability value |
        | P2 | Business-value calculator | Adoption |
        | P3 | Multi-site federation | Enterprise scale |

        ### User stories with acceptance criteria
        - **US1**: As a sustainability engineer, I want to see baseline vs predicted power, so I can identify savings.  
          **AC**: Dashboard shows observed, baseline, predicted, uncertainty, and data quality.
        - **US2**: As an operations manager, I want the system to hold during sensor faults.  
          **AC**: Injected fault produces quarantine, fallback_used=True, and Hold/Protect recommendation.
        - **US3**: As an operator, I want to approve only safe setpoint changes.  
          **AC**: Raise-setpoint recommendation appears only when predicted inlet remains below guardrail.
        - **US4**: As an auditor, I want to inspect evidence behind each recommendation.  
          **AC**: Each recommendation includes JSON evidence and is stored in audit history.

        ### Glossary
        - **PUE**: Power Usage Effectiveness = total facility power / IT power.
        - **CHWS**: Chilled-water supply temperature.
        - **Economizer**: Free-cooling mode using favorable outdoor air.
        - **Baseline**: Transparent expected power from historical behavior.
        - **Fallback**: Deterministic safe behavior when model/input quality is inadequate.
        - **Guardrail**: Hard operational limit such as max inlet temperature or SLA.
        """
    )

    st.subheader("Self-checks")
    if st.button("Run embedded smoke checks"):
        with st.spinner("Running smoke checks..."):
            checks = run_self_checks()
        st.dataframe(checks, use_container_width=True, hide_index=True)
        if checks["Pass"].all():
            st.success("All embedded smoke checks passed.")
        else:
            st.warning("Some checks failed. Review details above.")

with tabs[4]:
    st.header("Data contract and quality")

    st.markdown(
        """
        The canonical model is rack-level, timestamped, and provenance-tagged.
        All demo records are synthetic. No personal data is used.
        """
    )

    st.subheader("Schema")
    st.dataframe(pd.DataFrame(DATA_CONTRACT), use_container_width=True, hide_index=True)

    st.subheader("Pydantic model schema")
    try:
        st.json(to_jsonable(Reading.model_json_schema()))
    except Exception as exc:
        st.warning(f"Could not render Pydantic schema: {exc}")

    st.subheader("Example valid record")
    sample_valid = st.session_state["historical"].head(1).to_dict(orient="records")
    st.json(to_jsonable(sample_valid))

    st.subheader("Example quarantined record")
    if not st.session_state["quarantine"].empty:
        sample_quar = st.session_state["quarantine"].head(1).to_dict(orient="records")
        st.json(to_jsonable(sample_quar))
    else:
        st.info("No quarantined rows in current synthetic batch.")

    st.subheader("Handling rules")
    st.markdown(
        """
        - **Missing values**: critical numeric NaNs are quarantined; fallback uses last valid row.
        - **Duplicates**: same site/rack/timestamp keeps latest ingested record; others quarantined.
        - **Late arrivals**: accepted but quality score reduced if ingestion is more than 2 hours after event time.
        - **Malformed input**: invalid enums, negative power, non-finite values, or missing columns are quarantined.
        - **Outliers**: implausible rack power or PUE is quarantined and flagged.
        - **Schema changes**: unknown columns are ignored in prototype; required-column drift quarantines rows.
        """
    )

    st.subheader("Quality report")
    st.json(to_jsonable(st.session_state["quality_report"]))

    st.subheader("Acceptance checks")
    st.markdown(
        """
        - Valid rows contain no NaN in critical numeric fields.
        - Duplicate timestamps do not appear in valid set.
        - Quarantined rows retain validation_errors for inspection.
        - Every row has provenance and schema_version.
        """
    )

with tabs[5]:
    st.header("Models, baselines, and evaluation")

    models = st.session_state["models"]
    metrics = models["metrics"]

    st.markdown(
        """
        ### Transparent baseline before ML
        - **Naive rolling baseline**: previous 6 observations per rack.
        - **Linear regression**: interpretable parametric model.
        - **Random forest**: nonlinear ensemble with tree-level uncertainty.

        The prototype compares all three. The random forest is used for prediction, but the rolling baseline remains the fallback anchor.
        """
    )

    metrics_df = pd.DataFrame(
        [
            {
                "Algorithm": "Naive rolling baseline",
                "MAE": metrics["naive_baseline"]["MAE"],
                "RMSE": metrics["naive_baseline"]["RMSE"],
                "R2": metrics["naive_baseline"]["R2"],
            },
            {
                "Algorithm": "Linear regression",
                "MAE": metrics["linear"]["MAE"],
                "RMSE": metrics["linear"]["RMSE"],
                "R2": metrics["linear"]["R2"],
            },
            {
                "Algorithm": "Random forest",
                "MAE": metrics["random_forest"]["MAE"],
                "RMSE": metrics["random_forest"]["RMSE"],
                "R2": metrics["random_forest"]["R2"],
            },
        ]
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("80% interval coverage", f"{metrics['coverage_80']:.2f}")
    c2.metric("Historical fallback rate", f"{metrics['fallback_rate']:.2f}")

    st.markdown(
        """
        ### Metric formulas
        - **MAE** = mean(|y - y_hat|)
        - **RMSE** = sqrt(mean((y - y_hat)^2))
        - **R2** = 1 - SSE / SST
        - **Coverage** = fraction of true values inside predicted mean ± 1.28 * ensemble std
        - **Confidence** = 1 - clip(std / (abs(mean) + epsilon), 0, 1)

        These metrics describe held-out synthetic performance. They do not prove real-world energy savings without metered controlled trials.
        """
    )

    st.subheader("Actual vs predicted on chronological test split")
    te = models["test_eval"]
    min_v = float(min(te["total_power_kw"].min(), te["pred_total_power_kw"].min()))
    max_v = float(max(te["total_power_kw"].max(), te["pred_total_power_kw"].max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="Perfect prediction",
            line=dict(dash="dot", color="gray"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=te["total_power_kw"],
            y=te["pred_total_power_kw"],
            mode="markers",
            name="Random forest",
            marker=dict(size=8, color="#2ca02c"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=te["total_power_kw"],
            y=te["baseline_total_power_kw"],
            mode="markers",
            name="Baseline",
            marker=dict(size=6, color="#ff7f0e", symbol="x"),
        )
    )
    fig.update_layout(
        xaxis_title="Observed total power (kW)",
        yaxis_title="Predicted total power (kW)",
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance")
    fi = models["feature_importance"].reset_index()
    fi.columns = ["Feature", "Importance"]
    fig2 = px.bar(fi, x="Feature", y="Importance", title="Random forest feature importance")
    fig2.update_layout(height=320, margin=dict(l=40, r=20, t=50, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Uncertainty and deterministic fallback")
    st.markdown(
        """
        Fallback triggers when:
        - row quality_score < 0.75, or
        - model_confidence < 0.55, or
        - sensor_status is missing/stale/outlier/fallback.

        Fallback behavior:
        - use rolling baseline as expected power,
        - suppress energy-saving recommendations,
        - emit Hold or Protect action,
        - log audit event.
        """
    )

    st.subheader("Controlled evaluation design")
    st.markdown(
        """
        To establish incremental value:
        1. Select comparable racks or time windows.
        2. Use metered historical baseline and weather/carbo-normalization.
        3. Randomize recommendation exposure where operationally safe.
        4. Measure kWh saved, CO2e avoided, PUE delta, thermal violations, SLA incidents, and operator acceptance.
        5. Report confidence intervals and subgroup performance by rack role and priority.

        A metric such as synthetic MAE cannot prove commercial savings. Real value requires controlled, metered evidence.
        """
    )

with tabs[6]:
    st.header("Recommendations and audit history")

    pending = [r for r in st.session_state["recommendations"] if r["status"] == "pending"]
    history = [r for r in st.session_state["recommendations"] if r["status"] != "pending"]

    st.subheader("Pending recommendations")
    if not pending:
        st.info("No pending recommendations. Advance the simulation or inject a scenario.")
    else:
        for rec in pending[:8]:
            with st.expander(f"{rec['action']} | score {rec['score']:.2f} | confidence {rec['confidence']:.2f}"):
                st.write(f"**Created:** {rec['created_at']}")
                st.write(f"**Scope:** {rec['scope']}")
                st.write(f"**Current value:** {rec['current_value']:.3f}")
                st.write(f"**Proposed value:** {rec['proposed_value']:.3f}")
                st.write(f"**Predicted savings:** {rec['predicted_savings_kw']:.3f} kW")
                st.write(f"**Predicted CO2 reduction:** {rec['predicted_co2_kg_per_hour']:.3f} kg/h")
                st.write("**Requires human approval:** Yes" if rec["requires_approval"] else "**Requires human approval:** No")

                try:
                    evidence = json.loads(rec["evidence_json"])
                except Exception:
                    evidence = {"error": "evidence not parseable"}
                st.json(to_jsonable(evidence))

                bc1, bc2 = st.columns(2)
                if bc1.button("Approve", key=f"approve_{rec['id']}", use_container_width=True):
                    approve_recommendation(rec["id"])
                    st.rerun()
                if bc2.button("Reject", key=f"reject_{rec['id']}", use_container_width=True):
                    reject_recommendation(rec["id"])
                    st.rerun()

    st.subheader("Recommendation history")
    if history:
        hist_df = pd.DataFrame(history)[
            ["created_at", "action", "status", "score", "confidence", "approver", "approved_at"]
        ]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No approved or rejected recommendations yet.")

    st.subheader("Audit log")
    audit_df = pd.DataFrame(st.session_state["audit"])
    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No audit events yet.")

    st.markdown(
        """
        ### Actions requiring human approval
        - Raise chilled-water setpoint.
        - Lower chilled-water setpoint for protection.
        - Reduce or increase fan speed.
        - Enable economizer/free cooling.
        - Shift batch workload.
        - Any action that changes physical or logical operating setpoints.

        Safe failure mode: if model or data quality is inadequate, the system holds and alerts rather than actuating.
        """
    )

with tabs[7]:
    st.header("Business value calculator")

    st.caption("Enter assumptions. This calculator does not fabricate customer results.")

    c1, c2, c3 = st.columns(3)
    avg_power_kw = c1.number_input("Average site power (kW)", min_value=100.0, max_value=100000.0, value=850.0, step=10.0)
    annual_hours = c2.number_input("Annual operating hours", min_value=1000.0, max_value=8760.0, value=8760.0, step=10.0)
    coverage_pct = c3.number_input("Optimizer coverage (%)", min_value=0.0, max_value=100.0, value=80.0, step=5.0)

    c4, c5, c6 = st.columns(3)
    elec_cost = c4.number_input("Electricity cost ($/kWh)", min_value=0.01, max_value=0.50, value=float(st.session_state[CONTROL_KEY_MAP["electricity_cost_per_kwh"]]), step=0.005)
    carbon_intensity = c5.number_input("Average carbon intensity (gCO2e/kWh)", min_value=0.0, max_value=1000.0, value=320.0, step=10.0)
    carbon_price = c6.number_input("Carbon price ($/tCO2e)", min_value=0.0, max_value=250.0, value=float(st.session_state[CONTROL_KEY_MAP["carbon_price_per_tco2"]]), step=5.0)

    c7, c8, c9 = st.columns(3)
    savings_pct = c7.number_input("Assumed realized savings (%)", min_value=0.0, max_value=15.0, value=2.5, step=0.25)
    implementation_cost = c8.number_input("Implementation cost ($)", min_value=0.0, max_value=10000000.0, value=120000.0, step=10000.0)
    annual_opex = c9.number_input("Annual operating cost ($)", min_value=0.0, max_value=2000000.0, value=25000.0, step=1000.0)

    annual_energy_kwh = avg_power_kw * annual_hours * (coverage_pct / 100.0)
    saved_kwh = annual_energy_kwh * (savings_pct / 100.0)
    energy_savings_usd = saved_kwh * elec_cost
    co2_tonnes = saved_kwh * carbon_intensity / 1_000_000.0
    carbon_value_usd = co2_tonnes * carbon_price
    total_benefit = energy_savings_usd + carbon_value_usd
    net_annual = total_benefit - annual_opex
    payback_years = implementation_cost / net_annual if net_annual > 0 else float("inf")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual saved energy", f"{saved_kwh:,.0f} kWh")
    k2.metric("Energy savings", f"${energy_savings_usd:,.0f}")
    k3.metric("Carbon value", f"${carbon_value_usd:,.0f}")
    k4.metric("Payback", "Never" if math.isinf(payback_years) else f"{payback_years:.2f} years")

    st.subheader("Sensitivity analysis")
    sens_rows = []
    for s in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        skwh = annual_energy_kwh * (s / 100.0)
        ben = skwh * elec_cost + (skwh * carbon_intensity / 1_000_000.0) * carbon_price
        net = ben - annual_opex
        pb = implementation_cost / net if net > 0 else float("inf")
        sens_rows.append(
            {
                "Savings %": s,
                "Saved kWh": skwh,
                "Total benefit $": ben,
                "Net annual $": net,
                "Payback years": pb,
            }
        )
    st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

    st.warning(
        "These outputs are assumption-driven. Real deployment requires metered baseline, controlled evaluation, and operational safety review."
    )

with tabs[8]:
    st.header("Risk register and governance")

    st.dataframe(pd.DataFrame(RISK_REGISTER), use_container_width=True, hide_index=True)

    st.subheader("Privacy and data protection")
    st.markdown(
        """
        - Initial demo uses only synthetic operational telemetry.
        - No personal data is collected or displayed.
        - Local SQLite stores audit and recommendation metadata.
        - Production deployment should enforce RBAC, encryption at rest/in transit, retention policy, and access auditing.
        """
    )

    st.subheader("Fairness and segment checks")
    st.markdown(
        """
        The prototype segments outcomes by:
        - rack_id,
        - workload_priority (critical vs batch),
        - sensor_status,
        - provenance.

        Governance review should ensure batch-shifting burden is not systematically assigned to one team or business unit without quota and transparency.
        """
    )

    st.subheader("Threat model summary")
    st.markdown(
        """
        - **Spoofed telemetry**: mitigated by schema validation, quality scores, duplicate checks, and provenance.
        - **Model manipulation**: mitigated by chronological evaluation, uncertainty, fallback, and human approval.
        - **Unsafe automation**: mitigated by hard thermal guardrails and no automatic actuation in prototype.
        - **Audit tampering**: production should use append-only audit storage and access controls.
        """
    )

    st.subheader("Rollback and kill switch")
    st.markdown(
        """
        - Pause simulation stops new recommendations.
        - Reset clears live state but preserves audit history.
        - Reject button reverses pending recommendation without applying control changes.
        - Safe failure mode is Hold + alert, not silent actuation.
        """
    )

with tabs[9]:
    st.header("Architecture, scale, and delivery")

    st.markdown(
        """
        ### Modular architecture
        The prototype is a modular monolith in one file, but boundaries are explicit:
        - **Source adapters**: synthetic historical generator and live simulator.
        - **Ingestion/validation**: schema checks, duplicate/late/outlier handling, quarantine.
        - **Canonical data model**: rack-level Reading schema with provenance and quality flags.
        - **Feature/signal processing**: time features, economizer flag, priority encoding.
        - **Baseline analytics**: rolling per-rack baseline.
        - **Model layer**: linear regression and random forest with ensemble uncertainty.
        - **Decision/scoring layer**: guardrailed recommendation engine.
        - **API/dashboard**: Streamlit UI and local actions.
        - **Audit history**: SQLite recommendations and audit events.
        - **Observability**: data-quality report, model confidence, CPU/RAM caption.

        ### Ownership and failure boundaries
        - Ingestion owns row validity; downstream consumes only valid or fallback rows.
        - Model layer owns prediction and uncertainty; it does not decide actuation.
        - Decision layer owns guardrails and scoring; it cannot bypass approval.
        - Dashboard owns presentation; it does not mutate canonical data directly.
        - Audit service records decisions and approvals immutably in prototype.
        """
    )

    st.subheader("Mermaid architecture diagram")
    st.markdown(
        """
        ```mermaid
        flowchart LR
          SIM[Synthetic Simulator / Historical Adapter] --> ING[Ingestion + Validation]
          ING --> CAN[Canonical Readings]
          CAN --> BASE[Baseline Analytics]
          CAN --> ML[Prediction Model + Uncertainty]
          BASE --> DEC[Decision / Scoring Layer]
          ML --> DEC
          DEC --> UI[Streamlit Dashboard]
          DEC --> AUD[(Audit + Recommendations DB)]
          UI --> HUM[Human Approval]
          HUM --> AUD
          DEC --> FB[Deterministic Fallback]
          ING --> Q[Quarantine]
          Q --> UI
        ```
        """
    )

    st.subheader("Mermaid sequence diagram")
    st.markdown(
        """
        ```mermaid
        sequenceDiagram
          participant Operator
          participant Dashboard
          participant Ingestion
          participant Baseline
          participant Model
          participant Decision
          participant AuditDB

          Operator->>Dashboard: Start / adjust scenario
          Dashboard->>Ingestion: New telemetry batch
          Ingestion->>Ingestion: Validate, dedupe, flag quality
          Ingestion->>Baseline: Valid rows
          Ingestion->>Model: Valid rows
          Baseline-->>Decision: Rolling expected power
          Model-->>Decision: Prediction + uncertainty
          Decision->>Decision: Apply thermal/SLA guardrails
          Decision-->>Dashboard: Ranked recommendations
          Dashboard->>Operator: Show evidence and request approval
          Operator->>Dashboard: Approve or reject
          Dashboard->>AuditDB: Persist decision and audit event
        ```
        """
    )

    st.subheader("Scale plan without premature complexity")
    st.markdown(
        """
        Initial modular monolith is appropriate because:
        - demo data volume is small,
        - CPU inference is fast,
        - local audit is simple,
        - operational safety benefits from single-process determinism.

        Planned scaling mechanisms:
        - **Batching**: process telemetry in 15-minute windows.
        - **Caching**: cache trained models and synthetic historical data.
        - **Queues**: future ingestion queue for multi-site streams.
        - **Retries**: idempotent writes using primary keys.
        - **Idempotency**: recommendation IDs and reading primary keys prevent duplicates.
        - **Pagination**: dashboard queries limited to recent windows.
        - **Backpressure**: pause simulation if processing falls behind.
        - **Partitioning**: future partition by site_id and time.
        - **Graceful degradation**: fallback to baseline and Hold.

        Evidence that would justify splitting services:
        - sustained ingestion > 10k rows/min,
        - multiple independent sites with different SLAs,
        - model latency budget separate from UI latency,
        - regulatory requirement for isolated audit pipeline,
        - need for independent scaling of BMS connectors.
        """
    )

    st.subheader("Repository layout for productionization")
    st.markdown(
        """
        Although this deliverable is a single runnable `app.py`, a production repo would use:

        ```text
        /src
          /domain          # canonical models, guardrails, policies
          /data            # adapters, synthetic generators, repositories
          /schemas         # pydantic contracts, migration definitions
          /models          # baseline and ML trainers/inference
          /services        # orchestration, recommendation, audit
          /api             # FastAPI service layer
          /dashboard       # Streamlit or web UI modules
          /configs         # environment configs, feature flags
        /scripts           # setup, seed, evaluation, video generation
        /migrations        # SQL or Alembic migrations
        /tests
          /unit
          /integration
          /e2e
        /docs              # architecture, data dictionary, license inventory
        ```
        """
    )

    st.subheader("Windows PowerShell setup")
    st.markdown(
        """
        ```powershell
        python -m venv .venv
        .\\.venv\\Scripts\\Activate.ps1
        pip install -r requirements.txt
        streamlit run app.py
        ```
        """
    )

    st.subheader("Optional Docker Compose route")
    st.markdown(
        """
        A production deployment can containerize the API and dashboard separately.
        For this prototype, Docker is optional and not required.
        """
    )

    st.subheader("License inventory guidance")
    license_df = pd.DataFrame(
        [
            {"Component": "streamlit", "Typical license": "Apache-2.0", "Commercial use": "Generally permitted", "Attribution": "Check notices"},
            {"Component": "pandas", "Typical license": "BSD-3-Clause", "Commercial use": "Generally permitted", "Attribution": "Preserve copyright"},
            {"Component": "numpy", "Typical license": "BSD-3-Clause / BSD-2-Clause components", "Commercial use": "Generally permitted", "Attribution": "Preserve copyright"},
            {"Component": "scikit-learn", "Typical license": "BSD-3-Clause", "Commercial use": "Generally permitted", "Attribution": "Preserve copyright"},
            {"Component": "pydantic", "Typical license": "MIT", "Commercial use": "Generally permitted", "Attribution": "Preserve copyright"},
            {"Component": "matplotlib", "Typical license": "PSF-based / matplotlib license", "Commercial use": "Generally permitted", "Attribution": "Check notices"},
            {"Component": "opencv-python", "Typical license": "Apache-2.0", "Commercial use": "Generally permitted", "Attribution": "Check notices"},
            {"Component": "plotly", "Typical license": "MIT", "Commercial use": "Generally permitted", "Attribution": "Preserve copyright"},
            {"Component": "psutil", "Typical license": "BSD-3-Clause", "Commercial use": "Generally permitted", "Attribution": "Preserve copyright"},
            {"Component": "Synthetic dataset", "Typical license": "Generated in-app", "Commercial use": "No external restriction", "Attribution": "None required"},
        ]
    )
    st.dataframe(license_df, use_container_width=True, hide_index=True)

    st.warning(
        "Before commercial distribution, generate a locked dependency tree and review direct/transitive licenses, attributions, and dataset/model terms."
    )

    st.subheader("Known limitations and roadmap")
    st.markdown(
        """
        - Prototype uses synthetic physics-like relationships, not certified BMS models.
        - No real hardware actuation.
        - Local SQLite is not HA.
        - Video generation is offline simulated, not screen-recorded live control.
        - Roadmap: real DCIM adapter, model registry, policy engine, RBAC, metered A/B evaluation, multi-site federation, and certified safety interlocks.
        """
    )
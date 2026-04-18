from __future__ import annotations

from datetime import date
from pathlib import Path
import re
import warnings

import holidays
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency at runtime
    shap = None


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


DATE_COLUMNS = ["From Date", "To Date", "Applied On", "Approved On"]
DROP_COLUMNS = ["Comments", "Approver Comments"]
TEXT_FILL_COLUMNS = [
    "Department",
    "Location",
    "Leave Type",
    "Leave Reason",
    "Approved By",
    "Business Area",
    "Cost Centre",
    "Work Contract External",
    "Sub Group Category",
    "Sub Department 1",
    "Sub Department 2",
    "Sub Department 3",
    "Type",
    "SourceApp",
    "Status",
]
TARGET_COLUMN = "Leave_Count"
MASTER_FORECAST_BUFFER_DAYS = 730
FUTURE_FORECAST_WINDOW_DAYS = 60


def get_project_paths(base_dir: Path | None = None) -> dict[str, Path]:
    project_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    return {
        "project_dir": project_dir,
        "data_path": project_dir / "Data" / "Combined_All_Leave_Data.csv",
        "employee_master_path": project_dir / "Employee Master - Feb 2026 Team Member.xlsx",
        "model_path": project_dir / "artifacts" / "leave_forecasting_model.pkl",
        "metadata_path": project_dir / "artifacts" / "leave_forecasting_metadata.pkl",
    }


def build_unavailable_bundle(base_dir: Path | None = None, missing_paths: list[Path] | None = None) -> dict[str, object]:
    paths = get_project_paths(base_dir)
    missing_paths = missing_paths or []
    metadata = {
        "bundle_status": "unavailable",
        "missing_paths": [str(path) for path in missing_paths],
        "setup_hint": (
            "Run retrain_model.py after restoring the Data folder and employee master file, "
            "or open streamlit_sql_visualization.py for the non-ML dashboard."
        ),
    }
    empty_frame = pd.DataFrame({"Date": pd.to_datetime([])})
    return {
        "raw_frame": pd.DataFrame(),
        "clean_frame": pd.DataFrame(),
        "expanded_frame": pd.DataFrame(),
        "full_expanded_frame": pd.DataFrame(),
        "feature_df": empty_frame.copy(),
        "model_df": empty_frame.copy(),
        "holiday_calendar": build_holiday_calendar(date.today().year, date.today().year),
        "feature_columns": [],
        "engineered_feature_columns": [],
        "feature_fill_columns": [],
        "master_workforce_features": empty_frame.copy(),
        "master_feature_columns": [],
        "department_daily_features": empty_frame.copy(),
        "department_encoded": empty_frame.copy(),
        "leave_type_daily_features": empty_frame.copy(),
        "leave_type_monthly_features": empty_frame.copy(),
        "current_live_headcount": 0,
        "model": None,
        "metadata": metadata,
    }


def bundle_is_ready(bundle: dict[str, object]) -> bool:
    model = bundle.get("model")
    feature_df = bundle.get("feature_df")
    model_df = bundle.get("model_df")
    feature_columns = bundle.get("feature_columns", [])
    return (
        model is not None
        and isinstance(feature_df, pd.DataFrame)
        and not feature_df.empty
        and isinstance(model_df, pd.DataFrame)
        and not model_df.empty
        and bool(feature_columns)
    )


def slugify_label(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    return cleaned.strip("_") or "unknown"


def build_holiday_calendar(start_year: int, end_year: int):
    return holidays.India(years=list(range(start_year, end_year + 1)))


def bucket_holiday_name(name: str) -> str:
    if not isinstance(name, str) or name == "No Holiday":
        return "None"
    lowered = name.lower()
    if "diwali" in lowered or "deepavali" in lowered:
        return "Diwali"
    if "holi" in lowered:
        return "Holi"
    if "eid" in lowered or "id-ul" in lowered or "bakrid" in lowered:
        return "Eid"
    if "christmas" in lowered:
        return "Christmas"
    if "republic" in lowered:
        return "Republic Day"
    if "independence" in lowered:
        return "Independence Day"
    return "Other Public Holiday"


def is_long_weekend(date_value: pd.Timestamp, holiday_calendar) -> int:
    neighbours = [date_value + pd.Timedelta(days=offset) for offset in (-1, 0, 1)]
    has_holiday = any(day.date() in holiday_calendar for day in neighbours)
    has_weekend = any(day.weekday() >= 5 for day in neighbours)
    return int(has_holiday and has_weekend)


def add_calendar_features(frame: pd.DataFrame, holiday_calendar) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["day_of_week"] = enriched["Date"].dt.dayofweek
    enriched["day_name"] = enriched["Date"].dt.day_name()
    enriched["month"] = enriched["Date"].dt.month
    enriched["day_of_month"] = enriched["Date"].dt.day
    enriched["week_of_year"] = enriched["Date"].dt.isocalendar().week.astype(int)
    enriched["quarter"] = enriched["Date"].dt.quarter
    enriched["is_weekend"] = enriched["day_of_week"].isin([5, 6]).astype(int)
    enriched["is_month_start"] = enriched["Date"].dt.is_month_start.astype(int)
    enriched["is_month_end"] = enriched["Date"].dt.is_month_end.astype(int)
    enriched["month_sin"] = np.sin(2 * np.pi * enriched["month"] / 12)
    enriched["month_cos"] = np.cos(2 * np.pi * enriched["month"] / 12)
    enriched["holiday_name"] = enriched["Date"].apply(lambda value: holiday_calendar.get(value.date(), "No Holiday"))
    enriched["is_holiday"] = enriched["holiday_name"].ne("No Holiday").astype(int)
    enriched["festival_name"] = enriched["holiday_name"].apply(bucket_holiday_name)
    enriched["is_long_weekend"] = enriched["Date"].apply(lambda value: is_long_weekend(value, holiday_calendar))
    
    # Add post-holiday and day-of-week emphasis features for better daily predictions
    enriched["is_post_holiday"] = enriched["Date"].apply(
        lambda d: int((d - pd.Timedelta(days=1)).date() in holiday_calendar)
    )
    enriched["is_monday"] = (enriched["day_of_week"] == 0).astype(int)
    enriched["is_friday"] = (enriched["day_of_week"] == 4).astype(int)
    
    return enriched


def add_history_features(frame: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    enriched = frame.copy()
    for lag in [1, 7, 14, 30]:
        enriched[f"leave_lag_{lag}"] = enriched[target_col].shift(lag)
    enriched["rolling_mean_7"] = enriched[target_col].shift(1).rolling(window=7).mean()
    enriched["rolling_std_7"] = enriched[target_col].shift(1).rolling(window=7).std()
    enriched["rolling_mean_30"] = enriched[target_col].shift(1).rolling(window=30).mean()
    return enriched


def clean_leave_data(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    frame.columns = frame.columns.str.strip()

    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    for column in DATE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", dayfirst=True)

    for column in ["Days", "Delay"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[frame["Status"].eq("Approved")].copy()

    columns_to_drop = DROP_COLUMNS + [column for column in frame.columns if column.lower().startswith("userdefined")]
    frame = frame.drop(columns=[column for column in columns_to_drop if column in frame.columns], errors="ignore")

    for column in TEXT_FILL_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna("Unknown")

    for column in ["Days", "Delay"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna(frame[column].median())

    frame = frame[frame["From Date"].notna() & frame["To Date"].notna()].copy()
    frame = frame[frame["To Date"] >= frame["From Date"]].copy()

    dedupe_key = [column for column in ["EmpNo", "Leave Type", "From Date", "To Date", "Applied On"] if column in frame.columns]
    frame = frame.drop_duplicates(subset=dedupe_key, keep="first").copy()
    return frame


def read_employee_master_sheet(path: Path, sheet_name: str, header_row: int) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def prepare_employee_master(path: Path):
    live_raw = read_employee_master_sheet(path, sheet_name="Live", header_row=4)
    left_raw = read_employee_master_sheet(path, sheet_name="Left", header_row=0)

    required_columns = [
        "SAP Emp No",
        "Name",
        "Department",
        "Division",
        "Current Designation",
        "Business Area",
        "Direct / Indirect",
        "State",
        "Local/Non Local",
        "Category",
        "Sex",
        "Marrital Status",
        "D.O.J",
        "D.O.L",
        "Date of Retirement",
        "Years of Service",
        "Age",
    ]
    rename_map = {
        "SAP Emp No": "EmpNo",
        "Name": "Master_Name",
        "Department": "Master_Department",
        "Division": "Master_Division",
        "Current Designation": "Master_Designation",
        "Business Area": "Master_Business_Area",
        "Direct / Indirect": "Master_Direct_Indirect",
        "State": "Master_State",
        "Local/Non Local": "Master_Local_Non_Local",
        "Category": "Master_Category",
        "Sex": "Master_Sex",
        "Marrital Status": "Master_Marital_Status",
        "D.O.J": "Date_of_Joining",
        "D.O.L": "Date_of_Leaving",
        "Date of Retirement": "Date_of_Retirement",
        "Years of Service": "Years_of_Service",
        "Age": "Employee_Age",
    }
    text_columns = [
        "Master_Name",
        "Master_Department",
        "Master_Division",
        "Master_Designation",
        "Master_Business_Area",
        "Master_Direct_Indirect",
        "Master_State",
        "Master_Local_Non_Local",
        "Master_Category",
        "Master_Sex",
        "Master_Marital_Status",
    ]

    def select_master_columns(frame: pd.DataFrame, employment_status: str) -> pd.DataFrame:
        prepared = frame.copy()
        for column in required_columns:
            if column not in prepared.columns:
                prepared[column] = np.nan
        prepared = prepared[required_columns].copy()
        prepared["employment_status"] = employment_status
        prepared = prepared.rename(columns=rename_map)
        prepared["EmpNo"] = pd.to_numeric(prepared["EmpNo"], errors="coerce").astype("Int64")
        for column in ["Date_of_Joining", "Date_of_Leaving", "Date_of_Retirement"]:
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce")
        for column in ["Years_of_Service", "Employee_Age"]:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        for column in text_columns:
            cleaned = prepared[column].astype(str).str.strip()
            prepared[column] = cleaned.where(~cleaned.isin(["nan", ""]), np.nan)
        return prepared

    live_master = select_master_columns(live_raw, employment_status="Live")
    left_master = select_master_columns(left_raw, employment_status="Left")

    employee_master = pd.concat([live_master, left_master], ignore_index=True)
    employee_master = employee_master[employee_master["EmpNo"].notna()].copy()
    employee_master["status_priority"] = employee_master["employment_status"].map({"Live": 0, "Left": 1}).fillna(2)
    employee_master = (
        employee_master
        .sort_values(["status_priority", "Date_of_Leaving", "Date_of_Joining"], ascending=[True, False, True])
        .drop_duplicates(subset=["EmpNo"], keep="first")
        .drop(columns="status_priority")
        .reset_index(drop=True)
    )
    employee_master["employment_start"] = employee_master["Date_of_Joining"].dt.normalize()
    employee_master["employment_end"] = employee_master["Date_of_Leaving"].dt.normalize()
    return live_master, left_master, employee_master


def build_active_headcount_series(master_frame: pd.DataFrame, calendar_frame: pd.DataFrame, feature_name: str, filter_fn=None) -> pd.DataFrame:
    workforce = master_frame.copy()
    if filter_fn is not None:
        workforce = workforce.loc[filter_fn(workforce)].copy()
    workforce = workforce[workforce["employment_start"].notna()].copy()

    if workforce.empty:
        return calendar_frame[["Date"]].assign(**{feature_name: 0})

    calendar_min = calendar_frame["Date"].min()
    calendar_max = calendar_frame["Date"].max()
    workforce["start_effective"] = workforce["employment_start"].clip(lower=calendar_min)
    workforce = workforce[workforce["start_effective"] <= calendar_max].copy()
    workforce["end_effective"] = workforce["employment_end"].fillna(calendar_max).clip(lower=calendar_min, upper=calendar_max)
    workforce = workforce[workforce["start_effective"] <= workforce["end_effective"]].copy()

    join_delta = workforce.groupby("start_effective").size()
    exit_delta = (workforce["end_effective"] + pd.Timedelta(days=1)).value_counts().sort_index() * -1
    delta_series = pd.concat([join_delta, exit_delta]).groupby(level=0).sum().sort_index()

    output = calendar_frame[["Date"]].copy()
    output[feature_name] = output["Date"].map(delta_series).fillna(0).cumsum().astype(int)
    return output


def expand_leave_records(clean_frame: pd.DataFrame) -> pd.DataFrame:
    expanded_base = clean_frame[["EmpNo", "Department", "Location", "Leave Type", "From Date", "To Date"]].copy()
    day_counts = (expanded_base["To Date"] - expanded_base["From Date"]).dt.days.add(1).astype(int)
    repeated = expanded_base.loc[expanded_base.index.repeat(day_counts)].copy()
    repeated["Date"] = np.concatenate(
        [pd.date_range(start_date, end_date, freq="D").to_numpy() for start_date, end_date in zip(expanded_base["From Date"], expanded_base["To Date"])]
    )
    return repeated[["Date", "EmpNo", "Department", "Location", "Leave Type"]].reset_index(drop=True)


# Leave types that do NOT count as ON Duty absence (not requiring staffing replacement)
SPECIAL_LEAVE_TYPES = {"Special Leave [Not Call ON Duty]", "Comp-Off"}
# Leave types also excluded from training/certification attendance counts
NON_TRAINING_LEAVE_TYPES = SPECIAL_LEAVE_TYPES


def expand_leave_records_full(clean_frame: pd.DataFrame) -> pd.DataFrame:
    """Expand leave records to daily rows keeping all analytics dimensions."""
    analytics_cols = ["EmpNo", "Department", "Location", "Leave Type", "Cost Centre", "Type", "Leave Reason", "From Date", "To Date"]
    available_cols = [c for c in analytics_cols if c in clean_frame.columns]
    expanded_base = clean_frame[available_cols].copy()
    day_counts = (expanded_base["To Date"] - expanded_base["From Date"]).dt.days.add(1).astype(int)
    repeated = expanded_base.loc[expanded_base.index.repeat(day_counts)].copy()
    repeated["Date"] = np.concatenate(
        [pd.date_range(s, e, freq="D").to_numpy() for s, e in zip(expanded_base["From Date"], expanded_base["To Date"])]
    )
    result_cols = ["Date"] + [c for c in available_cols if c not in ("From Date", "To Date")]
    return repeated[result_cols].reset_index(drop=True)


def clip_leave_records_to_as_of_date(clean_frame: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    if as_of_date is None:
        return clean_frame.copy()

    cutoff = pd.Timestamp(as_of_date).normalize()
    clipped = clean_frame.copy()
    clipped = clipped[clipped["From Date"] <= cutoff].copy()
    clipped["To Date"] = clipped["To Date"].clip(upper=cutoff)
    clipped = clipped[clipped["To Date"] >= clipped["From Date"]].copy()
    return clipped.reset_index(drop=True)


def align_feature_columns(
    feature_frame: pd.DataFrame,
    requested_feature_columns: list[str] | tuple[str, ...] | None,
) -> tuple[pd.DataFrame, list[str]]:
    aligned = feature_frame.copy()
    selected_columns = [str(column) for column in (requested_feature_columns or []) if str(column).strip()]
    if not selected_columns:
        selected_columns = [
            column for column in aligned.columns
            if column not in {"Date", TARGET_COLUMN, "Leave_Events", "year_month", "day_name", "holiday_name", "festival_name"}
        ]
    for column in selected_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned[selected_columns] = aligned[selected_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    return aligned, selected_columns


def ensure_model_ready_features(bundle: dict[str, object], feature_frame: pd.DataFrame) -> pd.DataFrame:
    aligned = feature_frame.copy()
    feature_columns = bundle["feature_columns"]
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned[feature_columns] = aligned[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    return aligned[feature_columns]


def safe_model_predict(model, features: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(features)
    return np.clip(np.asarray(predictions, dtype=float), 0, None)


def derive_planning_columns(
    forecast_window: pd.DataFrame,
    total_workforce: int,
    required_present_workforce: int,
    known_absent_employees: int,
) -> pd.DataFrame:
    planned = forecast_window.copy()
    total_workforce = max(int(total_workforce), 1)
    required_present_workforce = max(int(required_present_workforce), 1)
    known_absent_employees = max(int(known_absent_employees), 0)

    planned["Predicted_Leave_Count"] = planned["Predicted_Leave_Count"].round().clip(lower=0).astype(int)
    if "Prediction_Lower_Bound" in planned.columns:
        planned["Prediction_Lower_Bound"] = planned["Prediction_Lower_Bound"].round().clip(lower=0).astype(int)
    if "Prediction_Upper_Bound" in planned.columns:
        planned["Prediction_Upper_Bound"] = planned["Prediction_Upper_Bound"].round().clip(lower=0).astype(int)

    planned["Known_Absent_Employees"] = known_absent_employees
    planned["Total_Expected_Absent"] = planned["Predicted_Leave_Count"] + known_absent_employees
    planned["Projected_Available"] = np.maximum(total_workforce - planned["Total_Expected_Absent"], 0)
    planned["Coverage_Gap"] = np.maximum(required_present_workforce - planned["Projected_Available"], 0)
    planned["Total_Staff_Needed"] = required_present_workforce + planned["Total_Expected_Absent"]
    planned["Additional_Headcount_Needed"] = np.maximum(planned["Total_Staff_Needed"] - total_workforce, 0)

    upper_bound = planned["Prediction_Upper_Bound"] if "Prediction_Upper_Bound" in planned.columns else planned["Predicted_Leave_Count"]
    planned["Conservative_Total_Expected_Absent"] = upper_bound + known_absent_employees
    planned["Conservative_Projected_Available"] = np.maximum(total_workforce - planned["Conservative_Total_Expected_Absent"], 0)
    planned["Conservative_Coverage_Gap"] = np.maximum(required_present_workforce - planned["Conservative_Projected_Available"], 0)
    return planned


def build_feature_dataset(base_dir: Path | None = None, as_of_date=None) -> dict[str, object]:
    paths = get_project_paths(base_dir)
    raw_frame = pd.read_csv(paths["data_path"], low_memory=False)
    clean_frame = clean_leave_data(raw_frame)
    clean_frame = clip_leave_records_to_as_of_date(clean_frame, as_of_date)
    employee_master_live, employee_master_left, employee_master = prepare_employee_master(paths["employee_master_path"])
    expanded_frame = expand_leave_records(clean_frame)
    employee_master_join_columns = [
        "EmpNo",
        "Master_Department",
        "Master_Division",
        "Master_Designation",
        "Master_Direct_Indirect",
        "Master_Local_Non_Local",
        "Master_Category",
        "Master_Sex",
        "Years_of_Service",
        "Employee_Age",
        "employment_status",
    ]
    expanded_frame = expanded_frame.merge(employee_master[employee_master_join_columns], on="EmpNo", how="left")
    expanded_frame["year_month"] = expanded_frame["Date"].dt.to_period("M").astype(str)

    daily_leave = (
        expanded_frame.groupby("Date")
        .agg(Leave_Count=("EmpNo", "nunique"), Leave_Events=("EmpNo", "size"))
        .reset_index()
    )

    department_daily = (
        expanded_frame.groupby(["Date", "Department"])["EmpNo"]
        .nunique()
        .reset_index(name="Department_Leave_Count")
    )
    department_daily_features = (
        department_daily.groupby("Date")
        .agg(
            department_avg_leave=("Department_Leave_Count", "mean"),
            department_leave_frequency=("Department", "nunique"),
        )
        .reset_index()
    )
    top_departments = expanded_frame["Department"].value_counts().head(5).index.tolist()
    department_encoded = (
        department_daily[department_daily["Department"].isin(top_departments)]
        .assign(feature_name=lambda frame: frame["Department"].apply(lambda value: f"dept_{slugify_label(value)}"))
        .pivot_table(index="Date", columns="feature_name", values="Department_Leave_Count", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    department_encoded_columns = [column for column in department_encoded.columns if column != "Date"]

    leave_type_daily = (
        expanded_frame.groupby(["Date", "Leave Type"])["EmpNo"]
        .count()
        .reset_index(name="Leave_Type_Count")
    )
    top_leave_types = expanded_frame["Leave Type"].value_counts().head(5).index.tolist()
    leave_type_daily_features = (
        leave_type_daily[leave_type_daily["Leave Type"].isin(top_leave_types)]
        .assign(feature_name=lambda frame: frame["Leave Type"].apply(lambda value: f"leave_type_daily_{slugify_label(value)}"))
        .pivot_table(index="Date", columns="feature_name", values="Leave_Type_Count", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    leave_type_daily_columns = [column for column in leave_type_daily_features.columns if column != "Date"]

    leave_type_monthly = (
        expanded_frame[expanded_frame["Leave Type"].isin(top_leave_types)]
        .groupby(["year_month", "Leave Type"])["EmpNo"]
        .count()
        .reset_index(name="Leave_Type_Monthly_Frequency")
    )
    leave_type_monthly["monthly_total"] = leave_type_monthly.groupby("year_month")["Leave_Type_Monthly_Frequency"].transform("sum")
    leave_type_monthly["Leave_Type_Monthly_Share"] = np.where(
        leave_type_monthly["monthly_total"] == 0,
        0,
        leave_type_monthly["Leave_Type_Monthly_Frequency"] / leave_type_monthly["monthly_total"],
    )
    leave_type_monthly_features = (
        leave_type_monthly.assign(feature_name=lambda frame: frame["Leave Type"].apply(lambda value: f"leave_type_share_{slugify_label(value)}"))
        .pivot_table(index="year_month", columns="feature_name", values="Leave_Type_Monthly_Share", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    leave_type_monthly_columns = [column for column in leave_type_monthly_features.columns if column != "year_month"]

    full_calendar = pd.DataFrame({"Date": pd.date_range(daily_leave["Date"].min(), daily_leave["Date"].max(), freq="D")})
    daily_leave = full_calendar.merge(daily_leave, on="Date", how="left")
    daily_leave[["Leave_Count", "Leave_Events"]] = daily_leave[["Leave_Count", "Leave_Events"]].fillna(0)
    daily_leave[["Leave_Count", "Leave_Events"]] = daily_leave[["Leave_Count", "Leave_Events"]].astype(int)

    master_feature_calendar = pd.DataFrame(
        {"Date": pd.date_range(daily_leave["Date"].min(), daily_leave["Date"].max() + pd.Timedelta(days=MASTER_FORECAST_BUFFER_DAYS), freq="D")}
    )
    active_employee_feature = build_active_headcount_series(employee_master, master_feature_calendar, "active_employee_count")
    active_team_member_feature = build_active_headcount_series(
        employee_master,
        master_feature_calendar,
        "active_team_member_count",
        lambda frame: frame["Master_Designation"].fillna("").str.contains("Team Member", case=False, na=False),
    )
    active_indirect_feature = build_active_headcount_series(
        employee_master,
        master_feature_calendar,
        "active_indirect_count",
        lambda frame: frame["Master_Direct_Indirect"].fillna("").str.contains("Indirect", case=False, na=False),
    )
    active_local_feature = build_active_headcount_series(
        employee_master,
        master_feature_calendar,
        "active_local_count",
        lambda frame: frame["Master_Local_Non_Local"].fillna("").str.strip().eq("Local"),
    )

    master_workforce_features = master_feature_calendar.copy()
    for feature_frame in [active_employee_feature, active_team_member_feature, active_indirect_feature, active_local_feature]:
        master_workforce_features = master_workforce_features.merge(feature_frame, on="Date", how="left")

    join_events = employee_master["employment_start"].dropna().value_counts().sort_index()
    exit_events = employee_master["employment_end"].dropna().value_counts().sort_index()
    master_workforce_features["join_count_30d"] = master_workforce_features["Date"].map(join_events).fillna(0).rolling(window=30, min_periods=1).sum()
    master_workforce_features["exit_count_30d"] = master_workforce_features["Date"].map(exit_events).fillna(0).rolling(window=30, min_periods=1).sum()
    master_workforce_features["workforce_growth_30d"] = master_workforce_features["active_employee_count"].diff(30).fillna(0)
    master_workforce_features["indirect_workforce_share"] = np.where(
        master_workforce_features["active_employee_count"] == 0,
        0,
        master_workforce_features["active_indirect_count"] / master_workforce_features["active_employee_count"],
    )
    master_workforce_features["local_workforce_share"] = np.where(
        master_workforce_features["active_employee_count"] == 0,
        0,
        master_workforce_features["active_local_count"] / master_workforce_features["active_employee_count"],
    )

    top_master_departments = employee_master_live["Master_Department"].fillna("Unknown").value_counts().head(3).index.tolist()
    master_department_headcount_columns = []
    for department_name in top_master_departments:
        feature_name = f"active_master_headcount_{slugify_label(department_name)}"
        department_feature = build_active_headcount_series(
            employee_master,
            master_feature_calendar,
            feature_name,
            lambda frame, department_name=department_name: frame["Master_Department"].fillna("Unknown").eq(department_name),
        )
        master_workforce_features = master_workforce_features.merge(department_feature, on="Date", how="left")
        master_department_headcount_columns.append(feature_name)

    master_feature_columns = [
        "active_employee_count",
        "active_team_member_count",
        "active_indirect_count",
        "active_local_count",
        "indirect_workforce_share",
        "local_workforce_share",
        "join_count_30d",
        "exit_count_30d",
        "workforce_growth_30d",
        *master_department_headcount_columns,
    ]
    master_workforce_features[master_feature_columns] = master_workforce_features[master_feature_columns].fillna(0)

    holiday_calendar = build_holiday_calendar(int(daily_leave["Date"].dt.year.min()), int(daily_leave["Date"].dt.year.max()) + 2)
    feature_df = add_calendar_features(daily_leave, holiday_calendar)
    feature_df["year_month"] = feature_df["Date"].dt.to_period("M").astype(str)
    feature_df = feature_df.merge(master_workforce_features[["Date"] + master_feature_columns], on="Date", how="left")
    feature_df = feature_df.merge(department_daily_features, on="Date", how="left")
    feature_df = feature_df.merge(department_encoded, on="Date", how="left")
    feature_df = feature_df.merge(leave_type_daily_features, on="Date", how="left")
    feature_df = feature_df.merge(leave_type_monthly_features, on="year_month", how="left")

    department_feature_columns = ["department_avg_leave", "department_leave_frequency", *department_encoded_columns]
    leave_type_feature_columns = [*leave_type_daily_columns, *leave_type_monthly_columns]
    feature_fill_columns = [*master_feature_columns, *department_feature_columns, *leave_type_feature_columns]
    feature_df[feature_fill_columns] = feature_df[feature_fill_columns].fillna(0)
    feature_df = add_history_features(feature_df)

    history_feature_columns = [
        "leave_lag_1",
        "leave_lag_7",
        "leave_lag_14",
        "leave_lag_30",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_30",
    ]
    engineered_feature_columns = [
        "day_of_week",
        "month",
        "day_of_month",
        "week_of_year",
        "quarter",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_holiday",
        "is_long_weekend",
        "month_sin",
        "month_cos",
        *master_feature_columns,
        *department_feature_columns,
        *leave_type_feature_columns,
        *history_feature_columns,
    ]
    metadata = {}
    if paths["metadata_path"].exists():
        loaded_metadata = joblib.load(paths["metadata_path"])
        if isinstance(loaded_metadata, dict):
            metadata = loaded_metadata
    requested_feature_columns = metadata.get("feature_columns", engineered_feature_columns)
    feature_df, feature_columns = align_feature_columns(feature_df, requested_feature_columns)
    model_df = feature_df.dropna(subset=[column for column in history_feature_columns if column in feature_columns]).reset_index(drop=True)

    full_expanded_frame = expand_leave_records_full(clean_frame)

    return {
        "raw_frame": raw_frame,
        "clean_frame": clean_frame,
        "expanded_frame": expanded_frame,
        "full_expanded_frame": full_expanded_frame,
        "feature_df": feature_df,
        "model_df": model_df,
        "holiday_calendar": holiday_calendar,
        "feature_columns": feature_columns,
        "engineered_feature_columns": engineered_feature_columns,
        "feature_fill_columns": feature_fill_columns,
        "master_workforce_features": master_workforce_features,
        "master_feature_columns": master_feature_columns,
        "department_daily_features": department_daily_features,
        "department_encoded": department_encoded,
        "leave_type_daily_features": leave_type_daily_features,
        "leave_type_monthly_features": leave_type_monthly_features,
        "current_live_headcount": int(employee_master_live["EmpNo"].dropna().nunique()),
    }
import os

def load_model_bundle(base_dir: Path | None = None) -> dict[str, object]:
    paths = get_project_paths(base_dir)
    required_inputs = [paths["data_path"], paths["employee_master_path"], paths["model_path"]]
    missing_inputs = [path for path in required_inputs if not path.exists()]
    if missing_inputs:
        return build_unavailable_bundle(base_dir, missing_inputs)

    metadata = joblib.load(paths["metadata_path"]) if paths["metadata_path"].exists() else {}
    if not isinstance(metadata, dict):
        metadata = {}
    required_metadata_keys = ["feature_columns", "training_end_date"]
    missing_keys = [key for key in required_metadata_keys if key not in metadata]
    if missing_keys:
        st.warning(f"Model metadata is missing keys: {missing_keys}. Using safe fallbacks.")

    try:
        model = joblib.load(paths["model_path"])
        bundle_cutoff_date = metadata.get("as_of_date") or metadata.get("training_end_date") or str(date.today())
        bundle = build_feature_dataset(base_dir, as_of_date=bundle_cutoff_date)
        bundle["model"] = model
        bundle["metadata"] = metadata
        bundle["feature_columns"] = list(metadata.get("feature_columns", bundle["feature_columns"]))
        return bundle
    except FileNotFoundError:
        return build_unavailable_bundle(base_dir, [paths["model_path"], paths["data_path"], paths["employee_master_path"]])
    except Exception as exc:
        st.warning(f"Unable to load the forecasting bundle: {exc}. Starting in limited mode.")
        return build_unavailable_bundle(base_dir, [paths["model_path"], paths["data_path"], paths["employee_master_path"]])


def weighted_absolute_percentage_error(y_true, y_pred) -> float:
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / denominator)


def mean_absolute_percentage_error_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero_mask = y_true != 0
    if not non_zero_mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])))


def symmetric_mean_absolute_percentage_error(y_true, y_pred) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    non_zero_mask = denominator != 0
    if not non_zero_mask.any():
        return 0.0
    return float(np.mean(2 * np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / denominator[non_zero_mask]))


def evaluate_saved_model(base_dir: Path | None = None, bundle: dict[str, object] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = bundle or load_model_bundle(base_dir)
    model = bundle["model"]
    model_df = bundle["model_df"]
    feature_columns = bundle["feature_columns"]
    metadata = bundle.get("metadata", {})

    model_ready_df = model_df[["Date", TARGET_COLUMN] + feature_columns].copy()
    test_start_date = metadata.get("test_start_date")
    test_end_date = metadata.get("test_end_date")

    if test_start_date and test_end_date:
        test_df = model_ready_df[
            (model_ready_df["Date"] >= pd.Timestamp(test_start_date))
            & (model_ready_df["Date"] <= pd.Timestamp(test_end_date))
        ].copy()
    else:
        holdout_size = max(30, int(len(model_ready_df) * 0.15))
        test_df = model_ready_df.iloc[-holdout_size:].copy()

    if test_df.empty:
        raise ValueError("No evaluation rows were found for the saved model test window.")

    X_test = ensure_model_ready_features(bundle, test_df)
    y_test = test_df[TARGET_COLUMN]

    model_predictions = safe_model_predict(model, X_test)
    naive_predictions = np.clip(test_df["leave_lag_1"].to_numpy(), 0, None)

    metrics_frame = pd.DataFrame(
        [
            {
                "Model": "Saved Forecast Model",
                "MAE": mean_absolute_error(y_test, model_predictions),
                "RMSE": mean_squared_error(y_test, model_predictions) ** 0.5,
                "MAPE": mean_absolute_percentage_error_safe(y_test, model_predictions),
                "R2": r2_score(y_test, model_predictions),
                "WAPE": weighted_absolute_percentage_error(y_test, model_predictions),
                "SMAPE": symmetric_mean_absolute_percentage_error(y_test, model_predictions),
            },
            {
                "Model": "Naive Lag-1 Baseline",
                "MAE": mean_absolute_error(y_test, naive_predictions),
                "RMSE": mean_squared_error(y_test, naive_predictions) ** 0.5,
                "MAPE": mean_absolute_percentage_error_safe(y_test, naive_predictions),
                "R2": r2_score(y_test, naive_predictions),
                "WAPE": weighted_absolute_percentage_error(y_test, naive_predictions),
                "SMAPE": symmetric_mean_absolute_percentage_error(y_test, naive_predictions),
            },
        ]
    )

    comparison_frame = pd.DataFrame(
        {
            "Date": test_df["Date"].to_numpy(),
            "Actual_Leave_Count": y_test.to_numpy(),
            "Predicted_Leave_Count": model_predictions,
            "Naive_Lag1_Prediction": naive_predictions,
        }
    )
    comparison_frame["Residual"] = comparison_frame["Actual_Leave_Count"] - comparison_frame["Predicted_Leave_Count"]
    comparison_frame["Absolute_Error"] = comparison_frame["Residual"].abs()
    return metrics_frame, comparison_frame


def apply_prediction_interval(prediction_frame: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    enriched = prediction_frame.copy()
    interval = metadata.get("prediction_interval", {}) if isinstance(metadata, dict) else {}
    try:
        lower_residual = float(interval.get("residual_p05", 0.0))
        upper_residual = float(interval.get("residual_p95", 0.0))
        absolute_error_p90 = float(interval.get("absolute_error_p90", 0.0))
    except (TypeError, ValueError):
        lower_residual = 0.0
        upper_residual = 0.0
        absolute_error_p90 = 0.0

    prediction_col = "Predicted_Leave_Count" if "Predicted_Leave_Count" in enriched.columns else None
    if prediction_col is None:
        return enriched

    enriched["Prediction_Lower_Bound"] = np.maximum(enriched[prediction_col] + lower_residual, 0)
    enriched["Prediction_Upper_Bound"] = np.maximum(enriched[prediction_col] + upper_residual, 0)
    enriched["Prediction_Error_Band_P90"] = absolute_error_p90
    return enriched


def extend_intelligence_summary_with_forecast(
    intelligence_summary: pd.DataFrame,
    metadata: dict[str, object],
    forecast_window_end,
) -> pd.DataFrame:
    if intelligence_summary.empty:
        return intelligence_summary

    combined = intelligence_summary.copy().sort_values("Date").reset_index(drop=True)
    forecast_rows = metadata.get("next_30_days_forecast", []) if isinstance(metadata, dict) else []
    if not forecast_rows:
        combined["Data_Source"] = "Actual"
        return combined

    forecast_frame = pd.DataFrame(forecast_rows).copy()
    if forecast_frame.empty or "Date" not in forecast_frame.columns or "Predicted_Leave_Count" not in forecast_frame.columns:
        combined["Data_Source"] = "Actual"
        return combined

    forecast_frame["Date"] = pd.to_datetime(forecast_frame["Date"], errors="coerce").dt.normalize()
    forecast_frame = forecast_frame.dropna(subset=["Date"]).copy()
    last_actual_date = combined["Date"].max()
    forecast_frame = forecast_frame[
        (forecast_frame["Date"] > last_actual_date)
        & (forecast_frame["Date"] <= pd.Timestamp(forecast_window_end))
    ].copy()

    if forecast_frame.empty:
        combined["Data_Source"] = "Actual"
        return combined

    future_summary = pd.DataFrame(
        {
            "Date": forecast_frame["Date"],
            "Employees_On_Leave": forecast_frame["Predicted_Leave_Count"].round().clip(lower=0).astype(int),
            "Staffing_Relevant_Employees": forecast_frame["Predicted_Leave_Count"].round().clip(lower=0).astype(int),
            "Unplanned_Days": 0,
            "Cost_Centres_Affected": 0,
            "Departments_Affected": 0,
            "Unplanned_Share": 0.0,
            "Special_Leave_Share": 0.0,
            "Sick_Leave": 0,
            "Casual_Leave": 0,
            "Others": forecast_frame["Predicted_Leave_Count"].round().clip(lower=0).astype(int),
            "Data_Source": "Forecast",
        }
    )

    combined["Data_Source"] = "Actual"
    combined = pd.concat([combined, future_summary], ignore_index=True, sort=False)
    combined = combined.sort_values("Date").drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)
    return combined


def load_feature_importance_from_metadata(metadata: dict[str, object]) -> pd.DataFrame:
    if not isinstance(metadata, dict):
        return pd.DataFrame()

    versioned_model_path = metadata.get("versioned_model_path")
    if not versioned_model_path:
        return pd.DataFrame()

    importance_path = Path(versioned_model_path).with_name(f"{Path(versioned_model_path).stem}_feature_importance.csv")
    if not importance_path.exists():
        return pd.DataFrame()

    return pd.read_csv(importance_path)


def build_forecast_confidence_chart(forecast_window: pd.DataFrame, prediction_date) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=forecast_window["Date"],
            y=forecast_window["Prediction_Upper_Bound"],
            mode="lines",
            line=dict(width=0),
            name="Upper bound",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=forecast_window["Date"],
            y=forecast_window["Prediction_Lower_Bound"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(15, 76, 92, 0.14)",
            name="Confidence band",
            hoverinfo="skip",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=forecast_window["Date"],
            y=forecast_window["Predicted_Leave_Count"],
            mode="lines+markers",
            name="Predicted leave",
            line=dict(color=INTELLIGENCE_COLORS["primary"], width=3),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=forecast_window["Date"],
            y=forecast_window["Projected_Available"],
            mode="lines+markers",
            name="Projected available",
            line=dict(color=INTELLIGENCE_COLORS["success"], width=2),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=forecast_window["Date"],
            y=forecast_window["Total_Staff_Needed"],
            mode="lines+markers",
            name="Total staff needed",
            line=dict(color=INTELLIGENCE_COLORS["danger"], width=2, dash="dot"),
        )
    )
    chart.update_layout(
        title=f"Workforce plan from {prediction_date}",
        template="plotly_white",
        hovermode="x unified",
    )
    return chart


def get_feature_row_for_date(bundle: dict[str, object], target_date) -> tuple[pd.DataFrame, float | None]:
    target_date = pd.Timestamp(target_date)
    feature_df = bundle["feature_df"]
    model_df = bundle["model_df"]
    feature_columns = bundle["feature_columns"]

    if target_date <= feature_df["Date"].max():
        historical_row = model_df.loc[model_df["Date"].eq(target_date), ["Date", TARGET_COLUMN] + feature_columns].copy()
        if historical_row.empty:
            raise ValueError("Selected date does not have enough lag history for explanation.")
        actual_value = float(historical_row[TARGET_COLUMN].iloc[0])
        return ensure_model_ready_features(bundle, historical_row), actual_value

    horizon = (target_date - feature_df["Date"].max()).days
    forecast_row = iterative_forecast(bundle, horizon, return_features=True).tail(1).copy()
    return ensure_model_ready_features(bundle, forecast_row), None


def explain_forecast_reason(bundle: dict[str, object], target_date, top_n: int = 8) -> pd.DataFrame:
    feature_row, _ = get_feature_row_for_date(bundle, target_date)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feature_row)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            contribution_values = np.asarray(shap_values).reshape(-1)
            explanation = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "feature_value": feature_row.iloc[0].to_numpy(),
                    "contribution": contribution_values,
                }
            )
        except Exception:
            explanation = pd.DataFrame()
    else:
        explanation = pd.DataFrame()
    if explanation.empty and hasattr(model, "feature_importances_"):
        importance_values = np.asarray(model.feature_importances_).reshape(-1)
        centered_values = feature_row.iloc[0].to_numpy() - feature_row.iloc[0].mean()
        explanation = pd.DataFrame(
            {
                "feature": feature_columns,
                "feature_value": feature_row.iloc[0].to_numpy(),
                "contribution": importance_values * centered_values,
            }
        )
    if explanation.empty:
        explanation = pd.DataFrame(
            {
                "feature": feature_columns,
                "feature_value": feature_row.iloc[0].to_numpy(),
                "contribution": np.zeros(len(feature_columns)),
            }
        )

    explanation["abs_contribution"] = explanation["contribution"].abs()
    explanation["direction"] = np.where(explanation["contribution"] >= 0, "Pushes forecast up", "Pushes forecast down")
    return explanation.sort_values("abs_contribution", ascending=False).head(top_n).reset_index(drop=True)


def iterative_forecast(bundle: dict[str, object], forecast_horizon: int, return_features: bool = False) -> pd.DataFrame:
    forecast_horizon = max(int(forecast_horizon), 0)
    if forecast_horizon == 0:
        columns = ["Date", "Predicted_Leave_Count", *bundle["feature_columns"]] if return_features else ["Date", "Predicted_Leave_Count"]
        return pd.DataFrame(columns=columns)

    history = bundle["feature_df"][["Date", TARGET_COLUMN]].copy().sort_values("Date").reset_index(drop=True)
    forecasts = []

    for _ in range(forecast_horizon):
        next_date = history["Date"].max() + pd.Timedelta(days=1)
        provisional = pd.concat(
            [history, pd.DataFrame({"Date": [next_date], TARGET_COLUMN: [np.nan]})],
            ignore_index=True,
        )
        provisional = add_calendar_features(provisional, bundle["holiday_calendar"])
        provisional["year_month"] = provisional["Date"].dt.to_period("M").astype(str)
        provisional = provisional.merge(bundle["master_workforce_features"][["Date"] + bundle["master_feature_columns"]], on="Date", how="left")
        provisional = provisional.merge(bundle["department_daily_features"], on="Date", how="left")
        provisional = provisional.merge(bundle["department_encoded"], on="Date", how="left")
        provisional = provisional.merge(bundle["leave_type_daily_features"], on="Date", how="left")
        provisional = provisional.merge(bundle["leave_type_monthly_features"], on="year_month", how="left")
        provisional[bundle["feature_fill_columns"]] = provisional[bundle["feature_fill_columns"]].ffill().bfill().fillna(0)
        provisional = add_history_features(provisional)
        next_features = ensure_model_ready_features(bundle, provisional.iloc[[-1]])
        prediction = float(safe_model_predict(bundle["model"], next_features)[0])
        history = pd.concat([history, pd.DataFrame({"Date": [next_date], TARGET_COLUMN: [prediction]})], ignore_index=True)
        forecast_record = {"Date": next_date, "Predicted_Leave_Count": prediction}
        if return_features:
            forecast_record.update(next_features.iloc[0].to_dict())
        forecasts.append(forecast_record)

    return pd.DataFrame(forecasts)


def forecast_for_specific_date(bundle: dict[str, object], target_date) -> pd.DataFrame:
    target_date = pd.Timestamp(target_date)
    feature_df = bundle["feature_df"]
    model_df = bundle["model_df"]
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    if target_date <= feature_df["Date"].max():
        historical_row = model_df.loc[model_df["Date"].eq(target_date), ["Date", TARGET_COLUMN] + feature_columns]
        if historical_row.empty:
            raise ValueError("Target date is inside the historical range but lacks sufficient lag history for prediction.")
        predicted_value = float(safe_model_predict(model, ensure_model_ready_features(bundle, historical_row))[0])
        actual_value = float(historical_row[TARGET_COLUMN].iloc[0])
        return pd.DataFrame({"Date": [target_date], "Predicted_Leave_Count": [predicted_value], "Actual_Leave_Count": [actual_value]})

    horizon = (target_date - feature_df["Date"].max()).days
    forecast_row = iterative_forecast(bundle, horizon).tail(1)
    forecast_row["Actual_Leave_Count"] = np.nan
    return forecast_row


@st.cache_data(ttl=3600)
def load_approved_daily_leave_facts(data_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_path, low_memory=False)
    clean = clean_leave_data(raw)
    expanded = expand_leave_records_full(clean)
    if expanded.empty:
        return pd.DataFrame(columns=["Date", "EmpNo", "Department", "Leave Type", "Cost Centre", "Type", "Leave Reason"])
    expanded["Date"] = pd.to_datetime(expanded["Date"]).dt.normalize()
    for column in ["Department", "Leave Type", "Cost Centre", "Type", "Leave Reason"]:
        if column not in expanded.columns:
            expanded[column] = "Unknown"
        expanded[column] = expanded[column].fillna("Unknown")
    return expanded


@st.cache_data(ttl=3600)
def get_todays_actual_leaves(data_path: Path) -> dict:
    """Get today's live approved leave count from cached daily leave facts."""
    try:
        facts = load_approved_daily_leave_facts(data_path)
        today = pd.Timestamp.now().normalize()
        today_leaves = facts[facts["Date"].eq(today)].copy()
        total_on_leave = today_leaves["EmpNo"].nunique()
        breakdown = today_leaves.groupby("Leave Type")["EmpNo"].nunique().sort_values(ascending=False).to_dict() if not today_leaves.empty else {}
        return {
            "total": int(total_on_leave),
            "breakdown": breakdown,
            "date": today.date(),
            "employee_details": today_leaves[["EmpNo", "Leave Type", "Department"]].drop_duplicates() if not today_leaves.empty else pd.DataFrame()
        }
    except Exception:
        return {"total": 0, "breakdown": {}, "date": pd.Timestamp.now().date(), "employee_details": pd.DataFrame()}


@st.cache_data(ttl=3600)
def get_daily_leaves_range(data_path: Path, start_date, end_date) -> pd.DataFrame:
    """Get historical daily actuals for past N days, aggregated by date and leave type."""
    try:
        facts = load_approved_daily_leave_facts(data_path).copy()
        start_date = pd.Timestamp(start_date).normalize()
        end_date = pd.Timestamp(end_date).normalize()
        facts = facts[(facts["Date"] >= start_date) & (facts["Date"] <= end_date)].copy()
        if facts.empty:
            return pd.DataFrame(columns=["Date", "Total_Employees", "Sick_Leave", "Casual_Leave", "Others"])
        facts["Leave_Type"] = facts["Leave Type"]
        daily_summary = (
            facts.groupby("Date")
            .agg(
                Total_Employees=("EmpNo", "nunique"),
                Sick_Leave=("Leave_Type", lambda s: int(s.eq("Sick Leave").sum())),
                Casual_Leave=("Leave_Type", lambda s: int(s.eq("Casual Leave").sum())),
            )
            .reset_index()
        )
        daily_summary["Others"] = np.maximum(
            facts.groupby("Date").size().reindex(daily_summary["Date"]).to_numpy() - daily_summary["Sick_Leave"] - daily_summary["Casual_Leave"],
            0,
        )
        return daily_summary.sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Date", "Total_Employees", "Sick_Leave", "Casual_Leave", "Others"])


@st.cache_data(ttl=3600)
def get_daily_dept_breakdown(data_path: Path, start_date, end_date) -> pd.DataFrame:
    """Get daily leave counts broken down by department (from actuals)."""
    try:
        facts = load_approved_daily_leave_facts(data_path).copy()
        start_date = pd.Timestamp(start_date).normalize()
        end_date = pd.Timestamp(end_date).normalize()
        facts = facts[(facts["Date"] >= start_date) & (facts["Date"] <= end_date)].copy()
        if facts.empty:
            return pd.DataFrame()
        dept_breakdown = (
            facts.groupby(["Date", "Department"])["EmpNo"]
            .nunique()
            .reset_index(name="Employees_On_Leave")
        )
        return dept_breakdown.sort_values(["Date", "Employees_On_Leave"], ascending=[True, False]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def detect_leave_type_spikes(data_path: Path, lookback_days: int = 30) -> dict:
    """Detect which leave types spike on which days of week (Mon=0, Fri=4, etc)."""
    try:
        facts = load_approved_daily_leave_facts(data_path).copy()
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
        facts = facts[facts["Date"] >= cutoff].copy()
        if facts.empty:
            return {}
        facts["Leave_Type"] = facts["Leave Type"]
        facts["day_of_week"] = facts["Date"].dt.dayofweek
        spike_summary = facts.groupby(["day_of_week", "Leave_Type"]).size().reset_index(name="Count")
        spike_summary["day_name"] = spike_summary["day_of_week"].map(
            {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        )
        
        spikes = {}
        for leave_type in spike_summary["Leave_Type"].unique():
            type_data = spike_summary[spike_summary["Leave_Type"] == leave_type]
            peak_day_row = type_data.loc[type_data["Count"].idxmax()]
            spikes[leave_type] = {"peak_day": peak_day_row["day_name"], "peak_count": int(peak_day_row["Count"])}
        
        return spikes
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_daily_anomalies(forecast_df: pd.DataFrame, actuals_df: pd.DataFrame = None, std_threshold: float = 1.5) -> pd.DataFrame:
    """Flag unusual days (peaks, dips, forecast vs actual mismatch) using statistical thresholds."""
    try:
        if forecast_df.empty or "Predicted_Leave_Count" not in forecast_df.columns:
            return pd.DataFrame()
        
        forecast_copy = forecast_df.copy()
        forecast_copy["Date"] = pd.to_datetime(forecast_copy["Date"])
        mean_leave = forecast_copy["Predicted_Leave_Count"].mean()
        std_leave = forecast_copy["Predicted_Leave_Count"].std()
        
        anomalies = []
        for _, row in forecast_copy.iterrows():
            z_score = abs((row["Predicted_Leave_Count"] - mean_leave) / (std_leave + 0.001))
            if z_score > std_threshold:
                anomalies.append({
                    "Date": row["Date"],
                    "Prediction": row["Predicted_Leave_Count"],
                    "Anomaly_Type": "Peak" if row["Predicted_Leave_Count"] > mean_leave else "Dip",
                    "Z_Score": z_score
                })
        
        if actuals_df is not None and not actuals_df.empty:
            actuals_df = actuals_df.copy()
            actuals_df["Date"] = pd.to_datetime(actuals_df["Date"])
            merged = forecast_copy.merge(actuals_df, on="Date", how="inner", suffixes=("_forecast", "_actual"))
            if "Total_Employees" in merged.columns:
                merged["Mismatch"] = abs(merged["Predicted_Leave_Count"] - merged["Total_Employees"])
                high_mismatch = merged[merged["Mismatch"] > merged["Mismatch"].quantile(0.75)]
                for _, row in high_mismatch.iterrows():
                    anomalies.append({
                        "Date": row["Date"],
                        "Prediction": row["Predicted_Leave_Count"],
                        "Anomaly_Type": "Forecast_Actual_Mismatch",
                        "Z_Score": row["Mismatch"]
                    })
        
        return pd.DataFrame(anomalies).drop_duplicates(subset=["Date"]) if anomalies else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def predict_date_range(bundle: dict[str, object], start_date, periods: int) -> pd.DataFrame:
    start_date = pd.Timestamp(start_date)
    periods = max(int(periods), 1)
    end_date = start_date + pd.Timedelta(days=periods - 1)
    feature_df = bundle["feature_df"]
    model_df = bundle["model_df"]
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    in_range = model_df.loc[(model_df["Date"] >= start_date) & (model_df["Date"] <= min(end_date, feature_df["Date"].max()))].copy()
    if not in_range.empty:
        in_range["Predicted_Leave_Count"] = safe_model_predict(model, ensure_model_ready_features(bundle, in_range))
        in_range["Actual_Leave_Count"] = in_range[TARGET_COLUMN]
        result = in_range[["Date", "Predicted_Leave_Count", "Actual_Leave_Count"]].copy()
    else:
        result = pd.DataFrame(columns=["Date", "Predicted_Leave_Count", "Actual_Leave_Count"])

    if end_date > feature_df["Date"].max():
        future_horizon = (end_date - feature_df["Date"].max()).days
        future_frame = iterative_forecast(bundle, future_horizon)
        future_frame = future_frame[future_frame["Date"] >= max(start_date, feature_df["Date"].max() + pd.Timedelta(days=1))].copy()
        future_frame["Actual_Leave_Count"] = np.nan
        result = pd.concat([result, future_frame], ignore_index=True)

    return result.sort_values("Date").head(periods).reset_index(drop=True)


def build_staffing_plan(predicted_leave: int, total_workforce: int, required_present_workforce: int, known_absent_employees: int) -> dict[str, int | str]:
    predicted_leave = max(int(round(predicted_leave)), 0)
    known_absent_employees = max(int(known_absent_employees), 0)
    total_expected_absent = predicted_leave + known_absent_employees
    projected_available = max(int(total_workforce) - total_expected_absent, 0)
    total_staff_needed = int(required_present_workforce) + total_expected_absent
    additional_headcount_needed = max(total_staff_needed - int(total_workforce), 0)
    coverage_gap = max(int(required_present_workforce) - projected_available, 0)

    if projected_available >= int(required_present_workforce):
        recommendation = "Planned workforce looks sufficient for the selected day."
    elif projected_available >= int(required_present_workforce * 0.9):
        recommendation = "A mild staffing gap is expected. Consider shift balancing or backup allocation."
    else:
        recommendation = "A significant staffing gap is expected. Plan overtime, floaters, or replacement staff."

    return {
        "predicted_leave": predicted_leave,
        "known_absent_employees": known_absent_employees,
        "total_expected_absent": total_expected_absent,
        "projected_available": projected_available,
        "required_present_workforce": int(required_present_workforce),
        "total_staff_needed": total_staff_needed,
        "additional_headcount_needed": additional_headcount_needed,
        "coverage_gap": coverage_gap,
        "recommendation": recommendation,
    }


INTELLIGENCE_COLORS = {
    "primary": "#0f4c5c",
    "accent": "#e36414",
    "success": "#2a9d8f",
    "warning": "#f4a261",
    "danger": "#c1121f",
    "neutral": "#7f8c8d",
}


def normalize_plan_type(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    cleaned = str(value).strip().lower()
    if cleaned == "planned":
        return "Planned"
    if cleaned in {"un-planned", "unplanned"}:
        return "Unplanned"
    return "Unknown"


def build_leave_intelligence_dataset(expanded_frame: pd.DataFrame) -> pd.DataFrame:
    if expanded_frame.empty:
        return pd.DataFrame()

    frame = expanded_frame.copy()
    frame["Type_Normalized"] = frame["Type"].apply(normalize_plan_type) if "Type" in frame.columns else "Unknown"
    frame["Is_Special_Leave"] = frame["Leave Type"].isin(SPECIAL_LEAVE_TYPES)
    frame["Is_Staffing_Relevant"] = ~frame["Is_Special_Leave"]
    frame["Is_Planned"] = frame["Type_Normalized"].eq("Planned")
    frame["Is_Unplanned"] = frame["Type_Normalized"].eq("Unplanned")
    frame["Employee_Day"] = 1
    frame["Week_Start"] = frame["Date"].dt.to_period("W").apply(lambda period: period.start_time)
    frame["Month_Label"] = frame["Date"].dt.to_period("M").astype(str)
    frame["Day_Name"] = frame["Date"].dt.day_name()
    return frame


def build_leave_intelligence_summary(intelligence_frame: pd.DataFrame) -> pd.DataFrame:
    if intelligence_frame.empty:
        return pd.DataFrame()

    summary = (
        intelligence_frame.groupby("Date")
        .agg(
            Employees_On_Leave=("EmpNo", "nunique"),
            Employee_Days=("Employee_Day", "sum"),
            Staffing_Relevant_Days=("Is_Staffing_Relevant", "sum"),
            Special_Leave_Days=("Is_Special_Leave", "sum"),
            Planned_Days=("Is_Planned", "sum"),
            Unplanned_Days=("Is_Unplanned", "sum"),
            Cost_Centres_Affected=("Cost Centre", "nunique"),
            Departments_Affected=("Department", "nunique"),
        )
        .reset_index()
    )

    staffing_employees = (
        intelligence_frame[intelligence_frame["Is_Staffing_Relevant"]]
        .groupby("Date")["EmpNo"]
        .nunique()
        .rename("Staffing_Relevant_Employees")
    )
    special_employees = (
        intelligence_frame[intelligence_frame["Is_Special_Leave"]]
        .groupby("Date")["EmpNo"]
        .nunique()
        .rename("Special_Leave_Employees")
    )
    planned_employees = (
        intelligence_frame[intelligence_frame["Is_Planned"]]
        .groupby("Date")["EmpNo"]
        .nunique()
        .rename("Planned_Employees")
    )
    unplanned_employees = (
        intelligence_frame[intelligence_frame["Is_Unplanned"]]
        .groupby("Date")["EmpNo"]
        .nunique()
        .rename("Unplanned_Employees")
    )

    summary = summary.merge(staffing_employees, on="Date", how="left")
    summary = summary.merge(special_employees, on="Date", how="left")
    summary = summary.merge(planned_employees, on="Date", how="left")
    summary = summary.merge(unplanned_employees, on="Date", how="left")
    fill_columns = [
        "Staffing_Relevant_Employees",
        "Special_Leave_Employees",
        "Planned_Employees",
        "Unplanned_Employees",
    ]
    summary[fill_columns] = summary[fill_columns].fillna(0).astype(int)
    summary["Unplanned_Share"] = np.where(summary["Employee_Days"].eq(0), 0.0, summary["Unplanned_Days"] / summary["Employee_Days"])
    summary["Special_Leave_Share"] = np.where(summary["Employee_Days"].eq(0), 0.0, summary["Special_Leave_Days"] / summary["Employee_Days"])
    summary["Operational_Risk_Index"] = (
        summary["Staffing_Relevant_Employees"] * 0.55
        + summary["Unplanned_Employees"] * 0.30
        + summary["Cost_Centres_Affected"] * 0.15
    )
    return summary.sort_values("Date").reset_index(drop=True)


def build_cost_centre_risk_summary(intelligence_frame: pd.DataFrame) -> pd.DataFrame:
    if intelligence_frame.empty:
        return pd.DataFrame()

    summary = (
        intelligence_frame.groupby("Cost Centre")
        .agg(
            Employees_On_Leave=("EmpNo", "nunique"),
            Employee_Days=("Employee_Day", "sum"),
            Staffing_Relevant_Days=("Is_Staffing_Relevant", "sum"),
            Special_Leave_Days=("Is_Special_Leave", "sum"),
            Planned_Days=("Is_Planned", "sum"),
            Unplanned_Days=("Is_Unplanned", "sum"),
            Departments_Affected=("Department", "nunique"),
        )
        .reset_index()
    )

    staffing_employees = (
        intelligence_frame[intelligence_frame["Is_Staffing_Relevant"]]
        .groupby("Cost Centre")["EmpNo"]
        .nunique()
        .rename("Staffing_Relevant_Employees")
    )
    summary = summary.merge(staffing_employees, on="Cost Centre", how="left")
    summary["Staffing_Relevant_Employees"] = summary["Staffing_Relevant_Employees"].fillna(0).astype(int)
    summary["Risk_Score"] = (
        summary["Staffing_Relevant_Employees"] * 0.50
        + summary["Unplanned_Days"] * 0.30
        + summary["Departments_Affected"] * 0.20
    )
    return summary.sort_values(["Risk_Score", "Staffing_Relevant_Employees"], ascending=[False, False]).reset_index(drop=True)


def build_leave_type_intelligence_summary(intelligence_frame: pd.DataFrame) -> pd.DataFrame:
    if intelligence_frame.empty:
        return pd.DataFrame()

    summary = (
        intelligence_frame.groupby("Leave Type")
        .agg(
            Employees_On_Leave=("EmpNo", "nunique"),
            Employee_Days=("Employee_Day", "sum"),
            Planned_Days=("Is_Planned", "sum"),
            Unplanned_Days=("Is_Unplanned", "sum"),
            Special_Leave_Days=("Is_Special_Leave", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values("Employee_Days", ascending=False).reset_index(drop=True)


def build_operational_baseline_forecast(daily_summary: pd.DataFrame, start_date, periods: int) -> pd.DataFrame:
    if daily_summary.empty:
        return pd.DataFrame(columns=["Date", "Predicted_Staffing_Relevant_Employees"])

    history = daily_summary[["Date", "Staffing_Relevant_Employees"]].copy().sort_values("Date").reset_index(drop=True)
    start_date = pd.Timestamp(start_date).normalize()
    rolling_history = history.copy()
    forecasts: list[dict[str, object]] = []
    warmup_horizon = max((start_date - history["Date"].max()).days, 0)
    total_steps = max(periods + warmup_horizon, periods)

    for _ in range(total_steps):
        next_date = rolling_history["Date"].max() + pd.Timedelta(days=1)
        same_weekday_mask = rolling_history["Date"].dt.dayofweek.eq(next_date.dayofweek)
        same_weekday_history = rolling_history.loc[same_weekday_mask, "Staffing_Relevant_Employees"].tail(8)
        trailing_history = rolling_history["Staffing_Relevant_Employees"].tail(28)

        weekday_component = same_weekday_history.mean() if not same_weekday_history.empty else trailing_history.mean()
        trend_component = trailing_history.mean() if not trailing_history.empty else history["Staffing_Relevant_Employees"].mean()
        prediction = max(int(round((weekday_component * 0.65) + (trend_component * 0.35))), 0)

        rolling_history = pd.concat(
            [rolling_history, pd.DataFrame({"Date": [next_date], "Staffing_Relevant_Employees": [prediction]})],
            ignore_index=True,
        )
        if next_date >= start_date:
            forecasts.append(
                {
                    "Date": next_date,
                    "Predicted_Staffing_Relevant_Employees": prediction,
                    "Forecast_Source": "Operational Baseline",
                }
            )

    return pd.DataFrame(forecasts).head(periods)


def build_operational_staffing_scenarios(
    forecast_frame: pd.DataFrame,
    total_workforce: int,
    required_present_workforce: int,
    known_absent_employees: int = 0,
    safety_buffer_ratio: float = 0.03,
) -> pd.DataFrame:
    if forecast_frame.empty:
        return pd.DataFrame()

    scenario_frame = forecast_frame.copy()
    scenario_frame["Known_Absent_Employees"] = max(int(known_absent_employees), 0)
    scenario_frame["Safety_Buffer"] = np.ceil(scenario_frame["Predicted_Staffing_Relevant_Employees"] * float(safety_buffer_ratio)).astype(int)
    scenario_frame["Total_Expected_Absent"] = (
        scenario_frame["Predicted_Staffing_Relevant_Employees"]
        + scenario_frame["Known_Absent_Employees"]
        + scenario_frame["Safety_Buffer"]
    )
    scenario_frame["Projected_Available"] = np.maximum(int(total_workforce) - scenario_frame["Total_Expected_Absent"], 0)
    scenario_frame["Coverage_Gap"] = np.maximum(int(required_present_workforce) - scenario_frame["Projected_Available"], 0)
    scenario_frame["Total_Staff_Needed"] = int(required_present_workforce) + scenario_frame["Total_Expected_Absent"]
    scenario_frame["Additional_Headcount_Needed"] = np.maximum(scenario_frame["Total_Staff_Needed"] - int(total_workforce), 0)
    scenario_frame["Risk_Level"] = np.select(
        [scenario_frame["Coverage_Gap"].eq(0), scenario_frame["Coverage_Gap"].le(max(int(required_present_workforce * 0.05), 1))],
        ["Stable", "Watch"],
        default="Critical",
    )
    return scenario_frame


def plot_leave_intelligence_trend(daily_summary: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=daily_summary["Date"],
            y=daily_summary["Employees_On_Leave"],
            mode="lines",
            name="Employees on Leave",
            line=dict(color=INTELLIGENCE_COLORS["primary"], width=3),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=daily_summary["Date"],
            y=daily_summary["Staffing_Relevant_Employees"],
            mode="lines",
            name="Staffing Relevant Employees",
            line=dict(color=INTELLIGENCE_COLORS["accent"], width=2, dash="dot"),
        )
    )
    figure.update_layout(
        title="Daily Leave Intelligence Trend",
        template="plotly_white",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def plot_cost_centre_risk(cost_centre_risk: pd.DataFrame, top_n: int = 12) -> go.Figure:
    chart_data = cost_centre_risk.head(top_n).sort_values("Risk_Score", ascending=True)
    figure = px.bar(
        chart_data,
        x="Risk_Score",
        y="Cost Centre",
        orientation="h",
        color="Staffing_Relevant_Employees",
        color_continuous_scale=["#dde7ec", INTELLIGENCE_COLORS["danger"]],
        title="Highest Risk Cost Centres",
        labels={"Risk_Score": "Risk Score", "Cost Centre": ""},
    )
    figure.update_layout(template="plotly_white", height=450, coloraxis_showscale=False)
    return figure


def plot_operational_staffing_gap(scenario_frame: pd.DataFrame, required_present_workforce: int) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=scenario_frame["Date"],
            y=scenario_frame["Projected_Available"],
            name="Projected Available",
            marker_color=INTELLIGENCE_COLORS["success"],
        )
    )
    figure.add_trace(
        go.Scatter(
            x=scenario_frame["Date"],
            y=[int(required_present_workforce)] * len(scenario_frame),
            mode="lines+markers",
            name="Required Present Workforce",
            line=dict(color=INTELLIGENCE_COLORS["danger"], width=3),
        )
    )
    figure.update_layout(
        title="Projected Availability vs Required Presence",
        template="plotly_white",
        hovermode="x unified",
        height=420,
    )
    return figure


@st.cache_data
def prepare_leave_intelligence(expanded_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    intelligence_frame = build_leave_intelligence_dataset(expanded_frame)
    return {
        "daily_fact": intelligence_frame,
        "daily_summary": build_leave_intelligence_summary(intelligence_frame),
        "cost_centre_risk": build_cost_centre_risk_summary(intelligence_frame),
        "leave_type_summary": build_leave_type_intelligence_summary(intelligence_frame),
    }


st.set_page_config(page_title="Employee Leave Forecasting", layout="wide")


@st.cache_resource
def load_bundle():
    return load_model_bundle(Path(__file__).resolve().parent)


@st.cache_data
def load_evaluation():
    loaded_bundle = load_bundle()
    if not bundle_is_ready(loaded_bundle):
        empty_metrics = pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE", "R2", "WAPE", "SMAPE"])
        empty_comparison = pd.DataFrame(columns=["Date", "Actual_Leave_Count", "Predicted_Leave_Count", "Naive_Lag1_Prediction"])
        return empty_metrics, empty_comparison
    try:
        return evaluate_saved_model(Path(__file__).resolve().parent, bundle=loaded_bundle)
    except Exception as exc:
        st.warning(f"Unable to evaluate the saved model: {exc}. Showing an empty evaluation view.")
        empty_metrics = pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE", "R2", "WAPE", "SMAPE"])
        empty_comparison = pd.DataFrame(columns=["Date", "Actual_Leave_Count", "Predicted_Leave_Count", "Naive_Lag1_Prediction"])
        return empty_metrics, empty_comparison


bundle = load_bundle()
metrics_df, comparison_df = load_evaluation()
project_paths = get_project_paths(Path(__file__).resolve().parent)
if not bundle_is_ready(bundle):
    st.error("The forecasting model bundle is missing from this checkout, so the ML dashboard cannot start.")
    st.info(
        f"Expected files: {project_paths['model_path']}, {project_paths['metadata_path']}, "
        f"{project_paths['data_path']}, {project_paths['employee_master_path']}"
    )
    st.info("Restore those files and run retrain_model.py, or open streamlit_sql_visualization.py for the data-only dashboard.")
    st.stop()

feature_df = bundle["feature_df"]
metadata = bundle.get("metadata", {})
feature_importance_df = load_feature_importance_from_metadata(metadata)
historical_start_date = feature_df["Date"].min().date()
last_observed_date = feature_df["Date"].max().date()
forecast_max_date = (
    bundle["master_workforce_features"]["Date"].max().date()
    if "master_workforce_features" in bundle and not bundle["master_workforce_features"].empty
    else (feature_df["Date"].max() + pd.Timedelta(days=MASTER_FORECAST_BUFFER_DAYS)).date()
)
future_start_min = historical_start_date
future_end_max = min(forecast_max_date, date.today() + pd.Timedelta(days=FUTURE_FORECAST_WINDOW_DAYS))
default_prediction_date = min(max(date.today(), historical_start_date), forecast_max_date)
full_exp = bundle.get("full_expanded_frame", pd.DataFrame())
intelligence_bundle = prepare_leave_intelligence(full_exp) if not full_exp.empty else {}

if future_end_max < future_start_min:
    st.error(
        f"No forecast dates are available from {future_start_min} through the next {FUTURE_FORECAST_WINDOW_DAYS} days from today. "
        f"The current forecast horizon ends on {forecast_max_date}."
    )
    st.stop()

st.title("OptiShift Intelligence")
# st.caption("Forecast daily leave demand and convert it into workforce planning numbers.")

# Initialize variables for sidebar
prediction_date = default_prediction_date
forecast_window_days = 1
start_date = default_prediction_date
end_date = default_prediction_date

with st.sidebar:
    st.header("Forecast Inputs")
    
    forecast_type = st.radio("Select Forecast Type", ["Daily", "Weekly"], horizontal=True)
    
    if forecast_type == "Daily":
        prediction_date = st.date_input(
            "Select Date",
            value=default_prediction_date,
            min_value=future_start_min,
            max_value=future_end_max,
            key="daily_date",
        )
        forecast_window_days = 1
        start_date = prediction_date
        end_date = prediction_date
    else:  # Weekly
        st.write("**Select Date Range**")
        col1, col2 = st.columns(2)
        with col1:
            default_weekly_start = min(max(default_prediction_date, future_start_min), future_end_max)
            start_date = st.date_input(
                "From",
                value=default_weekly_start,
                min_value=future_start_min,
                max_value=future_end_max,
                key="weekly_start",
            )
        with col2:
            default_weekly_end = min(start_date + pd.Timedelta(days=6), future_end_max)
            end_date = st.date_input(
                "To",
                value=max(start_date, default_weekly_end),
                min_value=start_date,
                max_value=future_end_max,
                key="weekly_end",
            )
        prediction_date = start_date
        forecast_window_days = (end_date - start_date).days + 1

    st.divider()
    st.caption(
        f"You can select historical dates or future forecast dates through {forecast_max_date}. Historical data currently runs through {last_observed_date}."
    )
    default_total_workforce = int(bundle.get("current_live_headcount", 1000) or 1000)
    total_workforce_input = st.number_input(
        "Current total workforce",
        min_value=1,
        value=default_total_workforce,
        step=10,
    )
    required_present_input = st.number_input(
        "Required present workforce",
        min_value=1,
        value=max(1, int(round(default_total_workforce * 0.9))),
        step=10,
    )
    known_absent_input = st.number_input(
        "Known planned absences",
        min_value=0,
        value=75,
        step=5,
    )
    
    st.divider()
    generate_forecast = st.button("Generate Forecast", type="primary", width="stretch")

if forecast_type == "Weekly" and forecast_window_days <= 0:
    st.error("The weekly end date must be on or after the start date.")
    st.stop()

if forecast_type == "Weekly":
    if start_date < historical_start_date:
        st.error(f"The weekly start date must be on or after {historical_start_date}.")
        st.stop()
    if end_date > future_end_max:
        st.error(f"The weekly end date must be on or before {future_end_max}.")
        st.stop()

if required_present_input > total_workforce_input:
    st.warning("Required present workforce is higher than current total workforce. The planner will show a staffing gap unless you increase available headcount.")

st.caption(
    f"Model artifact: {project_paths['model_path'].name} | Data source: {project_paths['data_path'].name} | Employee master: {project_paths['employee_master_path'].name} | Historical coverage through {last_observed_date}"
)

tab_forecast, tab_intelligence, tab_special, tab_costcentre, tab_planned, tab_reason = st.tabs([
    "📈 Forecasting",
    "🧭 Executive Intelligence",
    "🔵 Special Leave & Comp-Off",
    "🏭 Cost Centre Analysis",
    "📊 Planned vs Unplanned",
    "🔍 Leave Reason & Prediction",
])

with tab_forecast:
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("Model Evaluation")
        with st.expander("📊 How to Read These Metrics", expanded=False):
            st.markdown("""
            **Key Performance Metrics Explained:**
            - **MAE** (Mean Absolute Error): Average daily forecast error in # employees. Lower is better.
            - **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily. Lower is better.
            - **MAPE** (Mean Absolute % Error): Percentage error ignoring zero-leave days. Lower is better.
            - **R²** (Coefficient of Determination): % of variance explained. Closer to 1.0 is better (>0.70 = good).
            - **WAPE** (Weighted Absolute % Error): Emphasizes high-leave days. Primary metric for leave forecasting.
            - **SMAPE** (Symmetric MAPE): Fair metric for both high and low forecasts. Lower is better.
            
            👉 **Best Model Selection**: Ranked by WAPE first, then MAE, then RMSE.
            """)
        st.dataframe(
            metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2%}", "R2": "{:.3f}", "WAPE": "{:.2%}", "SMAPE": "{:.2%}"}),
            width="stretch",
        )
        with st.expander("📈 Holdout Evaluation Chart - What It Shows", expanded=False):
            st.markdown("""
            **Three Lines Explained:**
            1. **Actual_Leave_Count** (blue): True employee absences from historical data
            2. **Predicted_Leave_Count** (orange): Model's forecast for same dates
            3. **Naive_Lag1_Prediction** (green): Simple baseline (yesterday's count repeat)
            
            ✅ **Good Sign**: Predicted line closely tracks Actual line  
            ⚠️ **Watch Out**: Large gaps on specific days indicate those are harder to forecast (e.g., holidays)  
            🔴 **Problem**: If Naive baseline outperforms model, data lacks predictable patterns
            """)
        evaluation_chart = px.line(
            comparison_df,
            x="Date",
            y=["Actual_Leave_Count", "Predicted_Leave_Count", "Naive_Lag1_Prediction"],
            title="Holdout Evaluation: Actual vs Predicted Leave Count",
            template="plotly_white",
        )
        st.plotly_chart(evaluation_chart, width="stretch")

    with right_col:
        st.subheader("Model Context")
        context_rows = [
            {"Field": "Best model", "Value": metadata.get("best_model_name", type(bundle["model"]).__name__)},
            {"Field": "Training end date", "Value": metadata.get("training_end_date", str(feature_df["Date"].max().date()))},
            {"Field": "Feature count", "Value": len(metadata.get("feature_columns", bundle["feature_columns"]))},
            {"Field": "Default forecast horizon", "Value": metadata.get("forecast_horizon", 30)},
            {"Field": "Current live headcount from master", "Value": metadata.get("current_live_headcount_from_master", bundle.get("current_live_headcount", "n/a"))},
        ]
        context_df = pd.DataFrame(context_rows)
        context_df["Value"] = context_df["Value"].astype(str)
        st.dataframe(context_df, hide_index=True, width="stretch")
        
        model_test_metrics = metadata.get("test_metrics", [{}])[0] if metadata.get("test_metrics") else {}
        model_interval = metadata.get("prediction_interval", {})
        model_balance = metadata.get("model_balance", {})
        
        # ════════════════════════════════════════════════════════════════
        # OVERFITTING DETECTION - GENERALIZATION METRICS
        # ════════════════════════════════════════════════════════════════
        if model_balance:
            st.subheader("🔍 Overfitting Detection")
            with st.expander("What are these metrics?", expanded=False):
                st.markdown("""
                **Health Indicators for Model Generalization:**
                - **Validation WAPE**: Model accuracy on validation data (seen during training)
                - **Test WAPE**: Model accuracy on completely unseen test data (true generalization)
                - **Overfitting Signal**: Val WAPE - Train WAPE. If > 0.05, model is overfitting
                - **Generalization Gap**: |Val WAPE - Test WAPE|. If > 0.04, validation doesn't reflect test performance
                - **Stability Score**: How consistent validation and test performance are. Closer to 1.0 is better
                
                ✅ **Healthy Model**: Small gaps (<0.04), stability > 0.85, same performance across all sets
                """)
            
            # Create warning badges based on thresholds
            overfitting_signal = model_balance.get('Overfitting_Signal', 0)
            gen_gap = model_balance.get('Generalization_Gap_WAPE', 0)
            stability = model_balance.get('Stability_Score', 1.0)
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                # Overfitting signal badge
                if overfitting_signal > 0.10:
                    st.error(f"⚠️ Overfitting Signal: {overfitting_signal:.2%}")
                elif overfitting_signal > 0.05:
                    st.warning(f"⚠️ Overfitting Signal: {overfitting_signal:.2%}")
                else:
                    st.success(f"✅ Overfitting Signal: {overfitting_signal:.2%}")
            
            with metric_col2:
                # Generalization gap badge
                if gen_gap > 0.08:
                    st.error(f"⚠️ Gen. Gap: {gen_gap:.2%}")
                elif gen_gap > 0.04:
                    st.warning(f"⚠️ Gen. Gap: {gen_gap:.2%}")
                else:
                    st.success(f"✅ Gen. Gap: {gen_gap:.2%}")
            
            with metric_col3:
                # Stability score badge
                if stability < 0.75:
                    st.error(f"⚠️ Stability: {stability:.2%}")
                elif stability < 0.85:
                    st.warning(f"⚠️ Stability: {stability:.2%}")
                else:
                    st.success(f"✅ Stability: {stability:.2%}")
        
        health_col1, health_col2 = st.columns(2)
        with health_col1:
            st.metric("Test WAPE", f"{model_test_metrics.get('WAPE', np.nan):.2%}" if model_test_metrics else "n/a")
            st.metric("Test R2", f"{model_test_metrics.get('R2', np.nan):.3f}" if model_test_metrics else "n/a")
        with health_col2:
            st.metric("Test SMAPE", f"{model_test_metrics.get('SMAPE', np.nan):.2%}" if model_test_metrics else "n/a")
            st.metric("90% error band", f"+/- {model_interval.get('absolute_error_p90', 0.0):.1f}")
        
        if model_balance:
            balance_col1, balance_col2 = st.columns(2)
            with balance_col1:
                st.metric("Validation WAPE", f"{model_balance.get('Validation_WAPE', np.nan):.2%}")
            with balance_col2:
                st.metric("Generalization gap", f"{model_balance.get('Generalization_Gap_WAPE', np.nan):.2%}")
        
        st.markdown(
            "`Total staff needed` is calculated as required present workforce + known planned absences + model-predicted leave."
        )
        st.markdown(
            "For example, if you want 1000 employees present and 200 people are already known to be absent, the dashboard starts from 1200 total staff needed before adding any extra model-predicted leave."
        )
        if metadata.get("test_start_date") and metadata.get("test_end_date"):
            st.caption(
                f"Production holdout window: {metadata['test_start_date']} to {metadata['test_end_date']}"
            )

    if not feature_importance_df.empty:
        st.subheader("Top Forecast Drivers")
        with st.expander("🎯 How Forecast Drivers Work", expanded=False):
            st.markdown("""
            **Feature Importance Analysis:**
            
            Shows which data points the model relies on most for predictions. Longer bars = more important.
            
            **Common Top Drivers:**
            - **leave_lag_1, leave_lag_7, leave_lag_30**: Historical leave patterns (always important)
            - **is_holiday, is_long_weekend**: Holidays and extended weekends
            - **day_of_week, month**: Calendar patterns (Mondays have more sick leave, Dec has more casual leave)
            - **department_*, active_employee_count**: Organizational patterns
            
            📊 **Interpretation:**
            - If lags dominate: Model relies on "yesterday was like today" assumption
            - If calendar features are high: Seasonal patterns are strong
            - If department features are high: Different teams have different leave behaviors
            """)
        top_feature_chart = px.bar(
            feature_importance_df.head(12).sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 12 Feature Importances",
            template="plotly_white",
        )
        top_feature_chart.update_layout(height=420)
        st.plotly_chart(top_feature_chart, width="stretch")

    # ── Previous Year: Actual vs Predicted Leave Count ──────────────────────
    st.subheader("Previous Year: Actual vs Predicted Leave Count")
    with st.expander("📅 Year-over-Year Seasonal Accuracy", expanded=False):
        st.markdown("""
        **What This Shows:**
        
        Compares model accuracy across a full 12-month period (previous year) to identify seasonal strengths/weaknesses.
        
        **How to Interpret:**
        - **Red line (Actual)**: True employee absences  
        - **Blue line (Predicted)**: Model forecast  
        
        ✅ **Good Pattern**: Lines run close together all year  
        ⚠️ **Seasonal Issue**: Lines diverge in specific months (e.g., high error in December)  
        
        **Metrics Shown:**
        - **Period**: Date range analyzed  
        - **Mean Actual Daily Leave**: Average employees on leave per day  
        - **Mean Absolute Error**: Average forecast deviation  
        """)
    st.caption("Model accuracy over the 12 months before the current month")

    _today = pd.Timestamp.now().normalize()
    _prev_year_start = pd.Timestamp(_today.year - 1, _today.month, 1)
    _prev_month_end = pd.Timestamp(_today.year, _today.month, 1) - pd.Timedelta(days=1)
    _model_df = bundle["model_df"]
    _prev_year_df = _model_df[
        (_model_df["Date"] >= _prev_year_start) & (_model_df["Date"] <= _prev_month_end)
    ].copy()

    if _prev_year_df.empty:
        st.warning("No historical data available for the previous year range.")
    else:
        _prev_year_df["Predicted_Leave_Count"] = np.clip(
            bundle["model"].predict(_prev_year_df[bundle["feature_columns"]]), 0, None
        )
        _prev_year_df["Absolute_Error"] = (
            _prev_year_df[TARGET_COLUMN] - _prev_year_df["Predicted_Leave_Count"]
        ).abs()

        _py_kpi1, _py_kpi2, _py_kpi3 = st.columns(3)
        _py_kpi1.metric(
            "Period",
            f"{_prev_year_start.strftime('%b %Y')} – {_prev_month_end.strftime('%b %Y')}",
        )
        _py_kpi2.metric("Mean Actual Daily Leave", f"{_prev_year_df[TARGET_COLUMN].mean():.1f}")
        _py_kpi3.metric("Mean Absolute Error", f"{_prev_year_df['Absolute_Error'].mean():.2f}")

        _prev_year_chart = px.line(
            _prev_year_df,
            x="Date",
            y=[TARGET_COLUMN, "Predicted_Leave_Count"],
            markers=True,
            title=f"Actual vs Predicted Leave Count ({_prev_year_start.strftime('%b %Y')} – {_prev_month_end.strftime('%b %Y')})",
            template="plotly_white",
            labels={"value": "Leave Count", "variable": ""},
        )
        _prev_year_chart.update_traces(marker=dict(size=4))
        _prev_year_chart.update_layout(
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                tickformat="%d %b\n%Y",
                dtick="D1",
                tickangle=45,
                tickfont=dict(size=9),
                rangeslider=dict(visible=True, thickness=0.06),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#f0f2f6",
                    activecolor="#4c72b0",
                ),
            ),
            hovermode="x unified",
        )
        st.plotly_chart(_prev_year_chart, width="stretch")

        _prev_year_df["Month"] = _prev_year_df["Date"].dt.to_period("M").astype(str)
        _monthly_summary = (
            _prev_year_df.groupby("Month")
            .agg(
                Actual_Leave=(TARGET_COLUMN, "sum"),
                Predicted_Leave=("Predicted_Leave_Count", lambda x: int(round(x.sum()))),
                Days=("Date", "count"),
            )
            .reset_index()
        )
        _monthly_summary["Monthly_Error"] = (
            _monthly_summary["Actual_Leave"] - _monthly_summary["Predicted_Leave"]
        ).abs()
        st.dataframe(_monthly_summary, hide_index=True, width="stretch")

    forecast_horizon = int(metadata.get("forecast_horizon", 30) or 30)

    # ── NEXT 30 DAYS FORECAST ──────────────────────────────────────────────
    st.divider()
    st.subheader(f"📅 Next {forecast_horizon} Days Leave Forecast")
    
    next_30_forecast = metadata.get('next_30_days_forecast', [])
    if next_30_forecast:
        with st.expander(f"📊 Operational Planning - Next {forecast_horizon} Days", expanded=True):
            st.markdown("""
            **Workforce Planning for the saved future forecast window:**
            
            This forecast helps you:
            - Plan contingency staffing for high-leave days
            - Schedule team activities around absence patterns
            - Prepare for peak leave periods
            - Balance team workloads
            
            **Confidence Bands:** Gray area shows the 90% uncertainty range.  
            Plan your buffers within this range to handle 9 out of 10 scenarios.
            """)
            
            # Create dataframe from forecast data
            forecast_display_df = pd.DataFrame(next_30_forecast)
            forecast_display_df['Date'] = pd.to_datetime(forecast_display_df['Date']).dt.date
            
            # Display as table
            st.dataframe(
                forecast_display_df[['Date', 'Day_of_Week', 'Predicted_Leave_Count', 'Lower_Bound', 'Upper_Bound']],
                hide_index=True,
                width="stretch",
            )
            
            # Visualization
            forecast_df_viz = pd.DataFrame(next_30_forecast)
            forecast_df_viz['Date'] = pd.to_datetime(forecast_df_viz['Date'])
            
            next_30_chart = px.line(
                forecast_df_viz,
                x='Date',
                y='Predicted_Leave_Count',
                title=f'Next {forecast_horizon} Days: Leave Count Forecast with 90% Confidence Band',
                template='plotly_white',
                labels={'Predicted_Leave_Count': 'Employees on Leave', 'Date': 'Date'},
                markers=True,
            )
            
            # Add confidence band
            next_30_chart.add_scatter(
                x=forecast_df_viz['Date'],
                y=forecast_df_viz['Upper_Bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
            next_30_chart.add_scatter(
                x=forecast_df_viz['Date'],
                y=forecast_df_viz['Lower_Bound'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(100, 180, 255, 0.2)',
                fill='tonexty',
                name='90% Confidence Band',
                hoverinfo='skip'
            )
            
            next_30_chart.update_layout(height=480, hovermode='x unified')
            st.plotly_chart(next_30_chart, width="stretch")
            
            # Summary statistics
            forecast_stats_col1, forecast_stats_col2, forecast_stats_col3 = st.columns(3)
            with forecast_stats_col1:
                st.metric(
                    f"Avg Daily Leave ({forecast_horizon} days)",
                    f"{forecast_df_viz['Predicted_Leave_Count'].mean():.0f} employees"
                )
            with forecast_stats_col2:
                if forecast_df_viz.empty or "Predicted_Leave_Count" not in forecast_df_viz.columns:
                    st.metric("Peak Leave Day", "n/a")
                else:
                    peak_day = forecast_df_viz.loc[forecast_df_viz["Predicted_Leave_Count"].idxmax()]
                    st.metric(
                        "Peak Leave Day",
                        peak_day["Date"].strftime("%a, %d %b"),
                        delta=f"{int(peak_day['Predicted_Leave_Count'])} employees",
                    )
            with forecast_stats_col3:
                st.metric(
                    f"Total {forecast_horizon}-Day Employee-Days",
                    f"{int(forecast_df_viz['Predicted_Leave_Count'].sum())} days"
                )
    else:
        st.info(f"💡 {forecast_horizon}-day forecast not yet generated. Run `retrain_model.py` to generate it.")

    if generate_forecast:
        prediction_frame = forecast_for_specific_date(bundle, prediction_date)
        prediction_frame = apply_prediction_interval(prediction_frame, metadata)
        if prediction_frame.empty or "Predicted_Leave_Count" not in prediction_frame.columns:
            st.error("No forecast data is available for the selected date. Try another date.")
            st.stop()

        predicted_leave = int(round(float(prediction_frame["Predicted_Leave_Count"].iloc[0])))
        prediction_upper_bound = int(round(float(prediction_frame["Prediction_Upper_Bound"].iloc[0])))
        
        total_workforce = int(total_workforce_input)
        required_present_workforce = int(required_present_input)
        known_absent_employees = int(known_absent_input)
        
        staffing_plan = build_staffing_plan(
            predicted_leave=predicted_leave,
            total_workforce=int(total_workforce),
            required_present_workforce=int(required_present_workforce),
            known_absent_employees=int(known_absent_employees),
        )

        forecast_window = predict_date_range(bundle, prediction_date, forecast_window_days)
        forecast_window = apply_prediction_interval(forecast_window, metadata)
        if forecast_window.empty:
            st.error("No forecast window could be generated for the selected range.")
            st.stop()

        forecast_window = derive_planning_columns(
            forecast_window,
            total_workforce=int(total_workforce),
            required_present_workforce=int(required_present_workforce),
            known_absent_employees=int(known_absent_employees),
        )

        kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
        kpi_1.metric("Predicted leave", staffing_plan["predicted_leave"])
        kpi_2.metric("Upper bound leave", prediction_upper_bound)
        kpi_3.metric("Projected available", staffing_plan["projected_available"])
        kpi_4.metric("Additional headcount needed", staffing_plan["additional_headcount_needed"])

        summary_col, table_col = st.columns([1.0, 1.1])

        with summary_col:
            st.subheader("Planning Summary")
            summary_rows = [
                {"Metric": "Prediction date", "Value": str(prediction_date)},
                {"Metric": "Current total workforce", "Value": total_workforce},
                {"Metric": "Required present workforce", "Value": required_present_workforce},
                {"Metric": "Known planned absences", "Value": known_absent_employees},
                {"Metric": "Model-predicted leave", "Value": staffing_plan["predicted_leave"]},
                {"Metric": "Prediction lower bound", "Value": int(prediction_frame["Prediction_Lower_Bound"].iloc[0]) if "Prediction_Lower_Bound" in prediction_frame.columns else 0},
                {"Metric": "Prediction upper bound", "Value": prediction_upper_bound},
                {"Metric": "Total expected absences", "Value": staffing_plan["total_expected_absent"]},
                {"Metric": "Projected available workforce", "Value": staffing_plan["projected_available"]},
                {"Metric": "Coverage gap", "Value": staffing_plan["coverage_gap"]},
                {"Metric": "Conservative projected available", "Value": int(forecast_window["Conservative_Projected_Available"].iloc[0]) if "Conservative_Projected_Available" in forecast_window.columns else "n/a"},
                {"Metric": "Conservative coverage gap", "Value": int(forecast_window["Conservative_Coverage_Gap"].iloc[0]) if "Conservative_Coverage_Gap" in forecast_window.columns else "n/a"},
                {"Metric": "Total staff needed", "Value": staffing_plan["total_staff_needed"]},
            ]
            if not prediction_frame["Actual_Leave_Count"].isna().all():
                summary_rows.insert(5, {"Metric": "Actual leave count in history", "Value": int(round(float(prediction_frame["Actual_Leave_Count"].iloc[0])))})
            summary_df = pd.DataFrame(summary_rows)
            summary_df["Value"] = summary_df["Value"].astype(str)
            st.dataframe(summary_df, hide_index=True, width="stretch")
            st.success(staffing_plan["recommendation"])
            st.caption(
                f"Prediction interval is derived from production residual quantiles. 90% absolute error band: +/- {metadata.get('prediction_interval', {}).get('absolute_error_p90', 0.0):.1f}"
            )

        with table_col:
            st.subheader("Forecast Window")
            st.dataframe(forecast_window, width="stretch")

        forecast_chart = build_forecast_confidence_chart(forecast_window, prediction_date)
        st.plotly_chart(forecast_chart, width="stretch")

        st.subheader("Why This Forecast")
        try:
            explanation_df = explain_forecast_reason(bundle, prediction_date, top_n=10)
            reason_chart = px.bar(
                explanation_df.sort_values("contribution"),
                x="contribution",
                y="feature",
                orientation="h",
                color="direction",
                title=f"Top feature contributions for {prediction_date}",
                template="plotly_white",
                color_discrete_map={
                    "Pushes forecast up": INTELLIGENCE_COLORS["danger"],
                    "Pushes forecast down": INTELLIGENCE_COLORS["success"],
                },
                hover_data={"feature_value": True, "contribution": ":.3f"},
            )
            reason_chart.update_layout(height=460, yaxis_title="")
            st.plotly_chart(reason_chart, width="stretch")
            st.dataframe(
                explanation_df[["feature", "feature_value", "contribution", "direction"]],
                hide_index=True,
                width="stretch",
            )
            st.caption("Feature contributions explain the selected forecast date using the deployed model inputs for that date.")
        except Exception as exc:
            st.info(f"Prediction reason details are not available for this date yet: {exc}")
    else:
        st.subheader("How to use")
        st.write(
            "Select a date, enter your current workforce size, the number of employees you need present, and any already-known absences. Then click Generate forecast."
        )
        st.write(
            "Set a prediction date, then tune current total workforce, required present workforce, and known planned absences from the sidebar to generate scenario-ready forecasts."
        )
        st.caption("Forecast outputs include an interval band so you can plan for both point and conservative staffing scenarios.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Special Leave & Comp-Off
# Special Leave [Not Call ON Duty] and Comp-Off do NOT count as ON Duty absence
# They are also excluded from training/certification attendance calculations.
# ════════════════════════════════════════════════════════════════════════════
with tab_intelligence:
    st.subheader("Executive Leave Intelligence")
    st.caption("A decision-focused operational layer built from the expanded daily leave fact table.")
    
    with st.expander("📊 How This Tab Works", expanded=False):
        st.markdown("""
        **Executive Intelligence Overview:**
        
        This tab analyzes organizational leave patterns to identify operational risks.
        
        **Key Concepts:**
        - **Total Employees on Leave**: Includes all leave types (Casual, Sick, Comp-Off, Special, etc.)
        - **Staffing-Relevant Employees**: Excludes Comp-Off and Special Leave (no replacement needed)
        - **Risk Score**: Weighted formula based on staffing impact, unplanned absence ratio, and departmental spread
        
        **Why This Matters:**
        - Identifies high-risk departments that need contingency plans
        - Shows whether absences are concentrated or spread organization-wide
        - Highlights leaves that require actual backfill vs. administrative absences
        """)

    if not intelligence_bundle:
        st.warning("Expanded leave intelligence data is not available.")
    else:
        intelligence_daily = intelligence_bundle["daily_fact"]
        intelligence_summary = extend_intelligence_summary_with_forecast(
            intelligence_bundle["daily_summary"],
            metadata,
            future_end_max,
        )

        intel_min_date = intelligence_summary["Date"].min().date()
        intel_max_date = intelligence_summary["Date"].max().date()
        default_intel_end = min(future_end_max, intel_max_date)
        default_intel_start = max(intel_min_date, min(date.today(), default_intel_end) - pd.Timedelta(days=29))

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            intelligence_start = st.date_input(
                "Analysis start",
                value=default_intel_start,
                min_value=intel_min_date,
                max_value=future_end_max,
                key="intel_start",
            )
        with filter_col2:
            intelligence_end = st.date_input(
                "Analysis end",
                value=max(intelligence_start, default_intel_end),
                min_value=intelligence_start,
                max_value=future_end_max,
                key="intel_end",
            )

        filtered_summary = intelligence_summary[
            (intelligence_summary["Date"] >= pd.Timestamp(intelligence_start))
            & (intelligence_summary["Date"] <= pd.Timestamp(intelligence_end))
        ].copy()
        filtered_daily = intelligence_daily[
            (intelligence_daily["Date"] >= pd.Timestamp(intelligence_start))
            & (intelligence_daily["Date"] <= pd.Timestamp(intelligence_end))
        ].copy()

        if filtered_summary.empty:
            st.warning("No intelligence data is available for the selected range.")
        else:
            filtered_cost_centre = build_cost_centre_risk_summary(filtered_daily)
            filtered_leave_type = build_leave_type_intelligence_summary(filtered_daily)
            peak_row = filtered_summary.loc[filtered_summary["Employees_On_Leave"].idxmax()]
            top_risk_cost_centre = filtered_cost_centre["Cost Centre"].iloc[0] if not filtered_cost_centre.empty else "n/a"
            risk_score = filtered_cost_centre["Risk_Score"].iloc[0] if not filtered_cost_centre.empty else 0

            # Daily-Focused Metrics (replacing average)
            st.markdown("### Daily Leave Dashboard")
            
            # Metrics Row 1: Today's status, week average, peak day
            metric_1, metric_2, metric_3, metric_4 = st.columns(4)
            
            data_path = get_project_paths()["data_path"]
            todays_data = get_todays_actual_leaves(data_path)
            week_avg = filtered_summary["Employees_On_Leave"].mean()
            next_7_days = filtered_summary.head(7) if len(filtered_summary) >= 7 else filtered_summary
            peak_upcoming = next_7_days["Employees_On_Leave"].max() if len(next_7_days) > 0 else 0
            
            with metric_1:
                st.metric("Today's Actual On Leave", f"{todays_data['total']} employees")
            with metric_2:
                st.metric("Week Average Leave", f"{week_avg:.1f} employees")
            with metric_3:
                st.metric("Peak Day (Next 7)", f"{int(peak_upcoming)} employees")
            with metric_4:
                st.metric("Highest Risk Centre", top_risk_cost_centre, delta=f"{risk_score:.1f} score")
            
            # Daily Summary Table
            st.markdown("### Daily Leave Summary (Next 30 Days)")
            daily_summary_display = filtered_summary[["Date", "Employees_On_Leave", "Staffing_Relevant_Employees", "Unplanned_Days"]].copy()
            daily_summary_display["Date"] = daily_summary_display["Date"].dt.strftime("%Y-%m-%d")
            daily_summary_display["Date"] = daily_summary_display["Date"].astype(str)
            daily_summary_display["Status"] = daily_summary_display["Employees_On_Leave"].apply(
                lambda x: "🔴 High" if x > week_avg * 1.3 else ("🟡 Medium" if x > week_avg else "🟢 Normal")
            )
            daily_summary_display.columns = ["Date", "Total On Leave", "Staffing Relevant", "Unplanned Days", "Status"]
            if "Data_Source" in filtered_summary.columns:
                daily_summary_display["Source"] = filtered_summary["Data_Source"].to_numpy()
            
            st.dataframe(daily_summary_display, width="stretch", hide_index=True)
            
            # Interactive Daily Drill-Down
            st.markdown("### Daily Employee Details")
            selected_date_idx = st.selectbox(
                "Select a date to see employee details:",
                range(len(filtered_summary)),
                format_func=lambda i: f"{filtered_summary.iloc[i]['Date'].strftime('%Y-%m-%d')} - {int(filtered_summary.iloc[i]['Employees_On_Leave'])} employees",
                key="daily_drill_down"
            )
            
            if selected_date_idx is not None:
                selected_date = filtered_summary.iloc[selected_date_idx]["Date"]
                selected_source = filtered_summary.iloc[selected_date_idx].get("Data_Source", "Actual")
                dept_breakdown = get_daily_dept_breakdown(data_path, selected_date, selected_date)
                
                if selected_source == "Forecast":
                    st.info(f"Predicted summary available for {selected_date.strftime('%Y-%m-%d')}. Employee-level actual details are not available for forecast dates.")
                elif not dept_breakdown.empty:
                    st.info(f"{int(filtered_summary.iloc[selected_date_idx]['Employees_On_Leave'])} employees on leave on {selected_date.strftime('%Y-%m-%d')}")
                    dept_display = dept_breakdown.copy()
                    dept_display["Date"] = dept_display["Date"].dt.strftime("%Y-%m-%d")
                    dept_display.columns = ["Date", "Department", "Employees on Leave"]
                    st.dataframe(dept_display[["Department", "Employees on Leave"]], width="stretch", hide_index=True)
                else:
                    st.info(f"No detailed breakdown available for {selected_date.strftime('%Y-%m-%d')}")
            
            # Daily Insights Section
            st.markdown("### Daily Insights & Patterns")
            col_insights, col_spikes = st.columns(2)
            
            with col_insights:
                st.subheader("Peak Days Alert")
                peak_3_days = filtered_summary.nlargest(3, "Employees_On_Leave")
                for idx, (_, row) in enumerate(peak_3_days.iterrows(), 1):
                    st.warning(f"**Day {idx}**: {row['Date'].strftime('%a, %d %b')} - {int(row['Employees_On_Leave'])} on leave (Unplanned: {int(row['Unplanned_Days'])})")
            
            with col_spikes:
                st.subheader("Leave Type Patterns")
                spikes = detect_leave_type_spikes(data_path, lookback_days=30)
                if spikes:
                    for leave_type, info in spikes.items():
                        st.info(f"**{leave_type}** peaks on **{info['peak_day']}**s (~{info['peak_count']} instances)")
                else:
                    st.info("Insufficient data to detect spike patterns")

            chart_col, risk_col = st.columns([1.2, 1])
            with chart_col:
                st.empty()
            with risk_col:
                st.plotly_chart(plot_cost_centre_risk(filtered_cost_centre), width="stretch")
                with st.expander("🎯 Risk Score Explained", expanded=False):
                    st.markdown("""
                    **Risk Score Formula:**
                    - 50% weight: # of staffing-relevant employees on leave  
                    - 30% weight: Total unplanned leave days  
                    - 20% weight: # of departments affected  
                    
                    **High Risk** (red) → That cost centre losing many people  
                    **Action Items:**
                    - Create contingency staffing plans
                    - Cross-train adjacent teams
                    - Negotiate leave timing with managers
                    """)

            insight_col, scenario_col = st.columns([1, 1.2])
            with insight_col:
                st.markdown("### Operational Signals")
                insight_rows = [
                    {"Metric": "Unplanned day share", "Value": f"{filtered_summary['Unplanned_Share'].mean() * 100:.1f}%"},
                    {"Metric": "Special leave day share", "Value": f"{filtered_summary['Special_Leave_Share'].mean() * 100:.1f}%"},
                    {"Metric": "Average cost centres affected/day", "Value": f"{filtered_summary['Cost_Centres_Affected'].mean():.1f}"},
                    {"Metric": "Average departments affected/day", "Value": f"{filtered_summary['Departments_Affected'].mean():.1f}"},
                    {"Metric": "Peak staffing-relevant employees", "Value": int(filtered_summary["Staffing_Relevant_Employees"].max())},
                ]
                st.dataframe(pd.DataFrame(insight_rows), hide_index=True, width="stretch")

                st.markdown("### Dominant Leave Types")
                st.dataframe(filtered_leave_type.head(10), hide_index=True, width="stretch")

            with scenario_col:
                st.markdown("### Staffing Scenario Planner")
                planner_col1, planner_col2 = st.columns(2)
                default_workforce = int(bundle.get("current_live_headcount", 1000) or 1000)
                default_scenario_start = min(max(prediction_date, future_start_min), future_end_max)
                max_scenario_days = max(1, (future_end_max - default_scenario_start).days + 1)
                with planner_col1:
                    scenario_start_date = st.date_input(
                        "Scenario start",
                        value=default_scenario_start,
                        min_value=future_start_min,
                        max_value=future_end_max,
                        key="intel_scenario_start",
                    )
                    max_scenario_days = max(1, (future_end_max - scenario_start_date).days + 1)
                    scenario_periods = st.slider(
                        "Scenario days",
                        min_value=1,
                        max_value=max_scenario_days,
                        value=min(max(7, int(forecast_window_days)), max_scenario_days),
                        key="intel_scenario_days",
                    )
                    total_workforce_input = st.number_input("Total workforce", min_value=1, value=default_workforce, step=10, key="intel_total_workforce")
                with planner_col2:
                    required_present_input = st.number_input(
                        "Required present workforce",
                        min_value=1,
                        value=max(1, int(round(default_workforce * 0.9))),
                        step=10,
                        key="intel_required_present",
                    )
                    known_absent_input = st.number_input("Known absences", min_value=0, value=75, step=5, key="intel_known_absent")
                    safety_buffer_ratio = st.slider("Safety buffer %", min_value=0.0, max_value=15.0, value=3.0, step=0.5, key="intel_buffer") / 100

                baseline_forecast = build_operational_baseline_forecast(filtered_summary, scenario_start_date, int(scenario_periods))
                scenario_table = build_operational_staffing_scenarios(
                    baseline_forecast,
                    total_workforce=int(total_workforce_input),
                    required_present_workforce=int(required_present_input),
                    known_absent_employees=int(known_absent_input),
                    safety_buffer_ratio=float(safety_buffer_ratio),
                )

                if scenario_table.empty:
                    st.info("Not enough data to build the operational staffing scenario.")
                else:
                    st.plotly_chart(
                        plot_operational_staffing_gap(scenario_table, int(required_present_input)),
                        width="stretch",
                    )
                    st.dataframe(scenario_table, hide_index=True, width="stretch")
                    st.caption("This planner uses a weekday-aware operational baseline from the expanded daily leave history.")


with tab_special:
    st.subheader("Special Leave & Comp-Off Analysis")
    with st.expander("ℹ️ What is Special Leave & Comp-Off?", expanded=False):
        st.markdown("""
        **Special Leave [Not Call ON Duty]**: Administrative absence (company events, emergencies, etc.)  
        → Does NOT count as absence for operational staffing  
        → No need to hire replacements
        
        **Comp-Off (Compensatory Off)**: Given when employees work on holidays/weekends  
        → Employee settlement period after overtime  
        → Doesn't require backfill (it's owed time)
        
        **Why Track Separately:**
        - Operational forecasting focuses on leave that needs coverage
        - Special Leave and Comp-Off are **administrative**, not operational
        - They're tracked separately to avoid inflating "true" leave forecasts
        """)
    
    st.info(
        "**Special Leave [Not Call ON Duty]** and **Comp-Off** are NOT counted as ON Duty absence "
        "and are NOT required for training/certification attendance — they are tracked separately here."
    )

    if full_exp.empty:
        st.warning("Full expanded leave data not available.")
    else:
        _sl_df = full_exp[full_exp["Leave Type"].isin(SPECIAL_LEAVE_TYPES)].copy()
        if _sl_df.empty:
            st.warning("No Special Leave or Comp-Off records found.")
        else:
            _sl_date_min = _sl_df["Date"].min().date()
            _sl_date_max = _sl_df["Date"].max().date()
            _sl_col1, _sl_col2 = st.columns(2)
            with _sl_col1:
                _sl_start = st.date_input("From", value=_sl_date_min, min_value=_sl_date_min, max_value=_sl_date_max, key="sl_start")
            with _sl_col2:
                _sl_end = st.date_input("To", value=_sl_date_max, min_value=_sl_date_min, max_value=_sl_date_max, key="sl_end")

            _sl_filtered = _sl_df[
                (_sl_df["Date"] >= pd.Timestamp(_sl_start)) & (_sl_df["Date"] <= pd.Timestamp(_sl_end))
            ].copy()

            _sl_kpi1, _sl_kpi2, _sl_kpi3 = st.columns(3)
            _sl_kpi1.metric("Total Special Leave Days", int(_sl_filtered[_sl_filtered["Leave Type"] == "Special Leave [Not Call ON Duty]"].shape[0]))
            _sl_kpi2.metric("Total Comp-Off Days", int(_sl_filtered[_sl_filtered["Leave Type"] == "Comp-Off"].shape[0]))
            _sl_kpi3.metric("Unique Employees", int(_sl_filtered["EmpNo"].nunique()))

            # Weekly chart — separate bars for each type
            _sl_filtered["Week"] = _sl_filtered["Date"].dt.to_period("W").apply(lambda r: r.start_time)
            _sl_weekly = (
                _sl_filtered.groupby(["Week", "Leave Type"])
                .size()
                .reset_index(name="Days")
            )
            _sl_weekly["Week_Label"] = _sl_weekly["Week"].dt.strftime("W/C %d %b %Y")

            _sl_weekly_chart = px.bar(
                _sl_weekly,
                x="Week_Label",
                y="Days",
                color="Leave Type",
                barmode="group",
                title="Weekly Special Leave & Comp-Off (Separate — Not Counted as ON Duty)",
                template="plotly_white",
                color_discrete_map={
                    "Special Leave [Not Call ON Duty]": "#3a86ff",
                    "Comp-Off": "#ff6b6b",
                },
            )
            _sl_weekly_chart.update_xaxes(tickangle=45)
            _sl_weekly_chart.update_layout(height=420, xaxis_title="Week Starting", yaxis_title="Leave Days")
            st.plotly_chart(_sl_weekly_chart, width="stretch")
            with st.expander("📅 Weekly Pattern - What to Look For", expanded=False):
                st.markdown("""
                **Chart Shows:**
                - Blue bars = Special Leave days per week
                - Red bars = Comp-Off days per week
                
                **Interpretation:**
                - **Regular spikes**: Systematic policy (e.g., comp-off settlements happen every 2 weeks)
                - **Scattered spikes**: Ad-hoc company events
                - **Increasing trend**: Growing comp-off liability (overtime culture warning)
                - **Seasonal pattern**: More comp-offs before year-end (usage deadline pressure)
                """)

            # Monthly trend side-by-side
            _sl_filtered["Month"] = _sl_filtered["Date"].dt.to_period("M").astype(str)
            _sl_monthly = (
                _sl_filtered.groupby(["Month", "Leave Type"])
                .size()
                .reset_index(name="Days")
            )
            _sl_monthly_chart = px.bar(
                _sl_monthly,
                x="Month",
                y="Days",
                color="Leave Type",
                barmode="group",
                title="Monthly Special Leave & Comp-Off Trend",
                template="plotly_white",
                color_discrete_map={
                    "Special Leave [Not Call ON Duty]": "#3a86ff",
                    "Comp-Off": "#ff6b6b",
                },
            )
            _sl_monthly_chart.update_xaxes(tickangle=45)
            _sl_monthly_chart.update_layout(height=380, yaxis_title="Leave Days")
            st.plotly_chart(_sl_monthly_chart, width="stretch")
            with st.expander("📊 Monthly Trend - Spotting Issues", expanded=False):
                st.markdown("""
                **What This Chart Reveals:**
                - Which months have heavy comp-off settlement burden
                - Whether special leave is random or tied to company events
                - Comp-off liability accumulation over time
                
                **Red Flags:**
                - Comp-off growing month-over-month (employee burnout risk)
                - Sudden spike mid-month (unplanned event or policy change)
                - December spike (year-end comp-off rush)
                """)

            # Day-of-week pattern
            _sl_filtered["Day_of_Week"] = _sl_filtered["Date"].dt.day_name()
            _dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            _sl_dow = (
                _sl_filtered.groupby(["Day_of_Week", "Leave Type"])
                .size()
                .reset_index(name="Days")
            )
            _sl_dow["Day_of_Week"] = pd.Categorical(_sl_dow["Day_of_Week"], categories=_dow_order, ordered=True)
            _sl_dow = _sl_dow.sort_values("Day_of_Week")
            _sl_dow_chart = px.bar(
                _sl_dow,
                x="Day_of_Week",
                y="Days",
                color="Leave Type",
                barmode="group",
                title="Day-of-Week Pattern — Special Leave & Comp-Off",
                template="plotly_white",
                color_discrete_map={
                    "Special Leave [Not Call ON Duty]": "#3a86ff",
                    "Comp-Off": "#ff6b6b",
                },
            )
            _sl_dow_chart.update_layout(height=350, yaxis_title="Leave Days")
            st.plotly_chart(_sl_dow_chart, width="stretch")

            st.caption("⚠️ These leave types are excluded from ON Duty counts and training attendance requirements.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Cost Centre Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_costcentre:
    st.subheader("Cost Centre Wise Leave Analysis")
    with st.expander("🏭 Understanding Cost Centre Analysis", expanded=False):
        st.markdown("""
        **Why Cost Centre Matters:**
        - Identify which departments have highest leave volumes
        - Understand leave distribution patterns by organization
        - Plan department-specific retention/engagement strategies
        - Allocate contingency staffing to high-risk areas
        
        **Key Charts:**
        - **Pie/Donut**: Total leave days by department (% breakdown)
        - **Treemaps**: Drill-down by cost centre → leave type
        - **Heatmaps**: Time patterns (which days/weeks/months problematic)
        - **Daily/Weekly/Monthly**: Trends over time per department
        """)

    if full_exp.empty or "Cost Centre" not in full_exp.columns:
        st.warning("Cost Centre data not available.")
    else:
        _cc_all_types = sorted(full_exp["Leave Type"].dropna().unique().tolist())
        _cc_date_min = full_exp["Date"].min().date()
        _cc_date_max = full_exp["Date"].max().date()

        _cc_f1, _cc_f2, _cc_f3 = st.columns(3)
        with _cc_f1:
            _cc_start = st.date_input("From", value=_cc_date_min, min_value=_cc_date_min, max_value=_cc_date_max, key="cc_start")
        with _cc_f2:
            _cc_end = st.date_input("To", value=_cc_date_max, min_value=_cc_date_min, max_value=_cc_date_max, key="cc_end")
        with _cc_f3:
            _cc_leave_filter = st.multiselect("Leave Type filter", options=_cc_all_types, default=[], key="cc_leave_filter", placeholder="All types")

        _cc_df = full_exp[
            (full_exp["Date"] >= pd.Timestamp(_cc_start)) & (full_exp["Date"] <= pd.Timestamp(_cc_end))
        ].copy()
        if _cc_leave_filter:
            _cc_df = _cc_df[_cc_df["Leave Type"].isin(_cc_leave_filter)]

        if _cc_df.empty:
            st.warning("No data for selected filters.")
        else:
            # ── KPIs
            _cc_total_emp = int(_cc_df["EmpNo"].nunique())
            _cc_k1, _cc_k2, _cc_k3, _cc_k4 = st.columns(4)
            _cc_k1.metric("Total Leave Days", int(_cc_df.shape[0]))
            _cc_k2.metric("Cost Centres", int(_cc_df["Cost Centre"].nunique()))
            _cc_k3.metric("Employees on Leave", _cc_total_emp)
            _cc_k4.metric("Avg Days per Employee", f"{_cc_df.shape[0] / _cc_total_emp:.1f}" if _cc_total_emp else "0")

            # ════════════════════════════════════════════════════════════
            # SECTION A — Employees on Leave segregated by Cost Centre
            # "If 10 employees are on leave, how many come from each CC?"
            # ════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 👥 Employee Count Segregation by Cost Centre")
            st.caption(
                "Shows how many **unique employees** are on leave from each cost centre. "
                "Example: if 10 employees are on leave, this breaks down those 10 across cost centres."
            )

            _cc_emp_by_cc = (
                _cc_df.groupby("Cost Centre")["EmpNo"]
                .nunique()
                .reset_index(name="Employees on Leave")
                .sort_values("Employees on Leave", ascending=False)
            )
            _cc_emp_by_cc["% of Total"] = (
                _cc_emp_by_cc["Employees on Leave"] / _cc_total_emp * 100
            ).round(1)

            _pie_col, _bar_col = st.columns([1, 1.2])

            with _pie_col:
                _cc_donut = px.pie(
                    _cc_emp_by_cc,
                    names="Cost Centre",
                    values="Employees on Leave",
                    title=f"Employee Distribution Across Cost Centres\n(Total: {_cc_total_emp} employees)",
                    template="plotly_white",
                    hole=0.45,
                )
                _cc_donut.update_traces(
                    textposition="outside",
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>Employees: %{value}<br>Share: %{percent}<extra></extra>",
                )
                _cc_donut.update_layout(height=420, showlegend=False)
                st.plotly_chart(_cc_donut, width="stretch")

            with _bar_col:
                _cc_emp_bar = px.bar(
                    _cc_emp_by_cc.sort_values("Employees on Leave"),
                    x="Employees on Leave",
                    y="Cost Centre",
                    orientation="h",
                    title="Employees on Leave per Cost Centre",
                    template="plotly_white",
                    color="Employees on Leave",
                    color_continuous_scale="Teal",
                    text="Employees on Leave",
                )
                _cc_emp_bar.update_traces(textposition="outside")
                _cc_emp_bar.update_layout(
                    height=max(320, len(_cc_emp_by_cc) * 55),
                    yaxis_title="",
                    showlegend=False,
                )
                st.plotly_chart(_cc_emp_bar, width="stretch")

            # ── Employee segregation table (like "10 employees → CC1: 4, CC2: 3 …")
            st.dataframe(
                _cc_emp_by_cc.rename(columns={"% of Total": "% of Total Employees on Leave"}),
                hide_index=True,
                width="stretch",
            )

            # ── Treemap — employee count by Cost Centre + Leave Type
            st.markdown("#### Employee Count by Cost Centre & Leave Type")
            _cc_tree_df = (
                _cc_df.groupby(["Cost Centre", "Leave Type"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees")
            )
            _cc_treemap = px.treemap(
                _cc_tree_df,
                path=["Cost Centre", "Leave Type"],
                values="Employees",
                title="Employee Count Treemap — Cost Centre → Leave Type",
                template="plotly_white",
                color="Employees",
                color_continuous_scale="Blues",
            )
            _cc_treemap.update_traces(
                texttemplate="<b>%{label}</b><br>%{value} emp",
                hovertemplate="<b>%{label}</b><br>Employees: %{value}<extra></extra>",
            )
            _cc_treemap.update_layout(height=480)
            st.plotly_chart(_cc_treemap, width="stretch")

            # ════════════════════════════════════════════════════════════
            # SECTION B — Pick a single date: who is on leave & from which CC?
            # ════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📅 Single-Date Employee Breakdown by Cost Centre")
            st.caption("Pick any date to see exactly how many employees from each cost centre are on leave that day.")

            _cc_single_date = st.date_input(
                "Select date",
                value=_cc_date_max,
                min_value=_cc_date_min,
                max_value=_cc_date_max,
                key="cc_single_date",
            )
            _cc_day_df = full_exp[full_exp["Date"] == pd.Timestamp(_cc_single_date)].copy()
            if _cc_leave_filter:
                _cc_day_df = _cc_day_df[_cc_day_df["Leave Type"].isin(_cc_leave_filter)]

            if _cc_day_df.empty:
                st.info(f"No leave records found for {_cc_single_date}.")
            else:
                _cc_day_total = int(_cc_day_df["EmpNo"].nunique())
                st.success(
                    f"**{_cc_day_total} employee(s)** on leave on **{_cc_single_date}** — "
                    f"segregated below across {int(_cc_day_df['Cost Centre'].nunique())} cost centre(s)."
                )

                _cc_day_by_cc = (
                    _cc_day_df.groupby("Cost Centre")["EmpNo"]
                    .nunique()
                    .reset_index(name="Employees on Leave")
                    .sort_values("Employees on Leave", ascending=False)
                )
                _cc_day_by_cc["% of Day Total"] = (
                    _cc_day_by_cc["Employees on Leave"] / _cc_day_total * 100
                ).round(1)

                _day_pie_col, _day_tbl_col = st.columns([1, 1])

                with _day_pie_col:
                    _cc_day_pie = px.pie(
                        _cc_day_by_cc,
                        names="Cost Centre",
                        values="Employees on Leave",
                        title=f"{_cc_day_total} Employees on Leave — {_cc_single_date}",
                        template="plotly_white",
                        hole=0.4,
                    )
                    _cc_day_pie.update_traces(
                        textposition="outside",
                        textinfo="label+value+percent",
                        hovertemplate="<b>%{label}</b><br>Employees: %{value}<br>Share: %{percent}<extra></extra>",
                    )
                    _cc_day_pie.update_layout(height=380, showlegend=False)
                    st.plotly_chart(_cc_day_pie, width="stretch")

                with _day_tbl_col:
                    st.markdown(f"**Breakdown of {_cc_day_total} employees on {_cc_single_date}**")
                    st.dataframe(_cc_day_by_cc, hide_index=True, width="stretch")

                    # leave type split for that day
                    _cc_day_lt = (
                        _cc_day_df.groupby(["Cost Centre", "Leave Type"])["EmpNo"]
                        .nunique()
                        .reset_index(name="Employees")
                    )
                    _cc_day_lt_chart = px.bar(
                        _cc_day_lt,
                        x="Cost Centre",
                        y="Employees",
                        color="Leave Type",
                        barmode="stack",
                        title=f"Leave Type Split on {_cc_single_date}",
                        template="plotly_white",
                    )
                    _cc_day_lt_chart.update_xaxes(tickangle=30)
                    _cc_day_lt_chart.update_layout(height=320, yaxis_title="Employees")
                    st.plotly_chart(_cc_day_lt_chart, width="stretch")

            # ════════════════════════════════════════════════════════════
            # SECTION C — Daily employee headcount on leave per Cost Centre
            # ════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📈 Daily Employee Count on Leave by Cost Centre")
            _cc_daily_emp = (
                _cc_df.groupby(["Date", "Cost Centre"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees on Leave")
            )
            _cc_daily_emp_chart = px.line(
                _cc_daily_emp,
                x="Date",
                y="Employees on Leave",
                color="Cost Centre",
                markers=False,
                title="Daily Employees on Leave per Cost Centre",
                template="plotly_white",
            )
            _cc_daily_emp_chart.update_layout(
                height=420,
                yaxis_title="Employees on Leave",
                xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
            )
            st.plotly_chart(_cc_daily_emp_chart, width="stretch")

            # ── Weekly employee headcount by Cost Centre
            _cc_df["Week"] = _cc_df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
            _cc_weekly_emp = (
                _cc_df.groupby(["Week", "Cost Centre"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees on Leave")
            )
            _cc_weekly_emp_chart = px.bar(
                _cc_weekly_emp,
                x="Week",
                y="Employees on Leave",
                color="Cost Centre",
                barmode="stack",
                title="Weekly Employee Count on Leave by Cost Centre",
                template="plotly_white",
            )
            _cc_weekly_emp_chart.update_layout(height=420, xaxis_title="Week Starting", yaxis_title="Employees on Leave")
            st.plotly_chart(_cc_weekly_emp_chart, width="stretch")

            # ── Monthly heatmap — employees (not days)
            _cc_df["Month"] = _cc_df["Date"].dt.to_period("M").astype(str)
            _cc_heat = (
                _cc_df.groupby(["Cost Centre", "Month"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees")
                .pivot(index="Cost Centre", columns="Month", values="Employees")
                .fillna(0)
            )
            _cc_heatmap = go.Figure(data=go.Heatmap(
                z=_cc_heat.values,
                x=_cc_heat.columns.tolist(),
                y=_cc_heat.index.tolist(),
                colorscale="YlOrRd",
                hoverongaps=False,
                hovertemplate="Cost Centre: <b>%{y}</b><br>Month: %{x}<br>Employees: %{z}<extra></extra>",
            ))
            _cc_heatmap.update_layout(
                title="Monthly Employees on Leave Heatmap by Cost Centre",
                xaxis_title="Month",
                yaxis_title="Cost Centre",
                height=max(300, len(_cc_heat) * 50 + 100),
                template="plotly_white",
            )
            st.plotly_chart(_cc_heatmap, width="stretch")

            # ── Summary table
            _cc_summary = (
                _cc_df.groupby("Cost Centre")
                .agg(
                    Total_Leave_Days=("EmpNo", "size"),
                    Unique_Employees=("EmpNo", "nunique"),
                    Most_Common_Leave_Type=("Leave Type", lambda x: x.mode().iloc[0] if not x.empty else ""),
                )
                .reset_index()
                .sort_values("Unique_Employees", ascending=False)
            )
            _cc_summary["% of Total Employees"] = (
                _cc_summary["Unique_Employees"] / _cc_total_emp * 100
            ).round(1)
            st.markdown("#### Summary Table")
            st.dataframe(_cc_summary, hide_index=True, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Planned vs Unplanned
# ════════════════════════════════════════════════════════════════════════════
with tab_planned:
    st.subheader("Planned vs Unplanned Leave Dashboard")
    
    with st.expander("📋 Planned vs Unplanned Explained", expanded=False):
        st.markdown("""
        **Planned Leave**: Employee gives advance notice (casual leave, vacation)  
        → Forecastable, can arrange coverage in advance  
        → Key metric for resource planning
        
        **Unplanned Leave**: No advance notice (sick leave, emergencies)  
        → Creates operational disruption  
        → Harder to forecast; need larger safety buffers  
        → May indicate health/morale issues if high
        
        **Why This Matters:**
        - High unplanned % → Urgent action needed (engagement survey, wellness program)
        - Low unplanned % → Mature leave culture; easy to predict
        - Cost centre comparison → Identify departments with engagement issues
        
        **Decision Framework:**
        - Unplanned > 50%? → Hire buffers; can't rely on forecasts
        - Unplanned < 30%? → Most leaves are planned; forecasts reliable
        - Growing unplanned trend? → Investigate root causes (burnout, policy changes)
        """)

    if full_exp.empty or "Type" not in full_exp.columns:
        st.warning("Leave Type/Plan data not available.")
    else:
        _pu_date_min = full_exp["Date"].min().date()
        _pu_date_max = full_exp["Date"].max().date()

        _pu_f1, _pu_f2, _pu_f3 = st.columns(3)
        with _pu_f1:
            _pu_start = st.date_input("From", value=_pu_date_min, min_value=_pu_date_min, max_value=_pu_date_max, key="pu_start")
        with _pu_f2:
            _pu_end = st.date_input("To", value=_pu_date_max, min_value=_pu_date_min, max_value=_pu_date_max, key="pu_end")
        with _pu_f3:
            _pu_cc = st.multiselect(
                "Cost Centre",
                options=sorted(full_exp["Cost Centre"].dropna().unique().tolist()) if "Cost Centre" in full_exp.columns else [],
                default=[],
                key="pu_cc",
                placeholder="All cost centres",
            )

        _pu_df = full_exp[
            (full_exp["Date"] >= pd.Timestamp(_pu_start)) & (full_exp["Date"] <= pd.Timestamp(_pu_end))
        ].copy()
        if _pu_cc:
            _pu_df = _pu_df[_pu_df["Cost Centre"].isin(_pu_cc)]
        _pu_df = _pu_df[_pu_df["Type"].notna()].copy()
        _pu_df["Type"] = _pu_df["Type"].str.strip().str.title()  # normalise to "Planned" / "Un-Planned"

        if _pu_df.empty:
            st.warning("No data for selected filters.")
        else:
            _planned_emp   = int(_pu_df[_pu_df["Type"] == "Planned"]["EmpNo"].nunique())
            _unplanned_emp = int(_pu_df[_pu_df["Type"] == "Un-Planned"]["EmpNo"].nunique())
            _total_emp     = int(_pu_df["EmpNo"].nunique())
            _planned_count = int((_pu_df["Type"] == "Planned").sum())
            _unplanned_count = int((_pu_df["Type"] == "Un-Planned").sum())
            _total = _planned_count + _unplanned_count

            # ── KPI row 1: Leave Days
            st.markdown("#### Leave Days")
            _pu_k1, _pu_k2, _pu_k3, _pu_k4 = st.columns(4)
            _pu_k1.metric("Total Leave Days", _total)
            _pu_k2.metric("Planned Days", _planned_count)
            _pu_k3.metric("Unplanned Days", _unplanned_count)
            _pu_k4.metric("Unplanned %", f"{_unplanned_count / _total * 100:.1f}%" if _total else "0%")

            # ── KPI row 2: Employee Headcount
            st.markdown("#### Employee Headcount on Leave")
            _pu_e1, _pu_e2, _pu_e3, _pu_e4 = st.columns(4)
            _pu_e1.metric("Total Employees on Leave", _total_emp)
            _pu_e2.metric("Employees — Planned", _planned_emp)
            _pu_e3.metric("Employees — Unplanned", _unplanned_emp)
            _pu_e4.metric("Unplanned Emp %", f"{_unplanned_emp / _total_emp * 100:.1f}%" if _total_emp else "0%")

            # ── Split pie — Days (removed duplicate employee split pie)
            _pie_l, _pie_r = st.columns([1.2, 1])
            with _pie_l:
                _pu_pie = px.pie(
                    values=[_planned_count, _unplanned_count],
                    names=["Planned", "Un-Planned"],
                    title="Leave Days Split (Planned vs Unplanned)",
                    template="plotly_white",
                    hole=0.4,
                    color_discrete_sequence=["#2ecc71", "#e74c3c"],
                )
                _pu_pie.update_traces(textposition="inside", textinfo="percent+label+value")
                _pu_pie.update_layout(height=340, showlegend=False)
                st.plotly_chart(_pu_pie, width="stretch")
            with _pie_r:
                # Key metrics summary instead of duplicate pie chart
                st.markdown("#### Key Summary")
                _summary_metrics = [
                    {"Metric": "Planned Days", "Value": f"{_planned_count} ({_planned_count/_total*100:.1f}%)"},
                    {"Metric": "Unplanned Days", "Value": f"{_unplanned_count} ({_unplanned_count/_total*100:.1f}%)"},
                    {"Metric": "Planned Employees", "Value": f"{_planned_emp}"},
                    {"Metric": "Unplanned Employees", "Value": f"{_unplanned_emp}"},
                    {"Metric": "Avg Days/Employee", "Value": f"{_total/_total_emp:.2f}"},
                ]
                _st_df = pd.DataFrame(_summary_metrics)
                st.dataframe(_st_df, hide_index=True, width="stretch")

            # ══════════════════════════════════════════════════════════
            # EMPLOYEE HEADCOUNT BY LEAVE TYPE — Planned vs Unplanned
            # ══════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 👥 Employee Headcount by Leave Type")
            st.caption(
                "Unique employees on leave split by leave type and whether their leave was Planned or Unplanned."
            )

            _pu_lt_emp = (
                _pu_df.groupby(["Leave Type", "Type"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees")
                .sort_values("Employees", ascending=False)
            )

            _lt_bar_col, _lt_tbl_col = st.columns([1.6, 1])
            with _lt_bar_col:
                _pu_lt_chart = px.bar(
                    _pu_lt_emp,
                    x="Leave Type",
                    y="Employees",
                    color="Type",
                    barmode="group",
                    title="Employees on Leave by Leave Type (Planned vs Unplanned)",
                    template="plotly_white",
                    color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
                    text="Employees",
                )
                _pu_lt_chart.update_traces(textposition="outside")
                _pu_lt_chart.update_xaxes(tickangle=30)
                _pu_lt_chart.update_layout(height=430, yaxis_title="Employees on Leave")
                st.plotly_chart(_pu_lt_chart, width="stretch")

            with _lt_tbl_col:
                # Pivot into a clean table: Leave Type | Planned Emp | Unplanned Emp | Total Emp
                _pu_lt_pivot = (
                    _pu_lt_emp.pivot(index="Leave Type", columns="Type", values="Employees")
                    .fillna(0)
                    .astype(int)
                    .reset_index()
                )
                # Ensure both columns exist even if one type is missing for a leave type
                for _col in ("Planned", "Un-Planned"):
                    if _col not in _pu_lt_pivot.columns:
                        _pu_lt_pivot[_col] = 0
                _pu_lt_pivot["Total Employees"] = _pu_lt_pivot.get("Planned", 0) + _pu_lt_pivot.get("Un-Planned", 0)
                _pu_lt_pivot = _pu_lt_pivot.rename(columns={"Planned": "Planned Emp", "Un-Planned": "Unplanned Emp"})
                _pu_lt_pivot = _pu_lt_pivot.sort_values("Total Employees", ascending=False).reset_index(drop=True)
                st.markdown("**Headcount Table by Leave Type**")
                st.dataframe(_pu_lt_pivot, hide_index=True, width="stretch")

            # ── Stacked horizontal bar — headcount per leave type
            _pu_lt_h = px.bar(
                _pu_lt_emp,
                x="Employees",
                y="Leave Type",
                color="Type",
                barmode="stack",
                orientation="h",
                title="Total Employee Headcount by Leave Type (stacked Planned + Unplanned)",
                template="plotly_white",
                color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
                text="Employees",
            )
            _pu_lt_h.update_traces(textposition="inside")
            _pu_lt_h.update_layout(height=max(320, len(_pu_lt_emp["Leave Type"].unique()) * 50), yaxis_title="")
            st.plotly_chart(_pu_lt_h, width="stretch")

            # ══════════════════════════════════════════════════════════
            # DAILY / WEEKLY — Employee headcount (not leave days)
            # ══════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📅 Daily & Weekly Employee Headcount")

            _pu_daily_emp = (
                _pu_df.groupby(["Date", "Type"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees")
            )
            _pu_daily_emp_chart = px.bar(
                _pu_daily_emp,
                x="Date",
                y="Employees",
                color="Type",
                barmode="stack",
                title="Daily Employee Headcount — Planned vs Unplanned",
                template="plotly_white",
                color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
            )
            _pu_daily_emp_chart.update_layout(
                height=380,
                xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
                yaxis_title="Employees on Leave",
            )
            st.plotly_chart(_pu_daily_emp_chart, width="stretch")

            _pu_df["Week"] = _pu_df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
            _pu_weekly_emp = (
                _pu_df.groupby(["Week", "Type"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees")
            )
            _pu_weekly_emp["Week_Label"] = _pu_weekly_emp["Week"].dt.strftime("W/C %d %b %Y")
            _pu_weekly_emp_chart = px.bar(
                _pu_weekly_emp,
                x="Week_Label",
                y="Employees",
                color="Type",
                barmode="stack",
                title="Weekly Employee Headcount — Planned vs Unplanned",
                template="plotly_white",
                color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
            )
            _pu_weekly_emp_chart.update_xaxes(tickangle=45)
            _pu_weekly_emp_chart.update_layout(height=400, xaxis_title="Week Starting", yaxis_title="Employees on Leave")
            st.plotly_chart(_pu_weekly_emp_chart, width="stretch")

            # ── Original leave-days daily/weekly charts (kept for reference)
            st.markdown("---")
            st.markdown("### 📄 Leave Days (for reference)")
            _pu_daily = (
                _pu_df.groupby(["Date", "Type"])
                .size()
                .reset_index(name="Days")
            )
            _pu_daily_chart = px.bar(
                _pu_daily,
                x="Date",
                y="Days",
                color="Type",
                barmode="stack",
                title="Daily Planned vs Unplanned Leave Days",
                template="plotly_white",
                color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
            )
            _pu_daily_chart.update_layout(
                height=380,
                xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
                yaxis_title="Leave Days",
            )
            st.plotly_chart(_pu_daily_chart, width="stretch")

            _pu_weekly = (
                _pu_df.groupby(["Week", "Type"])
                .size()
                .reset_index(name="Days")
            )
            _pu_weekly["Week_Label"] = _pu_weekly["Week"].dt.strftime("W/C %d %b %Y")
            _pu_weekly_chart = px.bar(
                _pu_weekly,
                x="Week_Label",
                y="Days",
                color="Type",
                barmode="stack",
                title="Weekly Planned vs Unplanned Leave Days",
                template="plotly_white",
                color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
            )
            _pu_weekly_chart.update_xaxes(tickangle=45)
            _pu_weekly_chart.update_layout(height=400, xaxis_title="Week Starting", yaxis_title="Leave Days")
            st.plotly_chart(_pu_weekly_chart, width="stretch")

            # ── Cost Centre breakdown of planned/unplanned (employees)
            if "Cost Centre" in _pu_df.columns:
                _pu_cc_breakdown = (
                    _pu_df.groupby(["Cost Centre", "Type"])["EmpNo"]
                    .nunique()
                    .reset_index(name="Employees")
                )
                _pu_cc_chart = px.bar(
                    _pu_cc_breakdown,
                    x="Cost Centre",
                    y="Employees",
                    color="Type",
                    barmode="group",
                    title="Employees on Leave by Cost Centre — Planned vs Unplanned",
                    template="plotly_white",
                    color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
                    text="Employees",
                )
                _pu_cc_chart.update_traces(textposition="outside")
                _pu_cc_chart.update_xaxes(tickangle=30)
                _pu_cc_chart.update_layout(height=400, yaxis_title="Employees on Leave")
                st.plotly_chart(_pu_cc_chart, width="stretch")

            # ── Day-of-week pattern (employees)
            _pu_df["Day_of_Week"] = _pu_df["Date"].dt.day_name()
            _dow_order2 = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            _pu_dow = (
                _pu_df.groupby(["Day_of_Week", "Type"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees")
            )
            _pu_dow["Day_of_Week"] = pd.Categorical(_pu_dow["Day_of_Week"], categories=_dow_order2, ordered=True)
            _pu_dow = _pu_dow.sort_values("Day_of_Week")
            _pu_dow_chart = px.bar(
                _pu_dow,
                x="Day_of_Week",
                y="Employees",
                color="Type",
                barmode="group",
                title="Day-of-Week Pattern — Employees (Planned vs Unplanned)",
                template="plotly_white",
                color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
            )
            _pu_dow_chart.update_layout(height=350, yaxis_title="Employees on Leave")
            st.plotly_chart(_pu_dow_chart, width="stretch")

            # ── Monthly trend line (employees)
            _pu_df["Month"] = _pu_df["Date"].dt.to_period("M").astype(str)
            _pu_monthly_emp = (
                _pu_df.groupby(["Month", "Type"])["EmpNo"]
                .nunique()
                .reset_index(name="Employees")
            )
            _pu_monthly_chart = px.line(
                _pu_monthly_emp,
                x="Month",
                y="Employees",
                color="Type",
                markers=True,
                title="Monthly Employee Headcount Trend — Planned vs Unplanned",
                template="plotly_white",
                color_discrete_map={"Planned": "#2ecc71", "Un-Planned": "#e74c3c"},
            )
            _pu_monthly_chart.update_xaxes(tickangle=45)
            _pu_monthly_chart.update_layout(height=380, yaxis_title="Employees on Leave")
            st.plotly_chart(_pu_monthly_chart, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Leave Reason & Prediction Context
# ════════════════════════════════════════════════════════════════════════════
with tab_reason:
    st.subheader("Leave Reason & Prediction Context by Cost Centre")
    st.caption("Understand what leave reasons drive absence — filter by cost centre and leave type before predicting.")
    
    with st.expander("🔍 Leave Reason Analysis - Why This Tab?", expanded=False):
        st.markdown("""
        **Leave Reason Categories:**
        - **Casual Leave**: Discretionary absence (vacation, personal days)
        - **Sick Leave**: Health-related (illness, medical appointments)
        - **Comp-Off**: Earned time from overtime
        - **Special Leave**: Company events, emergencies (no coverage needed)
        - **ON-Duty**: Official work outside office (training, meetings)
        
        **Why Analyze by Reason:**
        1. **Casual > 50%?** → Discretionary; employees have work-life balance
        2. **Sick > 30%?** → Health/wellness concern; consider wellness programs
        3. **Comp-Off > 20%?** → Excessive overtime culture; burnout risk
        4. **Special Leave spike?** → Indicates company events (planned) vs. emergencies (sudden)
        
        **Action Triggers:**
        - High sick in specific dept? → Investigate morale/environment
        - Comp-off growing? → Review overtime policies
        - Reason doesn't match season? → Possible data quality issue
        
        **Use This Tab To:**
        - Benchmark departments against org average
        - Identify behavioral patterns by leave reason
        - Support HR policy decisions
        """)

    if full_exp.empty:
        st.warning("Full expanded leave data not available.")
    else:
        _lr_all_cc = sorted(full_exp["Cost Centre"].dropna().unique().tolist()) if "Cost Centre" in full_exp.columns else []
        _lr_all_lt = sorted(full_exp["Leave Type"].dropna().unique().tolist())

        _lr_f1, _lr_f2 = st.columns(2)
        with _lr_f1:
            _lr_cc_sel = st.multiselect("Cost Centre", options=_lr_all_cc, default=[], key="lr_cc", placeholder="All cost centres")
        with _lr_f2:
            _lr_lt_sel = st.multiselect("Leave Type", options=_lr_all_lt, default=[], key="lr_lt", placeholder="All leave types")

        _lr_date_min = full_exp["Date"].min().date()
        _lr_date_max = full_exp["Date"].max().date()
        _lrd1, _lrd2 = st.columns(2)
        with _lrd1:
            _lr_start = st.date_input("From", value=_lr_date_min, min_value=_lr_date_min, max_value=_lr_date_max, key="lr_start")
        with _lrd2:
            _lr_end = st.date_input("To", value=_lr_date_max, min_value=_lr_date_min, max_value=_lr_date_max, key="lr_end")

        _lr_df = full_exp[
            (full_exp["Date"] >= pd.Timestamp(_lr_start)) & (full_exp["Date"] <= pd.Timestamp(_lr_end))
        ].copy()
        if _lr_cc_sel:
            _lr_df = _lr_df[_lr_df["Cost Centre"].isin(_lr_cc_sel)]
        if _lr_lt_sel:
            _lr_df = _lr_df[_lr_df["Leave Type"].isin(_lr_lt_sel)]

        if _lr_df.empty:
            st.warning("No data for selected filters.")
        else:
            # ── Top leave reasons overall
            if "Leave Reason" in _lr_df.columns:
                _lr_reasons = (
                    _lr_df.groupby("Leave Reason")
                    .size()
                    .reset_index(name="Days")
                    .sort_values("Days", ascending=False)
                    .head(15)
                )
                _lr_reason_chart = px.bar(
                    _lr_reasons,
                    x="Days",
                    y="Leave Reason",
                    orientation="h",
                    title="Top 15 Leave Reasons (filtered selection)",
                    template="plotly_white",
                    color="Days",
                    color_continuous_scale="Oranges",
                )
                _lr_reason_chart.update_layout(height=450, yaxis_title="", showlegend=False)
                st.plotly_chart(_lr_reason_chart, width="stretch")

                # Top reasons table
                st.markdown("#### Top Leave Reasons Summary")
                st.dataframe(_lr_reasons.rename(columns={"Days": "Leave Days"}), hide_index=True, width="stretch")

            # ── Leave type breakdown by Cost Centre (filtered)
            if "Cost Centre" in _lr_df.columns:
                _lr_lt_cc = (
                    _lr_df.groupby(["Cost Centre", "Leave Type"])
                    .size()
                    .reset_index(name="Days")
                )
                _lr_lt_cc_chart = px.bar(
                    _lr_lt_cc,
                    x="Cost Centre",
                    y="Days",
                    color="Leave Type",
                    barmode="stack",
                    title="Leave Type by Cost Centre (filtered)",
                    template="plotly_white",
                )
                _lr_lt_cc_chart.update_xaxes(tickangle=30)
                _lr_lt_cc_chart.update_layout(height=400, yaxis_title="Leave Days")
                st.plotly_chart(_lr_lt_cc_chart, width="stretch")

            # ── Prediction context box — show dominant leave types/reasons for a date
            st.markdown("---")
            st.subheader("Prediction Context")
            st.caption("For the selected prediction date, see which cost centres and leave reasons are historically dominant.")
            _ctx_date = st.date_input(
                "Context date for prediction",
                value=default_prediction_date,
                min_value=historical_start_date,
                max_value=forecast_max_date,
                key="lr_ctx_date",
            )
            _ctx_month = pd.Timestamp(_ctx_date).strftime("%B")
            _ctx_dow = pd.Timestamp(_ctx_date).strftime("%A")
            st.info(f"Selected: **{_ctx_date}** ({_ctx_dow}, {_ctx_month})")

            _ctx_hist = full_exp[
                (full_exp["Date"].dt.month == pd.Timestamp(_ctx_date).month) &
                (full_exp["Date"].dt.day_of_week == pd.Timestamp(_ctx_date).dayofweek)
            ].copy()

            if not _ctx_hist.empty:
                _ctx_c1, _ctx_c2 = st.columns(2)
                with _ctx_c1:
                    st.markdown("**Top Cost Centres** (same month & weekday historically)")
                    _ctx_cc_top = (
                        _ctx_hist.groupby("Cost Centre").size().reset_index(name="Days").sort_values("Days", ascending=False).head(6)
                    )
                    st.dataframe(_ctx_cc_top, hide_index=True, width="stretch")
                with _ctx_c2:
                    st.markdown("**Top Leave Reasons** (same month & weekday historically)")
                    if "Leave Reason" in _ctx_hist.columns:
                        _ctx_lr_top = (
                            _ctx_hist.groupby("Leave Reason").size().reset_index(name="Days").sort_values("Days", ascending=False).head(6)
                        )
                        st.dataframe(_ctx_lr_top, hide_index=True, width="stretch")
            else:
                st.info("No historical data found for same month and weekday.")


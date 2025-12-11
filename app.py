# app.py
import numpy as np
import pandas as pd
import streamlit as st

from dataclasses import dataclass
from typing import List, Dict, Tuple
import joblib

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Leadership Pipeline Simulation (Digital Twin Inspired)",
    layout="wide"
)

# ---------------------------------------------------------
# Global constants
# ---------------------------------------------------------

# Features expected by your attrition_model.pkl
MODEL_FEATURE_COLS = [
    "age",
    "years_at_company",
    "performance_rating",
    "gender",
    "race",
    "role_level",
]

RACE_CATEGORIES = ["Asian", "Black", "Hispanic", "Other", "White"]
ROLE_LEVELS = ["IC", "Mid", "Senior"]


# ---------------------------------------------------------
# 1. LOAD MODEL & DATA
# ---------------------------------------------------------

@st.cache_resource
def load_attrition_model():
    """Load the trained ML model (sklearn Pipeline)."""
    model = joblib.load("attrition_model.pkl")
    return model


@st.cache_data
def load_base_dataframe() -> pd.DataFrame:
    """
    Load IBM HR dataset (Data.csv) and derive:
    - age
    - years_at_company
    - performance_rating
    - gender
    - race            (synthetic but aligned with categories)
    - role_level      (from JobLevel)
    - leadership / technical / strategic skills (synthetic, correlated)
    - ur_group        (UR vs Non-UR from race)
    """
    df_raw = pd.read_csv("Data.csv")

    # Basic core fields
    df = pd.DataFrame()
    df["emp_id"] = df_raw["EmployeeNumber"]
    df["age"] = df_raw["Age"]
    df["years_at_company"] = df_raw["YearsAtCompany"]
    df["performance_rating"] = df_raw["PerformanceRating"]
    df["gender"] = df_raw["Gender"]

    # Map JobLevel -> role_level (simple IC / Mid / Senior mapping)
    def map_role_level(job_level: int) -> str:
        if job_level <= 1:
            return "IC"
        elif job_level <= 3:
            return "Mid"
        else:
            return "Senior"

    df["role_level"] = df_raw["JobLevel"].apply(map_role_level)

    # Synthetic race (since IBM dataset has no race column)
    # Use fixed RNG seed for reproducibility and align with model categories
    rng = np.random.default_rng(42)
    df["race"] = rng.choice(RACE_CATEGORIES, size=len(df))

    # Approximate skills from job level + performance
    # This is just to support "skill coverage" logic in the simulation.
    jl = df_raw["JobLevel"]
    pr = df["performance_rating"]

    # Normalize helper
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    jl_n = norm(jl)
    pr_n = norm(pr)

    # Leadership skill: more with higher job level + rating
    df["skill_leadership"] = np.clip(
        0.2 + 0.4 * jl_n + 0.3 * pr_n + rng.normal(0, 0.05, len(df)),
        0.0,
        1.0,
    )

    # Technical skill: everyone has some; slightly higher at IC/Mid
    df["skill_technical"] = np.clip(
        0.4 + 0.2 * (1 - jl_n) + 0.2 * pr_n + rng.normal(0, 0.05, len(df)),
        0.0,
        1.0,
    )

    # Strategic skill: higher at senior levels
    df["skill_strategic"] = np.clip(
        0.2 + 0.5 * jl_n + 0.1 * pr_n + rng.normal(0, 0.05, len(df)),
        0.0,
        1.0,
    )

    # UR group derived from race
    df["ur_group"] = np.where(
        df["race"].isin(["Black", "Hispanic", "Other"]),
        "UR",
        "Non-UR",
    )

    return df


# ---------------------------------------------------------
# 2. SIMULATION CONFIG & DATA CLASSES
# ---------------------------------------------------------

@dataclass
class SimulationConfig:
    years: int = 5
    base_voluntary_attrition: float = 0.12
    retirement_age: int = 62
    retirement_shock: float = 0.0
    promotion_rate_ic_to_mid: float = 0.10
    promotion_rate_mid_to_senior: float = 0.08
    external_hiring_rate_mid: float = 0.06
    external_hiring_rate_senior: float = 0.03
    diversity_boost_mid: float = 0.0
    diversity_boost_senior: float = 0.0
    automation_risk: float = 0.0


@dataclass
class YearSnapshot:
    year: int
    headcount_total: int
    headcount_ic: int
    headcount_mid: int
    headcount_senior: int
    required_mid: int
    required_senior: int
    gap_mid: int
    gap_senior: int
    avg_attrition_prob_mid: float
    avg_attrition_prob_senior: float
    skill_coverage_mid: float
    skill_coverage_senior: float
    diversity_mid_share_ur: float
    diversity_senior_share_ur: float
    successors_mid_ready: int
    successors_senior_ready: int
    successors_mid_available: int
    successors_senior_available: int


# ---------------------------------------------------------
# 3. READINESS & ML-BASED ATTRITION
# ---------------------------------------------------------

def compute_readiness_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute readiness for Mid and Senior roles based on:
    - performance_rating
    - years_at_company
    - leadership & strategic skills
    """
    d = df.copy()

    def norm(col):
        cmin, cmax = col.min(), col.max()
        if cmax == cmin:
            return np.ones_like(col, dtype=float)
        return (col - cmin) / (cmax - cmin)

    perf_n = norm(d["performance_rating"])
    tenure_n = norm(d["years_at_company"])
    lead_n = norm(d["skill_leadership"])
    strat_n = norm(d["skill_strategic"])

    d["readiness_mid"] = (
        0.45 * perf_n +
        0.30 * lead_n +
        0.25 * tenure_n
    )

    d["readiness_senior"] = (
        0.40 * perf_n +
        0.25 * lead_n +
        0.15 * tenure_n +
        0.20 * strat_n
    )

    return d


def compute_attrition_probabilities(
    df: pd.DataFrame,
    config: SimulationConfig,
    model
) -> np.ndarray:
    """
    Use your trained ML model (attrition_model.pkl) to predict
    individual attrition probabilities, then apply retirement effects
    and an overall calibration via the 'baseline attrition' slider.
    """
    # Ensure we have all required feature columns
    X = df[MODEL_FEATURE_COLS].copy()

    # ML-predicted attrition probabilities
    ml_probs = model.predict_proba(X)[:, 1]

    # Calibrate up/down with baseline slider
    # Assume 0.12 is the "nominal" baseline used when training / thinking
    base_ref = 0.12
    scale = config.base_voluntary_attrition / base_ref if base_ref > 0 else 1.0
    probs = ml_probs * scale

    # Retirement effect (force higher attrition near retirement age)
    retirement_flag = (df["age"] >= config.retirement_age).astype(float)
    probs = np.where(
        retirement_flag == 1.0,
        np.clip(probs + 0.5 + config.retirement_shock, 0.01, 1.0),
        probs
    )

    # Safety bounds
    probs = np.clip(probs, 0.01, 0.99)
    return probs


# ---------------------------------------------------------
# 4. EXTERNAL HIRES & YEARLY STEP
# ---------------------------------------------------------

def external_hires(
    df: pd.DataFrame,
    hires_mid: int,
    hires_senior: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Add external hires (Mid & Senior) as synthetic rows.
    """
    new_rows = []

    current_max_id = df["emp_id"].max() if len(df) > 0 else 0
    next_id = current_max_id + 1

    def make_hires(n, role_level):
        nonlocal next_id
        rows = []
        for _ in range(n):
            age = rng.normal(40 if role_level != "IC" else 30, 5)
            years_at_company = 0.3  # new join
            perf = rng.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            skill_lead = rng.beta(2.5, 2.0)
            skill_tech = rng.beta(2.0, 2.0)
            skill_strat = rng.beta(2.0, 2.5)

            gender = rng.choice(["Male", "Female"], p=[0.5, 0.5])
            race = rng.choice(RACE_CATEGORIES)

            rows.append({
                "emp_id": next_id,
                "age": np.clip(age, 23, 65),
                "years_at_company": years_at_company,
                "performance_rating": perf,
                "gender": gender,
                "race": race,
                "role_level": role_level,
                "skill_leadership": skill_lead,
                "skill_technical": skill_tech,
                "skill_strategic": skill_strat,
                "ur_group": "UR" if race in ["Black", "Hispanic", "Other"] else "Non-UR"
            })
            next_id += 1
        return rows

    if hires_mid > 0:
        new_rows.extend(make_hires(hires_mid, "Mid"))
    if hires_senior > 0:
        new_rows.extend(make_hires(hires_senior, "Senior"))

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df


def simulate_year(
    df: pd.DataFrame,
    config: SimulationConfig,
    rng: np.random.Generator,
    base_required_mid: int,
    base_required_senior: int,
    model
) -> Tuple[pd.DataFrame, YearSnapshot]:
    """
    Run one simulation year:
    - attrition (using ML model + retirement logic)
    - promotions IC->Mid, Mid->Senior
    - external hiring to close gaps
    - compute bench strength & diversity metrics
    """

    # Compute attrition probabilities
    probs = compute_attrition_probabilities(df, config, model)
    leave_flags = rng.binomial(n=1, p=probs).astype(bool)

    # Compute readiness to track "ready vs available" successors
    df_with_readiness = compute_readiness_scores(df)
    successors_mid_mask = (
        (df_with_readiness["role_level"] == "IC") &
        (df_with_readiness["readiness_mid"] >= 0.7)
    )
    successors_senior_mask = (
        (df_with_readiness["role_level"] == "Mid") &
        (df_with_readiness["readiness_senior"] >= 0.7)
    )

    successors_mid_ready = int(successors_mid_mask.sum())
    successors_senior_ready = int(successors_senior_mask.sum())

    successors_mid_available = int((successors_mid_mask & ~leave_flags).sum())
    successors_senior_available = int((successors_senior_mask & ~leave_flags).sum())

    # Apply attrition
    df_after_attrition = df.loc[~leave_flags].copy()

    # Update age & years at company
    df_after_attrition["age"] += 1
    df_after_attrition["years_at_company"] += 1

    # Recompute readiness after attrition and time progression
    df_after_attrition = compute_readiness_scores(df_after_attrition)

    # Promotions IC -> Mid
    ic_mask = df_after_attrition["role_level"] == "IC"
    candidates_ic = df_after_attrition[ic_mask].copy()
    candidates_ic = candidates_ic.sort_values("readiness_mid", ascending=False)

    n_promote_ic = int(config.promotion_rate_ic_to_mid * len(candidates_ic))
    to_promote_ic_ids = candidates_ic.head(n_promote_ic)["emp_id"].tolist()

    df_after_attrition.loc[
        df_after_attrition["emp_id"].isin(to_promote_ic_ids),
        "role_level"
    ] = "Mid"

    # Promotions Mid -> Senior, diversity aware
    mid_mask = df_after_attrition["role_level"] == "Mid"
    candidates_mid = df_after_attrition[mid_mask].copy()
    candidates_mid = candidates_mid.sort_values("readiness_senior", ascending=False)

    base_promos = int(config.promotion_rate_mid_to_senior * len(candidates_mid))
    mid_ur_mask = (candidates_mid["ur_group"] == "UR")
    extra_ur_promos = int(config.diversity_boost_senior * len(candidates_mid[mid_ur_mask]))
    n_promote_mid = base_promos + extra_ur_promos

    to_promote_mid_ids = candidates_mid.head(n_promote_mid)["emp_id"].tolist()

    df_after_attrition.loc[
        df_after_attrition["emp_id"].isin(to_promote_mid_ids),
        "role_level"
    ] = "Senior"

    # Demand with automation risk (role obsolescence)
    demand_mid = int(base_required_mid * (1 - config.automation_risk))
    demand_senior = int(base_required_senior * (1 - config.automation_risk))

    # Current role counts
    count_ic = (df_after_attrition["role_level"] == "IC").sum()
    count_mid = (df_after_attrition["role_level"] == "Mid").sum()
    count_senior = (df_after_attrition["role_level"] == "Senior").sum()
    total = len(df_after_attrition)

    gap_mid = demand_mid - count_mid
    gap_senior = demand_senior - count_senior

    # External hiring to close gaps
    hires_mid = max(gap_mid, 0)
    hires_senior = max(gap_senior, 0)
    df_after_attrition = external_hires(
        df_after_attrition,
        hires_mid=hires_mid,
        hires_senior=hires_senior,
        rng=rng
    )

    # Recalculate counts after hiring
    count_ic = (df_after_attrition["role_level"] == "IC").sum()
    count_mid = (df_after_attrition["role_level"] == "Mid").sum()
    count_senior = (df_after_attrition["role_level"] == "Senior").sum()
    total = len(df_after_attrrition := df_after_attrition)

    # Skill coverage – share of Mid/Senior above readiness thresholds
    df_after_attrrition = compute_readiness_scores(df_after_attrrition)
    mid_people = df_after_attrrition[df_after_attrrition["role_level"] == "Mid"]
    senior_people = df_after_attrrition[df_after_attrrition["role_level"] == "Senior"]

    skill_cov_mid = (mid_people["readiness_mid"] >= 0.7).mean() if len(mid_people) > 0 else 0.0
    skill_cov_senior = (senior_people["readiness_senior"] >= 0.7).mean() if len(senior_people) > 0 else 0.0

    # Diversity representation
    diversity_mid_share_ur = (mid_people["ur_group"] == "UR").mean() if len(mid_people) > 0 else 0.0
    diversity_senior_share_ur = (senior_people["ur_group"] == "UR").mean() if len(senior_people) > 0 else 0.0

    # Attrition probability averages by role (for reporting only)
    probs_mid = compute_attrition_probabilities(mid_people, config, model) if len(mid_people) > 0 else np.array([0.0])
    probs_senior = compute_attrition_probabilities(senior_people, config, model) if len(senior_people) > 0 else np.array([0.0])

    snapshot = YearSnapshot(
        year=0,  # filled later by caller
        headcount_total=int(total),
        headcount_ic=int(count_ic),
        headcount_mid=int(count_mid),
        headcount_senior=int(count_senior),
        required_mid=int(demand_mid),
        required_senior=int(demand_senior),
        gap_mid=int(demand_mid - count_mid),
        gap_senior=int(demand_senior - count_senior),
        avg_attrition_prob_mid=float(probs_mid.mean()),
        avg_attrition_prob_senior=float(probs_senior.mean()),
        skill_coverage_mid=float(skill_cov_mid),
        skill_coverage_senior=float(skill_cov_senior),
        diversity_mid_share_ur=float(diversity_mid_share_ur),
        diversity_senior_share_ur=float(diversity_senior_share_ur),
        successors_mid_ready=int(successors_mid_ready),
        successors_senior_ready=int(successors_senior_ready),
        successors_mid_available=int(successors_mid_available),
        successors_senior_available=int(successors_senior_available),
    )

    return df_after_attrrition, snapshot


def run_simulation(
    initial_df: pd.DataFrame,
    config: SimulationConfig,
    model,
    random_state: int = 123
) -> Tuple[List[YearSnapshot], pd.DataFrame]:
    """Run multi-year simulation."""
    rng = np.random.default_rng(random_state)
    df = initial_df.copy()

    df = compute_readiness_scores(df)

    base_required_mid = (df["role_level"] == "Mid").sum()
    base_required_senior = (df["role_level"] == "Senior").sum()

    snapshots: List[YearSnapshot] = []

    for year in range(1, config.years + 1):
        df, snap = simulate_year(
            df=df,
            config=config,
            rng=rng,
            base_required_mid=base_required_mid,
            base_required_senior=base_required_senior,
            model=model
        )
        snap.year = year
        snapshots.append(snap)

    snap_df = pd.DataFrame([s.__dict__ for s in snapshots])
    return snapshots, snap_df


def static_successors(df: pd.DataFrame) -> Dict[str, int]:
    """
    Static snapshot of "ready now" successors — ignores flow.
    """
    d = compute_readiness_scores(df)
    static_mid = ((d["role_level"] == "IC") & (d["readiness_mid"] >= 0.7)).sum()
    static_senior = ((d["role_level"] == "Mid") & (d["readiness_senior"] >= 0.7)).sum()

    return {
        "static_mid": int(static_mid),
        "static_senior": int(static_senior)
    }


# ---------------------------------------------------------
# 5. VISUALIZATION HELPERS
# ---------------------------------------------------------

def line_chart_from_snapshots(df: pd.DataFrame, cols: List[str], title: str):
    st.subheader(title)
    st.line_chart(df.set_index("year")[cols])


# ---------------------------------------------------------
# 6. PAGES
# ---------------------------------------------------------

def page_overview(initial_df: pd.DataFrame):
    st.title("Interactive Workforce Simulation (Digital Twin-Inspired)")

    st.markdown("""
This tool is an **interactive workforce simulation** inspired by digital twin principles.  
Instead of relying on static succession lists, it models **flows** over time:

- Employee **attrition** (predicted by an ML model trained on IBM HR data)
- **Promotions** from IC → Mid → Senior
- **External hiring** to close emerging gaps
- **Skill coverage** and **diversity representation** in leadership roles

It is **not** a full enterprise digital twin (no live HRIS integration), but a **proof-of-concept**
meant to show why **dynamic simulation** reveals vulnerabilities that **static succession planning** misses.
""")

    st.markdown("### 📘 Methodology & Assumptions")
    with st.expander("Click to view methodology, assumptions & limitations"):
        st.markdown("""
**Data & Model**

- Underlying dataset: IBM HR Attrition (structured employee snapshot).
- Attrition model: Scikit-learn Pipeline (`attrition_model.pkl`) trained on:
  - Age  
  - Years at company  
  - Performance rating  
  - Gender  
  - Race (synthetic but realistic categories)  
  - Role level (IC / Mid / Senior)  
- The model outputs individual **attrition probabilities**, which the simulation uses each year.

**Succession & readiness logic**

- Readiness for Mid and Senior roles uses:
  - Performance rating  
  - Years at company  
  - Leadership & strategic skills (synthetic but correlated with level & rating)
- Promotions are capacity-constrained and parameterized via sliders.

**Demand & automation**

- Baseline demand for Mid and Senior roles equals the current headcount.
- The **Automation Risk** slider reduces demand over time to simulate role obsolescence.

**Limitations**

- No live HRIS integration → this is a **simulation**, not a production digital twin.
- Skills and race attributes are partly synthetic for demo purposes.
- Attrition drivers beyond the modeled features are not included.
- Real-world deployment would require:
  - Organization-specific data
  - Historical validation
  - Integration into HR workflows.
""")

    st.markdown("### 👀 Baseline Workforce Snapshot")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total employees", len(initial_df))
        st.metric("IC headcount", int((initial_df["role_level"] == "IC").sum()))
        st.metric("Mid-level headcount", int((initial_df["role_level"] == "Mid").sum()))
        st.metric("Senior headcount", int((initial_df["role_level"] == "Senior").sum()))

    with col2:
        mid_ur = ((initial_df["role_level"] == "Mid") & (initial_df["ur_group"] == "UR")).mean()
        senior_ur = ((initial_df["role_level"] == "Senior") & (initial_df["ur_group"] == "UR")).mean()
        st.metric("UR share in mid-level roles", f"{mid_ur * 100:.1f}%")
        st.metric("UR share in senior roles", f"{senior_ur * 100:.1f}%")


def page_static_vs_dynamic(initial_df: pd.DataFrame, config: SimulationConfig, model):
    st.title("Static Succession vs Dynamic Simulation")

    st.markdown("""
Static succession planning often answers:

> *“How many ready successors do we have **today** for each role?”*

But it **ignores**:

- Who leaves before they can be promoted  
- Promotion velocity and bottlenecks  
- Retirement shocks  
- Automation-driven demand shifts  

This page compares a **static snapshot** to **multi-year simulation**.
""")

    snapshots, snap_df = run_simulation(initial_df, config, model)
    static = static_successors(initial_df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧊 Static 'ready-now' successors")
        st.metric("IC → Mid static successors", static["static_mid"])
        st.metric("Mid → Senior static successors", static["static_senior"])
        st.write("Static lists typically show these numbers as **constant each year**.")

    with col2:
        st.markdown("#### 🔄 Simulated available successors over time")
        st.line_chart(
            snap_df.set_index("year")[
                ["successors_mid_available", "successors_senior_available"]
            ]
        )

    st.markdown("""
In your presentation, you can narrate a concrete example:

> “Static planning says we have 5 ready successors for a senior role.  
> After simulating attrition, promotions, and retirement shocks, only 2 are actually available
> when the role opens — a **60% shortfall**. This is exactly the blind spot of static succession planning.”
""")


def page_succession_planning(initial_df: pd.DataFrame, config: SimulationConfig, model):
    st.title("Succession Planning & Bench Strength")

    st.markdown("""
This page focuses on **succession quality**:

- Bench strength (successors per role)  
- Pipeline leakage (successors who exit before promotion)  
- Readiness over time  
- Diversity within the succession pool  
""")

    snapshots, snap_df = run_simulation(initial_df, config, model)

    initial_mid_roles = (initial_df["role_level"] == "Mid").sum()
    initial_senior_roles = (initial_df["role_level"] == "Senior").sum()

    last = snap_df.iloc[-1]

    bench_mid = last["successors_mid_available"] / max(initial_mid_roles, 1)
    bench_senior = last["successors_senior_available"] / max(initial_senior_roles, 1)

    def bench_label(score):
        if score >= 1.5:
            return "Strong"
        elif score >= 0.75:
            return "Moderate"
        else:
            return "Weak"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bench strength at end of horizon")
        st.metric(
            "Mid-level bench strength (successors/role)",
            f"{bench_mid:.2f}",
            bench_label(bench_mid)
        )
        st.metric(
            "Senior bench strength (successors/role)",
            f"{bench_senior:.2f}",
            bench_label(bench_senior)
        )

    with col2:
        st.subheader("Pipeline leakage")
        base_mid_ready = snap_df.iloc[0]["successors_mid_ready"]
        base_senior_ready = snap_df.iloc[0]["successors_senior_ready"]

        leak_mid = base_mid_ready - last["successors_mid_available"]
        leak_senior = base_senior_ready - last["successors_senior_available"]

        st.metric("Mid successors lost before promotion", int(leak_mid))
        st.metric("Senior successors lost before promotion", int(leak_senior))

    st.markdown("### 📈 Available successors over time")
    line_chart_from_snapshots(
        snap_df,
        ["successors_mid_available", "successors_senior_available"],
        "Available successors (IC→Mid and Mid→Senior) by year"
    )

    st.markdown("""
You can phrase this in your conclusion as:

> “Bench strength isn't just how many successors we list today.  
> It’s how many remain in the organization **and are still ready** when roles actually open.
> Our simulation turns abstract ideas like *bench strength* and *pipeline leakage* into concrete, year-by-year numbers.”
""")


def page_skills_diversity(initial_df: pd.DataFrame, config: SimulationConfig, model):
    st.title("Skill Coverage & Diversity Representation")

    st.markdown("""
Many succession tools track **titles**, not **skills**.  
This page asks:

- Do promoted leaders actually meet readiness thresholds?  
- Does the leadership pipeline reflect **underrepresented groups (UR)** fairly?  
""")

    snapshots, snap_df = run_simulation(initial_df, config, model)

    col1, col2 = st.columns(2)
    with col1:
        line_chart_from_snapshots(
            snap_df,
            ["skill_coverage_mid", "skill_coverage_senior"],
            "Share of Mid/Senior leaders meeting readiness thresholds"
        )

    with col2:
        line_chart_from_snapshots(
            snap_df,
            ["diversity_mid_share_ur", "diversity_senior_share_ur"],
            "UR share in Mid & Senior roles over time"
        )

    st.markdown("""
### 🔎 Connecting the Diversity sliders to research

The **Diversity Boost** sliders in the sidebar approximate interventions like:

- Sponsorship & mentorship programs  
- Equitable performance calibration  
- Transparent promotion criteria  
- Leadership development targeted at UR talent  

In your write-up, you can say:

> “Our simulation operationalizes the literature on structural barriers by allowing HR
> to test how targeted interventions (e.g., sponsorship for UR talent) change promotion
> flows and leadership representation over time.”
""")


def page_sandbox_and_limits(initial_df: pd.DataFrame, config: SimulationConfig, model):
    st.title("Scenario Sandbox & Limitations")

    st.markdown("""
This page works as a **what-if sandbox**.  
Try scenarios like:

- Lowering retirement age  
- Increasing promotion velocity  
- Doubling external hiring  
- Adding strong diversity interventions  
- Applying automation risk to mid-level demand  
""")

    snapshots, snap_df = run_simulation(initial_df, config, model)

    st.subheader("Headcount vs demand over time")
    col1, col2 = st.columns(2)
    with col1:
        line_chart_from_snapshots(
            snap_df,
            ["headcount_mid", "required_mid"],
            "Mid-level headcount vs required demand"
        )
    with col2:
        line_chart_from_snapshots(
            snap_df,
            ["headcount_senior", "required_senior"],
            "Senior headcount vs required demand"
        )

    st.subheader("Leadership gaps over time")
    line_chart_from_snapshots(
        snap_df,
        ["gap_mid", "gap_senior"],
        "Leadership gaps (negative values = shortages)"
    )

    st.markdown("""
### 📌 How to talk about limitations (for your report / pitch)

You can acknowledge:

- **Attrition modeling is still simplified**:  
  It uses your ML model but does not capture every possible driver.
- **Race and skills are partly synthetic**:  
  Real deployment would use actual DEI and capability data.
- **No live HRIS integration yet**:  
  This is a **simulation demo**, not a production system.
- **Automation risk is coarse**:  
  Modeled as a demand reduction factor rather than job-family level analysis.

Then conclude:

> “Even with these simplifications, the simulation already reveals
> dynamic vulnerabilities — mid-level leakage, diversity shortfalls,
> and timing misalignments — that static succession methods simply cannot see.”
""")


# ---------------------------------------------------------
# 7. SIDEBAR CONTROLS
# ---------------------------------------------------------

def sidebar_controls() -> SimulationConfig:
    st.sidebar.title("Simulation Controls")

    years = st.sidebar.slider("Simulation horizon (years)", 2, 10, 5)

    st.sidebar.markdown("### Attrition & Retirement")
    base_attr = st.sidebar.slider("Baseline attrition calibration", 0.05, 0.25, 0.12, 0.01)
    retirement_age = st.sidebar.slider("Retirement age", 55, 70, 62, 1)
    retirement_shock = st.sidebar.slider("Retirement shock (extra probability)", 0.0, 0.30, 0.0, 0.05)

    st.sidebar.markdown("### Promotions")
    promo_ic_mid = st.sidebar.slider("IC → Mid promotion rate", 0.00, 0.30, 0.10, 0.01)
    promo_mid_sen = st.sidebar.slider("Mid → Senior promotion rate", 0.00, 0.25, 0.08, 0.01)

    st.sidebar.markdown("### External Hiring")
    hire_mid = st.sidebar.slider("External hiring rate for Mid (used for gaps)", 0.00, 0.20, 0.06, 0.01)
    hire_sen = st.sidebar.slider("External hiring rate for Senior (used for gaps)", 0.00, 0.15, 0.03, 0.01)

    st.sidebar.markdown("### Diversity Interventions")
    diversity_boost_mid = st.sidebar.slider(
        "Diversity boost for Mid promotions (UR candidates)",
        0.0, 0.30, 0.0, 0.05
    )
    diversity_boost_senior = st.sidebar.slider(
        "Diversity boost for Senior promotions (UR candidates)",
        0.0, 0.30, 0.0, 0.05
    )

    st.sidebar.info(
        "The Diversity Boost sliders approximate structural interventions like "
        "sponsorship, equitable calibration, and targeted leadership programs for "
        "underrepresented employees."
    )

    st.sidebar.markdown("### Automation Risk (Future of Work)")
    automation_risk = st.sidebar.slider(
        "Automation risk affecting demand for Mid/Senior roles",
        0.0, 0.50, 0.0, 0.05
    )

    cfg = SimulationConfig(
        years=years,
        base_voluntary_attrition=base_attr,
        retirement_age=retirement_age,
        retirement_shock=retirement_shock,
        promotion_rate_ic_to_mid=promo_ic_mid,
        promotion_rate_mid_to_senior=promo_mid_sen,
        external_hiring_rate_mid=hire_mid,
        external_hiring_rate_senior=hire_sen,
        diversity_boost_mid=diversity_boost_mid,
        diversity_boost_senior=diversity_boost_senior,
        automation_risk=automation_risk
    )
    return cfg


# ---------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------

def main():
    model = load_attrition_model()
    df_initial = load_base_dataframe()
    config = sidebar_controls()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview & Methodology",
            "Static vs Dynamic",
            "Succession Planning",
            "Skills & Diversity",
            "Scenario Sandbox & Limitations",
        ]
    )

    if page == "Overview & Methodology":
        page_overview(df_initial)
    elif page == "Static vs Dynamic":
        page_static_vs_dynamic(df_initial, config, model)
    elif page == "Succession Planning":
        page_succession_planning(df_initial, config, model)
    elif page == "Skills & Diversity":
        page_skills_diversity(df_initial, config, model)
    elif page == "Scenario Sandbox & Limitations":
        page_sandbox_and_limits(df_initial, config, model)


if __name__ == "__main__":
    main()

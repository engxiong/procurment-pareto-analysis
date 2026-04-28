"""
Procurement Pareto Dashboard
-----------------------------
Run:  streamlit run procurement_dashboard.py

Chart 1 — Pareto Curve:
    X = Cumulative value %,  Y = Cumulative volume %

Chart 2 — Distribution Curve (matches whiteboard sketch):
    X = Actual value ($), Y = Cumulative volume count
    - Vertical threshold line (orange dashed) — driven by slider
    - Horizontal target line (purple dotted)  — driven by vol_target_pct slider
    - Purple ◆ marks the exact $ threshold needed to hit the volume target
    - Shaded tail region left of threshold
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procurement Pareto Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── palette ───────────────────────────────────────────────────────────────────
BLUE   = "#185FA5"
TEAL   = "#1D9E75"
ORANGE = "#D85A30"
PURPLE = "#7B2D8B"
GRAY   = "#888780"


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #fafaf8; }
[data-testid="stSidebar"]          { background: #f5f4f0; border-right: 1px solid #e0ded6; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def fmt_cur(v):
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def compute_pareto(df, value_col):
    """Sort ascending; return cumulative vol % and val % columns."""
    s = df[[value_col]].copy().sort_values(value_col).reset_index(drop=True)
    n   = len(s)
    tv  = s[value_col].sum()
    s["cum_vol_pct"] = np.arange(1, n + 1) / n * 100
    s["cum_val_pct"] = s[value_col].cumsum() / tv * 100
    orig = pd.DataFrame({value_col: [0], "cum_vol_pct": [0.0], "cum_val_pct": [0.0]})
    return pd.concat([orig, s], ignore_index=True), tv, n


def compute_distribution(df, value_col):
    """Sort ascending; Y = cumulative count of projects."""
    s = df[[value_col]].copy().sort_values(value_col).reset_index(drop=True)
    s["cum_count"] = np.arange(1, len(s) + 1)
    orig = pd.DataFrame({value_col: [0], "cum_count": [0]})
    return pd.concat([orig, s], ignore_index=True)


def classify(df, value_col, threshold):
    a = df[df[value_col] >= threshold]
    b = df[(df[value_col] >= threshold * 0.5) & (df[value_col] < threshold)]
    c = df[df[value_col] < threshold * 0.5]
    return a, b, c


def class_stats(subset, total_n, total_val, value_col):
    n    = len(subset)
    vsum = subset[value_col].sum() if n else 0
    return n, n / total_n * 100, vsum, vsum / total_val * 100 if total_val else 0


def pill(label, color, bg, border, n, vol_pct, val_pct, spend):
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:8px;
                padding:10px 14px;margin-bottom:8px;">
      <div style="font-weight:600;color:{color};font-size:13px;">{label}</div>
      <div style="font-size:20px;font-weight:700;color:{color};margin:4px 0;">{n:,}</div>
      <div style="font-size:11px;color:{color};opacity:0.85;">{vol_pct:.1f}% of volume</div>
      <div style="font-size:11px;color:{color};opacity:0.85;">{val_pct:.1f}% of value</div>
      <div style="font-size:11px;color:{color};opacity:0.85;margin-top:4px;">{spend}</div>
    </div>""", unsafe_allow_html=True)


# ── sample data ───────────────────────────────────────────────────────────────
def sample_data(n=80, seed=42):
    rng = np.random.default_rng(seed)
    values = np.clip(rng.lognormal(9.5, 1.4, n), 500, 800_000).round(-2)
    return pd.DataFrame({
        "Project":    [f"PO-{i+1:04d}" for i in range(n)],
        "Department": rng.choice(["IT","Operations","HR","Finance","Marketing","Facilities"], n),
        "Category":   rng.choice(["Software","Hardware","Services","Consumables","Consulting","Maintenance"], n),
        "Region":     rng.choice(["APAC","EMEA","Americas"], n),
        "Supplier":   [f"Supplier {chr(65+rng.integers(0,12))}" for _ in range(n)],
        "Value":      values,
        "Year":       rng.choice([2022, 2023, 2024], n),
    })


# ── Chart 1: Pareto ───────────────────────────────────────────────────────────
def make_pareto_chart(curve_df, thresh_vol_pct, thresh_val_pct):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve_df["cum_val_pct"], y=curve_df["cum_vol_pct"],
        mode="lines", name="Cumulative curve",
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(24,95,165,0.07)",
        hovertemplate="Value: %{x:.1f}%<br>Volume: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines", name="Equal spread",
        line=dict(color="rgba(136,135,128,0.35)", width=1, dash="dash"),
        hoverinfo="skip",
    ))
    # vertical threshold
    fig.add_trace(go.Scatter(
        x=[thresh_val_pct, thresh_val_pct], y=[0, thresh_vol_pct],
        mode="lines", name="Value threshold",
        line=dict(color=ORANGE, width=1.8, dash="dash"), hoverinfo="skip",
    ))
    # horizontal threshold
    fig.add_trace(go.Scatter(
        x=[0, thresh_val_pct], y=[thresh_vol_pct, thresh_vol_pct],
        mode="lines", name="Volume at threshold",
        line=dict(color=TEAL, width=1.8, dash="dash"), hoverinfo="skip",
    ))
    # intersection dot
    fig.add_trace(go.Scatter(
        x=[thresh_val_pct], y=[thresh_vol_pct], mode="markers", name="Intersection",
        marker=dict(color=ORANGE, size=10, line=dict(color="white", width=2)),
        hovertemplate=(f"<b>Threshold point</b><br>Value: {thresh_val_pct:.1f}%"
                       f"<br>Volume: {thresh_vol_pct:.1f}%<extra></extra>"),
    ))

    fig.update_layout(
        title=dict(text="Chart 1 — Pareto Curve: Cumulative Volume % vs Cumulative Value %",
                   font=dict(size=13, color="#2C2C2A")),
        xaxis=dict(title="Cumulative value (%)", range=[0, 100],
                   ticksuffix="%", gridcolor="#eeede8", zeroline=False),
        yaxis=dict(title="Cumulative volume (%)", range=[0, 100],
                   ticksuffix="%", gridcolor="#eeede8", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=10)),
        margin=dict(l=55, r=20, t=60, b=50), height=420, hovermode="closest",
    )
    return fig


# ── Chart 2: Distribution ─────────────────────────────────────────────────────
def make_distribution_chart(dist_df, value_col, threshold, vol_target_pct, total_n):
    """
    Y = cumulative count of projects (not %)
    X = actual value ($)
    The % target is a parameter — the chart translates it into an absolute count.
    """
    # count at current threshold
    tail_df          = dist_df[dist_df[value_col] <= threshold]
    count_at_thresh  = int(tail_df["cum_count"].max()) if not tail_df.empty else 0
    pct_at_thresh    = count_at_thresh / total_n * 100

    # absolute count for the volume target
    vol_target_count = total_n * vol_target_pct / 100

    # find the value where the curve first reaches the target count
    target_rows  = dist_df[dist_df["cum_count"] >= vol_target_count]
    target_value = float(target_rows[value_col].iloc[0]) if not target_rows.empty else None

    fig = go.Figure()

    # ── shaded tail ──────────────────────────────────────────────────────────
    if not tail_df.empty:
        sx = list(tail_df[value_col]) + [threshold, 0]
        sy = list(tail_df["cum_count"]) + [0, 0]
        fig.add_trace(go.Scatter(
            x=sx, y=sy, fill="toself",
            fillcolor="rgba(216,90,48,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Low-value tail (below threshold)",
            hoverinfo="skip",
        ))

    # ── main curve ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dist_df[value_col], y=dist_df["cum_count"],
        mode="lines", name="Cumulative volume",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="value: $%{x:,.0f}<br>Cumulative count: %{y:,}<extra></extra>",
    ))

    # ── threshold vertical line ───────────────────────────────────────────────
    fig.add_vline(
        x=threshold,
        line=dict(color=ORANGE, width=2, dash="dash"),
        annotation=dict(
            text=f"<b>Threshold: {fmt_cur(threshold)}</b>",
            font=dict(color=ORANGE, size=11),
            bgcolor="white", bordercolor=ORANGE, borderwidth=1, borderpad=4,
            yanchor="top", y=0.97, xanchor="left",
        ),
    )

    # ── horizontal line: current count at threshold ───────────────────────────
    fig.add_hline(
        y=count_at_thresh,
        line=dict(color=TEAL, width=1.5, dash="dash"),
        annotation=dict(
            text=f"<b>{count_at_thresh} projects = {pct_at_thresh:.1f}% of volume</b>",
            font=dict(color=TEAL, size=10), bgcolor="white",
            xanchor="right", x=0.99,
        ),
    )

    # ── volume target line (% is the variable input) ──────────────────────────
    fig.add_hline(
        y=vol_target_count,
        line=dict(color=PURPLE, width=1.8, dash="dot"),
        annotation=dict(
            text=f"<b>{vol_target_pct}% target = {int(vol_target_count)} projects</b>",
            font=dict(color=PURPLE, size=10), bgcolor="white",
            xanchor="left", x=0.01,
        ),
    )

    # ── orange dot: intersection of threshold with curve ─────────────────────
    fig.add_trace(go.Scatter(
        x=[threshold], y=[count_at_thresh],
        mode="markers", name="Current threshold point",
        marker=dict(color=ORANGE, size=11, line=dict(color="white", width=2)),
        hovertemplate=(
            f"<b>Current threshold: {fmt_cur(threshold)}</b><br>"
            f"projects ≤ threshold: {count_at_thresh} ({pct_at_thresh:.1f}%)<extra></extra>"
        ),
    ))

    # ── purple diamond: optimal threshold to hit the target ───────────────────
    if target_value is not None and target_value > 0:
        fig.add_trace(go.Scatter(
            x=[target_value], y=[vol_target_count],
            mode="markers",
            name=f"Optimal threshold for {vol_target_pct}% target",
            marker=dict(color=PURPLE, size=13, symbol="diamond",
                        line=dict(color="white", width=2)),
            hovertemplate=(
                f"<b>To reach {vol_target_pct}% volume:</b><br>"
                f"Set threshold ≥ {fmt_cur(target_value)}<extra></extra>"
            ),
        ))
        # vertical drop line from diamond to x-axis
        fig.add_shape(
            type="line",
            x0=target_value, x1=target_value, y0=0, y1=vol_target_count,
            line=dict(color=PURPLE, width=1.2, dash="dot"),
        )
        fig.add_annotation(
            x=target_value, y=0,
            text=f"<b>{fmt_cur(target_value)}</b>",
            showarrow=False, font=dict(color=PURPLE, size=10),
            yanchor="top", yshift=-10, bgcolor="white",
        )

    fig.update_layout(
        title=dict(
            text=(
                "Chart 2 — Distribution: Cumulative Volume vs Value  "
                f"<span style='font-size:11px;color:#888780'>"
                f"| Target: {vol_target_pct}% of volume ({int(vol_target_count)} projects)"
                f"</span>"
            ),
            font=dict(size=13, color="#2C2C2A"),
        ),
        xaxis=dict(title="Value ($)", gridcolor="#eeede8",
                   zeroline=False, tickprefix="$", tickformat=",.0f"),
        yaxis=dict(title="Cumulative number of projects", gridcolor="#eeede8", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=10)),
        margin=dict(l=55, r=20, t=70, b=60), height=460, hovermode="closest",
    )
    return fig, count_at_thresh, pct_at_thresh, target_value


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Procurement Pareto")
    st.markdown("---")

    uploaded = st.file_uploader("Upload Excel / CSV file", type=["xlsx", "xls", "csv"])

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Sheet", xls.sheet_names)
                df_raw = pd.read_excel(uploaded, sheet_name=sheet)
            st.success(f"{len(df_raw):,} rows loaded")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df_raw = sample_data()
    else:
        df_raw = sample_data()
        st.info("Using built-in sample data (80 projects).")

    st.markdown("---")
    st.markdown("**Column setup**")
    num_cols = df_raw.select_dtypes(include="number").columns.tolist()
    all_cols = df_raw.columns.tolist()

    if not num_cols:
        st.error("No numeric columns found.")
        st.stop()

    default_val = "Value" if "Value" in num_cols else num_cols[0]
    value_col = st.selectbox("Value column (PO spend)", num_cols,
                              index=num_cols.index(default_val))

    st.markdown("---")
    st.markdown("**Filters / slicers**")

    # ── reset button ──────────────────────────────────────────────────────────
    def _reset_all():
        for key in ["_filter_col", "_filter_vals", "threshold_val",
                    "_filter_vals_data", "_vol_target"]:
            if key in st.session_state:
                del st.session_state[key]

    st.button("↺  Reset all filters & threshold", use_container_width=True,
              on_click=_reset_all,
              help="Resets slice column, multi-select values, threshold, and volume target to defaults")

    st.markdown("")

    filter_col = st.selectbox(
        "Slice by",
        ["(none)"] + [c for c in all_cols if c != value_col],
        key="_filter_col",
    )
    filter_vals = []
    if filter_col != "(none)":
        opts = sorted(df_raw[filter_col].dropna().unique().tolist(), key=str)
        stored = st.session_state.get("_filter_vals_data", opts)
        valid_stored = [v for v in stored if v in opts]
        filter_vals = st.multiselect(
            f"Select {filter_col}", opts,
            default=valid_stored if valid_stored else opts,
            key="_filter_vals",
        )
        st.session_state["_filter_vals_data"] = filter_vals

    st.markdown("---")
    st.markdown("**Threshold  *(Chart 1 & 2)***")
    vmin = float(df_raw[value_col].min())
    vmax = float(df_raw[value_col].max())
    step = max(1.0, (vmax - vmin) / 200)
    default_thresh = float(np.percentile(df_raw[value_col], 72))

    # ── single source of truth ────────────────────────────────────────────────
    # Both widgets share the key "threshold_val" directly.
    # Streamlit reads from and writes to session_state["threshold_val"] for
    # both — no secondary sync variable needed, no on_change callbacks needed.
    # We only set a default when the key is absent (first load or after reset).
    if "threshold_val" not in st.session_state:
        st.session_state["threshold_val"] = default_thresh

    # Clamp in case the data range changed (e.g. different file uploaded)
    st.session_state["threshold_val"] = float(
        np.clip(st.session_state["threshold_val"], vmin, vmax)
    )

    # ── Slider: owns the single source-of-truth key ──────────────────────────
    # Streamlit writes the slider value directly into session_state["threshold_val"].
    st.slider(
        "Slide to set threshold",
        min_value=vmin, max_value=vmax,
        step=step,
        format="$%.0f",
        key="threshold_val",
        label_visibility="collapsed",
    )

    # ── Number input: reads from the shared key, writes back via on_change ───
    # Uses its own internal key "_thresh_num" to avoid DuplicateWidgetID.
    # on_change copies its value into "threshold_val", keeping both in sync.
    # We set its value from session_state["threshold_val"] every render so it
    # always reflects whatever the slider is showing.
    st.session_state["_thresh_num"] = st.session_state["threshold_val"]

    def _sync_num_to_slider():
        val = float(np.clip(st.session_state["_thresh_num"], vmin, vmax))
        st.session_state["threshold_val"] = val
        st.session_state["_thresh_num"]   = val  # keep input tidy if clamped

    st.number_input(
        "Or type an exact value ($)",
        min_value=vmin, max_value=vmax,
        step=step,
        format="%.0f",
        key="_thresh_num",
        on_change=_sync_num_to_slider,
    )

    threshold = st.session_state["threshold_val"]
    st.caption(
        f"Threshold: **{fmt_cur(threshold)}**"
        f"  ·  range {fmt_cur(vmin)} – {fmt_cur(vmax)}"
    )

    st.markdown("---")
    st.markdown("**Volume target %  *(Chart 2 only)***")
    vol_target_pct = st.slider(
        "Target volume %",
        min_value=10, max_value=95, value=60, step=5, format="%d%%", key="_vol_target",
        help=(
            "The purple dotted line on Chart 2 is drawn at this % of total volume. "
            "The purple ◆ shows the exact $ threshold needed to contain that many projects."
        ),
    )
    st.caption(
        f"Target: **{vol_target_pct}%** of projects ≤ threshold. "
        "Move the threshold slider until the orange dot meets the purple ◆."
    )


# ══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ══════════════════════════════════════════════════════════════════════════════
df = df_raw.copy()
if filter_col != "(none)" and filter_vals:
    df = df[df[filter_col].isin(filter_vals)]

df = df.dropna(subset=[value_col])
df = df[df[value_col] > 0]

if df.empty:
    st.warning("No data after filters. Adjust your selection.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════════════════
curve_df, total_val, total_n = compute_pareto(df, value_col)
dist_df                      = compute_distribution(df, value_col)
class_a, class_b, class_c    = classify(df, value_col, threshold)

a_n, a_vol_pct, a_vsum, a_val_pct = class_stats(class_a, total_n, total_val, value_col)
b_n, b_vol_pct, b_vsum, b_val_pct = class_stats(class_b, total_n, total_val, value_col)
c_n, c_vol_pct, c_vsum, c_val_pct = class_stats(class_c, total_n, total_val, value_col)

# pareto threshold position
thresh_row     = (curve_df[curve_df[value_col] <= threshold].iloc[-1]
                  if not curve_df[curve_df[value_col] <= threshold].empty
                  else curve_df.iloc[1])
thresh_vol_pct = thresh_row["cum_vol_pct"]
thresh_val_pct = thresh_row["cum_val_pct"]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════
title_sfx = (f" — {filter_col}: {', '.join(str(v) for v in filter_vals)}"
             if filter_col != "(none)" and filter_vals else "")
st.markdown(f"## Procurement Pareto Analysis{title_sfx}")
st.markdown(
    f"**{total_n:,} projects** · "
    f"Total spend **{fmt_cur(total_val)}** · "
    f"Threshold **{fmt_cur(threshold)}**"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, (lbl, val, sub) in zip(
    [c1, c2, c3, c4, c5, c6],
    [
        ("Threshold",      fmt_cur(threshold),        "value cutoff"),
        ("Total projects", f"{total_n:,}",          "in filtered set"),
        ("Class A count",  f"{a_n:,}",              f"{a_vol_pct:.1f}% of volume"),
        ("Cum. volume",    f"{thresh_vol_pct:.1f}%","at threshold (vol %)"),
        ("Cum. value",     f"{thresh_val_pct:.1f}%","at threshold (val %)"),
        ("Class A spend",  fmt_cur(a_vsum),             f"{a_val_pct:.1f}% of total"),
    ],
):
    col.metric(lbl, val, sub)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Pareto curve
# ══════════════════════════════════════════════════════════════════════════════
ch1_left, ch1_right = st.columns([3, 1])

with ch1_left:
    st.plotly_chart(
        make_pareto_chart(curve_df, thresh_vol_pct, thresh_val_pct),
        use_container_width=True,
    )

with ch1_right:
    st.markdown("**ABC Classification**")
    pill("Class A — above threshold", "#0C447C", "#E6F1FB", "#85B7EB",
         a_n, a_vol_pct, a_val_pct, fmt_cur(a_vsum))
    pill("Class B — mid range",       "#27500A", "#EAF3DE", "#97C459",
         b_n, b_vol_pct, b_val_pct, fmt_cur(b_vsum))
    pill("Class C — tail",            "#444441", "#F1EFE8", "#B4B2A9",
         c_n, c_vol_pct, c_val_pct, fmt_cur(c_vsum))

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Distribution curve
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Distribution — Find the optimal threshold value")
st.markdown(
    "Y axis = **cumulative volume**. &nbsp;&nbsp;"
    "X axis = **actual value ($)**. &nbsp;&nbsp;"
    "Drag the **threshold slider** (sidebar) until the 🟠 orange dot meets the 🟣 purple ◆ "
    "to achieve your target volume %."
)

fig_dist, count_at_thresh, pct_at_thresh, target_value = make_distribution_chart(
    dist_df, value_col, threshold, vol_target_pct, total_n
)

ch2_left, ch2_right = st.columns([3, 1])

with ch2_left:
    st.plotly_chart(fig_dist, use_container_width=True)

with ch2_right:
    st.markdown("**Distribution insights**")

    # status: on-target, above, below
    gap     = count_at_thresh - (total_n * vol_target_pct / 100)
    on_tgt  = abs(gap) <= total_n * 0.03
    above   = gap > total_n * 0.03
    c_bg    = "#EAF3DE" if on_tgt else ("#E6F1FB" if above else "#FAECE7")
    c_col   = "#27500A" if on_tgt else ("#0C447C" if above else "#712B13")
    c_bdr   = "#97C459" if on_tgt else ("#85B7EB" if above else "#D85A30")
    status  = "✓ On target" if on_tgt else ("↑ Above target" if above else "↓ Below target")

    st.markdown(f"""
    <div style="background:{c_bg};border:1px solid {c_bdr};border-radius:8px;
                padding:10px 14px;margin-bottom:8px;">
      <div style="font-weight:600;color:{c_col};font-size:12px;">Current threshold</div>
      <div style="font-size:20px;font-weight:700;color:{c_col};margin:4px 0;">{fmt_cur(threshold)}</div>
      <div style="font-size:12px;color:{c_col};">{count_at_thresh:,} projects · {pct_at_thresh:.1f}% of volume</div>
      <div style="font-size:11px;color:{c_col};margin-top:4px;">{status}</div>
    </div>""", unsafe_allow_html=True)

    optimal_txt = f"≥ {fmt_cur(target_value)}" if target_value else "n/a"
    st.markdown(f"""
    <div style="background:#F3EAF7;border:1px solid #A855B5;border-radius:8px;
                padding:10px 14px;margin-bottom:8px;">
      <div style="font-weight:600;color:#5B1A6E;font-size:12px;">{vol_target_pct}% volume target</div>
      <div style="font-size:20px;font-weight:700;color:#5B1A6E;margin:4px 0;">
          {int(total_n * vol_target_pct / 100):,} projects</div>
      <div style="font-size:12px;color:#5B1A6E;">Optimal threshold: {optimal_txt}</div>
      <div style="font-size:11px;color:#5B1A6E;margin-top:4px;">◆ purple diamond on chart</div>
    </div>""", unsafe_allow_html=True)

    tail_spend     = df[df[value_col] <= threshold][value_col].sum()
    tail_spend_pct = tail_spend / total_val * 100 if total_val else 0
    st.markdown(f"""
    <div style="background:#F5F4F0;border:1px solid #D3D1C7;border-radius:8px;
                padding:10px 14px;margin-bottom:8px;">
      <div style="font-weight:600;color:#444441;font-size:12px;">Tail (shaded region)</div>
      <div style="font-size:12px;color:#444441;margin-top:4px;">
          {count_at_thresh:,} projects · {pct_at_thresh:.1f}% of volume</div>
      <div style="font-size:12px;color:#444441;">
          {fmt_cur(tail_spend)} · {tail_spend_pct:.1f}% of spend</div>
      <div style="font-size:11px;color:#888780;margin-top:4px;">
          Candidates for P-card / catalogue buying</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["Class A projects", "Breakdown by column", "Raw data"])

with tab1:
    if class_a.empty:
        st.info("No Class A projects at this threshold.")
    else:
        dcols = [value_col] + [c for c in df.columns if c != value_col]
        st.dataframe(
            class_a[dcols].sort_values(value_col, ascending=False)
                          .style.format({value_col: "${:,.0f}"}),
            use_container_width=True, height=300,
        )

with tab2:
    cat_candidates = [c for c in df.columns
                      if c != value_col and (df[c].dtype == object or df[c].nunique() <= 30)]
    if cat_candidates:
        bc = st.selectbox("Group by", cat_candidates, key="breakdown")
        grp = (df.groupby(bc)[value_col]
                 .agg(count="count", total_value="sum")
                 .assign(
                     pct_value=lambda x: x["total_value"] / total_val * 100,
                     pct_volume=lambda x: x["count"] / total_n * 100,
                 )
                 .sort_values("total_value", ascending=False)
                 .reset_index())
        grp["fmt_value"] = grp["total_value"].apply(fmt_cur)

        bl, br = st.columns(2)
        with bl:
            fb = go.Figure(go.Bar(
                y=grp[bc], x=grp["total_value"], orientation="h",
                marker_color=BLUE,
                text=grp["pct_value"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                hovertemplate="%{y}: %{x:$,.0f}<extra></extra>",
            ))
            fb.update_layout(
                xaxis_title="Total value", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                height=max(280, len(grp) * 36 + 80),
                margin=dict(l=10, r=60, t=30, b=40),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fb, use_container_width=True)
        with br:
            st.dataframe(
                grp.rename(columns={
                    "count": "# projects", "fmt_value": "Total value",
                    "pct_value": "Value %", "pct_volume": "Volume %",
                })[[bc, "# projects", "Total value", "Value %", "Volume %"]]
                .style.format({"Value %": "{:.1f}%", "Volume %": "{:.1f}%"}),
                use_container_width=True, height=300,
            )
    else:
        st.info("No categorical columns available for breakdown.")

with tab3:
    st.dataframe(
        df.sort_values(value_col, ascending=False)
          .style.format({value_col: "${:,.0f}"}),
        use_container_width=True, height=350,
    )
    st.download_button(
        "Download filtered data as CSV",
        df.to_csv(index=False).encode(),
        "procurement_filtered.csv", "text/csv",
    )

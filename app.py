"""EDF Viewer — Streamlit + MNE + Plotly"""

import os
import glob
import numpy as np
import streamlit as st
import mne
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="EDF Viewer", layout="wide")

# hide default Streamlit header padding / hamburger for a cleaner look
st.markdown(
    """
    <style>
    /* tighten top padding */
    .block-container { padding-top: 1.2rem; }
    /* epoch nav buttons */
    div[data-testid="stColumns"] button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N4",
    "Sleep stage R": "REM",
    "Sleep stage ?": "?",
    "Movement time": "MT",
}

STAGE_COLORS = {
    "W": "#e74c3c",
    "N1": "#f39c12",
    "N2": "#2ecc71",
    "N3": "#3498db",
    "N4": "#2980b9",
    "REM": "#9b59b6",
    "?": "#95a5a6",
    "MT": "#7f8c8d",
}

STAGE_JUMP_TARGETS = ["W", "N1", "N2", "N3", "REM"]

# Hypnogram overview constants
VALID_HYP_STAGES = {"W", "N1", "N2", "N3", "REM"}
HYP_STAGE_Y = {"W": 0, "REM": 1, "N1": 2, "N2": 3, "N3": 4}
HYP_COLORS = {
    "W": "#F5A623",    # 橘黃
    "REM": "#E74C3C",  # 紅
    "N1": "#7EC8E3",   # 淺藍
    "N2": "#2C5F8A",   # 深藍
    "N3": "#8E44AD",   # 紫
}


# ── helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EDF…")
def load_raw(path: str) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    return raw


@st.cache_data(show_spinner=False)
def load_hypnogram(path: str):
    """Return list of (onset, duration, stage_label) sorted by onset."""
    annot = mne.read_annotations(path)
    stages = []
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        label = STAGE_MAP.get(desc)
        if label is not None:
            stages.append((float(onset), float(dur), label))
    stages.sort(key=lambda x: x[0])
    return stages


def find_hypnogram(psg_path: str) -> str | None:
    """Heuristic: look for *-Hypnogram.edf in the same directory."""
    d = os.path.dirname(psg_path)
    base = os.path.basename(psg_path)
    # SC4002E0-PSG.edf → SC4002 prefix
    prefix = base.split("E")[0] if "E" in base else base[:6]
    candidates = glob.glob(os.path.join(d, "*Hypnogram*.[Ee][Dd][Ff]"))
    for c in candidates:
        if prefix in os.path.basename(c):
            return c
    # fallback: any hypnogram in the same dir
    return candidates[0] if candidates else None


def stage_at(stages, t: float) -> str | None:
    for onset, dur, label in stages:
        if onset <= t < onset + dur:
            return label
    return None


def epoch_stages(stages, epoch_len: float, n_epochs: int) -> list[str | None]:
    """Pre-compute stage for each epoch (by epoch midpoint)."""
    result = []
    for i in range(n_epochs):
        mid = i * epoch_len + epoch_len / 2
        result.append(stage_at(stages, mid))
    return result


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def normalize_stages(stages):
    """Merge N4 → N3 and keep only scorable stages (exclude ? / MT)."""
    out = []
    for onset, dur, label in stages:
        if label == "N4":
            label = "N3"
        if label in VALID_HYP_STAGES:
            out.append((onset, dur, label))
    return out


def build_hyp_segments(stages_norm):
    """Group consecutive same-stage epochs into (start, end, stage) segments."""
    if not stages_norm:
        return []
    segs: list[tuple[float, float, str]] = []
    s_stage = stages_norm[0][2]
    s_start = stages_norm[0][0]
    s_end = stages_norm[0][0] + stages_norm[0][1]
    for onset, dur, label in stages_norm[1:]:
        if label == s_stage and abs(onset - s_end) < 1:
            s_end = onset + dur
        else:
            segs.append((s_start, s_end, s_stage))
            s_stage, s_start, s_end = label, onset, onset + dur
    segs.append((s_start, s_end, s_stage))
    return segs


def compute_sleep_metrics(stages_norm, stages_all, trim_wake=True):
    """Return dict of standard sleep statistics from normalised hypnogram.

    *stages_norm*  – N4→N3 merged, ? / MT removed
    *stages_all*   – original list (needed for TIB boundary)
    *trim_wake*    – True: TIB starts 30 min before sleep onset;
                     False: TIB starts at record start (0)
    """
    if not stages_norm:
        return None

    # Sleep onset = first non-W epoch onset
    first_sleep = next((o for o, _, s in stages_norm if s != "W"), None)
    if first_sleep is None:
        return None

    # Final awakening = end of last non-W epoch
    final_awk = max(
        (o + d for o, d, s in stages_norm if s != "W"), default=None
    )
    if final_awk is None:
        return None

    # ── TIB boundaries ──────────────────────────────────────────────
    if trim_wake:
        tib_start = max(0.0, first_sleep - 1800)  # 30 min before onset
    else:
        tib_start = 0.0
    tib_end = final_awk
    tib_s = tib_end - tib_start

    # ── TST: all non-W time ─────────────────────────────────────────
    tst_s = sum(d for _, d, s in stages_norm if s != "W")

    # ── SE ──────────────────────────────────────────────────────────
    se = tst_s / tib_s * 100 if tib_s else 0.0

    # ── SOL: TIB start → sleep onset ───────────────────────────────
    sol_s = first_sleep - tib_start

    # ── REM latency: sleep onset → first REM ───────────────────────
    first_rem = next((o for o, _, s in stages_norm if s == "REM"), None)
    rem_lat_s = (first_rem - first_sleep) if first_rem is not None else None

    # ── WASO & awakening count (sleep onset → final awakening) ─────
    waso_s = sum(
        d for o, d, s in stages_norm
        if s == "W" and o >= first_sleep and o < final_awk
    )
    awakenings = 0
    in_w = False
    for o, d, s in stages_norm:
        if o < first_sleep or o >= final_awk:
            continue
        if s == "W":
            if not in_w:
                awakenings += 1
                in_w = True
        else:
            in_w = False

    def _m(sec):
        return round(sec / 60, 1) if sec is not None else None

    return dict(
        se=round(se, 1),
        tst=_m(tst_s),
        sol=_m(sol_s),
        rem_latency=_m(rem_lat_s),
        waso=_m(waso_s),
        awakenings=awakenings,
        tib=_m(tib_s),
        tib_start_s=tib_start,
        tib_end_s=tib_end,
        trimmed=trim_wake,
    )


# ── sidebar: file selection ─────────────────────────────────────────────────
st.sidebar.subheader("📂 File")

existing = sorted(glob.glob(os.path.join(DATA_DIR, "*.[Ee][Dd][Ff]")))
existing_psg = [f for f in existing if "Hypnogram" not in os.path.basename(f)]

source = st.sidebar.radio("Source", ["Local files", "Upload"], horizontal=True, label_visibility="collapsed")

edf_path = None
if source == "Local files":
    if existing_psg:
        labels = [os.path.basename(f) for f in existing_psg]
        sel = st.sidebar.selectbox("Choose file", labels, label_visibility="collapsed")
        edf_path = existing_psg[labels.index(sel)]
    else:
        st.sidebar.info("No EDF files in `data/`. Upload one instead.")
else:
    uploaded = st.sidebar.file_uploader("Upload .edf", type=["edf"], label_visibility="collapsed")
    if uploaded:
        save_dir = os.path.join(DATA_DIR, "uploads")
        os.makedirs(save_dir, exist_ok=True)
        edf_path = os.path.join(save_dir, uploaded.name)
        with open(edf_path, "wb") as f:
            f.write(uploaded.getbuffer())

if edf_path is None:
    st.info("← Select or upload an EDF file to begin.")
    st.stop()

# ── load data ────────────────────────────────────────────────────────────────
raw = load_raw(edf_path)
sfreq = raw.info["sfreq"]
n_times = raw.n_times
duration = n_times / sfreq
meas_date = raw.info.get("meas_date")
meas_str = meas_date.strftime("%Y-%m-%d %H:%M") if meas_date else "N/A"

# file info caption
st.caption(
    f"**{os.path.basename(edf_path)}** · "
    f"{len(raw.ch_names)} channels · "
    f"{sfreq:.0f} Hz · "
    f"{format_duration(duration)} · "
    f"recorded {meas_str}"
)

# ── sidebar: channel & epoch settings ────────────────────────────────────────
st.sidebar.subheader("📊 Channels")
all_chs = raw.ch_names
selected_chs = st.sidebar.multiselect("Select channels", all_chs, default=all_chs[:3], label_visibility="collapsed")

st.sidebar.subheader("⏱ Epoch")
epoch_len = st.sidebar.number_input("Epoch length (s)", min_value=1, max_value=300, value=30, step=1)
n_epochs = int(np.ceil(duration / epoch_len))

# ── hypnogram ────────────────────────────────────────────────────────────────
hyp_path = find_hypnogram(edf_path)
stages = load_hypnogram(hyp_path) if hyp_path else None
per_epoch_stages = epoch_stages(stages, epoch_len, n_epochs) if stages else None

# ── sidebar: hypnogram overview toggle ──────────────────────────────────────
if stages:
    st.sidebar.subheader("🌙 Hypnogram")
    show_hypnogram = st.sidebar.checkbox("顯示 hypnogram", value=False)
    trim_wake = st.sidebar.checkbox(
        "排除記錄前後的長 W 區段",
        value=True,
        help="勾選時 TIB 從 sleep onset 前 30 分鐘算起；取消則從記錄起點算起",
    )
else:
    show_hypnogram = False
    trim_wake = True

# ── epoch state ──────────────────────────────────────────────────────────────
# The number_input owns st.session_state.epoch via key="epoch".
# Button callbacks use on_click to modify it *before* the widget renders on
# the next rerun, avoiding StreamlitAPIException.
if "epoch" not in st.session_state:
    st.session_state.epoch = 0


def clamp_epoch(e: int) -> int:
    return max(0, min(e, n_epochs - 1))


def _step(delta: int):
    st.session_state.epoch = clamp_epoch(st.session_state.epoch + delta)


def _jump_to_stage(target: str):
    cur = st.session_state.epoch
    # search forward from current epoch
    for j in range(cur + 1, n_epochs):
        if per_epoch_stages[j] == target:
            st.session_state.epoch = j
            return
    # wrap around from the beginning
    for j in range(0, cur):
        if per_epoch_stages[j] == target:
            st.session_state.epoch = j
            return


# ── epoch navigation bar ────────────────────────────────────────────────────
nav_cols = st.columns([1, 1, 2, 1, 1] + ([1] * len(STAGE_JUMP_TARGETS) if stages else []))

with nav_cols[0]:
    st.button("⏪ −100", use_container_width=True, on_click=_step, args=(-100,))
with nav_cols[1]:
    st.button("◀ −1", use_container_width=True, on_click=_step, args=(-1,))
with nav_cols[2]:
    st.number_input(
        "Epoch",
        min_value=0,
        max_value=n_epochs - 1,
        step=1,
        label_visibility="collapsed",
        key="epoch",
    )
with nav_cols[3]:
    st.button("+1 ▶", use_container_width=True, on_click=_step, args=(1,))
with nav_cols[4]:
    st.button("+100 ⏩", use_container_width=True, on_click=_step, args=(100,))

# stage jump buttons
if stages and per_epoch_stages:
    for i, target in enumerate(STAGE_JUMP_TARGETS):
        with nav_cols[5 + i]:
            st.button(f"→{target}", use_container_width=True,
                      on_click=_jump_to_stage, args=(target,))

cur_epoch = clamp_epoch(st.session_state.epoch)

# ── current sleep stage display ──────────────────────────────────────────────
if stages:
    cur_stage = stage_at(stages, cur_epoch * epoch_len + epoch_len / 2)
    if cur_stage:
        color = STAGE_COLORS.get(cur_stage, "#888")
        st.markdown(
            f'<p style="font-size:1.8rem; font-weight:700; color:{color}; margin:0 0 0.3rem 0;">'
            f"{cur_stage}</p>",
            unsafe_allow_html=True,
        )

# ── main waveform plot ───────────────────────────────────────────────────────
if not selected_chs:
    st.warning("Select at least one channel in the sidebar.")
    st.stop()

t_start = cur_epoch * epoch_len
t_end = min(t_start + epoch_len, duration)
i_start = int(t_start * sfreq)
i_end = int(t_end * sfreq)

data, times = raw[raw.ch_names.index(selected_chs[0]):raw.ch_names.index(selected_chs[0]) + 1, i_start:i_end]
times_sec = times  # mne returns times in seconds

# get data for all selected channels at once
picks = [raw.ch_names.index(ch) for ch in selected_chs]
data_all = raw.get_data(picks=picks, start=i_start, stop=i_end)
times_arr = np.arange(i_start, min(i_end, n_times)) / sfreq

n_ch = len(selected_chs)
fig = make_subplots(
    rows=n_ch,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    subplot_titles=selected_chs,
)

plot_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

for idx, ch in enumerate(selected_chs):
    color = plot_colors[idx % len(plot_colors)]
    fig.add_trace(
        go.Scattergl(
            x=times_arr,
            y=data_all[idx] * 1e6,  # to µV for EEG
            mode="lines",
            line=dict(width=0.8, color=color),
            name=ch,
            hovertemplate="%{y:.1f} µV<extra>" + ch + "</extra>",
        ),
        row=idx + 1,
        col=1,
    )
    fig.update_yaxes(title_text="µV", row=idx + 1, col=1)

fig.update_xaxes(title_text="Time (s)", row=n_ch, col=1)
fig.update_layout(
    height=max(250 * n_ch, 400),
    margin=dict(l=60, r=20, t=30, b=40),
    showlegend=False,
    dragmode="zoom",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ── hypnogram overview & sleep metrics ──────────────────────────────────────
if stages and show_hypnogram:
    stages_norm = normalize_stages(stages)
    if stages_norm:
        metrics = compute_sleep_metrics(stages_norm, stages, trim_wake=trim_wake)

        # ── sleep metrics (2 rows × 3 cols) ────────────────────────────
        if metrics:
            row1 = st.columns(3)
            row1[0].metric(
                "SE",
                f"{metrics['se']}%",
                help="睡眠效率 Sleep Efficiency = TST ÷ TIB × 100",
            )
            row1[1].metric(
                "TST",
                f"{metrics['tst']} min",
                help="總睡眠時間 Total Sleep Time：所有非 W 階段的總時長（不含 Movement time / Sleep stage ?）",
            )
            row1[2].metric(
                "SOL",
                f"{metrics['sol']} min" if metrics["sol"] is not None else "—",
                help="入睡潛伏期 Sleep Onset Latency：從 TIB 起點到第一個非 W epoch",
            )

            row2 = st.columns(3)
            row2[0].metric(
                "REM Latency",
                f"{metrics['rem_latency']} min" if metrics["rem_latency"] is not None else "—",
                help="REM 潛伏期：從第一個非 W epoch 到第一個 REM epoch",
            )
            row2[1].metric(
                "WASO",
                f"{metrics['waso']} min" if metrics["waso"] is not None else "—",
                help="入睡後清醒 Wake After Sleep Onset：sleep onset 到 final awakening 之間的 W 總時長",
            )
            row2[2].metric(
                "Awakenings",
                f"{metrics['awakenings']}" if metrics["awakenings"] is not None else "—",
                help="覺醒次數：sleep onset 到 final awakening 之間 W 片段（bout）的個數",
            )

            # caption: show TIB range in clock time
            if meas_date:
                t0 = meas_date + timedelta(seconds=metrics["tib_start_s"])
                t1 = meas_date + timedelta(seconds=metrics["tib_end_s"])
                tib_range = f"{t0.strftime('%H:%M')}–{t1.strftime('%H:%M')}"
            else:
                tib_range = (
                    f"{int(metrics['tib_start_s'] // 60)}′–"
                    f"{int(metrics['tib_end_s'] // 60)}′"
                )
            if metrics["trimmed"]:
                st.caption(
                    f"TIB {tib_range}（{metrics['tib']} min），"
                    "已排除記錄前後長 W 區段（onset 前 30 min 起算）"
                )
            else:
                st.caption(
                    f"TIB {tib_range}（{metrics['tib']} min），"
                    "從記錄起點到最後一個非 W epoch"
                )

        # ── hypnogram chart ─────────────────────────────────────────────
        segments = build_hyp_segments(stages_norm)
        rec_start = meas_date if meas_date else datetime(2000, 1, 1)

        # per-stage traces with None gaps
        stage_x: dict[str, list] = {s: [] for s in HYP_STAGE_Y}
        stage_y_data: dict[str, list] = {s: [] for s in HYP_STAGE_Y}
        for s0, s1, stg in segments:
            t0 = rec_start + timedelta(seconds=s0)
            t1 = rec_start + timedelta(seconds=s1)
            stage_x[stg].extend([t0, t1, None])
            stage_y_data[stg].extend([HYP_STAGE_Y[stg], HYP_STAGE_Y[stg], None])

        fig_hyp = go.Figure()
        for stg in ("W", "REM", "N1", "N2", "N3"):
            if stage_x[stg]:
                fig_hyp.add_trace(
                    go.Scatter(
                        x=stage_x[stg],
                        y=stage_y_data[stg],
                        mode="lines",
                        line=dict(color=HYP_COLORS[stg], width=3),
                        name=stg,
                    )
                )

        # vertical connectors between stages
        conn_x: list = []
        conn_y: list = []
        for i in range(len(segments) - 1):
            _, e1, stg1 = segments[i]
            s2, _, stg2 = segments[i + 1]
            if abs(e1 - s2) < 1:
                t = rec_start + timedelta(seconds=e1)
                conn_x.extend([t, t, None])
                conn_y.extend([HYP_STAGE_Y[stg1], HYP_STAGE_Y[stg2], None])
        if conn_x:
            fig_hyp.add_trace(
                go.Scatter(
                    x=conn_x,
                    y=conn_y,
                    mode="lines",
                    line=dict(color="#888", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # current-epoch vertical marker
        epoch_t = rec_start + timedelta(seconds=cur_epoch * epoch_len + epoch_len / 2)
        fig_hyp.add_vline(
            x=epoch_t,
            line_dash="dash",
            line_color="#FFD700",
            line_width=1.5,
            annotation_text=f"Epoch {cur_epoch}",
            annotation_position="top",
            annotation_font_color="#FFD700",
        )

        fig_hyp.update_layout(
            height=250,
            margin=dict(l=60, r=80, t=10, b=40),
            yaxis=dict(
                tickvals=[0, 1, 2, 3, 4],
                ticktext=["W", "REM", "N1", "N2", "N3"],
                range=[4.5, -0.5],
            ),
            xaxis=dict(title="Time", tickformat="%H:%M"),
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01,
            ),
            hovermode="x",
            dragmode="zoom",
        )

        st.plotly_chart(fig_hyp, use_container_width=True)

# ── channel statistics (collapsed) ──────────────────────────────────────────
with st.expander("📈 Channel statistics", expanded=False):
    stat_cols = st.columns(min(n_ch, 4))
    for idx, ch in enumerate(selected_chs):
        col = stat_cols[idx % len(stat_cols)]
        sig = data_all[idx] * 1e6
        col.markdown(f"**{ch}**")
        col.text(
            f"  Max:  {sig.max():.2f} µV\n"
            f"  Min:  {sig.min():.2f} µV\n"
            f"  Mean: {sig.mean():.2f} µV\n"
            f"  Std:  {sig.std():.2f} µV"
        )

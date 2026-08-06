"""EDF Viewer — Streamlit + MNE + Plotly"""

import os
import glob
import numpy as np
import streamlit as st
import mne
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

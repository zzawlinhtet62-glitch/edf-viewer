"""EDF Viewer — Streamlit + MNE + Plotly"""

import os
import glob
import numpy as np
import streamlit as st
import mne
import yasa
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, cohen_kappa_score

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="EDF Viewer", layout="wide")

# hide default Streamlit header padding / hamburger for a cleaner look
st.markdown(
    """
    <style>
    /* tighten top padding */
    .block-container { padding-top: 1.2rem; }
    /* epoch nav buttons — prevent text wrapping */
    div[data-testid="stColumns"] button {
        width: 100%;
        white-space: nowrap;
        padding-left: 0.3rem;
        padding-right: 0.3rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Canonical aliases → normalised label (W / N1 / N2 / N3 / REM).
# Anything that doesn't map is treated as unscored (None).
_STAGE_ALIAS: dict[str, str] = {
    "W": "W", "WAKE": "W",
    "1": "N1", "N1": "N1", "S1": "N1",
    "2": "N2", "N2": "N2", "S2": "N2",
    "3": "N3", "N3": "N3", "S3": "N3",
    "4": "N3", "N4": "N3", "S4": "N3",
    "R": "REM", "REM": "REM",
}


def normalize_label(raw_label: str | None) -> str | None:
    """Normalise any sleep-stage string to W / N1 / N2 / N3 / REM.

    Handles 'Sleep stage ...' prefixes, YASA labels (WAKE / R), and various
    legacy notations.  Returns ``None`` for unscored / movement / unknown.
    """
    if not raw_label:
        return None
    s = raw_label.strip()
    # strip common prefix from Sleep-EDF hypnograms
    if s.lower().startswith("sleep stage "):
        s = s[len("sleep stage "):]
    s = s.strip().upper()
    return _STAGE_ALIAS.get(s)

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
        label = normalize_label(desc)
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
    """Expand annotation segments into per-epoch labels.

    For each annotation (onset, duration, label), every epoch whose start
    falls inside [onset, onset+duration) is assigned that label.  Epochs
    not covered by any annotation remain ``None``.
    """
    result: list[str | None] = [None] * n_epochs
    for onset, dur, label in stages:
        if label is None:
            continue
        i_start = max(0, int(onset / epoch_len))
        i_end = min(n_epochs, int((onset + dur) / epoch_len))
        for i in range(i_start, i_end):
            result[i] = label
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


# ── YASA helpers ─────────────────────────────────────────────────────────────

def detect_channels(ch_names: list[str]) -> dict:
    """Auto-detect EEG / EOG / EMG channel names for YASA."""
    result = {}
    ch_lower = {ch: ch.lower() for ch in ch_names}

    # EEG: prefer Fpz-Cz
    for ch, low in ch_lower.items():
        if "fpz-cz" in low or "fpz_cz" in low:
            result["eeg_name"] = ch
            break
    if "eeg_name" not in result:
        for ch, low in ch_lower.items():
            if "eeg" in low or "fpz" in low or "cz" in low or "c4" in low or "c3" in low:
                result["eeg_name"] = ch
                break
    if "eeg_name" not in result:
        # fallback: first channel
        result["eeg_name"] = ch_names[0]

    # EOG: prefer 'horizontal'
    for ch, low in ch_lower.items():
        if "horizontal" in low:
            result["eog_name"] = ch
            break
    if "eog_name" not in result:
        for ch, low in ch_lower.items():
            if "eog" in low:
                result["eog_name"] = ch
                break

    # EMG: prefer 'submental'
    for ch, low in ch_lower.items():
        if "submental" in low:
            result["emg_name"] = ch
            break
    if "emg_name" not in result:
        for ch, low in ch_lower.items():
            if "emg" in low:
                result["emg_name"] = ch
                break

    return result


@st.cache_data(show_spinner=False)
def run_yasa_staging(_raw, edf_path: str, eeg_name: str,
                     eog_name: str | None = None,
                     emg_name: str | None = None) -> list[str]:
    """Run YASA SleepStaging; return per-30s-epoch labels (W/N1/N2/N3/REM).

    *_raw* is prefixed with _ so st.cache_data skips hashing it;
    *edf_path* (+ channel names) serves as the cache key.
    """
    kwargs = {"eeg_name": eeg_name}
    if eog_name:
        kwargs["eog_name"] = eog_name
    if emg_name:
        kwargs["emg_name"] = emg_name
    sls = yasa.SleepStaging(_raw, **kwargs)
    pred = sls.predict()
    # yasa ≥0.7 returns a Hypnogram object with .hypno (pd.Series);
    # older versions return a plain list / ndarray of strings.
    if hasattr(pred, "hypno"):
        labels = list(pred.hypno)
    else:
        labels = list(np.asarray(pred).ravel())
    # normalise to W / N1 / N2 / N3 / REM
    return [normalize_label(s) or s for s in labels]


def add_hyp_traces_to_fig(fig, segments, rec_start, row, show_legend=True):
    """Add hypnogram traces (coloured stage lines + connectors) to a subplot row."""
    stage_x: dict[str, list] = {s: [] for s in HYP_STAGE_Y}
    stage_y_data: dict[str, list] = {s: [] for s in HYP_STAGE_Y}
    for s0, s1, stg in segments:
        t0 = rec_start + timedelta(seconds=s0)
        t1 = rec_start + timedelta(seconds=s1)
        stage_x[stg].extend([t0, t1, None])
        stage_y_data[stg].extend([HYP_STAGE_Y[stg], HYP_STAGE_Y[stg], None])

    for stg in ("W", "REM", "N1", "N2", "N3"):
        if stage_x[stg]:
            fig.add_trace(
                go.Scatter(
                    x=stage_x[stg],
                    y=stage_y_data[stg],
                    mode="lines",
                    line=dict(color=HYP_COLORS[stg], width=3),
                    name=stg,
                    showlegend=show_legend,
                    legendgroup=stg,
                ),
                row=row, col=1,
            )

    # vertical connectors between adjacent stages
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
        fig.add_trace(
            go.Scatter(
                x=conn_x, y=conn_y, mode="lines",
                line=dict(color="#888", width=1),
                showlegend=False, hoverinfo="skip",
            ),
            row=row, col=1,
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

# ── hypnogram (technician) ──────────────────────────────────────────────────
hyp_path = find_hypnogram(edf_path)
stages = load_hypnogram(hyp_path) if hyp_path else None
per_epoch_stages = epoch_stages(stages, epoch_len, n_epochs) if stages else None

# ── sidebar: YASA auto-staging ──────────────────────────────────────────────
st.sidebar.subheader("🤖 自動判期")
st.sidebar.caption(f"yasa v{yasa.__version__}")
yasa_labels: list[str] | None = None
yasa_per_epoch: list[str | None] | None = None
yasa_can_run = (epoch_len == 30)

if not yasa_can_run:
    st.sidebar.warning("YASA 固定使用 30 秒 epoch，目前設定不是 30 秒，已停用自動判期。")
else:
    ch_map = detect_channels(raw.ch_names)
    ch_info_parts = [f"EEG: {ch_map['eeg_name']}"]
    if "eog_name" in ch_map:
        ch_info_parts.append(f"EOG: {ch_map['eog_name']}")
    if "emg_name" in ch_map:
        ch_info_parts.append(f"EMG: {ch_map['emg_name']}")
    st.sidebar.caption("偵測頻道：" + "、".join(ch_info_parts))

    if st.sidebar.button("執行自動判期（YASA）"):
        st.session_state.yasa_path = edf_path

    if st.session_state.get("yasa_path") == edf_path:
        with st.spinner("🤖 YASA 自動分期運算中，請稍候…"):
            yasa_labels = run_yasa_staging(
                raw, edf_path,
                eeg_name=ch_map["eeg_name"],
                eog_name=ch_map.get("eog_name"),
                emg_name=ch_map.get("emg_name"),
            )
        # Length-mismatch guard: warn if YASA vs technician differ by >2 epochs
        if per_epoch_stages:
            tech_epoch_count = sum(1 for s in per_epoch_stages if s is not None)
            if abs(len(yasa_labels) - tech_epoch_count) > 2:
                st.warning(
                    f"⚠️ YASA 與技師判讀的 epoch 數不一致"
                    f"（YASA: {len(yasa_labels)}, 技師: {tech_epoch_count}），"
                    "以較短的為準對齊比較。"
                )
        # Build per-epoch list aligned to viewer epochs; pad / trim as needed
        yasa_per_epoch = list(yasa_labels)
        while len(yasa_per_epoch) < n_epochs:
            yasa_per_epoch.append(None)
        yasa_per_epoch = yasa_per_epoch[:n_epochs]
        st.sidebar.success("自動：YASA（已快取）")

# ── sidebar: hypnogram overview toggle ──────────────────────────────────────
has_tech = stages is not None
has_yasa = yasa_per_epoch is not None

if has_tech or has_yasa:
    st.sidebar.subheader("🌙 Hypnogram")
    show_hypnogram = st.sidebar.checkbox("顯示 hypnogram", value=False)
    if has_tech:
        trim_wake = st.sidebar.checkbox(
            "排除記錄前後的長 W 區段",
            value=True,
            help="勾選時 TIB 從 sleep onset 前 30 分鐘算起；取消則從記錄起點算起",
        )
    else:
        trim_wake = True
else:
    show_hypnogram = False
    trim_wake = True

# ── sidebar: jump source radio ──────────────────────────────────────────────
if has_tech and has_yasa:
    jump_source = st.sidebar.radio("分期跳轉依據", ["技師", "自動"], horizontal=True)
elif has_tech:
    jump_source = "技師"
elif has_yasa:
    jump_source = "自動"
else:
    jump_source = None

# Which per-epoch list drives the stage-jump buttons?
active_per_epoch = (
    yasa_per_epoch if jump_source == "自動" else per_epoch_stages
)

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
    src = active_per_epoch
    if src is None:
        return
    # search forward from current epoch
    for j in range(cur + 1, n_epochs):
        if src[j] == target:
            st.session_state.epoch = j
            return
    # wrap around from the beginning
    for j in range(0, cur):
        if src[j] == target:
            st.session_state.epoch = j
            return


def _jump_to_disagree():
    """Jump to the next epoch where technician and YASA labels disagree."""
    cur = st.session_state.epoch
    if not per_epoch_stages or not yasa_per_epoch:
        return
    for j in range(cur + 1, n_epochs):
        t = per_epoch_stages[j] if j < len(per_epoch_stages) else None
        a = yasa_per_epoch[j] if j < len(yasa_per_epoch) else None
        if t is not None and a is not None and t != a:
            st.session_state.epoch = j
            return
    # wrap around
    for j in range(0, cur):
        t = per_epoch_stages[j] if j < len(per_epoch_stages) else None
        a = yasa_per_epoch[j] if j < len(yasa_per_epoch) else None
        if t is not None and a is not None and t != a:
            st.session_state.epoch = j
            return


# ── epoch navigation bar ────────────────────────────────────────────────────
show_stage_btns = active_per_epoch is not None
has_both = has_tech and has_yasa
# Build column weights: widen the epoch input, keep buttons compact
_nav_weights = [1, 0.7, 2.5, 0.7, 1]
if show_stage_btns:
    _nav_weights += [0.8] * len(STAGE_JUMP_TARGETS)
if has_both:
    _nav_weights += [0.7]
nav_cols = st.columns(_nav_weights)

with nav_cols[0]:
    st.button("⏪−100", use_container_width=True, on_click=_step, args=(-100,))
with nav_cols[1]:
    st.button("◀−1", use_container_width=True, on_click=_step, args=(-1,))
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
    st.button("+1▶", use_container_width=True, on_click=_step, args=(1,))
with nav_cols[4]:
    st.button("+100⏩", use_container_width=True, on_click=_step, args=(100,))

# stage jump buttons (use short labels: →R instead of →REM)
_STAGE_SHORT = {"W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "R"}
if show_stage_btns:
    for i, target in enumerate(STAGE_JUMP_TARGETS):
        with nav_cols[5 + i]:
            st.button(f"→{_STAGE_SHORT[target]}", use_container_width=True,
                      on_click=_jump_to_stage, args=(target,))

# ≠ button — jump to next disagreement (only when both tech & YASA exist)
if has_both:
    _neq_idx = 5 + (len(STAGE_JUMP_TARGETS) if show_stage_btns else 0)
    with nav_cols[_neq_idx]:
        st.button("≠", use_container_width=True, on_click=_jump_to_disagree,
                  help="跳到下一個技師與自動判期不一致的 epoch")

cur_epoch = clamp_epoch(st.session_state.epoch)

# ── current sleep stage display ──────────────────────────────────────────────
tech_stage = (
    per_epoch_stages[cur_epoch]
    if per_epoch_stages and cur_epoch < len(per_epoch_stages)
    else None
)
auto_stage = (
    yasa_per_epoch[cur_epoch]
    if yasa_per_epoch and cur_epoch < len(yasa_per_epoch)
    else None
)

if tech_stage or auto_stage:
    parts: list[str] = []
    if tech_stage:
        tc = STAGE_COLORS.get(tech_stage, "#888")
        parts.append(f'<span style="color:{tc};">技師：{tech_stage}</span>')
    if auto_stage:
        ac = STAGE_COLORS.get(auto_stage, "#888")
        parts.append(f'<span style="color:{ac};">自動：{auto_stage}</span>')
    if tech_stage and auto_stage and tech_stage == auto_stage:
        parts.append('<span style="color:#2ecc71; font-size:1.6rem;">✓</span>')
    st.markdown(
        '<p style="font-size:1.8rem; font-weight:700; margin:0 0 0.3rem 0;">'
        + " &nbsp;│&nbsp; ".join(parts)
        + "</p>",
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

# ── pre-compute stage data (used by metrics, hypnogram, and comparison) ────
stages_norm = normalize_stages(stages) if stages else []
yasa_stages_norm: list[tuple[float, float, str]] = []
if yasa_labels:
    yasa_stages_tuples = [(i * 30.0, 30.0, lbl) for i, lbl in enumerate(yasa_labels)]
    yasa_stages_norm = normalize_stages(yasa_stages_tuples)

# ── sleep metrics (always visible when technician data exists) ─────────────
if stages_norm:
    metrics = compute_sleep_metrics(stages_norm, stages, trim_wake=trim_wake)

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

# ── hypnogram chart (shown when checkbox is on) ───────────────────────────
if (has_tech or has_yasa) and show_hypnogram:
    rec_start = meas_date if meas_date else datetime(2000, 1, 1)
    epoch_marker_t = rec_start + timedelta(seconds=cur_epoch * epoch_len + epoch_len / 2)

    tech_segments = build_hyp_segments(stages_norm) if stages_norm else []
    yasa_segments = build_hyp_segments(yasa_stages_norm) if yasa_stages_norm else []

    y_axis_cfg = dict(
        tickvals=[0, 1, 2, 3, 4],
        ticktext=["W", "REM", "N1", "N2", "N3"],
        range=[4.5, -0.5],
    )

    if tech_segments and yasa_segments:
        fig_hyp = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["技師判讀", "自動判期（YASA）"],
            vertical_spacing=0.12,
        )
        add_hyp_traces_to_fig(fig_hyp, tech_segments, rec_start, row=1, show_legend=True)
        add_hyp_traces_to_fig(fig_hyp, yasa_segments, rec_start, row=2, show_legend=False)
        for r in (1, 2):
            fig_hyp.add_vline(
                x=epoch_marker_t, line_dash="dash",
                line_color="#FFD700", line_width=1.5,
                row=r, col=1,
            )
        fig_hyp.add_annotation(
            x=epoch_marker_t, y=-0.3, yref="y domain",
            text=f"Epoch {cur_epoch}", font_color="#FFD700",
            showarrow=False, row=1, col=1,
        )
        fig_hyp.update_yaxes(**y_axis_cfg, row=1, col=1)
        fig_hyp.update_yaxes(**y_axis_cfg, row=2, col=1)
        fig_hyp.update_xaxes(title="Time", tickformat="%H:%M", row=2, col=1)
        fig_hyp.update_layout(
            height=400,
            margin=dict(l=60, r=80, t=30, b=40),
            legend=dict(
                orientation="v", yanchor="middle", y=0.5,
                xanchor="left", x=1.01,
            ),
            hovermode="x", dragmode="zoom",
        )
        st.plotly_chart(fig_hyp, use_container_width=True)

    elif tech_segments or yasa_segments:
        segments = tech_segments or yasa_segments
        fig_hyp = go.Figure()
        stage_x: dict[str, list] = {s: [] for s in HYP_STAGE_Y}
        stage_y_data: dict[str, list] = {s: [] for s in HYP_STAGE_Y}
        for s0, s1, stg in segments:
            t0 = rec_start + timedelta(seconds=s0)
            t1 = rec_start + timedelta(seconds=s1)
            stage_x[stg].extend([t0, t1, None])
            stage_y_data[stg].extend([HYP_STAGE_Y[stg], HYP_STAGE_Y[stg], None])
        for stg in ("W", "REM", "N1", "N2", "N3"):
            if stage_x[stg]:
                fig_hyp.add_trace(
                    go.Scatter(
                        x=stage_x[stg], y=stage_y_data[stg], mode="lines",
                        line=dict(color=HYP_COLORS[stg], width=3), name=stg,
                    )
                )
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
                    x=conn_x, y=conn_y, mode="lines",
                    line=dict(color="#888", width=1),
                    showlegend=False, hoverinfo="skip",
                )
            )
        fig_hyp.add_vline(
            x=epoch_marker_t, line_dash="dash", line_color="#FFD700",
            line_width=1.5, annotation_text=f"Epoch {cur_epoch}",
            annotation_position="top", annotation_font_color="#FFD700",
        )
        fig_hyp.update_layout(
            height=250,
            margin=dict(l=60, r=80, t=10, b=40),
            yaxis=y_axis_cfg,
            xaxis=dict(title="Time", tickformat="%H:%M"),
            legend=dict(orientation="v", yanchor="middle", y=0.5,
                        xanchor="left", x=1.01),
            hovermode="x", dragmode="zoom",
        )
        st.plotly_chart(fig_hyp, use_container_width=True)

# ── A. 判期比對 (Staging Agreement) ────────────────────────────────────────
if has_both:
    # Pair epochs where both have a label
    _COMPARE_STAGES = ["W", "N1", "N2", "N3", "REM"]
    tech_arr = per_epoch_stages or []
    auto_arr = yasa_per_epoch or []
    paired = [
        (tech_arr[i], auto_arr[i])
        for i in range(min(len(tech_arr), len(auto_arr)))
        if tech_arr[i] is not None and auto_arr[i] is not None
    ]

    if paired:
        y_true = [p[0] for p in paired]
        y_pred = [p[1] for p in paired]
        n_total = len(paired)
        n_agree = sum(1 for t, a in paired if t == a)
        n_disagree = n_total - n_agree
        agree_pct = n_agree / n_total * 100
        kappa = cohen_kappa_score(y_true, y_pred)

        with st.expander(
            f"📊 判期比對　一致率 {agree_pct:.1f}%　"
            f"Cohen κ {kappa:.3f}　"
            f"（{n_total} 個 epoch，{n_disagree} 個不一致）",
            expanded=False,
        ):
            # Confusion matrix
            labels_present = [s for s in _COMPARE_STAGES
                              if s in set(y_true) or s in set(y_pred)]
            cm = sk_confusion_matrix(y_true, y_pred, labels=labels_present)

            # Build DataFrame: rows=tech, cols=YASA, + subtotal + agreement rate
            df_cm = pd.DataFrame(cm, index=labels_present, columns=labels_present)
            df_cm.insert(0, "技師小計", df_cm.sum(axis=1))
            per_stage_agree = []
            for i, stg in enumerate(labels_present):
                row_sum = cm[i].sum()
                diag = cm[i][i]
                per_stage_agree.append(
                    f"{diag / row_sum * 100:.1f}%" if row_sum > 0 else "—"
                )
            df_cm["逐期一致率"] = per_stage_agree
            df_cm.index.name = "技師＼YASA"

            st.dataframe(df_cm, use_container_width=True)

            st.caption(
                "列＝技師判讀，欄＝YASA 自動判期。"
                "N1 兩邊最難對上是常態——人類判讀者之間在 N1 的一致度本來就最低。"
                "自動判期是輔助定位用的，不能取代技師判讀。"
            )

    # ── B. 各階段時間與佔比 ────────────────────────────────────────────────
    if paired:
        with st.expander("📊 各階段時間與佔比", expanded=False):
            # Count epochs per stage
            tech_counts = {s: 0 for s in _COMPARE_STAGES}
            auto_counts = {s: 0 for s in _COMPARE_STAGES}
            for t_lbl, a_lbl in paired:
                if t_lbl in tech_counts:
                    tech_counts[t_lbl] += 1
                if a_lbl in auto_counts:
                    auto_counts[a_lbl] += 1

            epoch_min = epoch_len / 60.0  # seconds per epoch → minutes

            tech_tst_epochs = sum(v for k, v in tech_counts.items() if k != "W")
            auto_tst_epochs = sum(v for k, v in auto_counts.items() if k != "W")
            tech_tib_epochs = sum(tech_counts.values())
            auto_tib_epochs = sum(auto_counts.values())

            # Bar chart
            fig_bar = go.Figure()
            tech_mins = [tech_counts[s] * epoch_min for s in _COMPARE_STAGES]
            auto_mins = [auto_counts[s] * epoch_min for s in _COMPARE_STAGES]

            fig_bar.add_trace(go.Bar(
                name="技師判讀",
                x=_COMPARE_STAGES, y=tech_mins,
                text=[f"{v:.1f}" for v in tech_mins],
                textposition="outside",
                marker_color="#2C5F8A",
            ))
            fig_bar.add_trace(go.Bar(
                name="自動判期",
                x=_COMPARE_STAGES, y=auto_mins,
                text=[f"{v:.1f}" for v in auto_mins],
                textposition="outside",
                marker_color="#7EC8E3",
            ))
            fig_bar.update_layout(
                barmode="group",
                yaxis_title="分鐘",
                height=350,
                margin=dict(l=50, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Table
            rows = []
            for s in _COMPARE_STAGES:
                t_min = tech_counts[s] * epoch_min
                a_min = auto_counts[s] * epoch_min
                if s == "W":
                    t_pct = (tech_counts[s] / tech_tib_epochs * 100
                             if tech_tib_epochs else 0)
                    a_pct = (auto_counts[s] / auto_tib_epochs * 100
                             if auto_tib_epochs else 0)
                else:
                    t_pct = (tech_counts[s] / tech_tst_epochs * 100
                             if tech_tst_epochs else 0)
                    a_pct = (auto_counts[s] / auto_tst_epochs * 100
                             if auto_tst_epochs else 0)
                rows.append({
                    "階段": s,
                    "技師(分)": round(t_min, 1),
                    "技師%": f"{t_pct:.1f}%",
                    "自動(分)": round(a_min, 1),
                    "自動%": f"{a_pct:.1f}%",
                    "差異(分)": round(t_min - a_min, 1),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            st.caption("W 的百分比以 TIB 為分母，其餘各期以 TST 為分母")

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

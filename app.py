# Black-Scholes Option Strategy Dashboard
# Interactive dashboard for pricing option strategies using Black-Scholes model (with dividend yield).
# Computes prices, Greeks (Delta, Gamma, Theta, Vega, Rho), payoffs, time value, and premiums.
# Plots combined or separate metrics vs. underlying price (S).

import streamlit as st
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Dark theme for matplotlib ───
plt.style.use('dark_background')
PLOT_BG   = '#0f1117'
GRID_COL  = '#2a2d3e'
AXIS_COL  = '#8b8fa8'

# ─── Page config ───
st.set_page_config(page_title="Option Greeks Simulator", layout="wide", initial_sidebar_state="expanded")

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e2e4ef;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #16192a;
        border-right: 1px solid #2a2d3e;
        width: 300px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #16192a;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 14px 18px;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: #4f6ef7;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #8b8fa8 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        color: #e2e4ef !important;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        background: #1e2235;
        border: 1px solid #2a2d3e;
        color: #c9cde0;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #4f6ef7;
        border-color: #4f6ef7;
        color: #fff;
    }

    /* Headers */
    h1 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #e2e4ef !important;
        letter-spacing: -0.02em;
    }
    h2 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #8b8fa8 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    h3 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.9rem !important;
        color: #c9cde0 !important;
    }

    /* Leg pills */
    .leg-pill {
        background: #1e2235;
        border: 1px solid #2a2d3e;
        border-radius: 20px;
        padding: 4px 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #c9cde0;
        margin-bottom: 4px;
        display: inline-block;
    }

    /* Divider */
    hr {
        border-color: #2a2d3e !important;
        margin: 12px 0 !important;
    }

    /* Select boxes and inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stMultiSelect > div > div {
        background-color: #1e2235 !important;
        border-color: #2a2d3e !important;
        color: #e2e4ef !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
    }

    /* Section label */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4f6ef7;
        margin-bottom: 8px;
        margin-top: 4px;
    }

    /* Strategy badge */
    .strategy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e2235, #252a40);
        border: 1px solid #3a4060;
        border-radius: 6px;
        padding: 3px 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #7fa8ff;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Vectorized Black-Scholes ───
def bs_price_and_greeks_vec(S_arr, K, T, r, q, sigma, option_type='call'):
    """Vectorized BS: accepts numpy array for S, returns dict of arrays."""
    if T <= 0 or sigma <= 0:
        raise ValueError("Time to maturity and volatility must be positive.")
    S_arr = np.asarray(S_arr, dtype=float)
    d1 = (np.log(S_arr / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S_arr * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -(S_arr * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S_arr * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S_arr * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -(S_arr * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S_arr * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S_arr * sigma * np.sqrt(T))
    vega  = S_arr * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    # ── Market conventions ──
    theta_daily = theta / 365      # per calendar day
    vega_1pct   = vega  / 100     # per 1% move in vol

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta_daily,
        'vega':  vega_1pct,
        'rho':   rho,
    }

def bs_scalar(S, K, T, r, q, sigma, option_type='call'):
    """Scalar wrapper for display metrics."""
    res = bs_price_and_greeks_vec(np.array([S]), K, T, r, q, sigma, option_type)
    return {k: float(v[0]) for k, v in res.items()}

# ─── Predefined strategies ───
STRATEGIES = {
    "Long Call":      [{'type': 'call', 'strike': 100.0, 'position':  1}],
    "Long Put":       [{'type': 'put',  'strike': 100.0, 'position':  1}],
    "Short Call":     [{'type': 'call', 'strike': 100.0, 'position': -1}],
    "Short Put":      [{'type': 'put',  'strike': 100.0, 'position': -1}],
    "Straddle":       [{'type': 'call', 'strike': 100.0, 'position':  1},
                       {'type': 'put',  'strike': 100.0, 'position':  1}],
    "Strangle":       [{'type': 'call', 'strike': 105.0, 'position':  1},
                       {'type': 'put',  'strike':  95.0, 'position':  1}],
    "Bull Call Spread":[{'type': 'call', 'strike':  95.0, 'position':  1},
                        {'type': 'call', 'strike': 105.0, 'position': -1}],
    "Bear Put Spread": [{'type': 'put',  'strike': 105.0, 'position':  1},
                        {'type': 'put',  'strike':  95.0, 'position': -1}],
    "Iron Condor":    [{'type': 'put',  'strike':  90.0, 'position':  1},
                       {'type': 'put',  'strike':  95.0, 'position': -1},
                       {'type': 'call', 'strike': 105.0, 'position': -1},
                       {'type': 'call', 'strike': 110.0, 'position':  1}],
    "Butterfly":      [{'type': 'call', 'strike':  95.0, 'position':  1},
                       {'type': 'call', 'strike': 100.0, 'position': -2},
                       {'type': 'call', 'strike': 105.0, 'position':  1}],
}

# ─── Session state defaults ───
defaults = {'S': 100.0, 'T': 1.0, 'r': 0.05, 'q': 0.0, 'sigma': 0.20}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if 'legs' not in st.session_state:
    st.session_state.legs = [{'type': 'call', 'strike': 100.0, 'position': 1}]
if 'single_plots' not in st.session_state:
    st.session_state.single_plots = []

# ─── Slider/number sync callbacks ───
for param in ['S', 'T', 'r', 'q', 'sigma']:
    exec(f"""
def update_{param}():
    st.session_state['{param}'] = st.session_state['num_{param}']
    st.session_state['slider_{param}'] = st.session_state['num_{param}']
def update_slider_{param}():
    st.session_state['{param}'] = st.session_state['slider_{param}']
    st.session_state['num_{param}'] = st.session_state['slider_{param}']
""")

# ─── Sidebar ───
with st.sidebar:
    st.markdown('<p class="section-label">Market Parameters</p>', unsafe_allow_html=True)

    params_cfg = [
        ('S',     'Underlying Price (S)',   50.0,  200.0, 1.0),
        ('T',     'Time to Maturity (yrs)', 0.01,  5.0,   0.01),
        ('r',     'Risk-Free Rate',         0.0,   0.20,  0.005),
        ('q',     'Dividend Yield',         0.0,   0.20,  0.005),
        ('sigma', 'Volatility (σ)',         0.01,  1.0,   0.01),
    ]
    for param, label, lo, hi, step in params_cfg:
        st.slider(label, lo, hi, st.session_state[param], step,
                  key=f'slider_{param}', on_change=eval(f'update_slider_{param}'))
        st.number_input(label, lo, hi, st.session_state[param], step,
                        key=f'num_{param}', on_change=eval(f'update_{param}'),
                        label_visibility="collapsed")

    S     = st.session_state['S']
    T     = st.session_state['T']
    r     = st.session_state['r']
    q     = st.session_state['q']
    sigma = st.session_state['sigma']

    st.divider()

    # ── Predefined strategies ──
    st.markdown('<p class="section-label">Predefined Strategies</p>', unsafe_allow_html=True)
    strategy_cols = st.columns(2)
    for i, name in enumerate(STRATEGIES):
        with strategy_cols[i % 2]:
            if st.button(name, use_container_width=True, key=f"strat_{name}"):
                st.session_state.legs = [leg.copy() for leg in STRATEGIES[name]]
                st.rerun()

    st.divider()

    # ── Add custom leg ──
    st.markdown('<p class="section-label">Custom Leg</p>', unsafe_allow_html=True)
    col_type, col_pos = st.columns(2)
    with col_type:
        new_type = st.selectbox("Type", ["call", "put"], key="new_type", label_visibility="collapsed")
    with col_pos:
        new_pos_label = st.selectbox("Pos", ["Long", "Short"], key="new_position", label_visibility="collapsed")
    new_strike = st.number_input("Strike (K)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
    if st.button("＋ Add Leg", use_container_width=True):
        sign = 1 if new_pos_label == "Long" else -1
        st.session_state.legs.append({'type': new_type, 'strike': new_strike, 'position': sign})
        st.rerun()

    # ── Active legs ──
    if st.session_state.legs:
        st.divider()
        st.markdown('<p class="section-label">Active Legs</p>', unsafe_allow_html=True)
        for i, leg in enumerate(st.session_state.legs):
            pos_label = "▲ Long" if leg['position'] > 0 else "▼ Short"
            qty = abs(leg['position'])
            qty_str = f"×{qty}" if qty != 1 else ""
            col_leg, col_rm = st.columns([4, 1])
            with col_leg:
                st.markdown(
                    f'<span class="leg-pill">{pos_label} {leg["type"].upper()} '
                    f'K={leg["strike"]:.0f} {qty_str}</span>',
                    unsafe_allow_html=True
                )
            with col_rm:
                if st.button("✕", key=f"rm_{i}"):
                    del st.session_state.legs[i]
                    st.rerun()

    st.divider()

    # ── Plot settings ──
    st.markdown('<p class="section-label">Plot Settings</p>', unsafe_allow_html=True)
    plot_options = ["Payoff", "Delta", "Gamma", "Theta", "Vega", "Rho", "Time Value", "Premium"]
    selected_plots = st.multiselect("Overlay on Combined Graph", plot_options, default=["Payoff"])
    show_separate = st.checkbox("Separate graph per metric (+ Payoff)")

    st.divider()
    st.markdown('<p class="section-label">Individual Graphs</p>', unsafe_allow_html=True)
    single_metric = st.selectbox("Metric", ["Delta", "Gamma", "Theta", "Vega", "Rho", "Time Value", "Premium"])
    if st.button("＋ Add Graph", use_container_width=True):
        if single_metric not in st.session_state.single_plots:
            st.session_state.single_plots.append(single_metric)
        st.rerun()

    if st.session_state.single_plots:
        for i, m in enumerate(st.session_state.single_plots):
            c1, c2 = st.columns([4, 1])
            c1.caption(m)
            with c2:
                if st.button("✕", key=f"rm_single_{i}"):
                    del st.session_state.single_plots[i]
                    st.rerun()

# ─── Color palette ───
colors = {
    'Payoff':     '#f5f5f5',
    'Time Value': '#00c9b1',
    'Premium':    '#a3e635',
    'Delta':      '#4f6ef7',
    'Gamma':      '#22d3ee',
    'Theta':      '#f87171',
    'Vega':       '#c084fc',
    'Rho':        '#fb923c',
}

# ─── Helper: styled figure ───
def make_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.grid(True, color=GRID_COL, linewidth=0.6, linestyle='--')
    ax.tick_params(colors=AXIS_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    return fig, ax

def style_twin(ax2):
    ax2.set_facecolor(PLOT_BG)
    ax2.tick_params(colors=AXIS_COL, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COL)

# ─── Main ───
st.title("Option Greeks Simulator")
st.caption("Black-Scholes · Multi-leg · Theta/day · Vega/1%vol")

try:
    if not st.session_state.legs:
        st.warning("Add at least one option leg in the sidebar.")
    else:
        # ── Current scalar values ──
        combined = {'price': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        combined_payoff = 0
        combined_time_value = 0

        for leg in st.session_state.legs:
            res = bs_scalar(S, leg['strike'], T, r, q, sigma, leg['type'])
            sign = leg['position']
            for key in combined:
                combined[key] += sign * res[key]
            intrinsic = max(S - leg['strike'], 0) if leg['type'] == 'call' else max(leg['strike'] - S, 0)
            combined_payoff      += sign * intrinsic
            combined_time_value  += sign * (res['price'] - intrinsic)

        # ── Metrics grid ──
        st.header("Strategy Snapshot")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Premium",      f"{combined['price']:.4f}")
        m2.metric("Payoff",       f"{combined_payoff:.4f}")
        m3.metric("Time Value",   f"{combined_time_value:.4f}")
        m4.metric("Delta (Δ)",    f"{combined['delta']:.4f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Gamma (Γ)",    f"{combined['gamma']:.4f}")
        m6.metric("Theta / day",  f"{combined['theta']:.4f}", help="Theta normalized: value change per calendar day")
        m7.metric("Vega / 1% σ",  f"{combined['vega']:.4f}", help="Vega normalized: value change per 1% move in implied vol")
        m8.metric("Rho (ρ)",      f"{combined['rho']:.4f}")

        st.divider()

        # ── Vectorized plot data ──
        S_range = np.linspace(max(10, S - 60), S + 60, 300)

        def compute_metric(metric, S_arr):
            out = np.zeros(len(S_arr))
            for leg in st.session_state.legs:
                sign = leg['position']
                if metric == 'Payoff':
                    if leg['type'] == 'call':
                        out += sign * np.maximum(S_arr - leg['strike'], 0)
                    else:
                        out += sign * np.maximum(leg['strike'] - S_arr, 0)
                else:
                    res = bs_price_and_greeks_vec(S_arr, leg['strike'], T, r, q, sigma, leg['type'])
                    if metric == 'Time Value':
                        intrinsic = (np.maximum(S_arr - leg['strike'], 0) if leg['type'] == 'call'
                                     else np.maximum(leg['strike'] - S_arr, 0))
                        out += sign * (res['price'] - intrinsic)
                    elif metric == 'Premium':
                        out += sign * res['price']
                    else:
                        out += sign * res[metric.lower()]
            return out

        plot_data = {m: compute_metric(m, S_range) for m in plot_options}

        # ── Break-even detection ──
        def find_breakevens(payoff_arr, S_arr):
            bes = []
            for i in range(len(payoff_arr) - 1):
                if payoff_arr[i] * payoff_arr[i + 1] < 0:
                    be = S_arr[i] - payoff_arr[i] * (S_arr[i+1] - S_arr[i]) / (payoff_arr[i+1] - payoff_arr[i])
                    bes.append(be)
            return bes

        breakevens = find_breakevens(plot_data['Payoff'], S_range)

        def add_breakevens(ax, color='#fbbf24', ymin=0, ymax=1):
            for be in breakevens:
                ax.axvline(be, color=color, linestyle=':', linewidth=1.2, alpha=0.7)
                ax.text(be, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
                        f' BE\n {be:.1f}', color=color, fontsize=7,
                        fontfamily='monospace', va='bottom')

        def add_spot(ax):
            ax.axvline(S, color='#8b8fa8', linestyle='--', linewidth=1, alpha=0.5)

        # ── Combined plot ──
        if selected_plots:
            st.header("Combined Strategy Plot")
            fig, ax = make_fig(10, 5)
            axes_list = [ax]
            lines = []

            for i, metric in enumerate(selected_plots):
                col = colors[metric]
                if i == 0:
                    line, = ax.plot(S_range, plot_data[metric], color=col, linewidth=1.8, label=metric)
                    ax.set_ylabel(metric, color=col, fontsize=9, fontfamily='monospace')
                    ax.tick_params(axis='y', colors=col)
                    ax.yaxis.label.set_color(col)
                else:
                    new_ax = ax.twinx()
                    new_ax.spines['right'].set_position(('axes', 1.0 + 0.12 * (i - 1)))
                    style_twin(new_ax)
                    line, = new_ax.plot(S_range, plot_data[metric], color=col, linewidth=1.8, label=metric)
                    new_ax.set_ylabel(metric, color=col, fontsize=9, fontfamily='monospace')
                    new_ax.tick_params(axis='y', colors=col)
                    axes_list.append(new_ax)
                lines.append(line)

            ax.set_xlabel('Underlying Price (S)', color=AXIS_COL, fontsize=9)
            ax.axhline(0, color=GRID_COL, linewidth=0.8)
            add_spot(ax)
            add_breakevens(ax)
            ax.legend(lines, [l.get_label() for l in lines],
                      loc='upper left', fontsize=8, framealpha=0.2,
                      facecolor='#1e2235', edgecolor=GRID_COL,
                      labelcolor='white', prop={'family': 'monospace'})
            fig.tight_layout()
            st.pyplot(fig)

        # ── Separate graphs ──
        if show_separate:
            metrics_sel = [p for p in selected_plots if p != "Payoff"]
            if not metrics_sel:
                st.info("Select additional metrics (beyond Payoff) to enable separate plots.")
            else:
                st.header("Separate Metric Graphs")
                for idx in range(0, len(metrics_sel), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if idx + j < len(metrics_sel):
                            metric = metrics_sel[idx + j]
                            with col:
                                fig, ax = make_fig(6, 4)
                                ax.plot(S_range, plot_data["Payoff"], color=colors['Payoff'],
                                        linewidth=1.5, label='Payoff')
                                ax.set_ylabel('Payoff', color=colors['Payoff'], fontsize=8, fontfamily='monospace')
                                ax.tick_params(axis='y', colors=colors['Payoff'])
                                ax.axhline(0, color=GRID_COL, linewidth=0.7)
                                add_spot(ax)
                                add_breakevens(ax)

                                ax2 = ax.twinx()
                                style_twin(ax2)
                                ax2.plot(S_range, plot_data[metric], color=colors[metric],
                                         linewidth=1.5, label=metric)
                                ax2.set_ylabel(metric, color=colors[metric], fontsize=8, fontfamily='monospace')
                                ax2.tick_params(axis='y', colors=colors[metric])

                                ax.set_xlabel('S', color=AXIS_COL, fontsize=8)
                                ax.set_title(f'{metric}  ·  Payoff', color='#c9cde0', fontsize=9,
                                             fontfamily='monospace', pad=8)

                                all_lines = ax.get_lines() + ax2.get_lines()
                                ax.legend(all_lines, [l.get_label() for l in all_lines],
                                          loc='upper left', fontsize=7, framealpha=0.2,
                                          facecolor='#1e2235', edgecolor=GRID_COL,
                                          labelcolor='white', prop={'family': 'monospace'})
                                fig.tight_layout()
                                st.pyplot(fig)

        # ── Individual graphs ──
        if st.session_state.single_plots:
            st.header("Individual Metric Graphs")
            for idx in range(0, len(st.session_state.single_plots), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if idx + j < len(st.session_state.single_plots):
                        metric = st.session_state.single_plots[idx + j]
                        with col:
                            fig, ax = make_fig(6, 4)
                            ax.plot(S_range, plot_data["Payoff"], color=colors['Payoff'],
                                    linewidth=1.5, label='Payoff')
                            ax.set_ylabel('Payoff', color=colors['Payoff'], fontsize=8, fontfamily='monospace')
                            ax.tick_params(axis='y', colors=colors['Payoff'])
                            ax.axhline(0, color=GRID_COL, linewidth=0.7)
                            add_spot(ax)
                            add_breakevens(ax)

                            ax2 = ax.twinx()
                            style_twin(ax2)
                            ax2.plot(S_range, plot_data[metric], color=colors[metric],
                                     linewidth=1.5, label=metric)
                            ax2.set_ylabel(metric, color=colors[metric], fontsize=8, fontfamily='monospace')
                            ax2.tick_params(axis='y', colors=colors[metric])

                            ax.set_xlabel('S', color=AXIS_COL, fontsize=8)
                            ax.set_title(f'{metric}  ·  Payoff', color='#c9cde0', fontsize=9,
                                         fontfamily='monospace', pad=8)

                            all_lines = ax.get_lines() + ax2.get_lines()
                            ax.legend(all_lines, [l.get_label() for l in all_lines],
                                      loc='upper left', fontsize=7, framealpha=0.2,
                                      facecolor='#1e2235', edgecolor=GRID_COL,
                                      labelcolor='white', prop={'family': 'monospace'})
                            fig.tight_layout()
                            st.pyplot(fig)

except ValueError as e:
    st.error(f"Error: {e}")

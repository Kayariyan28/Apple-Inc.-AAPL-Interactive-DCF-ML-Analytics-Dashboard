"""
Apple Inc. (AAPL) — Interactive DCF Valuation & ML Analytics Dashboard
Enhanced Streamlit Web App — Converted from Excel DCF Model
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Apple (AAPL) DCF & ML Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 12px; }
    .metric-card { background: linear-gradient(135deg, #1F4E79, #2E75B6);
                   border-radius: 10px; padding: 16px; color: white; text-align: center; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1F4E79;
                      border-bottom: 2px solid #1F4E79; padding-bottom: 4px; margin-bottom: 12px; }
    div[data-testid="stSidebarContent"] { background: #f0f4f8; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=40)
st.sidebar.title("Model Controls")

scenario = st.sidebar.radio("Active Scenario", ["Management Case", "Street Consensus"])

st.sidebar.subheader("Revenue Growth Assumptions")
defaults_mgmt   = [0.08, 0.09, 0.08, 0.07, 0.065]
defaults_street = [0.06, 0.065, 0.06, 0.055, 0.05]
defaults = defaults_mgmt if scenario == "Management Case" else defaults_street
growth_rates = [
    st.sidebar.slider(f"FY{2025+i}E Growth", 0.0, 0.15, defaults[i], 0.005, format="%.1f%%")
    for i in range(5)
]

st.sidebar.subheader("Margin & Valuation")
gross_margin = st.sidebar.slider(
    "Steady-State Gross Margin", 0.40, 0.55,
    0.475 if scenario == "Management Case" else 0.465, 0.005, format="%.1f%%"
)
wacc = st.sidebar.slider("WACC", 0.06, 0.14,
    0.09 if scenario == "Management Case" else 0.095, 0.005, format="%.1f%%")
tgr  = st.sidebar.slider("Terminal Growth Rate", 0.01, 0.06,
    0.035 if scenario == "Management Case" else 0.03, 0.005, format="%.1f%%")
tax_rate = st.sidebar.slider("Effective Tax Rate", 0.10, 0.25, 0.162, 0.005, format="%.1f%%")

st.sidebar.subheader("Simulation Settings")
n_sims = st.sidebar.selectbox("Monte Carlo Simulations", [5_000, 10_000, 25_000, 50_000], index=1)

# ─── BASE DATA (FY2024 10-K + FY2025 Actuals) ────────────────────────────────
BASE_REV = 394_328   # $mm
CASH     = 156_652
DEBT     =  96_842
SHARES   =  15_115   # mm diluted shares
RD_PCT, SGA_PCT, DA_PCT, CAPEX_PCT, NWC_PCT = 0.077, 0.063, 0.028, 0.025, 0.010

CURRENT_PRICE = 266.18

hist_years      = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
hist_revenue    = [265.6, 260.17, 274.52, 365.82, 394.33, 383.29, 391.04, 416.16]
hist_net_income = [59.53, 55.26, 57.41, 94.68, 99.80, 97.00, 93.74, 112.01]
hist_gm         = [38.3, 37.8, 38.2, 41.8, 43.3, 44.1, 46.2, 46.9]
hist_services   = [37.19, 46.29, 53.77, 68.43, 78.13, 85.20, 96.17, 109.16]
hist_products   = [r - s for r, s in zip(hist_revenue, hist_services)]
hist_eps        = [2.98, 3.28, 3.28, 5.61, 6.11, 6.13, 6.08, 7.40]
hist_price      = [39.5, 73.4, 132.7, 177.6, 129.9, 192.5, 254.5, 285.9]
hist_op_income  = [70.9, 63.93, 66.29, 108.95, 119.44, 114.3, 123.22, 133.05]

# Street scenario reference data
street_rev = [417987.68, 445156.88, 471866.29, 497818.94, 522709.88]

# ─── DCF ENGINE ──────────────────────────────────────────────────────────────
def run_dcf(growth_rates, gross_margin, wacc, tgr, tax_rate):
    revs = [BASE_REV]
    for g in growth_rates:
        revs.append(revs[-1] * (1 + g))
    pr = revs[1:]
    ebit   = [r * (gross_margin - RD_PCT - SGA_PCT) for r in pr]
    nopat  = [e * (1 - tax_rate) for e in ebit]
    fcf    = [n + r*DA_PCT - r*CAPEX_PCT - (pr[i] - revs[i])*NWC_PCT
              for i, (n, r) in enumerate(zip(nopat, pr))]
    pv_fcf = [f / (1 + wacc)**(i + 1) for i, f in enumerate(fcf)]
    tv     = fcf[-1] * (1 + tgr) / (wacc - tgr)
    pv_tv  = tv / (1 + wacc)**5
    ev     = sum(pv_fcf) + pv_tv
    equity = ev + CASH - DEBT
    return {
        "years":       [f"FY{2025+i}E" for i in range(5)],
        "revenue":     pr,
        "ebit":        ebit,
        "ebit_margin": [e / r for e, r in zip(ebit, pr)],
        "nopat":       nopat,
        "fcf":         fcf,
        "pv_fcf":      pv_fcf,
        "tv":          tv,
        "pv_tv":       pv_tv,
        "ev":          ev,
        "equity":      equity,
        "price":       equity / SHARES,
        "tv_pct":      pv_tv / (sum(pv_fcf) + pv_tv),
        "sum_pv_fcf":  sum(pv_fcf),
    }

dcf = run_dcf(growth_rates, gross_margin, wacc, tgr, tax_rate)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("🍎 Apple Inc. (AAPL) — Interactive DCF & ML Analytics Dashboard")
st.caption(
    f"5-Year DCF Valuation · Monte Carlo · GBM Stochastic · ML Predictions · Risk Analytics  |  "
    f"Active Scenario: **{scenario}**  |  Current Market Price: **${CURRENT_PRICE:,.2f}**"
)

# ─── KPI BANNER ──────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
implied = dcf["price"]
premium = (CURRENT_PRICE / implied - 1) * 100
k1.metric("Implied Share Price", f"${implied:,.2f}",
          f"{'+' if implied > CURRENT_PRICE else ''}{(implied/CURRENT_PRICE-1)*100:.1f}% vs mkt")
k2.metric("Enterprise Value",    f"${dcf['ev']/1e6:.2f}T")
k3.metric("FY2029E Revenue",     f"${dcf['revenue'][-1]/1000:.0f}B")
k4.metric("FY2029E EBIT Margin", f"{dcf['ebit_margin'][-1]:.1%}")
k5.metric("TV % of EV",          f"{dcf['tv_pct']:.1%}")
k6.metric("Mkt Premium to DCF",  f"{premium:+.1f}%",
          "Overvalued" if premium > 10 else ("Undervalued" if premium < -10 else "Fair"))

st.divider()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 DCF Model",
    "🎲 Monte Carlo",
    "📉 Stochastic & GBM",
    "🔥 Sensitivity Heatmaps",
    "🤖 ML Predictions",
    "📊 Historical Performance",
    "⚠️ Risk Analytics",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DCF MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">5-Year DCF Projection — Income Statement & FCF Build</div>',
                unsafe_allow_html=True)

    # Projection table
    df_dcf = pd.DataFrame({
        "Year":           dcf["years"],
        "Revenue ($mm)":  [f"${r:,.0f}" for r in dcf["revenue"]],
        "Revenue Growth": [f"{g:.1%}"   for g in growth_rates],
        "EBIT ($mm)":     [f"${e:,.0f}" for e in dcf["ebit"]],
        "EBIT Margin":    [f"{m:.1%}"   for m in dcf["ebit_margin"]],
        "NOPAT ($mm)":    [f"${n:,.0f}" for n in dcf["nopat"]],
        "UFCF ($mm)":     [f"${f:,.0f}" for f in dcf["fcf"]],
        "PV of FCF ($mm)":[f"${p:,.0f}" for p in dcf["pv_fcf"]],
    })
    st.dataframe(df_dcf, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_bar(x=dcf["years"], y=[r/1000 for r in dcf["revenue"]],
                    name="Revenue ($B)", marker_color="#1F4E79")
        fig.add_scatter(x=dcf["years"], y=[e/1000 for e in dcf["ebit"]],
                        name="EBIT ($B)", mode="lines+markers",
                        line=dict(color="#FF6B35", width=3), marker=dict(size=8))
        fig.update_layout(title="Revenue & EBIT Projection ($B)", height=380,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure()
        fig2.add_bar(x=dcf["years"], y=[f/1000 for f in dcf["fcf"]],
                     name="UFCF ($B)", marker_color="#00B050")
        fig2.add_scatter(x=dcf["years"], y=[p/1000 for p in dcf["pv_fcf"]],
                         name="PV of FCF ($B)", mode="lines+markers",
                         line=dict(color="#1F4E79", width=3, dash="dash"), marker=dict(size=8))
        fig2.update_layout(title="Free Cash Flow Build ($B)", height=380,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    # EBIT Margin trend
    fig_margin = go.Figure()
    all_years_m  = [2024] + [2025+i for i in range(5)]
    all_margins  = [0.3192] + dcf["ebit_margin"]
    fig_margin.add_scatter(x=[str(y) for y in all_years_m], y=[m*100 for m in all_margins],
                           mode="lines+markers+text",
                           text=[f"{m:.1%}" for m in all_margins],
                           textposition="top center",
                           line=dict(color="#1F4E79", width=3), marker=dict(size=10))
    fig_margin.update_layout(title="EBIT Margin Trend (%)", height=300,
                             yaxis_title="EBIT Margin (%)", xaxis_title="")
    st.plotly_chart(fig_margin, use_container_width=True)

    # Equity bridge waterfall
    st.markdown('<div class="section-header">Equity Value Bridge</div>', unsafe_allow_html=True)
    fig_bridge = go.Figure(go.Waterfall(
        x=["Sum PV FCFs", "PV Terminal Value", "Enterprise Value",
           "+ Cash & Inv.", "− Total Debt", "Equity Value"],
        y=[dcf["sum_pv_fcf"]/1e6, dcf["pv_tv"]/1e6, 0,
           CASH/1e6, -DEBT/1e6, 0],
        measure=["relative","relative","total","relative","relative","total"],
        text=[f"${dcf['sum_pv_fcf']/1e6:.2f}T",
              f"${dcf['pv_tv']/1e6:.2f}T", "",
              f"${CASH/1e6:.2f}T", f"-${DEBT/1e6:.2f}T",
              f"${dcf['equity']/1e6:.2f}T"],
        textposition="outside",
        connector={"line": {"color": "#888"}},
        decreasing={"marker": {"color": "#C00000"}},
        increasing={"marker": {"color": "#00B050"}},
        totals={"marker": {"color": "#1F4E79"}},
    ))
    fig_bridge.update_layout(
        title=f"DCF Equity Bridge → Implied Price: ${dcf['price']:.2f} per share",
        height=420, yaxis_title="$T (Trillions)"
    )
    st.plotly_chart(fig_bridge, use_container_width=True)

    # Scenario comparison table
    st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)
    dcf_street = run_dcf(
        [0.06, 0.065, 0.06, 0.055, 0.05], 0.465, 0.095, 0.03, 0.165
    )
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Management Case", f"${dcf['price']:.2f}",
                  f"{(dcf['price']/CURRENT_PRICE-1)*100:+.1f}% vs market")
    col_s2.metric("Street Consensus", f"${dcf_street['price']:.2f}",
                  f"{(dcf_street['price']/CURRENT_PRICE-1)*100:+.1f}% vs market")
    midpoint = (dcf['price'] + dcf_street['price']) / 2
    col_s3.metric("Midpoint", f"${midpoint:.2f}",
                  f"{(midpoint/CURRENT_PRICE-1)*100:+.1f}% vs market")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-header">Monte Carlo Simulation — {n_sims:,} Scenarios</div>',
                unsafe_allow_html=True)
    st.caption("Randomises: WACC (±1%), Terminal Growth Rate (±0.5%), Revenue Growth (±2.5%), Margins")

    np.random.seed(42)
    sim_prices = []
    for _ in range(n_sims):
        w = np.clip(np.random.normal(wacc, 0.010), 0.06, 0.14)
        t = np.clip(np.random.normal(tgr,  0.005), 0.01, 0.05)
        if w <= t:
            t = w - 0.01
        rev = BASE_REV
        pv  = 0.0
        lf  = 0.0
        for yr in range(1, 6):
            g   = np.clip(np.random.normal(np.mean(growth_rates), 0.025), -0.02, 0.15)
            m   = np.clip(np.random.normal(gross_margin - SGA_PCT - RD_PCT, 0.015), 0.25, 0.42)
            rev *= (1 + g)
            fcf_s = rev * m * (1 - tax_rate) * 1.002
            pv   += fcf_s / (1 + w)**yr
            lf    = fcf_s
        price_s = (pv + lf*(1+t)/(w-t)/(1+w)**5 + CASH - DEBT) / SHARES
        sim_prices.append(price_s)

    sp = np.array(sim_prices)
    mean_p   = sp.mean()
    median_p = np.median(sp)
    p5, p25, p75, p95 = (np.percentile(sp, q) for q in [5, 25, 75, 95])

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Mean",           f"${mean_p:.0f}")
    mc2.metric("Median",         f"${median_p:.0f}")
    mc3.metric("5th Percentile", f"${p5:.0f}")
    mc4.metric("25th Percentile",f"${p25:.0f}")
    mc5.metric("75th Percentile",f"${p75:.0f}")
    mc6.metric("95th Percentile",f"${p95:.0f}")

    # Histogram
    fig_mc = go.Figure()
    fig_mc.add_histogram(x=sp, nbinsx=100, marker_color="#1F4E79",
                         opacity=0.8, name="Simulated Prices")
    fig_mc.add_vline(x=CURRENT_PRICE,  line_dash="dash", line_color="red",   line_width=2,
                     annotation_text=f"Market ${CURRENT_PRICE}", annotation_position="top right")
    fig_mc.add_vline(x=median_p, line_dash="dash", line_color="#00B050", line_width=2,
                     annotation_text=f"Median ${median_p:.0f}", annotation_position="top left")
    fig_mc.add_vline(x=p5,  line_dash="dot", line_color="orange", line_width=1.5,
                     annotation_text=f"P5 ${p5:.0f}")
    fig_mc.add_vline(x=p95, line_dash="dot", line_color="green",  line_width=1.5,
                     annotation_text=f"P95 ${p95:.0f}")
    fig_mc.update_layout(
        title=f"Monte Carlo Fair Value Distribution ({n_sims:,} simulations)",
        xaxis_title="Implied Share Price ($)", yaxis_title="Frequency", height=480,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # CDF
    sorted_sp   = np.sort(sp)
    cdf_vals    = np.arange(1, len(sorted_sp)+1) / len(sorted_sp)
    fig_cdf = go.Figure()
    fig_cdf.add_scatter(x=sorted_sp, y=cdf_vals*100, mode="lines",
                        line=dict(color="#1F4E79", width=2.5), name="CDF")
    fig_cdf.add_vline(x=CURRENT_PRICE, line_dash="dash", line_color="red",
                      annotation_text=f"Market ${CURRENT_PRICE}")
    prob_below_mkt = (sp < CURRENT_PRICE).mean() * 100
    fig_cdf.update_layout(
        title=f"Cumulative Distribution — {prob_below_mkt:.1f}% of scenarios price below market",
        xaxis_title="Implied Share Price ($)", yaxis_title="Percentile (%)", height=350,
    )
    st.plotly_chart(fig_cdf, use_container_width=True)

    # Excel model reference stats
    st.markdown('<div class="section-header">Reference: Excel Model Monte Carlo Results</div>',
                unsafe_allow_html=True)
    mc_ref = pd.DataFrame({
        "Statistic":    ["Mean", "Median", "Standard Deviation",
                         "5th Pctile (Very Bear)", "25th Pctile (Bear)",
                         "50th Pctile (Base)", "75th Pctile (Bull)", "95th Pctile (Very Bull)"],
        "Excel Model":  ["$172", "$164.6", "$41.1", "$122.5", "$144.0",
                         "$164.6", "$190.2", "$246.1"],
        "Interpretation": [
            "Central tendency", "Median fair value",
            "Spread of outcomes",
            "95% chance price is above this", "75% chance price is above this",
            "Equal probability up/down", "25% chance price is above this",
            "5% chance price is above this",
        ]
    })
    st.dataframe(mc_ref, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STOCHASTIC & GBM
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Stochastic Price Models</div>',
                unsafe_allow_html=True)

    # Calibrate from historical
    log_rets = np.diff(np.log(hist_price))
    mu_ann   = np.mean(log_rets)
    sig_ann  = np.std(log_rets, ddof=1)
    S0       = CURRENT_PRICE

    g3a, g3b = st.columns(2)
    g3a.metric("Calibrated Drift (μ)", f"{mu_ann:.1%}")
    g3b.metric("Calibrated Volatility (σ)", f"{sig_ann:.1%}")

    st.subheader("Geometric Brownian Motion (GBM) — dS = μS dt + σS dW")
    st.caption(f"50,000 simulated paths · μ={mu_ann:.1%} · σ={sig_ann:.1%} · S₀=${S0}")

    np.random.seed(42)
    n_paths  = min(n_sims, 10_000)
    n_steps  = 5
    gbm      = np.zeros((n_paths, n_steps + 1))
    gbm[:,0] = S0
    for i in range(n_steps):
        z = np.random.standard_normal(n_paths)
        gbm[:,i+1] = gbm[:,i] * np.exp((mu_ann - 0.5*sig_ann**2) + sig_ann*z)

    fig_gbm = go.Figure()
    # Sample paths
    for i in range(min(300, n_paths)):
        fig_gbm.add_scatter(
            x=list(range(6)), y=gbm[i],
            mode="lines", line=dict(width=0.4, color="rgba(31,78,121,0.08)"),
            showlegend=False
        )
    # Percentile bands
    for pct, col, nm in [(5, "#C00000", "5th Pctile"), (25, "orange", "25th Pctile"),
                         (50, "blue", "Median"), (75, "lightgreen", "75th Pctile"),
                         (95, "#00B050", "95th Pctile")]:
        fig_gbm.add_scatter(
            x=list(range(6)), y=np.percentile(gbm, pct, axis=0),
            mode="lines+markers", line=dict(width=2.5, color=col), name=nm,
            marker=dict(size=6)
        )
    labels = ["Current"] + [f"Yr {i}" for i in range(1, 6)]
    fig_gbm.update_layout(
        title=f"GBM Simulated Price Paths (μ={mu_ann:.1%}, σ={sig_ann:.1%})",
        xaxis=dict(ticktext=labels, tickvals=list(range(6))),
        yaxis_title="Stock Price ($)", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_gbm, use_container_width=True)

    # GBM statistics table
    gbm_stats = pd.DataFrame({
        "Metric":          ["Mean Price", "Median Price", "Std Deviation",
                            "5th Pctile (Bear)", "95th Pctile (Bull)",
                            "P(Price > $300)", "P(Price > $400)", "P(Price < $150)"],
        "Year 1":  [f"${np.mean(gbm[:,1]):.0f}",   f"${np.median(gbm[:,1]):.0f}",
                    f"${np.std(gbm[:,1]):.0f}",     f"${np.percentile(gbm[:,1],5):.0f}",
                    f"${np.percentile(gbm[:,1],95):.0f}",
                    f"{(gbm[:,1]>300).mean():.1%}", f"{(gbm[:,1]>400).mean():.1%}",
                    f"{(gbm[:,1]<150).mean():.1%}"],
        "Year 3":  [f"${np.mean(gbm[:,3]):.0f}",   f"${np.median(gbm[:,3]):.0f}",
                    f"${np.std(gbm[:,3]):.0f}",     f"${np.percentile(gbm[:,3],5):.0f}",
                    f"${np.percentile(gbm[:,3],95):.0f}",
                    f"{(gbm[:,3]>300).mean():.1%}", f"{(gbm[:,3]>400).mean():.1%}",
                    f"{(gbm[:,3]<150).mean():.1%}"],
        "Year 5":  [f"${np.mean(gbm[:,5]):.0f}",   f"${np.median(gbm[:,5]):.0f}",
                    f"${np.std(gbm[:,5]):.0f}",     f"${np.percentile(gbm[:,5],5):.0f}",
                    f"${np.percentile(gbm[:,5],95):.0f}",
                    f"{(gbm[:,5]>300).mean():.1%}", f"{(gbm[:,5]>400).mean():.1%}",
                    f"{(gbm[:,5]<150).mean():.1%}"],
        "Excel Ref (Yr1)": ["$353.1","$336.0","$114.7","$199.8","$566.1","63.9%","29.0%","0.6%"],
    })
    st.dataframe(gbm_stats, use_container_width=True, hide_index=True)

    # Jump-Diffusion model
    st.subheader("Merton Jump-Diffusion Model")
    st.caption("Adds Poisson jump shocks: λ=0.3/yr · mean jump = -5% · jump vol = 15%")
    lambda_j, mu_j, sig_j = 0.3, -0.05, 0.15
    np.random.seed(123)
    jd_paths      = np.zeros((n_paths, n_steps + 1))
    jd_paths[:,0] = S0
    for i in range(n_steps):
        z       = np.random.standard_normal(n_paths)
        n_jumps = np.random.poisson(lambda_j, n_paths)
        J       = np.array([np.sum(np.random.normal(mu_j, sig_j, max(n, 1)))
                            if n > 0 else 0 for n in n_jumps])
        jd_paths[:,i+1] = jd_paths[:,i] * np.exp(
            (mu_ann - 0.5*sig_ann**2 - lambda_j*(np.exp(mu_j+0.5*sig_j**2)-1)) + sig_ann*z
        ) * np.exp(J)

    fig_jd = go.Figure()
    for pct, col, nm in [(5, "#C00000","5th (Bear)"), (50,"blue","Median"), (95,"#00B050","95th (Bull)")]:
        fig_jd.add_scatter(
            x=list(range(6)), y=np.percentile(jd_paths, pct, axis=0),
            mode="lines+markers", line=dict(width=2.5, color=col), name=f"JD {nm}",
            marker=dict(size=6)
        )
        fig_jd.add_scatter(
            x=list(range(6)), y=np.percentile(gbm, pct, axis=0),
            mode="lines+markers", line=dict(width=2, color=col, dash="dot"),
            name=f"GBM {nm}", marker=dict(size=5), opacity=0.6
        )
    fig_jd.update_layout(
        title="Jump-Diffusion vs GBM (solid=JD, dashed=GBM)",
        xaxis=dict(ticktext=labels, tickvals=list(range(6))),
        yaxis_title="Stock Price ($)", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
    )
    st.plotly_chart(fig_jd, use_container_width=True)

    st.subheader("GBM 5-Year Price Distribution")
    gbm_bins  = [75, 125, 175, 225, 275, 375, 475, 575, 675, 825, 975]
    gbm_freq  = [64, 338, 672, 1012, 1407, 1928, 2130, 2060, 1988, 1697, 1400]
    fig_gbm_d = go.Figure()
    fig_gbm_d.add_bar(x=[str(b) for b in gbm_bins], y=gbm_freq,
                      marker_color="#1F4E79", name="GBM Distribution")
    fig_gbm_d.update_layout(
        title="GBM Year-5 Distribution (50,000 paths · Excel Model Reference)",
        xaxis_title="Price Band ($)", yaxis_title="Frequency", height=360,
    )
    st.plotly_chart(fig_gbm_d, use_container_width=True)

    # Methodology notes
    with st.expander("📚 Stochastic Model Methodology"):
        st.markdown("""
**Geometric Brownian Motion (GBM)**
- SDE: `dS = μS dt + σS dW`, calibrated from 8-year AAPL annual log-returns
- μ = 28.3% (drift), σ = 31.7% (volatility) — 50,000 paths over 5 years
- Key finding: 93% probability stock exceeds $300 within 5 years

**Ornstein-Uhlenbeck (OU) — Revenue Growth**
- SDE: `dX = θ(μ – X) dt + σ dW`, θ=0.5, μ=7.2%, σ=12.2%
- Mean-reverting: prevents runaway exponential assumptions
- Median FY2029E Revenue: ~$570B

**Merton Jump-Diffusion**
- Adds Poisson jumps (λ=0.3/yr) modelling sudden macro shocks
- Mean jump = -5%, jump vol = 15% — captures tail risk beyond GBM
- Median Year-5 price ($839) similar to GBM ($850) but fatter left tail

**Bayesian Regression Revenue**
- Normal-Inverse-Gamma prior · Revenue slope = $24.6B/year
- 95% credible interval FY2029: $421B – $635B
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENSITIVITY HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Valuation Sensitivity — WACC vs Terminal Growth Rate</div>',
                unsafe_allow_html=True)

    col4a, col4b = st.columns([1, 1])

    with col4a:
        # Live heatmap from current DCF
        wacc_range = np.arange(0.07, 0.125, 0.005)
        tgr_range  = np.arange(0.015, 0.055, 0.005)
        hm = np.zeros((len(wacc_range), len(tgr_range)))
        for i, w in enumerate(wacc_range):
            for j, t in enumerate(tgr_range):
                if w <= t:
                    hm[i, j] = np.nan
                    continue
                pv = sum(f / (1+w)**(yr+1) for yr, f in enumerate(dcf["fcf"]))
                hm[i, j] = (pv + dcf["fcf"][-1]*(1+t)/(w-t)/(1+w)**5 + CASH - DEBT) / SHARES

        fig_h = px.imshow(
            hm,
            x=[f"{t:.1%}" for t in tgr_range],
            y=[f"{w:.1%}" for w in wacc_range],
            color_continuous_scale=["#C00000","#FF6B6B","#FFD700","#90EE90","#006100"],
            color_continuous_midpoint=CURRENT_PRICE,
            text_auto=".0f",
            title="Live DCF — Implied Price (current FCFs)",
        )
        fig_h.update_layout(
            xaxis_title="Terminal Growth Rate", yaxis_title="WACC", height=500,
        )
        st.plotly_chart(fig_h, use_container_width=True)
        st.info("🟢 Green = above market $266  |  🔴 Red = below market  |  🟡 Yellow = near market")

    with col4b:
        # Excel model reference heatmap (11×8 grid)
        wacc_vals = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
        tgr_vals  = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        hm_excel  = np.array([
            [182.6,197.4,215.6,238.2,267.4,306.3,360.7,442.3],
            [168.8,181.7,197.5,217.0,241.8,274.5,319.9,387.3],
            [156.8,168.2,181.9,199.0,220.2,247.8,285.7,342.0],
            [146.2,156.4,168.5,183.3,201.4,224.9,256.5,302.2],
            [136.8,146.0,156.8,169.8,185.6,205.6,232.1,270.0],
            [128.4,136.7,146.3,157.8,171.6,189.0,211.4,242.6],
            [120.8,128.3,137.0,147.1,159.2,174.0,193.1,218.7],
            [114.0,120.8,128.7,137.8,148.7,161.8,178.1,199.5],
            [107.8,114.0,121.1,129.4,139.2,150.8,165.3,183.6],
            [102.2,107.9,114.3,121.8,130.7,141.2,154.1,170.1],
            [97.0, 102.3,108.1,114.9,123.0,132.5,144.2,158.5],
        ])
        fig_he = px.imshow(
            hm_excel,
            x=[f"{t}%" for t in tgr_vals],
            y=[f"{w}%" for w in wacc_vals],
            color_continuous_scale=["#C00000","#FF6B6B","#FFD700","#90EE90","#006100"],
            color_continuous_midpoint=CURRENT_PRICE,
            text_auto=".0f",
            title="Excel Model Reference — Full 11×8 WACC/TGR Grid",
        )
        fig_he.update_layout(
            xaxis_title="Terminal Growth Rate", yaxis_title="WACC (%)", height=500,
        )
        st.plotly_chart(fig_he, use_container_width=True)

    # Scenario comparison bar chart
    st.markdown('<div class="section-header">Scenario Comparison — Management vs Street</div>',
                unsafe_allow_html=True)
    proj_years = [f"FY{2025+i}E" for i in range(5)]
    fig_sc = go.Figure()
    fig_sc.add_bar(x=proj_years, y=[r/1000 for r in dcf["revenue"]],
                   name="Management Case", marker_color="#1F4E79")
    fig_sc.add_bar(x=proj_years, y=[r/1000 for r in street_rev],
                   name="Street Consensus", marker_color="#7F7F7F")
    fig_sc.update_layout(
        title="Revenue Projections: Management vs Street ($B)",
        barmode="group", height=380,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Summary table
    sum_tbl = pd.DataFrame({
        "Metric":           ["Implied Share Price","5-Yr Revenue CAGR","Terminal UFCF Margin","Valuation Range"],
        "Management Case":  [f"${dcf['price']:.2f}", "7.7%", "28.3%", "$128–$171"],
        "Street Consensus": ["$128.08",              "5.8%", "27.0%", "$128–$171"],
        "Midpoint":         ["$149.34",              "—",    "—",    "—"],
    })
    st.dataframe(sum_tbl, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Machine Learning Predictions — Multi-Model Ensemble</div>',
                unsafe_allow_html=True)
    st.caption("Models: Linear Regression · Polynomial · Exponential Growth · Log-Linear · Monte Carlo")

    proj_yrs = [2025, 2026, 2027, 2028, 2029]
    proj_labels = ["FY2025E","FY2026E","FY2027E","FY2028E","FY2029E"]

    # ── Revenue ML models ──
    st.subheader("Model 1: Revenue Prediction Ensemble  (R² ≈ 0.84–0.86)")
    rev_linear   = [429.8, 454.4, 479.0, 503.5, 528.1]
    rev_poly     = [415.2, 423.1, 426.8, 426.3, 421.6]
    rev_exp      = [438.8, 472.7, 509.3, 548.7, 591.1]
    rev_ensemble = [429.0, 452.3, 475.4, 498.4, 521.3]
    rev_dcf_mgmt = [r/1000 for r in dcf["revenue"]]
    rev_dcf_st   = [r/1000 for r in street_rev]

    fig_rev = go.Figure()
    # Historical
    fig_rev.add_scatter(x=hist_years[-4:], y=hist_revenue[-4:],
                        mode="lines+markers", name="Historical",
                        line=dict(color="#888", width=2, dash="dash"))
    fig_rev.add_scatter(x=proj_yrs, y=rev_linear,   mode="lines+markers", name="Linear Reg",
                        line=dict(color="#5DA5DA", width=2))
    fig_rev.add_scatter(x=proj_yrs, y=rev_poly,     mode="lines+markers", name="Polynomial",
                        line=dict(color="#FAA43A", width=2))
    fig_rev.add_scatter(x=proj_yrs, y=rev_exp,      mode="lines+markers", name="Exponential",
                        line=dict(color="#60BD68", width=2))
    fig_rev.add_scatter(x=proj_yrs, y=rev_ensemble, mode="lines+markers", name="★ Ensemble",
                        line=dict(color="#1F4E79", width=4), marker=dict(size=10))
    fig_rev.add_scatter(x=proj_yrs, y=rev_dcf_mgmt, mode="lines+markers", name="DCF Mgmt",
                        line=dict(color="#B94A8C", width=2, dash="dot"))
    fig_rev.add_scatter(x=proj_yrs, y=rev_dcf_st,   mode="lines+markers", name="DCF Street",
                        line=dict(color="#B94A8C", width=2, dash="longdash"))
    fig_rev.update_layout(title="Revenue Forecast Comparison ($B)", height=420,
                          yaxis_title="Revenue ($B)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_rev, use_container_width=True)

    # ── Gross Margin prediction ──
    col5a, col5b = st.columns(2)
    with col5a:
        st.subheader("Model 2: Gross Margin (R² ≈ 0.77)")
        gm_pred    = [46.5, 47.7, 49.0, 50.2, 51.5]
        svc_pct    = [25.7, 27.3, 28.8, 30.3, 31.9]
        fig_gm = make_subplots(specs=[[{"secondary_y": True}]])
        fig_gm.add_bar(x=proj_labels, y=gm_pred, name="Predicted GM %",
                       marker_color="#1F4E79", secondary_y=False)
        fig_gm.add_scatter(x=proj_labels, y=svc_pct, mode="lines+markers",
                           name="Services % of Rev", line=dict(color="#00B050", width=2.5),
                           secondary_y=True)
        fig_gm.update_yaxes(title_text="Gross Margin (%)", secondary_y=False)
        fig_gm.update_yaxes(title_text="Services Mix (%)", secondary_y=True)
        fig_gm.update_layout(title="Gross Margin vs Services Mix", height=360)
        st.plotly_chart(fig_gm, use_container_width=True)

    with col5b:
        st.subheader("Model 3: EPS Prediction (R² ≈ 0.86)")
        eps_pred   = [7.79, 8.93, 10.23, 11.72, 13.42]
        pe_implied = [34.1, 29.8, 26.0,  22.7,  19.8]
        fig_eps = make_subplots(specs=[[{"secondary_y": True}]])
        fig_eps.add_bar(x=proj_labels, y=eps_pred, name="Predicted EPS",
                        marker_color="#1F4E79", secondary_y=False)
        fig_eps.add_scatter(x=proj_labels, y=pe_implied, mode="lines+markers",
                            name="Implied P/E (at $266)", line=dict(color="#FF6B35", width=2.5),
                            secondary_y=True)
        fig_eps.update_yaxes(title_text="EPS ($)", secondary_y=False)
        fig_eps.update_yaxes(title_text="Implied P/E Multiple", secondary_y=True)
        fig_eps.update_layout(title="EPS & Implied P/E", height=360)
        st.plotly_chart(fig_eps, use_container_width=True)

    # ── Stock Price Prediction ──
    st.subheader("Model 4: Stock Price Ensemble  (R² ≈ 0.85)")
    st.caption("Factors: Time Trend 30%, Revenue Growth 35%, EPS Growth 35%")
    price_upper    = [405.4, 482.2, 573.6, 683.4, 816.2]
    price_ensemble = [283.6, 337.3, 401.3, 478.1, 571.0]
    price_lower    = [198.4, 236.0, 280.7, 334.5, 399.5]
    cy_labels      = ["CY2025E","CY2026E","CY2027E","CY2028E","CY2029E"]

    fig_sp = go.Figure()
    fig_sp.add_scatter(x=cy_labels, y=price_upper, mode="lines+markers",
                       name="Bull Band", line=dict(color="#00B050", width=2, dash="dash"),
                       marker=dict(size=7))
    fig_sp.add_scatter(x=cy_labels, y=price_ensemble, mode="lines+markers+text",
                       name="★ Ensemble Target", line=dict(color="#1F4E79", width=4),
                       marker=dict(size=12), text=[f"${p:.0f}" for p in price_ensemble],
                       textposition="top center")
    fig_sp.add_scatter(x=cy_labels, y=price_lower, mode="lines+markers",
                       name="Bear Band", line=dict(color="#C00000", width=2, dash="dash"),
                       marker=dict(size=7))
    # Fill between bands
    fig_sp.add_traces([
        go.Scatter(x=cy_labels+cy_labels[::-1],
                   y=price_upper+price_lower[::-1],
                   fill="toself", fillcolor="rgba(0,176,80,0.10)",
                   line=dict(width=0), showlegend=False, name="Uncertainty Band")
    ])
    fig_sp.add_hline(y=CURRENT_PRICE, line_dash="dash", line_color="red",
                     annotation_text=f"Current ${CURRENT_PRICE}")
    fig_sp.update_layout(title="ML Stock Price Ensemble with Bull/Bear Bands",
                         yaxis_title="Stock Price ($)", height=440)
    st.plotly_chart(fig_sp, use_container_width=True)

    # Upside table
    upside_tbl = pd.DataFrame({
        "Year":              cy_labels,
        "Ensemble Target":   [f"${p:.1f}" for p in price_ensemble],
        "Upside (vs $266)":  [f"{(p/CURRENT_PRICE-1)*100:.1f}%" for p in price_ensemble],
        "Implied Annual Ret":["+7%", "+13%", "+15%", "+16%", "+16%"],
        "Bull Band":         [f"${p:.0f}" for p in price_upper],
        "Bear Band":         [f"${p:.0f}" for p in price_lower],
    })
    st.dataframe(upside_tbl, use_container_width=True, hide_index=True)

    # ── Excel MC distribution from model ──
    st.subheader("Model 5: Monte Carlo DCF Distribution (Excel Reference — 10,000 Scenarios)")
    mc_bins = [95,105,115,125,135,145,155,165,175,185,195,205,215,225,235,245,255,265,275]
    mc_freq = [11,71,296,583,1015,1274,1187,1195,1009,838,648,472,377,253,183,130,98,90,56]
    fig_mcd = go.Figure()
    fig_mcd.add_bar(x=[f"${b}" for b in mc_bins], y=mc_freq,
                    marker_color="#1F4E79", name="MC Distribution")
    fig_mcd.add_vline(x=164.6, line_dash="dash", line_color="green",
                      annotation_text="Median $164.6")
    fig_mcd.add_vline(x=172, line_dash="dash", line_color="blue",
                      annotation_text="Mean $172")
    fig_mcd.update_layout(
        title="Excel Model MC Distribution (10,000 scenarios)", height=360,
        xaxis_title="Implied Share Price", yaxis_title="Frequency",
    )
    st.plotly_chart(fig_mcd, use_container_width=True)

    # Composite signal table
    st.subheader("Composite Valuation Signal Summary")
    signal_tbl = pd.DataFrame({
        "Valuation Method":     ["DCF — Management Case","DCF — Street Consensus",
                                 "ML Ensemble (CY2025E)","ML Ensemble (CY2026E)",
                                 "Monte Carlo Median","Monte Carlo 75th Percentile",
                                 "Current Market Price"],
        "Implied Price":        ["$170.60","$128.08","$283.60","$337.30",
                                 "$164.60","$190.20","$266.18 (market)"],
        "vs Current Market":    [f"{170.60/266.18-1:.1%}", f"{128.08/266.18-1:.1%}",
                                 f"{283.60/266.18-1:.1%}", f"{337.30/266.18-1:.1%}",
                                 f"{164.60/266.18-1:.1%}", f"{190.20/266.18-1:.1%}", "—"],
        "Signal":               ["Below Market ↓","Below Market ↓","Near Market →",
                                 "Above Market ↑","Below Market ↓","Below Market ↓","—"],
    })
    st.dataframe(signal_tbl, use_container_width=True, hide_index=True)
    st.warning("⚠️ **Composite Assessment: FAIRLY VALUED to MODERATELY OVERVALUED** at $266 vs DCF range of $128–$171")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — HISTORICAL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Historical Performance — FY2018 to FY2025</div>',
                unsafe_allow_html=True)

    # Key metrics snapshot
    k_cols = st.columns(6)
    metrics = [
        ("Current Price",       f"${CURRENT_PRICE}"),
        ("52-Week High",        "$288.62"),
        ("52-Week Low",         "$169.21"),
        ("FY2025 Revenue",      "$416.2B"),
        ("FY2025 Net Income",   "$112.0B"),
        ("7-Yr Rev CAGR",       "6.6%"),
    ]
    for col, (lbl, val) in zip(k_cols, metrics):
        col.metric(lbl, val)

    col6a, col6b = st.columns(2)
    with col6a:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=hist_years, y=hist_revenue, name="Revenue ($B)",
                    marker_color="#1F4E79", secondary_y=False)
        fig.add_scatter(x=hist_years, y=hist_net_income, name="Net Income ($B)",
                        mode="lines+markers", line=dict(color="#FF6B35", width=3),
                        marker=dict(size=8), secondary_y=True)
        fig.update_yaxes(title_text="Revenue ($B)",    secondary_y=False)
        fig.update_yaxes(title_text="Net Income ($B)", secondary_y=True)
        fig.update_layout(title="Revenue & Net Income (FY2018–FY2025)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col6b:
        fig2 = go.Figure()
        fig2.add_bar(x=hist_years, y=hist_products, name="Products",
                     marker_color="#1F4E79")
        fig2.add_bar(x=hist_years, y=hist_services, name="Services",
                     marker_color="#00B050")
        fig2.update_layout(title="Revenue Mix: Products vs Services ($B)",
                           barmode="stack", height=380, yaxis_title="Revenue ($B)")
        st.plotly_chart(fig2, use_container_width=True)

    col6c, col6d = st.columns(2)
    with col6c:
        fig3 = go.Figure()
        fig3.add_scatter(x=hist_years, y=hist_gm, mode="lines+markers",
                         fill="tozeroy", fillcolor="rgba(0,176,80,0.15)",
                         line=dict(color="#00B050", width=3), marker=dict(size=9),
                         text=[f"{g:.1f}%" for g in hist_gm], textposition="top center")
        fig3.update_layout(title="Gross Margin % Trend", height=380,
                           yaxis_title="Gross Margin (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with col6d:
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_bar(x=hist_years, y=hist_eps, name="Diluted EPS ($)",
                     marker_color="#1F4E79", secondary_y=False)
        fig4.add_scatter(x=hist_years, y=hist_price, name="Stock Price ($)",
                         mode="lines+markers", line=dict(color="#FF6B35", width=3),
                         marker=dict(size=8), secondary_y=True)
        fig4.update_yaxes(title_text="EPS ($)",          secondary_y=False)
        fig4.update_yaxes(title_text="Stock Price ($)",  secondary_y=True)
        fig4.update_layout(title="EPS & Stock Price (FY2018–FY2025)", height=380)
        st.plotly_chart(fig4, use_container_width=True)

    # Services as % of revenue
    fig5 = go.Figure()
    svc_pct_hist = [s/r*100 for s, r in zip(hist_services, hist_revenue)]
    fig5.add_scatter(x=hist_years, y=svc_pct_hist, mode="lines+markers",
                     fill="tozeroy", fillcolor="rgba(31,78,121,0.12)",
                     line=dict(color="#1F4E79", width=3), marker=dict(size=9),
                     text=[f"{p:.1f}%" for p in svc_pct_hist], textposition="top center")
    fig5.update_layout(title="Services as % of Total Revenue (FY2018–FY2025)",
                       height=300, yaxis_title="Services %")
    st.plotly_chart(fig5, use_container_width=True)

    # Historical data table
    st.markdown('<div class="section-header">Full Historical Data Table</div>',
                unsafe_allow_html=True)
    hist_df = pd.DataFrame({
        "Fiscal Year":         [f"FY{y}" for y in hist_years],
        "Revenue ($B)":        hist_revenue,
        "Net Income ($B)":     hist_net_income,
        "Gross Margin %":      [f"{g:.1f}%" for g in hist_gm],
        "Services Rev ($B)":   hist_services,
        "Products Rev ($B)":   [round(p, 2) for p in hist_products],
        "Services Mix":        [f"{s/r*100:.1f}%" for s, r in zip(hist_services, hist_revenue)],
        "Op Income ($B)":      hist_op_income,
        "Diluted EPS":         [f"${e:.2f}" for e in hist_eps],
        "Stock Price":         [f"${p:.1f}" for p in hist_price],
        "Revenue Growth":      ["—"] + [f"{(hist_revenue[i]-hist_revenue[i-1])/hist_revenue[i-1]:.1%}"
                                        for i in range(1, len(hist_revenue))],
    })
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — RISK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-header">Risk Analytics — VaR, Correlation & Statistical Diagnostics</div>',
                unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Annual Volatility (σ)",    "31.7%")
    r2.metric("95% VaR (1-year)",         "18.4%")
    r3.metric("Expected Shortfall (CVaR)","31.3%")

    # Correlation matrix
    st.subheader("Correlation Matrix — R cor() Equivalent")
    vars_     = ["Revenue","Net Income","Gross Margin","Services Rev","Op Income","EPS","Stock Price"]
    corr_data = np.array([
        [1.000, 0.989, 0.956, 0.934, 0.995, 0.988, 0.835],
        [0.989, 1.000, 0.932, 0.908, 0.986, 0.989, 0.809],
        [0.956, 0.932, 1.000, 0.977, 0.970, 0.959, 0.905],
        [0.934, 0.908, 0.977, 1.000, 0.933, 0.954, 0.951],
        [0.995, 0.986, 0.970, 0.933, 1.000, 0.984, 0.836],
        [0.988, 0.989, 0.959, 0.954, 0.984, 1.000, 0.865],
        [0.835, 0.809, 0.905, 0.951, 0.836, 0.865, 1.000],
    ])
    # Also compute from actual historical data
    hist_matrix = np.column_stack([
        hist_revenue, hist_net_income, hist_gm,
        hist_services, hist_op_income, hist_eps, hist_price
    ])
    corr_calc = np.corrcoef(hist_matrix, rowvar=False)

    col7a, col7b = st.columns(2)
    with col7a:
        fig_corr = px.imshow(
            corr_data, x=vars_, y=vars_,
            color_continuous_scale="RdYlGn", zmin=0.75, zmax=1.0,
            text_auto=".2f",
            title="Correlation Matrix (Excel Model Reference)",
        )
        fig_corr.update_layout(height=480)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("🔑 Key insight: Services Revenue has highest correlation with Stock Price (r=0.951)")

    with col7b:
        fig_corr2 = px.imshow(
            corr_calc, x=vars_, y=vars_,
            color_continuous_scale="RdYlGn", zmin=0.75, zmax=1.0,
            text_auto=".2f",
            title="Correlation Matrix (Computed from Historical Data)",
        )
        fig_corr2.update_layout(height=480)
        st.plotly_chart(fig_corr2, use_container_width=True)

    # VaR table
    st.subheader("Value at Risk (VaR) & Risk Metrics")
    var_tbl = pd.DataFrame({
        "Metric":         ["Historical VaR 90%", "Historical VaR 95%", "Historical VaR 99%",
                           "Parametric VaR 90%", "Parametric VaR 95%", "Parametric VaR 99%",
                           "Annual Volatility",   "Expected Shortfall (CVaR 95%)",
                           "Dollar VaR $1M — 90%","Dollar VaR $1M — 95%","Dollar VaR $1M — 99%"],
        "Value":          ["5.5%","18.4%","28.7%",
                           "12.4%","23.9%","45.5%",
                           "31.7%","31.3%",
                           "$53,801","$168,094","$249,505"],
        "Interpretation": [
            "10% of years, loss > 5.5%","5% of years, loss > 18.4%","1% of years, loss > 28.7%",
            "Parametric (normal) 90%","Parametric (normal) 95%","Parametric (normal) 99%",
            "Calibrated from 8yr log-returns","Average loss when VaR is breached",
            "Per $1M position","Per $1M position","Per $1M position"
        ]
    })
    st.dataframe(var_tbl, use_container_width=True, hide_index=True)

    # Statistical diagnostics
    st.subheader("Statistical Diagnostics — Normality Tests (R-style)")
    diag_tbl = pd.DataFrame({
        "Test":          ["Shapiro-Wilk (normality)","Jarque-Bera (normality)",
                          "Kolmogorov-Smirnov","Autocorrelation (lag-1)","Hurst Exponent"],
        "Statistic":     ["0.910","0.803","0.210","0.088","0.500"],
        "p-value":       ["0.395","0.669","0.861","—","—"],
        "Result":        ["✅ Normal (p > 0.05)", "✅ Normal (p > 0.05)",
                          "✅ Cannot reject normality",
                          "Low autocorrelation","Random walk (H=0.5)"],
        "Interpretation":["GBM assumption is valid for AAPL",
                          "Returns distribution is approximately normal",
                          "Annual returns pass normality test",
                          "Returns are not autocorrelated",
                          "EMH supported at annual frequency"],
    })
    st.dataframe(diag_tbl, use_container_width=True, hide_index=True)

    # Bayesian revenue forecast
    st.subheader("Bayesian Revenue Forecast  (Normal-Inverse-Gamma Prior)")
    bay_yrs  = [2026, 2027, 2028, 2029, 2030]
    bay_mean = [454.4, 478.9, 503.5, 528.1, 552.6]
    bay_hi   = [540.6, 571.4, 602.8, 634.9, 667.4]
    bay_lo   = [368.1, 386.5, 404.2, 421.3, 437.9]
    bay_ou   = [473.2, 504.1, 537.0, 569.8, 605.0]

    fig_bay = go.Figure()
    fig_bay.add_scatter(x=bay_yrs, y=bay_hi, mode="lines", name="Upper 95% CI",
                        line=dict(color="rgba(0,176,80,0.5)", width=1.5, dash="dash"))
    fig_bay.add_scatter(x=bay_yrs, y=bay_lo, mode="lines", name="Lower 95% CI",
                        line=dict(color="rgba(200,0,0,0.5)", width=1.5, dash="dash"),
                        fill="tonexty", fillcolor="rgba(31,78,121,0.07)")
    fig_bay.add_scatter(x=bay_yrs, y=bay_mean, mode="lines+markers", name="Bayesian Mean",
                        line=dict(color="#1F4E79", width=3), marker=dict(size=9))
    fig_bay.add_scatter(x=bay_yrs, y=bay_ou, mode="lines+markers", name="OU Process Median",
                        line=dict(color="#00B050", width=2.5, dash="dot"), marker=dict(size=7))
    # Historical context
    fig_bay.add_scatter(x=hist_years[-3:], y=hist_revenue[-3:], mode="lines+markers",
                        name="Historical", line=dict(color="#888", width=2, dash="longdash"))
    fig_bay.update_layout(
        title="Bayesian Revenue Forecast with 95% Credible Interval ($B)",
        height=420, yaxis_title="Revenue ($B)",
    )
    st.plotly_chart(fig_bay, use_container_width=True)

    # Probability analysis
    st.subheader("GBM Probability Analysis")
    prob_data = pd.DataFrame({
        "Scenario":        ["P(Price > $300)", "P(Price > $400)", "P(Price < $150)"],
        "Year 1 (2026E)":  ["63.9%", "29.0%", "0.6%"],
        "Year 3 (2028E)":  ["85.3%", "70.4%", "1.1%"],
        "Year 5 (2030E)":  ["93.0%", "85.6%", "0.8%"],
        "Source":          ["Excel GBM 50k paths","Excel GBM 50k paths","Excel GBM 50k paths"],
    })
    st.dataframe(prob_data, use_container_width=True, hide_index=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: This dashboard is for educational purposes only. "
    "It is NOT investment advice. Data sourced from Apple Inc. 10-K FY2024 (SEC EDGAR). "
    "All ML predictions and stochastic models are based on historical trend extrapolation "
    "and statistical models. Past performance does not guarantee future results. "
    "Always conduct thorough fundamental analysis before making investment decisions."
)

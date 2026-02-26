# Apple Inc. (AAPL) — Interactive DCF & ML Analytics Dashboard

> A comprehensive, interactive financial valuation and analytics platform for Apple Inc. (AAPL), combining Discounted Cash Flow modeling, Monte Carlo simulation, stochastic processes, machine learning forecasting, and risk analytics — all in a single Streamlit web application.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Live Demo](#live-demo)
- [Dashboard Modules](#dashboard-modules)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
  - [Interactive Controls](#interactive-controls)
  - [Scenario Analysis](#scenario-analysis)
- [Methodology](#methodology)
  - [DCF Valuation](#dcf-valuation)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
  - [Stochastic Models](#stochastic-models)
  - [Machine Learning Models](#machine-learning-models)
  - [Risk Analytics](#risk-analytics)
- [Data Sources](#data-sources)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Limitations & Disclaimer](#limitations--disclaimer)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This dashboard was built as an end-to-end financial analytics project to demonstrate how modern quantitative finance techniques can be combined with machine learning and interactive tooling to analyze a single equity — Apple Inc. (AAPL).

Rather than relying on a single valuation methodology, this tool layers **six distinct analytical lenses** on top of the same underlying financial data, allowing users to stress-test assumptions, compare outputs across frameworks, and develop a more nuanced, probabilistic view of Apple's intrinsic value and price trajectory.

**Who is this for?**

| Audience | Use Case |
|---|---|
| Finance students | Learn DCF modeling, stochastic processes, and risk analytics in one interactive environment |
| Individual investors | Explore Apple's valuation from multiple angles and stress-test key assumptions |
| Data scientists | See how ML/statistical models are applied in a real equity context |
| Analysts | Use as a cross-validation reference against your own Excel models |

**Bottom Line (as of FY2024 data snapshot):**
Apple at **$266.18/share** is assessed as **fairly valued to moderately overvalued** under DCF methodology (fair value range: $128–$171), while machine learning ensemble models suggest potential upside to **$337–$571 by 2029** under base-case assumptions.

---

## Key Features

- **7 Fully Interactive Tabs** — each tab is a self-contained analytical module
- **Live DCF Engine** — adjust growth rates, margins, WACC, and terminal growth rate in real-time and see the implied price update instantly
- **10,000-Path Monte Carlo** — probabilistic scenario analysis with configurable simulation sizes (5k, 10k, 25k, 50k paths)
- **3 Stochastic Price Path Models** — Geometric Brownian Motion (GBM), Merton Jump-Diffusion, and Ornstein-Uhlenbeck process
- **5 ML Ensemble Models** — revenue, gross margin, EPS, stock price, and MC distribution reference models
- **WACC/TGR Sensitivity Heatmaps** — see how every combination of key assumptions maps to an implied share price
- **8 Years of Historical Data** — FY2018–FY2025 financials with trend analysis and growth rate decomposition
- **Full Risk Analytics Suite** — VaR (Historical + Parametric), CVaR/Expected Shortfall, correlation matrix, and statistical diagnostics
- **Bayesian Revenue Forecasts** — Normal-Inverse-Gamma priors with 95% credible intervals
- **Excel Model Cross-Validation** — comparison tables and reference heatmaps for verifying model output integrity
- **Custom Professional UI** — corporate-grade gradient cards, responsive layout, and Plotly interactive charts

---

## Live Demo

To run the dashboard locally:

```bash
git clone https://github.com/Kayariyan28/-Apple-Inc.-AAPL-Interactive-DCF-ML-Analytics-Dashboard.git
cd -Apple-Inc.-AAPL-Interactive-DCF-ML-Analytics-Dashboard
pip install -r requirements.txt
streamlit run apple_dcf_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Dashboard Modules

### Tab 1 — DCF Model

The core valuation engine. Build a 5-year discounted cash flow model interactively.

- **Revenue projections**: Per-year growth rate sliders (FY2025–FY2029)
- **P&L build**: Gross margin → EBIT → NOPAT → Unlevered Free Cash Flow (UFCF)
- **Terminal Value**: Gordon Growth Model with adjustable Terminal Growth Rate
- **Equity Bridge**: Enterprise value → net debt adjustment → equity value → implied price per share (waterfall chart)
- **Scenario Comparison**: Management Case vs. Street Consensus side-by-side

Base Year (FY2024) Assumptions:

| Metric | Value |
|---|---|
| Revenue | $394,328M |
| Cash & Equivalents | $156,652M |
| Total Debt | $96,842M |
| Diluted Shares Outstanding | 15,115M |
| Market Price | $266.18 |

---

### Tab 2 — Monte Carlo Simulation

Run up to 50,000 probabilistic scenarios by randomizing the four key value drivers simultaneously.

**Randomized Variables:**

| Variable | Distribution | Std Dev |
|---|---|---|
| WACC | Normal | ±1.0% |
| Terminal Growth Rate | Normal | ±0.5% |
| Revenue Growth | Normal | ±2.5% |
| Gross Margins | Normal | ±1.5% |

**Output:** Full distribution of implied share prices with percentile breakdown (P5, P25, P50, P75, P95), histogram overlay with current market price, and CDF for probability-weighted analysis.

---

### Tab 3 — Stochastic & GBM Models

Applies three stochastic process models to simulate Apple's stock price paths over a 5-year horizon.

**Geometric Brownian Motion (GBM)**
- 50,000 simulated paths calibrated from 8 years of annual log-returns
- Drift (μ): 28.3% annual | Volatility (σ): 31.7% annual
- Percentile fan chart (P5 / P25 / P50 / P75 / P95)

**Merton Jump-Diffusion Model**
- Extends GBM with Poisson-distributed jump events (λ = 0.3 per year)
- Mean jump size: −5% | Jump volatility: 15%
- Captures fat-tail, event-driven drawdown risk

**Ornstein-Uhlenbeck (OU) Process**
- Applied to revenue growth (mean-reverting assumption)
- Parameters: θ = 0.5 (speed of reversion), μ = 7.2%, σ = 12.2%

**Bayesian Regression Revenue Forecast**
- Normal-Inverse-Gamma conjugate prior
- Point estimates + 95% credible intervals through FY2029

---

### Tab 4 — Sensitivity Heatmaps

Understand how sensitive the DCF output is to changes in the two most impactful assumptions.

- **Live Heatmap**: WACC (7.0%–12.0%) vs. Terminal Growth Rate (1.5%–5.0%) — 11×8 grid, color-coded relative to current market price ($266)
- **Excel Reference Heatmap**: Hardcoded from external Excel model for cross-validation
- **Scenario Revenue Comparison**: Management Case vs. Street Consensus charted over the 5-year explicit period

---

### Tab 5 — ML Predictions

Five machine learning ensemble models applied to forecast key Apple financial metrics.

| Model | Target | Approach | R² |
|---|---|---|---|
| Model 1 | Revenue (FY2025–2029) | Linear + Polynomial + Exponential ensemble | ~0.85 |
| Model 2 | Gross Margin (FY2025–2029) | Trend regression with Services mix overlay | ~0.77 |
| Model 3 | EPS (FY2025–2029) | Multi-factor regression | ~0.86 |
| Model 4 | Stock Price (CY2025–2029) | Weighted ensemble (30% time, 35% revenue, 35% EPS) | ~0.85 |
| Model 5 | MC Distribution Reference | 10,000-scenario histogram vs. live simulation | Reference |

**Composite Valuation Signal**: All methods aggregated into a single signal table comparing DCF, GBM, Jump-Diffusion, and ML price targets with "Below / Near / Above Market" indicators.

---

### Tab 6 — Historical Performance

8 years of Apple financial history (FY2018–FY2025) visualized across multiple dimensions.

- Revenue trajectory and YoY growth rates
- Net income and margin expansion
- Products vs. Services revenue split — tracking the services business mix shift
- Services as % of total revenue (trend from ~14% to ~28%)
- EPS evolution and stock price correlation
- Full historical data table with computed growth metrics

---

### Tab 7 — Risk Analytics

A comprehensive risk measurement module covering both market risk and statistical validation.

**Risk Metrics:**

| Metric | Value |
|---|---|
| Annual Volatility (σ) | 31.7% |
| 95% Value at Risk (1-Year) | 18.4% |
| Expected Shortfall (CVaR) | 31.3% |

**Modules:**
- **Correlation Matrix**: 7×7 heatmap of key financial variables (highest: Services Revenue vs. Stock Price at 0.951)
- **Value at Risk**: Historical VaR (90%, 95%, 99%) + Parametric VaR + Dollar VaR for $1M position sizing
- **Statistical Diagnostics**: Shapiro-Wilk, Jarque-Bera, and Kolmogorov-Smirnov normality tests; autocorrelation analysis; Hurst Exponent estimation
- **GBM Probability Table**: Probability of price exceeding $300, $400, or falling below $150 — across Year 1 through Year 5
- **Bayesian Revenue Forecast**: Credible interval projections through 2030

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Recommended: a virtual environment (venv or conda)

### Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/Kayariyan28/-Apple-Inc.-AAPL-Interactive-DCF-ML-Analytics-Dashboard.git
cd -Apple-Inc.-AAPL-Interactive-DCF-ML-Analytics-Dashboard
```

**Step 2: Create and activate a virtual environment (recommended)**

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run apple_dcf_app.py
```

The app will launch and automatically open in your default browser at `http://localhost:8501`.

**Dependencies installed:**

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
plotly>=5.15.0
```

---

## Usage Guide

### Interactive Controls

The sidebar contains the primary controls that drive the DCF model and scenario analysis:

| Control | Range | Default | Impact |
|---|---|---|---|
| Revenue Growth FY2025 | 0% – 20% | 7.5% | Direct revenue projection |
| Revenue Growth FY2026 | 0% – 20% | 8.0% | Direct revenue projection |
| Revenue Growth FY2027 | 0% – 20% | 8.5% | Direct revenue projection |
| Revenue Growth FY2028 | 0% – 20% | 9.0% | Direct revenue projection |
| Revenue Growth FY2029 | 0% – 20% | 9.0% | Direct revenue projection |
| Gross Margin | 40% – 55% | 46.5% | EBIT and FCF generation |
| WACC | 6% – 14% | 9.0% | Discount rate for PV calculations |
| Terminal Growth Rate | 1% – 6% | 3.0% | Terminal value magnitude |
| Tax Rate | 10% – 25% | 15.7% | NOPAT calculation |
| Monte Carlo Scenarios | 5k / 10k / 25k / 50k | 10,000 | Simulation precision |

### Scenario Analysis

Switch between two built-in scenarios using the radio button at the top of Tab 1:

- **Management Case**: Apple's internal forward guidance and consensus analyst estimates — more optimistic growth assumptions
- **Street Consensus**: Broader Wall Street median estimates — generally more conservative

You can then overlay your own custom assumptions via the sidebar sliders to create a third, fully customized scenario.

---

## Methodology

### DCF Valuation

The DCF model uses a standard 2-stage approach:

1. **Explicit Forecast Period (FY2025–FY2029)**: Year-by-year revenue, margin, and free cash flow projections
   - UFCF = NOPAT + D&A − CapEx − Change in NWC
   - Each year's FCF is discounted at WACC to present value

2. **Terminal Value**: Gordon Growth Model applied to FY2029 normalized FCF
   - TV = FCF₂₀₂₉ × (1 + TGR) / (WACC − TGR)
   - Terminal value discounted back 5 years

3. **Equity Bridge**:
   - Enterprise Value = Sum of PV(FCFs) + PV(Terminal Value)
   - Equity Value = Enterprise Value + Cash − Total Debt
   - Implied Price = Equity Value / Diluted Shares Outstanding

Fixed margin assumptions used across the model:

| Line Item | % of Revenue |
|---|---|
| R&D Expense | 7.7% |
| SG&A Expense | 6.3% |
| Depreciation & Amortization | 2.8% |
| Capital Expenditure | 2.5% |
| Change in Net Working Capital | 1.0% |

### Monte Carlo Simulation

The Monte Carlo engine runs N independent trials (configurable 5k–50k), each time drawing random values from normal distributions centered on the base-case DCF assumptions. Each trial runs the full DCF model and records the implied share price. The output distribution represents the model's probabilistic estimate of fair value.

### Stochastic Models

**GBM** models continuous price evolution under the assumption that log-returns are normally distributed. Calibrated parameters come from 8 years of observed annual returns.

**Jump-Diffusion** extends GBM by superimposing discrete jump events modeled via a Poisson process — capturing tail risks like earnings shocks, regulatory actions, or macro dislocations.

**Ornstein-Uhlenbeck** applies mean-reversion to revenue growth, reflecting the tendency for hypergrowth rates to normalize toward long-run GDP-level growth over time.

### Machine Learning Models

The ML module uses ensemble averaging across multiple regression specifications (linear, polynomial, and exponential trend models) trained on the 8-year historical dataset. Ensemble predictions are weighted and combined to produce a final point estimate with confidence bands.

> Note: Given the small training dataset (8 data points), these models are best interpreted as trend extrapolations rather than fully trained predictive models. Results should be used directionally.

### Risk Analytics

**Value at Risk (VaR)**: Estimated using both historical simulation (empirical percentiles of the return distribution) and parametric methods (normal distribution assumption with observed σ).

**Expected Shortfall (CVaR)**: Average loss in the worst-case tail (beyond VaR threshold) — a more complete measure of downside risk than VaR alone.

**Statistical Tests**: The Shapiro-Wilk, Jarque-Bera, and Kolmogorov-Smirnov tests are applied to validate distributional assumptions underlying the parametric risk models.

**Hurst Exponent**: Tests whether the return series exhibits trending (H > 0.5), random walk (H ≈ 0.5), or mean-reverting (H < 0.5) behavior.

---

## Data Sources

All financial data in this application is sourced from publicly available documents:

| Data | Source |
|---|---|
| FY2024 Annual Financials | Apple Inc. Form 10-K (SEC EDGAR) |
| Historical Price Data (FY2018–FY2025) | Public market data |
| Analyst Consensus Estimates | Street consensus (as of model build date) |
| Balance Sheet Items | Apple Inc. Form 10-K FY2024 |

**Important**: This application uses a static data snapshot. It does **not** connect to live market data APIs. Financial figures, stock prices, and analyst estimates reflect the point-in-time data available when the model was built and may not reflect current market conditions.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Web Framework | Streamlit >= 1.28.0 | Interactive UI, sliders, tabs, layout |
| Data Manipulation | Pandas >= 2.0.0 | DataFrames, tabular display, calculations |
| Numerical Computing | NumPy >= 1.24.0 | Array operations, simulation paths |
| Statistical Computing | SciPy >= 1.10.0 | Hypothesis tests, distributions |
| Visualization | Plotly >= 5.15.0 | All interactive charts, heatmaps, subplots |
| Styling | Custom CSS (via Streamlit) | Gradient cards, corporate UI design |

---

## Project Structure

```
-Apple-Inc.-AAPL-Interactive-DCF-ML-Analytics-Dashboard/
│
├── apple_dcf_app.py        # Main application — all logic, models, and UI (965 lines)
├── requirements.txt        # Python package dependencies
├── .gitignore              # Excludes venv/, __pycache__/, .DS_Store, *.pyc
└── README.md               # This file
```

**Key functions inside `apple_dcf_app.py`:**

| Function | Description |
|---|---|
| `run_dcf()` | Core DCF engine — takes growth rates, margin, WACC, TGR, tax rate → returns full model output dict |
| Monte Carlo loop | Lines ~253–271: N-path simulation over randomized DCF assumptions |
| GBM path generator | Lines ~358–365: Standard Wiener process with calibrated drift and volatility |
| Jump-Diffusion model | Lines ~420–431: Poisson jump events superimposed on GBM paths |
| Sensitivity grid | Lines ~500–508: WACC × TGR nested loop for heatmap generation |
| ML model implementations | Lines ~592–684: Ensemble forecasting for revenue, margin, EPS, and price |

---

## Key Findings

Based on the FY2024 data snapshot used in this model:

| Valuation Method | Implied Price | vs. Market ($266.18) |
|---|---|---|
| DCF — Management Case | $170.60 | −35.9% |
| DCF — Street Consensus | $128.08 | −51.9% |
| DCF Midpoint | $149.34 | −43.9% |
| GBM Median (5-Year) | ~$430–$470 | +61%–+77% |
| ML Ensemble (CY2029) | $337–$571 | +27%–+115% |
| Monte Carlo P50 | ~$165–$185 | −30%–−35% |

**Interpretation:**
- Pure DCF analysis suggests Apple is trading at a significant premium to intrinsic value under typical discount rate assumptions
- However, stochastic and ML models — which embed the momentum of Apple's historical compounding — suggest the market may be pricing in continued outperformance
- The divergence between DCF and price-based models reflects the market's embedded expectation of sustained high returns on equity and continued Services business mix shift
- Services Revenue shows the strongest correlation with stock price (0.951), making it the most critical variable to monitor

---

## Limitations & Disclaimer

**This dashboard is for educational and informational purposes only. It does not constitute financial advice, investment recommendations, or a solicitation to buy or sell any security.**

Key limitations to be aware of:

- **Static Data**: The model reflects a point-in-time data snapshot and does not update automatically
- **Small ML Training Set**: ML models are trained on 8 annual data points — sufficient for trend analysis but not robust predictive modeling
- **Hardcoded Assumptions**: Fixed margin assumptions (R&D, SG&A, D&A, CapEx, NWC) do not respond to sidebar inputs — only growth rates and top-level metrics are dynamic
- **No Macro Linkage**: The model does not incorporate macroeconomic variables (interest rates, GDP, FX exposure)
- **Single-Stock Focus**: All analysis is specific to Apple and is not designed to generalize to other equities without modification
- **Past Performance**: All historical calibration (GBM drift, volatility, correlations) is based on past data and does not guarantee future results

Always conduct your own independent research and consult a qualified financial professional before making investment decisions.

---

## Contributing

Contributions are welcome. If you'd like to improve the model, add new analytical modules, or fix issues:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes with clear, descriptive commits
4. Push to your fork and open a Pull Request

**Ideas for future contributions:**
- [ ] Live data integration via `yfinance` or `Alpha Vantage` API
- [ ] Peer comparison module (vs. MSFT, GOOGL, AMZN)
- [ ] Automated report export (PDF summary of all tabs)
- [ ] Segment-level revenue modeling (iPhone, Mac, iPad, Wearables, Services)
- [ ] DCF sensitivity to Services margin expansion assumptions
- [ ] Options pricing module (Black-Scholes implied vol surface)
- [ ] Macroeconomic scenario overlays (Fed rate paths)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built as an independent side project. Apple Inc. (AAPL) data sourced from public filings. Not affiliated with or endorsed by Apple Inc.*

# Feri Radar v4

Feri Radar v4 is a local Streamlit dashboard that tracks selected macro and market instruments with public Yahoo Finance data through `yfinance`.

The application does not trade and does not provide financial advice. It is an analytical radar that produces a Feri Score from 0 to 100, a BUY / WAIT / SELL action, automatic 24-hour signal evaluation, and a daily performance summary.

## Markets

| Market | Yahoo Finance symbol |
|---|---|
| DXY | DX-Y.NYB |
| VIX | ^VIX |
| USD/JPY | JPY=X |
| Bitcoin | BTC-USD |
| Gold | GC=F |
| Silver | SI=F |
| Oil | CL=F |
| Copper | HG=F |

## Features

- Dark modern Streamlit UI
- Last update date and time in Europe/Bratislava timezone
- Feri Score from 0 to 100
- BUY / WAIT / SELL action
- Top 3 Markets ranked by Feri Score
- Current price and daily percentage change
- Trend, RSI(14), and short signal explanation
- 30-day price and RSI(14) charts for the selected market
- `signals_history.csv` with one daily signal per market and automatic 24-hour evaluation
- `performance.csv` with one daily summary row per date
- Signal success rate by market: DXY, VIX, BTC, Gold, Silver, Oil, Copper, USDJPY
- Signal success rate by signal type: BUY, WAIT, SELL
- Manual data refresh button

## Signal Logic

Feri Radar combines simple technical momentum with macro filters:

- Moving averages and RSI estimate market momentum.
- DXY is used as the main commodity filter.
- VIX, Bitcoin, and USD/JPY help classify RISK-ON or RISK-OFF conditions.
- Commodity scores can improve when DXY weakens and the commodity is rising.
- Bitcoin, Oil, and Copper are treated as more sensitive to RISK-ON conditions.
- Gold and Silver receive a small defensive boost during RISK-OFF conditions.

The resulting score is converted into:

| Feri Score | Action |
|---|---|
| 67-100 | BUY |
| 34-66 | WAIT |
| 0-33 | SELL |

## Performance Tracking

Feri Radar automatically creates these CSV files in the application folder if they do not exist:

- `signals_history.csv`
- `performance.csv`

Each app run records the current daily signal once per market. It does not duplicate signals for the same market on the same date.

Signals are evaluated automatically after 24 hours:

| Signal | Correct when |
|---|---|
| BUY | Evaluation price is above the signal price |
| SELL | Evaluation price is below the signal price |
| WAIT | Price stays within 0.5% of the signal price |

`performance.csv` keeps one daily summary per date with:

- date
- total signals
- correct signals
- wrong signals
- accuracy %
- best market
- worst market
- best signal type
- weakest signal type

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Start

Run the application:

```bash
py -m streamlit run app.py
```

Open the dashboard at:

```text
http://localhost:8501
```

## Notes

Yahoo Finance availability can vary by symbol and region. If a market cannot be loaded, the dashboard keeps running and marks that row as `NO DATA`.

Feri Radar v4 is for analysis only. It does not execute trades.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


APP_TITLE = "Feri Radar v4"
LOCAL_TZ = ZoneInfo("Europe/Bratislava")
PERIOD = "6mo"
INTERVAL = "1d"
RSI_PERIOD = 14
CHART_DAYS = 30
SIGNALS_HISTORY_FILE = Path("signals_history.csv")
PERFORMANCE_FILE = Path("performance.csv")
EVALUATION_HOURS = 24
WAIT_MOVE_THRESHOLD_PCT = 0.5

SIGNALS_HISTORY_COLUMNS = [
    "signal_id",
    "created_at",
    "signal_date",
    "market",
    "symbol",
    "action",
    "score",
    "price_at_signal",
    "evaluated_at",
    "evaluation_price",
    "price_change_pct",
    "result",
    "is_correct",
]

PERFORMANCE_COLUMNS = [
    "date",
    "total signals",
    "correct signals",
    "wrong signals",
    "accuracy %",
    "best market",
    "worst market",
    "best signal type",
    "weakest signal type",
]

MARKETS: Dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "USD/JPY": "JPY=X",
    "Bitcoin": "BTC-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Oil": "CL=F",
    "Brent Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Coffee": "KC=F",
    "Cocoa": "CC=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Orange Juice": "OJ=F",
    "Sugar": "SB=F",
}

PERFORMANCE_MARKET_NAMES = {
    "Bitcoin": "BTC",
    "USD/JPY": "USDJPY",
}

COMMODITIES = {"Gold", "Silver", "Oil", "Copper"}
CYCLICAL_RISK_MARKETS = {"Bitcoin", "Oil", "Copper"}
DEFENSIVE_MARKETS = {"Gold", "Silver"}

SIGNAL_ORDER = {"BUY": 0, "WAIT": 1, "SELL": 2}
@dataclass(frozen=True)
class MarketContext:
    dxy_trend: str = "NEUTRAL"
    dxy_change: float = 0.0
    vix_trend: str = "NEUTRAL"
    vix_change: float = 0.0
    btc_trend: str = "NEUTRAL"
    btc_change: float = 0.0
    usdjpy_trend: str = "NEUTRAL"
    usdjpy_change: float = 0.0
    risk_mode: str = "NEUTRAL"


@st.cache_data(ttl=300, show_spinner=False)
def download_history(symbol: str, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    available = [column for column in required if column in df.columns]
    if "Close" not in available:
        return pd.DataFrame()

    return df[available].dropna().copy()


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    enriched = df.copy()
    enriched["RSI"] = calculate_rsi(enriched["Close"])
    enriched["MA5"] = enriched["Close"].rolling(5).mean()
    enriched["MA20"] = enriched["Close"].rolling(20).mean()
    enriched["MA50"] = enriched["Close"].rolling(50).mean()
    return enriched.dropna()


def pct_change_last(df: pd.DataFrame) -> float:
    if df.empty or "Close" not in df.columns or len(df) < 2:
        return 0.0

    previous = float(df["Close"].iloc[-2])
    if previous == 0:
        return 0.0

    latest = float(df["Close"].iloc[-1])
    return ((latest - previous) / previous) * 100


def pct_change_days(df: pd.DataFrame, days: int) -> float:
    if df.empty or "Close" not in df.columns or len(df) <= days:
        return 0.0

    base = float(df["Close"].iloc[-days - 1])
    if base == 0:
        return 0.0

    latest = float(df["Close"].iloc[-1])
    return ((latest - base) / base) * 100


def detect_trend(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 20:
        return "NEUTRAL"

    last_close = float(df["Close"].iloc[-1])
    previous_close = float(df["Close"].iloc[-2])
    ma5 = float(df["MA5"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])

    if last_close > ma5 > ma20 and last_close >= previous_close:
        return "UP"
    if last_close < ma5 < ma20 and last_close <= previous_close:
        return "DOWN"
    return "NEUTRAL"


def base_momentum_score(df: pd.DataFrame) -> int:
    if df.empty or len(df) < 20:
        return 50

    latest = float(df["Close"].iloc[-1])
    previous = float(df["Close"].iloc[-2])
    ma5 = float(df["MA5"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])
    change_5d = pct_change_days(df, 5)
    change_20d = pct_change_days(df, 20)

    score = 50
    score += 10 if latest > previous else -10
    score += 10 if latest > ma5 else -10
    score += 10 if ma5 > ma20 else -10

    if 48 <= rsi <= 68:
        score += 10
    elif rsi > 78:
        score -= 15
    elif rsi > 70:
        score -= 5
    elif rsi < 30:
        score += 5
    elif rsi < 42:
        score -= 5

    if change_5d > 2:
        score += 8
    elif change_5d < -2:
        score -= 8

    if change_20d > 4:
        score += 7
    elif change_20d < -4:
        score -= 7

    return clamp_score(score)


def clamp_score(score: float) -> int:
    return int(max(0, min(100, round(score))))


def build_context(histories: Dict[str, pd.DataFrame]) -> MarketContext:
    def trend_and_change(name: str) -> Tuple[str, float]:
        df = histories.get(name, pd.DataFrame())
        if df.empty:
            return "NEUTRAL", 0.0
        return detect_trend(df), pct_change_last(df)

    dxy_trend, dxy_change = trend_and_change("DXY")
    vix_trend, vix_change = trend_and_change("VIX")
    btc_trend, btc_change = trend_and_change("Bitcoin")
    usdjpy_trend, usdjpy_change = trend_and_change("USD/JPY")

    risk_score = 0
    risk_score += -1 if vix_change > 0 else 1 if vix_change < 0 else 0
    risk_score += 1 if btc_change > 0 else -1 if btc_change < 0 else 0
    risk_score += 1 if usdjpy_change > 0 else -1 if usdjpy_change < 0 else 0

    risk_mode = "NEUTRAL"
    if risk_score >= 2:
        risk_mode = "RISK-ON"
    elif risk_score <= -2:
        risk_mode = "RISK-OFF"

    return MarketContext(
        dxy_trend=dxy_trend,
        dxy_change=dxy_change,
        vix_trend=vix_trend,
        vix_change=vix_change,
        btc_trend=btc_trend,
        btc_change=btc_change,
        usdjpy_trend=usdjpy_trend,
        usdjpy_change=usdjpy_change,
        risk_mode=risk_mode,
    )


def adjusted_momentum(name: str, base_score: int, daily_change: float, ctx: MarketContext) -> int:
    score = base_score

    if name in COMMODITIES:
        if ctx.dxy_change < 0 and daily_change > 0:
            score += 15
        elif ctx.dxy_change > 0 and daily_change < 0:
            score -= 15
        elif ctx.dxy_change > 0 and daily_change > 0:
            score -= 5

    if ctx.risk_mode == "RISK-OFF":
        if name in CYCLICAL_RISK_MARKETS:
            score -= 12
        elif name in DEFENSIVE_MARKETS:
            score += 5
    elif ctx.risk_mode == "RISK-ON":
        if name in CYCLICAL_RISK_MARKETS:
            score += 10
        elif name == "VIX":
            score -= 8

    if name == "DXY" and ctx.dxy_trend == "UP":
        score += 4
    if name == "VIX" and ctx.vix_change > 3:
        score -= 8

    return clamp_score(score)


def generate_signal(name: str, trend: str, rsi: float, score: int, ctx: MarketContext) -> Tuple[str, str]:
    reasons = []

    if trend == "UP" and score >= 65 and rsi < 74:
        signal = "BUY"
        reasons.append("positive trend and strong momentum")
    elif trend == "DOWN" and score <= 35 and rsi > 26:
        signal = "SELL"
        reasons.append("negative trend and weak momentum")
    elif score >= 75 and rsi < 72:
        signal = "BUY"
        reasons.append("high score despite a mixed trend")
    elif score <= 25 and rsi > 28:
        signal = "SELL"
        reasons.append("low score confirms market weakness")
    else:
        signal = "WAIT"
        reasons.append("signal is not clean enough")

    if name in COMMODITIES:
        if ctx.dxy_change < 0:
            reasons.append("weaker DXY supports commodities")
        elif ctx.dxy_change > 0:
            reasons.append("stronger DXY pressures commodities")

    if ctx.risk_mode == "RISK-ON":
        reasons.append("risk mode is RISK-ON")
    elif ctx.risk_mode == "RISK-OFF":
        reasons.append("risk mode is RISK-OFF")

    if rsi >= 72:
        reasons.append("RSI is elevated")
    elif rsi <= 30:
        reasons.append("RSI is oversold")

    return signal, "; ".join(reasons[:4]) + "."


def calculate_feri_score(radar_df: pd.DataFrame, ctx: MarketContext) -> int:
    if radar_df.empty or "Feri Score" not in radar_df.columns:
        return 50

    scores = pd.to_numeric(radar_df["Feri Score"], errors="coerce").dropna()
    if scores.empty:
        return 50

    score = float(scores.mean())
    if ctx.risk_mode == "RISK-ON":
        score += 5
    elif ctx.risk_mode == "RISK-OFF":
        score -= 5

    if ctx.vix_change > 5:
        score -= 5

    return clamp_score(score)


def signal_from_score(score: int) -> str:
    if score >= 67:
        return "BUY"
    if score <= 33:
        return "SELL"
    return "WAIT"


def report_market_name(name: str) -> str:
    return PERFORMANCE_MARKET_NAMES.get(name, name)


def ensure_csv(path: Path, columns: list[str]) -> None:
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    changed = False
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
            changed = True

    extra_columns = [column for column in df.columns if column not in columns]
    ordered = df[columns + extra_columns]
    if changed or list(df.columns) != list(ordered.columns):
        ordered.to_csv(path, index=False)


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    ensure_csv(path, columns)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns].copy()


def write_csv_if_changed(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output = output[columns]

    try:
        existing = pd.read_csv(path)
        existing = existing[columns]
        if existing.fillna("").astype(str).equals(output.fillna("").astype(str)):
            return
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        pass

    output.to_csv(path, index=False)


def parse_local_datetime(value: object) -> datetime | None:
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        return parsed.to_pydatetime().replace(tzinfo=LOCAL_TZ)
    return parsed.tz_convert(LOCAL_TZ).to_pydatetime()


def current_price_by_market(radar_df: pd.DataFrame) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    if radar_df.empty:
        return prices

    for _, row in radar_df.iterrows():
        market = str(row.get("Market", ""))
        price = pd.to_numeric(row.get("Price", np.nan), errors="coerce")
        if market and pd.notna(price):
            prices[market] = float(price)
    return prices


def evaluate_signal(action: str, entry_price: float, evaluation_price: float) -> Tuple[str, bool, float]:
    if entry_price == 0:
        return "WRONG", False, 0.0

    change_pct = ((evaluation_price - entry_price) / entry_price) * 100
    if action == "BUY":
        is_correct = evaluation_price > entry_price
    elif action == "SELL":
        is_correct = evaluation_price < entry_price
    else:
        is_correct = abs(change_pct) <= WAIT_MOVE_THRESHOLD_PCT

    return "CORRECT" if is_correct else "WRONG", is_correct, round(change_pct, 4)


def record_daily_signals(history: pd.DataFrame, radar_df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if radar_df.empty:
        return history

    today = now.date().isoformat()
    existing_ids = set(history.get("signal_id", pd.Series(dtype=str)).dropna().astype(str))
    new_rows = []

    for _, row in radar_df.iterrows():
        market = str(row.get("Market", ""))
        price = pd.to_numeric(row.get("Price", np.nan), errors="coerce")
        action = str(row.get("Action", "WAIT"))
        if not market or pd.isna(price):
            continue

        signal_id = f"{today}|{market}"
        if signal_id in existing_ids:
            continue
        score_value = pd.to_numeric(row.get("Feri Score", 50), errors="coerce")
        if pd.isna(score_value):
            score_value = 50

        new_rows.append(
            {
                "signal_id": signal_id,
                "created_at": now.isoformat(timespec="seconds"),
                "signal_date": today,
                "market": market,
                "symbol": row.get("Symbol", ""),
                "action": action,
                "score": int(score_value),
                "price_at_signal": round(float(price), 6),
                "evaluated_at": pd.NA,
                "evaluation_price": pd.NA,
                "price_change_pct": pd.NA,
                "result": pd.NA,
                "is_correct": pd.NA,
            }
        )

    if not new_rows:
        return history

    return pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)


def evaluate_due_signals(history: pd.DataFrame, radar_df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if history.empty:
        return history

    updated = history.copy()
    for column in ["evaluated_at", "evaluation_price", "price_change_pct", "result", "is_correct"]:
        if column in updated.columns:
            updated[column] = updated[column].astype(object)

    prices = current_price_by_market(radar_df)
    due_before = now - timedelta(hours=EVALUATION_HOURS)

    for index, row in updated.iterrows():
        if pd.notna(row.get("evaluated_at")):
            continue

        created_at = parse_local_datetime(row.get("created_at"))
        if created_at is None or created_at > due_before:
            continue

        market = str(row.get("market", ""))
        if market not in prices:
            continue

        entry_price = pd.to_numeric(row.get("price_at_signal"), errors="coerce")
        if pd.isna(entry_price):
            continue

        result, is_correct, change_pct = evaluate_signal(
            str(row.get("action", "WAIT")),
            float(entry_price),
            prices[market],
        )
        updated.at[index, "evaluated_at"] = now.isoformat(timespec="seconds")
        updated.at[index, "evaluation_price"] = round(prices[market], 6)
        updated.at[index, "price_change_pct"] = change_pct
        updated.at[index, "result"] = result
        updated.at[index, "is_correct"] = bool(is_correct)

    return updated


def evaluated_for_date(history: pd.DataFrame, date_text: str) -> pd.DataFrame:
    if history.empty or "evaluated_at" not in history.columns:
        return pd.DataFrame(columns=history.columns)

    evaluated = history[history["evaluated_at"].notna()].copy()
    if evaluated.empty:
        return evaluated

    evaluated["evaluation_date"] = pd.to_datetime(
        evaluated["evaluated_at"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(LOCAL_TZ).dt.date.astype(str)
    return evaluated[evaluated["evaluation_date"] == date_text].copy()


def accuracy_leader(df: pd.DataFrame, group_column: str, highest: bool) -> str:
    if df.empty or group_column not in df.columns:
        return "N/A"

    grouped = (
        df.assign(_correct=df["is_correct"].astype(str).str.lower().eq("true"))
        .groupby(group_column, dropna=True)["_correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    if grouped.empty:
        return "N/A"

    grouped["accuracy"] = grouped["sum"] / grouped["count"]
    grouped = grouped.sort_values(
        ["accuracy", "count", group_column],
        ascending=[not highest, False, True],
    )
    return str(grouped.iloc[0][group_column])


def build_daily_summary(history: pd.DataFrame, now: datetime) -> dict:
    today = now.date().isoformat()
    evaluated_today = evaluated_for_date(history, today)
    if evaluated_today.empty:
        return {
            "date": today,
            "total signals": 0,
            "correct signals": 0,
            "wrong signals": 0,
            "accuracy %": 0.0,
            "best market": "N/A",
            "worst market": "N/A",
            "best signal type": "N/A",
            "weakest signal type": "N/A",
        }

    evaluated_today["market_report"] = evaluated_today["market"].map(report_market_name)
    correct = evaluated_today["is_correct"].astype(str).str.lower().eq("true")
    total = int(len(evaluated_today))
    correct_count = int(correct.sum())
    wrong_count = total - correct_count

    return {
        "date": today,
        "total signals": total,
        "correct signals": correct_count,
        "wrong signals": wrong_count,
        "accuracy %": round((correct_count / total) * 100, 2) if total else 0.0,
        "best market": accuracy_leader(evaluated_today, "market_report", True),
        "worst market": accuracy_leader(evaluated_today, "market_report", False),
        "best signal type": accuracy_leader(evaluated_today, "action", True),
        "weakest signal type": accuracy_leader(evaluated_today, "action", False),
    }


def update_daily_performance(performance: pd.DataFrame, summary: dict) -> pd.DataFrame:
    date_text = str(summary["date"])
    retained = performance[performance["date"].astype(str) != date_text].copy()
    updated = pd.concat([retained, pd.DataFrame([summary])], ignore_index=True)
    return updated.sort_values("date", ascending=True).reset_index(drop=True)


def update_signal_files(radar_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    now = datetime.now(LOCAL_TZ)
    history = read_csv(SIGNALS_HISTORY_FILE, SIGNALS_HISTORY_COLUMNS)
    history = record_daily_signals(history, radar_df, now)
    history = evaluate_due_signals(history, radar_df, now)
    write_csv_if_changed(SIGNALS_HISTORY_FILE, history, SIGNALS_HISTORY_COLUMNS)

    performance = read_csv(PERFORMANCE_FILE, PERFORMANCE_COLUMNS)
    summary = build_daily_summary(history, now)
    performance = update_daily_performance(performance, summary)
    write_csv_if_changed(PERFORMANCE_FILE, performance, PERFORMANCE_COLUMNS)

    return history, performance, summary


def success_rate_table(history: pd.DataFrame, group_column: str, labels: list[str] | None = None) -> pd.DataFrame:
    base_columns = [group_column, "Total", "Correct", "Wrong", "Accuracy %"]
    if history.empty or "evaluated_at" not in history.columns:
        return pd.DataFrame(columns=base_columns)

    evaluated = history[history["evaluated_at"].notna()].copy()
    if evaluated.empty:
        table = pd.DataFrame({group_column: labels or []})
        for column in base_columns[1:]:
            table[column] = 0 if column != "Accuracy %" else 0.0
        return table

    if group_column == "Market":
        evaluated[group_column] = evaluated["market"].map(report_market_name)
    else:
        evaluated[group_column] = evaluated["action"]

    evaluated["_correct"] = evaluated["is_correct"].astype(str).str.lower().eq("true")
    grouped = (
        evaluated.groupby(group_column, dropna=False)["_correct"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "Total", "sum": "Correct"})
    )
    grouped["Correct"] = grouped["Correct"].astype(int)
    grouped["Wrong"] = grouped["Total"] - grouped["Correct"]
    grouped["Accuracy %"] = np.where(
        grouped["Total"] > 0,
        (grouped["Correct"] / grouped["Total"] * 100).round(2),
        0.0,
    )

    if labels:
        grouped = pd.DataFrame({group_column: labels}).merge(grouped, on=group_column, how="left")
        grouped[["Total", "Correct", "Wrong", "Accuracy %"]] = grouped[
            ["Total", "Correct", "Wrong", "Accuracy %"]
        ].fillna(0)

    return grouped[base_columns].sort_values(["Accuracy %", "Total"], ascending=[False, False])


def build_radar() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], MarketContext, str]:
    histories: Dict[str, pd.DataFrame] = {}

    for name, symbol in MARKETS.items():
        raw = download_history(symbol)
        histories[name] = add_indicators(raw) if not raw.empty else pd.DataFrame()

    ctx = build_context(histories)
    updated_at = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    rows = []

    for name, symbol in MARKETS.items():
        df = histories.get(name, pd.DataFrame())
        if df.empty or len(df) < 20:
            rows.append(
                {
                    "Market": name,
                    "Symbol": symbol,
                    "Last update": updated_at,
                    "Price": np.nan,
                    "Daily change %": np.nan,
                    "Trend": "NO DATA",
                    "RSI(14)": np.nan,
                    "Feri Score": 50,
                    "Action": "WAIT",
                    "Reason": "Not enough data from yfinance.",
                }
            )
            continue

        price = float(df["Close"].iloc[-1])
        daily_change = pct_change_last(df)
        trend = detect_trend(df)
        rsi = float(df["RSI"].iloc[-1])
        base_score = base_momentum_score(df)
        score = adjusted_momentum(name, base_score, daily_change, ctx)
        signal, reason = generate_signal(name, trend, rsi, score, ctx)

        rows.append(
            {
                "Market": name,
                "Symbol": symbol,
                "Last update": updated_at,
                "Price": round(price, 4),
                "Daily change %": round(daily_change, 2),
                "Trend": trend,
                "RSI(14)": round(rsi, 2),
                "Feri Score": score,
                "Action": signal,
                "Reason": reason,
            }
        )

    return pd.DataFrame(rows), histories, ctx, updated_at


def inject_dark_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07090d;
            --panel: #111821;
            --panel-2: #151f2b;
            --text: #f3f7fb;
            --muted: #a8b3c2;
            --line: #26384d;
            --buy: #2ee66b;
            --wait: #ffd84d;
            --sell: #ff5c5c;
            --accent: #58c7ff;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(7, 9, 13, 0.88);
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1 {
            color: var(--text);
            font-size: clamp(3rem, 5vw, 5rem) !important;
            line-height: 1 !important;
            letter-spacing: 0 !important;
            margin-bottom: 0.4rem !important;
        }

        h2, h3, p, div, span, label {
            letter-spacing: 0 !important;
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 20px;
            min-height: 116px;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text) !important;
        }

        .feri-hero {
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                radial-gradient(circle at 18% 0%, rgba(88, 199, 255, 0.18), transparent 34%),
                linear-gradient(135deg, #0b1017 0%, #121b27 56%, #080b10 100%);
            padding: 34px;
            margin-bottom: 24px;
        }

        .feri-subtitle {
            color: var(--muted);
            font-size: 1.18rem;
            max-width: 1040px;
        }

        .status-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 20px;
            min-height: 132px;
        }

        .status-label {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 8px;
        }

        .status-value {
            color: var(--text);
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.05;
        }

        .status-note {
            color: var(--muted);
            font-size: 0.98rem;
            margin-top: 9px;
        }

        .signal-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 88px;
            padding: 7px 12px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 800;
        }

        .signal-buy {
            color: var(--buy);
            background: rgba(46, 230, 107, 0.13);
            border: 1px solid rgba(46, 230, 107, 0.48);
        }

        .signal-wait {
            color: var(--wait);
            background: rgba(255, 216, 77, 0.13);
            border: 1px solid rgba(255, 216, 77, 0.45);
        }

        .signal-sell {
            color: var(--sell);
            background: rgba(255, 92, 92, 0.13);
            border: 1px solid rgba(255, 92, 92, 0.48);
        }

        .top-card {
            background: var(--panel-2);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 20px;
            min-height: 178px;
        }

        .top-rank {
            color: var(--muted);
            font-size: 0.94rem;
        }

        .top-name {
            color: var(--text);
            font-size: 1.32rem;
            font-weight: 800;
            margin-top: 4px;
        }

        .top-price {
            color: var(--text);
            font-size: 1.7rem;
            font-weight: 800;
            margin: 8px 0 4px;
        }

        .top-meta {
            color: var(--muted);
            font-size: 0.98rem;
            margin-bottom: 8px;
        }

        .top-details {
            color: var(--muted);
            font-size: 0.94rem;
            margin-bottom: 12px;
        }

        .score-good { color: var(--buy); }
        .score-mid { color: var(--wait); }
        .score-bad { color: var(--sell); }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def score_class(score: int) -> str:
    if score >= 67:
        return "score-good"
    if score <= 33:
        return "score-bad"
    return "score-mid"


def signal_badge(signal: str) -> str:
    css_signal = signal.lower() if signal in SIGNAL_ORDER else "wait"
    return f'<span class="signal-badge signal-{css_signal}">{signal}</span>'


def render_status_card(label: str, value: str, note: str = "", css_class: str = "") -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value {css_class}">{value}</div>
            <div class="status-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_price(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def format_change(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def render_top_market(row: pd.Series, rank: int) -> None:
    score = int(row.get("Feri Score", 50))
    action = str(row.get("Action", "WAIT"))
    st.markdown(
        f"""
        <div class="top-card">
            <div class="top-rank">#{rank} Top Market</div>
            <div class="top-name">{row.get("Market", "N/A")}</div>
            <div class="top-price">{format_price(row.get("Price", np.nan))}</div>
            <div class="top-meta">Daily {format_change(row.get("Daily change %", np.nan))} | Trend {row.get("Trend", "N/A")}</div>
            <div class="top-details">RSI(14) {row.get("RSI(14)", "N/A")} | Symbol {row.get("Symbol", "N/A")}</div>
            <div class="{score_class(score)}"><strong>Feri Score {score}/100</strong></div>
            <div style="margin-top: 12px;">{signal_badge(action)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_table(radar_df: pd.DataFrame) -> pd.DataFrame:
    if radar_df.empty:
        return radar_df

    table = radar_df.copy()
    table["_order"] = table["Action"].map(SIGNAL_ORDER).fillna(9)
    table["_score"] = pd.to_numeric(table["Feri Score"], errors="coerce").fillna(0)
    return table.sort_values(["_order", "_score"], ascending=[True, False]).drop(columns=["_order", "_score"])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":chart_with_upwards_trend:", layout="wide")
    inject_dark_theme()

    st.markdown(
        f"""
        <div class="feri-hero">
            <h1>{APP_TITLE}</h1>
            <div class="feri-subtitle">
                Dark modern market radar for DXY, VIX, USD/JPY, Bitcoin, Gold, Silver, Oil, and Copper.
                It calculates RSI(14), a 0-100 Feri Score, BUY / WAIT / SELL actions, and the Top 3 Markets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        selected_market = st.selectbox("Market chart", list(MARKETS.keys()))
        st.caption("Data is cached for 5 minutes.")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading market data from yfinance..."):
        radar_df, histories, context, updated_at = build_radar()

    signals_history_df, performance_df, daily_summary = update_signal_files(radar_df)
    feri_score = calculate_feri_score(radar_df, context)
    feri_signal = signal_from_score(feri_score)
    buy_count = int((radar_df["Action"] == "BUY").sum()) if not radar_df.empty else 0
    wait_count = int((radar_df["Action"] == "WAIT").sum()) if not radar_df.empty else 0
    sell_count = int((radar_df["Action"] == "SELL").sum()) if not radar_df.empty else 0

    st.subheader("Market State")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_status_card("Feri Score", f"{feri_score}/100", "Overall 0-100 market score", score_class(feri_score))
    with col2:
        render_status_card("Last update", updated_at, "Europe/Bratislava time")
    with col3:
        render_status_card("Risk mode", context.risk_mode, "Macro filter from VIX, BTC, USD/JPY")
    with col4:
        render_status_card("Action", feri_signal, f"BUY {buy_count} | WAIT {wait_count} | SELL {sell_count}", score_class(feri_score))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("DXY", context.dxy_trend, f"{context.dxy_change:.2f}%")
    col6.metric("VIX", context.vix_trend, f"{context.vix_change:.2f}%")
    col7.metric("Bitcoin", context.btc_trend, f"{context.btc_change:.2f}%")
    col8.metric("USD/JPY", context.usdjpy_trend, f"{context.usdjpy_change:.2f}%")

    st.subheader("Top 3 Markets")
    if radar_df.empty:
        st.info("No market data is available yet.")
    else:
        top_markets = radar_df.copy()
        top_markets["_score"] = pd.to_numeric(top_markets["Feri Score"], errors="coerce").fillna(0)
        top_markets = top_markets.sort_values("_score", ascending=False).head(3)
        top_cols = st.columns(3)
        for index, (_, row) in enumerate(top_markets.iterrows(), start=1):
            with top_cols[index - 1]:
                render_top_market(row, index)

    st.subheader("Radar Table")
    st.dataframe(display_table(radar_df), use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("BUY", buy_count)
    m2.metric("WAIT", wait_count)
    m3.metric("SELL", sell_count)

    st.subheader("Daily Performance Summary")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Summary date", daily_summary["date"])
    p2.metric("Total signals", int(daily_summary["total signals"]))
    p3.metric("Correct / Wrong", f"{int(daily_summary['correct signals'])} / {int(daily_summary['wrong signals'])}")
    p4.metric("Accuracy", f"{float(daily_summary['accuracy %']):.2f}%")

    p5, p6, p7, p8 = st.columns(4)
    p5.metric("Best market", daily_summary["best market"])
    p6.metric("Worst market", daily_summary["worst market"])
    p7.metric("Best signal type", daily_summary["best signal type"])
    p8.metric("Weakest signal type", daily_summary["weakest signal type"])

    st.subheader("Signal Success Rate")
    market_labels = ["DXY", "VIX", "BTC", "Gold", "Silver", "Oil", "Copper", "USDJPY"]
    signal_labels = ["BUY", "WAIT", "SELL"]
    success_col1, success_col2 = st.columns(2)
    with success_col1:
        st.caption("By market")
        st.dataframe(
            success_rate_table(signals_history_df, "Market", market_labels),
            use_container_width=True,
            hide_index=True,
        )
    with success_col2:
        st.caption("By signal type")
        st.dataframe(
            success_rate_table(signals_history_df, "Signal type", signal_labels),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("CSV history"):
        st.caption("signals_history.csv keeps each daily market signal and its automatic 24-hour evaluation.")
        st.dataframe(signals_history_df.tail(30), use_container_width=True, hide_index=True)
        st.caption("performance.csv keeps one daily summary row per date.")
        st.dataframe(performance_df.tail(30), use_container_width=True, hide_index=True)

    st.subheader(f"30-day chart: {selected_market}")
    chart_df = histories.get(selected_market, pd.DataFrame())
    if chart_df.empty:
        st.warning("No chart data is available for the selected market.")
    else:
        chart_30d = chart_df.tail(CHART_DAYS)
        price_fig = px.line(chart_30d, x=chart_30d.index, y="Close", title=f"{selected_market} 30-day price")
        price_fig.update_layout(template="plotly_dark", font_size=15, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(price_fig, use_container_width=True)

        rsi_fig = px.line(chart_30d, x=chart_30d.index, y="RSI", title=f"{selected_market} RSI(14) - 30 days")
        rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="RSI 70")
        rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="RSI 30")
        rsi_fig.update_layout(template="plotly_dark", font_size=15, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(rsi_fig, use_container_width=True)

    st.warning(
        "Feri Radar is an analytical tool only. It is not financial advice and it does not place real trades."
    )


if __name__ == "__main__":
    main()

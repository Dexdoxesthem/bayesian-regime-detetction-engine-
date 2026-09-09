"""Data source configuration for Indian equity regime detection."""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class YFinanceSource:
    """Configuration for yfinance-based data sources."""
    ticker: str
    field: str  # 'history' or 'info'
    alias: str
    start_date: str = "2010-01-01"
    end_date: Optional[str] = None  # None = today
    interval: str = "1d"
    auto_adjust: bool = True


# NSE indices via yfinance (use .NS suffix)
# NOTE: Nifty Smallcap 100 not available on Yahoo Finance.
# Using Nifty Bank + Nifty IT as substitutes for cap/sector diversification.
YFINANCE_SOURCES: List[YFinanceSource] = [
    YFinanceSource(ticker="^NSEI", field="history", alias="nifty50",
                   start_date="2010-01-01"),
    YFinanceSource(ticker="^NSEMDCP50", field="history", alias="nifty_midcap",
                   start_date="2010-01-01"),
    YFinanceSource(ticker="^NSEBANK", field="history", alias="nifty_bank",
                   start_date="2010-01-01"),
    YFinanceSource(ticker="^INDIAVIX", field="history", alias="india_vix",
                   start_date="2015-01-01"),  # VIX available from ~2015
    YFinanceSource(ticker="INR=X", field="history", alias="usdinr",
                   start_date="2010-01-01"),
]


@dataclass
class WebSource:
    """Configuration for web-scraped data sources."""
    name: str
    url: str
    description: str
    frequency: str  # 'daily', 'monthly'


# FII/DII data sources
FIIDII_SOURCES = {
    "nse_fiidii": WebSource(
        name="NSE FII/DII Cash Market",
        url="https://www.nseindia.com/api/fiidiiTradeReact",
        description="Daily FII and DII cash market buy/sell",
        frequency="daily",
    ),
}

# AMFI SIP data
AMFI_SIP_SOURCE = WebSource(
    name="AMFI Monthly SIP",
    url="https://www.amfiindia.com/SpicesReport/Flow-Spices-Report.html",
    description="Monthly mutual fund SIP inflows",
    frequency="monthly",
)

# RBI DBIE for gilt yields
RBI_GILT_SOURCE = WebSource(
    name="RBI DBIE Gilt Yields",
    url="https://dbie.rbi.org.in/DBIE/DDF.do",
    description="10-year G-Sec yield, AAA-G-Sec spread",
    frequency="daily",
)


@dataclass
class RegimeBreakpoints:
    """Known Indian market regime breakpoints for validation."""
    events: Dict[str, str] = field(default_factory=lambda: {
        "2008-01-21": "Global Financial Crisis",
        "2013-05-22": "Taper Tantrum",
        "2015-08-24": "China Devaluation",
        "2016-11-08": "Demonetization",
        "2018-09-20": "IL&FS / NBFC Crisis",
        "2020-03-23": "COVID Crash",
        "2022-09-28": "Gilt Rout / INR Weakness",
        "2024-06-04": "Election Results Volatility",
    })


REGIME_BREAKPOINTS = RegimeBreakpoints()

# Regime definitions for the 5-state model
REGIME_LABELS = {
    0: "Risk-On",
    1: "Risk-Off",
    2: "Transitional",
    3: "Late-Cycle",
    4: "Post-Shock",
}

# Data quality thresholds
DATA_QUALITY_CONFIG = {
    "max_missing_pct": 0.05,       # Alert if >5% missing in any column
    "max_gap_days": 5,             # Alert if >5 consecutive business days missing
    "outlier_std_threshold": 5.0,  # Flag observations >5σ from rolling mean
    "min_history_years": 10,       # Minimum years of data required
}

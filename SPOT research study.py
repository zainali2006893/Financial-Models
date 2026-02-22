import time
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

SECTORS = {
    "SaaS": [
        "CRM","NOW","WDAY","HUBS","DDOG","SNOW","NET","MDB","ESTC","CFLT",
        "CRWD","ZS","OKTA","DOCU","MNDY","ASAN","TEAM","GTLB","ZM","RNG",
        "FIVN","TWLO","DBX","BOX",
    ],
    "Consumer Internet": [
        "SPOT","NFLX","DIS","WBD","PARA","CMCSA","RDDT","SNAP","PINS","MTCH",
        "BMBL","GRND","RBLX","DUOL","COUR","UDMY","CHGG","TWOU","YELP","IAC",
        "ANGI","VMEO","ROKU","KIND","EB","FVRR","UPWK","PTON",
    ],
    "Fintech": [
        "SQ","PYPL","HOOD","COIN","SOFI","AFRM","TOST","LSPD","FOUR",
        "DKNG","PENN","FLUT","ZIP",
    ],
    "E-commerce": [
        "SHOP","ETSY","EBAY","MELI","CPNG","SE","TDUP","REAL","BIGC","WIX",
        "GDDY","OPEN","COMP","CSGP","RDFN","Z","BKNG","EXPE","TRIP","ABNB",
        "DASH","UBER","LYFT","AMZN","GOOGL","META","MSFT","AAPL","EA","TTWO",
    ],
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}


def scrape_finviz(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    time.sleep(random.uniform(1.2, 2.0))
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 429:
        time.sleep(20)
        r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return None
    cells = [td.text.strip() for td in BeautifulSoup(r.text, "html.parser").select("table.snapshot-table2 td")]
    return dict(zip(cells[::2], cells[1::2]))


def parse_pct(val):
    try:
        return float(val.replace("%", ""))
    except:
        return float("nan")


def run_regression(df):
    X = np.column_stack([np.ones(len(df)), df["sales_qoq"], df["oper_margin"]])
    y = df["log_ev_sales"].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    df["predicted"] = np.exp(X @ coeffs)
    df["residual"] = df["ev_sales"] - df["predicted"]
    r2 = 1 - ((df["log_ev_sales"] - np.log(df["predicted"])) ** 2).sum() / ((df["log_ev_sales"] - df["log_ev_sales"].mean()) ** 2).sum()
    return df, coeffs, r2


# scrape all tickers
rows = []
for sector, tickers in SECTORS.items():
    print(f"Scraping {sector}...")
    for t in tickers:
        snap = scrape_finviz(t)
        if not snap:
            continue
        ev, growth, margin, profit = (
            snap.get("EV/Sales", "-"),
            snap.get("Sales Q/Q", "-"),
            snap.get("Oper. Margin", "-"),
            snap.get("Profit Margin", "-"),
        )
        if "-" in (ev, growth, margin):
            continue
        rows.append({
            "ticker":        t,
            "sector":        sector,
            "ev_sales":      float(ev),
            "sales_qoq":     parse_pct(growth),
            "oper_margin":   parse_pct(margin),
            "profit_margin": parse_pct(profit),
        })

df_all = pd.DataFrame(rows).dropna(subset=["ev_sales", "sales_qoq", "oper_margin"])
print(f"\nTotal tickers scraped: {len(df_all)}")

# winsorise and log transform per sector
processed = []
for sector, group in df_all.groupby("sector"):
    lower = group["ev_sales"].quantile(0.05)
    upper = group["ev_sales"].quantile(0.95)
    group = group.copy()
    group["ev_sales"] = group["ev_sales"].clip(lower, upper)
    group["log_ev_sales"] = np.log(group["ev_sales"])
    processed.append(group)

df_all = pd.concat(processed).reset_index(drop=True)

# run regression per sector, plot each
all_residuals = []
fig_list = []

for sector, group in df_all.groupby("sector"):
    group = group.copy()
    group, coeffs, r2 = run_regression(group)
    all_residuals.append(group)

    print(f"\n=== {sector} ===")
    print(f"R²: {r2:.3f} | Intercept: {coeffs[0]:.3f} | Sales Q/Q: {coeffs[1]:.3f} | Oper. Margin: {coeffs[2]:.3f}")
    print(f"Tickers: {len(group)}")

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(group["sales_qoq"], group["ev_sales"], alpha=0.6)
    for _, row in group.iterrows():
        ax.annotate(row["ticker"], (row["sales_qoq"], row["ev_sales"]), fontsize=7, va="bottom")

    x_range = np.linspace(group["sales_qoq"].min(), group["sales_qoq"].max(), 100)
    ax.plot(x_range, np.exp(coeffs[0] + coeffs[1] * x_range + coeffs[2] * group["oper_margin"].mean()),
            color="red", linewidth=1.5, label=f"Regression (R²={r2:.2f})")

    ax.set_xlabel("Sales Q/Q (%)")
    ax.set_ylabel("EV/Sales")
    ax.set_title(f"{sector}: EV/Sales vs Revenue Growth")
    ax.legend()
    plt.tight_layout()
    filename = f"spot_comps_{sector.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    print(f"Chart saved: {filename}")

# combined summary table
df_combined = pd.concat(all_residuals).reset_index(drop=True)

print("\n=== TOP 5 RICHEST ACROSS ALL SECTORS ===")
print(df_combined.nlargest(5, "residual")[["ticker", "sector", "ev_sales", "predicted", "residual"]].to_string(index=False))

print("\n=== TOP 5 CHEAPEST ACROSS ALL SECTORS ===")
print(df_combined.nsmallest(5, "residual")[["ticker", "sector", "ev_sales", "predicted", "residual"]].to_string(index=False))

# where does SPOT sit
spot = df_combined[df_combined["ticker"] == "SPOT"]
if not spot.empty:
    print("\n=== SPOT ===")
    print(spot[["ticker", "sector", "ev_sales", "predicted", "residual"]].to_string(index=False))
import time
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

TICKERS = [
    "SPOT","NFLX","GOOGL","META","AAPL","AMZN","MSFT","DIS","WBD","PARA",
    "CMCSA","ROKU","SNAP","PINS","RDDT","MTCH","BMBL","GRND","RBLX","U",
    "EA","TTWO","DKNG","PENN","FLUT","DUOL","COUR","UDMY","CHGG","TWOU",
    "PTON","UBER","LYFT","DASH","ABNB","BKNG","EXPE","TRIP","Z","RDFN",
    "COMP","CSGP","OPEN","ETSY","EBAY","TDUP","REAL","SHOP","BIGC","GDDY",
    "WIX","CPNG","SE","MELI","SQ","PYPL","HOOD","COIN","SOFI","AFRM",
    "TOST","LSPD","FOUR","DBX","BOX","CRM","ZM","RNG","FIVN","TWLO",
    "ADBE","DOCU","MNDY","ASAN","TEAM","GTLB","NOW","WDAY","HUBS","DDOG",
    "SNOW","NET","MDB","ESTC","CFLT","CRWD","ZS","OKTA","YELP","IAC",
    "ANGI","KIND","EB","FVRR","UPWK","ZIP","VMEO",
]

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


rows = []
for t in TICKERS:
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
        "ev_sales":      float(ev),
        "sales_qoq":     parse_pct(growth),
        "oper_margin":   parse_pct(margin),
        "profit_margin": parse_pct(profit),
    })

df = pd.DataFrame(rows).dropna(subset=["ev_sales", "sales_qoq", "oper_margin"])
print(f"Tickers scraped: {len(df)}")

# OLS regression: EV/Sales ~ sales_qoq + oper_margin
X = np.column_stack([np.ones(len(df)), df["sales_qoq"], df["oper_margin"]])
y = df["ev_sales"].values
coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

df["predicted"] = X @ coeffs
df["residual"] = df["ev_sales"] - df["predicted"]
r2 = 1 - ((df["ev_sales"] - df["predicted"]) ** 2).sum() / ((df["ev_sales"] - df["ev_sales"].mean()) ** 2).sum()

print(f"\nR²: {r2:.3f}")
print(f"Intercept: {coeffs[0]:.3f} | Sales Q/Q: {coeffs[1]:.3f} | Oper. Margin: {coeffs[2]:.3f}")
print("\nCheapest vs model:")
print(df.nsmallest(5, "residual")[["ticker", "ev_sales", "predicted", "residual"]].to_string(index=False))
print("\nRichest vs model:")
print(df.nlargest(5, "residual")[["ticker", "ev_sales", "predicted", "residual"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(df["sales_qoq"], df["ev_sales"], alpha=0.6)
for _, row in df.iterrows():
    ax.annotate(row["ticker"], (row["sales_qoq"], row["ev_sales"]), fontsize=7, va="bottom")

x_range = np.linspace(df["sales_qoq"].min(), df["sales_qoq"].max(), 100)
ax.plot(x_range, coeffs[0] + coeffs[1] * x_range + coeffs[2] * df["oper_margin"].mean(),
        color="red", linewidth=1.5, label=f"Regression (R²={r2:.2f})")

ax.set_xlabel("Sales Q/Q (%)")
ax.set_ylabel("EV/Sales")
ax.set_title("Platform Peers: EV/Sales vs Revenue Growth")
ax.legend()
plt.tight_layout()
plt.savefig("spot_comps.png", dpi=150)
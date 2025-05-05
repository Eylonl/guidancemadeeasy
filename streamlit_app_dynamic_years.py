import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import re
import io

# ─── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SEC 8‑K Guidance Extractor", layout="centered")
st.title("📄 SEC 8‑K Guidance Extractor")

# ─── INPUTS ────────────────────────────────────────────────────────────────────
ticker     = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
api_key    = st.text_input("Enter OpenAI API Key", type="password")
year_input = st.text_input("How many years back to search for 8‑K filings? (Leave blank for most recent only)", "")

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
number_token = r'[-+]?\d[\d,\.]*\s*(?:[KMB]|million|billion)?'

def extract_number(token: str):
    if not token or not isinstance(token, str):
        return None
    neg = token.strip().startswith('(') and token.strip().endswith(')')
    tok = token.replace('(', '').replace(')', '').replace('$','') \
               .replace(',', '').strip().lower()
    factor = 1.0
    if tok.endswith('billion'):
        tok, factor = tok[:-7].strip(), 1000
    elif tok.endswith('million'):
        tok, factor = tok[:-7].strip(), 1
    elif tok.endswith('b'):
        tok, factor = tok[:-1].strip(), 1000
    elif tok.endswith('m'):
        tok, factor = tok[:-1].strip(), 1
    elif tok.endswith('k'):
        tok, factor = tok[:-1].strip(), 0.001
    try:
        val = float(tok) * factor
        return -val if neg else val
    except:
        return None

def parse_value_range(text: str):
    """
    Return (low, high, avg) with percentages preserved.
      - Ranges    => strings like "20%", "25%", "22.5%"
      - Singles   => strings like "5%", "-5%" for "(5)%"
      - Otherwise falls back to numeric millions parsing.
    """
    if not isinstance(text, str):
        return None, None, None

    # ------ Percentage handling ------
    if '%' in text:
        # Range A–B% or A to B%
        m = re.search(
            r'\(?\s*([+-]?\d+(?:\.\d+)?)\s*\)?\s*(?:[-–—~]|to)\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*\)?\s*%',
            text
        )
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            avg = (lo + hi) / 2
            return f"{lo}%", f"{hi}%", f"{avg}%"

        # Single percentage, e.g. "5%" or "(5)%"
        m2 = re.search(r'\(?\s*([+-]?\d+(?:\.\d+)?)\s*\)?\s*%', text)
        if m2:
            val = float(m2.group(1))
            # parentheses imply negative
            if text.strip().startswith('('):
                val = -abs(val)
            return f"{val}%", f"{val}%", f"{val}%"

        return None, None, None

    # ------ Flat/unchanged ------
    if re.search(r'\b(flat|unchanged)\b', text, re.I):
        return 0.0, 0.0, 0.0

    # ------ Numeric range A–B or A to B ------
    rng = re.search(
        rf'({number_token})\s*(?:[-–—~]|to)\s*({number_token})',
        text, re.I
    )
    if rng:
        lo = extract_number(rng.group(1))
        hi = extract_number(rng.group(2))
        avg = (lo + hi)/2 if lo is not None and hi is not None else None
        return lo, hi, avg

    # ------ Single numeric value ------
    single = re.search(number_token, text, re.I)
    if single:
        v = extract_number(single.group(0))
        return v, v, v

    return None, None, None

@st.cache_data(show_spinner=False)
def lookup_cik(ticker: str) -> str:
    headers = {'User-Agent': 'Your Name contact@domain.com'}
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = resp.json()
    for ent in data.values():
        if ent["ticker"].upper() == ticker:
            return str(ent["cik_str"]).zfill(10)
    return None

def get_accessions(cik: str, years_back: int):
    headers = {'User-Agent': 'Your Name contact@domain.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    js = requests.get(url, headers=headers).json()["filings"]["recent"]
    cutoff = datetime.today() - timedelta(days=365*years_back)
    return [
        (ad, fd) for f, fd, ad in
        zip(js["form"], js["filingDate"], js["accessionNumber"])
        if f == "8-K" and datetime.strptime(fd, "%Y-%m-%d") >= cutoff
    ]

def get_most_recent_accession(cik: str):
    accs = get_accessions(cik, 10)
    return accs[:1] if accs else []

def get_ex99_1_links(cik: str, accessions):
    headers = {'User-Agent': 'Your Name contact@domain.com'}
    links = []
    for an, fd in accessions:
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{an.replace('-','')}/"
        idx  = base + f"{an}-index.htm"
        r = requests.get(idx, headers=headers)
        if r.status_code != 200:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        # look for any exhibit file starting with ex991*.htm/html
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if re.match(r"(?i)^ex991.*\.(htm|html)$", href):
                full = href if href.lower().startswith("http") else base + href
                links.append((fd, full))
                break
    return links

def extract_guidance(text: str, ticker: str, client: OpenAI):
    prompt = (
        f"You are a financial analyst assistant. Extract all forward-looking guidance "
        f"given in this earnings release for {ticker}.\n\n"
        f"Return a Markdown table with columns Metric|Value|Period.\n\n"
        f"{text}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4", messages=[{"role":"user","content":prompt}], temperature=0
        )
        return resp.choices[0].message.content
    except:
        return None

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if st.button("🔍 Extract Guidance"):
    if not ticker:
        st.error("Please enter a ticker."); st.stop()
    cik = lookup_cik(ticker)
    if not cik:
        st.error("CIK not found."); st.stop()

    # choose filings
    if year_input.strip().isdigit():
        yrs = int(year_input.strip())
        accs = get_accessions(cik, yrs)
    else:
        accs = get_most_recent_accession(cik)

    if not accs:
        st.warning("No 8-K filings found."); st.stop()

    links = get_ex99_1_links(cik, accs)
    if not links:
        st.warning("No Ex‑99.1 links found."); st.stop()

    client = OpenAI(api_key=api_key) if api_key else None
    results = []

    for fd, url in links:
        st.write(f"📄 Processing {url}")
        html = requests.get(url, headers={'User-Agent':'MyCo'}).text
        text = BeautifulSoup(html, "html.parser").get_text()
        cut = text.lower().find("forward looking statements")
        if cut != -1:
            text = text[:cut]

        # GPT extraction
        df = None
        if client:
            md = extract_guidance(text, ticker, client)
            if md and "|" in md:
                lines = [l for l in md.splitlines() if l.strip().startswith("|")]
                if len(lines) >= 2:
                    hdr = [h.strip() for h in lines[0].split("|")[1:-1]]
                    rows = [
                        [p.strip() for p in ln.split("|")[1:-1]]
                        for ln in lines[2:]
                        if len(ln.split("|")[1:-1]) == len(hdr)
                    ]
                    if rows:
                        df = pd.DataFrame(rows, columns=hdr)

        # finalize
        if df is None:
            st.warning("⚠️ No guidance found; skipping.")
            continue

        # add Low/High/Average
        if "Value" in df.columns:
            df[["Low","High","Average"]] = df["Value"].apply(lambda v: pd.Series(parse_value_range(v)))

        # normalize periods
        if "Period" in df.columns:
            df["Period"] = df["Period"].apply(lambda p: normalise_period(p) or p)

        df["FilingDate"] = fd
        df["8K_Link"]    = url
        results.append(df)
        st.success("✅ Guidance extracted.")

    # combine & download
    if results:
        combined = pd.concat(results, ignore_index=True)
        buf = io.BytesIO()
        combined.to_excel(buf, index=False)
        st.download_button(
            label="📥 Download Excel",
            data=buf.getvalue(),
            file_name=f"{ticker}_guidance.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No guidance extracted.")

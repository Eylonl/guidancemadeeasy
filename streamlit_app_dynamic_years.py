
import re
import io
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI

# ─── PAGE SETUP ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SEC 8‑K Guidance Extractor", layout="centered")
st.title("📄 SEC 8‑K Guidance Extractor")

# ─── INPUTS ──────────────────────────────────────────────────────────────────────
ticker     = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
api_key    = st.text_input("Enter OpenAI API Key (for GPT fallback)", type="password")
year_input = st.text_input("Years back to search for 8‑K filings (blank = most recent only)", "")

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────────
number_token = r'[-+]?\d[\d,\.]*\s*(?:[KMB]|million|billion)?'

def extract_number(token: str) -> Optional[float]:
    if not token or not isinstance(token, str):
        return None
    neg = token.strip().startswith('(') and token.strip().endswith(')')
    tok = token.replace('(', '').replace(')', '').replace('$', '') \
               .replace(',', '').strip().lower()
    factor = 1.0
    if tok.endswith('billion'):
        tok, factor = tok[:-7].strip(), 1_000
    elif tok.endswith('million'):
        tok, factor = tok[:-7].strip(), 1
    elif tok.endswith('b'):
        tok, factor = tok[:-1].strip(), 1_000
    elif tok.endswith('m'):
        tok, factor = tok[:-1].strip(), 1
    elif tok.endswith('k'):
        tok, factor = tok[:-1].strip(), 0.001
    try:
        val = float(tok) * factor
        return -val if neg else val
    except ValueError:
        return None

def parse_value_range(text: str) -> Tuple[Optional[float],Optional[float],Optional[float]]:
    if not text or not isinstance(text, str):
        return None, None, None
    if re.search(r'\b(flat|unchanged)\b', text, re.I):
        return 0.0, 0.0, 0.0
    rng = re.search(rf'({number_token})\s*(?:[-–—~]|to)\s*({number_token})', text, re.I)
    if rng:
        lo  = extract_number(rng.group(1))
        hi  = extract_number(rng.group(2))
        avg = (lo + hi)/2 if lo is not None and hi is not None else None
        return lo, hi, avg
    single = re.search(number_token, text, re.I)
    if single:
        v = extract_number(single.group(0))
        return v, v, v
    return None, None, None

_ORDINAL = {"first":"Q1","second":"Q2","third":"Q3","fourth":"Q4"}

def normalise_period(txt: str) -> Optional[str]:
    if not txt or not isinstance(txt, str):
        return None
    t = txt.strip().lower()
    m = re.search(r'(q[1-4])\s*fy\s*(\d{2,4})', t)
    if m:
        return f"{m.group(1).upper()} FY{m.group(2)[-2:]}"
    m = re.search(r'(first|second|third|fourth)\s+quarter\s+fiscal\s+year\s+(\d{4})', t)
    if m:
        return f"{_ORDINAL[m.group(1)]} FY{m.group(2)[-2:]}"
    m = re.search(r'(?:fiscal|full)\s+year\s+(\d{4})', t)
    if m:
        return f"FY{m.group(1)[-2:]}"
    return txt.strip()

# ─── EDGAR FETCHING ───────────────────────────────────────────────────────────────
@st.cache_data
def lookup_cik(ticker: str) -> Optional[str]:
    hdr = {'User-Agent': 'Your Name contact@domain.com'}
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=hdr)
    d = r.json()
    for ent in d.values():
        if ent["ticker"].upper() == ticker:
            return str(ent["cik_str"]).zfill(10)
    return None

def get_accessions(cik: str, years_back: int):
    hdr = {'User-Agent': 'Your Name contact@domain.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=hdr).json()["filings"]["recent"]
    cutoff = datetime.today() - timedelta(days=365*years_back)
    acc = []
    for form, dt, an in zip(data["form"], data["filingDate"], data["accessionNumber"]):
        if form=="8-K" and datetime.strptime(dt,"%Y-%m-%d")>=cutoff:
            acc.append((an, dt))
    return acc

def get_most_recent_accession(cik: str):
    return get_accessions(cik, 10)[:1]

def get_ex99_links(cik: str, accessions):
    hdr = {'User-Agent': 'Your Name contact@domain.com'}
    links = []
    for an, dt in accessions:
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{an.replace('-','')}/"
        idx  = base + f"{an}-index.htm"
        r = requests.get(idx, headers=hdr)
        if r.status_code!=200:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for tr in soup.find_all("tr"):
            if "99.1" in tr.get_text().lower():
                tds = tr.find_all("td")
                if len(tds)>=3:
                    fn = tds[2].text.strip()
                    links.append((dt, base+fn))
                    break
    return links

# ─── GPT EXTRACTION ───────────────────────────────────────────────────────────────
def extract_via_gpt(text: str, ticker: str, client: OpenAI) -> Optional[str]:
    prompt = (
        f"You are a financial analyst assistant. Extract forward-looking guidance "
        f"for {ticker} in a Markdown table Metric|Value|Period.\n\n{text}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4", messages=[{"role":"user","content":prompt}], temperature=0
        )
        return resp.choices[0].message.content
    except Exception:
        return None

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if st.button("🔍 Extract Guidance"):
    if not ticker:
        st.error("Please enter a ticker."); st.stop()
    cik = lookup_cik(ticker)
    if not cik:
        st.error("CIK not found."); st.stop()
    # choose filings
    if year_input.strip():
        try: yrs = int(year_input.strip()); accs = get_accessions(cik, yrs)
        except: st.error("Invalid years‑back."); st.stop()
    else:
        accs = get_most_recent_accession(cik)
    links = get_ex99_links(cik, accs)
    if not links:
        st.warning("No Ex‑99.1 links found."); st.stop()
    client = OpenAI(api_key=api_key) if api_key else None
    results = []
    for dt, url in links:
        st.write(f"📄 Processing {url}")
        try:
            html = requests.get(url, headers={'User-Agent':'MyCo'}).text
            text = BeautifulSoup(html, "html.parser").get_text()
            idx = text.lower().find("forward looking statements")
            if idx>-1: text = text[:idx]
            table_md = extract_via_gpt(text, ticker, client) if client else None
            if not table_md:
                st.warning("⚠️ GPT skip or no table."); continue
            # robust markdown parsing
            md_lines = [l for l in table_md.splitlines() if l.strip().startswith("|")]
            if len(md_lines) < 2:
                st.warning("⚠️ No table detected; skipping."); continue
            header = [h.strip() for h in md_lines[0].strip().split("|")[1:-1]]
            data_lines = md_lines[2:]
            data = []
            for line in data_lines:
                parts = line.strip().split("|")[1:-1]
                if len(parts) != len(header):
                    st.warning(f"⚠️ Skipping malformed row: {line}")
                    continue
                data.append([p.strip() for p in parts])
            if not data:
                st.warning("⚠️ No data rows; skipping."); continue
            df = pd.DataFrame(data, columns=header)
            if "Value" in df.columns:
                df[["Low","High","Avg"]] = df["Value"].apply(lambda v: pd.Series(parse_value_range(v)))
            if "Period" in df.columns:
                df["Period"] = df["Period"].apply(lambda p: normalise_period(p) or p)
            df["FilingDate"] = dt
            df["8K_Link"] = url
            results.append(df)
            st.success("✅ Guidance extracted.")
        except Exception as e:
            st.warning(f"Failed {url}: {e}")
    if results:
        combined = pd.concat(results, ignore_index=True)
        buf = io.BytesIO()
        combined.to_excel(buf, index=False)
        st.download_button("📥 Download Excel", buf.getvalue(),
                           file_name=f"{ticker}_guidance.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("No guidance extracted.")

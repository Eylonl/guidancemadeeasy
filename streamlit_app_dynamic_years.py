
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import re
import io

# ─── PAGE SETUP ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SEC 8‑K Guidance Extractor", layout="centered")
st.title("📄 SEC 8‑K Guidance Extractor")

# ─── INPUTS ──────────────────────────────────────────────────────────────────────
ticker = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
api_key = st.text_input("Enter OpenAI API Key (for GPT fallback)", type="password")
years_back = st.text_input("Years back to search for 8‑K filings (blank = most recent only)", "")

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────────
number_token = r'[-+]?\d[\d,\.]*\s*(?:[KMB]|million|billion)?'

def extract_number(token: str):
    if not token:
        return None
    neg = token.strip().startswith('(') and token.strip().endswith(')')
    tok = token.replace('(', '').replace(')', '').replace('$','') \
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
    except:
        return None

def parse_value_range(text: str):
    if not isinstance(text, str):
        return None, None, None
    if re.search(r'\b(flat|unchanged)\b', text, re.I):
        return 0.0, 0.0, 0.0
    rng = re.search(rf'({number_token})\s*(?:[-–—~]|to)\s*({number_token})', text, re.I)
    if rng:
        lo, hi = extract_number(rng.group(1)), extract_number(rng.group(2))
        avg = (lo+hi)/2 if lo is not None and hi is not None else None
        return lo, hi, avg
    single = re.search(number_token, text, re.I)
    if single:
        v = extract_number(single.group(0))
        return v, v, v
    return None, None, None

_ORD = {"first":"Q1","second":"Q2","third":"Q3","fourth":"Q4"}

def normalise_period(txt: str):
    if not isinstance(txt, str):
        return None
    t = txt.strip().lower()
    m = re.search(r'(q[1-4])\s*fy\s*(\d{2,4})', t)
    if m:
        return f"{m.group(1).upper()} FY{m.group(2)[-2:]}"
    m = re.search(r'(first|second|third|fourth)\s+quarter\s+fiscal\s+year\s+(\d{4})', t)
    if m:
        return f"{_ORD[m.group(1)]} FY{m.group(2)[-2:]}"
    m = re.search(r'(?:fiscal|full)\s+year\s+(\d{4})', t)
    if m:
        return f"FY{m.group(1)[-2:]}"
    return txt.strip()

def local_extract_guidance(text: str):
    lines = text.splitlines()
    guidance = []
    period = None
    in_targets = False
    for ln in lines:
        low = ln.lower().strip()
        if 'financial targets' in low:
            in_targets = True
            continue
        if in_targets:
            hdr = ln.strip().rstrip(':')
            # detect period line
            if re.search(r'(quarter|year)\s+fiscal\s+year', hdr, re.I):
                period = normalise_period(hdr)
                continue
            if ln.strip().startswith('•') and period:
                bullet = ln.strip().lstrip('•').strip().rstrip('.')
                m = re.match(r'(.+?)(?:is expected to|are expected to)\s*(.+)', bullet, re.I)
                if m:
                    metric = m.group(1).strip()
                    value = m.group(2).strip()
                    guidance.append((metric, value, period))
                continue
            if period and not ln.strip().startswith('•') and ln.strip()=='':
                break
    if guidance:
        df = pd.DataFrame(guidance, columns=['Metric','Value','Period'])
        return df
    return None

# ─── EDGAR FETCHING ───────────────────────────────────────────────────────────────
@st.cache_data
def lookup_cik(tick):
    hdr = {'User-Agent': 'Your Name contact@domain.com'}
    data = requests.get("https://www.sec.gov/files/company_tickers.json", headers=hdr).json()
    for ent in data.values():
        if ent['ticker'].upper()==tick:
            return str(ent['cik_str']).zfill(10)
    return None

def get_accessions(cik, yrs):
    hdr = {'User-Agent':'Your Name'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    js = requests.get(url, headers=hdr).json()['filings']['recent']
    cutoff = datetime.today() - timedelta(days=365*yrs)
    acc = []
    for f, d, an in zip(js['form'], js['filingDate'], js['accessionNumber']):
        if f=='8-K' and datetime.strptime(d,'%Y-%m-%d')>=cutoff:
            acc.append((an, d))
    return acc

def get_recent(cik):
    lst = get_accessions(cik,10)
    return lst[:1]

def get_ex99_links(cik, accs):
    hdr = {'User-Agent':'Your Name'}
    links=[]
    for an, d in accs:
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{an.replace('-','')}/"
        idx = base+f"{an}-index.htm"
        r = requests.get(idx, headers=hdr)
        if r.status_code!=200: continue
        soup=BeautifulSoup(r.text,'html.parser')
        for tr in soup.find_all('tr'):
            if '99.1' in tr.get_text().lower():
                tds=tr.find_all('td')
                if len(tds)>=3:
                    links.append((d, base+tds[2].text.strip()))
                break
    return links

# ─── GPT EXTRACTION ───────────────────────────────────────────────────────────────
def extract_guidance(text, tick, client):
    prompt = (
        f"You are a financial analyst assistant. Extract forward-looking guidance for {tick} "
        f"in a Markdown table with columns Metric|Value|Period.\n\n{text}"
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
        st.error("Enter a ticker."); st.stop()
    cik = lookup_cik(ticker)
    if not cik:
        st.error("CIK not found."); st.stop()
    try:
        yrs = int(years_back) if years_back.strip() else None
    except:
        st.error("Invalid years"); st.stop()
    accs = get_accessions(cik, yrs) if yrs else get_recent(cik)
    links = get_ex99_links(cik, accs)
    if not links:
        st.warning("No Ex-99.1 links."); st.stop()
    client = OpenAI(api_key=api_key) if api_key else None
    results=[]
    for date_str, url in links:
        st.write(f"📄 Processing {url}")
        try:
            html=requests.get(url, headers={'User-Agent':'MyCo'}).text
            text=BeautifulSoup(html,'html.parser').get_text()
            idx=text.lower().find('forward looking statements')
            if idx>=0: text=text[:idx]
            df=None
            # GPT path
            if client:
                md=extract_guidance(text, ticker, client)
                if md and '|' in md:
                    lines=[l for l in md.splitlines() if l.strip().startswith('|')]
                    if len(lines)>=2:
                        hdr=[h.strip() for h in lines[0].split('|')[1:-1]]
                        rows=[]
                        for ln in lines[2:]:
                            parts=ln.split('|')[1:-1]
                            if len(parts)==len(hdr):
                                rows.append([p.strip() for p in parts])
                        if rows:
                            df=pd.DataFrame(rows, columns=hdr)
            # fallback local parser
            if df is None:
                df = local_extract_guidance(text)
                if df is not None:
                    st.info("ℹ️ Used local guidance parser.")
            if df is None:
                st.warning("⚠️ No guidance found; skipping.")
                continue
            # post-process
            if 'Value' in df.columns:
                df[['Low','High','Avg']] = df['Value'].apply(lambda v: pd.Series(parse_value_range(v)))
            if 'Period' in df.columns:
                df['Period'] = df['Period'].apply(lambda p: normalise_period(p) or p)
            df['FilingDate']=date_str
            df['8K_Link']=url
            results.append(df)
            st.success("✅ Guidance extracted.")
        except Exception as e:
            st.warning(f"Failed {url}: {e}")
    if results:
        combined=pd.concat(results, ignore_index=True)
        buf=io.BytesIO()
        combined.to_excel(buf, index=False)
        st.download_button("📥 Download Excel", buf.getvalue(),
                           file_name=f"{ticker}_guidance.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("No guidance extracted.")

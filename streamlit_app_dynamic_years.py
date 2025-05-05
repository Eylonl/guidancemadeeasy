
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import re
import io

# PAGE SETUP
st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("📄 SEC 8-K Guidance Extractor")

# INPUTS
ticker = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")
year_input = st.text_input("How many years back to search for 8-K filings? (Leave blank for most recent only)", "")

# HELPER FUNCTIONS
number_token = r'[-+]?\d[\d,\.]*\s*(?:[KMB]|million|billion)?'

def extract_number(token: str):
    if not token or not isinstance(token, str):
        return None
    neg = token.strip().startswith('(') and token.strip().endswith(')')
    tok = token.replace('(', '').replace(')', '').replace('$', '').replace(',', '').strip().lower()
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
    if not isinstance(text, str):
        return None, None, None
    # percentage handling
    if '%' in text:
        neg_flag = '(' in text
        # range A–B%
        m = re.search(r'\(?([-+]?\d+(?:\.\d+)?)[\)]?\s*(?:[-–—~]|to)\s*\(?([-+]?\d+(?:\.\d+)?)[\)]?\s*%', text)
        if m:
            low = float(m.group(1))
            high = float(m.group(2))
            avg = (low + high) / 2
            if neg_flag:
                low, high, avg = -low, -high, -avg
            return f"{low}%", f"{high}%", f"{avg}%"
        # single value
        m2 = re.search(r'\(?([-+]?\d+(?:\.\d+)?)[\)]?\s*%', text)
        if m2:
            val = float(m2.group(1))
            if '(' in m2.group(0):
                val = -val
            return f"{val}%", f"{val}%", f"{val}%"
        return None, None, None

    # flat or unchanged
    if re.search(r'\b(flat|unchanged)\b', text, re.I):
        return 0.0, 0.0, 0.0

    # range A–B or A to B
    rng = re.search(rf'({number_token})\s*(?:[-–—~]|to)\s*({number_token})', text, re.I)
    if rng:
        lo = extract_number(rng.group(1))
        hi = extract_number(rng.group(2))
        avg = (lo + hi)/2 if lo is not None and hi is not None else None
        return lo, hi, avg

    # single value
    single = re.search(number_token, text, re.I)
    if single:
        v = extract_number(single.group(0))
        return v, v, v

    return None, None, None

# (Remaining file unchanged... definitions for lookup_cik, get_accessions, get_ex99_1_links, extract_guidance, local_extract_guidance, main logic follow here)
# After extracting df:
# df[["Low","High","Average"]] = df["Value"].apply(lambda v: pd.Series(parse_value_range(v)))
# ...


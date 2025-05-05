import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os
import re

# ─── NUMBER & RANGE PARSING HELPERS ───────────────────────────────────────────
number_token = r'[-+]?\d[\d,\.]*\s*(?:[KMB]|million|billion)?'

def extract_number(token: str):
    if not token or not isinstance(token, str):
        return None
    neg = token.strip().startswith('(') and token.strip().endswith(')' )
    tok = token.replace('(', '').replace(')', '').replace('$','') \
               .replace(',', '').strip().lower()
    factor = 1.0
    if tok.endswith('billion'): tok, factor = tok[:-7].strip(), 1000
    elif tok.endswith('million'): tok, factor = tok[:-7].strip(), 1
    elif tok.endswith('b'): tok, factor = tok[:-1].strip(), 1000
    elif tok.endswith('m'): tok, factor = tok[:-1].strip(), 1
    elif tok.endswith('k'): tok, factor = tok[:-1].strip(), 0.001
    try:
        val = float(tok) * factor
        return -val if neg else val
    except:
        return None


def format_percent(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return f"{val:.1f}%"
    return val

def parse_value_range(text: str):
    if not isinstance(text, str):
        return (None, None, None)
    if re.search(r'\b(flat|unchanged)\b', text, re.I):
        return (0.0, 0.0, 0.0)
    rng = re.search(fr'({number_token})\s*(?:[-–—~]|to)\s*({number_token})', text, re.I)
    if rng:
        lo = extract_number(rng.group(1))
        hi = extract_number(rng.group(2))
        avg = (lo+hi)/2 if lo is not None and hi is not None else None
        return (lo, hi, avg)
    single = re.search(number_token, text, re.I)
    if single:
        v = extract_number(single.group(0))
        return (v, v, v)
    return (None, None, None)


st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("📄 SEC 8-K Guidance Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")

# Both filter options displayed at the same time
year_input = st.text_input("How many years back to search for 8-K filings? (Leave blank for most recent only)", "")
quarter_input = st.text_input("OR enter specific quarter (e.g., 2Q25, Q4FY24)", "")


@st.cache_data(show_spinner=False)
def lookup_cik(ticker):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = res.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)


def get_accessions(cik, years_back=None, specific_quarter=None):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    filings = data["filings"]["recent"]
    accessions = []
    
    if years_back:
        cutoff = datetime.today() - timedelta(days=365 * years_back)
        
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if date >= cutoff:
                    accessions.append((accession, date_str))
    
    elif specific_quarter:
        # Parse quarter and year from input format like "3Q25" or "Q4FY24"
        match = re.search(r'(?:Q?(\d)Q?|Q(\d))(?:FY)?(\d{2}|\d{4})', specific_quarter.upper())
        if match:
            quarter = match.group(1) or match.group(2)
            year = match.group(3)
            
            # Convert 2-digit year to 4-digit year
            if len(year) == 2:
                year = '20' + year
                
            quarter_num = int(quarter)
            year_num = int(year)
            
            # Determine date ranges for the specified quarter
            # Fiscal quarters can vary by company, but we'll use calendar quarters as default
            # Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
            quarter_start_months = {1: 1, 2: 4, 3: 7, 4: 10}
            quarter_end_months = {1: 3, 2: 6, 3: 9, 4: 12}
            
            start_date = datetime(year_num, quarter_start_months[quarter_num], 1)
            end_month = quarter_end_months[quarter_num]
            if end_month == 12:
                end_date = datetime(year_num, end_month, 31)
            elif end_month in [4, 6, 9, 11]:
                end_date = datetime(year_num, end_month, 30)
            else:  # February
                if (year_num % 4 == 0 and year_num % 100 != 0) or (year_num % 400 == 0):
                    end_date = datetime(year_num, end_month, 29)  # Leap year
                else:
                    end_date = datetime(year_num, end_month, 28)
            
            # Add a buffer period after quarter end for earnings releases (typically 1-2 months)
            end_date = end_date + timedelta(days=60)
            
            for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
                if form == "8-K":
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    if start_date <= date <= end_date:
                        accessions.append((accession, date_str))
    
    else:  # Default: most recent only
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                accessions.append((accession, date_str))
                break
    
    return accessions

def get_most_recent_accession(cik):
    all_recent = get_accessions(cik)
    return all_recent[:1] if all_recent else []

def get_ex99_1_links(cik, accessions):
    links = []
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    for accession, date_str in accessions:
        base_folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
        index_url = base_folder + f"{accession}-index.htm"
        res = requests.get(index_url, headers=headers)
        if res.status_code != 200:
            continue
        soup = BeautifulSoup(res.text, "html.parser")
        for row in soup.find_all("tr"):
            if "99.1" in row.get_text().lower():
                tds = row.find_all("td")
                if len(tds) >= 3:
                    filename = tds[2].text.strip()
                    links.append((date_str, accession, base_folder + filename))
                    break
    return links

def extract_guidance(text, ticker, client):
    prompt = f"""You are a financial analyst assistant. Extract all forward-looking guidance given in this earnings release for {ticker}. 

Return a structured list containing:
- metric (e.g. Revenue, EPS, Operating Margin)
- value or range (e.g. $1.5B–$1.6B or $2.05)
- applicable period (e.g. Q3 FY24, Full Year 2025)

VERY IMPORTANT: For any percentage values, always include the % symbol in your output:
- If the guidance mentions "operating margin of 5 to 7 percent", output it as "5% to 7%" or "5%-7%"
- If the guidance mentions a negative percentage like "(5%)" or "decrease of 5%", output it as "-5%"
- Preserve any descriptive text like "Approximately" or "Around" in your output

Respond in table format without commentary.\n\n{text}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
                st.warning("⚠️ Skipped, no guidance found in filing.")


def split_gaap_non_gaap(df):
    if 'Value' not in df.columns or 'Metric' not in df.columns:
        return df  # Avoid crash if column names are missing

    rows = []
    for _, row in df.iterrows():
        val = str(row['Value'])
        match = re.search(r'(\d[\d\.\s%to–-]*)\s*on a GAAP basis.*?(\d[\d\.\s%to–-]*)\s*on a non-GAAP basis', val, re.I)
        if match:
            gaap_val = match.group(1).strip() + " GAAP"
            non_gaap_val = match.group(2).strip() + " non-GAAP"
            for new_val, label in [(gaap_val, "GAAP"), (non_gaap_val, "Non-GAAP")]:
                new_row = row.copy()
                new_row["Value"] = new_val
                new_row["Metric"] = f"{row['Metric']} ({label})"
                lo, hi, avg = parse_value_range(new_val)
                new_row["Low"], new_row["High"], new_row["Average"] = format_percent(lo), format_percent(hi), format_percent(avg)
                rows.append(new_row)
        else:
            rows.append(row)
    return pd.DataFrame(rows)


if st.button("🔍 Extract Guidance"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        cik = lookup_cik(ticker)
        if not cik:
            st.error("CIK not found for ticker.")
        else:
            client = OpenAI(api_key=api_key)
            
            # Handle different filtering options
            if quarter_input.strip():
                # Quarter input takes precedence if both are filled
                accessions = get_accessions(cik, specific_quarter=quarter_input.strip())
                if not accessions:
                    st.warning(f"No 8-K filings found for {quarter_input}. Please check the format (e.g., 2Q25, Q4FY24).")
            elif year_input.strip():
                try:
                    years_back = int(year_input.strip())
                    accessions = get_accessions(cik, years_back=years_back)
                except:
                    st.error("Invalid year input. Must be a number.")
                    accessions = []
            else:
                # Default to most recent if neither input is provided
                accessions = get_most_recent_accession(cik)

            links = get_ex99_1_links(cik, accessions)
            results = []

            for date_str, acc, url in links:
                st.write(f"📄 Processing {url}")
                try:
                    html = requests.get(url, headers={"User-Agent": "MyCompanyName Data Research Contact@mycompany.com"}).text
                    text = BeautifulSoup(html, "html.parser").get_text()
                    forw_idx = text.lower().find("forward looking statements")
                    if forw_idx != -1:
                        text = text[:forw_idx]
                    table = extract_guidance(text, ticker, client)
                    if table and "|" in table:
                        rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                        df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                        
                        # Store which rows have percentages in the Value column
                        percentage_rows = []
                        for idx, row in df.iterrows():
                            if '%' in str(row[df.columns[1]]):
                                percentage_rows.append(idx)
                        
                        # Parse low, high, and average from Value column
                        value_col = df.columns[1]
                        df[['Low','High','Average']] = df[value_col].apply(lambda v: pd.Series(parse_value_range(v)))
                        
                        # Apply GAAP/non-GAAP split
                        df = split_gaap_non_gaap(df)
                        
                        # For rows that originally had % in the Value column, make sure Low, High, Average have % too
                        for idx in df.index:
                            # Check if the original row had a percentage
                            if idx in percentage_rows:
                                # Add % to Low, High, Average columns
                                for col in ['Low', 'High', 'Average']:
                                    if pd.notnull(df.loc[idx, col]) and isinstance(df.loc[idx, col], (int, float)):
                                        df.loc[idx, col] = f"{df.loc[idx, col]:.1f}%"
                        
                        df["FilingDate"] = date_str
                        df["8K_Link"] = url
                        results.append(df)
                        st.success("✅ Guidance extracted from this 8-K.")
                    else:
                        st.warning("⚠️ Skipped, no guidance found in filing.")
                except Exception as e:
                    st.warning(f"Could not process: {url}. Error: {str(e)}")

            if results:
                combined = pd.concat(results, ignore_index=True)
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No guidance data extracted.")

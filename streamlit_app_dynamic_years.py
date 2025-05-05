import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os
import re
import traceback

st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("📄 SEC 8-K Guidance Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")
year_input = st.text_input("How many years back to search for 8-K filings? (Leave blank for most recent only)", "")


@st.cache_data(show_spinner=False)
def lookup_cik(ticker):
    headers = {"User-Agent": "Your Name Contact@domain.com"}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = res.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_accessions(cik, years_back):
    headers = {"User-Agent": "Your Name Contact@domain.com"}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    filings = data["filings"]["recent"]
    accessions = []
    cutoff = datetime.today() - timedelta(days=365 * years_back)

    for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
        if form == "8-K":
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date >= cutoff:
                accessions.append((accession, date_str))
    return accessions


def get_most_recent_accession(cik):
    all_recent = get_accessions(cik, 10)
    return all_recent[:1] if all_recent else []


def get_ex99_1_links(cik, accessions):
    links = []
    headers = {"User-Agent": "Your Name Contact@domain.com"}
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
    prompt = f"""You are a financial analyst assistant. Extract ALL forward-looking guidance given in this earnings release for {ticker}. 

IMPORTANT INSTRUCTIONS:
1. When a metric has both GAAP and non-GAAP values, split them into separate rows in your table.
   For example, if you see: "Gross margin of 81.5% to 82.0% on a GAAP basis and in the range of 84.0% to 84.5% on a non-GAAP basis"
   Create TWO separate rows:
   - Metric: Gross margin (GAAP), Value: 81.5% to 82.0%, Period: [applicable period]
   - Metric: Gross margin (non-GAAP), Value: 84.0% to 84.5%, Period: [applicable period]

2. Extract BOTH quarterly and full-year guidance:
   - Look for phrases like "for Q4'24" and "in FY24" or "full year FY24"
   - Clearly identify the period in your table (Q4 FY24, Full Year FY24, etc.)

3. Look for ALL types of forward-looking statements, including:
   - Expected margins (operating margin, gross margin)
   - Revenue guidance (total, cloud, data center)
   - Growth percentages and rates
   - Stock dilution estimates
   - Free cash flow projections
   - Any targets for future fiscal years/quarters

4. Pay special attention to statements that mention:
   - "We expect..."
   - "We continue to expect..."
   - "Our guidance assumes..."
   - "This guidance implies..."  
   - "...to be approximately X%"
   - Long-term targets (e.g., "by FY27")
   - All future periods (quarters or years)

5. Look for full-year guidance implied by quarterly guidance:
   - For example: "This guidance implies full year FY24 revenue growth of approximately 23%"
   - Create a separate row for this implied full-year guidance

Return a structured table with these columns:
- metric (e.g. Revenue, EPS, Operating Margin, Total Revenue Growth)
- value or range (e.g. $1.5B–$1.6B or $2.05, 30%, etc.)
- applicable period (e.g. Q1 FY25, Full Year 2025, FY27)

Respond in table format without commentary. If no forward-looking guidance is found, respond with "No guidance found."
\n\n{text}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning("⚠️ Skipped, no guidance found in filing.")
        return None


def extract_number(text):
    # This helper function extracts numeric values from text
    if not text:
        return None
        
    # Handle parentheses as negative numbers - e.g., (5.0) means -5.0
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1]  # Remove parentheses
    
    # Clean the text
    clean_text = text.replace("$", "").replace(",", "")
    
    # Extract the number
    try:
        # Check if there's an explicit negative sign
        if clean_text.startswith("-"):
            is_negative = True
            clean_text = clean_text[1:]  # Remove the negative sign for processing
            
        # Handle units
        if "B" in clean_text.upper():
            value = float(clean_text.upper().replace("B", "")) * 1000000000
        elif "M" in clean_text.upper():
            value = float(clean_text.upper().replace("M", "")) * 1000000
        else:
            value = float(clean_text)
            
        # Apply negative sign if needed
        return -value if is_negative else value
    except ValueError:
        return None


def parse_value_range(text):
    # Return tuple of (low, high, average)
    if not text or not isinstance(text, str):
        return None, None, None
    
    text = text.strip()
    
    # Case 1: "flat" or "unchanged" guidance
    if "flat" in text.lower() or "unchanged" in text.lower():
        return 0, 0, 0
    
    # Case 2: Percentage values (including approximate)
    # Handle both positive and negative percentages
    percent_match = re.search(r'(?:approximately|about|around|roughly|~|circa)?\s*(\(?([-+]?\d+\.?\d*)\)?%)', text, re.IGNORECASE)
    if percent_match:
        # Get the percentage value, ensuring proper handling of negative values
        percent_str = percent_match.group(2)
        
        # If wrapped in parentheses without explicit negative sign, it's negative
        if "(" in percent_match.group(1) and ")" in percent_match.group(1) and not percent_str.startswith("-"):
            percent_value = -float(percent_str)
        else:
            percent_value = float(percent_str)
            
        return percent_value, percent_value, percent_value
    
    # Case 3: Range values like "$1.5B-$1.6B"
    range_match = re.search(r'[$]?([-+]?[\d\.]+[KMB]?)(?:[ ]*[-–—~][ ]*|\s+to\s+)[$]?([-+]?[\d\.]+[KMB]?)', text, re.IGNORECASE)
    if range_match:
        low = extract_number(range_match.group(1))
        high = extract_number(range_match.group(2))
        if low is not None and high is not None:
            return low, high, (low + high) / 2
    
    # Case 4: Single values like "$1.5B"
    single_match = re.search(r'[$]?([-+]?[\d\.]+[KMB]?)(?:\s|$)', text, re.IGNORECASE)
    if single_match:
        value = extract_number(single_match.group(1))
        if value is not None:
            return value, value, value
    
    # No numeric value found, return original text as average
    return None, None, text


if st.button("🔍 Extract Guidance"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        cik = lookup_cik(ticker)
        if not cik:
            st.error("CIK not found for ticker.")
        else:
            client = OpenAI(api_key=api_key)
            if year_input.strip():
                try:
                    years_back = int(year_input)
                    accessions = get_accessions(cik, years_back)
                except:
                    st.error("Invalid year input. Must be a number.")
                    accessions = []
            else:
                accessions = get_most_recent_accession(cik)

            links = get_ex99_1_links(cik, accessions)
            results = []

            for date_str, acc, url in links:
                st.write(f"📄 Processing {url}")
                try:
                    html = requests.get(url, headers={"User-Agent": "MyCompanyName Data Research Contact@mycompany.com"}).text
                    text = BeautifulSoup(html, "html.parser").get_text()
                    # forw_idx trimming removed – guidance kept
#
                    
                    # Get guidance table from OpenAI
                    table = extract_guidance(text, ticker, client)
                    if table and "|" in table:
                        # Process table
                        rows = []
                        for r in table.strip().split("\n"):
                            if "|" in r:
                                # Skip separator rows that consist of dashes
                                if re.match(r'^[\|\s\-]+$', r):
                                    continue
                                # Extract cells from table format
                                cells = r.strip().split("|")
                                # Remove empty cells at start/end if present
                                if cells and not cells[0].strip():
                                    cells = cells[1:]
                                if cells and not cells[-1].strip():
                                    cells = cells[:-1]
                                rows.append([cell.strip() for cell in cells])
                        
                        # Create DataFrame if we have header and data
                        if len(rows) >= 2:
                            try:
                                # Check for column count mismatch
                                num_header_cols = len(rows[0])
                                for i, row in enumerate(rows[1:], 1):
                                    if len(row) != num_header_cols:
                                        # Fix row length to match header
                                        if len(row) < num_header_cols:
                                            # If row is shorter, pad with empty strings
                                            rows[i] = row + [''] * (num_header_cols - len(row))
                                        else:
                                            # If row is longer, truncate
                                            rows[i] = row[:num_header_cols]
                                            
                                # Now create the DataFrame with fixed rows
                                df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                                
                                # Normalize column names
                                column_mapping = {
                                    'metric': 'Metric',
                                    'value or range': 'Value',
                                    'applicable period': 'Period',
                                    'period': 'Period',
                                    'value': 'Value'
                                }
                                
                                # Rename columns
                                df = df.rename(columns={k: v for k, v in column_mapping.items() 
                                                      if k in df.columns})
                                
                                # Make sure required columns exist
                                if 'Metric' not in df.columns and len(df.columns) > 0:
                                    df['Metric'] = df.iloc[:, 0]
                                
                                if 'Value' not in df.columns and len(df.columns) > 1:
                                    df['Value'] = df.iloc[:, 1]
                                    
                                if 'Period' not in df.columns:
                                    # If Period column doesn't exist, create it
                                    if len(df.columns) > 2:
                                        df['Period'] = df.iloc[:, 2]
                                    else:
                                        # If we don't have enough columns, add an empty Period column
                                        df['Period'] = ''
                                
                                # Parse values to get Low, High, Average
                                parsed_values = df["Value"].apply(parse_value_range)
                                
                                # Check if the original value is a percentage
                                is_percentage = df["Value"].str.contains("%", regex=False)
                                
                                # Add new columns
                                df["Low"] = [v[0] if isinstance(v[0], (int, float)) else None for v in parsed_values]
                                df["High"] = [v[1] if isinstance(v[1], (int, float)) else None for v in parsed_values]
                                
                                # For Average: use calculated average if available, otherwise use the text value
                                df["Average"] = [
                                    v[2] if isinstance(v[2], (int, float)) else 
                                    (v[2] if isinstance(v[2], str) else None) 
                                    for v in parsed_values
                                ]
                                
                                # Format percentage values with % symbol
                                for col in ["Low", "High", "Average"]:
                                    # Only format cells where the original value was a percentage
                                    mask = (is_percentage & df[col].notna())
                                    df.loc[mask, col] = df.loc[mask, col].apply(lambda x: f"{x}%" if isinstance(x, (int, float)) else x)
                                
                                # Add filing information
                                df["FilingDate"] = date_str
                                df["8K_Link"] = url
                                
                                # Keep only the columns we need
                                cols_to_keep = [
                                    "Metric", "Value", "Low", "High", "Average", 
                                    "Period", "FilingDate", "8K_Link"
                                ]
                                df = df[[col for col in cols_to_keep if col in df.columns]]
                                
                                results.append(df)
                                st.success("✅ Guidance extracted from this 8-K.")
                            except Exception as e:
                                st.warning(f"Error creating DataFrame: {str(e)}")
                                st.expander("Table Structure Debug").write(rows)
                                # Try a fallback approach for irregularly structured tables
                                try:
                                    # Create a more simplified DataFrame with just the data we have
                                    # Extract column names from first row
                                    if rows and len(rows) >= 2:
                                        headers = rows[0]
                                        # Create a dictionary to build the DataFrame
                                        data_dict = {header: [] for header in headers}
                                        
                                        # Fill in the data, handling missing values
                                        for row in rows[1:]:
                                            for i, header in enumerate(headers):
                                                if i < len(row):
                                                    data_dict[header].append(row[i])
                                                else:
                                                    data_dict[header].append('')
                                        
                                        # Create DataFrame from dictionary
                                        df = pd.DataFrame(data_dict)
                                        
                                        # Add necessary columns and continue processing
                                        if 'Metric' not in df.columns and len(df.columns) > 0:
                                            df['Metric'] = df.iloc[:, 0]
                                            
                                        if 'Value' not in df.columns:
                                            if len(df.columns) > 1:
                                                df['Value'] = df.iloc[:, 1]
                                            else:
                                                df['Value'] = ''
                                                
                                        if 'Period' not in df.columns:
                                            df['Period'] = ''
                                            
                                        # Simplified version of processing from above
                                        df["FilingDate"] = date_str
                                        df["8K_Link"] = url
                                        
                                        # Keep only essential columns
                                        essential_cols = ["Metric", "Value", "Period", "FilingDate", "8K_Link"]
                                        df = df[[col for col in essential_cols if col in df.columns]]
                                        
                                        results.append(df)
                                        st.success("✅ Guidance extracted with simplified processing.")
                                except Exception as e2:
                                    st.error(f"Fallback approach also failed: {str(e2)}")
                        else:
                            st.warning("⚠️ Skipped, no valid table structure found in the response.")
                    else:
                        st.warning("⚠️ Skipped, no guidance found in filing.")
                except Exception as e:
                    st.warning(f"Could not process: {url}")
                    st.error(f"Error: {str(e)}")
                    st.expander("Debug Information").code(traceback.format_exc())

            if results:
                try:
                    combined = pd.concat(results, ignore_index=True)
                    combined['Period'] = combined['Period'].apply(lambda x: normalise_period(str(x)) or x)
                    
                    # Display the table in the app
                    st.subheader("Extracted Guidance")
                    st.dataframe(combined)
                    
                    # CSV download only
                    import io
                    csv_buffer = io.BytesIO()
                    combined.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"{ticker}_guidance_output.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error combining results: {str(e)}")
                    # Try to display individual DataFrames
                    st.subheader("Individual Results (Could not combine)")
                    for i, df in enumerate(results):
                        st.write(f"Result {i+1}:")
                        st.dataframe(df)
            else:
                st.warning("No guidance data extracted.")


# ───────────────────────── UPDATED HELPERS (Auto‑generated) ────────────────────────────
import re
from typing import Optional, Dict

# Token that matches “$1,085 million”, “2.3B”, “-900K”, etc.
number_token = r'[-+]?\d[\d,\.]*\s*(?:[KMB]|million|billion)?'

def extract_number(token: str) -> Optional[float]:
    """
    Parse a token and return the numeric value **in millions**.

        '$1,085 million'  -> 1085
        '2.3B'            -> 2300
        '(900K)'          -> -0.9
    """
    if not token:
        return None

    # Handle negatives written like "(1.5M)"
    neg = token.strip().startswith('(') and token.strip().endswith(')')
    token = token.replace('(', '').replace(')', '')

    token = token.replace('$', '').replace(',', '').strip().lower()

    factor = 1.0            # default: token already in millions
    if token.endswith('billion'):
        token = token[:-7].strip();  factor = 1_000          # 1 B = 1000 M
    elif token.endswith('million'):
        token = token[:-7].strip();  factor = 1
    elif token.endswith('k'):
        token = token[:-1].strip();  factor = 0.001          # 1 K = 0.001 M
    elif token.endswith('m'):
        token = token[:-1].strip();  factor = 1
    elif token.endswith('b'):
        token = token[:-1].strip();  factor = 1_000

    try:
        value = float(token) * factor
        return -value if neg else value
    except ValueError:
        return None


def parse_value_range(text: str) -> Dict[str, Optional[float]]:
    """
    Find a single number or a range “A – B”/“A to B” in `text`.
    Returns a dict with numeric values **in millions**.
    """
    range_pat = rf'({number_token})(?:\s*[-–—~]\s*|\s+to\s+)({number_token})'
    m = re.search(range_pat, text, flags=re.I)
    if m:
        low  = extract_number(m.group(1))
        high = extract_number(m.group(2))
        avg  = (low + high) / 2 if low is not None and high is not None else None
        return {"low": low, "high": high, "avg": avg}

    m = re.search(number_token, text, flags=re.I)
    value = extract_number(m.group(0)) if m else None
    return {"low": value, "high": value, "avg": value}
# ──────────────────────────────────────────────────────────────────────

# ===== Added helper functions for number parsing and period normalisation =====
from typing import Optional, Tuple

_ORDINAL_TO_Q = {
    "first": "Q1",
    "second": "Q2",
    "third": "Q3",
    "fourth": "Q4",
}

def normalise_period(txt: str) -> Optional[str]:
    """Return standardized period name like 'Q1 FY25' or 'FY25'.
    Returns None if not recognised.
    """
    if not txt:
        return None
    text = txt.strip().lower()

    # Qx FYyyyy
    m = re.search(r'(q[1-4])\s*fy\s*(\d{2,4})', text)
    if m:
        q, yr = m.groups()
        return f"{q.upper()} FY{yr[-2:]}"

    # written-out quarter
    m = re.search(r'(first|second|third|fourth)\s+quarter\s+fiscal\s+year\s+(\d{4})', text)
    if m:
        ord_word, yr = m.groups()
        return f"{_ORDINAL_TO_Q[ord_word]} FY{yr[-2:]}"    

    # fiscal year
    m = re.search(r'fiscal\s+year\s+(\d{4})', text)
    if m:
        yr = m.group(1)
        return f"FY{yr[-2:]}"

    # full year xxxx
    m = re.search(r'full\s+year\s+(\d{4})', text)
    if m:
        yr = m.group(1)
        return f"FY{yr[-2:]}"

    return None


def extract_number(token: str) -> Optional[float]:
    """
    Parse a token like '$1,085 million', '2.3B', '(900K)' and return a float **in millions**.
    Returns None if the token cannot be parsed.
    """
    if not token:
        return None

    # Check for wrapped negative numbers: (1.2M)
    neg = token.strip().startswith('(') and token.strip().endswith(')')
    token = token.replace('(', '').replace(')', '')

    token = token.replace('$', '').replace(',', '').strip().lower()

    factor = 1.0  # default token already in millions
    if token.endswith('billion'):
        token = token[:-7].strip()
        factor = 1000        # 1 billion = 1000 million
    elif token.endswith('million'):
        token = token[:-7].strip()
        factor = 1
    elif token.endswith('k'):
        token = token[:-1].strip()
        factor = 0.001
    elif token.endswith('m'):
        token = token[:-1].strip()
        factor = 1
    elif token.endswith('b'):
        token = token[:-1].strip()
        factor = 1000

    # remove any trailing percentage sign, though Value shouldn't include it
    token = token.replace('%','')

    try:
        val = float(token) * factor
        return -val if neg else val
    except ValueError:
        return None


number_token = r'[-+]?\d[\d,\.]*\s*(?:[KMB]|million|billion)?'

def parse_value_range(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (low, high, average) from a snippet of guidance text.
    All numbers are normalised to **millions**.
    """
    if not text or not isinstance(text, str):
        return None, None, None

    text = text.strip()

    # Detect "flat" or "unchanged"
    if re.search(r'\b(flat|unchanged)\b', text, re.I):
        return 0.0, 0.0, 0.0

    # Look for range A to B or A – B
    rng = re.search(rf'({number_token})(?:\s*(?:[-–—~]|to)\s*)({number_token})', text, re.I)
    if rng:
        low = extract_number(rng.group(1))
        high = extract_number(rng.group(2))
        avg = (low + high) / 2 if low is not None and high is not None else None
        return low, high, avg

    # Fallback single value
    single = re.search(number_token, text, re.I)
    if single:
        val = extract_number(single.group(0))
        return val, val, val

    return None, None, None
    

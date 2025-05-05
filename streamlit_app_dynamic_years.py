def get_fiscal_year_end(ticker, cik=None):
    """
    Get the fiscal year end month for a company by querying SEC data.
    Returns the month number (1-12) of the fiscal year end.
    """
    try:
        if not cik:
            cik = lookup_cik(ticker)
            
        if not cik:
            st.warning(f"Could not find CIK for {ticker}. Using default fiscal year (December).")
            return 12, 31  # Default to calendar year (December 31)
            
        # Query the SEC submission data to find fiscal year end
        headers = {'User-Agent': 'Your Name Contact@domain.com'}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        
        # Get fiscal year end date
        if 'fiscalYearEnd' in data:
            fiscal_year_end = data['fiscalYearEnd']  # Format is "MMDD"
            month = int(fiscal_year_end[:2])
            day = int(fiscal_year_end[2:])
            
            month_name = datetime(2000, month, 1).strftime('%B')
            st.write(f"Retrieved fiscal year end for {ticker}: {month_name} {day}")
            
            return month, day
        else:
            st.warning(f"Could not find fiscal year end for {ticker} in SEC data. Using default fiscal year (December).")
            return 12, 31  # Default to calendar year (December 31)
    except Exception as e:
        st.error(f"Error retrieving fiscal year end for {ticker}: {str(e)}")
        return 12, 31  # Default to calendar year (December 31)import streamlit as st
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

# Default fiscal configuration - only need to specify the fiscal year end month
# All other calculations will be dynamic based on this
DEFAULT_FISCAL_CONFIG = {
    'fiscal_year_end_month': 12,  # Default to December (calendar year)
    'fiscal_year_end_day': 31
}

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., TEAM, FRSH)", "TEAM").upper()
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


def get_fiscal_dates(ticker, quarter_num, year_num, cik=None):
    """
    Calculate the appropriate date range for a company's fiscal quarter
    based on its fiscal year end from SEC data.
    """
    # Get fiscal year end month and day from SEC data
    fiscal_year_end_month, fiscal_year_end_day = get_fiscal_year_end(ticker, cik)
    
    # Build a config object with fiscal year end information
    config = {
        'fiscal_year_end_month': fiscal_year_end_month,
        'fiscal_year_end_day': fiscal_year_end_day
    }
    
    # Calculate fiscal year start month (the month after fiscal year end)
    fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
    
    # Generate quarter configuration dynamically based on fiscal year end
    # This ensures correct quarters regardless of when the fiscal year ends
    quarters = {}
    current_month = fiscal_year_start_month
    
    for q in range(1, 5):  # Generate all 4 quarters
        start_month = current_month
        
        # Each quarter is 3 months
        end_month = (start_month + 2) % 12
        if end_month == 0:  # Handle December (month 0 becomes month 12)
            end_month = 12
            
        quarters[q] = {'start_month': start_month, 'end_month': end_month}
        
        # Move to next quarter's start month
        current_month = (end_month % 12) + 1
    
    # Use our dynamically generated quarters
    quarter_info = quarters.get(quarter_num, {})
    start_month = quarter_info['start_month']
    end_month = quarter_info['end_month']
    
    # Show fiscal configuration information
    st.write(f"Using fiscal calendar for {ticker} (Fiscal year ends: {datetime(2000, fiscal_year_end_month, 1).strftime('%B')} {fiscal_year_end_day})")
    st.write(f"Quarter {quarter_num} spans months {start_month}-{end_month} " + 
             f"({datetime(2000, start_month, 1).strftime('%B')}-{datetime(2000, end_month, 1).strftime('%B')})")
    
    # CALENDAR YEAR CALCULATION
    # For FY2024 that ends in May 2024:
    #   Q1 (Jun-Aug 2023): First quarter, in previous calendar year
    #   Q2 (Sep-Nov 2023): Second quarter, in previous calendar year
    #   Q3 (Dec 2023-Feb 2024): Third quarter, spans calendar years
    #   Q4 (Mar-May 2024): Fourth quarter, in current calendar year
    
    # First determine the calendar year for the START date of this quarter
    if fiscal_year_end_month == 12:
        # Calendar fiscal year (Jan-Dec) - simple case
        start_calendar_year = year_num
    else:
        # Non-calendar fiscal year
        # If quarter starts in months after fiscal year end but before calendar year end
        if start_month > fiscal_year_end_month:
            start_calendar_year = year_num - 1
        else:
            start_calendar_year = year_num
    
    # Next determine the calendar year for the END date of this quarter
    end_calendar_year = start_calendar_year
    
    # If end month is less than start month, it means the quarter spans calendar years
    # Example: Q3 of Oracle is Dec-Feb
    if end_month < start_month:
        end_calendar_year = start_calendar_year + 1
    
    # Create actual date objects
    start_date = datetime(start_calendar_year, start_month, 1)
    
    # Calculate end date (last day of the end month)
    if end_month == 2:
        # Handle February and leap years
        if (end_calendar_year % 4 == 0 and end_calendar_year % 100 != 0) or (end_calendar_year % 400 == 0):
            end_day = 29  # Leap year
        else:
            end_day = 28
    elif end_month in [4, 6, 9, 11]:
        end_day = 30
    else:
        end_day = 31
    
    end_date = datetime(end_calendar_year, end_month, end_day)
    
    # Calculate expected earnings report dates
    report_start = end_date + timedelta(days=15)
    report_end = report_start + timedelta(days=45)
    
    # Output info about the dates
    quarter_period = f"Q{quarter_num} FY{year_num}"
    period_description = f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
    expected_report = f"~{report_start.strftime('%B %d, %Y')} to {report_end.strftime('%B %d, %Y')}"
    
    # Debug: Show all quarters
    st.write("All quarters for this fiscal calendar:")
    for q, dates in quarters.items():
        st.write(f"Q{q}: {datetime(2000, dates['start_month'], 1).strftime('%B')}-{datetime(2000, dates['end_month'], 1).strftime('%B')}")
    
    return {
        'quarter_period': quarter_period,
        'start_date': start_date,
        'end_date': end_date,
        'report_start': report_start,
        'report_end': report_end,
        'period_description': period_description,
        'expected_report': expected_report
    }
    
    return {
        'quarter_period': quarter_period,
        'start_date': start_date,
        'end_date': end_date,
        'report_start': report_start,
        'report_end': report_end,
        'period_description': period_description,
        'expected_report': expected_report
    }


def get_accessions(cik, years_back=None, specific_quarter=None):
    """General function for finding filings"""
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    filings = data["filings"]["recent"]
    accessions = []
    
    # Get ticker from session state
    ticker = st.session_state.get('ticker', '').upper()
    
    if years_back:
        # Modified to add one extra quarter (approximately 91.25 days)
        # For example: 1 year = 365 + 91.25 days = 456.25 days
        cutoff = datetime.today() - timedelta(days=(365 * years_back) + 91.25)
        
        st.write(f"Looking for filings from the past {years_back} years plus 1 quarter (from {cutoff.strftime('%Y-%m-%d')} to present)")
        
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if date >= cutoff:
                    accessions.append((accession, date_str))
    
    elif specific_quarter:
        # Parse quarter and year from input - handle various formats
        # Examples: 2Q25, Q4FY24, Q3 2024, Q1 FY 2025, etc.
        match = re.search(r'(?:Q?(\d)Q?|Q(\d))(?:\s*FY\s*|\s*)?(\d{2}|\d{4})', specific_quarter.upper())
        if match:
            quarter = match.group(1) or match.group(2)
            year = match.group(3)
            
            # Convert 2-digit year to 4-digit year
            if len(year) == 2:
                year = '20' + year
                
            quarter_num = int(quarter)
            year_num = int(year)
            
            # Get fiscal dates based on company's fiscal calendar from SEC data
            fiscal_info = get_fiscal_dates(ticker, quarter_num, year_num, cik)
            
            # Display fiscal quarter information
            st.write(f"Looking for {ticker} {fiscal_info['quarter_period']} filings")
            st.write(f"Fiscal quarter period: {fiscal_info['period_description']}")
            st.write(f"Expected earnings reporting window: {fiscal_info['expected_report']}")
            
            # We want to find filings around the expected earnings report date
            start_date = fiscal_info['report_start'] - timedelta(days=15)  # Include potential early reports
            end_date = fiscal_info['report_end'] + timedelta(days=15)  # Include potential late reports
            
            st.write(f"Searching for filings between: {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
            
            # Find filings in this date range
            for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
                if form == "8-K":
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    if start_date <= date <= end_date:
                        accessions.append((accession, date_str))
                        st.write(f"Found filing from {date_str}: {accession}")
    
    else:  # Default: most recent only
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                accessions.append((accession, date_str))
                break
    
    # Show debug info about the selected accessions
    if accessions:
        st.write(f"Found {len(accessions)} relevant 8-K filings")
    else:
        # Show all available dates for reference
        available_dates = []
        for form, date_str in zip(filings["form"], filings["filingDate"]):
            if form == "8-K":
                available_dates.append(date_str)
        
        if available_dates:
            available_dates.sort(reverse=True)  # Show most recent first
            st.write("All available 8-K filing dates:")
            for date in available_dates[:15]:  # Show only the first 15 to avoid cluttering
                st.write(f"- {date}")
            if len(available_dates) > 15:
                st.write(f"... and {len(available_dates) - 15} more")
    
    return accessions


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

VERY IMPORTANT:
- Look for sections titled 'Outlook', 'Guidance', 'Financial Outlook', 'Business Outlook', or similar
- For any percentage values, always include the % symbol in your output (e.g., "5% to 7%" or "5%-7%")
- If the guidance mentions a negative percentage like "(5%)" or "decrease of 5%", output it as "-5%"
- Preserve any descriptive text like "Approximately" or "Around" in your output
- Be sure to capture year-over-year growth metrics as well as absolute values

Respond in table format without commentary.\n\n{text}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"⚠️ Error extracting guidance: {str(e)}")
        return None


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
            
            # Store the ticker for later use
            st.session_state['ticker'] = ticker
            
            # Handle different filtering options
            if quarter_input.strip():
                # Quarter input takes precedence
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
                accessions = get_accessions(cik)

            links = get_ex99_1_links(cik, accessions)
            results = []

            for date_str, acc, url in links:
                st.write(f"📄 Processing {url}")
                try:
                    html = requests.get(url, headers={"User-Agent": "MyCompanyName Data Research Contact@mycompany.com"}).text
                    text = BeautifulSoup(html, "html.parser").get_text()
                    
                    # Search for guidance-related sections in the text
                    outlook_sections = []
                    outlook_keywords = ["outlook", "guidance", "financial outlook", "business outlook", "forecast"]
                    
                    # First, try to find sections with these keywords in headers
                    for keyword in outlook_keywords:
                        pattern = re.compile(fr'(?i)(?:^|\n|\r)\s*{keyword}[^\n\r]*(?:\n|\r)', re.MULTILINE)
                        matches = pattern.finditer(text)
                        for match in matches:
                            start_pos = match.start()
                            # Find a reasonable endpoint for this section (next header or a certain number of paragraphs)
                            next_header = re.search(r'(?:\n|\r)\s*[A-Z][^\n\r]*(?:\n|\r)', text[start_pos+100:])
                            end_pos = start_pos + 100 + next_header.start() if next_header else min(start_pos + 5000, len(text))
                            outlook_sections.append(text[start_pos:end_pos])
                    
                    # If we couldn't find specific sections, try a broader approach
                    if not outlook_sections:
                        # Look for forward-looking statements
                        forw_idx = text.lower().find("forward-looking statements")
                        if forw_idx != -1:
                            # Capture the text leading up to forward-looking statements disclaimer
                            outlook_sections.append(text[:forw_idx])
                        else:
                            # If all else fails, use the full document
                            outlook_sections.append(text)
                    
                    # If we found multiple sections, prioritize them by relevance
                    # Combine the most promising sections for extraction
                    combined_text = "\n\n".join(outlook_sections[:3])  # Use up to 3 most relevant sections
                    
                    # Now extract guidance from the identified sections
                    table = extract_guidance(combined_text, ticker, client)
                    
                    if table and "|" in table:
                        rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                        if len(rows) > 1:  # Check if we have header and at least one row of data
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
                            st.warning(f"⚠️ Table format was detected but no data rows were found in {url}")
                    else:
                        st.warning(f"⚠️ No guidance table found in {url}")
                        
                        # Show a sample of the text to help debug
                        sample_length = min(500, len(text))
                        st.write(f"Sample of document text (first {sample_length} characters):")
                        st.text(text[:sample_length] + "...")
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

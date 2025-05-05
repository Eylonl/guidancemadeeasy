import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import os
import re
import io

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
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, ORCL)", "MSFT").upper()

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


def get_fiscal_year_end(ticker, cik):
    """
    Get the fiscal year end month for a company from SEC data.
    Returns the month (1-12) and day.
    """
    try:
        headers = {'User-Agent': 'Your Name Contact@domain.com'}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        
        # Extract fiscal year end info - format is typically "MMDD" 
        if 'fiscalYearEnd' in data:
            fiscal_year_end = data['fiscalYearEnd']
            if len(fiscal_year_end) == 4:  # MMDD format
                month = int(fiscal_year_end[:2])
                day = int(fiscal_year_end[2:])
                
                month_name = datetime(2000, month, 1).strftime('%B')
                st.success(f"✅ Retrieved fiscal year end for {ticker}: {month_name} {day}")
                
                return month, day
        
        # If not found, default to December 31 (calendar year)
        st.warning(f"⚠️ Could not determine fiscal year end for {ticker} from SEC data. Using December 31 (calendar year).")
        return 12, 31
        
    except Exception as e:
        st.warning(f"⚠️ Error retrieving fiscal year end: {str(e)}. Using December 31 (calendar year).")
        return 12, 31


def generate_fiscal_quarters(fiscal_year_end_month):
    """
    Dynamically generate fiscal quarters based on the fiscal year end month.
    """
    # Calculate the first month of the fiscal year (month after fiscal year end)
    fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
    
    # Generate all four quarters
    quarters = {}
    current_month = fiscal_year_start_month
    
    for q in range(1, 5):
        start_month = current_month
        
        # Each quarter is 3 months
        end_month = (start_month + 2) % 12
        if end_month == 0:  # Handle December (month 0 becomes month 12)
            end_month = 12
            
        quarters[q] = {'start_month': start_month, 'end_month': end_month}
        
        # Move to next quarter's start month
        current_month = (end_month % 12) + 1
    
    return quarters


def get_fiscal_dates(ticker, quarter_num, year_num, fiscal_year_end_month, fiscal_year_end_day):
    """
    Calculate the appropriate date range for a fiscal quarter
    based on the fiscal year end month.
    """
    # Generate quarters dynamically based on fiscal year end
    quarters = generate_fiscal_quarters(fiscal_year_end_month)
    
    # Get the specified quarter
    if quarter_num < 1 or quarter_num > 4:
        st.error(f"Invalid quarter number: {quarter_num}. Must be 1-4.")
        return None
        
    quarter_info = quarters[quarter_num]
    start_month = quarter_info['start_month']
    end_month = quarter_info['end_month']
    
    # Determine if the quarter spans calendar years
    spans_calendar_years = end_month < start_month
    
    # Determine the calendar year for each quarter
    if fiscal_year_end_month == 12:
        # Simple case: Calendar year matches fiscal year
        start_calendar_year = year_num
    else:
        # For non-calendar fiscal years, determine which calendar year the quarter falls in
        fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
        
        if start_month >= fiscal_year_start_month:
            # This quarter starts in the previous calendar year
            # Example: For fiscal year ending in June (FY2024 = Jul 2023-Jun 2024)
            # Q1 (Jul-Sep) and Q2 (Oct-Dec) start in calendar year 2023
            start_calendar_year = year_num - 1
        else:
            # This quarter starts in the current calendar year
            # Example: For fiscal year ending in June (FY2024 = Jul 2023-Jun 2024)
            # Q3 (Jan-Mar) and Q4 (Apr-Jun) start in calendar year 2024
            start_calendar_year = year_num
    
    # For quarters that span calendar years, the end date is in the next calendar year
    end_calendar_year = start_calendar_year
    if spans_calendar_years:
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
    
    # Calculate expected earnings report dates (typically a few weeks after quarter end)
    report_start = end_date + timedelta(days=15)
    report_end = report_start + timedelta(days=45)  # Typical reporting window
    
    # Output info about the dates
    quarter_period = f"Q{quarter_num} FY{year_num}"
    period_description = f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
    expected_report = f"~{report_start.strftime('%B %d, %Y')} to {report_end.strftime('%B %d, %Y')}"
    
    # Display fiscal quarter information
    st.write(f"Fiscal year ends in {datetime(2000, fiscal_year_end_month, 1).strftime('%B')} {fiscal_year_end_day}")
    st.write(f"Quarter {quarter_num} spans: {datetime(2000, start_month, 1).strftime('%B')}-{datetime(2000, end_month, 1).strftime('%B')}")
    
    # Show all quarters
    st.write("All quarters for this fiscal pattern:")
    for q, q_info in quarters.items():
        st.write(f"Q{q}: {datetime(2000, q_info['start_month'], 1).strftime('%B')}-{datetime(2000, q_info['end_month'], 1).strftime('%B')}")
    
    return {
        'quarter_period': quarter_period,
        'start_date': start_date,
        'end_date': end_date,
        'report_start': report_start,
        'report_end': report_end,
        'period_description': period_description,
        'expected_report': expected_report
    }


def get_accessions(cik, ticker, years_back=None, specific_quarter=None):
    """General function for finding filings"""
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    filings = data["filings"]["recent"]
    accessions = []
    
    # Auto-detect fiscal year end from SEC data
    fiscal_year_end_month, fiscal_year_end_day = get_fiscal_year_end(ticker, cik)
    
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
            
            # Get fiscal dates based on fiscal year end month
            fiscal_info = get_fiscal_dates(ticker, quarter_num, year_num, fiscal_year_end_month, fiscal_year_end_day)
            
            if not fiscal_info:
                return []
            
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


def generate_guidance_prompt(text, ticker):
    """
    Generates the prompt that would be sent to OpenAI's API.
    Returns the prompt text.
    """
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
    return prompt


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
    cik = lookup_cik(ticker)
    if not cik:
        st.error("CIK not found for ticker.")
    else:
        # Store the ticker for later use
        st.session_state['ticker'] = ticker
        
        # Handle different filtering options
        if quarter_input.strip():
            # Quarter input takes precedence
            accessions = get_accessions(cik, ticker, specific_quarter=quarter_input.strip())
            if not accessions:
                st.warning(f"No 8-K filings found for {quarter_input}. Please check the format (e.g., 2Q25, Q4FY24).")
        elif year_input.strip():
            try:
                years_back = int(year_input.strip())
                accessions = get_accessions(cik, ticker, years_back=years_back)
            except:
                st.error("Invalid year input. Must be a number.")
                accessions = []
        else:
            # Default to most recent if neither input is provided
            accessions = get_accessions(cik, ticker)

        links = get_ex99_1_links(cik, accessions)
        
        # For each link, create a download button for the OpenAI prompt
        for i, (date_str, acc, url) in enumerate(links):
            st.write(f"📄 Filing from {date_str}: {url}")
            
            # Create an expander for each filing to avoid cluttering the UI
            with st.expander(f"View Details for Filing {i+1}"):
                try:
                    headers = {"User-Agent": "MyCompanyName Data Research Contact@mycompany.com"}
                    html = requests.get(url, headers=headers).text
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
                    
                    # Generate the prompt
                    prompt = generate_guidance_prompt(combined_text, ticker)
                    
                    # Create download button for the prompt
                    st.download_button(
                        label=f"📥 Download OpenAI Prompt for {date_str}",
                        data=prompt,
                        file_name=f"{ticker}_guidance_prompt_{date_str}.txt",
                        mime="text/plain"
                    )
                    
                    # Show a preview of the prompt
                    with st.expander("Preview Prompt"):
                        st.text_area("OpenAI Prompt", prompt, height=400)
                        
                    st.success(f"✅ Prompt generated for {date_str} filing. Click the download button above to save it.")
                    
                    # Show a sample of the text to help debug
                    with st.expander("Preview Document Text"):
                        sample_length = min(500, len(text))
                        st.write(f"Sample of document text (first {sample_length} characters):")
                        st.text(text[:sample_length] + "...")
                    
                except Exception as e:
                    st.warning(f"Could not process: {url}. Error: {str(e)}")
        
        # If no links were found
        if not links:
            st.warning("No filings found to generate prompts from.")

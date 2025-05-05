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
    """
    Enhanced function to parse value ranges from guidance text.
    Handles formats like "$0.08-$0.09" with dollar signs on both numbers.
    """
    if not isinstance(text, str):
        return (None, None, None)
    
    # Check for "flat" or "unchanged"
    if re.search(r'\b(flat|unchanged)\b', text, re.I):
        return (0.0, 0.0, 0.0)
    
    # First try to match ranges with dollar signs on both sides like "$0.08-$0.09"
    dollar_range = re.search(r'\$\s*(\d+(?:\.\d+)?)\s*[-–—~]\s*\$\s*(\d+(?:\.\d+)?)', text, re.I)
    if dollar_range:
        lo = float(dollar_range.group(1))
        hi = float(dollar_range.group(2))
        avg = (lo + hi) / 2 if lo is not None and hi is not None else None
        return (lo, hi, avg)
    
    # Try to match percentage ranges with percent signs on both sides like "5%-7%"
    percent_range = re.search(r'(\d+(?:\.\d+)?)\s*%\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*%', text, re.I)
    if percent_range:
        lo = float(percent_range.group(1))
        hi = float(percent_range.group(2))
        avg = (lo + hi) / 2 if lo is not None and hi is not None else None
        return (lo, hi, avg)
    
    # Then try standard range formats
    rng = re.search(fr'({number_token})\s*(?:[-–—~]|to)\s*({number_token})', text, re.I)
    if rng:
        lo = extract_number(rng.group(1))
        hi = extract_number(rng.group(2))
        avg = (lo+hi)/2 if lo is not None and hi is not None else None
        return (lo, hi, avg)
    
    # Finally check for a single value
    single = re.search(number_token, text, re.I)
    if single:
        v = extract_number(single.group(0))
        return (v, v, v)
    
    return (None, None, None)


st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("📄 SEC 8-K Guidance Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, ORCL)", "MSFT").upper()
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


def find_guidance_paragraphs(text):
    """
    Extract paragraphs from text that are likely to contain guidance information.
    Returns both the filtered paragraphs and a boolean indicating if any were found.
    """
    # Define patterns to identify guidance sections
    guidance_patterns = [
        r'(?i)outlook',
        r'(?i)guidance',
        r'(?i)financial outlook',
        r'(?i)business outlook',
        r'(?i)forward[\s-]*looking',
        r'(?i)for (?:the )?(?:fiscal|next|coming|upcoming) (?:quarter|year)',
        r'(?i)(?:we|company) expect(?:s)?',
        r'(?i)revenue (?:is|to be) (?:in the range of|expected to|anticipated to)',
        r'(?i)to be (?:in the range of|approximately)',
        r'(?i)margin (?:is|to be) (?:expected|anticipated|forecast)',
        r'(?i)growth of (?:approximately|about)',
        r'(?i)for (?:fiscal|the fiscal)',
        r'(?i)next quarter',
        r'(?i)full year',
        r'(?i)current quarter',
        r'(?i)future quarter',
        r'(?i)Q[1-4]'
    ]
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', text)
    
    # Find paragraphs matching guidance patterns
    guidance_paragraphs = []
    
    for para in paragraphs:
        if any(re.search(pattern, para) for pattern in guidance_patterns):
            # Check if it's likely a forward-looking statement (not a disclaimer)
            if not (re.search(r'(?i)safe harbor', para) or 
                    (re.search(r'(?i)forward-looking statements', para) and 
                     re.search(r'(?i)risks', para))):
                guidance_paragraphs.append(para)
    
    # Check if we found any guidance paragraphs
    found_paragraphs = len(guidance_paragraphs) > 0
    
    # If no guidance paragraphs found, get a small sample of the document
    if not found_paragraphs:
        # Extract a small sample from sections that might contain guidance
        for section_name in ["outlook", "guidance", "forward", "future", "expect", "anticipate"]:
            section_pattern = re.compile(fr'(?i)(?:^|\n|\. )([^.]*{section_name}[^.]*\. [^.]*\. [^.]*\.)', re.MULTILINE)
            matches = section_pattern.findall(text)
            for match in matches:
                if len(match.strip()) > 50:  # Ensure it's not just a brief mention
                    guidance_paragraphs.append(match.strip())
    
    # If still no paragraphs found, get first few paragraphs and any with financial terms
    if not guidance_paragraphs:
        # Add first few paragraphs (might contain summary of results including guidance)
        first_few = paragraphs[:5] if len(paragraphs) > 5 else paragraphs
        guidance_paragraphs.extend([p for p in first_few if len(p.strip()) > 100])
        
        # Add paragraphs with financial terms
        financial_terms = ["revenue", "earnings", "eps", "income", "margin", "growth", "forecast"]
        for para in paragraphs:
            if any(term in para.lower() for term in financial_terms) and para not in guidance_paragraphs:
                if len(para.strip()) > 100:  # Ensure it's substantial
                    guidance_paragraphs.append(para)
                    if len(guidance_paragraphs) > 15:  # Limit sample size
                        break
    
    # Combine paragraphs and add a note about the original document
    formatted_paragraphs = "\n\n".join(guidance_paragraphs)
    
    # Add metadata about the document to help GPT understand the context
    if guidance_paragraphs:
        formatted_paragraphs = (
            f"DOCUMENT TYPE: SEC 8-K Earnings Release for {ticker}\n\n"
            f"POTENTIAL GUIDANCE INFORMATION (extracted from full document):\n\n{formatted_paragraphs}\n\n"
            "Note: These are selected paragraphs that may contain forward-looking guidance."
        )
    
    return formatted_paragraphs, found_paragraphs


def extract_guidance(text, ticker, client):
    """
    Modified to work with just the guidance paragraphs instead of the full text.
    """
    prompt = f"""You are a financial analyst assistant. Extract ALL forward-looking guidance, projections, and outlook statements given in this earnings release for {ticker}. 

Return a structured list containing:
- metric (e.g. Revenue, EPS, Operating Margin)
- value or range (e.g. $1.5B–$1.6B or $2.05)
- applicable period (e.g. Q3 FY24, Full Year 2025)

VERY IMPORTANT:
- Look for sections titled 'Outlook', 'Guidance', 'Financial Outlook', 'Business Outlook', or similar
- Also look for statements containing phrases like "expect", "anticipate", "forecast", "will be", "to be in the range of"
- Review the ENTIRE document for ANY forward-looking statements about future performance
- Pay special attention to sections describing "For the fiscal quarter", "For the fiscal year", "For next quarter", etc.
- For any percentage values, always include the % symbol in your output (e.g., "5% to 7%" or "5%-7%")
- If the guidance mentions a negative percentage like "(5%)" or "decrease of 5%", output it as "-5%"
- Preserve any descriptive text like "Approximately" or "Around" in your output
- Be sure to capture year-over-year growth metrics as well as absolute values
- Look for common financial metrics: Revenue, EPS, Operating Margin, Free Cash Flow, Gross Margin, etc.
- Include both quarterly and full-year guidance if available
- If guidance includes both GAAP and non-GAAP measures, include both with clear labels

CRITICAL FORMATTING FOR BILLION VALUES:
- When a value is expressed in billions (e.g., "$1.10 billion" or "$1.11B"), you MUST multiply it by 1000 and express it in millions in your output.
  - Example: "$1.10 billion" should be output as "$1,100 million"
  - Example: "$1.11B" should be output as "$1,110 million"
  - Example: A range of "$1.10-$1.11 billion" should be output as "$1,100-$1,110 million"
- Always maintain the exact decimal precision when converting (multiply the exact number by 1000)

IMPORTANT FORMATTING INSTRUCTIONS:
- For dollar ranges, use the format "$X-$Y" (with dollar sign before each number)
  - Example: "$0.08-$0.09" (not "$0.08-0.09")
- For percentage ranges, use the format "X%-Y%" (with % after each number)
  - Example: "5%-7%" (not "5-7%")
- For other numeric ranges, use "X-Y" format
  - Example: "100-110" (not "100 to 110")
- Keep numbers exactly as stated (don't convert $0.08 to $0.8, etc.) EXCEPT for billion values which must be converted as instructed above

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
            results = []

            for date_str, acc, url in links:
                st.write(f"📄 Processing {url}")
                try:
                    html = requests.get(url, headers={"User-Agent": "MyCompanyName Data Research Contact@mycompany.com"}).text
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Extract text while preserving structure
                    text = soup.get_text(" ", strip=True)
                    
                    # Find paragraphs containing guidance patterns
                    guidance_paragraphs, found_guidance = find_guidance_paragraphs(text)
                    
                    # Check if we found any guidance paragraphs
                    if found_guidance:
                        st.success(f"✅ Found potential guidance information.")
                        
                        # Extract guidance from the highlighted text
                        table = extract_guidance(guidance_paragraphs, ticker, client)
                    else:
                        st.warning(f"⚠️ No guidance paragraphs found. Trying with a sample of the document.")
                        # Use a sample of the document to reduce token usage
                        sample_text = "DOCUMENT TYPE: SEC 8-K Earnings Release for " + ticker + "\n\n"
                        paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', text)
                        sample_text += "\n\n".join(paragraphs[:15])  # Just use first few paragraphs
                        table = extract_guidance(sample_text, ticker, client)
                    
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
                            
                            # Show a sample of the text to help debug
                            st.write("Sample of text sent to OpenAI:")
                            sample_length = min(500, len(guidance_paragraphs))
                            st.text(guidance_paragraphs[:sample_length] + "..." if len(guidance_paragraphs) > sample_length else guidance_paragraphs)
                    else:
                        st.warning(f"⚠️ No guidance table found in {url}")
                        
                        # Show a sample of the text to help debug
                        st.write("Sample of text sent to OpenAI:")
                        sample_length = min(500, len(guidance_paragraphs))
                        st.text(guidance_paragraphs[:sample_length] + "..." if len(guidance_paragraphs) > sample_length else guidance_paragraphs)
                except Exception as e:
                    st.warning(f"Could not process: {url}. Error: {str(e)}")

            if results:
                combined = pd.concat(results, ignore_index=True)
                
                # Preview the table
                st.subheader("🔍 Preview of Extracted Guidance")
                
                # Select the most relevant columns for display
                display_cols = ["Metric", "Value", "Period", "Low", "High", "Average", "FilingDate"]
                display_df = combined[display_cols] if all(col in combined.columns for col in display_cols) else combined
                
                # Display the table with formatting
                st.dataframe(
                    display_df.style.format({
                        "Low": lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
                        "High": lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
                        "Average": lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                    }),
                    use_container_width=True
                )
                
                # Add download button
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No guidance data extracted.")

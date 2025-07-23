import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os
import re

def format_percent(val):
“”“Format a value as a percentage with consistent decimal places”””
if val is None:
return None
if isinstance(val, (int, float)):
return f”{val:.1f}%”
return val

def format_dollar(val):
“”“Format a value as a dollar amount with consistent decimal places”””
if val is None:
return None
if isinstance(val, (int, float)):
# Use different formatting based on value magnitude
if abs(val) >= 100:  # Large values, use whole numbers
return f”${val:.0f}”
elif abs(val) >= 10:  # Medium values, use 1 decimal place
return f”${val:.1f}”
else:  # Small values, use 2 decimal places
return f”${val:.2f}”
return val

def extract_guidance(text, ticker, client, model_name):
“””
Enhanced function to extract guidance from SEC filings.
Now directly extracting Low, High, and Average values from the language model.
“””
prompt = f””“You are a financial analyst assistant. Extract ALL forward-looking guidance, projections, and outlook statements given in this earnings release for {ticker}.

Return a structured table containing the following columns:

- metric (e.g. Revenue, EPS, Operating Margin)
- value_or_range (e.g. $1.5B–$1.6B or $2.05 or $(0.05) to $0.10 - EXACTLY as it appears in the text)
- period (e.g. Q3 FY24, Full Year 2025)
- period_type (MUST be either “Quarter” or “Full Year” based on the period text)
- low (numeric low end of the range, or the single value if not a range)
- high (numeric high end of the range, or the single value if not a range)
- average (average of low and high, or just the value if not a range)

VERY IMPORTANT:

- Look for sections titled ‘Outlook’, ‘Guidance’, ‘Financial Outlook’, ‘Business Outlook’, or similar
- Also look for statements containing phrases like “expect”, “anticipate”, “forecast”, “will be”, “to be in the range of”
- Review the ENTIRE document for ANY forward-looking statements about future performance
- Pay special attention to sections describing “For the fiscal quarter”, “For the fiscal year”, “For next quarter”, etc.

CRITICAL GUIDANCE FOR THE NUMERIC COLUMNS (low, high, average):

- For low, high, and average columns, provide ONLY numeric values (no $ signs, no % symbols, no “million” or “billion” text)
- Use negative numbers for negative values: -1 instead of “(1)” and -5 instead of “(5%)”
- For mixed sign ranges like “$(1) million to $1 million”, make sure low is negative (-1) and high is positive (1)
- Convert all billions to millions (multiply by 1000): $1.2 billion → 1200
- For percentages, just give the number without % sign: “5% to 7%” → low=5, high=7
- For dollar amounts, omit the $ sign: “$0.05 to $0.10” → low=0.05, high=0.10

FOR THE PERIOD TYPE COLUMN:

- Classify each period as either “Quarter” or “Full Year” based on the applicable period
- Use “Quarter” for: Q1, Q2, Q3, Q4, First Quarter, Next Quarter, Current Quarter, etc.
- Use “Full Year” for: Full Year, Fiscal Year, FY, Annual, Year Ending, etc.
- If a period just mentions a year (e.g., “2023” or “FY24”) without specifying a quarter, classify it as “Full Year”
- THIS COLUMN IS REQUIRED AND MUST ONLY CONTAIN “Quarter” OR “Full Year” - NO OTHER VALUES

FORMATTING INSTRUCTIONS FOR VALUE_OR_RANGE COLUMN:

- Always preserve the original notation exactly as it appears in the document (maintain parentheses, $ signs, % symbols)
- Example: If document says “($0.05) to $0.10”, use exactly “($0.05) to $0.10” in value_or_range column
- Example: If document says “(5%) to 2%”, use exactly “(5%) to 2%” in value_or_range column
- For billion values, keep them as billions in this column: “$1.10 billion to $1.11 billion”

Respond in table format without commentary.\n\n{text}”””

```
try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content
except Exception as e:
    st.warning(f"⚠️ Error extracting guidance: {str(e)}")
    return None
```

def split_gaap_non_gaap(df):
“”“Split rows that contain both GAAP and non-GAAP guidance into separate rows”””
if ‘value_or_range’ not in df.columns or ‘metric’ not in df.columns:
return df  # Avoid crash if column names are missing

```
rows = []
for _, row in df.iterrows():
    val = str(row['value_or_range'])
    match = re.search(r'(\d[\d\.\s%to–-]*)\s*on a GAAP basis.*?(\d[\d\.\s%to–-]*)\s*on a non-GAAP basis', val, re.I)
    if match:
        gaap_val = match.group(1).strip() + " GAAP"
        non_gaap_val = match.group(2).strip() + " non-GAAP"
        for new_val, label in [(gaap_val, "GAAP"), (non_gaap_val, "Non-GAAP")]:
            new_row = row.copy()
            new_row["value_or_range"] = new_val
            new_row["metric"] = f"{row['metric']} ({label})"
            rows.append(new_row)
    else:
        rows.append(row)
return pd.DataFrame(rows)
```

def format_guidance_values(df):
“”“Format the numeric values to appropriate formats based on the metric and value types”””
# Make a copy to avoid modifying the original
formatted_df = df.copy()

```
for idx, row in df.iterrows():
    value_text = str(row.get('value_or_range', ''))
    
    # Determine if it's a percentage value
    is_percentage = '%' in value_text
    
    # Determine if it's a dollar value
    is_dollar = '$' in value_text
    
    # Format the Low, High, Average columns based on the value type
    for col in ['low', 'high', 'average']:
        if col in df.columns and not pd.isnull(row.get(col)):
            try:
                val = float(row[col])
                
                if is_percentage:
                    formatted_df.at[idx, col] = f"{val:.1f}%"
                elif is_dollar:
                    if abs(val) >= 100:
                        formatted_df.at[idx, col] = f"${val:.0f}"
                    elif abs(val) >= 10:
                        formatted_df.at[idx, col] = f"${val:.1f}"
                    else:
                        formatted_df.at[idx, col] = f"${val:.2f}"
            except:
                # Skip if we can't parse as float
                continue

return formatted_df
```

# Streamlit App Setup

st.set_page_config(page_title=“SEC 8-K Guidance Extractor”, layout=“centered”)
st.title(“📄 SEC 8-K Guidance Extractor”)

# Modified inputs section to support both ticker and CIK

input_method = st.radio(“Choose input method:”, [“Ticker Symbol”, “CIK Code”])

if input_method == “Ticker Symbol”:
identifier = st.text_input(“Enter Stock Ticker (e.g., MSFT, ORCL)”, “MSFT”).upper()
ticker = identifier  # For display purposes
cik = None  # Will be looked up
else:
identifier = st.text_input(“Enter CIK Code (e.g., 0000789019)”, “”)
ticker = identifier  # Will use CIK as ticker for display
cik = identifier.zfill(10) if identifier else None  # Pad with zeros to 10 digits

api_key = st.text_input(“Enter OpenAI API Key”, type=“password”)

# Add model selection dropdown

openai_models = {
“GPT-4 Turbo”: “gpt-4-turbo-preview”,
“GPT-4”: “gpt-4”,
“GPT-3.5 Turbo”: “gpt-3.5-turbo”
}
selected_model = st.selectbox(
“Select OpenAI Model”,
list(openai_models.keys()),
index=0  # Default to first option (GPT-4 Turbo)
)

# Both filter options displayed at the same time

year_input = st.text_input(“How many years back to search for 8-K filings? (Leave blank for most recent only)”, “”)
quarter_input = st.text_input(“OR enter specific quarter (e.g., 2Q25, Q4FY24)”, “”)

# This is Part 2 - Add this after Part 1

@st.cache_data(show_spinner=False)
def lookup_cik(ticker):
headers = {‘User-Agent’: ‘Your Name Contact@domain.com’}
res = requests.get(“https://www.sec.gov/files/company_tickers.json”, headers=headers)
data = res.json()
for entry in data.values():
if entry[“ticker”].upper() == ticker:
return str(entry[“cik_str”]).zfill(10)

def get_ticker_from_cik(cik):
“”“Lookup ticker symbol from CIK for display purposes”””
try:
headers = {‘User-Agent’: ‘Your Name Contact@domain.com’}
res = requests.get(“https://www.sec.gov/files/company_tickers.json”, headers=headers)
data = res.json()
cik_int = int(cik)
for entry in data.values():
if entry[“cik_str”] == cik_int:
return entry[“ticker”].upper()
return None
except:
return None

def get_fiscal_year_end(ticker, cik):
“””
Get the fiscal year end month for a company from SEC data.
Returns the month (1-12) and day.
“””
try:
headers = {‘User-Agent’: ‘Your Name Contact@domain.com’}
url = f”https://data.sec.gov/submissions/CIK{cik}.json”
resp = requests.get(url, headers=headers)
data = resp.json()

```
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
```

def generate_fiscal_quarters(fiscal_year_end_month):
“””
Dynamically generate fiscal quarters based on the fiscal year end month.
“””
# Calculate the first month of the fiscal year (month after fiscal year end)
fiscal_year_start_month = (fiscal_year_end_month % 12) + 1

```
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
```

def get_fiscal_dates(ticker, quarter_num, year_num, fiscal_year_end_month, fiscal_year_end_day):
“””
Calculate the appropriate date range for a fiscal quarter
based on the fiscal year end month.
“””
# Generate quarters dynamically based on fiscal year end
quarters = generate_fiscal_quarters(fiscal_year_end_month)

```
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
        start_calendar_year = year_num - 1
    else:
        # This quarter starts in the current calendar year
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
```

def get_accessions(cik, ticker, years_back=None, specific_quarter=None):
“”“General function for finding filings”””
headers = {‘User-Agent’: ‘Your Name Contact@domain.com’}
url = f”https://data.sec.gov/submissions/CIK{cik}.json”
resp = requests.get(url, headers=headers)
data = resp.json()
filings = data[“filings”][“recent”]
accessions = []

```
# Auto-detect fiscal year end from SEC data
fiscal_year_end_month, fiscal_year_end_day = get_fiscal_year_end(ticker, cik)

if years_back:
    # Modified to add one extra quarter (approximately 91.25 days)
    cutoff = datetime.today() - timedelta(days=(365 * years_back) + 91.25)
    
    st.write(f"Looking for filings from the past {years_back} years plus 1 quarter (from {cutoff.strftime('%Y-%m-%d')} to present)")
    
    for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
        if form == "8-K":
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date >= cutoff:
                accessions.append((accession, date_str))

elif specific_quarter:
    # Parse quarter and year from input - handle various formats
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
        start_date = fiscal_info['report_start'] - timedelta(days=15)
        end_date = fiscal_info['report_end'] + timedelta(days=15)
        
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
```

# This is Part 3a - Add this after Part 2

def get_ex99_1_links(cik, accessions):
“”“Enhanced function to find exhibit 99.1 files with better searching”””
links = []
headers = {‘User-Agent’: ‘Your Name Contact@domain.com’}

```
for accession, date_str in accessions:
    # Remove dashes from accession number for folder name
    accession_no_dashes = accession.replace('-', '')
    base_folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/"
    index_url = base_folder + f"{accession}-index.htm"
    
    try:
        res = requests.get(index_url, headers=headers, timeout=30)
        if res.status_code != 200:
            continue
            
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Look for exhibit 99.1 in multiple ways
        found_exhibit = False
        
        # Method 1: Look for "99.1" or "99.01" in table rows
        for row in soup.find_all("tr"):
            row_text = row.get_text().lower()
            if "99.1" in row_text or "99.01" in row_text:
                tds = row.find_all("td")
                if len(tds) >= 3:
                    filename = tds[2].text.strip()
                    exhibit_url = base_folder + filename
                    links.append((date_str, accession, exhibit_url))
                    found_exhibit = True
                    break
        
        # Method 2: If not found, look for any .htm files that might be exhibits
        if not found_exhibit:
            for row in soup.find_all("tr"):
                tds = row.find_all("td")
                if len(tds) >= 3:
                    filename = tds[2].text.strip()
                    if filename.endswith('.htm') and ('ex' in filename.lower() or 'exhibit' in filename.lower()):
                        exhibit_url = base_folder + filename
                        links.append((date_str, accession, exhibit_url))
                        found_exhibit = True
                        break
                        
        # Method 3: If still not found, try common exhibit naming patterns
        if not found_exhibit:
            date_no_dash = date_str.replace('-', '')
            common_patterns = [
                f"ex-991x{date_no_dash}x8k.htm",
                f"ex991x{date_no_dash}x8k.htm",
                f"ex-99_1x{date_no_dash}x8k.htm",
                f"ex991{date_no_dash}.htm", 
                f"exhibit991.htm",
                f"ex99-1.htm",
                f"ex991.htm",
                f"ex-99.1.htm",
                f"exhibit99_1.htm"
            ]
            
            for pattern in common_patterns:
                test_url = base_folder + pattern
                try:
                    test_res = requests.head(test_url, headers=headers, timeout=10)
                    if test_res.status_code == 200:
                        links.append((date_str, accession, test_url))
                        found_exhibit = True
                        break
                except:
                    continue
                    
    except Exception as e:
        continue
        
return links
```

def find_guidance_paragraphs(text):
“””
Extract paragraphs from text that are likely to contain guidance information.
Returns both the filtered paragraphs and a boolean indicating if any were found.
“””
# Define patterns to identify guidance sections
guidance_patterns = [
r’(?i)outlook’,
r’(?i)guidance’,
r’(?i)financial outlook’,
r’(?i)business outlook’,
r’(?i)forward[\s-]*looking’,
r’(?i)for (?:the )?(?:fiscal|next|coming|upcoming) (?:quarter|year)’,
r’(?i)(?:we|company) expect(?:s)?’,
r’(?i)revenue (?:is|to be) (?:in the range of|expected to|anticipated to)’,
r’(?i)to be (?:in the range of|approximately)’,
r’(?i)margin (?:is|to be) (?:expected|anticipated|forecast)’,
r’(?i)growth of (?:approximately|about)’,
r’(?i)for (?:fiscal|the fiscal)’,
r’(?i)next quarter’,
r’(?i)full year’,
r’(?i)current quarter’,
r’(?i)future quarter’,
r’(?i)Q[1-4]’
]

```
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
```

# This is Part 3B - Add this after Part 3A

# Main execution logic

if st.button(“🔍 Extract Guidance”):
if not api_key:
st.error(“Please enter your OpenAI API key.”)
elif not identifier:
st.error(“Please enter a ticker symbol or CIK code.”)
else:
# Handle CIK lookup or validation
if input_method == “Ticker Symbol”:
cik = lookup_cik(identifier)
if not cik:
st.error(“CIK not found for ticker.”)
st.stop()
ticker = identifier
else:
# For CIK input, validate and try to get ticker for display
if not cik or len(cik) != 10:
st.error(“Invalid CIK format. CIK should be 10 digits.”)
st.stop()
# Try to get ticker symbol for better display
found_ticker = get_ticker_from_cik(cik)
ticker = found_ticker if found_ticker else f”CIK-{cik}”

```
    # Get the selected model ID from the dropdown
    model_id = openai_models[selected_model]
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Display model information once at the beginning
    st.info(f"Using OpenAI model: {selected_model}")
    
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
                
                # Extract guidance from the highlighted text using the selected model
                table = extract_guidance(guidance_paragraphs, ticker, client, model_id)
            else:
                st.warning(f"⚠️ No guidance paragraphs found. Trying with a sample of the document.")
                # Use a sample of the document to reduce token usage
                sample_text = "DOCUMENT TYPE: SEC 8-K Earnings Release for " + ticker + "\n\n"
                paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', text)
                sample_text += "\n\n".join(paragraphs[:15])  # Just use first few paragraphs
                table = extract_guidance(sample_text, ticker, client, model_id)
            
            if table and "|" in table:
                rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                if len(rows) > 1:  # Check if we have header and at least one row of data
                    # Standardize the column names
                    column_names = [c.strip().lower().replace(' ', '_') for c in rows[0]]
                    
                    # Create DataFrame with standardized column names
                    df = pd.DataFrame(rows[1:], columns=column_names)
                    
                    # Format the numeric columns to display appropriate symbols
                    df = format_guidance_values(df)
                    
                    # Apply GAAP/non-GAAP split
                    if 'value_or_range' in df.columns:
                        df = split_gaap_non_gaap(df.rename(columns={'value_or_range': 'Value or range'}))
                        # Rename back to standard naming
                        if 'Value or range' in df.columns:
                            df.rename(columns={'Value or range': 'value_or_range'}, inplace=True)
                    
                    # Add metadata columns
                    df["filing_date"] = date_str
                    df["filing_url"] = url
                    df["model_used"] = selected_model
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
        # Combine all results
        combined = pd.concat(results, ignore_index=True)
        
        # Display human-friendly column names
        display_rename = {
            'metric': 'Metric',
            'value_or_range': 'Value or Range',
            'period': 'Period', 
            'period_type': 'Period Type',
            'low': 'Low',
            'high': 'High', 
            'average': 'Average',
            'filing_date': 'Filing Date',
            'filing_url': 'Filing URL', 
            'model_used': 'Model Used'
        }
        
        # Preview the table
        st.subheader("🔍 Preview of Extracted Guidance")
        
        # Select the most relevant columns for display
        display_cols = ['metric', 'value_or_range', 'period', 'period_type', 'low', 'high', 'average', 'filing_date']
        display_df = combined[display_cols] if all(col in combined.columns for col in display_cols) else combined
        
        # Rename columns for display
        display_df = display_df.rename(columns={c: display_rename.get(c, c) for c in display_df.columns})
        
        # Display the table with formatting
        st.dataframe(display_df, use_container_width=True)
        
        # Add download button
        import io
        excel_buffer = io.BytesIO()
        
        # Rename columns for Excel export
        excel_df = combined.rename(columns={c: display_rename.get(c, c) for c in combined.columns})
        excel_df.to_excel(excel_buffer, index=False)
        
        st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("No guidance data extracted.")
```
             

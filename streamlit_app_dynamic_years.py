import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os
import re

def format_percent(val):
    """Format a value as a percentage with consistent decimal places"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return f"{val:.1f}%"
    return val

def format_dollar(val):
    """Format a value as a dollar amount with consistent decimal places"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if abs(val) >= 100:
            return f"${val:.0f}"
        elif abs(val) >= 10:
            return f"${val:.1f}"
        else:
            return f"${val:.2f}"
    return val
    def extract_guidance(text, ticker, client, model_name):
    """Enhanced function to extract guidance from SEC filings"""
    prompt = f"""You are a financial analyst assistant. Extract ALL forward-looking guidance, projections, and outlook statements given in this earnings release for {ticker}.

Return a structured table containing the following columns:

- metric (e.g. Revenue, EPS, Operating Margin)
- value_or_range (e.g. $1.5B–$1.6B or $2.05 or $(0.05) to $0.10 - EXACTLY as it appears in the text)
- period (e.g. Q3 FY24, Full Year 2025)
- period_type (MUST be either "Quarter" or "Full Year" based on the period text)
- low (numeric low end of the range, or the single value if not a range)
- high (numeric high end of the range, or the single value if not a range)
- average (average of low and high, or just the value if not a range)

VERY IMPORTANT:
- Look for sections titled ‘Outlook’, ‘Guidance’, ‘Financial Outlook’, ‘Business Outlook’, or similar
- Also look for statements containing phrases like “expect”, “anticipate”, “forecast”, “will be”, “to be in the range of”
- Review the ENTIRE document for ANY forward-looking statements about future performance
- Pay special attention to sections describing “For the fiscal quarter”, “For the fiscal year”, “For next quarter”, etc.

CRITICAL GUIDANCE FOR THE NUMERIC COLUMNS:
- Only numeric values (no $ signs, % symbols, or text)
- Use negative numbers for negative values
- Convert billions to millions (×1000)
- Percentages: just numbers
- Dollar values: no $ signs

Respond in table format without commentary.

{text}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Error extracting guidance: {str(e)}")
        return None

def split_gaap_non_gaap(df):
    """Split rows that contain both GAAP and non-GAAP guidance into separate rows"""
    if 'value_or_range' not in df.columns or 'metric' not in df.columns:
        return df

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

def format_guidance_values(df):
    """Format the numeric values to appropriate formats based on the metric and value types"""
    formatted_df = df.copy()
    for idx, row in df.iterrows():
        value_text = str(row.get('value_or_range', ''))
        is_percentage = '%' in value_text
        is_dollar = '$' in value_text
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
                    continue
    return formatted_df

def is_cik_format(input_str):
    """Check if input looks like a CIK (10 digits)"""
    return input_str.strip().isdigit() and len(input_str.strip()) == 10

def get_ticker_from_cik(cik):
    """Get ticker symbol from CIK for display purposes"""
    try:
        headers = {'User-Agent': 'Your Name Contact@domain.com'}
        res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
        data = res.json()
        for entry in data.values():
            if str(entry["cik_str"]).zfill(10) == cik:
                return entry["ticker"].upper()
        return None
    except:
        return None

@st.cache_data(show_spinner=False)
def lookup_cik(ticker):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = res.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)

def get_fiscal_year_end(ticker, cik):
    """Get the fiscal year end month for a company from SEC data"""
    try:
        headers = {'User-Agent': 'Your Name Contact@domain.com'}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        if 'fiscalYearEnd' in data:
            fiscal_year_end = data['fiscalYearEnd']
            if len(fiscal_year_end) == 4:
                month = int(fiscal_year_end[:2])
                day = int(fiscal_year_end[2:])
                month_name = datetime(2000, month, 1).strftime('%B')
                st.success(f"Retrieved fiscal year end for {ticker}: {month_name} {day}")
                return month, day
        st.warning("Could not determine fiscal year end. Using December 31.")
        return 12, 31
    except Exception as e:
        st.warning(f"Error retrieving fiscal year end: {str(e)}. Using December 31.")
        return 12, 31

def generate_fiscal_quarters(fiscal_year_end_month):
    """Dynamically generate fiscal quarters based on the fiscal year end month"""
    fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
    quarters = {}
    current_month = fiscal_year_start_month
    for q in range(1, 5):
        start_month = current_month
        end_month = (start_month + 2) % 12
        if end_month == 0:
            end_month = 12
        quarters[q] = {'start_month': start_month, 'end_month': end_month}
        current_month = (end_month % 12) + 1
    return quarters

def get_fiscal_dates(ticker, quarter_num, year_num, fiscal_year_end_month, fiscal_year_end_day):
    """Calculate the appropriate date range for a fiscal quarter"""
    quarters = generate_fiscal_quarters(fiscal_year_end_month)
    if quarter_num < 1 or quarter_num > 4:
        st.error(f"Invalid quarter number: {quarter_num}. Must be 1-4.")
        return None
    quarter_info = quarters[quarter_num]
    start_month = quarter_info['start_month']
    end_month = quarter_info['end_month']
    spans_calendar_years = end_month < start_month

    if fiscal_year_end_month == 12:
        start_calendar_year = year_num
    else:
        fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
        if start_month >= fiscal_year_start_month:
            start_calendar_year = year_num - 1
        else:
            start_calendar_year = year_num

    end_calendar_year = start_calendar_year
    if spans_calendar_years:
        end_calendar_year = start_calendar_year + 1

    start_date = datetime(start_calendar_year, start_month, 1)
    if end_month == 2:
        end_day = 29 if (end_calendar_year % 4 == 0 and end_calendar_year % 100 != 0) or (end_calendar_year % 400 == 0) else 28
    elif end_month in [4, 6, 9, 11]:
        end_day = 30
    else:
        end_day = 31
    end_date = datetime(end_calendar_year, end_month, end_day)
    report_start = end_date + timedelta(days=15)
    report_end = report_start + timedelta(days=45)

    return {
        'quarter_period': f"Q{quarter_num} FY{year_num}",
        'start_date': start_date,
        'end_date': end_date,
        'report_start': report_start,
        'report_end': report_end,
        'period_description': f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}",
        'expected_report': f"~{report_start.strftime('%B %d, %Y')} to {report_end.strftime('%B %d, %Y')}"
    }

def get_accessions(cik, ticker, years_back=None, specific_quarter=None):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    filings = data["filings"]["recent"]
    accessions = []
    fiscal_year_end_month, fiscal_year_end_day = get_fiscal_year_end(ticker, cik)

    if years_back:
        cutoff = datetime.today() - timedelta(days=(365 * years_back) + 91.25)
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K" and datetime.strptime(date_str, "%Y-%m-%d") >= cutoff:
                accessions.append((accession, date_str))
    elif specific_quarter:
        match = re.search(r'(?:Q?(\d)Q?|Q(\d))(?:\s*FY\s*|\s*)?(\d{2}|\d{4})', specific_quarter.upper())
        if match:
            quarter = match.group(1) or match.group(2)
            year = match.group(3)
            if len(year) == 2:
                year = '20' + year
            fiscal_info = get_fiscal_dates(ticker, int(quarter), int(year), fiscal_year_end_month, fiscal_year_end_day)
            start_date = fiscal_info['report_start'] - timedelta(days=15)
            end_date = fiscal_info['report_end'] + timedelta(days=15)
            for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
                if form == "8-K":
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    if start_date <= date <= end_date:
                        accessions.append((accession, date_str))
    else:
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                accessions.append((accession, date_str))
                break
    return accessions

def get_ex99_1_links(cik, accessions):
    """Find exhibit 99.1 URLs from SEC index pages"""
    links = []
    headers = {'User-Agent': 'Your Name Contact@domain.com'}

    for accession, date_str in accessions:
        acc_no = accession.replace('-', '')
        base_folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/"
        index_url = base_folder + f"{accession}-index.htm"
        try:
            res = requests.get(index_url, headers=headers, timeout=30)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            for row in soup.find_all("tr"):
                row_text = row.get_text().lower()
                if "99.1" in row_text or "99.01" in row_text:
                    tds = row.find_all("td")
                    if len(tds) >= 3:
                        filename = tds[2].text.strip()
                        links.append((date_str, accession, base_folder + filename))
                        break
        except:
            continue
    return links

# Streamlit App Setup
st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("SEC 8-K Guidance Extractor")

ticker_or_cik = st.text_input("Enter Stock Ticker or CIK (e.g., MSFT or 0000789019)", "MSFT").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")

openai_models = {
    "GPT-4 Turbo": "gpt-4-turbo-preview",
    "GPT-4": "gpt-4",
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}
selected_model = st.selectbox("Select OpenAI Model", list(openai_models.keys()), index=0)

year_input = st.text_input("How many years back to search for 8-K filings? (Leave blank for most recent only)", "")
quarter_input = st.text_input("OR enter specific quarter (e.g., 2Q25, Q4FY24)", "")

if st.button("Extract Guidance"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        if is_cik_format(ticker_or_cik):
            cik = ticker_or_cik.strip()
            ticker = get_ticker_from_cik(cik) or f"CIK-{cik}"
        else:
            ticker = ticker_or_cik.strip()
            cik = lookup_cik(ticker)
            if not cik:
                st.error("CIK not found for ticker.")
                st.stop()

        st.info(f"Using ticker {ticker} (CIK: {cik})")
        model_id = openai_models[selected_model]
        client = OpenAI(api_key=api_key)

        if quarter_input.strip():
            accessions = get_accessions(cik, ticker, specific_quarter=quarter_input.strip())
        elif year_input.strip():
            try:
                years_back = int(year_input.strip())
                accessions = get_accessions(cik, ticker, years_back=years_back)
            except:
                st.error("Invalid year input. Must be a number.")
                accessions = []
        else:
            accessions = get_accessions(cik, ticker)

        links = get_ex99_1_links(cik, accessions)
        results = []

        for date_str, acc, url in links:
            st.write(f"Processing {url}")
            try:
                html = requests.get(url, headers={"User-Agent": "MyCompanyName Data Research Contact@mycompany.com"}).text
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(" ", strip=True)

                prompt_text = f"DOCUMENT TYPE: SEC 8-K Earnings Release for {ticker}\n\n{text[:4000]}"
                table = extract_guidance(prompt_text, ticker, client, model_id)

                if table and "|" in table:
                    rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                    if len(rows) > 1:
                        column_names = [c.strip().lower().replace(' ', '_') for c in rows[0]]
                        df = pd.DataFrame(rows[1:], columns=column_names)
                        df = format_guidance_values(df)

                        if 'value_or_range' in df.columns:
                            df = split_gaap_non_gaap(df)
                        
                        df["filing_date"] = date_str
                        df["filing_url"] = url
                        df["model_used"] = selected_model
                        results.append(df)
                        st.success("Guidance extracted.")
                    else:
                        st.warning("Table structure found, but no rows.")
                else:
                    st.warning("No guidance table found.")

            except Exception as e:
                st.warning(f"Error processing {url}: {str(e)}")

        if results:
            combined = pd.concat(results, ignore_index=True)
            display_cols = ['metric', 'value_or_range', 'period', 'period_type', 'low', 'high', 'average', 'filing_date']
            if all(col in combined.columns for col in display_cols):
                st.dataframe(combined[display_cols], use_container_width=True)
            else:
                st.dataframe(combined, use_container_width=True)

            import io
            buffer = io.BytesIO()
            combined.to_excel(buffer, index=False)
            st.download_button("Download as Excel", data=buffer.getvalue(), file_name=f"{ticker}_guidance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("No guidance data extracted.")

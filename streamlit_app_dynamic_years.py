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
    prompt = f"""You are a financial analyst assistant. Extract all forward-looking guidance given in this earnings release for {ticker}. 
Return a structured list containing:
- metric (e.g. Revenue, EPS, Operating Margin)
- value or range (e.g. $1.5B–$1.6B or $2.05)
- applicable period (e.g. Q3 FY24, Full Year 2025)

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
        return None


def extract_number(text):
    # This is a simplified helper function to extract numeric values
    if not text:
        return None
    clean_text = text.replace("$", "").replace(",", "")
    try:
        if "B" in clean_text.upper():
            return float(clean_text.upper().replace("B", "")) * 1000000000
        elif "M" in clean_text.upper():
            return float(clean_text.upper().replace("M", "")) * 1000000
        else:
            return float(clean_text)
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
    percent_match = re.search(r'(?:approximately|about|around|roughly|~|circa)?\s*(\d+\.?\d*)%', text, re.IGNORECASE)
    if percent_match:
        percent_value = float(percent_match.group(1))
        return percent_value, percent_value, percent_value
    
    # Case 3: Range values like "$1.5B-$1.6B"
    range_match = re.search(r'[$]?([\d\.]+[KMB]?)(?:[ ]*[-–—~][ ]*|\s+to\s+)[$]?([\d\.]+[KMB]?)', text, re.IGNORECASE)
    if range_match:
        low = extract_number(range_match.group(1))
        high = extract_number(range_match.group(2))
        if low is not None and high is not None:
            return low, high, (low + high) / 2
    
    # Case 4: Single values like "$1.5B"
    single_match = re.search(r'[$]?([\d\.]+[KMB]?)(?:\s|$)', text, re.IGNORECASE)
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
                    forw_idx = text.lower().find("forward looking statements")
                    if forw_idx != -1:
                        text = text[:forw_idx]
                    
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
                            # Create dataframe from rows
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
                                
                            if 'Period' not in df.columns and len(df.columns) > 2:
                                df['Period'] = df.iloc[:, 2]
                            
                            # Parse values to get Low, High, Average
                            parsed_values = df["Value"].apply(parse_value_range)
                            
                            # Add new columns
                            df["Low"] = [v[0] if isinstance(v[0], (int, float)) else None for v in parsed_values]
                            df["High"] = [v[1] if isinstance(v[1], (int, float)) else None for v in parsed_values]
                            
                            # For Average: use calculated average if available, otherwise use the text value
                            df["Average"] = [
                                v[2] if isinstance(v[2], (int, float)) else 
                                (v[2] if isinstance(v[2], str) else None) 
                                for v in parsed_values
                            ]
                            
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
                        else:
                            st.warning("⚠️ Skipped, no valid table structure found in the response.")
                    else:
                        st.warning("⚠️ Skipped, no guidance found in filing.")
                except Exception as e:
                    st.warning(f"Could not process: {url}")
                    st.error(f"Error: {str(e)}")
                    st.expander("Debug Information").code(traceback.format_exc())

            if results:
                combined = pd.concat(results, ignore_index=True)
                
                # Display the table in the app
                st.subheader("Extracted Guidance")
                st.dataframe(combined)
                
                # Provide download options
                import io
                
                # Excel download
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                st.download_button(
                    "📥 Download Excel", 
                    data=excel_buffer.getvalue(), 
                    file_name=f"{ticker}_guidance_output.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # CSV download
                csv_buffer = io.BytesIO()
                combined.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"{ticker}_guidance_output.csv",
                    mime="text/csv",
                    key="csv-download"
                )
            else:
                st.warning("No guidance data extracted.")

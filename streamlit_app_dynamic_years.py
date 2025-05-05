import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os
import re

st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("📄 SEC 8-K Guidance Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")
year_input = st.text_input("How many years back to search for 8-K filings? (Leave blank for most recent only)", "")


@st.cache_data(show_spinner=False)
def lookup_cik(ticker):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = res.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)


def get_accessions(cik, years_back):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
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

def parse_value_range(value_str):
    """Parse value ranges to extract low, high and average values"""
    if not value_str or not isinstance(value_str, str):
        return None, None, None
        
    # Clean up the input string
    value_str = value_str.strip()
    
    # Initialize values
    low, high, avg = None, None, None
    
    # Function to convert string to float, handling common financial formats
    def to_float(s):
        if not s:
            return None
        # Remove $ and commas
        s = s.replace('

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
                    table = extract_guidance(text, ticker, client)
                    if table and "|" in table:
                        # Handle tables with or without markdown formatting
                        rows = []
                        for r in table.strip().split("\n"):
                            if "|" in r:
                                # Skip separator rows that consist of dashes
                                if re.match(r'^[\|\s\-]+
                        
                        # Parse values to get Low, High, and Average
                        parsed_values = df["Value"].apply(parse_value_range)
                        
                        # Create new columns from the parsed values
                        df["Low"] = [v[0] if isinstance(v[0], (int, float)) else None for v in parsed_values]
                        df["High"] = [v[1] if isinstance(v[1], (int, float)) else None for v in parsed_values]
                        df["Average"] = [v[2] if isinstance(v[2], (int, float)) else None for v in parsed_values]
                        
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
                        st.warning("⚠️ Skipped, no guidance found in filing.")
                except Exception as e:
                    st.warning(f"Could not process: {url}")
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.expander("Debug Information").code(traceback.format_exc())

            if results:
                combined = pd.concat(results, ignore_index=True)
                # Display the table in the app
                st.subheader("Extracted Guidance")
                st.dataframe(combined)
                
                # Provide download option
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                # Also provide CSV download option
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
, '').replace(',', '')
        # Handle 'B' for billions and 'M' for millions
        if 'B' in s.upper():
            s = s.upper().replace('B', '')
            return float(s) * 1000000000
        elif 'M' in s.upper():
            s = s.upper().replace('M', '')
            return float(s) * 1000000
        else:
            return float(s)
    
    # Try to find a range with various separators
    # This simpler pattern avoids character class issues
    range_pattern = r'[$]?([\d\.]+[KMB]?)(?:[ ]*[-–—~][ ]*|\s+to\s+)[$]?([\d\.]+[KMB]?)'
    match = re.search(range_pattern, value_str, re.IGNORECASE)
    
    if match:
        try:
            low = to_float(match.group(1))
            high = to_float(match.group(2))
            if low is not None and high is not None:
                avg = (low + high) / 2
                return low, high, avg
        except (ValueError, IndexError):
            pass
    
    # Try to find a single value
    single_value_pattern = r'[$]?([\d\.]+[KMB]?)(?:\s|$)'
    match = re.search(single_value_pattern, value_str, re.IGNORECASE)
    if match:
        try:
            val = to_float(match.group(1))
            if val is not None:
                return val, val, val  # Same value for low, high, avg
        except (ValueError, IndexError):
            pass
            
    # If we get here, we couldn't parse the value
    return value_str, None, None

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
                    table = extract_guidance(text, ticker, client)
                    if table and "|" in table:
                        rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                        df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                        
                        # Add the new columns
                        df["Value"] = df.get("value or range", df.get("Value", ""))
                        df["Metric"] = df.get("metric", df.get("Metric", ""))
                        df["Period"] = df.get("applicable period", df.get("Period", ""))
                        
                        # Parse values to get Low, High, and Average
                        parsed_values = df["Value"].apply(parse_value_range)
                        
                        # Create new columns from the parsed values
                        df["Low"] = [v[0] if isinstance(v[0], (int, float)) else None for v in parsed_values]
                        df["High"] = [v[1] if isinstance(v[1], (int, float)) else None for v in parsed_values]
                        df["Average"] = [v[2] if isinstance(v[2], (int, float)) else None for v in parsed_values]
                        
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
                        st.warning("⚠️ Skipped, no guidance found in filing.")
                except Exception as e:
                    st.warning(f"Could not process: {url}")
                    st.error(f"Error: {str(e)}")

            if results:
                combined = pd.concat(results, ignore_index=True)
                # Display the table in the app
                st.subheader("Extracted Guidance")
                st.dataframe(combined)
                
                # Provide download option
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                # Also provide CSV download option
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
, r):
                                    continue
                                # Extract cells, handling both standard and markdown table formats
                                cells = r.strip().split("|")
                                # Remove empty cells at start/end if present
                                if cells and not cells[0].strip():
                                    cells = cells[1:]
                                if cells and not cells[-1].strip():
                                    cells = cells[:-1]
                                rows.append([cell.strip() for cell in cells])
                        
                        # Ensure we have at least headers and one data row
                        if len(rows) >= 2:
                            df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                            
                            # Normalize column names to handle various formats from the API
                            column_mapping = {
                                'metric': 'Metric',
                                'value or range': 'Value',
                                'applicable period': 'Period',
                                'period': 'Period',
                                'value': 'Value'
                            }
                            
                            # Rename columns if they exist in the dataframe
                            df = df.rename(columns={k: v for k, v in column_mapping.items() 
                                                  if k in df.columns})
                            
                            # Ensure we have the essential columns with fallbacks
                            if 'Metric' not in df.columns and len(df.columns) > 0:
                                df['Metric'] = df.iloc[:, 0]  # Use first column as Metric
                            
                            if 'Value' not in df.columns and len(df.columns) > 1:
                                df['Value'] = df.iloc[:, 1]  # Use second column as Value
                                
                            if 'Period' not in df.columns and len(df.columns) > 2:
                                df['Period'] = df.iloc[:, 2]  # Use third column as Period
                        
                        # Parse values to get Low, High, and Average
                        parsed_values = df["Value"].apply(parse_value_range)
                        
                        # Create new columns from the parsed values
                        df["Low"] = [v[0] if isinstance(v[0], (int, float)) else None for v in parsed_values]
                        df["High"] = [v[1] if isinstance(v[1], (int, float)) else None for v in parsed_values]
                        df["Average"] = [v[2] if isinstance(v[2], (int, float)) else None for v in parsed_values]
                        
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
                        st.warning("⚠️ Skipped, no guidance found in filing.")
                except Exception as e:
                    st.warning(f"Could not process: {url}")
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.expander("Debug Information").code(traceback.format_exc())

            if results:
                combined = pd.concat(results, ignore_index=True)
                # Display the table in the app
                st.subheader("Extracted Guidance")
                st.dataframe(combined)
                
                # Provide download option
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                # Also provide CSV download option
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
, '').replace(',', '')
        # Handle 'B' for billions and 'M' for millions
        if 'B' in s.upper():
            s = s.upper().replace('B', '')
            return float(s) * 1000000000
        elif 'M' in s.upper():
            s = s.upper().replace('M', '')
            return float(s) * 1000000
        else:
            return float(s)
    
    # Try to find a range with various separators
    # This simpler pattern avoids character class issues
    range_pattern = r'[$]?([\d\.]+[KMB]?)(?:[ ]*[-–—~][ ]*|\s+to\s+)[$]?([\d\.]+[KMB]?)'
    match = re.search(range_pattern, value_str, re.IGNORECASE)
    
    if match:
        try:
            low = to_float(match.group(1))
            high = to_float(match.group(2))
            if low is not None and high is not None:
                avg = (low + high) / 2
                return low, high, avg
        except (ValueError, IndexError):
            pass
    
    # Try to find a single value
    single_value_pattern = r'[$]?([\d\.]+[KMB]?)(?:\s|$)'
    match = re.search(single_value_pattern, value_str, re.IGNORECASE)
    if match:
        try:
            val = to_float(match.group(1))
            if val is not None:
                return val, val, val  # Same value for low, high, avg
        except (ValueError, IndexError):
            pass
            
    # If we get here, we couldn't parse the value
    return value_str, None, None

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
                    table = extract_guidance(text, ticker, client)
                    if table and "|" in table:
                        rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                        df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                        
                        # Add the new columns
                        df["Value"] = df.get("value or range", df.get("Value", ""))
                        df["Metric"] = df.get("metric", df.get("Metric", ""))
                        df["Period"] = df.get("applicable period", df.get("Period", ""))
                        
                        # Parse values to get Low, High, and Average
                        parsed_values = df["Value"].apply(parse_value_range)
                        
                        # Create new columns from the parsed values
                        df["Low"] = [v[0] if isinstance(v[0], (int, float)) else None for v in parsed_values]
                        df["High"] = [v[1] if isinstance(v[1], (int, float)) else None for v in parsed_values]
                        df["Average"] = [v[2] if isinstance(v[2], (int, float)) else None for v in parsed_values]
                        
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
                        st.warning("⚠️ Skipped, no guidance found in filing.")
                except Exception as e:
                    st.warning(f"Could not process: {url}")
                    st.error(f"Error: {str(e)}")

            if results:
                combined = pd.concat(results, ignore_index=True)
                # Display the table in the app
                st.subheader("Extracted Guidance")
                st.dataframe(combined)
                
                # Provide download option
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                # Also provide CSV download option
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

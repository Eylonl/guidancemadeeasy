import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os
import io

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
    return None


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

def extract_guidance(text, ticker, date_str, client, max_chars=30000):
    # Truncate text if too long
    if len(text) > max_chars:
        truncated_text = text[:max_chars]
        note = "\n[Note: Text truncated due to length]"
    else:
        truncated_text = text
        note = ""
        
    # Look for guidance-specific sections
    guidance_keywords = [
        "outlook", "guidance", "forecast", "financial outlook", 
        "fiscal", "expects", "projected", "estimates", "anticipated",
        "expected", "future", "next quarter", "next fiscal"
    ]
    
    # Extract guidance sections
    guidance_section = ""
    for keyword in guidance_keywords:
        keyword_idx = truncated_text.lower().find(keyword)
        if keyword_idx != -1:
            # Get text around the keyword (1000 chars before and after)
            start_idx = max(0, keyword_idx - 1000)
            end_idx = min(len(truncated_text), keyword_idx + 1000)
            section = truncated_text[start_idx:end_idx]
            guidance_section += section + "\n\n"
    
    # If guidance sections found, prioritize them
    if guidance_section:
        processed_text = guidance_section + truncated_text[:10000]  # Add some context
    else:
        processed_text = truncated_text
    
    prompt = f"""You are a financial analyst assistant. Extract all forward-looking guidance given in this earnings release for {ticker} from the filing dated {date_str}. 

Return a structured list containing:
- metric (e.g. Revenue, EPS, Operating Margin)
- value or range (exactly as stated in the document)
- for range values, include separate columns for:
  - low (lower bound of the range)
  - high (upper bound of the range)
  - average (mathematical average of low and high)
- applicable period (e.g. Q3 FY24, Full Year 2025)

CRITICAL INSTRUCTIONS:
1. Group together BOTH quarterly AND full year guidance from this filing
2. For any range values (e.g. "$1.5B-$1.6B"), calculate and include the low, high, and average values
3. If you run into memory limitations, focus ONLY on extracting the guidance - skip any explanatory text
4. If you encounter memory errors, provide only the essential data

Respond in table format without commentary, using this format:
| Metric | Value | Low | High | Average | Period |
| ------ | ----- | --- | ---- | ------- | ------ |
| Revenue | $1.5B-$1.6B | $1.5B | $1.6B | $1.55B | Q3 FY24 |
| Revenue | $6.2B-$6.3B | $6.2B | $6.3B | $6.25B | Full Year 2024 |
| EPS | $0.57 | $0.57 | $0.57 | $0.57 | Q3 FY24 |
| EPS | $2.05-$2.10 | $2.05 | $2.10 | $2.075 | Full Year 2024 |

Document Text:{note}
{processed_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"⚠️ API Error: {str(e)}")
        return None


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
                    years_back = int(year_input.strip())
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
                    
                    table = extract_guidance(text, ticker, date_str, client)
                    
                    if table and "|" in table:
                        rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r and r.strip()]
                        if len(rows) > 1:  # Ensure we have header and at least one data row
                            columns = [c.strip() for c in rows[0]]
                            
                            # Handle if GPT doesn't return all expected columns
                            expected_columns = ["Metric", "Value", "Low", "High", "Average", "Period"]
                            missing_columns = [col for col in expected_columns if col not in columns]
                            
                            data_rows = rows[1:]
                            if data_rows:
                                # Ensure all rows have the expected number of columns
                                normalized_rows = []
                                for row in data_rows:
                                    # Skip empty rows or malformed data
                                    if not row or len(row) < 2:
                                        continue
                                        
                                    # If we have fewer columns than expected, pad with empty values
                                    if len(row) < len(columns):
                                        row = row + [""] * (len(columns) - len(row))
                                    # If we have more columns than expected, truncate
                                    elif len(row) > len(columns):
                                        row = row[:len(columns)]
                                    
                                    normalized_rows.append(row)
                                
                                if normalized_rows:
                                    df = pd.DataFrame(normalized_rows, columns=columns)
                                    
                                    # Add missing columns if any
                                    for col in missing_columns:
                                        df[col] = ""
                                    
                                    # Add metadata columns
                                    df["FilingDate"] = date_str
                                    df["8K_Link"] = url
                                    
                                    results.append(df)
                                    st.success("✅ Guidance extracted from this 8-K.")
                                else:
                                    st.warning("⚠️ No valid guidance data found after processing.")
                            else:
                                st.warning("⚠️ No guidance data rows found in the response.")
                        else:
                            st.warning("⚠️ Insufficient data in the response to create a dataframe.")
                    else:
                        st.warning("⚠️ No guidance found in filing or response format invalid.")
                except Exception as e:
                    st.warning(f"Could not process: {url}. Error: {str(e)}")

            if results:
                try:
                    # Ensure all dataframes have the same columns before concatenation
                    all_columns = set()
                    for df in results:
                        all_columns.update(df.columns)
                    
                    for i, df in enumerate(results):
                        for col in all_columns:
                            if col not in df.columns:
                                results[i][col] = ""
                    
                    combined = pd.concat(results, ignore_index=True)
                    
                    # Reorganize columns in a logical order
                    essential_columns = ["Metric", "Value", "Low", "High", "Average", "Period", "FilingDate", "8K_Link"]
                    other_columns = [col for col in combined.columns if col not in essential_columns]
                    final_columns = essential_columns + other_columns
                    
                    # Only keep columns that exist in the dataframe
                    final_columns = [col for col in final_columns if col in combined.columns]
                    combined = combined[final_columns]
                    
                    # Create Excel buffer for download
                    excel_buffer = io.BytesIO()
                    combined.to_excel(excel_buffer, index=False)
                    
                    # Offer download button
                    st.download_button(
                        "📥 Download Excel", 
                        data=excel_buffer.getvalue(), 
                        file_name=f"{ticker}_guidance_output.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Display the data in the app
                    st.write("### Extracted Guidance Data")
                    st.dataframe(combined)
                except Exception as e:
                    st.error(f"Error creating combined Excel file: {str(e)}")
            else:
                st.warning("No guidance data extracted from any filings.")
                
            if len(links) > 1:
                st.info("📝 **Tip:** If guidance extraction failed for some filings, try again with fewer years or focus on the most recent filing.")

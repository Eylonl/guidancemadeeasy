
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os

st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("📄 SEC 8-K Guidance Extractor")

# User inputs
ticker = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
years_back = st.number_input("How many years back to search for 8-K filings?", min_value=1, max_value=10, value=1)
api_key = st.text_input("Enter OpenAI API Key", type="password")
output_dir = "./"

@st.cache_data(show_spinner=False)
def lookup_cik(ticker):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = res.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None

def get_accessions_years_back(cik, years):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    filings = data["filings"]["recent"]
    accessions = []
    cutoff_date = datetime.today() - timedelta(days=365 * years)

    for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
        if form == "8-K":
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date >= cutoff_date:
                accessions.append((accession, date_str))
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

Respond in table format without commentary.\n\n{text}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
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
            accessions = get_accessions_years_back(cik, years_back)
            links = get_ex99_1_links(cik, accessions)
            results = []

            for date_str, acc, url in links:
                st.write(f"📄 Processing {url}")
                try:
                    html = requests.get(url).text
                    text = BeautifulSoup(html, "html.parser").get_text()
                    table = extract_guidance(text, ticker, client)
                    if table and "|" in table:
                        rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                        df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                        df["FilingDate"] = date_str
                        df["8K_Link"] = url
                        results.append(df)
                except:
                    st.warning(f"Could not process: {url}")

            if results:
                combined = pd.concat(results, ignore_index=True)
                excel_path = os.path.join(output_dir, f"{ticker}_guidance_1yr.xlsx")
                combined.to_excel(excel_path, index=False)
                with open(excel_path, "rb") as f:
                    st.download_button("📥 Download Excel", f, file_name=f"{ticker}_guidance_1yr.xlsx")
            else:
                st.warning("No guidance data extracted.")

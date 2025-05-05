
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os

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
            if year_input.strip():
                years_back = int(year_input.strip())
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
                        df["FilingDate"] = date_str
                        df["8K_Link"] = url
                        results.append(df)
                        st.success("✅ Guidance extracted from this 8-K.")
                except:
                    st.warning(f"Could not process: {url}")

            if results:
                combined = pd.concat(results, ignore_index=True)
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No guidance data extracted.")

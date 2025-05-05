import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import re
from openai import OpenAI
from utils import lookup_cik, get_accessions, get_ex99_1_links, extract_guidance, parse_value_range, split_gaap_non_gaap

st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("📄 SEC 8-K Guidance Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., TEAM)", "TEAM").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")
year_input = st.text_input("Enter period (e.g., 3Q25) or number of years to look back", "")

if st.button("🔍 Extract Guidance"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        cik = lookup_cik(ticker)
        if not cik:
            st.error("CIK not found for ticker.")
        else:
            client = OpenAI(api_key=api_key)
            accessions = []
            if year_input.strip():
                if re.match(r'\dQ\d{2,4}', year_input.upper()):
                    target = year_input.upper()
                    all_recent = get_accessions(cik, 10)
                    for acc, date_str in all_recent:
                        links = get_ex99_1_links(cik, [(acc, date_str)])
                        for _, _, url in links:
                            try:
                                html = requests.get(url, headers={"User-Agent": "MyCompanyBot Contact@example.com"}).text
                                text = BeautifulSoup(html, "html.parser").get_text()
                                norm_text = text.upper().replace("FISCAL YEAR ", "FY").replace("QUARTER", "Q")
                                if target in norm_text:
                                    accessions.append((acc, date_str))
                            except:
                                continue
                    if not accessions:
                        st.warning(f"No filings found mentioning {target} in document text.")
                else:
                    try:
                        years_back = int(year_input)
                        accessions = get_accessions(cik, years_back)
                    except:
                        st.error("Invalid year input. Must be a number or quarter (e.g., 1Q25).")
            else:
                accessions = get_most_recent_accession(cik)

            links = get_ex99_1_links(cik, accessions)
            results = []
            for date_str, acc, url in links:
                st.write(f"📄 Processing {url}")
                try:
                    html = requests.get(url, headers={"User-Agent": "MyCompanyBot Contact@example.com"}).text
                    text = BeautifulSoup(html, "html.parser").get_text()
                    forw_idx = text.lower().find("forward looking statements")
                    if forw_idx != -1:
                        text = text[:forw_idx]
                    table = extract_guidance(text, ticker, client)
                    if table and "|" in table:
                        rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                        df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                        df[['Low', 'High', 'Average']] = df['Value'].apply(lambda v: pd.Series(parse_value_range(v)))
                        df = split_gaap_non_gaap(df)
                        df['FilingDate'] = date_str
                        df['8K_Link'] = url
                        results.append(df)
                        st.success("✅ Guidance extracted from this 8-K.")
                    else:
                        st.warning("⚠️ Skipped, no guidance found in filing.")
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

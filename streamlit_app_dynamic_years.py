def extract_guidance(text, ticker, client):
    """
    Improved guidance extraction function that's fully dynamic and works for any company.
    """
    prompt = f"""You are a financial analyst assistant. Extract ALL forward-looking guidance, projections, and outlook statements given in this earnings release for {ticker}. 

Return a structured table containing:
- Metric (e.g. Revenue, EPS, Operating Margin, Growth Rate, etc.)
- Value or range (e.g. $1.5B–$1.6B, $2.05, 5% to 7%, etc.)
- Applicable period (e.g. Q2, Next Quarter, Full Year, etc.)

VERY IMPORTANT INSTRUCTIONS:
1. Look for "OUTLOOK" OR "GUIDANCE" SECTIONS that provide future financial projections
2. Pay attention to statements containing phrases like:
   - "For the fiscal" 
   - "For the next"
   - "For fiscal"
   - "expect"
   - "anticipate"
   - "outlook"
   - "guidance"
   - "forecast"
   - "revenue to be in the range of"
   - "to be in the range of"
   - "operating margin to be"
   - "growth of"
   - "expected to be"

3. Look for common financial metrics:
   - Revenue (including ranges and growth percentages)
   - Subscription Revenue
   - Cloud Revenue
   - Operating Margin
   - Net Income Per Share
   - EPS (Earnings Per Share)
   - Free Cash Flow
   - Adjusted Free Cash Flow
   - Gross Margin
   - Operating Income
   - Adjusted EBITDA

4. Include BOTH quarterly guidance AND full year guidance
5. For any percentage values, always include the % symbol in your output (e.g., "5% to 7%" or "5%-7%")
6. Be sure to capture year-over-year growth metrics
7. If guidance includes both GAAP and non-GAAP measures, include both with clear labels
8. If numbers are in thousands, millions, or billions, preserve that notation (e.g., "$100M" or "$1.2B")
9. DO NOT MISS ANY GUIDANCE INFORMATION - this is critical financial data
10. Respond ONLY with the table format with no additional commentary

Example table format:
| Metric | Value | Period |
|--------|-------|--------|
| Revenue | $267M-$268M | Q2 |
| Revenue Growth | 19% | Q2 |
| Revenue | $1.1B-$1.11B | Full Year |
| Revenue Growth | 19%-20% | Full Year |
| Operating Margin (Non-GAAP) | 5% | Q2 |
| EPS (Non-GAAP) | $0.08-$0.09 | Q2 |
| Operating Margin (Non-GAAP) | 6% | Full Year |
| EPS (Non-GAAP) | $0.36 | Full Year |

Now extract ALL guidance from this text:\n\n{text}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more consistent extraction
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"⚠️ Error extracting guidance: {str(e)}")
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


def process_8k_links(links, ticker, client):
    """
    Process 8-K links to extract guidance information using a dynamic approach.
    This function contains the improved text processing logic.
    """
    results = []
    
    for date_str, acc, url in links:
        st.write(f"📄 Processing {url}")
        try:
            headers = {"User-Agent": "MyCompanyName Data Research Contact@mycompany.com"}
            html = requests.get(url, headers=headers).text
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract text while preserving some structure
            text = soup.get_text(" ", strip=True)
            
            # Use regex patterns to identify potential guidance sections
            # These patterns are common across different companies' earnings releases
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
            
            # Find paragraphs containing guidance patterns
            paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', text)
            guidance_paragraphs = []
            
            for i, para in enumerate(paragraphs):
                if any(re.search(pattern, para) for pattern in guidance_patterns):
                    # Check if it's likely a forward-looking statement (not a disclaimer)
                    if not (re.search(r'(?i)safe harbor', para) or 
                            (re.search(r'(?i)forward-looking statements', para) and 
                            re.search(r'(?i)risks', para))):
                        guidance_paragraphs.append(para)
            
            # Check if we found any guidance paragraphs
            if guidance_paragraphs:
                st.success(f"✅ Found potential guidance information in {len(guidance_paragraphs)} paragraphs.")
                
                # Create a highlighted version of the text with guidance sections at the beginning
                highlighted_text = "POTENTIAL GUIDANCE SECTIONS:\n\n" + "\n\n".join(guidance_paragraphs) + "\n\n--- FULL DOCUMENT BELOW ---\n\n" + text
                
                # Now extract guidance from the highlighted text
                table = extract_guidance(highlighted_text, ticker, client)
                
                if table and "|" in table:
                    # Process the table as before
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
                        
                        # Show the identified guidance paragraphs
                        st.write("Potential guidance paragraphs found but not successfully extracted as a table:")
                        for i, para in enumerate(guidance_paragraphs[:3]):  # Show up to 3 paragraphs
                            st.write(f"Paragraph {i+1}:")
                            st.text(para[:300] + "..." if len(para) > 300 else para)
                else:
                    st.warning(f"⚠️ No guidance table found in {url}")
                    
                    # Show the identified guidance paragraphs
                    st.write("Potential guidance paragraphs were found but not successfully extracted as a table:")
                    for i, para in enumerate(guidance_paragraphs[:3]):  # Show up to 3 paragraphs
                        st.write(f"Paragraph {i+1}:")
                        st.text(para[:300] + "..." if len(para) > 300 else para)
            else:
                st.warning(f"⚠️ No guidance paragraphs found in {url}")
                
                # Show a sample of the text to help debug
                sample_length = min(500, len(text))
                st.write(f"Sample of document text (first {sample_length} characters):")
                st.text(text[:sample_length] + "...")
                
                # Try with the full text anyway as a fallback
                st.write("Attempting to extract guidance from the full document as a fallback...")
                table = extract_guidance(text, ticker, client)
                
                if table and "|" in table:
                    # Process the table as before
                    rows = [r.strip().split("|")[1:-1] for r in table.strip().split("\n") if "|" in r]
                    if len(rows) > 1:  # Check if we have header and at least one row of data
                        df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                        
                        # Continue processing as above...
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
                        st.success("✅ Guidance extracted from this 8-K using fallback method.")
                    else:
                        st.warning(f"⚠️ Table format was detected but no data rows were found in fallback attempt")
                else:
                    st.warning(f"⚠️ No guidance table found in fallback attempt")
        except Exception as e:
            st.warning(f"Could not process: {url}. Error: {str(e)}")
            st.error(f"Exception details: {type(e).__name__}: {str(e)}")
    
    return results


# Add this to the "if st.button" section, replacing the loop that processes links
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
            
            # Use our improved processing function
            results = process_8k_links(links, ticker, client)

            if results:
                combined = pd.concat(results, ignore_index=True)
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No guidance data extracted.")

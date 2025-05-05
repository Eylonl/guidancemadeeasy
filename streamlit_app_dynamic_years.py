# This code should replace the existing code in the loop that processes each link
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

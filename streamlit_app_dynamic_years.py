import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd
import os
import re

# ─── NUMBER & RANGE PARSING HELPERS ───────────────────────────────────────────
number_token = r'[-+]?\d[\d,\.]*\s*(?:[KMB]|million|billion)?'

def extract_number(token: str):
    """
    Enhanced function to extract numeric values from text, with improved handling of 
    both explicit negative signs and parenthetical negative notation.
    """
    if not token or not isinstance(token, str):
        return None
    
    # Store original token for reference
    original_token = token.strip()
    
    # Check if the token is a negative value with a leading - sign
    is_negative_sign = original_token.startswith('-')
    
    # Check for common financial notation of parentheses to indicate negative
    # Look for patterns like (0.05) or ( 0.05 ) - with optional whitespace
    is_parenthetical_negative = (
        (original_token.startswith('(') and original_token.endswith(')')) or
        re.match(r'^\(\s*\$?\s*\d+(?:\.\d+)?\s*\)$', original_token)
    )
    
    # Determine if the value should be treated as negative
    is_negative = is_negative_sign or is_parenthetical_negative
    
    # Remove parentheses, dollar signs, and commas for numerical processing
    tok = original_token.replace('(', '').replace(')', '').replace('$','') \
                        .replace(',', '').replace('-', '').replace('+', '').strip().lower()

# Convert billions to millions (multiply by 1000)
    factor = 1.0
    if 'billion' in tok or tok.endswith('b'):
        if 'billion' in tok:
            tok = tok.replace('billion', '').strip()
        elif tok.endswith('b'):
            tok = tok[:-1].strip()
        factor = 1000.0  # Convert to millions
    elif 'million' in tok or tok.endswith('m'):
        if 'million' in tok:
            tok = tok.replace('million', '').strip()
        elif tok.endswith('m'):
            tok = tok[:-1].strip()
        factor = 1.0
    elif tok.endswith('k'):
        tok = tok[:-1].strip()
        factor = 0.001
    
    try:
        val = float(tok) * factor
        return -val if is_negative else val
    except:
        return None


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
        # Use different formatting based on value magnitude
        if abs(val) >= 100:  # Large values, use whole numbers
            return f"${val:.0f}"
        elif abs(val) >= 10:  # Medium values, use 1 decimal place
            return f"${val:.1f}"
        else:  # Small values, use 2 decimal places
            return f"${val:.2f}"
    return val

def parse_value_range(text: str):
    """
    Enhanced function to parse value ranges from guidance text.
    Properly handles parenthetical negative notation throughout.
    """
    if not isinstance(text, str):
        return (None, None, None)
    
    # Check for "flat" or "unchanged"
    if re.search(r'\b(flat|unchanged)\b', text, re.I):
        return (0.0, 0.0, 0.0)
    
    # Detect patterns that clearly indicate negative values
    contains_negative_indicator = (
        '(' in text and ')' in text or  # Parenthetical notation
        '-' in text or                  # Explicit minus sign
        re.search(r'\b(?:loss|decrease|down|negative|deficit)\b', text, re.I)  # Negative words
    )
    
    # Handle negative values in parentheses (common financial notation)
    # Convert formats like "$(0.05)" or "(0.05)" to "-$0.05" or "-0.05"
    text = re.sub(r'\$\s*\(\s*(\d+(?:\.\d+)?)\s*\)', r'-$\1', text)
    text = re.sub(r'\(\s*(\d+(?:\.\d+)?)\s*\)%', r'-\1%', text)
    text = re.sub(r'\(\s*\$?\s*(\d+(?:\.\d+)?)\s*\)', r'-\1', text)
    
    # Additional handling for percentage cases with parentheses
    text = re.sub(r'\(\s*(\d+(?:\.\d+)?)\s*%\s*\)', r'-\1%', text)  # Replace (5%) with -5%
    text = re.sub(r'~\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)', r'-\1%', text)  # Replace ~(5%) with -5%
    
    # Handle "decrease of X%" pattern
    decrease_match = re.search(r'decrease\s+of\s+([0-9\.]+)%', text, re.I)
    if decrease_match:
        val = float(decrease_match.group(1))
        return (-val, -val, -val)
    
    # First look for precise ranges with "to" between values
    # 1. Dollar to dollar with amount qualifier: "$181 million to $183 million"
    dollar_to_full_range = re.search(r'(-?\$\s*\d+(?:,\d+)?(?:\.\d+)?)\s*(?:million|billion|M|B)?\s*(?:to|[-–—~])\s*(-?\$\s*\d+(?:,\d+)?(?:\.\d+)?)\s*(?:million|billion|M|B)?', text, re.I)
    if dollar_to_full_range:
        lo_text = dollar_to_full_range.group(1)
        hi_text = dollar_to_full_range.group(2)
        
        # Extract any unit qualifier (million/billion) that might come after the range
        unit_match = re.search(r'\b(million|billion|M|B)\b', text, re.I)
        unit = unit_match.group(1).lower() if unit_match else None
        
        # If unit exists, append it to each value for proper parsing
        if unit and not re.search(r'\b(million|billion|M|B)\b', lo_text, re.I):
            lo_text = f"{lo_text} {unit}"
        if unit and not re.search(r'\b(million|billion|M|B)\b', hi_text, re.I):
            hi_text = f"{hi_text} {unit}"
        
        lo = extract_number(lo_text)
        hi = extract_number(hi_text)
        avg = (lo + hi) / 2 if lo is not None and hi is not None else None
        return (lo, hi, avg)

    # 2. Dollar to dollar: "$0.08 to $0.09" or "-$0.08 to -$0.06"
    dollar_to_range = re.search(r'(-?\$\s*\d+(?:,\d+)?(?:\.\d+)?)\s*(?:to|[-–—~])\s*(-?\$\s*\d+(?:,\d+)?(?:\.\d+)?)', text, re.I)
    if dollar_to_range:
        lo = extract_number(dollar_to_range.group(1))
        hi = extract_number(dollar_to_range.group(2))
        avg = (lo + hi) / 2 if lo is not None and hi is not None else None
        return (lo, hi, avg)
    
    # 3. Simple numeric range with unit after: "181 to 183 million" or "181-183 million"
    numeric_unit_range = re.search(r'(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:to|[-–—~])\s*(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:million|billion|M|B)', text, re.I)
    if numeric_unit_range:
        lo_val = numeric_unit_range.group(1)
        hi_val = numeric_unit_range.group(2)
        unit_match = re.search(r'\b(million|billion|M|B)\b', text, re.I)
        unit = unit_match.group(1) if unit_match else ""
        
        lo = extract_number(f"{lo_val} {unit}")
        hi = extract_number(f"{hi_val} {unit}")
        avg = (lo + hi) / 2 if lo is not None and hi is not None else None
        return (lo, hi, avg)
    
    # 4. Percent to percent: "5% to 7%" or "-14% to -13%"
    percent_to_range = re.search(r'(-?\d+(?:\.\d+)?)\s*%\s*(?:to|[-–—~])\s*(-?\d+(?:\.\d+)?)\s*%', text, re.I)
    if percent_to_range:
        lo = float(percent_to_range.group(1))
        hi = float(percent_to_range.group(2))
        avg = (lo + hi) / 2 if lo is not None and hi is not None else None
        return (lo, hi, avg)
    
    # 5. Number to number: "100 to 110" or "-100 to -90" or "181 to 183"
    number_to_range = re.search(fr'(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:to|[-–—~])\s*(\d+(?:,\d+)?(?:\.\d+)?)', text, re.I)
    if number_to_range:
        lo = float(number_to_range.group(1).replace(',', ''))
        hi = float(number_to_range.group(2).replace(',', ''))
        
        # Check if we need to make both values negative based on context
        if contains_negative_indicator and not (str(lo).startswith('-') or str(hi).startswith('-')):
            # Both should be negative if we have negative indicators
            if re.search(r'\b(?:loss|decrease|down|negative|deficit)\b', text, re.I):
                lo, hi = -abs(lo), -abs(hi)
        
        avg = (lo + hi) / 2 if lo is not None and hi is not None else None
        return (lo, hi, avg)
    
    # Special handling for specific patterns
    
    # Handle parenthetical values like "(0.05)" or "($0.05)"
    parenthetical_value = re.search(r'\(\s*\$?\s*(\d+(?:\.\d+)?)\s*\)', text, re.I)
    if parenthetical_value:
        val = float(parenthetical_value.group(1))
        return (-val, -val, -val)  # Parenthetical values are always negative in financial context
    
    # Finally check for a single value
    single = re.search(number_token, text, re.I)
    if single:
        v = extract_number(single.group(0))
        
        # If the original text indicates the value is negative but our parsed value is positive,
        # correct it to be negative
        if contains_negative_indicator and v is not None and v > 0:
            if re.search(r'\b(?:loss|decrease|down|negative|deficit)\b', text, re.I):
                v = -v
        
        return (v, v, v)
    
    return (None, None, None)

def correct_value_signs(df):
    """
    Improved function to ensure the sign (positive/negative) of values 
    is correctly applied based on the context in the original text.
    
    This version is more cautious about applying negative signs and includes
    better detection of explicitly positive values.
    """
    if 'Value' not in df.columns:
        return df  # Skip if Value column is missing
    
    for idx, row in df.iterrows():
        value_text = str(row.get('Value', ''))
        
        # Extract actual numeric representations in the value text for direct analysis
        numeric_pattern = re.search(r'(-?\$?\s*\d+(?:,\d+)*(?:\.\d+)?|\(\s*\$?\s*\d+(?:,\d+)*(?:\.\d+)?\s*\))', value_text)
        numeric_repr = numeric_pattern.group(0) if numeric_pattern else ""
        
        # Determine if the value is explicitly represented as negative
        is_explicitly_negative = (
            # Check for parenthetical notation with a number
            (numeric_repr.startswith('(') and numeric_repr.endswith(')') and re.search(r'\(\s*\$?\s*\d', numeric_repr)) or
            # Check for explicit minus sign before a number
            (numeric_repr.startswith('-') and re.search(r'-\s*\$?\s*\d', numeric_repr))
        )
        
        # Look for explicit negative indicators in surrounding context
        has_negative_context = (
            # Only consider contextual words if the number itself isn't explicitly positive or negative
            not is_explicitly_negative and
            re.search(r'\b(?:loss|decrease|down|negative|deficit)\b', value_text, re.I) and
            # Make sure we don't have contradicting positive indicators
            not re.search(r'\b(?:growth|increase|up|positive|profit)\b', value_text, re.I) and
            # Check the metric name for clear negative terms
            (any(neg_term in str(row.get('Metric', '')).lower() for neg_term in ['loss', 'deficit', 'decrease']))
        )
        
        # Determine if the value should be negative
        should_be_negative = is_explicitly_negative or has_negative_context
        
        # Determine if the value is explicitly positive
        is_explicitly_positive = (
            # Check for dollar sign without negative indicators
            ('$' in numeric_repr and not numeric_repr.startswith('-') and 
             not (numeric_repr.startswith('(') and numeric_repr.endswith(')'))) or
            # Check for a plain positive number
            (re.search(r'^\d', numeric_repr.strip()) and not 
             (numeric_repr.startswith('(') and numeric_repr.endswith(')')))
        )
        
        # Look for explicit positive indicators in surrounding context
        has_positive_context = (
            not is_explicitly_negative and
            re.search(r'\b(?:growth|increase|up|positive|profit|gain)\b', value_text, re.I) and
            not re.search(r'\b(?:loss|decrease|down|negative|deficit)\b', value_text, re.I)
        )
        
        # Determine if the value should be positive
        should_be_positive = is_explicitly_positive or has_positive_context
        
        # Resolve conflicts - if we have contradicting indicators, 
        # explicit representation in the number takes precedence
        if should_be_negative and should_be_positive:
            if is_explicitly_negative:
                should_be_positive = False
            elif is_explicitly_positive:
                should_be_negative = False
            else:
                # If no explicit indicators in the numeric representation,
                # default to keeping the value as-is (don't change sign)
                should_be_negative = False
                should_be_positive = False
        
        # Apply corrections to Low, High, Average columns
        for col in ['Low', 'High', 'Average']:
            if col not in df.columns or pd.isnull(row.get(col)):
                continue
                
            val = row[col]
            
            # Skip if the value is already null
            if val is None:
                continue
                
            # Convert to numeric for comparison if it's a string
            if isinstance(val, str):
                # Extract numeric value from string
                try:
                    val_numeric = float(re.sub(r'[^\d.-]', '', val))
                    
                    # Only fix values that have the wrong sign based on our determination
                    if should_be_negative and val_numeric > 0:
                        # Value should be negative but is positive
                        if '%' in val:
                            df.at[idx, col] = f"-{abs(val_numeric):.1f}%"
                        elif '$' in val:
                            if abs(val_numeric) >= 100:
                                df.at[idx, col] = f"-${abs(val_numeric):.0f}"
                            elif abs(val_numeric) >= 10:
                                df.at[idx, col] = f"-${abs(val_numeric):.1f}"
                            else:
                                df.at[idx, col] = f"-${abs(val_numeric):.2f}"
                        else:
                            # Just a number
                            df.at[idx, col] = f"-{abs(val_numeric)}"
                            
                    elif should_be_positive and val_numeric < 0:
                        # Value should be positive but is negative
                        if '%' in val:
                            df.at[idx, col] = f"{abs(val_numeric):.1f}%"
                        elif '$' in val:
                            if abs(val_numeric) >= 100:
                                df.at[idx, col] = f"${abs(val_numeric):.0f}"
                            elif abs(val_numeric) >= 10:
                                df.at[idx, col] = f"${abs(val_numeric):.1f}"
                            else:
                                df.at[idx, col] = f"${abs(val_numeric):.2f}"
                        else:
                            # Just a number
                            df.at[idx, col] = f"{abs(val_numeric)}"
                except ValueError:
                    continue
                    
            else:  # Numeric value
                # Only fix values that have the wrong sign based on our determination
                if should_be_negative and val > 0:
                    # Value should be negative but is positive
                    df.at[idx, col] = -abs(val)
                elif should_be_positive and val < 0:
                    # Value should be positive but is negative
                    df.at[idx, col] = abs(val)
    
    return df

def check_range_consistency(df):
    """
    Improved function to check for and fix inconsistencies in range parsing.
    More cautious about applying corrections to avoid false negatives.
    """
    for idx, row in df.iterrows():
        # Get low and high values if they exist
        low, high = row.get('Low'), row.get('High')
        
        # Skip if either value is missing or null
        if low is None or high is None or pd.isnull(low) or pd.isnull(high):
            continue
            
        # Convert to float if they're strings with % or $ signs
        if isinstance(low, str):
            try:
                low_val = float(re.sub(r'[^\d.-]', '', low))
            except ValueError:
                continue  # Skip if we can't convert to float
        else:
            low_val = low
            
        if isinstance(high, str):
            try:
                high_val = float(re.sub(r'[^\d.-]', '', high))
            except ValueError:
                continue  # Skip if we can't convert to float
        else:
            high_val = high
        
        # Get original value text for context
        original_value = str(row.get('Value', ''))
        
        # Extract explicit negative indicators from the original value
        has_parenthetical_notation = '(' in original_value and ')' in original_value and re.search(r'\(\s*\$?\s*\d', original_value)
        has_minus_sign = '-' in original_value and re.search(r'-\s*\$?\s*\d', original_value)
        
        # Check if there's explicit negative language (not just generic "decrease" which could be positive)
        has_negative_language = re.search(r'\b(?:loss|deficit|negative)\b', original_value, re.I)
        
        # 1. If low is negative and high is positive, but clear indicators that both should be negative
        if low_val < 0 and high_val > 0:
            should_both_be_negative = (
                has_parenthetical_notation or 
                has_minus_sign or 
                has_negative_language
            )
            
            if should_both_be_negative:
                # Both values should be negative - fix the high value
                if isinstance(high, str):
                    if '%' in high:
                        df.at[idx, 'High'] = f"-{abs(high_val):.1f}%"
                    elif '$' in high:
                        if abs(high_val) >= 100:
                            df.at[idx, 'High'] = f"-${abs(high_val):.0f}"
                        elif abs(high_val) >= 10:
                            df.at[idx, 'High'] = f"-${abs(high_val):.1f}"
                        else:
                            df.at[idx, 'High'] = f"-${abs(high_val):.2f}"
                    else:
                        df.at[idx, 'High'] = f"-{abs(high_val)}"
                else:
                    df.at[idx, 'High'] = -abs(high_val)
                
                # Also fix the average
                if 'Average' in df.columns and not pd.isnull(row.get('Average')):
                    avg = (low_val + (-high_val)) / 2
                    if isinstance(row.get('Average'), str):
                        if '%' in row.get('Average', ''):
                            df.at[idx, 'Average'] = f"{avg:.1f}%"
                        elif '$' in row.get('Average', ''):
                            if abs(avg) >= 100:
                                df.at[idx, 'Average'] = f"${avg:.0f}"
                            elif abs(avg) >= 10:
                                df.at[idx, 'Average'] = f"${avg:.1f}"
                            else:
                                df.at[idx, 'Average'] = f"${avg:.2f}"
                        else:
                            df.at[idx, 'Average'] = avg
                    else:
                        df.at[idx, 'Average'] = avg
    
    return df

def split_gaap_non_gaap(df):
    if 'Value' not in df.columns or 'Metric' not in df.columns:
        return df  # Avoid crash if column names are missing

    rows = []
    for _, row in df.iterrows():
        val = str(row['Value'])
        match = re.search(r'(\d[\d\.\s%to–-]*)\s*on a GAAP basis.*?(\d[\d\.\s%to–-]*)\s*on a non-GAAP basis', val, re.I)
        if match:
            gaap_val = match.group(1).strip() + " GAAP"
            non_gaap_val = match.group(2).strip() + " non-GAAP"
            for new_val, label in [(gaap_val, "GAAP"), (non_gaap_val, "Non-GAAP")]:
                new_row = row.copy()
                new_row["Value"] = new_val
                new_row["Metric"] = f"{row['Metric']} ({label})"
                lo, hi, avg = parse_value_range(new_val)
                new_row["Low"], new_row["High"], new_row["Average"] = format_percent(lo), format_percent(hi), format_percent(avg)
                rows.append(new_row)
        else:
            rows.append(row)
    return pd.DataFrame(rows)

def standardize_metrics(df):
    """
    Standardize metric names to consistent format.
    Maps various financial metric variations to standard labels.
    """
    if 'Metric' not in df.columns:
        return df  # Return unchanged if no Metric column
    
    # Create a deep copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Create a dictionary to map patterns to standardized metric names
    metric_patterns = {
        # EPS PATTERNS
        'Non-GAAP EPS': [
            r'(?i).*non[\s-]*gaap.*(?:eps|earnings per share|per share|income per share).*',
            r'(?i).*adjusted.*(?:eps|earnings per share|per share|income per share).*',
        ],
        'GAAP EPS': [
            r'(?i).*gaap.*(?:eps|earnings per share|per share|income per share).*',
            r'(?i)^(?:eps|earnings per share|per share|income per share).*',
            r'(?i).*net income per share.*',
            r'(?i).*diluted (?:eps|earnings per share).*',
        ],
        
        # REVENUE PATTERNS
        'Revenue': [
            r'(?i).*revenue.*',
            r'(?i).*sales.*',
        ],
        
        # OPERATING INCOME PATTERNS
        'Non-GAAP Operating Income': [
            r'(?i).*non[\s-]*gaap.*operating[\s-]*(?:income|profit).*',
            r'(?i).*adjusted.*operating[\s-]*(?:income|profit).*',
        ],
        'GAAP Operating Income': [
            r'(?i).*operating[\s-]*(?:income|profit).*',
            r'(?i).*income[\s-]*from[\s-]*operations.*',
        ],
        
        # NET INCOME PATTERNS
        'Non-GAAP Net Income': [
            r'(?i).*non[\s-]*gaap.*net[\s-]*(?:income|profit|earnings).*',
            r'(?i).*adjusted.*net[\s-]*(?:income|profit|earnings).*',
        ],
        'GAAP Net Income': [
            r'(?i).*gaap.*net[\s-]*(?:income|profit|earnings).*',
            r'(?i)^(?:net[\s-]*income|net[\s-]*profit|net[\s-]*earnings).*',
        ],
        
        # MARGIN PATTERNS
        'Non-GAAP Operating Margins': [
            r'(?i).*non[\s-]*gaap.*operating[\s-]*margin.*',
            r'(?i).*adjusted.*operating[\s-]*margin.*',
        ],
        'GAAP Operating Margins': [
            r'(?i).*gaap.*operating[\s-]*margin.*',
            r'(?i)^operating[\s-]*margin.*',
        ],
        'Gross Margins': [
            r'(?i).*gross[\s-]*margin.*',
        ],
        
        # OTHER METRICS
        'Adj. EBITDA': [
            r'(?i).*adjusted.*ebitda.*',
            r'(?i).*adj.*ebitda.*',
        ],
        'Free Cash Flow': [
            r'(?i).*free[\s-]*cash[\s-]*flow.*',
            r'(?i).*fcf.*',
        ],
    }
    
    # Function to find the standardized metric name based on patterns
    def find_standardized_metric(metric_name):
        for standard_name, patterns in metric_patterns.items():
            for pattern in patterns:
                if re.search(pattern, metric_name):
                    return standard_name
        return metric_name  # Return original if no match
    
    # Apply standardization to the Metric column only
    df['Metric'] = df['Metric'].apply(find_standardized_metric)
    
    return df

def extract_guidance(text, ticker, client, model_name):
    """
    Enhanced function to extract guidance from SEC filings with improved handling of 
    parenthetical negative values and other formats. Now requests period type classification.
    """
    prompt = f"""You are a financial analyst assistant. Extract ALL forward-looking guidance, projections, and outlook statements given in this earnings release for {ticker}. 

Return a structured list containing:
- metric (e.g. Revenue, EPS, Operating Margin)
- value or range (e.g. $1.5B–$1.6B or $2.05)
- applicable period (e.g. Q3 FY24, Full Year 2025)
- period type (MUST be either "Quarter" or "Full Year" based on the period text)

VERY IMPORTANT:
- Look for sections titled 'Outlook', 'Guidance', 'Financial Outlook', 'Business Outlook', or similar
- Also look for statements containing phrases like "expect", "anticipate", "forecast", "will be", "to be in the range of"
- Review the ENTIRE document for ANY forward-looking statements about future performance
- Pay special attention to sections describing "For the fiscal quarter", "For the fiscal year", "For next quarter", etc.
- For any percentage values, always include the % symbol in your output (e.g., "5% to 7%" or "5%-7%")
- Be sure to capture year-over-year growth metrics as well as absolute values
- Look for common financial metrics: Revenue, EPS, Operating Margin, Free Cash Flow, Gross Margin, etc.
- Include both quarterly and full-year guidance if available
- If guidance includes both GAAP and non-GAAP measures, include both with clear labels

FOR THE PERIOD TYPE COLUMN:
- Classify each period as either "Quarter" or "Full Year" based on the applicable period
- Use "Quarter" for: Q1, Q2, Q3, Q4, First Quarter, Next Quarter, Current Quarter, etc.
- Use "Full Year" for: Full Year, Fiscal Year, FY, Annual, Year Ending, etc.
- If a period just mentions a year (e.g., "2023" or "FY24") without specifying a quarter, classify it as "Full Year"
- THIS COLUMN IS REQUIRED AND MUST ONLY CONTAIN "Quarter" OR "Full Year" - NO OTHER VALUES

CRITICAL: HANDLING POSITIVE AND NEGATIVE VALUES

1. CORRECTLY HANDLE NEGATIVE VALUES:
   - In financial reporting, values can be shown as negative in TWO ways:
   - With a minus sign: "-$0.05" or "-5%"
   - With parentheses: "($0.05)" or "(5%)" - THESE ARE ALWAYS NEGATIVE values
   - Always maintain parenthetical notation when it appears in the original text
   - Example: Value of "($0.05)" should be output as "($0.05)" (not "$0.05" or "-$0.05")
   - Example: Value of "(5%)" should be output as "(5%)" (not "5%" or "-5%")

2. CORRECTLY HANDLE POSITIVE VALUES:
   - Do NOT add minus signs or parentheses to positive values
   - Example: "$0.05" must be output as "$0.05" (not "-$0.05" or "($0.05)")
   - Example: "5%" must be output as "5%" (not "-5%" or "(5%)")
   - Only use minus signs or parentheses when the original text explicitly indicates a negative value

3. CRITICAL HANDLING OF NUMERIC RANGES:
   - ALWAYS include both the lower AND upper bounds in ranges, even when they differ by small amounts
   - Example: "$181 million to $183 million" must be output as "$181 million to $183 million" (not just "$181 million")
   - Example: "$181-$183 million" must be output as "$181-$183 million" (not just "$181 million")
   - Example: "181-183" must be output as "181-183" (not just "181")
   - For negative ranges, maintain consistent notation throughout the range:
   - Example: "($0.08) to ($0.06)" should be output exactly as "($0.08) to ($0.06)"
   - Example: "-$0.08 to -$0.06" should be output exactly as "-$0.08 to -$0.06"

CRITICAL FORMATTING FOR BILLION VALUES:
- When a value is expressed in billions (e.g., "$1.10 billion" or "$1.11B"), convert it to millions by multiplying by 1000:
  - Example: "$1.10 billion" should be output as "$1,100 million"
  - Example: "$1.11B" should be output as "$1,110 million"
  - Example: A range of "$1.10-$1.11 billion" should be output as "$1,100-$1,110 million"
  - Example: "$1.117 billion to $1.121 billion" should be output as "$1,117 million to $1,121 million"
- IMPORTANT: Do NOT add extra zeros beyond multiplying by 1000. Just convert the exact number from billions to millions.
- WRONG: "$1.117 billion" should NOT become "$1,117,000 million"
- CORRECT: "$1.117 billion" should become "$1,117 million"
- For ranges, convert each number individually: "$1.117-$1.121 billion" becomes "$1,117-$1,121 million"

IMPORTANT FORMATTING INSTRUCTIONS:
- For dollar ranges, use the format "$X to $Y" (with dollar sign before each number)
  - Example: "$0.08 to $0.09" (not "$0.08-0.09")
  - Example: "$181 million to $183 million" (not "$181-$183 million")
- For percentage ranges, use the format "X% to Y%" (with % after each number)
  - Example: "5% to 7%" (not "5-7%")
- For other numeric ranges, use "X to Y" format
  - Example: "100 to 110" (not "100-110")
- Keep numbers exactly as stated (don't convert $0.08 to $0.8, etc.) EXCEPT for billion values which must be converted as instructed above

Respond in table format without commentary.\n\n{text}"""
    
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

# Add this new function to fix mixed sign ranges
def fix_mixed_ranges(df):
    """
    Direct fix for mixed sign ranges like "$(1) million to $1 million".
    This will ensure the High column is positive when dealing with these ranges.
    """
    if 'Value or range' not in df.columns or 'Low' not in df.columns or 'High' not in df.columns:
        return df  # Skip if any required columns are missing
    
    for idx, row in df.iterrows():
        value_text = str(row.get('Value or range', ''))
        
        # Check for mixed-sign range patterns like "$(1) million to $1 million" or similar
        mixed_range_pattern = (
            # Has a negative parenthetical value followed by a positive value in a range
            (('$(' in value_text or '( ' in value_text) and 
             ('to $' in value_text or 'to ' in value_text) and
             not (' to (' in value_text) and # Second part is not in parentheses 
             not (' to -' in value_text))    # Second part is not negative
        )
        
        # If we detect a mixed-sign range
        if mixed_range_pattern:
            # Get the values
            low_val = row.get('Low')
            high_val = row.get('High')
            
            # Skip if either value is missing
            if pd.isnull(low_val) or pd.isnull(high_val):
                continue
                
            # Convert to numeric if needed
            if isinstance(low_val, str):
                try:
                    low_num = float(re.sub(r'[^\d.-]', '', low_val))
                except:
                    continue  # Skip if can't parse
            else:
                low_num = low_val
                
            if isinstance(high_val, str):
                try:
                    high_num = float(re.sub(r'[^\d.-]', '', high_val))
                except:
                    continue  # Skip if can't parse
            else:
                high_num = high_val
            
            # If low is negative and high is also negative (incorrectly)
            if low_num < 0 and high_num < 0:
                # Fix the high value - make it positive
                if isinstance(high_val, str):
                    # Preserve formatting while making positive
                    if '%' in high_val:
                        df.at[idx, 'High'] = f"{abs(high_num):.1f}%"
                    elif '$' in high_val:
                        if abs(high_num) >= 100:
                            df.at[idx, 'High'] = f"${abs(high_num):.0f}"
                        elif abs(high_num) >= 10:
                            df.at[idx, 'High'] = f"${abs(high_num):.1f}"
                        else:
                            df.at[idx, 'High'] = f"${abs(high_num):.2f}"
                    else:
                        df.at[idx, 'High'] = f"{abs(high_num)}"
                else:
                    df.at[idx, 'High'] = abs(high_num)
    
    return df

# Streamlit App Setup
st.set_page_config(page_title="SEC 8-K Guidance Extractor", layout="centered")
st.title("📄 SEC 8-K Guidance Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, ORCL)", "MSFT").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")

# Add model selection dropdown
openai_models = {
    "GPT-4 Turbo": "gpt-4-turbo-preview",
    "GPT-4": "gpt-4",
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}
selected_model = st.selectbox(
    "Select OpenAI Model",
    list(openai_models.keys()),
    index=0  # Default to first option (GPT-4 Turbo)
)

# Both filter options displayed at the same time
year_input = st.text_input("How many years back to search for 8-K filings? (Leave blank for most recent only)", "")
quarter_input = st.text_input("OR enter specific quarter (e.g., 2Q25, Q4FY24)", "")

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


def generate_fiscal_quarters(fiscal_year_end_month):
    """
    Dynamically generate fiscal quarters based on the fiscal year end month.
    """
    # Calculate the first month of the fiscal year (month after fiscal year end)
    fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
    
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

def get_fiscal_dates(ticker, quarter_num, year_num, fiscal_year_end_month, fiscal_year_end_day):
    """
    Calculate the appropriate date range for a fiscal quarter
    based on the fiscal year end month.
    """
    # Generate quarters dynamically based on fiscal year end
    quarters = generate_fiscal_quarters(fiscal_year_end_month)
    
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
            # Example: For fiscal year ending in June (FY2024 = Jul 2023-Jun 2024)
            # Q1 (Jul-Sep) and Q2 (Oct-Dec) start in calendar year 2023
            start_calendar_year = year_num - 1
        else:
            # This quarter starts in the current calendar year
            # Example: For fiscal year ending in June (FY2024 = Jul 2023-Jun 2024)
            # Q3 (Jan-Mar) and Q4 (Apr-Jun) start in calendar year 2024
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


def find_guidance_paragraphs(text):
    """
    Extract paragraphs from text that are likely to contain guidance information.
    Returns both the filtered paragraphs and a boolean indicating if any were found.
    """
    # Define patterns to identify guidance sections
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

if st.button("🔍 Extract Guidance"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        cik = lookup_cik(ticker)
        if not cik:
            st.error("CIK not found for ticker.")
        else:
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
                            # First, standardize the column names
                            column_names = [c.strip() for c in rows[0]]
                            
                            # Create DataFrame with original column names
                            df = pd.DataFrame(rows[1:], columns=column_names)
                            
                            # Find the value column - it might be called "Value" or "Value or range"
                            value_column = None
                            for col in df.columns:
                                if "value" in col.lower():
                                    value_column = col
                                    break
                            
                            # If no value column found, we have a problem
                            if not value_column:
                                st.warning("⚠️ No value column found in the extracted data.")
                            else:
                                # Save the original values for later
                                original_values = df[value_column].copy()
                                
                                # Rename the value column to "Value" for consistent processing
                                if value_column != "Value":
                                    df.rename(columns={value_column: "Value"}, inplace=True)
                                
                                # Store which rows have percentages or parenthetical values in the Value column
                                percentage_rows = []
                                parenthetical_rows = []
                                for idx, row in df.iterrows():
                                    val_text = str(row.get('Value', ''))
                                    if '%' in val_text:
                                        percentage_rows.append(idx)
                                    if '(' in val_text and ')' in val_text:
                                        parenthetical_rows.append(idx)
                                
                                # Parse low, high, and average from Value column
                                df[['Low','High','Average']] = df['Value'].apply(lambda v: pd.Series(parse_value_range(v)))
                                
                                # Apply special corrections for parenthetical values
                                for idx in parenthetical_rows:
                                    for col in ['Low', 'High', 'Average']:
                                        if col in df.columns and not pd.isnull(df.loc[idx, col]):
                                            val = df.loc[idx, col]
                                            if isinstance(val, (int, float)) and val > 0:
                                                df.at[idx, col] = -abs(val)
                                
                                # Apply comprehensive sign correction based on context
                                df = correct_value_signs(df)
                                
                                # Apply GAAP/non-GAAP split
                                df = split_gaap_non_gaap(df)
                                
                                # For rows that originally had % in the Value column, make sure Low, High, Average have % too
                                for idx in percentage_rows:
                                    # Add % to Low, High, Average columns
                                    for col in ['Low', 'High', 'Average']:
                                        if pd.notnull(df.loc[idx, col]) and isinstance(df.loc[idx, col], (int, float)):
                                            df.at[idx, col] = f"{df.loc[idx, col]:.1f}%"
                                
                                # Apply a final consistency check to fix any remaining issues with negative ranges
                                df = check_range_consistency(df)
                                
                                # Apply the targeted fix for mixed-sign ranges
                                df = fix_mixed_ranges(df)
                                
                                # Apply metric standardization (only changes the Metric column)
                                df = standardize_metrics(df)
                                
                                # Now rename the "Value" column back to "Value or range" and restore original values
                                if "Value" in df.columns:
                                    df.rename(columns={"Value": "Value or range"}, inplace=True)
                                    # Restore the original values
                                    df["Value or range"] = original_values
                                
                                # Add metadata columns
                                df["FilingDate"] = date_str
                                df["8K_Link"] = url
                                df["Model_Used"] = selected_model
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
                
                # Define the order of columns without removing any existing columns
                primary_columns = [
                    "Metric", "Value or range", "Period", "PeriodType", 
                    "Low", "High", "Average", "FilingDate", 
                    "8K_Link", "Model_Used"
                ]
                
                # Preview the table
                st.subheader("🔍 Preview of Extracted Guidance")
                
                # Select the most relevant columns for display
                display_cols = ["Metric", "Value or range", "Period", "PeriodType", "Low", "High", "Average", "FilingDate"]
                display_df = combined[display_cols] if all(col in combined.columns for col in display_cols) else combined
                
                # Display the table with formatting
                st.dataframe(display_df, use_container_width=True)
                
                # Add download button
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_guidance_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No guidance data extracted.")

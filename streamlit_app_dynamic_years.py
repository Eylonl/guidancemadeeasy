# MODIFY ONLY THIS FUNCTION:

def extract_number(token: str):
    """Modified to handle decimal precision in billion values"""
    if not token or not isinstance(token, str):
        return None
    
    neg = token.strip().startswith('(') and token.strip().endswith(')' )
    tok = token.replace('(', '').replace(')', '').replace('$','') \
               .replace(',', '').strip().lower()
    
    # Special handling for billion values with decimal precision
    # This handles cases like "$1.10 billion" → 1100, "$1.11 billion" → 1110
    billion_match = re.search(r'(\d+\.\d+)\s*(?:billion|b\b)', tok)
    if billion_match:
        # Extract the number and apply precise conversion
        num_str = billion_match.group(1)
        val = float(num_str) * 1000
        return -val if neg else val
    
    # Standard processing for other cases
    factor = 1.0
    if tok.endswith('billion'): tok, factor = tok[:-7].strip(), 1000
    elif tok.endswith('million'): tok, factor = tok[:-7].strip(), 1
    elif tok.endswith('b'): tok, factor = tok[:-1].strip(), 1000
    elif tok.endswith('m'): tok, factor = tok[:-1].strip(), 1
    elif tok.endswith('k'): tok, factor = tok[:-1].strip(), 0.001
    
    try:
        val = float(tok) * factor
        return -val if neg else val
    except:
        return None


# AND MODIFY THE DISPLAY CODE:

# Display the table with formatting
st.dataframe(
    display_df.style.format({
        "Low": lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and abs(x) >= 100 else (f"{x:.2f}" if isinstance(x, (int, float)) else x),
        "High": lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and abs(x) >= 100 else (f"{x:.2f}" if isinstance(x, (int, float)) else x),
        "Average": lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and abs(x) >= 100 else (f"{x:.2f}" if isinstance(x, (int, float)) else x)
    }),
    use_container_width=True
)

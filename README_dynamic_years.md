# SEC 8-K Guidance Extractor App

This app extracts forward-looking guidance from EX-99.1 exhibits in SEC 8-K filings using GPT-4.

## 🔧 Features
- Choose how many years back to search for earnings-related 8-K filings (default: 1 year)
- Enter any U.S. public company ticker (e.g. TEAM, AAPL, MSFT)
- Automatically pulls 8-K filings from the past year
- Extracts structured guidance using OpenAI's GPT-4
- Downloads a clean Excel report

## 🚀 How to Deploy on Streamlit Cloud
1. Clone or fork this repo to your GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“New app”**, and link to this repo.
4. Set the app file as `streamlit_app.py`
5. Deploy.

## 🔐 Requirements
- OpenAI API key (you’ll be prompted to paste it in the app)
- Internet access

## 📦 Dependencies
See `requirements.txt`

## 📄 Example Output
A downloadable Excel file with columns:
- Metric
- Value
- Applicable Period
- Filing Date
- 8-K Link

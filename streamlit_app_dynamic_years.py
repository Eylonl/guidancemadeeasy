
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="OpenAI API Diagnostic")

st.title("🧪 OpenAI API Test")
api_key = st.text_input("Enter your OpenAI API Key", type="password")
test_prompt = st.text_area("Enter a short prompt", "What is the capital of France?")

if st.button("Test OpenAI API"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            st.info("✅ OpenAI client initialized.")
            with st.spinner("Calling GPT-4..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": test_prompt}],
                    temperature=0.3
                )
                st.success("✅ Response received.")
                st.text_area("🔍 GPT Output", response.choices[0].message.content, height=200)
        except Exception as e:
            st.error(f"❌ OpenAI API Error: {e}")

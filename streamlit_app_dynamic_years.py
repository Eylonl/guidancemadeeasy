
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="OpenAI API Diagnostic")

st.title("🧪 OpenAI API Connectivity Test")

api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
prompt = st.text_area("💬 Prompt to send to GPT-4", "What is the capital of France?")

if st.button("🚀 Run Test"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        try:
            st.info("🔧 Initializing OpenAI client...")
            client = OpenAI(api_key=api_key)

            with st.spinner("⏳ Sending request to GPT-4..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                output = response.choices[0].message.content
                st.success("✅ GPT-4 responded successfully!")
                st.text_area("📥 GPT-4 Output", output, height=200)

        except Exception as e:
            st.error(f"❌ OpenAI API Error: {e}")

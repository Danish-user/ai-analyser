import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(page_title="AI Data Insights", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Business Data Analyzer")
st.write("Upload a CSV file to get instant executive summaries and insights.")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
        
    st.divider()
    
    st.subheader("🤖 Generate AI Executive Summary")
    
    if st.button("Analyze Data"):
        if not api_key:
            st.error("Please enter a Groq API key in the sidebar.")
        else:
            with st.spinner("Analyzing data with AI..."):
                try:
                    client = Groq(api_key=api_key)
                    
                    data_summary = f"Columns: {list(df.columns)}\nSample Data:\n{df.head(5).to_string()}"
                    
                    prompt = f"""
                    You are an expert business data analyst. Analyze this dataset overview and sample:
                    {data_summary}
                    
                    Provide:
                    1. A 3-bullet point executive summary of what this dataset contains.
                    2. 3 key potential business insights or trends to look out for.
                    3. Recommended next steps for action.
                    """
                    
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(page_title="AI Data Insights", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Business Data Analyzer")
st.write("Upload a CSV file to get instant executive summaries, statistical metrics, and charts.")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("1. Data Preview & Basic Metrics")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
        
    st.divider()
    
    # --- DATA SCIENCE FEATURE: AUTOMATED AGGREGATION & CHARTS ---
    st.subheader("2. Automated Data Analytics & Charts")
    
    # Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols:
        # User selects which numeric column to analyze
        selected_num_col = st.selectbox("Select a numeric column for calculation:", numeric_cols)
        
        # Calculate stats
        total_val = df[selected_num_col].sum()
        avg_val = df[selected_num_col].mean()
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(f"Total {selected_num_col}", f"{total_val:,.2f}")
        with m_col2:
            st.metric(f"Average {selected_num_col}", f"{avg_val:,.2f}")
            
        # Charting section
        if categorical_cols:
            selected_cat_col = st.selectbox("Group by column for chart:", categorical_cols)
            
            # Pandas aggregation: group by category and sum the numeric column
            chart_data = df.groupby(selected_cat_col)[selected_num_col].sum()
            
            st.write(f"### {selected_num_col} by {selected_cat_col}")
            st.bar_chart(chart_data)
    else:
        st.info("No numeric columns found in this dataset for automated chart generation.")
        
    st.divider()
    
    # --- AI EXECUTIVE SUMMARY ---
    st.subheader("3. 🤖 Generate AI Executive Summary")
    
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
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

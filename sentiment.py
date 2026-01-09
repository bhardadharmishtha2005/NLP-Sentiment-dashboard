import streamlit as st
import pandas as pd
from textblob import TextBlob
import io

# 1. Page Configuration
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="wide")

# --- SIDEBAR: ABOUT SECTION ---
st.sidebar.title("About this App")
st.sidebar.info("""
**AI Sentiment Analyzer**
This app uses **TextBlob** to perform Natural Language Processing (NLP) on CSV data. 
- **Tool:** Streamlit
- **Library:** TextBlob (NLP)
- **Goal:** Analyze text sentiment in real-time.
""")

st.sidebar.markdown("---")

# 2. Sidebar - Upload or Demo
st.sidebar.header("Step 1: Get Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV/TSV", type=["csv", "tsv"])
use_demo = st.sidebar.checkbox("Use Sample Testing Data")

# 3. Sentiment Analysis Function
def get_sentiment(text):
    if not isinstance(text, str):
        return "Neutral"
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# 4. Data Loading Logic with Robust Error Handling
df = None

if uploaded_file is not None:
    # We try multiple ways to read the file to avoid the errors in your screenshots
    encodings = ['utf-8', 'utf-16', 'latin1']
    delimiters = [',', '\t', ';']
    
    success = False
    for enc in encodings:
        if success: break
        for sep in delimiters:
            try:
                uploaded_file.seek(0) # Reset file pointer for each attempt
                df = pd.read_csv(uploaded_file, encoding=enc, sep=sep)
                # Check if it actually parsed columns or just one giant string
                if len(df.columns) > 1: 
                    success = True
                    break
            except Exception:
                continue
    
    if not success:
        st.error("Could not parse the file. Please ensure it is a valid CSV or TSV.")

elif use_demo:
    # Sample data for testing on phone without a file
    demo_data = {
        "User": ["Amit", "Sara", "John", "Meera", "Rahul"],
        "Feedback": [
            "This app is amazing and very helpful!",
            "I hate the new update, it is very slow.",
            "It is okay, but could be better.",
            "Great experience, I love the UI design.",
            "Worst service ever, very disappointed."
        ]
    }
    df = pd.DataFrame(demo_data)
    st.sidebar.success("Loaded Sample Data")

# 5. Main Screen Logic
st.title("🤖 AI Data Sentiment Analyzer")

if df is not None:
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 NLP Sentiment Analysis")
    
    # Selection for analysis
    text_col = st.selectbox("Select the column containing text/feedback:", df.columns)
    
    if st.button("Run TextBlob Analysis"):
        with st.spinner('Analyzing sentiments...'):
            df['Sentiment'] = df[text_col].apply(get_sentiment)
            
            st.success("Analysis Complete!")
            st.dataframe(df, use_container_width=True)
            
            # Summary Metrics
            col1, col2, col3 = st.columns(3)
            counts = df['Sentiment'].value_counts()
            col1.metric("Positive ✅", counts.get("Positive", 0))
            col2.metric("Neutral ↔️", counts.get("Neutral", 0))
            col3.metric("Negative ❌", counts.get("Negative", 0))
            
            # Visualization
            st.subheader("📊 Sentiment Distribution")
            st.bar_chart(counts)
else:
    st.warning("👈 Please upload a file or check 'Use Sample Testing Data' in the sidebar to begin.")
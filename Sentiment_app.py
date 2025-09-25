import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime
import os

# Enhanced import handling dengan fallbacks
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError as e:
    matplotlib_available = False
    st.sidebar.warning("📊 Matplotlib: Using Altair for visualizations")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk_available = True
except ImportError as e:
    nltk_available = False
    st.sidebar.error("❌ NLTK: Sentiment analysis disabled")

try:
    import google.generativeai as genai
    genai_available = True
except ImportError as e:
    genai_available = False
    st.sidebar.warning("🤖 Google AI: API features unavailable")

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer | Professional NLP Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dengan better error handling
@st.cache_resource
def initialize_components():
    # Initialize NLTK
    if nltk_available:
        try:
            nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()
        except Exception as e:
            st.error(f"❌ NLTK Error: {e}")
            sia = None
    else:
        sia = None
    
    # Initialize Gemini AI
    if genai_available:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            else:
                gemini_model = None
                st.sidebar.info("🔑 Add GOOGLE_API_KEY to secrets")
        except Exception as e:
            st.error(f"❌ Gemini AI Error: {e}")
            gemini_model = None
    else:
        gemini_model = None
    
    return sia, gemini_model

# Initialize
sia, gemini_model = initialize_components()

# ===== HEADER =====
st.title("🚀 AI-Powered Sentiment Analysis Platform")
st.markdown("""
*Professional business intelligence tool for real-time sentiment analysis and customer feedback monitoring.*
""")

# ===== SIDEBAR STATUS =====
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Dependencies status
    col1, col2 = st.columns(2)
    with col1:
        st.metric("NLTK", "✅ Active" if nltk_available else "❌ Offline")
    with col2:
        st.metric("Gemini AI", "✅ Active" if gemini_model else "❌ Setup Required")
    
    st.metric("Visualization", "📊 Matplotlib" if matplotlib_available else "📈 Altair")
    
    analysis_type = st.radio("Analysis Mode:", 
                           ["Single Text", "Batch Processing", "Real-time Dashboard"])

# ===== DEMO DATA UNTUK TEST =====
def create_demo_sentiment_scores(text):
    """Fallback function jika NLTK tidak tersedia"""
    # Simple rule-based sentiment sebagai fallback
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'awesome', 'perfect']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointing', 'poor']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = max(len(text.split()), 1)
    pos_score = positive_count / total_words
    neg_score = negative_count / total_words
    neu_score = 1 - pos_score - neg_score
    compound = pos_score - neg_score
    
    return {
        'neg': neg_score,
        'neu': neu_score,
        'pos': pos_score,
        'compound': compound
    }

# ===== SINGLE TEXT ANALYSIS =====
if analysis_type == "Single Text":
    st.header("📝 Text Sentiment Analysis")
    
    text_input = st.text_area("Enter text for analysis:", 
                             "This product is absolutely amazing! I love the quality and fast delivery.",
                             height=100)
    
    if st.button("Analyze Sentiment", type="primary"):
        if text_input.strip():
            if sia:
                # Gunakan NLTK jika available
                scores = sia.polarity_scores(text_input)
                st.success("✅ Using NLTK VADER Analysis")
            else:
                # Fallback ke demo function
                scores = create_demo_sentiment_scores(text_input)
                st.info("ℹ️ Using basic sentiment analysis (NLTK unavailable)")
            
            compound = scores['compound']
            sentiment = "😊 Positive" if compound > 0.05 else "😞 Negative" if compound < -0.05 else "😐 Neutral"
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", sentiment)
            with col2:
                st.metric("Confidence", f"{abs(compound):.3f}")
            with col3:
                st.metric("Text Length", f"{len(text_input)} chars")
            
            # Visualization
            if matplotlib_available:
                try:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    labels = ['Positive', 'Negative', 'Neutral']
                    values = [scores['pos'], scores['neg'], scores['neu']]
                    ax.bar(labels, values, color=['green', 'red', 'blue'])
                    ax.set_ylabel('Score')
                    ax.set_title('Sentiment Distribution')
                    st.pyplot(fig)
                except:
                    # Fallback ke Altair
                    chart_data = pd.DataFrame({'Sentiment': labels, 'Score': values})
                    st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
                        x='Sentiment', y='Score', color='Sentiment'
                    ), use_container_width=True)
            else:
                # Always use Altair
                labels = ['Positive', 'Negative', 'Neutral']
                values = [scores['pos'], scores['neg'], scores['neu']]
                chart_data = pd.DataFrame({'Sentiment': labels, 'Score': values})
                st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
                    x='Sentiment', y='Score', color='Sentiment'
                ), use_container_width=True)

# ===== BATCH PROCESSING =====
elif analysis_type == "Batch Processing":
    st.header("📁 Batch Text Analysis")
    
    sample_texts = """Excellent product, highly recommended!
This is okay, nothing special.
Very disappointed with the quality.
Outstanding service and support.
Good value for the money."""
    
    batch_input = st.text_area("Enter texts (one per line):", sample_texts, height=150)
    
    if st.button("Analyze Batch", type="primary"):
        if batch_input.strip():
            texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            results = []
            
            for text in texts:
                if sia:
                    scores = sia.polarity_scores(text)
                else:
                    scores = create_demo_sentiment_scores(text)
                
                compound = scores['compound']
                sentiment = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
                
                results.append({
                    'Text': text[:50] + "..." if len(text) > 50 else text,
                    'Sentiment': sentiment,
                    'Score': f"{compound:.3f}"
                })
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Summary
            sentiment_counts = df['Sentiment'].value_counts()
            st.altair_chart(alt.Chart(pd.DataFrame({
                'Sentiment': sentiment_counts.index,
                'Count': sentiment_counts.values
            })).mark_bar().encode(x='Sentiment', y='Count'), use_container_width=True)

# ===== REAL-TIME DASHBOARD =====
else:
    st.header("📊 Sentiment Analytics Dashboard")
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=24, freq='H')
    sentiment_scores = np.random.normal(0, 0.2, 24).cumsum()
    
    dashboard_data = pd.DataFrame({
        'Time': dates,
        'Sentiment_Score': sentiment_scores,
        'Sentiment': ['Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral' for x in sentiment_scores]
    })
    
    # Charts
    st.altair_chart(alt.Chart(dashboard_data).mark_line().encode(
        x='Time:T', y='Sentiment_Score:Q'
    ).properties(height=300, title='Sentiment Trend'), use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current", f"{sentiment_scores[-1]:.2f}")
    with col2:
        st.metric("Average", f"{np.mean(sentiment_scores):.2f}")
    with col3:
        st.metric("Positive", f"{sum(1 for x in sentiment_scores if x > 0)}")
    with col4:
        st.metric("Negative", f"{sum(1 for x in sentiment_scores if x < 0)}")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🚀 AI Sentiment Analysis Platform</strong></p>
    <p>Ready for enterprise deployment • Custom solutions available</p>
</div>
""", unsafe_allow_html=True)
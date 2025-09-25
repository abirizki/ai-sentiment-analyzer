import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime
import os

# Import dengan error handling
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not available. Some charts will be simplified.")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.error("NLTK not available. Sentiment analysis disabled.")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.warning("Google Gemini AI not available. AI features disabled.")

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer | Professional NLP Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components dengan error handling
@st.cache_resource
def initialize_analytics():
    if not NLTK_AVAILABLE:
        return None
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"NLTK initialization error: {e}")
        return None

@st.cache_resource
def initialize_gemini():
    if not GENAI_AVAILABLE:
        return None
        
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    if not api_key:
        st.sidebar.warning("🔑 Gemini AI: API Key Required")
        return None
    
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.sidebar.error("❌ Invalid API configuration")
        return None

# Initialize components
sia = initialize_analytics()
gemini_model = initialize_gemini()

# ===== HEADER SECTION =====
st.title("🚀 AI-Powered Sentiment Analysis Platform")
st.markdown("""
*Professional-grade NLP tool for real-time sentiment analysis, batch processing, and AI-driven insights. 
Built with cutting-edge technology for business intelligence applications.*
""")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Show availability status
    st.subheader("Component Status")
    if NLTK_AVAILABLE:
        st.success("✅ NLTK/VADER: Active")
    else:
        st.error("❌ NLTK/VADER: Unavailable")
        
    if GENAI_AVAILABLE and gemini_model:
        st.success("✅ Gemini AI: Active")
    elif GENAI_AVAILABLE:
        st.info("ℹ️ Gemini AI: Available with API key")
    else:
        st.error("❌ Gemini AI: Unavailable")
    
    analysis_type = st.radio(
        "Analysis Mode:",
        ["Single Text", "Batch Processing", "Real-time Dashboard"],
        help="Choose analysis method based on your needs"
    )
    
    st.markdown("---")
    st.markdown("""
    **🔧 Built With:**
    - Streamlit • Google Gemini AI • NLTK/VADER
    - Pandas • Altair • Matplotlib
    - Python 3.8+
    """)

# ===== SINGLE TEXT ANALYSIS =====
if analysis_type == "Single Text":
    st.header("📝 Single Text Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text for analysis:",
            "The product exceeded my expectations! Excellent quality and fast delivery. Highly recommended!",
            height=120,
            help="Enter any text: reviews, comments, feedback, etc."
        )
    
    with col2:
        st.metric("Text Length", f"{len(text_input)} chars")
        analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_btn and text_input.strip():
        if not NLTK_AVAILABLE:
            st.error("❌ Sentiment analysis unavailable. Required packages not installed.")
        else:
            with st.spinner("Analyzing sentiment with AI..."):
                # Perform sentiment analysis
                scores = sia.polarity_scores(text_input)
                compound_score = scores['compound']
                
                # Determine sentiment label
                if compound_score >= 0.05:
                    sentiment = "Positive 😊"
                    sentiment_color = "green"
                elif compound_score <= -0.05:
                    sentiment = "Negative 😞"
                    sentiment_color = "red"
                else:
                    sentiment = "Neutral 😐"
                    sentiment_color = "blue"
                
                # Display results in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sentiment", sentiment)
                with col2:
                    st.metric("Confidence Score", f"{abs(compound_score):.3f}")
                with col3:
                    st.metric("Positive", f"{scores['pos']:.3f}")
                with col4:
                    st.metric("Negative", f"{scores['neg']:.3f}")
                
                # Detailed scores expander
                with st.expander("📊 Detailed Analysis Scores"):
                    st.dataframe(pd.DataFrame([scores]), use_container_width=True)
                    
                    # Visualization - menggunakan Altair jika matplotlib tidak tersedia
                    if MATPLOTLIB_AVAILABLE:
                        try:
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Pie chart
                            labels = ['Positive', 'Negative', 'Neutral']
                            sizes = [scores['pos'], scores['neg'], scores['neu']]
                            colors = ['#4CAF50', '#F44336', '#2196F3']
                            ax[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                            ax[0].set_title('Sentiment Distribution')
                            
                            # Bar chart
                            ax[1].bar(labels, sizes, color=colors)
                            ax[1].set_title('Score Breakdown')
                            ax[1].set_ylim(0, 1)
                            
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning("Chart rendering issue. Using simplified visualization.")
                    
                    # Fallback ke Altair
                    chart_data = pd.DataFrame({
                        'Sentiment': ['Positive', 'Negative', 'Neutral'],
                        'Score': [scores['pos'], scores['neg'], scores['neu']]
                    })
                    bar_chart = alt.Chart(chart_data).mark_bar().encode(
                        x='Sentiment',
                        y='Score',
                        color='Sentiment'
                    ).properties(title='Sentiment Distribution')
                    st.altair_chart(bar_chart, use_container_width=True)
                
                # AI Insights (if Gemini available)
                if gemini_model:
                    with st.expander("🤖 AI-Powered Insights"):
                        try:
                            prompt = f"""
                            Analyze this text for sentiment and provide business insights:
                            "{text_input}"
                            
                            Provide:
                            1. Key emotional triggers
                            2. Business implications
                            3. Suggested actions
                            """
                            response = gemini_model.generate_content(prompt)
                            st.write(response.text)
                        except Exception as e:
                            st.info("AI insights temporarily unavailable")
                else:
                    st.info("🔑 Add Gemini API key in secrets for AI features")

# ... (lanjutkan dengan bagian Batch Processing dan Dashboard yang sama, dengan error handling)
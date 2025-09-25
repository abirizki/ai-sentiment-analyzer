import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np  # ← CRITICAL: This was missing
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import google.generativeai as genai
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer | Professional NLP Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize with error handling
@st.cache_resource
def initialize_analytics():
    """Initialize NLTK and sentiment analyzer"""
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return None

@st.cache_resource
def initialize_gemini():
    """Initialize Gemini AI with professional error handling"""
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    if not api_key:
        st.sidebar.warning("🔑 Gemini AI: API Key Required")
        st.sidebar.info("""
        **To enable AI features:**
        1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Add to Streamlit Cloud secrets
        """)
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
    analysis_type = st.radio(
        "Analysis Mode:",
        ["Single Text", "Batch Processing", "Real-time Dashboard"],
        help="Choose analysis method based on your needs"
    )
    
    st.markdown("---")
    st.header("📈 Performance")
    if gemini_model:
        st.success("✅ Gemini AI: Active")
    else:
        st.info("ℹ️ Gemini AI: Available with API key")
    
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
                
                # Visualization
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

# ===== BATCH PROCESSING =====
elif analysis_type == "Batch Processing":
    st.header("📁 Batch Text Analysis")
    
    st.info("💡 **Professional Use Case**: Analyze multiple customer reviews, social media comments, or survey responses at scale.")
    
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        """This product is amazing! I love it.
The quality could be better.
It's okay, nothing special.
Outstanding service and support.
Very disappointed with the purchase.
Good value for money.""",
        height=200
    )
    
    if st.button("🚀 Process Batch Analysis", type="primary"):
        if batch_input.strip():
            texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            progress_bar = st.progress(0)
            results = []
            
            for i, text in enumerate(texts):
                scores = sia.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= 0.05:
                    sentiment = "Positive"
                    emoji = "😊"
                elif compound <= -0.05:
                    sentiment = "Negative" 
                    emoji = "😞"
                else:
                    sentiment = "Neutral"
                    emoji = "😐"
                
                results.append({
                    'Text': text,
                    'Sentiment': f"{sentiment} {emoji}",
                    'Score': round(compound, 3),
                    'Positive': round(scores['pos'], 3),
                    'Negative': round(scores['neg'], 3),
                    'Neutral': round(scores['neu'], 3)
                })
                progress_bar.progress((i + 1) / len(texts))
            
            df = pd.DataFrame(results)
            
            # Display results
            st.subheader("📋 Analysis Results")
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            positive_count = len(df[df['Score'] > 0.05])
            negative_count = len(df[df['Score'] < -0.05])
            neutral_count = len(df[(df['Score'] >= -0.05) & (df['Score'] <= 0.05)])
            
            with col1:
                st.metric("Total Texts", len(df))
            with col2:
                st.metric("Positive", positive_count, delta=f"{(positive_count/len(df)*100):.1f}%")
            with col3:
                st.metric("Negative", negative_count)
            with col4:
                st.metric("Neutral", neutral_count)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_counts = df['Sentiment'].str.split(' ').str[0].value_counts()
                st.altair_chart(alt.Chart(pd.DataFrame({
                    'Sentiment': sentiment_counts.index,
                    'Count': sentiment_counts.values
                })).mark_bar().encode(
                    x='Sentiment',
                    y='Count',
                    color='Sentiment'
                ).properties(title='Sentiment Distribution'), use_container_width=True)
            
            with col2:
                # Score distribution
                st.altair_chart(alt.Chart(df).mark_bar().encode(
                    alt.X("Score:Q", bin=alt.Bin(maxbins=20)),
                    y='count()',
                    color=alt.Color('Score:Q', scale=alt.Scale(scheme='redyellowgreen'))
                ).properties(title='Score Distribution'), use_container_width=True)
            
            # Export functionality
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Export Results as CSV",
                data=csv_data,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# ===== REAL-TIME DASHBOARD =====
else:
    st.header("📊 Real-time Sentiment Dashboard")
    st.info("💡 **Enterprise Feature**: Monitor sentiment trends across multiple data sources in real-time.")
    
    # Demo dashboard with sample data
    sample_data = pd.DataFrame({
        'Time': pd.date_range('2024-01-01', periods=50, freq='H'),
        'Sentiment_Score': np.random.normal(0, 0.3, 50).cumsum()
    })
    sample_data['Sentiment'] = ['Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral' for x in sample_data['Sentiment_Score']]
    
    st.altair_chart(alt.Chart(sample_data).mark_line().encode(
        x='Time:T',
        y='Sentiment_Score:Q',
        color=alt.value('#FF6B6B')
    ).properties(title='Real-time Sentiment Trend', height=300), use_container_width=True)
    
    st.info("**🔒 Premium Feature**: Connect this dashboard to your live data sources (API, database, social media streams)")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🚀 Professional Sentiment Analysis Solution</strong></p>
    <p>Built with cutting-edge AI technology • Ready for enterprise deployment</p>
</div>
""", unsafe_allow_html=True)
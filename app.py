import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Ensure NLTK data is downloaded (runs quietly)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# ==========================================
# 1. SETUP & CONFIGURATION (CSS INJECTION)
# ==========================================
def setup_page():
    """Configures the Streamlit page and injects custom CSS to replace native styling."""
    st.set_page_config(page_title="Gadsby Analysis", page_icon="📚", layout="wide")
    
    custom_css = """
    <style>
        /* Hide the default Streamlit top menu and footer for a cleaner app look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main Typography and Headers */
        .main-title { 
            text-align: center; 
            color: #1a252f; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0px;
            padding-bottom: 5px;
        }
        .sub-title { 
            text-align: center; 
            color: #7f8c8d; 
            font-size: 1.2rem;
            padding-bottom: 30px; 
            font-style: italic;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .section-header { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 8px; 
            margin-top: 40px;
            margin-bottom: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
        }
        
        /* Custom Flexbox Layout for KPIs (Bypassing st.metric) */
        .kpi-container {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        
        /* Custom KPI Cards with Gradients and Hover Animations */
        .kpi-card {
            background: linear-gradient(135deg, #ffffff 0%, #f1f4f9 100%);
            border: 1px solid #e1e8ed;
            border-left: 5px solid #3498db;
            border-radius: 8px;
            padding: 25px 20px;
            flex: 1;
            min-width: 200px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        /* KPI Internal Typography */
        .kpi-title {
            font-size: 1rem;
            color: #95a5a6;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }
        .kpi-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            line-height: 1.2;
        }
        .kpi-delta {
            font-size: 1rem;
            font-weight: 600;
            margin-top: 5px;
        }
        .delta-good { color: #27ae60; }
        .delta-bad { color: #e74c3c; }
        
        /* Custom Divider */
        .custom-divider {
            height: 2px;
            background: linear-gradient(to right, transparent, #bdc3c7, transparent);
            margin: 40px 0;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 2. DATA PROCESSING
# ==========================================
@st.cache_data
def load_and_clean_text(filepath="Gadsby.txt"):
    """Loads text and returns raw text, sentences, and clean tokens."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
    except FileNotFoundError:
        return "", [], []

    sentences = [s.strip() for s in re.split(r'[.!?]+', raw_text) if len(s.strip()) > 0]
    tokens = re.findall(r'\b[a-zA-Z]+\b', raw_text)
    clean_tokens = [word.lower() for word in tokens]
    
    return raw_text, sentences, clean_tokens

# ==========================================
# 3. ANALYSIS FUNCTIONS
# ==========================================
def calculate_kpis(raw_text, clean_tokens):
    """Calculates top-level metrics."""
    total_words = len(clean_tokens)
    unique_words = len(set(clean_tokens))
    ttr = unique_words / total_words if total_words > 0 else 0
    
    chars = [char.lower() for char in raw_text if char.isalpha()]
    e_count = Counter(chars).get('e', 0)
    
    return total_words, unique_words, ttr, e_count

def get_word_lengths(clean_tokens):
    word_lengths = [len(word) for word in clean_tokens]
    return pd.DataFrame(list(Counter(word_lengths).items()), columns=['Length', 'Count']).sort_values('Length')

def get_sentence_lengths(sentences):
    return [len(re.findall(r'\b[a-zA-Z]+\b', sent)) for sent in sentences]

def get_pos_distribution(clean_tokens):
    pos_tags = nltk.pos_tag(clean_tokens)
    df_pos = pd.DataFrame(list(Counter([tag for _, tag in pos_tags]).items()), columns=['POS Tag', 'Count'])
    
    pos_mapping = {
        'NN': 'Noun', 'IN': 'Preposition', 'JJ': 'Adjective', 'DT': 'Determiner',
        'RB': 'Adverb', 'NNS': 'Noun (Plural)', 'VBD': 'Verb (Past)', 
        'VBG': 'Verb (-ing)', 'VB': 'Verb (Base)', 'CC': 'Conjunction'
    }
    df_pos['Readable Tag'] = df_pos['POS Tag'].map(lambda x: pos_mapping.get(x, x))
    return df_pos.sort_values('Count', ascending=False).head(10)

def get_sentiment_arc(sentences):
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(sent)['compound'] for sent in sentences]
    df_sent = pd.DataFrame({'Sentiment': sentiments})
    df_sent['Rolling Average'] = df_sent['Sentiment'].rolling(window=50, min_periods=1).mean()
    return df_sent

def generate_wordcloud_image(clean_tokens):
    stop_words = set(stopwords.words('english'))
    meaningful_words = [w for w in clean_tokens if w not in stop_words and 'e' not in w]
    text_for_cloud = ' '.join(meaningful_words)
    
    return WordCloud(width=1200, height=500, background_color='#ffffff', colormap='viridis', max_words=100).generate(text_for_cloud)

# ==========================================
# 4. UI RENDERING COMPONENTS
# ==========================================
def render_header_and_kpis(raw_text, clean_tokens):
    """Renders titles and pure HTML/CSS top metric cards."""
    st.markdown("<h1 class='main-title'>📖 Gadsby: The Lipogram Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>An interactive linguistic analysis of Ernest Vincent Wright's 50,000-word novel written entirely without the letter 'e'.</p>", unsafe_allow_html=True)

    if not raw_text:
        st.warning("Please ensure 'Gadsby.txt' is in the same directory to run the analysis.")
        return

    total_words, unique_words, ttr, e_count = calculate_kpis(raw_text, clean_tokens)
    
    # Logic for the "E" counter delta status
    e_delta_class = "delta-good" if e_count == 0 else "delta-bad"
    e_delta_text = "The Golden Rule intact" if e_count == 0 else "Uh oh! Rule broken."

    # Pure HTML/CSS structure to replace native st.metric columns
    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Total Words (Tokens)</div>
            <div class="kpi-value">{total_words:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Unique Words (Types)</div>
            <div class="kpi-value">{unique_words:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Lexical Diversity</div>
            <div class="kpi-value">{ttr:.4f}</div>
        </div>
        <div class="kpi-card" style="border-left-color: {'#27ae60' if e_count == 0 else '#e74c3c'};">
            <div class="kpi-title">Total 'E's Found</div>
            <div class="kpi-value">{e_count}</div>
            <div class="kpi-delta {e_delta_class}">{e_delta_text}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

def render_structural_charts(clean_tokens, sentences):
    """Renders charts relating to word and sentence structure."""
    if not clean_tokens: return
    
    c1, c2 = st.columns(2)
    sns.set_theme(style="whitegrid")

    with c1:
        st.markdown("<h3 class='section-header'>📏 Word Length Distribution</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=get_word_lengths(clean_tokens), x='Length', y='Count', ax=ax, palette="viridis")
        ax.set_xlabel("Number of Letters")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with c2:
        st.markdown("<h3 class='section-header'>📝 Sentence Length Distribution</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(get_sentence_lengths(sentences), bins=20, ax=ax, color="coral", kde=True)
        ax.set_xlabel("Number of Words per Sentence")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

def render_semantic_charts(clean_tokens, sentences):
    """Renders charts relating to grammar, emotion, and vocabulary."""
    if not clean_tokens: return
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<h3 class='section-header'>🧩 Part-of-Speech Anomalies</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=get_pos_distribution(clean_tokens), x='Readable Tag', y='Count', ax=ax, palette="mako")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with c2:
        st.markdown("<h3 class='section-header'>📈 The Emotional Arc (Rolling Sentiment)</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=get_sentiment_arc(sentences), x=get_sentiment_arc(sentences).index, y='Rolling Average', ax=ax, color="#e74c3c", linewidth=2)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel("Sentence Progression ->")
        ax.set_ylabel("Sentiment Polarity")
        st.pyplot(fig)

    st.markdown("<h3 class='section-header'>☁️ Core Vocabulary Word Cloud</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(generate_wordcloud_image(clean_tokens), interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    setup_page()
    
    # Load Data
    raw_text, sentences, clean_tokens = load_and_clean_text()
    
    # Render UI
    render_header_and_kpis(raw_text, clean_tokens)
    render_structural_charts(clean_tokens, sentences)
    render_semantic_charts(clean_tokens, sentences)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Built with Streamlit, NLTK & Seaborn. Data source: <em>Gadsby</em> by Ernest Vincent Wright.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

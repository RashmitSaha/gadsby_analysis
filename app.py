import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import base64

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from wordcloud import WordCloud

# Ensure NLTK data is downloaded (runs quietly)
nltk.download('all', quiet=True)

# ==========================================
# 0. HELPER FUNCTIONS
# ==========================================
def get_base64_of_bin_file(bin_file):
    """Reads a local image and converts it to a base64 string for CSS injection."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

# ==========================================
# 1. SETUP & CONFIGURATION (CSS INJECTION)
# ==========================================
def setup_page():
    """Configures the Streamlit page and injects custom CSS to replace native styling."""
    st.set_page_config(page_title="Gadsby Analysis", page_icon="📚", layout="wide")
    
    # Load the background image
    img_base64 = get_base64_of_bin_file("gadsby_image.png")
    bg_image_property = f"background-image: url('data:image/png;base64,{img_base64}');" if img_base64 else "background-color: #0f172a;"
    
    custom_css = f"""
    <style>
        /* Apply the base64 background image */
        .stApp {{
            {bg_image_property}
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Dark Glassmorphism container for better contrast */
        .block-container {{
            background-color: rgba(15, 23, 42, 0.85); /* Dark slate overlay */
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
            margin-top: 2rem;
            margin-bottom: 2rem;
            max-width: 1200px;
        }}

        /* Hide the default Streamlit top menu and footer */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{background: transparent !important;}}
        
        /* Main Typography and Headers */
        .main-title {{ 
            text-align: center; 
            color: #f8fafc !important; /* Pure white */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0px;
            padding-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        .sub-title {{ 
            text-align: center; 
            color: #94a3b8 !important; /* Slate gray */
            font-size: 1.2rem;
            padding-bottom: 30px; 
            font-style: italic;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .section-header {{ 
            color: #f1f5f9 !important; 
            border-bottom: 3px solid #3b82f6; 
            padding-bottom: 8px; 
            margin-top: 40px;
            margin-bottom: 10px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
        }}
        .section-desc {{
            color: #94a3b8 !important;
            font-size: 1rem;
            margin-bottom: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        /* Custom Flexbox Layout for KPIs */
        .kpi-container {{
            display: flex;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }}
        
        /* Dark Mode KPI Cards */
        .kpi-card {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid #334155;
            border-left: 5px solid #3b82f6;
            border-radius: 8px;
            padding: 25px 20px;
            flex: 1;
            min-width: 200px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.5);
        }}
        
        /* KPI Internal Typography */
        .kpi-title {{
            font-size: 1rem;
            color: #94a3b8 !important;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #f8fafc !important;
            line-height: 1.2;
        }}
        .kpi-delta {{
            font-size: 1rem;
            font-weight: 600;
            margin-top: 5px;
        }}
        .delta-good {{ color: #4ade80 !important; }}
        .delta-bad {{ color: #f87171 !important; }}
        
        /* Custom Divider */
        .custom-divider {{
            height: 2px;
            background: linear-gradient(to right, transparent, #334155, transparent);
            margin: 40px 0;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 2. DATA PROCESSING
# ==========================================
@st.cache_data
def load_and_clean_text(filepath="Gadsby.txt"):
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

@st.cache_data
def get_bigram_metrics(clean_tokens):
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(clean_tokens)
    finder.apply_freq_filter(5)
    
    freq_bigrams = finder.ngram_fd.most_common(15)
    df_freq = pd.DataFrame([
        (f"{bg[0]} {bg[1]}", count) for bg, count in freq_bigrams
    ], columns=['Bigram', 'Frequency'])
    
    pmi_scored = finder.score_ngrams(bigram_measures.pmi)[:15]
    df_pmi = pd.DataFrame([
        (f"{bg[0]} {bg[1]}", score) for bg, score in pmi_scored
    ], columns=['Bigram', 'PMI Score'])
    
    return df_freq, df_pmi

def generate_wordcloud_image(clean_tokens):
    stop_words = set(stopwords.words('english'))
    meaningful_words = [w for w in clean_tokens if w not in stop_words and 'e' not in w]
    text_for_cloud = ' '.join(meaningful_words)
    
    # Updated wordcloud to use a transparent background to blend with dark mode
    return WordCloud(width=1200, height=500, background_color=None, mode="RGBA", colormap='Blues', max_words=100).generate(text_for_cloud)

# ==========================================
# 4. CHART THEME CONFIGURATION
# ==========================================
def set_dark_chart_theme():
    """Configures Seaborn/Matplotlib for dark backgrounds with white text."""
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": (0, 0, 0, 0),
        "figure.facecolor":(0, 0, 0, 0),
        "text.color": "#f8fafc",
        "axes.labelcolor": "#f8fafc",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155"
    })

# ==========================================
# 5. UI RENDERING COMPONENTS
# ==========================================
def render_header_and_kpis(raw_text, clean_tokens):
    st.markdown("<h1 class='main-title'>📖 Gadsby: The Lipogram Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>An interactive linguistic analysis of Ernest Vincent Wright's 50,000-word novel written entirely without the letter 'e'.</p>", unsafe_allow_html=True)

    if not raw_text:
        st.warning("Please ensure 'Gadsby.txt' is in the same directory to run the analysis.")
        return

    total_words, unique_words, ttr, e_count = calculate_kpis(raw_text, clean_tokens)
    
    e_delta_class = "delta-good" if e_count == 0 else "delta-bad"
    e_delta_text = "The Golden Rule intact" if e_count == 0 else "Uh oh! Rule broken."

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
        <div class="kpi-card" style="border-left-color: {'#4ade80' if e_count == 0 else '#f87171'};">
            <div class="kpi-title">Total 'E's Found</div>
            <div class="kpi-value">{e_count}</div>
            <div class="kpi-delta {e_delta_class}">{e_delta_text}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

def render_structural_charts(clean_tokens, sentences):
    if not clean_tokens: return
    
    c1, c2 = st.columns(2)
    set_dark_chart_theme()

    with c1:
        st.markdown("<h3 class='section-header'>📏 Word Length Distribution</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=get_word_lengths(clean_tokens), x='Length', y='Count', ax=ax, palette="mako")
        ax.set_xlabel("Number of Letters")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with c2:
        st.markdown("<h3 class='section-header'>📝 Sentence Length Distribution</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(get_sentence_lengths(sentences), bins=20, ax=ax, color="#fb923c", kde=True)
        ax.set_xlabel("Number of Words per Sentence")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

def render_semantic_charts(clean_tokens, sentences):
    if not clean_tokens: return
    
    c1, c2 = st.columns(2)
    set_dark_chart_theme()

    with c1:
        st.markdown("<h3 class='section-header'>🧩 Part-of-Speech Anomalies</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=get_pos_distribution(clean_tokens), x='Readable Tag', y='Count', ax=ax, palette="viridis")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with c2:
        st.markdown("<h3 class='section-header'>📈 The Emotional Arc</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=get_sentiment_arc(sentences), x=get_sentiment_arc(sentences).index, y='Rolling Average', ax=ax, color="#f87171", linewidth=2)
        ax.axhline(0, color='#94a3b8', linestyle='--', linewidth=1)
        ax.set_xlabel("Sentence Progression ->")
        ax.set_ylabel("Sentiment Polarity")
        st.pyplot(fig)

def render_ngram_charts(clean_tokens):
    if not clean_tokens: return
    
    st.markdown("<h3 class='section-header'>🔗 Collocations & N-grams (Bigrams)</h3>", unsafe_allow_html=True)
    st.markdown("<p class='section-desc'>Comparing words that frequently appear together (Frequency) vs. words that have the strongest unique statistical associations (PMI).</p>", unsafe_allow_html=True)
    
    df_freq, df_pmi = get_bigram_metrics(clean_tokens)
    set_dark_chart_theme()
    
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df_freq, y='Bigram', x='Frequency', ax=ax1, palette="flare")
        ax1.set_title("Top 15 Bigrams by Frequency", fontsize=12, fontweight='bold', color="#f8fafc", pad=10)
        ax1.set_ylabel("")
        st.pyplot(fig1)
        
    with c2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df_pmi, y='Bigram', x='PMI Score', ax=ax2, palette="crest")
        ax2.set_title("Top 15 Bigrams by PMI (Strongest Associations)", fontsize=12, fontweight='bold', color="#f8fafc", pad=10)
        ax2.set_ylabel("")
        st.pyplot(fig2)

def render_wordcloud(clean_tokens):
    if not clean_tokens: return
    
    st.markdown("<h3 class='section-header'>☁️ Core Vocabulary Word Cloud</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(15, 6), facecolor=(0,0,0,0)) # Transparent figure background
    ax.imshow(generate_wordcloud_image(clean_tokens), interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    setup_page()
    
    raw_text, sentences, clean_tokens = load_and_clean_text()
    
    render_header_and_kpis(raw_text, clean_tokens)
    render_structural_charts(clean_tokens, sentences)
    render_semantic_charts(clean_tokens, sentences)
    render_ngram_charts(clean_tokens)
    render_wordcloud(clean_tokens)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-weight: bold;'>Built with Streamlit, NLTK & Seaborn. Data source: <em>Gadsby</em> by Ernest Vincent Wright.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# 📖 Gadsby: A Lipogram Linguistic Analysis Dashboard

## 📌 Project Overview
This project provides an interactive linguistic analysis of Ernest Vincent Wright's 1939 novel, *Gadsby*. The novel is famous for being a **lipogram**—a 50,000-word text written entirely without the letter 'e'. 

Because the letter 'e' is the most common letter in the English language and is heavily utilized in standard verb conjugation (e.g., past tense *-ed*) and common pronouns/articles (e.g., *the, he, she*), its absence forces massive structural, syntactic, and semantic shifts in the prose. This Streamlit dashboard uses Natural Language Processing (NLP) tools to visualize and quantify these linguistic anomalies.

## ✨ Features
The dashboard processes the raw text and generates several key insights:

*   **Golden Rule Verification:** A dynamic KPI counter that mathematically verifies if any 'e's slipped past the author's constraint.
*   **Structural Distributions:** Visualizes word length and sentence length distributions to see if the constraint forced shorter or longer phrasings.
*   **Part-of-Speech (POS) Anomalies:** Highlights the grammatical crutches the author relied on (e.g., increased use of auxiliary verbs and present participles to avoid standard past-tense verbs).
*   **The Emotional Arc:** A rolling 50-sentence sentiment analysis using VADER to chart the tonal progression of the novel.
*   **Core Vocabulary Word Cloud:** A stylized visualization of the text's most dominant themes, filtered for standard English stopwords.

## 🛠️ Tech Stack
*   **Frontend:** [Streamlit](https://streamlit.io/) (with custom CSS styling)
*   **Data Manipulation:** Pandas
*   **NLP Processing:** [NLTK](https://www.nltk.org/) (VADER Sentiment, POS Tagger, Tokenizer), WordCloud
*   **Data Visualization:** Matplotlib, Seaborn

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/RashmitSaha/gadsby-analysis.git
cd gadsby-analysis
```

**2. Install dependencies**
Make sure you have Python installed. You can install the required packages using pip:
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

**3. Add the Data**
Ensure you have the text file of the novel named exactly `Gadsby.txt` in the root directory of the project. You can also get them by running the notebooks in the repository and uploading the `Gadsby.pdf` file. It is preferable that you run the notebooks in Google Colab.

**4. Run the Application**
Launch the Streamlit server from your terminal:
```bash
streamlit run app.py
```

## 📂 Project Structure
```text
gadsby-analysis/
│
├── app.py             # Main Streamlit application and UI rendering
├── Gadsby.txt         # The raw text data of the novel (Not included in repo by default)
└── README.md          # Project documentation
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! If you want to add more advanced NLP metrics (like N-gram collocation finding or syntactic dependency parsing), feel free to fork the repository and submit a pull request.

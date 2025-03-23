"""
News analysis utilities including:
- News fetching and scraping (via NewsAPI)
- Sentiment analysis
- Topic extraction
- Translation and TTS
"""

import requests
from bs4 import BeautifulSoup
from transformers import pipeline, VitsModel, AutoTokenizer
import spacy
from googletrans import Translator
import soundfile as sf
from typing import List, Dict
import tempfile
import torch

# Initialize models once
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)
nlp = spacy.load("en_core_web_sm")
translator = Translator()
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-hin")

def fetch_news_articles(company: str, api_key: str, max_results: int = 10) -> List[Dict]:
    """Fetch news articles using NewsAPI."""
    news_api_url = "https://newsapi.org/v2/everything"
    keywords = "stock OR market OR shares OR revenue OR earnings OR merger OR acquisition"
    params = {
        "q": f'"{company}" AND ({keywords})',
        "language": "en",
        "pageSize": max_results,
        "sortBy": "publishedAt",
        "apiKey": api_key  # Use the provided API key
    }
    response = requests.get(news_api_url, params=params)
    data = response.json()
    return data.get("articles", [])

def scrape_article_content(url: str) -> dict:
    """Extract title and content from the article's webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract title
        title = soup.title.text.strip() if soup.title else "No title available"
        
        # Extract content
        content = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            content = meta_desc.get("content").strip()
        else:
            og_desc = soup.find("meta", property="og:description")
            if og_desc and og_desc.get("content"):
                content = og_desc.get("content").strip()
        
        # Fallback to body text
        if not content or len(content) < 100:
            article_body = soup.find("article") or soup
            paragraphs = article_body.find_all("p")
            content = "\n".join(p.get_text(strip=True) for p in paragraphs)
        
        return {"title": title, "content": content or "No content available."}
    
    except Exception as e:
        return {"title": "Error", "content": f"Error scraping article: {e}"}
    
def analyze_sentiment(text: str) -> str:
    """Perform sentiment analysis on text"""
    try:
        result = sentiment_analyzer(text[:512])[0]
        label_map = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }
        return label_map.get(result['label'], 'Neutral')
    except Exception as e:
        raise RuntimeError(f"Sentiment analysis failed: {str(e)}")

def extract_key_topics(text: str, max_topics: int = 5) -> List[str]:
    """Extract key topics using spaCy NER"""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'NORP']]
    return list(set(entities))[:max_topics]

def generate_hindi_tts(text: str) -> str:
    """Generate Hindi speech from text"""
    try:
        translated = translator.translate(text, dest='hi').text
        with torch.no_grad():
            inputs = tts_tokenizer(translated, return_tensors="pt")
            waveform = tts_model(**inputs).waveform
            audio_array = waveform.cpu().detach().numpy().T
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_array, tts_model.config.sampling_rate)
            return f.name
    except Exception as e:
        raise RuntimeError(f"TTS generation failed: {str(e)}")

"""if __name__ == "__main__":
    # Prompt for company name and fetch articles using NewsAPI
    company = input("Enter company name to search for business affecting news: ").strip()
    API_KEY = "8a22294933bb4cdfbef6ae2d15955be7"
    articles = fetch_news_articles(company, API_KEY)
    
    # Process each article
    for article in articles:
        title = article.get("title", "No title")
        description = article.get("description", "No description")
        url = article.get("url", "")
        published_at = article.get("publishedAt", "No published date")
        
        print("=" * 80)
        print(f"Title: {title}")
        print(f"Published At: {published_at}")
        print(f"API Description: {description}")
        print(f"URL: {url}")
        
        # Scrape article page for a complete summary
        complete_summary = get_complete_summary(url)
        print("Complete Summary:")
        print(complete_summary)
        
        # Perform sentiment analysis and topic extraction on the complete summary
        sentiment = analyze_sentiment(complete_summary)
        topics = extract_key_topics(complete_summary)
        print("Sentiment:", sentiment)
        print("Key Topics:", topics)
        
        # Generate Hindi audio from the complete summary
        tts_path = generate_hindi_tts(complete_summary)
        print("Hindi TTS Audio Path:", tts_path)
        
        print("=" * 80)
        print()"""

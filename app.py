"""
Streamlit frontend for news analysis application
Communicates with FastAPI backend via REST calls
"""

import streamlit as st
import requests
import os

# Configure API endpoint
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="News Analyzer", layout="wide")

def main():
    st.title("📰 Company News Analyzer")
    
    company = st.text_input("Enter company name", "Tesla")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing news articles..."):
            try:
                # Call FastAPI backend
                response = requests.get(f"{API_BASE}/analyze?company={company}")
                response.raise_for_status()
                data = response.json()

                # Display results
                st.header(f"Analysis for {data['company']}")
                
                # Show articles
                st.subheader("📋 Processed Articles")
                for idx, article in enumerate(data['articles']):
                    with st.expander(f"Article {idx+1}: {article['title']}"):
                        st.markdown(f"""
                        **Summary:** {article['summary']}  
                        **Sentiment:** {article['sentiment']}  
                        **Key Topics:** {", ".join(article['topics'])}
                        """)

                # Show TTS player
                st.subheader("🇮🇳 Hindi Summary")
                tts_response = requests.get(f"{API_BASE}{data['tts_url']}")
                st.audio(tts_response.content, format="audio/wav")

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {str(e)}")

if __name__ == "__main__":
    main()

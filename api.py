import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.logger import logger
from utils import (
    fetch_news_articles,
    scrape_article_content,
    analyze_sentiment,
    extract_key_topics,
    generate_hindi_tts
)

app = FastAPI(title="News Analyzer API")
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/analyze")
async def analyze_company_news(
    company: str, 
    background_tasks: BackgroundTasks
):
    """Main analysis endpoint"""
    try:
        news_items = fetch_news_articles(company)
        processed_articles = []

        for item in news_items:
            try:
                if not item['url'].startswith('http'):
                    continue
                
                article = scrape_article_content(item['url'])
                if not article['content']:
                    continue
                    
                processed_articles.append({
                    'title': article['title'],
                    'summary': article['summary'],
                    'sentiment': analyze_sentiment(article['content']),
                    'topics': extract_key_topics(article['content'])
                })
            except Exception as e:
                logger.error(f"Article processing failed: {str(e)}")
                continue

        if not processed_articles:
            raise HTTPException(
                status_code=404,
                detail="No articles could be processed"
            )

        tts_summary = f"{company} analysis: {len(processed_articles)} articles processed"
        tts_path = await asyncio.get_event_loop().run_in_executor(
            executor, 
            generate_hindi_tts, 
            tts_summary
        )

        background_tasks.add_task(os.remove, tts_path)
        
        return {
            'company': company,
            'articles': processed_articles,
            'tts_path': tts_path
        }
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
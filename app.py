import streamlit as st
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt

headers = {'User-Agent': 'Mozilla/5.0'}

# ====== SCRAPING ======
async def fetch(session, url):
    """Fetch content from a URL."""
    async with session.get(url) as response:
        return await response.text()

async def get_total_reviews(base_url):
    """Get total number of reviews from IMDb page."""
    async with aiohttp.ClientSession(headers=headers) as session:
        content = await fetch(session, base_url)
        soup = BeautifulSoup(content, 'html.parser')
        total_reviews_element = soup.find('div', {'data-testid': 'tturv-total-reviews'})
        if total_reviews_element:
            total_reviews_text = total_reviews_element.text.split()[0].replace(',', '').strip()
            return int(total_reviews_text) if total_reviews_text.isdigit() else 0
        return 0

async def fetch_review_detail(session, url):
    """Fetch individual review details."""
    content = await fetch(session, url)
    soup = BeautifulSoup(content, 'html.parser')
    review = soup.find('div', {'class': 'text show-more__control'})
    return review.text.strip() if review else "Review not found"

async def scrape_reviews(base_url):
    """Scrape reviews from IMDb."""
    reviews = []
    async with aiohttp.ClientSession(headers=headers) as session:
        total_reviews = await get_total_reviews(base_url)
        page = 1
        review_urls = []

        while len(review_urls) < total_reviews:
            content = await fetch(session, f"{base_url}?paginationKey={page}")
            soup = BeautifulSoup(content, 'html.parser')
            review_cards = soup.find_all('article', class_='sc-d99cd751-1 kzUfxa user-review-item')
            if not review_cards:
                break

            for card in review_cards:
                user = card.find('ul', class_='ipc-inline-list').li.text.strip()
                date = card.find('li', class_='ipc-inline-list__item review-date').text.strip()
                link = card.find('a', class_='ipc-title-link-wrapper').get('href')
                review_url = f"https://www.imdb.com{link}"
                review_urls.append({'user': user, 'date': date, 'url': review_url})
            page += 1
            await asyncio.sleep(0.5)

        tasks = [fetch_review_detail(session, item['url']) for item in review_urls]
        reviews_text = await asyncio.gather(*tasks)
        for i, item in enumerate(review_urls):
            reviews.append({'User': item['user'], 'Date': item['date'], 'Review': reviews_text[i]})
    return pd.DataFrame(reviews)

# ====== PREPROCESSING ======
def decontracted(phrase):
    """Expand contractions."""
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_text(text_data):
    """Clean and preprocess text."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_text = []
    for sentence in text_data:
        sentence = decontracted(sentence)
        sentence = re.sub(r"[^A-Za-z0-9]+", " ", sentence)
        sentence = re.sub(r"\b\w{1,2}\b", " ", sentence)
        sentence = re.sub(r"\d+", "", sentence)
        tokens = word_tokenize(sentence.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        preprocessed_text.append(" ".join(tokens).strip())
    return preprocessed_text

# ====== SENTIMENT ANALYSIS ======
def analyze_sentiment(text):
    """Perform sentiment analysis using TextBlob."""
    polarity = TextBlob(text).sentiment.polarity
    label = 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
    return label, polarity

# ====== STREAMLIT APP ======
st.title("Analisis Sentimen Review Film Pada IMDb")

imdb_link = st.sidebar.text_input("Enter IMDb Reviews URL", "")
if st.sidebar.button("Analyze"):
    st.write("Scraping reviews...")
    df = asyncio.run(scrape_reviews(imdb_link))
    df['Processed_Review'] = preprocess_text(df['Review'])
    sentiments = df['Processed_Review'].apply(analyze_sentiment)
    df['Sentiment'] = sentiments.apply(lambda x: x[0])
    df['Sentiment_Score'] = sentiments.apply(lambda x: x[1])
    st.write("Analysis complete!")
    st.dataframe(df)

    df.to_csv("imdb_reviews_with_sentiment.csv", index=False)
    
    sentiment_counts = df['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

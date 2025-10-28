import requests
import pandas as pd
import os

# -----------------------------
# 🔑 1. Enter your NewsData.io API key
# -----------------------------
API_KEY = "pub_b10c10920310401d8196f1709d0eda40"  # 👈 paste your key here

# -----------------------------
# 🌍 2. Choose category or topic
# -----------------------------
TOPICS = ["technology", "politics", "science", "health", "business"]

# -----------------------------
# 📥 3. Fetch latest real news
# -----------------------------
def fetch_real_news(api_key, topic, limit=10):
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={topic}&language=en&country=us"
    try:
        response = requests.get(url)
        data = response.json()

        articles = []
        for article in data.get("results", [])[:limit]:
            title = article.get("title", "")
            text = article.get("description", "") or article.get("content", "")
            date = article.get("pubDate", "")
            articles.append({"title": title, "text": text, "subject": topic, "date": date})

        return articles
    except Exception as e:
        print(f" Error fetching {topic} news:", e)
        return []

# -----------------------------
# 🧩 4. Combine with existing True.csv
# -----------------------------
def update_true_csv(file_path="True.csv"):
    all_articles = []
    for topic in TOPICS:
        print(f" Fetching real news for topic: {topic} ...")
        articles = fetch_real_news(API_KEY, topic)
        all_articles.extend(articles)

    if not all_articles:
        print(" No articles fetched — check your API key or internet connection.")
        return

    new_df = pd.DataFrame(all_articles)

    # Load existing file (if it exists)
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    # Save updated CSV
    updated_df.to_csv(file_path, index=False)
    print(f" Successfully updated {file_path} with {len(new_df)} new real news entries!")

# -----------------------------
# 🚀 5. Run the update
# -----------------------------
if __name__ == "__main__":
    update_true_csv()


import time

def update_true_csv():
    # your fetch & update logic here
    pass

while True:
    print("\n Updating real news dataset...")
    update_true_csv()
    print(" Done! Next update in 24 hours.")
    time.sleep(24 * 60 * 60)


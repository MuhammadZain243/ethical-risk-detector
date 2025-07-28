import requests
from bs4 import BeautifulSoup
import time
import os

BASE_URL = "https://www.gov.uk"
SEARCH_URL = "https://www.gov.uk/search/all"

KEYWORDS = [
    "AI ethics", "Responsible AI", "automated decision-making",
    "algorithm transparency", "AI procurement", "AI fairness"
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

def search_documents(keyword, max_results=10):
    print(f"Searching for keyword: {keyword}")
    params = {
        "q": keyword,
        "content_store_document_type": "publication",
        "order": "relevance"
    }
    results = []
    r = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    
    if r.status_code != 200:
        print(f"Error: Failed to retrieve search results for '{keyword}'")
        return results

    soup = BeautifulSoup(r.content, "html.parser")
    links = soup.select("li.gem-c-document-list__item a")

    for link in links[:max_results]:
        title = link.text.strip()
        url = BASE_URL + link['href']
        results.append({"title": title, "url": url})
    
    return results

def scrape_text_from_page(url):
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.content, "html.parser")
    article = soup.select_one("main")
    
    if not article:
        return None
    
    paragraphs = [p.get_text(strip=True) for p in article.find_all("p")]
    return "\n".join(paragraphs)

def save_scraped_documents(keyword, articles, save_dir="data/raw/govuk"):
    os.makedirs(save_dir, exist_ok=True)
    for i, entry in enumerate(articles):
        content = scrape_text_from_page(entry["url"])
        
        if content:
            filename = f"{save_dir}/{keyword.replace(' ', '_')}_{i}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"✅ Saved: {filename}")
        time.sleep(1.5)

if __name__ == "__main__":
    for kw in KEYWORDS:
        results = search_documents(kw, max_results=10)
        save_scraped_documents(kw, results)

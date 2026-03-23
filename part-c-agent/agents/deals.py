"""Deal data models and RSS feed scraping utilities."""

import re
import time
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import feedparser
import requests
from tqdm import tqdm

# RSS feed sources for deal monitoring
RSS_FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]


def extract_html_text(html_snippet: str) -> str:
    """
    Extract and clean text from HTML snippet.

    Args:
        html_snippet: HTML content to extract text from

    Returns:
        Cleaned text string
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet

    return result.replace("\n", " ")


class ScrapedDeal:
    """Represents a deal scraped from an RSS feed."""

    def __init__(self, entry: Dict):
        """Initialize from RSS feed entry."""
        self.title = entry["title"][:100]
        self.summary = extract_html_text(entry["summary"])
        self.url = entry["links"][0]["href"]
        self.details = ""
        self.features = ""

        try:
            response = requests.get(self.url, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
            content_section = soup.find("div", class_="content-section")
            if content_section:
                content = content_section.get_text()
                content = content.replace("\nmore", "").replace("\n", " ")
                if "Features" in content:
                    self.details, self.features = content.split("Features", 1)
                    self.details = self.details[:500]
                    self.features = self.features[:500]
                else:
                    self.details = content[:500]
        except Exception:
            self.details = self.summary[:500]

    def describe(self) -> str:
        """Return formatted description for model input."""
        return (
            f"Title: {self.title}\n"
            f"Details: {self.details.strip()}\n"
            f"Features: {self.features.strip()}\n"
            f"URL: {self.url}"
        )

    def __repr__(self) -> str:
        return f"<ScrapedDeal: {self.title}>"

    @classmethod
    def fetch(cls, show_progress: bool = False, max_per_feed: int = 10) -> List["ScrapedDeal"]:
        """
        Fetch deals from RSS feeds.

        Args:
            show_progress: Show progress bar
            max_per_feed: Maximum deals per feed

        Returns:
            List of scraped deals
        """
        deals = []
        feed_iter = tqdm(RSS_FEEDS) if show_progress else RSS_FEEDS

        for feed_url in feed_iter:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_per_feed]:
                    deals.append(cls(entry))
                    time.sleep(0.05)
            except Exception as e:
                print(f"Error fetching feed {feed_url}: {e}")

        return deals


class Deal(BaseModel):
    """Represents a deal with product description and price."""

    product_description: str = Field(
        description="Clear summary of the product in 3-4 sentences. Focus on item details, not discounts."
    )
    price: float = Field(
        description="Actual price of the product. If deal says '$100 off $300', use $200."
    )
    url: str = Field(description="URL of the deal")


class DealSelection(BaseModel):
    """Represents a selection of deals from scanning."""

    deals: List[Deal] = Field(
        description="Selection of deals with detailed descriptions and clear prices"
    )


class Opportunity(BaseModel):
    """Represents a deal opportunity with price estimate and discount."""

    deal: Deal
    estimate: float
    discount: float

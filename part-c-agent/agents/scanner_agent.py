"""Scanner agent for identifying and filtering deals from RSS feeds."""

import re
from typing import List, Optional
from .agent import Agent
from .deals import ScrapedDeal, DealSelection, Deal
from .utils import create_agent_model


class ScannerAgent(Agent):
    """Agent that scans RSS feeds and identifies promising deals."""

    name = "Scanner Agent"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, selecting deals that have the most detailed, high quality description and the most clear price.

Respond strictly in JSON format. Provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.

Most important is that you respond with the 5 deals that have the most detailed product description with price. Focus on a thorough description of the product itself, not the terms of the deal.

Be careful with products described as "$XXX off" or "reduced by $XXX" - this isn't the actual price. Only respond with products when you are highly confident about the price."""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.

Rephrase the description to be a summary of the product itself, not the terms of the deal. Remember to respond with a short paragraph of text in the product_description field for each of the 5 items.

Be careful with products described as "$XXX off" or "reduced by $XXX" - this isn't the actual price. Only respond with products when you are highly confident about the price.

Deals:

"""

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    def __init__(self, model_provider: str = "openrouter", model_name: Optional[str] = None):
        """
        Initialize scanner agent.

        Args:
            model_provider: Model provider (openrouter or ollama)
            model_name: Specific model name to use
        """
        super().__init__()
        self.log("Initializing Scanner Agent")
        self.model = create_agent_model(provider=model_provider, model_name=model_name)
        self.log("Scanner Agent ready")

    def fetch_deals(self, memory: List[str]) -> List[ScrapedDeal]:
        """
        Fetch new deals from RSS feeds not already in memory.

        Args:
            memory: List of URLs already processed

        Returns:
            List of new scraped deals
        """
        self.log("Fetching deals from RSS feeds")
        scraped = ScrapedDeal.fetch(show_progress=False)
        result = [deal for deal in scraped if deal.url not in memory]
        self.log(f"Found {len(result)} new deals not in memory")
        return result

    def _make_user_prompt(self, scraped: List[ScrapedDeal]) -> str:
        """Create user prompt from scraped deals."""
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([deal.describe() for deal in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def _extract_price(self, text: str) -> float:
        """Extract price from text."""
        text = text.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else 0.0

    def _parse_deal_selection(self, response_text: str, scraped_deals: List[ScrapedDeal]) -> Optional[DealSelection]:
        """Parse model response into DealSelection."""
        try:
            import json
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            data = json.loads(response_text)
            deals = []
            url_map = {deal.url: deal for deal in scraped_deals}

            for deal_data in data.get("deals", []):
                if "product_description" in deal_data and "price" in deal_data:
                    price = deal_data["price"]
                    if isinstance(price, str):
                        price = self._extract_price(price)
                    if price > 0:
                        url = deal_data.get("url", "")
                        if url in url_map:
                            deals.append(
                                Deal(
                                    product_description=deal_data["product_description"],
                                    price=float(price),
                                    url=url,
                                )
                            )

            if deals:
                return DealSelection(deals=deals[:5])
        except Exception as e:
            self.log(f"Error parsing deal selection: {e}", "WARNING")

        return None

    def scan(self, memory: List[str] = None) -> Optional[DealSelection]:
        """
        Scan for deals and return selection.

        Args:
            memory: List of URLs already processed

        Returns:
            DealSelection if deals found, None otherwise
        """
        if memory is None:
            memory = []

        scraped = self.fetch_deals(memory)
        if not scraped:
            self.log("No new deals found")
            return None

        user_prompt = self._make_user_prompt(scraped[:20])
        self.log("Calling model to select best deals")

        try:
            # Use the model's direct prediction method which returns text
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
            response_text = self.model.predict_price(full_prompt)

            selection = self._parse_deal_selection(response_text, scraped)
            if selection:
                valid_deals = [deal for deal in selection.deals if deal.price > 0]
                if valid_deals:
                    selection.deals = valid_deals[:5]
                    self.log(f"Selected {len(selection.deals)} deals with price > 0")
                    return selection
        except Exception as e:
            self.log(f"Error during scanning: {e}", "ERROR")

        return None

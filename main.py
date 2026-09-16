"""
Simple Finnhub News Chatbot
---------------------------
A single Pydantic AI agent with one tool: `get_company_news`.
The tool calls Finnhub's /api/v1/company-news endpoint, and the agent
answers the user's question using only the data that endpoint returns.
"""

import os
import asyncio
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

load_dotenv()



# Dependencies: things the tool needs at runtime (API key + shared HTTP client)

@dataclass
class Deps:
    finnhub_api_key: str
    http_client: httpx.AsyncClient



# Agent definition

agent = Agent(
    "google:gemini-2.5-flash",
    deps_type=Deps,
    instructions=(
        "You are a financial news assistant. You have exactly one tool, "
        "`get_company_news`, which fetches recent news for a stock ticker "
        "from Finnhub. Always call the tool before answering any question "
        "about a company's news. Base your answer ONLY on the articles "
        "returned by the tool — do not invent headlines, dates, or facts. "
        "If the tool returns no articles for the requested ticker or date "
        "range, say so plainly instead of guessing. When you mention an "
        "article, you may cite its source and a link if the user would "
        "find that useful."
    ),
)



# The single tool

@agent.tool
async def get_company_news(
    ctx: RunContext[Deps],
    ticker: str,
    days_back: int = 7,
) -> list[dict]:
    """Fetch recent news for a stock ticker from Finnhub.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL", "TSLA", "NVDA".
        days_back: How many days of history to fetch, default 7.

    Returns:
        A list of news articles, each with: category, datetime (unix
        timestamp), headline, id, image, related (ticker), source,
        summary, url. Returns an empty list if there is no news.
    """
    today = date.today()
    from_date = today - timedelta(days=days_back)

    response = await ctx.deps.http_client.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": ticker.upper(),
            "from": from_date.isoformat(),
            "to": today.isoformat(),
            "token": ctx.deps.finnhub_api_key,
        },
    )
    response.raise_for_status()
    articles = response.json()

    # Keep the payload lean — trim to the fields the agent actually needs,
    # and cap the count so we don't blow the model's context on a busy ticker.
    trimmed = [
        {
            "headline": a.get("headline"),
            "summary": a.get("summary"),
            "source": a.get("source"),
            "datetime": a.get("datetime"),
            "url": a.get("url"),
            "related": a.get("related"),
        }
        for a in articles[:15]
    ]
    return trimmed


# Simple CLI chat loop
# ---------------------------------------------------------------------------
async def main() -> None:
    finnhub_key = os.environ["FINNHUB_API_KEY"]

    async with httpx.AsyncClient(timeout=10) as client:
        deps = Deps(finnhub_api_key=finnhub_key, http_client=client)

        print("Finnhub News Chatbot — ask about a company's recent news.")
        print("Type 'exit' to quit.\n")

        message_history = None
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            result = await agent.run(
                user_input, deps=deps, message_history=message_history
            )
            print(f"\nBot: {result.output}\n")
            message_history = result.all_messages()


if __name__ == "__main__":
    asyncio.run(main())

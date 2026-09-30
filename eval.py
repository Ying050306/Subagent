"""
Evaluation script for the Finnhub News Chatbot, using Langfuse.

What this does:
  1. Enables Langfuse tracing for the Pydantic AI agent (every tool call,
     prompt, and response gets logged automatically).
  2. Pushes the 10 test questions to Langfuse as a "Dataset" — a fixed
     benchmark we can re-run every time we change the prompt or model.
  3. Runs the agent against every question in the dataset and records
     each trace, plus which ticker the tool was actually called with.

"""

import asyncio
import json
import os

import httpx
from dotenv import load_dotenv
from langfuse import get_client
from pydantic_ai import Agent

from main import agent, Deps

load_dotenv()

DATASET_NAME = "finnhub-news-chatbot-qna"

CACHE_PATH = "eval_cache.json"


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# Groq's free tier for openai/gpt-oss-120b is 30 requests/minute and
# 1,000 requests/day
MODEL_MIN_INTERVAL_SECONDS = 2.5
_rate_limit_lock = asyncio.Lock()
_last_call_time = 0.0


async def _throttle() -> None:
    """Block until at least MODEL_MIN_INTERVAL_SECONDS has passed since
    the last call. Safe to call from multiple tasks — the lock serializes
    them even if run_experiment's concurrency is set above 1."""
    global _last_call_time
    async with _rate_limit_lock:
        now = asyncio.get_event_loop().time()
        wait = _last_call_time + MODEL_MIN_INTERVAL_SECONDS - now
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_time = asyncio.get_event_loop().time()

# The 10 QnA from test_qna.md, turned into structured test cases.
# expected_tool_called / expected_ticker let us check the agent's
# *trajectory* (did it call the right tool, with the right ticker),
# separately from judging the *content* of its final answer.
TEST_CASES = [
    {"id": "q01", "question": "What's the latest news about Tesla?",
     "expected_tool_called": True, "expected_ticker": "TSLA"},
    {"id": "q02", "question": "Summarize the recent news on Apple.",
     "expected_tool_called": True, "expected_ticker": "AAPL"},
    {"id": "q03", "question": "Has there been any news about Amazon this week?",
     "expected_tool_called": True, "expected_ticker": "AMZN"},
    {"id": "q04", "question": "Where did the latest Microsoft news come from?",
     "expected_tool_called": True, "expected_ticker": "MSFT"},
    {"id": "q05", "question": "Give me a link to the most recent Nvidia article.",
     "expected_tool_called": True, "expected_ticker": "NVDA"},
    {"id": "q06", "question": "Is there any news about XOM in the last 3 days?",
     "expected_tool_called": True, "expected_ticker": "XOM"},
    {"id": "q07", "question": "What are people saying about Meta lately — good or bad?",
     "expected_tool_called": True, "expected_ticker": "META"},
    {"id": "q08", "question": "How many news articles came out about Google in the past month?",
     "expected_tool_called": True, "expected_ticker": "GOOGL"},
    {"id": "q09", "question": "What's the oldest news article you have on Netflix from this month?",
     "expected_tool_called": True, "expected_ticker": "NFLX"},
    {"id": "q10", "question": "Tell me about the news for a stock ticker that doesn't exist, like ZZZZ.",
     "expected_tool_called": True, "expected_ticker": "ZZZZ"},
]


def push_dataset(langfuse) -> None:
    """Create the dataset in Langfuse. Each item has a fixed id (q01..q10),
    so re-running this script updates the same 10 items instead of piling
    up duplicates."""
    langfuse.create_dataset(name=DATASET_NAME)
    for case in TEST_CASES:
        langfuse.create_dataset_item(
            id=case["id"],
            dataset_name=DATASET_NAME,
            input={"question": case["question"]},
            expected_output={
                "expected_tool_called": case["expected_tool_called"],
                "expected_ticker": case["expected_ticker"],
            },
        )


async def run_agent(item, finnhub_key: str, langfuse, cache: dict):
    """The 'task' Langfuse runs once per dataset item.

    Runs our actual agent (imported from main.py) against the question,
    and records the tool call trajectory + final answer as trace data
    so the LLM-as-judge evaluator (configured in the Langfuse UI) has
    everything it needs to score the run.

    If this item already succeeded on a previous run (per CACHE_PATH),
    the cached answer/tool_call_log is reused and no model call is made.
    """
    cached = cache.get(item.id)
    if cached and cached.get("status") == "success":
        langfuse.update_current_span(
            input=item.input,
            output=cached["answer"],
            metadata={
                "tool_call_log": cached["tool_call_log"],
                "articles_log": cached.get("articles_log", []),
                "cached": True,
            },
        )
        return {
            "answer": cached["answer"],
            "tool_call_log": cached["tool_call_log"],
            "articles_log": cached.get("articles_log", []),
        }

    tool_call_log: list[dict] = []
    articles_log: list[dict] = []

    await _throttle()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            deps = Deps(
                finnhub_api_key=finnhub_key,
                http_client=client,
                tool_call_log=tool_call_log,
                articles_log=articles_log,
            )
            result = await agent.run(item.input["question"], deps=deps)
        answer = result.output
    except Exception as e:
        # Don't crash the whole experiment over one bad item — record the
        # failure (so it's retried, not skipped, next run) and move on.
        cache[item.id] = {"status": "error", "error": str(e)}
        save_cache(cache)
        langfuse.update_current_span(
            input=item.input,
            output=f"ERROR: {e}",
            metadata={
                "tool_call_log": tool_call_log,
                "articles_log": articles_log,
                "error": str(e),
            },
        )
        return {
            "answer": f"ERROR: {e}",
            "tool_call_log": tool_call_log,
            "articles_log": articles_log,
        }

    # Success — cache immediately so a mid-run crash still preserves
    # progress made so far, not just progress as of script exit.
    cache[item.id] = {
        "status": "success",
        "answer": answer,
        "tool_call_log": tool_call_log,
        "articles_log": articles_log,
    }
    save_cache(cache)

    langfuse.update_current_span(
        input=item.input,
        output=answer,
        metadata={"tool_call_log": tool_call_log, "articles_log": articles_log},
    )

    return {
        "answer": answer,
        "tool_call_log": tool_call_log,
        "articles_log": articles_log,
    }


async def main() -> None:
    langfuse = get_client()
    assert langfuse.auth_check(), "Langfuse auth failed — check your keys"

    # Auto-instrument every Pydantic AI agent run from here on.
    Agent.instrument_all()

    push_dataset(langfuse)

    finnhub_key = os.environ["FINNHUB_API_KEY"]
    dataset = langfuse.get_dataset(DATASET_NAME)

    cache = load_cache()
    already_succeeded = sum(1 for c in cache.values() if c.get("status") == "success")
    if already_succeeded:
        print(f"Skipping {already_succeeded} already-succeeded item(s) from {CACHE_PATH}.")

    async def task(item):
        return await run_agent(item, finnhub_key, langfuse, cache)

    result = dataset.run_experiment(
        name="Baseline run",
        description="First evaluation pass over the 10 QnA test set",
        task=task,
        max_concurrency=1,
    )
    print(result.format())


if __name__ == "__main__":
    asyncio.run(main())
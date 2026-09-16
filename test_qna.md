# Test QnA Set: Finnhub News Chatbot

Ten questions to validate the agent's tool-calling and answer quality.
Answers depend on live data, so each entry notes what a _correct_ answer
should draw on rather than a fixed expected string.

1. "What's the latest news about Tesla?"
   Tool call: `ticker=TSLA`, small `days_back`. Answer should surface the
   newest headline + summary, attributed to its source.

2. "Summarize the recent news on Apple."
   Tool call: `ticker=AAPL`. Answer should synthesize several summaries
   into a short paragraph, not just list headlines.

3. "Has there been any news about Amazon this week?"
   Tests `days_back=7` filtering.

4. "Where did the latest Microsoft news come from?"
   Tests that the agent surfaces the `source` field.

5. "Give me a link to the most recent Nvidia article."
   Tests that the agent returns the `url` field correctly.

6. "Is there any news about XOM in the last 3 days?"
   Tests narrow date range + graceful "no results" handling if empty.

7. "What are people saying about Meta lately — good or bad?"
   Tests tone inference from headline/summary text only (no sentiment
   field exists in this endpoint — answer shouldn't overclaim precision).

8. "How many news articles came out about Google in the past month?"
   Tests counting/aggregating over the returned list.

9. "What's the oldest news article you have on Netflix from this month?"
   Tests sorting by `datetime` ascending.

10. "Tell me about the news for a stock ticker that doesn't exist, like ZZZZ."
    Edge case: tests graceful handling of an empty/invalid response instead
    of hallucinating headlines.

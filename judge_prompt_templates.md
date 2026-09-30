# LLM-as-Judge Evaluator Prompts (Langfuse)

Set these up in Langfuse under: Prompts -> Create Evaluator.
Because our data is live news (not fixed facts), both judges compare the
answer against what the tool ACTUALLY returned in that trace, rather than
a hardcoded expected answer.

------------------------------------------------------------------
## 1. Groundedness Evaluator (Final Response — Black Box)

Catches hallucinated headlines/facts not present in the tool's output.
Maps to test cases: all, but especially #7 (sentiment) and #10 (fake ticker).

Prompt template:
```
You are checking whether a financial news assistant's answer is grounded
in the source articles it was given — i.e. it did not invent any facts.

### Examples

Articles: [{"headline": "Tesla stock rises 5% after earnings beat"}]
Answer: "Tesla shares climbed 5% following a strong earnings report."
Reasoning: The answer's claim matches the article. No invented facts.
Score: 1

Articles: []
Answer: "Tesla announced a new factory in Berlin this week."
Reasoning: No articles were returned, but the answer states a specific
fact anyway. This is hallucinated.
Score: 0

Articles: []
Answer: "I couldn't find any recent news for that ticker."
Reasoning: No articles were returned, and the answer correctly says so
instead of inventing anything.
Score: 1

### This Case

Articles: {{tool_call_log}}
Answer: {{output}}
```

------------------------------------------------------------------
## 2. Tool Selection Evaluator (Trajectory — Glass Box)

Checks the agent called the tool with the correct ticker.
Maps to test cases: all — this is the simplest "did it do the right thing" check.

Prompt template:
```
You are checking whether an assistant called the correct tool with the
correct stock ticker for the user's question.

### Examples

Question: "What's the latest news about Tesla?"
Expected ticker: TSLA
Actual tool calls: [{"ticker": "TSLA", "days_back": 7}]
Reasoning: Correct ticker was used.
Score: 1

Question: "Give me a link to the most recent Nvidia article."
Expected ticker: NVDA
Actual tool calls: [{"ticker": "NVIDIA", "days_back": 7}]
Reasoning: Wrong ticker format used — "NVIDIA" is not a valid symbol,
should be "NVDA".
Score: 0

### This Case

Question: {{input}}
Expected ticker: {{expected_ticker}}
Actual tool calls: {{tool_call_log}}
```

------------------------------------------------------------------
## Notes for the team

- Both evaluators read `metadata.tool_call_log`, which `eval.py` attaches
  to each trace via `langfuse.update_current_span(metadata=...)`. Make sure
  the evaluator's variable mapping in the Langfuse UI points to that
  metadata field (not a hardcoded "expected_output" field), since our
  ground truth is the tool's own live response, not a fixed answer.
- Score scale is 0/1 (binary) to keep judging simple for a first pass.
  Once this is working, consider a 0-1 continuous or 1-5 scale for more
  nuance on the groundedness check.
- Run `python eval.py` again after any prompt/model change to compare
  experiment runs side-by-side in the Langfuse UI.

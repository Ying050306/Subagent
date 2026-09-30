"""
Streamlit frontend for the Finnhub News Chatbot.

Wraps the exact Pydantic AI agent + tool defined in main.py — no
duplicated logic, so anything you change in main.py (model, prompt,
tool) is picked up here automatically.

Run:
    pip install streamlit
    streamlit run app.py

Needs the same environment variables as main.py:
    FINNHUB_API_KEY
    GEMINI_API_KEY
(loaded from .env via python-dotenv, same as main.py)
"""

import asyncio
import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from main import agent, Deps  # reuse the exact agent + tool from main.py

load_dotenv()

st.set_page_config(page_title="Finnhub News Chatbot", page_icon="📈")
st.title("Finnhub News Chatbot")
st.caption("Ask about recent news for any public company (e.g. \"What's the latest on Tesla?\").")

# --- Session state: chat history for display, plus pydantic-ai's own
# internal message history (needed so the agent remembers context
# across turns within the same browser session). ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_history" not in st.session_state:
    st.session_state.message_history = None

# Render past turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask about a company's news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking that up..."):
            try:
                finnhub_key = os.environ["FINNHUB_API_KEY"]
            except KeyError:
                st.error("FINNHUB_API_KEY is not set. Add it to your .env file and restart.")
                st.stop()

            async def run_turn():
                async with httpx.AsyncClient(timeout=10) as client:
                    deps = Deps(finnhub_api_key=finnhub_key, http_client=client)
                    return await agent.run(
                        prompt,
                        deps=deps,
                        message_history=st.session_state.message_history,
                    )

            # Streamlit's execution model is synchronous per-run, so we
            # drive the async agent call with asyncio.run() here rather
            # than trying to make the whole script async.
            result = asyncio.run(run_turn())
            st.markdown(result.output)

    st.session_state.messages.append({"role": "assistant", "content": result.output})
    st.session_state.message_history = result.all_messages()

with st.sidebar:
    st.subheader("About")
    st.write(
        "This chatbot only answers using live news pulled from Finnhub's "
        "company-news endpoint for the ticker it infers from your question."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.message_history = None
        st.rerun()

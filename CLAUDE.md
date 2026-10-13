# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Chapter 12 of a Manning book on AI agents (LangChain/LangGraph, OpenAI). The code is a set of **standalone, progressive scripts** at the repo root (`main_04_01.py`, `main_05_01.py`, ...), each a self-contained version of the same "UK Travel Assistant" example. Each script supersedes the previous one; they do not import each other. `src/ch12/__init__.py` is just the uv scaffold stub (`ch12 = "ch12:main"`) and is unrelated to the scripts.

## Commands

Python >= 3.13, managed with `uv`.

- Install: `uv sync`
- Run a script: `uv run python main_05_01.py` (interactive CLI; type `exit`/`quit` to leave)
- Lint/format: `uv run ruff check .` / `uv run ruff format .` (ruff is the only dev tool; there is no test suite)
- Secrets: `secretspec.toml` declares `OPENAI_API_KEY`, provided by `pass`; scripts also call `load_dotenv()`, so a `.env` works too. The key must be set before launch.

## Architecture (main_05_01.py, the latest)

Everything runs at **import time**, top to bottom, before `chat_loop()` starts:

1. `get_travel_info_vectorstore()` downloads 4 WikiVoyage Cornwall pages (needs network + a custom `User-Agent`, else Wikimedia returns a stub page), splits them, embeds with `OpenAIEmbeddings` into an in-memory Chroma store, and raises `RuntimeError` if too few chunks come back. Every launch re-downloads and re-embeds (costs OpenAI calls).
2. Two `create_agent` ReAct agents share one `ChatOpenAI` (`gpt-5-nano`, Responses API):
   - `travel_info_agent`: tools `search_travel_info` (Chroma retriever) and `weather_forecast` (random mock).
   - `accommodation_booking_agent`: `SQLDatabaseToolkit` over `hotel_db/cornwall_hotels.db` (SQLite, schema in `hotel_db/cornwall_hotels_schema.sql`, path is relative to the cwd, so run from the repo root) plus `check_bnb_availability` (hardcoded mock offers in `BnBBookingService`).
3. A `StateGraph` wires `router_agent_node` (LLM with structured output `AgentTypeOutput`) to the two agents, both ending at `END`; compiled as `travel_assistant`. `chat_loop` calls `travel_assistant.invoke` with a fresh single-message state each turn, so there is no memory between turns.

Gotchas:
- `AgentState` is imported from `langchain.agents` and then **redefined locally** as a `TypedDict`, which shadows the import. It is only used by the router node and the `StateGraph`; `create_agent` uses its own built-in state.
- Earlier scripts (`main_04_01.py`) have no router: `chat_loop` there talks to a single agent directly. Check which agent `chat_loop` invokes before assuming the router is in play.
- Numbered `# A`, `# B` comments reference the explanatory callouts in the book; keep them aligned with their code when editing.
- `main_05_01_flow.png` is a hand-generated flow diagram of `main_05_01.py`; regenerate it if the flow changes.

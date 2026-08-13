# streamlit_pages/07_EV_Pulse.py
"""
EV Pulse — live news digest with AI summaries.

Data source : NewsAPI (free tier: 100 req/day, 1,000/month)
AI summary  : external AI service (optional)

Design for resilience:
  - Every external call wrapped in try/except with a specific user-facing message
  - Responses cached (1h for news, 24h for AI summaries) to conserve the free quota
  - Graceful degradation: missing AI key → still show headlines, just no summary
  - Rate-limit and timeout errors surfaced clearly, not swallowed silently
  - No partial UI state: if news fetch fails, the page explains why and stops cleanly

Setup: configure the required service credentials only in the local deployment secrets file.
Never commit that file or any credential value. AI summaries are optional.

Get a free NewsAPI key at: https://newsapi.org/register
"""

import streamlit as st
import pandas as pd
import requests
import textwrap
from datetime import datetime, timedelta

_NEWSAPI_URL   = "https://newsapi.org/v2/everything"
_AI_URL        = "https://api.anthropic.com/v1/messages"   # external AI provider endpoint
_TIMEOUT_NEWS  = 8
_TIMEOUT_AI    = 15


# ── Secrets access ───────────────────────────────────────────────────────────

def _get_secret(key: str) -> str:
    """Safely read a secret — never raises, returns empty string if absent."""
    try:
        return str(st.secrets.get(key, "") or "")
    except Exception:
        return ""


# ── NewsAPI ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_news(query: str, days: int, max_results: int, api_key: str) -> dict:
    """
    Fetch recent news from NewsAPI.

    Returns a dict with one of:
      {"ok": True,  "articles": [...]}
      {"ok": False, "error": "<human-readable reason>"}

    Caching by (query, days, max_results, api_key) avoids burning the
    100 req/day free quota on repeated identical queries within an hour.
    """
    if not api_key:
        return {"ok": False, "error": "no_key"}

    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": max_results,
        "apiKey":   api_key,
    }

    try:
        resp = requests.get(_NEWSAPI_URL, params=params, timeout=_TIMEOUT_NEWS)
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "connection"}
    except Exception as e:
        return {"ok": False, "error": f"unexpected: {e}"}

    if resp.status_code == 200:
        data = resp.json()
        articles = [
            a for a in data.get("articles", [])
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
        return {"ok": True, "articles": articles}

    if resp.status_code == 401:
        return {"ok": False, "error": "invalid_key"}
    if resp.status_code == 429:
        return {"ok": False, "error": "rate_limited"}
    if resp.status_code == 426:
        return {"ok": False, "error": "upgrade_required"}

    return {"ok": False, "error": f"http_{resp.status_code}"}


_NEWS_ERROR_MESSAGES = {
    "no_key":           "NewsAPI key not configured.",
    "timeout":          "NewsAPI did not respond in time. Try again shortly.",
    "connection":       "Could not reach NewsAPI. Check your network connection.",
    "invalid_key":      "NewsAPI key was rejected — verify it in secrets.toml.",
    "rate_limited":     "NewsAPI daily quota reached (free tier: 100 requests/day). Try again tomorrow.",
    "upgrade_required": "This query requires a paid NewsAPI plan (date range too far back on free tier).",
}


# ── AI Summary Service ────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _ai_summary(title: str, description: str, context: str, api_key: str) -> dict:
    """
    Generate a concise analytical summary via the external AI service.

    Returns {"ok": True, "text": "..."} or {"ok": False, "error": "..."}.
    Cached 24h per (title, description, context) — summaries don't change.
    """
    if not api_key:
        return {"ok": False, "error": "no_key"}

    prompt = f"""You are an EV market analyst. Summarise this news article for a data-driven audience.

Article title: {title}
Article description: {description}
Context: {context}

Write 2-3 sentences that:
1. State the key fact
2. Explain why it matters for EV transition
3. Note any uncertainty or caveat

Be direct, factual and analytical. No hype, no bullet points, no preamble."""

    try:
        resp = requests.post(
            _AI_URL,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",   # required by the provider
            },
            json={
                "model":      "claude-sonnet-4-6",   # provider-specific model
                "max_tokens": 300,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=_TIMEOUT_AI,
        )
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "connection"}
    except Exception as e:
        return {"ok": False, "error": f"unexpected: {e}"}

    if resp.status_code == 200:
        data   = resp.json()
        blocks = data.get("content", [])
        text   = " ".join(
            b.get("text", "") for b in blocks if b.get("type") == "text"
        ).strip()
        if text:
            return {"ok": True, "text": text}
        return {"ok": False, "error": "empty_response"}

    if resp.status_code == 401:
        return {"ok": False, "error": "invalid_key"}
    if resp.status_code == 429:
        return {"ok": False, "error": "rate_limited"}
    if resp.status_code == 529:
        return {"ok": False, "error": "overloaded"}

    return {"ok": False, "error": f"http_{resp.status_code}"}


_AI_ERROR_MESSAGES = {
    "no_key":      "AI summaries disabled — no AI API key configured.",
    "timeout":     "AI summary timed out.",
    "connection":  "Could not reach the AI service.",
    "invalid_key": "AI API key was rejected.",
    "rate_limited":"AI summary rate limit reached — showing headline only.",
    "overloaded":  "AI service temporarily overloaded — showing headline only.",
    "empty_response": "AI returned an empty summary.",
}


# ── Render ────────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame = None):
    # FIX: remover emoji duplicado — usar título limpo
    st.markdown("## EV Pulse")
    st.markdown(
        "Live news digest — EV market headlines with AI-generated analysis. "
        "Stories are pulled from global news sources and summarised by AI."
    )
    st.markdown("---")

    news_key = _get_secret("NEWSAPI_KEY")
    ai_key   = _get_secret("ANTHROPIC_API_KEY")   # key name kept for compatibility

    if not news_key:
        st.warning(
            "**NewsAPI key not configured.**  \n"
            "To enable live news, configure the NewsAPI credential in local deployment secrets. "
            "Never commit the secrets file. Get a free key (100 requests/day) at "
            "[newsapi.org/register](https://newsapi.org/register)."
        )
        st.markdown("---")
        st.markdown("### What you'll see here once configured")
        st.markdown(
            "- **Topic feed** — latest headlines on EV policy, market moves, "
            "infrastructure and technology\n"
            "- **Country feed** — news filtered by country\n"
            "- **AI summaries** — each article gets a 2-3 sentence analytical "
            "summary from an AI model, focused on market implications\n"
        )
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    col_topic, col_country, col_days = st.columns([2, 2, 1])
    with col_topic:
        topic = st.selectbox("Topic", [
            "electric vehicles",
            "EV policy",
            "EV charging infrastructure",
            "battery technology",
            "EV sales market",
            "EV transition emerging markets",
        ])
    with col_country:
        country_filter = st.text_input("Filter by country (optional)", placeholder="e.g. Brazil")
    with col_days:
        days = st.selectbox("Period (days)", [3, 7, 14, 30], index=1)

    query = f"{topic} {country_filter.strip()}" if country_filter.strip() else topic

    ai_available = bool(ai_key)
    ai_on = st.toggle(
        "Generate AI summaries" + ("" if ai_available else " (requires AI API key)"),
        value=ai_available,
        disabled=not ai_available,
    )

    st.markdown("---")

    # ── Fetch news ────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching latest news on '{query}'..."):
        result = _fetch_news(query, days=days, max_results=8, api_key=news_key)

    if not result["ok"]:
        msg = _NEWS_ERROR_MESSAGES.get(result["error"], f"Unexpected error: {result['error']}")
        st.error(msg)
        return

    articles = result["articles"]

    if not articles:
        st.info(
            f"No articles found for '{query}' in the last {days} days. "
            "Try a broader topic or a longer time window."
        )
        return

    st.markdown(f"**{len(articles)} articles** · Last {days} days · Query: `{query}`")
    st.markdown("---")

    # ── Display articles ──────────────────────────────────────────────────────
    ai_failures = 0

    for art in articles:
        title  = art.get("title", "No title")
        desc   = art.get("description", "") or ""
        url    = art.get("url", "#")
        source = art.get("source", {}).get("name", "Unknown source")
        pub_at = (art.get("publishedAt") or "")[:10]

        with st.container():
            # FIX: usar textwrap para truncar em palavras completas, não no meio
            if desc:
                # textwrap.shorten corta em palavras completas e adiciona "..."
                shown = textwrap.shorten(desc, width=300, placeholder="...")
            else:
                shown = ""

            # Card formatado com separação visual clara
            st.markdown(f"**{title}**")
            st.caption(f"{source} · {pub_at} · [Read full article →]({url})")

            if shown:
                st.markdown(f"{shown}")

            if ai_on:
                context = f"Query was '{query}'. Source: {source}. Focus on EV market implications."
                summary_result = _ai_summary(title, desc, context, ai_key)
                if summary_result["ok"]:
                    st.info(f"📊 **Analysis:** {summary_result['text']}")
                else:
                    ai_failures += 1
                    if ai_failures == 1:
                        err_msg = _AI_ERROR_MESSAGES.get(
                            summary_result["error"], summary_result["error"]
                        )
                        st.caption(f"ℹ️ {err_msg}")
            else:
                # Quando AI está desligado, mostrar um teaser sutil
                if not ai_available:
                    st.caption("💡 *Enable AI summaries by adding an AI API key to secrets.toml*")

            st.markdown("---")

    if ai_on and ai_failures > 1:
        st.caption(
            f"ℹ️ AI summaries failed for {ai_failures} of {len(articles)} articles "
            "— likely a temporary rate limit. Headlines are still shown above."
        )

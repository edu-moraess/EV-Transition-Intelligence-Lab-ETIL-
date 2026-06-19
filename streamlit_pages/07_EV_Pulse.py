# streamlit_pages/07_EV_Pulse.py
"""
EV Pulse — live news digest with AI summaries.

Uses NewsAPI (free tier: 100 req/day) to fetch EV-related headlines,
then Claude API (claude-sonnet-4-6) to generate a concise, analytical
summary of each article relevant to the country/topic selected.

Setup:
  Add to .streamlit/secrets.toml:
    NEWSAPI_KEY = "your_key_here"

  Get a free key at: https://newsapi.org/register
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta


# ── CSS para modebar transparente (opcional, mantido por consistência) ────
st.markdown("""
<style>
.modebar {
    background-color: transparent !important;
    box-shadow: none !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
}
.modebar .icon {
    fill: #1f3a6e !important;
    opacity: 0.7 !important;
    transition: opacity 0.2s !important;
}
.modebar .icon:hover {
    fill: #0d1117 !important;
    opacity: 1 !important;
}
.modebar-btn:hover {
    background-color: rgba(31, 58, 110, 0.08) !important;
    border-radius: 4px !important;
}
.modebar-group {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Configuração padrão para gráficos (não usado aqui, mas mantido) ──────
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d',
        'zoomIn2d', 'zoomOut2d', 'autoScale2d',
        'resetScale2d', 'toImage', 'sendDataToCloud'
    ],
    'responsive': True,
}


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def _fetch_news(query: str, days: int = 7, max_results: int = 8) -> list:
    """Fetch recent news from NewsAPI."""
    try:
        api_key = st.secrets.get("NEWSAPI_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        return []

    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":          query,
        "from":       from_date,
        "sortBy":     "publishedAt",
        "language":   "en",
        "pageSize":   max_results,
        "apiKey":     api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("articles", [])
    except Exception:
        pass
    return []


def _claude_summary(title: str, description: str, context: str) -> str:
    """Generate a concise analytical summary via Claude API."""
    prompt = f"""You are an EV market analyst. Summarise this news article for a data-driven audience.

Article title: {title}
Article description: {description}
Context: {context}

Write 2-3 sentences that:
1. State the key fact
2. Explain why it matters for EV transition
3. Note any uncertainty or caveat

Be direct, factual and analytical. No hype. No bullet points."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            blocks = data.get("content", [])
            return " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    except Exception:
        pass
    return ""


# ── Render ────────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame = None):
    import pandas as pd

    st.markdown("## ⚡ EV Pulse")
    st.markdown(
        "Live news digest — EV market headlines with AI-generated analysis. "
        "Stories are pulled from global news sources and summarised by Claude."
    )
    st.markdown("---")

    # Check API key
    try:
        has_key = bool(st.secrets.get("NEWSAPI_KEY", ""))
    except Exception:
        has_key = False

    if not has_key:
        st.warning(
            "**NewsAPI key not configured.**  \n"
            "To enable live news, add your key to `.streamlit/secrets.toml`:  \n"
            "```\nNEWSAPI_KEY = \"your_key_here\"\n```  \n"
            "Get a free key (100 req/day) at [newsapi.org](https://newsapi.org/register)."
        )
        st.markdown("---")
        st.markdown("### What you'll see here once configured:")
        st.markdown("""
- **Topic feed** — latest headlines on EV policy, market moves, infrastructure and technology
- **Country feed** — news filtered by country, linked to that country's EVQTI position
- **AI summaries** — each article gets a 2-3 sentence analytical summary from Claude, focusing on market implications
- **Trend signals** — recurring themes across stories (policy tailwinds, demand signals, supply constraints)
        """)
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
        days = st.selectbox("Period", [3, 7, 14, 30], index=1)

    # Build query
    query = topic
    if country_filter.strip():
        query = f"{topic} {country_filter.strip()}"

    # AI summary toggle
    ai_on = st.toggle("Generate AI summaries (uses Claude API)", value=True)

    st.markdown("---")

    # ── Fetch ─────────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching latest news on '{query}'..."):
        articles = _fetch_news(query, days=days, max_results=8)

    if not articles:
        st.info(
            "No articles found — try a different topic or time window, "
            "or check your NewsAPI key."
        )
        return

    st.markdown(f"**{len(articles)} articles** · Last {days} days · Query: `{query}`")
    st.markdown("---")

    # ── Display ───────────────────────────────────────────────────────────────
    for i, art in enumerate(articles):
        title   = art.get("title", "No title") or "No title"
        desc    = art.get("description", "") or ""
        url     = art.get("url", "#")
        source  = art.get("source", {}).get("name", "Unknown source")
        pub_at  = art.get("publishedAt", "")[:10]

        # Skip removed/paywalled articles
        if "[Removed]" in title:
            continue

        with st.container():
            st.markdown(f"**{title}**")
            st.caption(f"{source} · {pub_at} · [Read full article →]({url})")

            if desc:
                st.markdown(f"*{desc[:280]}{'...' if len(desc) > 280 else ''}*")

            if ai_on:
                context = (
                    f"Query was '{query}'. "
                    f"Source: {source}. "
                    "Focus on EV market implications."
                )
                with st.spinner("Summarising..."):
                    summary = _claude_summary(title, desc, context)
                if summary:
                    st.success(f"📊 **Analysis:** {summary}")

            st.markdown("---")
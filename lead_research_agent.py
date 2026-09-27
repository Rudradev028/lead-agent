"""
AI Lead Research & Outreach Agent
----------------------------------
Streamlit prototype: give it a company name or URL, it researches the
company (scraping + web search), scores it against your ICP, and drafts
a personalized cold email + LinkedIn message using Claude — showing its
reasoning steps along the way so a human can review before sending.

Run:
    pip install -r requirements.txt
    streamlit run lead_research_agent.py

You'll need:
    - An Anthropic API key (required)
    - A Tavily API key (optional, but strongly recommended — without it
      the agent falls back to scraping the company site only, with no
      external search for size/funding/news signals)
"""

import json
import re
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from anthropic import Anthropic

MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------

def get_client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)


def call_claude(client: Anthropic, system: str, prompt: str, max_tokens: int = 1200) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
    return json.loads(match.group(0))


def tavily_search(query: str, api_key: str, max_results: int = 5) -> list:
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        return [{"title": "search_error", "url": "", "content": str(e)}]


def scrape_page(url: str, timeout: int = 10) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; LeadResearchAgent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:6000]
    except Exception:
        return ""


# ---------------------------------------------------------------------
# Agent pipeline steps
# ---------------------------------------------------------------------

def normalize_company_input(raw: str, tavily_key: str, log: list) -> str:
    raw = raw.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if re.match(r"^[\w\-]+(\.[\w\-]+)+$", raw):
        return "https://" + raw

    log.append(f"'{raw}' doesn't look like a URL — searching for the official website")
    results = tavily_search(f"{raw} official website", tavily_key, max_results=5)
    for r in results:
        url = r.get("url", "")
        if url:
            log.append(f"Resolved '{raw}' to {url}")
            return url
    raise ValueError(
        f"Could not resolve a website for '{raw}'. Try entering the URL directly, "
        f"or add a Tavily API key so the agent can search for it."
    )


def scrape_company_site(base_url: str, log: list, max_extra_pages: int = 4) -> dict:
    pages = {}
    home_text = scrape_page(base_url)
    pages["homepage"] = home_text
    log.append(f"Scraped homepage ({len(home_text)} chars)")

    candidate_paths = ["/about", "/about-us", "/pricing", "/product", "/products", "/solutions", "/customers"]
    found = 0
    for path in candidate_paths:
        if found >= max_extra_pages:
            break
        text = scrape_page(urljoin(base_url, path))
        if text and len(text) > 200:
            pages[path] = text
            log.append(f"Scraped {path} ({len(text)} chars)")
            found += 1
    return pages


def research_company(client: Anthropic, company_input: str, tavily_key: str, log: list) -> dict:
    base_url = normalize_company_input(company_input, tavily_key, log)
    scraped = scrape_company_site(base_url, log)
    site_text = "\n\n".join(f"[{k}]\n{v}" for k, v in scraped.items() if v)

    log.append("Searching the web for size, funding, and news signals")
    search_results = tavily_search(
        f"{company_input} company size employees funding news", tavily_key, max_results=5
    )
    search_summary = "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')[:300]}"
        for r in search_results if r.get("content")
    )

    log.append("Synthesizing research findings with Claude")
    system = "You are a B2B sales research analyst. Respond with strict JSON only — no prose, no markdown fences."
    prompt = f"""Based on the scraped website content and web search snippets below, produce a JSON object describing this company:

{{
  "company_name": string,
  "website": string,
  "one_line_description": string,
  "industry": string,
  "estimated_size": string,   // e.g. "1-10", "11-50", "51-200", "201-1000", "1000+", or "unknown"
  "likely_pain_points": [string, ...],   // 3-5 short items
  "key_facts": [string, ...],            // 3-6 short items: funding, news, tech stack, notable customers
  "confidence": "low" | "medium" | "high"
}}

Website URL: {base_url}

--- SCRAPED WEBSITE CONTENT ---
{site_text[:8000] if site_text else "(no content could be scraped)"}

--- WEB SEARCH SNIPPETS ---
{search_summary if search_summary else "(no search results — no search API key provided or no results found)"}

Return ONLY the JSON object."""
    data = extract_json(call_claude(client, system, prompt, max_tokens=1200))
    data["website"] = data.get("website") or base_url
    return data


def score_lead(client: Anthropic, research: dict, icp: dict, log: list) -> dict:
    log.append("Scoring the lead against your Ideal Customer Profile")
    system = "You are a sales qualification assistant. Respond with strict JSON only."
    prompt = f"""Score this lead against the Ideal Customer Profile (ICP) on a 0-100 scale.

ICP:
- Target industries: {icp['industries']}
- Target company size: {icp['size']}
- Target pain points / buying triggers: {icp['pain_points']}
- Other notes: {icp['notes']}

Lead research:
{json.dumps(research, indent=2)}

Return a JSON object:
{{
  "score": integer 0-100,
  "verdict": "strong fit" | "possible fit" | "weak fit",
  "reasoning": string (2-4 sentences, citing specific overlaps or mismatches)
}}"""
    return extract_json(call_claude(client, system, prompt, max_tokens=500))


def generate_outreach(client: Anthropic, research: dict, score: dict, sender: dict, log: list) -> dict:
    log.append("Drafting personalized cold email and LinkedIn message")
    system = (
        "You are an expert SDR copywriter. Write concise, specific, non-generic outreach "
        "grounded in the actual research provided. No markdown, no unfilled placeholders."
    )
    prompt = f"""Write outreach for this lead.

Sender info:
- Rep name: {sender['rep_name']}
- Sender company: {sender['company_name']}
- Product/service: {sender['product_description']}
- Value prop angle to lead with: {sender['value_prop']}

Lead research:
{json.dumps(research, indent=2)}

Lead score/reasoning:
{json.dumps(score, indent=2)}

Return a JSON object:
{{
  "email_subject": string (under 60 chars),
  "email_body": string (80-130 words, personalized with specific facts from the research, one clear CTA, signed with the rep name),
  "linkedin_message": string (under 300 characters, casual, personalized, one CTA)
}}"""
    return extract_json(call_claude(client, system, prompt, max_tokens=900))


def run_pipeline(client, company_input, tavily_key, icp, sender):
    """Runs the full pipeline once, returns (research, score, outreach, full_log)."""
    full_log = []
    research = research_company(client, company_input, tavily_key, full_log)
    score = score_lead(client, research, icp, full_log)
    outreach = generate_outreach(client, research, score, sender, full_log)
    return research, score, outreach, full_log


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="Lead Research & Outreach Agent", layout="wide")
st.title("🔍 AI Lead Research & Outreach Agent")
st.caption("Give it a company. It researches, scores against your ICP, and drafts outreach — reasoning shown for review.")

with st.sidebar:
    st.header("🔑 API Keys")
    default_anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
    default_tavily_key = st.secrets.get("TAVILY_API_KEY", "") if hasattr(st, "secrets") else ""
    anthropic_key = st.text_input("Anthropic API key", type="password", value=default_anthropic_key)
    tavily_key = st.text_input("Tavily API key (optional, enables web search)", type="password", value=default_tavily_key)
    st.caption("Pulled from app secrets if configured; you can override per session. Never stored beyond this session.")

    st.header("🎯 Ideal Customer Profile")
    icp_industries = st.text_input("Target industries", "SaaS, Fintech")
    icp_size = st.text_input("Target company size", "51-500 employees")
    icp_pain_points = st.text_area("Target pain points / buying triggers", "Manual sales research, slow outbound, understaffed SDR team")
    icp_notes = st.text_area("Other ICP notes", "")

    st.header("✍️ Your info")
    rep_name = st.text_input("Your name", "Alex Rivera")
    sender_company = st.text_input("Your company", "Acme Analytics")
    product_description = st.text_area("What you sell", "An AI-powered analytics platform that helps sales teams prioritize leads")
    value_prop = st.text_input("Value prop angle to lead with", "Save reps hours of manual research per week")

icp = {"industries": icp_industries, "size": icp_size, "pain_points": icp_pain_points, "notes": icp_notes}
sender = {"rep_name": rep_name, "company_name": sender_company, "product_description": product_description, "value_prop": value_prop}

tab1, tab2 = st.tabs(["Single Lead", "Batch (CSV)"])

# ---------------- Single lead ----------------
with tab1:
    company_input = st.text_input("Company name or website", placeholder="e.g. Notion or notion.so")
    run = st.button("Research & Generate Outreach", type="primary")

    if run:
        if not anthropic_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        elif not company_input:
            st.error("Please enter a company name or URL.")
        else:
            if not tavily_key:
                st.info("No Tavily key set — the agent will only scrape the company's own site, with no external search.")
            client = get_client(anthropic_key)
            try:
                with st.status("Researching company...", expanded=True) as status:
                    log = []
                    research = research_company(client, company_input, tavily_key, log)
                    for line in log:
                        st.write("• " + line)
                    status.update(label="Research complete", state="complete")

                with st.status("Scoring against ICP...", expanded=True) as status:
                    log2 = []
                    score = score_lead(client, research, icp, log2)
                    for line in log2:
                        st.write("• " + line)
                    status.update(label="Scoring complete", state="complete")

                with st.status("Drafting outreach...", expanded=True) as status:
                    log3 = []
                    outreach = generate_outreach(client, research, score, sender, log3)
                    for line in log3:
                        st.write("• " + line)
                    status.update(label="Outreach drafted", state="complete")

                st.divider()
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("Lead Score")
                    st.metric(research.get("company_name", company_input), f"{score.get('score', '?')}/100", score.get("verdict", ""))
                    st.write(score.get("reasoning", ""))
                with col2:
                    st.subheader("Research Summary")
                    st.write(f"**{research.get('company_name', '')}** — {research.get('one_line_description', '')}")
                    st.write(f"Industry: {research.get('industry', '?')} · Size: {research.get('estimated_size', '?')} · Confidence: {research.get('confidence', '?')}")
                    st.write("**Likely pain points:**")
                    for p in research.get("likely_pain_points", []):
                        st.write(f"- {p}")
                    st.write("**Key facts:**")
                    for f in research.get("key_facts", []):
                        st.write(f"- {f}")

                st.divider()
                st.subheader("✉️ Cold Email (editable before sending)")
                subject = st.text_input("Subject", outreach.get("email_subject", ""))
                body = st.text_area("Body", outreach.get("email_body", ""), height=200)

                st.subheader("💬 LinkedIn Message (editable before sending)")
                li_msg = st.text_area("Message", outreach.get("linkedin_message", ""), height=100)

                result_row = {
                    "company_input": company_input,
                    **research,
                    **{f"score_{k}": v for k, v in score.items()},
                    "email_subject": subject,
                    "email_body": body,
                    "linkedin_message": li_msg,
                }
                st.download_button("Download result as JSON", json.dumps(result_row, indent=2), file_name="lead_result.json")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------------- Batch ----------------
with tab2:
    st.write("Upload a CSV with a `company` column (names or URLs) to process multiple leads at once.")
    csv_file = st.file_uploader("CSV file", type=["csv"])
    batch_run = st.button("Run batch")

    if batch_run:
        if not anthropic_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        elif not csv_file:
            st.error("Please upload a CSV file.")
        else:
            df = pd.read_csv(csv_file)
            if "company" not in df.columns:
                st.error("CSV must have a column named 'company'.")
            else:
                client = get_client(anthropic_key)
                results = []
                progress = st.progress(0.0)
                status_area = st.empty()

                for i, row in df.iterrows():
                    company_input = str(row["company"])
                    status_area.write(f"Processing **{company_input}** ({i + 1}/{len(df)})")
                    try:
                        research, score, outreach, _ = run_pipeline(client, company_input, tavily_key, icp, sender)
                        results.append({
                            "company_input": company_input,
                            "company_name": research.get("company_name"),
                            "website": research.get("website"),
                            "industry": research.get("industry"),
                            "estimated_size": research.get("estimated_size"),
                            "score": score.get("score"),
                            "verdict": score.get("verdict"),
                            "reasoning": score.get("reasoning"),
                            "email_subject": outreach.get("email_subject"),
                            "email_body": outreach.get("email_body"),
                            "linkedin_message": outreach.get("linkedin_message"),
                        })
                    except Exception as e:
                        results.append({"company_input": company_input, "error": str(e)})
                    progress.progress((i + 1) / len(df))
                    time.sleep(0.3)  # be polite to APIs

                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                st.download_button("Download results CSV", results_df.to_csv(index=False), file_name="lead_batch_results.csv")

st.divider()
st.caption(
    "Stretch goals not yet wired up: saving results to Google Sheets or a CRM (HubSpot/Salesforce) — "
    "would need OAuth/API-key setup for the target system. The batch tab's CSV export covers the same need for now."
)

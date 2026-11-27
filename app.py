import concurrent.futures
import hmac
import html
import io
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from trendspy import Trends as TrendsPyClient
    HAS_TRENDSPY = True
except Exception:
    HAS_TRENDSPY = False

# ==============================================================================
# CONNEXT GLOBAL BRAND TOKENS & CONSTANTS
# ==============================================================================
CNXT_COLORS = {
    "blue": "#005AAC",
    "sun": "#E58B25",
    "yellow": "#F1C916",
    "red": "#DC2626",
    "twilight": "#071776",
    "midnight": "#071739",
    "teal": "#0F818A",
    "teal_blue": "#90CBDB",
    "cream": "#F4F1E2",
    "bg": "#F7F8FB",
    "paper": "#FFFFFF",
    "ink": "#0F172A",
    "ash": "#64748B",
    "line": "#E2E8F0",
}

CNXT_LOGO_COLOR = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6a72a1e48c2cd52f0e5f2578_CONNEXT%20MAIN%20LOGO%20COLOR.png"
CNXT_LOGO_WHITE = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6ad6f9596637ac5988a5bf28_CONNEXT%20MAIN%20LOGO%20WHITE%20V.png"
CNXT_FAVICON = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6a965e51559ff3d4d6cad70f_Connext_Icons_Connext%20Logo-lg.png"

SITEMAP_XML_URL = "https://connextglobal.com/sitemap.xml"
SITEMAP_HTML_URL = "https://connextglobal.com/sitemap"

COMPETITORS_REGISTRY = [
    ("Solvo", "solvoglobal.com", r"\b(solvo(\s*global)?|solvoglobal)\b"),
    ("Genius", "joingenius.com", r"\b(join\s*)?genius\b"),
    ("Microsourcing", "microsourcing.com", r"\bmicro\s*sourcing\b"),
    ("Emapta", "emapta.com", r"\bemapta\b"),
    ("Outsourced", "outsourced.ph", r"\boutsourced(\.ph|\s*philippines)?\b"),
    ("OP360", "op360.com", r"\b(op\s*360|office\s*partners\s*360)\b"),
    ("Lean Solutions", "leangroup.com", r"\b(lean\s*solutions(\s*group)?|lean\s*group|leangroup)\b"),
    ("TaskUs", "taskus.com", r"\btask\s*us\b"),
    ("TOA Global", "toaglobal.com", r"\btoa(\s*global)?\b"),
    ("Cloudstaff", "cloudstaff.com", r"\bcloud\s*staff\b"),
    ("Near", "hirewithnear.com", r"\b(hire\s*with\s*near|near\s*(staffing|outsourcing|latam|talent))\b"),
    ("BairesDev", "bairesdev.com", r"\bbaires\s*dev\b"),
]

BLOG_INCLUDE_PATH_RE = re.compile(
    r"/(blog|blogs|insights|articles|article|resources/blog|learn/blog|news|guides|learning-center|knowledge-base|posts)/[^/]+",
    re.I,
)
BLOG_EXCLUDE_PATH_RE = re.compile(
    r"(^/(es|pt|fr|de|ja|zh|ph|co|mx)/|/category/|/categories/|/tag/|/tags/|/author/|/authors/|"
    r"/page/\d+|/feed/?$|/careers?|/jobs?/|/privacy|/terms|/cookie|/contact|/about|"
    r"/login|/signup|/case-stud|/white-?paper|/ebook|/e-book|/webinar|/event|/press-release|"
    r"/glossary|/podcast|/video|/sitemap|/wp-content|/team/|/people/|/locations?/|/industries/|/services/)",
    re.I,
)

BLOG_KW_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you", "are", "how", "what",
    "why", "when", "where", "which", "who", "can", "will", "into", "about", "more", "best",
    "top", "guide", "complete", "ultimate", "ways", "tips", "steps", "reasons", "things",
    "need", "know", "should", "make", "help", "using", "used", "their", "they", "have",
    "has", "had", "not", "but", "all", "new", "2024", "2025", "2026", "2027", "our", "out",
    "over", "than", "then", "also", "just", "most", "some", "such", "only", "very", "much",
    "between", "through", "after", "before", "during", "under", "within", "without", "blog",
    "article", "post", "read", "learn", "discover", "explore", "understanding", "explained",
    "everything", "really", "actually", "right", "better", "smart", "modern", "today", "future",
    "cut", "reduce", "boost", "grow", "increase", "rising", "benefits", "key", "essential",
    "business", "businesses", "company", "companies", "work", "working", "team", "teams",
}

COMP_BRAND_WORDS = {
    "solvo", "solvoglobal", "genius", "joingenius", "microsourcing", "emapta", "outsourced",
    "op360", "leangroup", "lean", "taskus", "toaglobal", "toa", "cloudstaff", "near",
    "hirewithnear", "bairesdev", "global", "solutions", "group", "inc", "llc",
}

INDUSTRY_NEWS_FEEDS = {
    "All B2B Outsourcing & Staffing": '("BPO outsourcing" OR "offshore staffing" OR "nearshore staffing" OR "staff augmentation") when:14d',
    "Healthcare & RCM": '("medical billing outsourcing" OR "revenue cycle management" OR "prior authorization" OR "healthcare BPO") when:14d',
    "Finance & Accounting": '("offshore accounting" OR "outsourced accounting" OR "CPA shortage" OR "finance outsourcing") when:14d',
    "Tech, IT & AI Staffing": '("nearshore software developers" OR "AI staffing" OR "IT staff augmentation" OR "offshore developers") when:14d',
    "Customer Experience (CX)": '("customer support outsourcing" OR "AI customer service" OR "contact center BPO" OR "virtual assistant") when:14d',
}

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Connext Global | Enterprise Search & Topic Radar",
    page_icon=CNXT_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_html_safe(html_str: str):
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        st.markdown(html_str, unsafe_allow_html=True)


def inject_connext_brand_css():
    """Renders Connext corporate styles via st.html() with zero blank lines so CSS never leaks as text."""
    css_rules = [
        "@import url('https://fonts.googleapis.com/css2?family=Arimo:wght@500;600;700&family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@500;600&display=swap');",
        ":root { --cnxt-blue: #005AAC; --cnxt-sun: #E58B25; --cnxt-yellow: #F1C916; --cnxt-twilight: #071776; --cnxt-midnight: #071739; --cnxt-teal: #0F818A; --cnxt-teal-blue: #90CBDB; --cnxt-bg: #F7F8FB; --cnxt-paper: #FFFFFF; --cnxt-ink: #0F172A; --cnxt-ash: #64748B; --cnxt-line: #E2E8F0; --font-display: 'Arimo', Arial, sans-serif; --font-body: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif; --font-mono: 'JetBrains Mono', ui-monospace, monospace; }",
        "html, body, .stApp { font-family: var(--font-body); background-color: var(--cnxt-bg) !important; color: var(--cnxt-ink) !important; }",
        'header[data-testid="stHeader"] { background-color: var(--cnxt-bg) !important; border-bottom: 1px solid var(--cnxt-line); }',
        ".block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1440px; }",
        "h1, h2, h3, h4 { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; letter-spacing: -0.015em; }",
        '.stApp [data-testid="stForm"] { background-color: #FFFFFF !important; border: 1px solid var(--cnxt-line) !important; border-radius: 12px !important; padding: 22px 24px !important; box-shadow: 0 4px 16px rgba(7, 23, 57, 0.06) !important; }',
        '.stApp .main label, .stApp [data-testid="stWidgetLabel"] p { color: #1E293B !important; font-weight: 500 !important; }',
        ".stApp .main input, .stApp .main textarea { background-color: #FFFFFF !important; color: #0F172A !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; }",
        '.stApp .main [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #0F172A !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; }',
        '.stApp .main [data-baseweb="select"] span, .stApp .main [data-baseweb="select"] input { color: #0F172A !important; }',
        '[data-baseweb="popover"], [data-baseweb="menu"] { background-color: #FFFFFF !important; color: #0F172A !important; }',
        '[data-baseweb="menu"] li { color: #0F172A !important; }',
        '.stApp .main [data-testid="stRadio"] label p, .stApp .main [data-testid="stCheckbox"] label p { color: #1E293B !important; }',
        '.stApp .main [data-testid="stExpander"] { background-color: #FFFFFF !important; border: 1px solid var(--cnxt-line) !important; border-radius: 10px !important; }',
        '.stApp .main [data-testid="stExpander"] summary p { color: var(--cnxt-twilight) !important; font-weight: 600 !important; }',
        '.stApp .main [data-testid="stCodeBlock"] pre { background-color: #0F172A !important; color: #F8FAFC !important; border-radius: 8px !important; }',
        '.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button, .stDownloadButton > button { background-color: var(--cnxt-sun) !important; color: #071739 !important; font-family: var(--font-body) !important; font-weight: 700 !important; border: none !important; border-radius: 999px !important; padding: 0.52rem 1.35rem !important; box-shadow: 0 4px 12px rgba(229, 139, 37, 0.30) !important; }',
        '.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button:hover, .stDownloadButton > button:hover { background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important; }',
        '.stButton > button:not([kind="primary"]) { background-color: #FFFFFF !important; color: var(--cnxt-twilight) !important; border: 1px solid #CBD5E1 !important; border-radius: 999px !important; font-weight: 600 !important; }',
        '.stButton > button:not([kind="primary"]):hover { border-color: var(--cnxt-twilight) !important; background-color: #F1F5F9 !important; }',
        'section[data-testid="stSidebar"] { background-color: var(--cnxt-midnight) !important; border-right: 1px solid rgba(144, 203, 219, 0.16); }',
        'section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #F8FAFC !important; }',
        'section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small { color: var(--cnxt-teal-blue) !important; }',
        'section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }',
        'section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea { background-color: #FFFFFF !important; color: #071739 !important; border-radius: 6px !important; }',
        'section[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #071739 !important; }',
        'section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #071739 !important; }',
        'section[data-testid="stSidebar"] [data-testid="stExpander"] { background-color: rgba(255,255,255,0.06) !important; border: 1px solid rgba(144,203,219,0.22) !important; }',
        'section[data-testid="stSidebar"] [data-testid="stExpander"] summary p { color: #F1C916 !important; }',
        'section[data-testid="stSidebar"] .stButton > button { background-color: rgba(255,255,255,0.10) !important; color: var(--cnxt-teal-blue) !important; border: 1px solid rgba(144,203,219,0.32) !important; border-radius: 999px !important; }',
        'section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background-color: var(--cnxt-sun) !important; color: var(--cnxt-midnight) !important; border: none !important; font-weight: 700 !important; }',
        ".cnxt-hero { background: linear-gradient(135deg, #071776 0%, #071739 100%); color: #FFFFFF; padding: 22px 28px; border-radius: 12px; margin-bottom: 16px; border: 1px solid rgba(144,203,219,0.2); box-shadow: 0 6px 20px rgba(7,23,57,0.08); }",
        ".cnxt-eyebrow { font-family: var(--font-mono); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.16em; color: #F1C916; margin-bottom: 5px; font-weight: 600; }",
        ".cnxt-hero h1 { font-size: clamp(1.35rem, 2.2vw, 1.9rem) !important; color: #FFFFFF !important; margin: 0 0 6px 0 !important; }",
        ".cnxt-hero h1 span { color: #E58B25; }",
        ".cnxt-hero p { color: #CBD5E1 !important; font-size: 0.93rem; margin: 0; max-width: 900px; line-height: 1.5; }",
        'div[data-testid="stMetric"] { background-color: var(--cnxt-paper); border: 1px solid var(--cnxt-line); border-top: 3px solid var(--cnxt-blue); border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04); }',
        'div[data-testid="stMetric"] label { font-family: var(--font-mono) !important; font-size: 0.70rem !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; color: var(--cnxt-ash) !important; }',
        'div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; font-weight: 700 !important; font-size: 1.6rem !important; }',
        '.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid var(--cnxt-line); margin-bottom: 14px; flex-wrap: wrap; }',
        '.stTabs [data-baseweb="tab"] { font-family: var(--font-display); font-weight: 600; font-size: 0.89rem; color: #334155 !important; background-color: #FFFFFF !important; border-radius: 8px 8px 0 0; padding: 10px 14px; border: 1px solid var(--cnxt-line); border-bottom: none; }',
        '.stTabs [aria-selected="true"] { background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important; border-color: var(--cnxt-twilight) !important; }',
        ".cnxt-callout { background-color: #FFFFFF; border: 1px solid var(--cnxt-line); border-left: 4px solid var(--cnxt-sun); border-radius: 8px; padding: 14px 18px; margin: 8px 0 14px 0; color: #1E293B; font-size: 0.92rem; line-height: 1.55; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03); }",
        ".cnxt-callout strong { color: var(--cnxt-twilight); }",
        ".cnxt-info-card { background-color: #FFFFFF; border: 1px solid var(--cnxt-line); border-radius: 12px; padding: 20px 22px; margin-bottom: 14px; color: #1E293B; font-size: 0.92rem; line-height: 1.6; box-shadow: 0 2px 10px rgba(7, 23, 57, 0.04); }",
        ".cnxt-info-card h3 { margin-top: 0 !important; margin-bottom: 10px !important; font-size: 1.08rem !important; }",
        ".cnxt-info-card ul { margin: 6px 0 10px 18px; padding: 0; }",
        ".cnxt-info-card li { margin-bottom: 6px; }",
        ".cnxt-timing-box { background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #E58B25; border-radius: 8px; padding: 12px 16px; margin: 12px 0; color: #78350F; font-size: 0.9rem; line-height: 1.5; }",
        ".cnxt-sec-sub { color: #64748B !important; font-size: 0.9rem; margin-top: -8px; margin-bottom: 12px; }",
        'span[data-baseweb="tag"] { background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important; }',
    ]
    _inject_html_safe("<style>\n" + "\n".join(css_rules) + "\n</style>")


inject_connext_brand_css()


def style_plotly_light(fig: go.Figure, height: int = 360, legend_bottom: bool = False) -> go.Figure:
    layout_kwargs = dict(
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Roboto, sans-serif", color="#0F172A", size=12),
        xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"), title_font=dict(color="#334155")),
        yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155"), title_font=dict(color="#334155")),
    )
    if legend_bottom:
        layout_kwargs["legend"] = dict(orientation="h", y=-0.22, font=dict(color="#0F172A"), title="")
    fig.update_layout(**layout_kwargs)
    return fig


def render_tab_guide(tab_name: str, what_you_see: str, what_to_expect: str, how_to_use: str):
    """Renders a consistent, plain-English 3-column guide box inside every tab."""
    with st.expander(f"How to Read & Use This Tab — {tab_name}", expanded=False):
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.markdown(f"**What You Are Looking At**\n\n{what_you_see}")
        with gc2:
            st.markdown(f"**What Results to Expect**\n\n{what_to_expect}")
        with gc3:
            st.markdown(f"**How to Act on This Data**\n\n{how_to_use}")


def require_connext_login():
    """Beginner-friendly team login screen with complete onboarding guidelines and loading-time notice."""
    if st.session_state.get("cnxt_authenticated"):
        return

    _inject_html_safe("<style>section[data-testid='stSidebar']{display:none !important;}</style>")

    st.image(CNXT_LOGO_COLOR, width=190)
    st.markdown(
        """<div class="cnxt-hero" style="margin-top: 12px; margin-bottom: 20px;">
<div class="cnxt-eyebrow">CONNEXT GLOBAL INTERNAL MARKETING &amp; GROWTH PLATFORM</div>
<h1>Connext Search &amp; Topic <span>Trend Radar</span></h1>
<p>100% verified Google Search Console data, live sitemap checks, Competitor US Ads Transparency, and competitor blog intelligence—strictly filtered to Connext B2B topics.</p>
</div>""",
        unsafe_allow_html=True,
    )

    col_login, col_guide = st.columns([1, 1.35], gap="large")

    with col_login:
        if "app_password" not in st.secrets:
            st.error("`app_password` is not configured in Streamlit Secrets yet. Add it in App Settings → Secrets.")
            st.stop()

        attempts = int(st.session_state.get("cnxt_login_attempts", 0))
        if attempts >= 5:
            st.error("Too many failed attempts in this browser session. Refresh the page and verify your team password.")
            st.stop()

        with st.form("cnxt_login_form", clear_on_submit=False):
            st.markdown("### Team Sign-In")
            st.caption("Enter the shared Connext marketing password provided in Microsoft Teams.")
            entered_pwd = st.text_input("Team Password", type="password", placeholder="Enter team password...")
            submitted = st.form_submit_button("Unlock Enterprise Radar", type="primary", use_container_width=True)
            if submitted:
                if hmac.compare_digest(entered_pwd.strip(), str(st.secrets["app_password"]).strip()):
                    st.session_state["cnxt_authenticated"] = True
                    st.session_state["cnxt_login_attempts"] = 0
                    st.rerun()
                else:
                    st.session_state["cnxt_login_attempts"] = attempts + 1
                    rem = max(0, 5 - (attempts + 1))
                    st.error(f"Incorrect password. ({rem} attempt{'s' if rem != 1 else ''} remaining)")

        st.markdown(
            """<div class="cnxt-timing-box">
<strong>Please Expect 30 to 60 Seconds to Load:</strong><br/>
Right after you click <strong>Unlock Enterprise Radar</strong>, the app connects live to Google Search Console, filters out job-seeker and off-topic noise, groups searches into Connext B2B topics, and indexes <code>connextglobal.com/sitemap.xml</code>. <strong>Please keep the tab open and do not refresh while it loads.</strong>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """<div class="cnxt-info-card" style="padding: 14px 18px; font-size: 0.86rem;">
<strong>Data Integrity &amp; Security:</strong>
<ul>
<li><strong>100% Real Data Only:</strong> Every impression, click, CTR, rank, and bid comes directly from Google Search Console, Google Keyword Planner, or live public pages—zero invented numbers.</li>
<li><strong>Read-Only Access:</strong> Nothing inside this dashboard can edit Google Search Console, Google Ads, or the website.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    with col_guide:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>What This Tool Does (In Plain English)</h3>
<ul>
<li><strong>Finds Rising Buyer Demand (Tab 1):</strong> Spots Connext-relevant B2B topics growing on Google (3-day spikes, 7-day surges, and 30-day growth) with real GSC Impressions, Clicks, CTR, and Rank.</li>
<li><strong>Catches Slipping Pages &amp; Audits HTML (Tab 2):</strong> Flags <em>Content Decay</em> when real GSC impressions or ranks drop, and grades live pages on SEO &amp; AI readiness.</li>
<li><strong>Builds Google &amp; Bing Ads Campaigns (Tab 3):</strong> Turns real buyer searches into Exact/Phrase keyword lists, harvests real wasted non-B2B negative keywords, and merges real Google Keyword Planner volumes/bids when uploaded.</li>
<li><strong>Plans New Content &amp; Internal Links (Tab 4):</strong> Distinguishes root-level &amp; <code>/blog/</code> posts from true commercial Service/Industry pages so blogs link into real Service Pages.</li>
<li><strong>Inspects Competitor US Ads &amp; Blogs (Tabs 5 &amp; 6):</strong> Dedicated <strong>Competitor US Ads Transparency</strong> tab for all 12 BPO rivals plus on-demand competitor new blog &amp; keyword discovery.</li>
</ul>
<h3>What to Expect Once Inside</h3>
<ul>
<li><strong>Tabs 1–4</strong> load automatically after the initial 30–60s sync (cached for 1 hour so filtering is instant).</li>
<li><strong>Tabs 5 &amp; 6</strong> run external checks <strong>only when you click their button</strong> (~10–20s) so the dashboard stays fast and never spends API credits automatically.</li>
<li>Open <strong>Tab 7 (User Guide &amp; Glossary)</strong> anytime for plain-English definitions of every metric.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    st.stop()


require_connext_login()

# ==============================================================================
# CONNEXT SITE ARCHITECTURE & STRICT B2B RELEVANCE TAXONOMY
# ==============================================================================
CORE_LIVE_PATHS = [
    "/", "/about-us", "/bpo-glossary", "/compare-by-model", "/connext-pricing", "/contact",
    "/general-outsourcing", "/hipaa-compliant-virtual-medical-assistants", "/insights", "/insights/faq",
    "/our-differentiator", "/our-service-model", "/roles-we-fill", "/trust-compliance",
    "/industries/construction", "/industries/education", "/industries/engineering",
    "/industries/engineering/engineering-outsourcing-solutions", "/industries/financial-services",
    "/industries/fintech", "/industries/healthcare", "/industries/healthcare/dental-billing-outsourcing",
    "/industries/healthcare/healthcare-software", "/industries/healthcare/medical-billing",
    "/industries/healthcare/medical-coding", "/industries/healthcare/outsource-claims-management",
    "/industries/healthcare/outsource-credentialing", "/industries/healthcare/outsource-prior-authorization",
    "/industries/healthcare/remote-monitoring", "/industries/healthcare/revenue-cycle-management",
    "/industries/healthcare/virtual-nursing", "/industries/hospitality", "/industries/human-resources",
    "/industries/insurance", "/industries/insurance/insurance-outsourcing", "/industries/legal",
    "/industries/private-equity", "/industries/real-estate", "/industries/real-estate/mortgage",
    "/industries/real-estate/title-escrow",
    "/teams/ai-staffing", "/teams/back-office/payroll-outsourcing-solutions",
    "/teams/business-operation-support", "/teams/customer-support",
    "/teams/customer-support/call-center-outsourcing", "/teams/customer-support/customer-success",
    "/teams/customer-support/technical-support-outsourcing", "/teams/customer-support/virtual-assistants",
    "/teams/customer-support/zendesk-staffing", "/teams/finance-and-accounting-outsourcing-services",
    "/teams/financial-services/accounting", "/teams/financial-services/accounts-softwares",
    "/teams/financial-services/debt-collection-outsourcing",
    "/teams/financial-services/hire-certified-public-accountants-with-connext",
    "/teams/financial-services/hire-expert-controllers-with-connext",
    "/teams/it-development", "/teams/it-development/machine-learing", "/teams/it-development/python-staffing",
    "/teams/it-development/servicenow-staffing", "/teams/marketing",
    "/teams/marketing/e-commerce-outsourcing", "/teams/marketing/outsource-digital-marketing",
    "/teams/operation-support/payroll-outsourcing",
    "/services/co-management", "/services/compliance", "/services/contractor-transitions",
    "/services/employer-of-record", "/services/office-space", "/services/process-it-support",
    "/services/recruiting", "/services/retention",
    "/locations", "/locations/outsource-to-colombia", "/locations/outsource-to-india",
    "/locations/outsource-to-mexico", "/locations/outsource-to-the-philippines",
    "/locations/outsource-to-united-states",
    "/tools", "/tools/co-management-fit", "/tools/country-explorer", "/tools/hiring-timeline-estimator",
    "/tools/pricing-calculator", "/tools/roi-calculator", "/tools/salary-benchmark", "/tools/team-readiness",
]

# Exact allowlist of root-level commercial & utility pages so migrated root-level blogs (without /blog/) are NEVER misclassified as Service Pages
EXACT_CORE_COMMERCIAL_PATHS = {
    "/our-service-model", "/our-differentiator", "/compare-by-model",
    "/connext-pricing", "/trust-compliance", "/roles-we-fill",
    "/general-outsourcing", "/hipaa-compliant-virtual-medical-assistants",
}

EXACT_UTILITY_PATHS = {
    "/about-us", "/contact", "/careers", "/jobs", "/sitemap",
    "/privacy-policy", "/terms", "/terms-of-service", "/terms-and-conditions",
    "/cookie-policy", "/thank-you",
}

EXACT_CONTENT_HUB_PATHS = {"/insights", "/insights/faq", "/bpo-glossary", "/blog"}

VERTICAL_DEFAULT_SERVICE_PAGE = {
    "Healthcare & RCM": "/industries/healthcare",
    "Finance, Accounting & Tax": "/teams/finance-and-accounting-outsourcing-services",
    "Tech, IT & AI Staffing": "/teams/it-development",
    "Customer Experience (CX) & VAs": "/teams/customer-support",
    "Back Office & Operations Support": "/teams/business-operation-support",
    "Real Estate, Mortgage & Title/Escrow": "/industries/real-estate",
    "Insurance & Claims": "/industries/insurance/insurance-outsourcing",
    "Legal & Compliance": "/industries/legal",
    "Fintech & Private Equity": "/industries/fintech",
    "Engineering & Construction": "/industries/engineering/engineering-outsourcing-solutions",
    "Sales, Marketing & E-Commerce": "/teams/marketing",
    "Hospitality, Education & HR": "/industries/human-resources",
    "General B2B Outsourcing & Staffing": "/our-service-model",
}

DEFAULT_SITE_URL = "sc-domain:connextglobal.com"
DEFAULT_BRAND_REGEX = r"\b(connext|connext\s*global(\s*solutions)?|connextglobal)\b"

STRICT_NON_INDUSTRY_REGEX = re.compile(
    r"\b("
    r"job|jobs|career|careers|hiring|hire me|salary|salaries|compensation|wage|pay rate|"
    r"apply|application|applicant|interview|resume|cv|cover letter|internship|intern|interns|"
    r"vacancy|vacancies|opening|openings|recruiter|recruiting|work from home|wfh|part time|"
    r"glassdoor|indeed|jobstreet|mynimo|kalibrr|ziprecruiter|monster|linkedin login|"
    r"employee|employees|staff portal|login|sign in|payslip|payroll login|benefits|reviews|"
    r"address|phone number|contact number|directions|map|free template|wikipedia"
    r")\b",
    re.I,
)

LOCAL_PH_AND_STUDENT_REGEX = re.compile(
    r"\b("
    r"school|schools|university|universities|college|colleges|student|students|tuition|courses|admission|strand|"
    r"baguio|clark|bgc|bonifacio global city|makati|ortigas|quezon city|alabang|iloilo|davao|cebu|pampanga|taguig|"
    r"list of bpo|top bpo companies in|call center companies in"
    r")\b",
    re.I,
)

OFF_TOPIC_NOISE_RE = re.compile(
    r"\b("
    r"holiday|holidays|weather|forecast|temperature|typhoon|storm|earthquake|"
    r"time in|current time|time zone|currency|exchange rate|peso to|usd to|php to|cop to|"
    r"flight|flights|hotel|hotels|resort|beach|tourist|tourism|visa requirements|passport|"
    r"translate|translation|in tagalog|in spanish|tagalog|filipino word|pronounce|"
    r"quote|quotes|meme|memes|joke|jokes|poem|song|lyrics|movie|netflix|nba|nfl|fifa|score|"
    r"recipe|restaurant|coffee|food|menu|mall|traffic|grab|uber|angkas|"
    r"w2 form|1040|1099 form|turbotax|irs refund|stimulus|unemployment benefits|sss|pag-?ibig|philhealth|bir form|"
    r"13th month pay|dole|labor code|sick leave|maternity leave|resignation letter|"
    r"excel formula|vlookup|pivot table|google sheets formula|powerpoint template|word template|pdf converter|"
    r"personal loan|credit card|car insurance|auto insurance|homeowners insurance|life insurance quote|"
    r"birthday|christmas|new year|valentine|halloween|thanksgiving|easter|good friday|holy week|"
    r"essay|homework|thesis|quiz|exam|flashcard|mcq|reviewer|"
    r"kpi|kpis|metrics|interview questions|job description|skills needed|how to become|cpt code|icd 10|icd-10"
    r")\b",
    re.I,
)

CONNEXT_DELIVERY_CORE_RE = re.compile(
    r"\b("
    r"outsourc\w*|offshor\w*|nearshor\w*|bpo|kpo|rpo|staffing|staff augmentation|"
    r"dedicated team|remote team|global team|extended team|co-?management|"
    r"employer of record|eor|rebadg\w*|contractor transition|"
    r"virtual assistant|virtual medical assistant|virtual nurs\w*|"
    r"medical billing|medical coding|revenue cycle|rcm|prior auth\w*|credentialing|"
    r"claims management|denial management|dental billing|remote patient monitoring|"
    r"call center|contact center|back office|back-office|shared services|"
    r"title and escrow|title escrow|mortgage processing|loan processing|"
    r"servicenow staffing|python staffing|ai staffing"
    r")\b",
    re.I,
)

CONNEXT_DOMAIN_FUNCTION_RE = re.compile(
    r"\b("
    r"bookkeeping|bookkeeper|accounting|accountant|cpa|controller|payroll|"
    r"accounts payable|accounts receivable|debt collection|"
    r"software developer|developer|engineer|python|servicenow|machine learning|it support|helpdesk|qa testing|devops|"
    r"customer support|customer service|customer success|zendesk|technical support|"
    r"data entry|data conversion|document processing|order processing|operations support|"
    r"property management|insurance underwriting|insurance claims|claims adjuster|"
    r"paralegal|legal support|litigation support|private equity|fintech|"
    r"cad drafting|bim modeling|construction estimating|digital marketing|e-?commerce|lead generation|sdr|bdr|human resources"
    r")\b",
    re.I,
)

STRICT_B2B_TEAM_CONTEXT_RE = re.compile(
    r"\b("
    r"outsourc\w*|offshor\w*|nearshor\w*|bpo|staffing|dedicated|remote|virtual|"
    r"hire|hiring|company|companies|firm|firms|provider|providers|partner|vendor|"
    r"philippines|colombia|mexico|india|latam|latin america|team|specialist|specialists|"
    r"co-?management|eor|rebadg\w*|cost|pricing|rates"
    r")\b",
    re.I,
)

# --- B2B CONVERSION FUNNEL CLASSIFIER (TOFU / MOFU / BOFU) ---
TOFU_DEF_RE = re.compile(
    r"\b(meaning|mean|means|definition|define|stands for|acronym|glossary|"
    r"what is|what are|what does|what do|explain|explained|overview|basics|"
    r"examples? of|history of|types of|introduction to)\b",
    re.I,
)
TOFU_EDU_RE = re.compile(
    r"\b(why is|why do|benefits of|advantages of|disadvantages of|challenges of|"
    r"trends|statistics|stats|role of|responsibilities|duties|skills needed|"
    r"importance of|how does .* work|guide to|what happens)\b",
    re.I,
)
BOFU_PRICE_RE = re.compile(
    r"\b(cost|costs|pricing|price|prices|rates?|fee|fees|quote|quotes|calculator|how much|roi)\b",
    re.I,
)
BOFU_BUY_RE = re.compile(
    r"\b(hire|hiring|agency|agencies|company|companies|firm|firms|provider|providers|"
    r"partner|partners|vendor|vendors|services?|consultants?|"
    r"dedicated team|staff augmentation|for hire|get started|book a|request a)\b",
    re.I,
)
BOFU_ACTION_VERB_RE = re.compile(
    r"\boutsource\s+(my|your|our|a)?\s*(medical|dental|accounting|bookkeeping|payroll|it|software|"
    r"customer|claims|billing|coding|prior|credentialing|back office|finance|hr|marketing|legal|"
    r"mortgage|title|insurance|engineering|data|support|operations|call center|rcm|tax|cpa|controller|rebadging)\b",
    re.I,
)
BUYER_DELIVERY_RE = re.compile(r"\b(outsourcing|outsource|staffing|bpo)\b", re.I)
HOW_TO_EVAL_RE = re.compile(
    r"\b(how to hire|how to outsource|how to build|how to transition|how to choose|how to find|how to set up|how to manage|steps to outsource|checklist)\b",
    re.I,
)
GEO_RE = re.compile(r"\b(philippines|colombia|mexico|india|latam|latin america|offshore|nearshore)\b", re.I)
CONNEXT_SPECIFIC_FUNCTION_RE = re.compile(
    r"\b(rebadging|contractor transitions?|employer of record|eor|co-?management|"
    r"medical billing|medical coding|revenue cycle|rcm|prior authorization|prior auth|"
    r"credentialing|claims management|dental billing|remote patient monitoring|virtual nursing|"
    r"virtual medical assistants?|bookkeeping|bookkeepers?|accounting|accountants?|cpas?|controllers?|"
    r"payroll|debt collection|accounts payable|accounts receivable|"
    r"software developers?|developers?|engineers?|python|servicenow|ai staffing|"
    r"machine learning|virtual assistants?|call centers?|technical support|customer support|"
    r"customer service|customer success|zendesk|back office|data entry|mortgage|title escrow|"
    r"insurance underwriting|paralegals?|digital marketing|e-?commerce)\b",
    re.I,
)
MOFU_EVAL_RE = re.compile(
    r"\b(best|top|vs|versus|compare|comparison|alternatives?|review|reviews|"
    r"framework|strategy|models?|in-?house vs|onshore vs|offshore vs|nearshore vs|"
    r"pros and cons|benefits of outsourcing|why outsource|co-?management|managed services|"
    r"rebadging|contractor transitions?|employer of record|eor|staffing model|dedicated|process|workflow)\b",
    re.I,
)

COMMERCIAL_MODIFIERS = re.compile(r"\b(outsourcing|outsource|staffing|offshore|nearshore|dedicated|bpo|agency|company|firm|services|solutions|cost|pricing|rates|quote|remote team|vs|versus|best|top|partner|provider|hire)\b", re.I)
AI_PROMPT_REGEX = re.compile(r"^(how|what|why|when|where|which|can|does|is|are|should|compare|difference between|pros and cons|steps to|best way to)\b", re.I)

ALL_CONNEXT_VERTICALS = [
    "Healthcare & RCM", "Finance, Accounting & Tax", "Tech, IT & AI Staffing",
    "Customer Experience (CX) & VAs", "Back Office & Operations Support",
    "Real Estate, Mortgage & Title/Escrow", "Insurance & Claims", "Legal & Compliance",
    "Fintech & Private Equity", "Engineering & Construction", "Sales, Marketing & E-Commerce",
    "Hospitality, Education & HR", "General B2B Outsourcing & Staffing",
]

SECTOR_PATTERNS = [
    ("Healthcare & RCM", re.compile(r"(/industries/healthcare|hipaa-compliant-virtual-medical|\b(medical|rcm|revenue cycle|healthcare|health care|billing|coding|icd|cpt|patient|prior auth|authorization|credentialing|clinic|hospital|dental|hipaa|ehr|emr|pharmacy|nursing|remote monitoring)\b)", re.I)),
    ("Finance, Accounting & Tax", re.compile(r"(/teams/finance|/teams/financial-services|payroll-outsourcing|\b(accounting|accountant|bookkeeping|bookkeeper|accounts payable|accounts receivable|cpa|controller|cfo|tax|audit|auditing|payroll|finance|financial|reconciliation|debt collection|ap|ar)\b)", re.I)),
    ("Tech, IT & AI Staffing", re.compile(r"(/teams/it-development|/teams/ai-staffing|\b(software|developer|developers|engineer|engineering|it support|helpdesk|help desk|qa|quality assurance|testing|data|cloud|devops|cybersecurity|tech|programmer|python|servicenow|machine learning|ai staffing|ai)\b)", re.I)),
    ("Customer Experience (CX) & VAs", re.compile(r"(/teams/customer-support|\b(customer service|customer care|customer support|client support|client success|call center|contact center|omnichannel|live chat|tier 1|tier 2|technical support|zendesk|virtual assistant)\b)", re.I)),
    ("Back Office & Operations Support", re.compile(r"(/teams/business-operation-support|/teams/back-office|\b(back office|back-office|admin|administrative|data entry|data conversion|document processing|order processing|catalog|content moderation|transcription|indexing|operations support|shared services|procurement)\b)", re.I)),
    ("Real Estate, Mortgage & Title/Escrow", re.compile(r"(/industries/real-estate|\b(real estate|property management|mortgage|title|escrow|leasing|appraisal|loan processing)\b)", re.I)),
    ("Insurance & Claims", re.compile(r"(/industries/insurance|\b(insurance|underwriting|claims|policy|actuarial|adjuster)\b)", re.I)),
    ("Legal & Compliance", re.compile(r"(/industries/legal|\b(legal|paralegal|attorney|law firm|contract|compliance|regulatory|litigation)\b)", re.I)),
    ("Fintech & Private Equity", re.compile(r"(/industries/fintech|/industries/private-equity|\b(fintech|private equity|portfolio company|portfolio companies|investment|wealth management|payment)\b)", re.I)),
    ("Engineering & Construction", re.compile(r"(/industries/engineering|/industries/construction|\b(civil engineering|structural|cad|bim|construction|estimating|architectural|architecture)\b)", re.I)),
    ("Sales, Marketing & E-Commerce", re.compile(r"(/teams/marketing|\b(sdr|bdr|sales development|lead generation|marketing|digital marketing|seo|paid media|ppc|graphic design|video editing|copywriting|social media|hubspot|salesforce|crm|ecommerce|e-commerce)\b)", re.I)),
    ("Hospitality, Education & HR", re.compile(r"(/industries/hospitality|/industries/education|/industries/human-resources|\b(hospitality|hotel|travel|education|university|higher ed|human resources|hr outsourcing|recruiting)\b)", re.I)),
]

MONEY_PAGE_BOOST_RULES = [
    (re.compile(r"\bdental\b", re.I), "/industries/healthcare/dental-billing-outsourcing"),
    (re.compile(r"\bprior auth", re.I), "/industries/healthcare/outsource-prior-authorization"),
    (re.compile(r"\bcredentialing\b", re.I), "/industries/healthcare/outsource-credentialing"),
    (re.compile(r"\bmedical claims|claims management|denial management\b", re.I), "/industries/healthcare/outsource-claims-management"),
    (re.compile(r"\bmedical coding|icd|cpt\b", re.I), "/industries/healthcare/medical-coding"),
    (re.compile(r"\bmedical billing\b", re.I), "/industries/healthcare/medical-billing"),
    (re.compile(r"\b(rcm|revenue cycle)\b", re.I), "/industries/healthcare/revenue-cycle-management"),
    (re.compile(r"\b(virtual nursing|nurse)\b", re.I), "/industries/healthcare/virtual-nursing"),
    (re.compile(r"\b(remote monitoring|rpm)\b", re.I), "/industries/healthcare/remote-monitoring"),
    (re.compile(r"\b(healthcare software|ehr|emr|healthtech)\b", re.I), "/industries/healthcare/healthcare-software"),
    (re.compile(r"\b(virtual medical assistant|hipaa)\b", re.I), "/hipaa-compliant-virtual-medical-assistants"),
    (re.compile(r"\b(healthcare|medical|patient|clinic)\b", re.I), "/industries/healthcare"),
    (re.compile(r"\b(cpa|certified public accountant)\b", re.I), "/teams/financial-services/hire-certified-public-accountants-with-connext"),
    (re.compile(r"\bcontroller\b", re.I), "/teams/financial-services/hire-expert-controllers-with-connext"),
    (re.compile(r"\bdebt collection\b", re.I), "/teams/financial-services/debt-collection-outsourcing"),
    (re.compile(r"\b(quickbooks|netsuite|xero|accounting software)\b", re.I), "/teams/financial-services/accounts-softwares"),
    (re.compile(r"\bpayroll\b", re.I), "/teams/operation-support/payroll-outsourcing"),
    (re.compile(r"\b(accounting|bookkeeping|accounts payable|accounts receivable|finance|tax)\b", re.I), "/teams/finance-and-accounting-outsourcing-services"),
    (re.compile(r"\bpython\b", re.I), "/teams/it-development/python-staffing"),
    (re.compile(r"\bservicenow\b", re.I), "/teams/it-development/servicenow-staffing"),
    (re.compile(r"\b(machine learning|ml)\b", re.I), "/teams/it-development/machine-learing"),
    (re.compile(r"\b(ai staffing|artificial intelligence|ai team)\b", re.I), "/teams/ai-staffing"),
    (re.compile(r"\b(software|developer|engineer|qa|it|tech)\b", re.I), "/teams/it-development"),
    (re.compile(r"\bzendesk\b", re.I), "/teams/customer-support/zendesk-staffing"),
    (re.compile(r"\bvirtual assistant\b", re.I), "/teams/customer-support/virtual-assistants"),
    (re.compile(r"\bcall center\b", re.I), "/teams/customer-support/call-center-outsourcing"),
    (re.compile(r"\btechnical support\b", re.I), "/teams/customer-support/technical-support-outsourcing"),
    (re.compile(r"\bcustomer success\b", re.I), "/teams/customer-support/customer-success"),
    (re.compile(r"\b(customer support|customer service)\b", re.I), "/teams/customer-support"),
    (re.compile(r"\b(e-commerce|ecommerce)\b", re.I), "/teams/marketing/e-commerce-outsourcing"),
    (re.compile(r"\b(digital marketing|marketing|seo|ppc)\b", re.I), "/teams/marketing/outsource-digital-marketing"),
    (re.compile(r"\b(back office|operations support|data entry|data conversion|shared services|procurement|admin)\b", re.I), "/teams/business-operation-support"),
    (re.compile(r"\b(title|escrow)\b", re.I), "/industries/real-estate/title-escrow"),
    (re.compile(r"\bmortgage\b", re.I), "/industries/real-estate/mortgage"),
    (re.compile(r"\breal estate\b", re.I), "/industries/real-estate"),
    (re.compile(r"\binsurance\b", re.I), "/industries/insurance/insurance-outsourcing"),
    (re.compile(r"\b(legal|paralegal|law)\b", re.I), "/industries/legal"),
    (re.compile(r"\bfintech\b", re.I), "/industries/fintech"),
    (re.compile(r"\bprivate equity\b", re.I), "/industries/private-equity"),
    (re.compile(r"\bconstruction\b", re.I), "/industries/construction"),
    (re.compile(r"\bengineering\b", re.I), "/industries/engineering/engineering-outsourcing-solutions"),
    (re.compile(r"\bhospitality\b", re.I), "/industries/hospitality"),
    (re.compile(r"\beducation\b", re.I), "/industries/education"),
    (re.compile(r"\b(human resources|hr)\b", re.I), "/industries/human-resources"),
    (re.compile(r"\bphilippines\b", re.I), "/locations/outsource-to-the-philippines"),
    (re.compile(r"\bcolombia\b", re.I), "/locations/outsource-to-colombia"),
    (re.compile(r"\bmexico\b", re.I), "/locations/outsource-to-mexico"),
    (re.compile(r"\bindia\b", re.I), "/locations/outsource-to-india"),
    (re.compile(r"\bco-?management\b", re.I), "/services/co-management"),
    (re.compile(r"\b(employer of record|eor)\b", re.I), "/services/employer-of-record"),
    (re.compile(r"\b(contractor transition|rebadging)\b", re.I), "/services/contractor-transitions"),
    (re.compile(r"\b(soc 2|soc2|compliance|data security)\b", re.I), "/services/compliance"),
    (re.compile(r"\b(pricing|cost|rates|salary|roi)\b", re.I), "/connext-pricing"),
]

GENERIC_BRAND_STOPWORDS = {
    "bpo", "services", "service", "outsourcing", "outsource", "offshore", "nearshore",
    "staffing", "company", "companies", "solutions", "team", "teams", "hire", "best",
    "top", "cost", "pricing", "philippines", "colombia", "india", "mexico",
    "how", "what", "why", "when", "where", "the", "and", "for", "with", "into", "from",
}


# ==============================================================================
# UTILITY, RELEVANCE & SITE ARCHITECTURE CLASSIFIERS
# ==============================================================================
def normalize_url(url: str) -> str:
    u = str(url).strip().lower()
    if u.startswith("/"):
        u = f"https://connextglobal.com{u}"
    u = u.replace("http://", "https://").replace("https://www.", "https://")
    u = u.split("?")[0].split("#")[0]
    return u.rstrip("/") if u.endswith("/") and len(u) > len("https://connextglobal.com/") else u


def url_to_path(url: str) -> str:
    path = urlparse(normalize_url(url)).path
    if not path:
        return "/"
    return path.rstrip("/") if path != "/" and path.endswith("/") else path


def is_relevant_to_connext(query: str, strict: bool = True) -> bool:
    """Strictly filters out off-topic queries unrelated to Connext Global's B2B staffing & outsourcing services."""
    q = str(query).lower().strip()
    if not q or len(q) < 4:
        return False
    if OFF_TOPIC_NOISE_RE.search(q):
        return False
    if not strict:
        return True
    if CONNEXT_DELIVERY_CORE_RE.search(q):
        return True
    if CONNEXT_DOMAIN_FUNCTION_RE.search(q) and STRICT_B2B_TEAM_CONTEXT_RE.search(q):
        return True
    return False


def classify_page_role(path: str, live_paths_set: set[str]) -> str:
    """
    Accurately classifies connextglobal.com URLs, including migrated root-level blogs
    published WITHOUT a /blog/ prefix so they are NEVER mislabeled as Service Pages.
    """
    p = path.rstrip("/") if path != "/" else "/"
    if p not in live_paths_set:
        return "Archived / Historical URL"
    if p == "/":
        return "Homepage"
    if p in EXACT_UTILITY_PATHS:
        return "Utility / Brand Page"
    if p in EXACT_CONTENT_HUB_PATHS:
        return "Content Hub / Glossary"
    if p == "/hipaa-compliant-virtual-medical-assistants" or (p.startswith("/industries/") and "/blog/" not in p):
        return "Industry Page"
    if p.startswith("/teams/") and "/blog/" not in p:
        return "Team Solution Page"
    if p.startswith("/services/") or p in EXACT_CORE_COMMERCIAL_PATHS:
        return "Service Model Page"
    if p.startswith("/locations"):
        return "Location Hub"
    if p.startswith("/tools"):
        return "Interactive Tool"
    if p.startswith("/case-studies/"):
        return "Case Study"
    if p.startswith("/white-paper/"):
        return "White Paper"
    if p.startswith("/one-pager/"):
        return "One-Pager"
    if p.startswith("/e-books/"):
        return "E-Book"
    if p.startswith("/blog/") or p.startswith("/insights/") or p.startswith("/general-outsourcing/"):
        return "Blog / Guide"
    # Any other root-level slug on connextglobal.com is a migrated editorial blog post without /blog/
    return "Blog / Guide (Root URL)"


def classify_single_query_funnel(query: str) -> tuple[str, str]:
    q = str(query).lower().strip()
    has_price = bool(BOFU_PRICE_RE.search(q))
    has_buy = bool(BOFU_BUY_RE.search(q))
    has_action = bool(BOFU_ACTION_VERB_RE.search(q))
    has_delivery = bool(BUYER_DELIVERY_RE.search(q))
    has_how_eval = bool(HOW_TO_EVAL_RE.search(q))
    has_def = bool(TOFU_DEF_RE.search(q))
    has_edu = bool(TOFU_EDU_RE.search(q))
    has_mofu = bool(MOFU_EVAL_RE.search(q))
    has_geo = bool(GEO_RE.search(q))
    has_func = bool(CONNEXT_SPECIFIC_FUNCTION_RE.search(q))
    has_service = has_func or has_delivery

    if has_price and (has_service or has_geo or has_buy):
        return "BOFU — Ready to Buy", "Pricing / Cost Intent"
    if has_def and not (has_buy or has_action):
        return "TOFU — Awareness", "Definition / Meaning Lookup"
    if has_how_eval or re.search(r"\b(vs|versus|compare|comparison|pros and cons)\b", q):
        return "MOFU — Evaluating", "Evaluating Execution / Comparison"
    if has_edu and not (has_buy or has_action or has_price):
        return "TOFU — Awareness", "Educational / Trends"
    if has_action or (has_buy and (has_service or has_geo)):
        return "BOFU — Ready to Buy", "Vendor / Hiring Intent"
    if (has_func and has_delivery) or (has_geo and has_service):
        return "BOFU — Ready to Buy", "Commercial Service Intent"
    if has_mofu:
        return "MOFU — Evaluating", "Evaluating B2B Model"
    if has_func or has_delivery:
        return "MOFU — Evaluating", "Core B2B Service Research"
    return "TOFU — Awareness", "General Research"


def classify_cluster_funnel(topic: str, query_imp_pairs: list[tuple[str, int]], top_page_role: str = "") -> tuple[str, str]:
    scores = {"TOFU — Awareness": 0.0, "MOFU — Evaluating": 0.0, "BOFU — Ready to Buy": 0.0}
    reason_weights = {}

    t_stage, t_reason = classify_single_query_funnel(topic)
    total_imp = sum(imp for _, imp in query_imp_pairs) or 1
    scores[t_stage] += total_imp * 0.45
    reason_weights[(t_stage, t_reason)] = reason_weights.get((t_stage, t_reason), 0.0) + total_imp * 0.45

    for q, imp in query_imp_pairs:
        st_lbl, rsn = classify_single_query_funnel(q)
        w = max(float(imp), 1.0)
        scores[st_lbl] += w
        reason_weights[(st_lbl, rsn)] = reason_weights.get((st_lbl, rsn), 0.0) + w

    if top_page_role in ("Industry Page", "Team Solution Page", "Service Model Page"):
        scores["BOFU — Ready to Buy"] += total_imp * 0.08
    elif "Blog" in top_page_role:
        scores["TOFU — Awareness"] += total_imp * 0.05

    winning_stage = max(scores.items(), key=lambda x: x[1])[0]
    best_reason = max(
        ((rsn, w) for (stg, rsn), w in reason_weights.items() if stg == winning_stage),
        key=lambda x: x[1],
        default=(t_reason, 1.0),
    )[0]
    return winning_stage, best_reason


def human_rank_label(pos: float) -> str:
    if pos <= 0:
        return "Not Ranking"
    if pos <= 3.0:
        return f"Top 3 (#{pos:.1f})"
    if pos <= 10.0:
        return f"Page 1 (#{pos:.1f})"
    if pos <= 20.0:
        return f"Page 2 (#{pos:.1f})"
    return f"Page 3+ (#{pos:.1f})"


def classify_sector(text: str) -> str:
    for sector_name, pattern in SECTOR_PATTERNS:
        if pattern.search(text):
            return sector_name
    return "General B2B Outsourcing & Staffing"


def build_enterprise_excel_bytes(sheets_dict: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets_dict.items():
            if df is None or df.empty:
                continue
            safe_name = re.sub(r"[\[\]:*?/\\]", "", name)[:31]
            out_df = df.copy()
            drop_cols = [c for c in out_df.columns if out_df[c].apply(lambda x: isinstance(x, (list, dict))).any()]
            if drop_cols:
                out_df = out_df.drop(columns=drop_cols)
            out_df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.sheets[safe_name]
            for col_cells in ws.columns:
                max_len = max((len(str(c.value or "")) for c in col_cells), default=10)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 3, 12), 48)
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_sitemap_paths() -> tuple[list[str], str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ConnextTrendRadar/3.4)"}
    try:
        with urllib.request.urlopen(urllib.request.Request(SITEMAP_XML_URL, headers=headers), timeout=8) as resp:
            root = ET.fromstring(resp.read())
        locs = [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text]
        child_sitemaps = [u for u in locs if u.lower().endswith(".xml")]
        if child_sitemaps:
            all_urls = []
            for sm_url in child_sitemaps[:10]:
                with urllib.request.urlopen(urllib.request.Request(sm_url, headers=headers), timeout=8) as c_resp:
                    c_root = ET.fromstring(c_resp.read())
                all_urls.extend([el.text.strip() for el in c_root.iter() if el.tag.endswith("loc") and el.text])
            locs = all_urls
        paths = sorted({url_to_path(u) for u in locs if "connextglobal.com" in u.lower()})
        if len(paths) >= 50:
            return paths, f"Live sitemap.xml ({len(paths):,} URLs)"
    except Exception:
        pass

    try:
        with urllib.request.urlopen(urllib.request.Request(SITEMAP_HTML_URL, headers=headers), timeout=8) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
        paths = sorted({url_to_path(h) for h in re.findall(r'href=["\']((?:https?://(?:www\.)?connextglobal\.com)?/[^"\'#?]*)["\']', html_text, re.I)})
        if len(paths) >= 50:
            return paths, f"Live /sitemap ({len(paths):,} URLs)"
    except Exception:
        pass

    fallback = sorted(set(CORE_LIVE_PATHS))
    return fallback, f"Core Pages Fallback ({len(fallback)} URLs)"


def _build_index_from_paths(all_paths: list[str]):
    live_set = set(all_paths)
    live_df = pd.DataFrame({"path": sorted(live_set)})
    live_df["full_url"] = live_df["path"].apply(lambda p: "https://connextglobal.com/" if p == "/" else f"https://connextglobal.com{p}")
    live_df["role"] = live_df["path"].apply(lambda p: classify_page_role(p, live_set))
    live_df["is_money_page"] = live_df["role"].isin(["Industry Page", "Team Solution Page", "Service Model Page", "Location Hub"])
    live_df["is_content_page"] = live_df["role"].isin([
        "Blog / Guide", "Blog / Guide (Root URL)", "Case Study", "White Paper", "One-Pager", "E-Book"
    ])
    live_df["slug_text"] = (
        live_df["path"]
        .str.replace(r"^/(industries|teams|services|locations|tools|blog|case-studies|white-paper|one-pager|e-books|insights)/", "", regex=True)
        .str.replace(r"[/\-_]", " ", regex=True)
        .str.strip()
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
    mat = vec.fit_transform(live_df["slug_text"].fillna(""))
    return live_set, live_df, vec, mat


def check_topic_against_sitemap(
    topic: str,
    queries_sample: str,
    vertical: str,
    live_set: set[str],
    live_df: pd.DataFrame,
    vec: TfidfVectorizer,
    mat,
    gsc_live_urls: list[str] | None = None,
) -> dict:
    """
    Strictly separates verified Commercial Service Pages (`best_money_url` — NEVER a blog)
    from existing Editorial Blogs (both `/blog/...` and root-level blogs without `/blog/`),
    and honors live URLs already ranking in GSC.
    """
    combined_text = f"{topic} {queries_sample}".strip()
    sims = cosine_similarity(vec.transform([combined_text]), mat).flatten()
    scored = live_df.copy()
    scored["raw_sim"] = sims
    scored["sim"] = sims

    topic_tokens = {w for w in re.findall(r"[a-z0-9]+", topic.lower()) if len(w) > 2 and w not in GENERIC_BRAND_STOPWORDS}
    scored["token_overlap"] = scored["slug_text"].apply(
        lambda s: (len(topic_tokens & set(re.findall(r"[a-z0-9]+", str(s).lower()))) / len(topic_tokens)) if topic_tokens else 0.0
    )

    boosted_path = None
    for pat, target_path in MONEY_PAGE_BOOST_RULES:
        if pat.search(combined_text) and target_path in live_set:
            boosted_path = target_path
            scored.loc[scored["path"] == target_path, "sim"] += 0.45
            break

    money_df = scored[scored["is_money_page"]].sort_values("sim", ascending=False)
    content_df = scored[scored["is_content_page"]].sort_values(["raw_sim", "token_overlap"], ascending=False)
    best_money_row = money_df.iloc[0] if not money_df.empty else scored.iloc[0]

    # Always resolve a verified commercial Service/Industry/Team/Location page (NEVER a blog!)
    if boosted_path is not None:
        best_money_url = f"https://connextglobal.com{boosted_path}"
        best_money_role = classify_page_role(boosted_path, live_set)
        has_strong_money = True
    elif not money_df.empty and float(money_df.iloc[0]["sim"]) >= 0.24:
        best_money_url = money_df.iloc[0]["full_url"]
        best_money_role = money_df.iloc[0]["role"]
        has_strong_money = True
    else:
        fallback_path = VERTICAL_DEFAULT_SERVICE_PAGE.get(vertical, "/our-service-model")
        best_money_url = f"https://connextglobal.com{fallback_path}"
        best_money_role = classify_page_role(fallback_path, live_set) if fallback_path in live_set else "Service Model Page"
        has_strong_money = False

    # Existing coverage check: check live GSC ranking URLs first, then sitemap slug/token matches
    eligible_df = scored[scored["is_money_page"] | scored["is_content_page"]]
    existing_matches = eligible_df[
        (eligible_df["raw_sim"] >= 0.30) | (eligible_df["token_overlap"] >= 0.67)
    ].sort_values(["token_overlap", "raw_sim"], ascending=False)

    valid_gsc_live = [
        u for u in (gsc_live_urls or [])
        if url_to_path(u) in live_set and classify_page_role(url_to_path(u), live_set) not in {"Homepage", "Utility / Brand Page", "Content Hub / Glossary"}
    ]

    if valid_gsc_live:
        already_exists = True
        closest_url = valid_gsc_live[0]
        closest_role = classify_page_role(url_to_path(closest_url), live_set)
        closest_pct = 100
    elif len(existing_matches) > 0:
        already_exists = True
        closest_url = existing_matches.iloc[0]["full_url"]
        closest_role = existing_matches.iloc[0]["role"]
        closest_pct = int(round(max(float(existing_matches.iloc[0]["raw_sim"]), float(existing_matches.iloc[0]["token_overlap"])) * 100))
    else:
        already_exists = False
        closest_url = best_money_url
        closest_role = best_money_role
        closest_pct = 0

    return {
        "best_money_url": best_money_url,
        "best_money_role": best_money_role,
        "has_strong_money_match": has_strong_money,
        "already_exists_in_sitemap": already_exists,
        "closest_existing_url": closest_url,
        "closest_existing_role": closest_role,
        "closest_overlap_pct": min(100, closest_pct),
        "existing_sitemap_matches": existing_matches.head(3)[["full_url", "role"]].to_dict("records"),
        "top_money_pages": money_df.head(3)[["full_url", "role", "sim"]].to_dict("records"),
        "top_content_pages": content_df[content_df["raw_sim"] >= 0.10].head(5)[["full_url", "role", "raw_sim"]].to_dict("records"),
    }


def build_rsa_assets(topic: str) -> tuple[list[tuple[str, str]], list[str]]:
    clean_kw = re.sub(r"\s+", " ", topic.strip()).title()
    short_kw = ""
    for w in clean_kw.split():
        cand = f"{short_kw} {w}".strip()
        if len(cand) <= 22:
            short_kw = cand
        else:
            break
    short_kw = short_kw or clean_kw[:22].strip()
    raw_h = [
        ("Keyword", short_kw[:30]), ("Keyword", f"Outsource {short_kw}"), ("Keyword", f"Dedicated {short_kw}"),
        ("Value", "Build Your Dedicated Team"), ("Value", "Global Scale, Local Touch"),
        ("Value", "Custom Offshore Staffing"), ("Value", "Seamless Team Integration"),
        ("CTA", "Scale Your Team Today"), ("CTA", "Book a Staffing Consultation"), ("CTA", "Request Custom Team Pricing"),
    ]
    valid_h, seen_h = [], set()
    for cat, h in raw_h:
        hc = h.strip()
        if len(hc) <= 30 and hc.lower() not in seen_h:
            valid_h.append((cat, hc))
            seen_h.add(hc.lower())
    raw_d = [
        f"Extend your team with dedicated {short_kw.lower()} talent built around your workflows.",
        "Overcome talent shortages and high overhead with transparent, custom offshore staffing.",
        "People-centric global teams that integrate seamlessly with your culture and operations.",
        "Scale faster with full operational support, high retention, and dedicated management.",
    ]
    return valid_h, [d[:90].strip() for d in raw_d if len(d) <= 90]


def build_faq_jsonld(topic: str, queries_list: list[str]) -> str:
    q_words = re.compile(r"\b(how|what|why|when|where|which|can|does|is|are|should|cost|pricing|rates|vs|versus|benefits)\b", re.I)
    questions = [q.capitalize() + ("?" if not q.endswith("?") else "") for q in queries_list if q_words.search(q)][:6]
    if not questions:
        questions = [
            f"How does {topic} work with Connext Global?",
            f"What is the cost of {topic} compared to in-house hiring?",
            f"How fast can Connext build a dedicated {topic} team?",
        ]
    schema_obj = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": q_text,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"Connext Global builds dedicated, co-managed {topic} teams tailored to your workflows, compliance requirements, and growth goals.",
                },
            }
            for q_text in questions
        ],
    }
    return f'<script type="application/ld+json">\n{json.dumps(schema_obj, indent=2)}\n</script>'


# ==============================================================================
# LIVE ON-PAGE SEO & AI-READINESS GRADER ($0)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def run_live_onpage_seo_audit(url: str, target_keywords: tuple[str, ...]) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as resp:
            raw_html = resp.read(250000).decode("utf-8", errors="ignore")
    except Exception as exc:
        return {"error": f"Could not fetch live page: {exc}"}

    clean_html = re.sub(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S)
    m_t = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
    title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m_t.group(1)))).strip() if m_t else ""

    m_d = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', raw_html, re.I | re.S)
    if not m_d:
        m_d = re.search(r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']', raw_html, re.I | re.S)
    meta_desc = re.sub(r"\s+", " ", html.unescape(m_d.group(1))).strip() if m_d else ""

    h1s = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip() for x in re.findall(r"<h1[^>]*>(.*?)</h1>", clean_html, re.I | re.S)]
    h2s = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip() for x in re.findall(r"<h2[^>]*>(.*?)</h2>", clean_html, re.I | re.S)]
    word_count = len(re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", clean_html))).split())

    has_jsonld = bool(re.search(r"application/ld\+json", raw_html, re.I))
    has_faq_schema = bool(re.search(r'"FAQPage"', raw_html, re.I))
    primary_kw = target_keywords[0].lower() if target_keywords else ""
    kw_in_title = bool(primary_kw and primary_kw in title.lower())
    kw_in_h1 = bool(primary_kw and any(primary_kw in h.lower() for h in h1s))

    checks, score = [], 0
    if 30 <= len(title) <= 65:
        score += 18; checks.append({"Check": "Title Tag Length", "Status": "Pass", "Detail": f"{len(title)} chars — {title[:75]}"})
    else:
        checks.append({"Check": "Title Tag Length", "Status": "Optimize", "Detail": f"{len(title)} chars (target 30–65) — {title[:75]}"})

    if 110 <= len(meta_desc) <= 165:
        score += 14; checks.append({"Check": "Meta Description", "Status": "Pass", "Detail": f"{len(meta_desc)} chars"})
    else:
        checks.append({"Check": "Meta Description", "Status": "Optimize", "Detail": f"{len(meta_desc)} chars (target 110–165)"})

    if len(h1s) == 1:
        score += 14; checks.append({"Check": "H1 Main Heading", "Status": "Pass", "Detail": f"1 H1: {h1s[0][:75]}"})
    else:
        checks.append({"Check": "H1 Main Heading", "Status": "Fix", "Detail": f"Found {len(h1s)} H1 tags (use exactly 1)"})

    if len(h2s) >= 3:
        score += 14; checks.append({"Check": "H2 Subheadings (Scannability)", "Status": "Pass", "Detail": f"{len(h2s)} H2 subheadings detected"})
    else:
        checks.append({"Check": "H2 Subheadings (Scannability)", "Status": "Optimize", "Detail": f"Only {len(h2s)} H2 tags (add 3+ question H2s)"})

    if kw_in_title or kw_in_h1:
        score += 16; checks.append({"Check": "Target Keyword Match", "Status": "Pass", "Detail": f"'{primary_kw}' appears in Title/H1"})
    else:
        checks.append({"Check": "Target Keyword Match", "Status": "Missing", "Detail": f"Include '{primary_kw}' in Title or H1"})

    if word_count >= 600:
        score += 12; checks.append({"Check": "Content Depth", "Status": "Pass", "Detail": f"~{word_count:,} words"})
    else:
        checks.append({"Check": "Content Depth", "Status": "Thin", "Detail": f"~{word_count:,} words (expand to 700+ words)"})

    if has_jsonld:
        score += 12; checks.append({"Check": "Structured Data (AI Readiness)", "Status": "Pass" if has_faq_schema else "Partial", "Detail": "FAQPage + JSON-LD active" if has_faq_schema else "JSON-LD found; add FAQPage from Tab 4"})
    else:
        checks.append({"Check": "Structured Data (AI Readiness)", "Status": "Missing", "Detail": "Add FAQPage JSON-LD from Tab 4"})

    return {"url": url, "score": score, "checks": pd.DataFrame(checks)}


# ==============================================================================
# KEYWORD PLANNER ($0 CSV PARSER) & GSC LOADERS (100% REAL CLICKS & IMPRESSIONS)
# ==============================================================================
def parse_keyword_planner_csv(uploaded_file) -> dict:
    fname = uploaded_file.name.lower()
    if fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raw_bytes = uploaded_file.read()
        for enc in ("utf-16", "utf-8-sig", "utf-8", "latin-1"):
            try:
                txt = raw_bytes.decode(enc)
                lines = [ln for ln in txt.splitlines() if ln.strip()]
                hdr_idx = next((i for i, ln in enumerate(lines[:10]) if "keyword" in ln.lower() and ("monthly searches" in ln.lower() or "bid" in ln.lower())), 0)
                df = pd.read_csv(io.StringIO("\n".join(lines[hdr_idx:])), sep=None, engine="python")
                break
            except Exception:
                continue
        else:
            raise ValueError("Could not decode Keyword Planner CSV.")

    df.columns = [str(c).lower().strip() for c in df.columns]
    kw_c = next((c for c in df.columns if "keyword" in c), None)
    vol_c = next((c for c in df.columns if "monthly searches" in c or "search volume" in c), None)
    low_c = next((c for c in df.columns if "low range" in c), None)
    high_c = next((c for c in df.columns if "high range" in c), None)
    comp_c = next((c for c in df.columns if c == "competition"), None)
    if not kw_c:
        raise ValueError("Missing 'Keyword' column in Keyword Planner CSV.")

    out = {}
    for _, r in df.iterrows():
        k = str(r[kw_c]).lower().strip()
        if not k or k == "nan":
            continue
        vol = int(pd.to_numeric(r[vol_c], errors="coerce") or 0) if vol_c else 0
        low_b = float(pd.to_numeric(r[low_c], errors="coerce") or 0.0) if low_c else 0.0
        high_b = float(pd.to_numeric(r[high_c], errors="coerce") or 0.0) if high_c else 0.0
        comp_s = str(r[comp_c]).strip() if comp_c and pd.notna(r[comp_c]) else "—"
        out[k] = {
            "kp_volume": vol,
            "kp_low_bid": round(low_b, 2),
            "kp_high_bid": round(high_b, 2),
            "kp_competition": comp_s,
        }
    return out


def _read_csv_or_zip(uploaded_file, preferred_names: list[str]) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".zip"):
        with zipfile.ZipFile(uploaded_file) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            if not csvs:
                raise ValueError("No CSV found in ZIP.")
            chosen = next((n for p in preferred_names for n in csvs if p in n.lower()), csvs[0])
            with zf.open(chosen) as f:
                return pd.read_csv(io.BytesIO(f.read()))
    return pd.read_csv(uploaded_file)


def parse_gsc_web_upload(uploaded_file, brand_regex: str, custom_negatives: str, exclude_local_ph: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_csv_or_zip(uploaded_file, ["quer", "page", "chart", "date"])
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={"top queries": "query", "top pages": "page"})
    missing = {"date", "query", "page", "impressions", "position"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}.")
    df["date"] = pd.to_datetime(df["date"])
    df["query"] = df["query"].astype(str).str.lower().str.strip()
    df["page"] = df["page"].astype(str).apply(normalize_url)
    df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce").fillna(0).astype(int) if "clicks" in df.columns else 0
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0).astype(int)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0)
    df = df[(df["query"] != "") & (df["impressions"] > 0)].copy()

    df["is_branded"] = df["query"].str.contains(brand_regex, regex=True, na=False) if brand_regex else False
    noise_mask = df["query"].str.contains(STRICT_NON_INDUSTRY_REGEX, regex=True, na=False)
    if exclude_local_ph:
        noise_mask = noise_mask | df["query"].str.contains(LOCAL_PH_AND_STUDENT_REGEX, regex=True, na=False)
    if custom_negatives.strip():
        noise_mask = noise_mask | df["query"].str.contains(custom_negatives, regex=True, na=False)

    neg_df = df[noise_mask].groupby("query", as_index=False)["impressions"].sum().sort_values("impressions", ascending=False).head(100)
    neg_df.columns = ["Wasted Non-B2B Query", "Wasted GSC Impr."]
    clean_df = df[~noise_mask].copy()
    return clean_df.reset_index(drop=True), neg_df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gsc_api_data(creds_dict: dict, site_url: str, lookback_days: int, brand_regex: str, custom_exclude_regex: str, exclude_local_ph: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/webmasters.readonly"])
    service = build("searchconsole", "v1", credentials=creds)
    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=lookback_days)

    rows, neg_counter = [], {}
    start_row, row_limit, max_rows = 0, 25000, 250000
    first_incomplete = None
    brand_pat = re.compile(brand_regex, re.I) if brand_regex.strip() else None
    custom_pat = re.compile(custom_exclude_regex, re.I) if custom_exclude_regex.strip() else None

    while start_row < max_rows:
        resp = service.searchanalytics().query(
            siteUrl=site_url.strip(),
            body={"startDate": start_dt.isoformat(), "endDate": end_dt.isoformat(), "dimensions": ["date", "query", "page"], "dataState": "all", "rowLimit": row_limit, "startRow": start_row},
        ).execute()
        if first_incomplete is None:
            first_incomplete = (resp.get("metadata") or {}).get("firstIncompleteDate", "None")
        batch = resp.get("rows", [])
        if not batch:
            break
        for r in batch:
            date_str, query, page = r["keys"]
            if (first_incomplete != "None" and date_str >= first_incomplete) or r["impressions"] <= 0:
                continue
            clean_q = query.lower().strip()
            if not clean_q:
                continue
            is_branded = bool(brand_pat.search(clean_q)) if brand_pat else False
            is_noise = bool(STRICT_NON_INDUSTRY_REGEX.search(clean_q)) or (exclude_local_ph and bool(LOCAL_PH_AND_STUDENT_REGEX.search(clean_q))) or (custom_pat and bool(custom_pat.search(clean_q)))
            if is_noise:
                neg_counter[clean_q] = neg_counter.get(clean_q, 0) + int(r["impressions"])
                continue
            rows.append({
                "date": date_str,
                "query": clean_q,
                "page": normalize_url(page),
                "clicks": int(r.get("clicks", 0)),
                "impressions": int(r["impressions"]),
                "position": float(r["position"]),
                "is_branded": is_branded,
            })
        if len(batch) < row_limit:
            break
        start_row += row_limit

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    neg_df = pd.DataFrame(sorted(neg_counter.items(), key=lambda x: -x[1])[:100], columns=["Wasted Non-B2B Query", "Wasted GSC Impr."])
    return df, neg_df


# ==============================================================================
# ON-DEMAND COMPETITOR NEW BLOGS + US ADS TRANSPARENCY + DISCOVERY
# ==============================================================================
def is_competitor_blog_url(url: str) -> bool:
    path = urlparse(url).path.strip()
    if not path or path == "/" or BLOG_EXCLUDE_PATH_RE.search(path):
        return False
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return False
    return bool(BLOG_INCLUDE_PATH_RE.search(path) and len(parts[-1]) >= 5 and not parts[-1].isdigit())


def extract_blog_keywords(slug: str, title: str, h1: str, meta_desc: str, h2s: list[str]) -> str:
    sources = [
        (slug.replace("-", " ").replace("_", " "), 3.2),
        (title, 3.0), (h1, 2.8), (" | ".join(h2s[:8]), 2.0), (meta_desc, 1.4),
    ]
    phrase_scores = {}
    for text, weight in sources:
        if not text:
            continue
        for chunk in re.split(r"[:|–—\-?,.;!()\[\]]+", text.lower()):
            tokens = [t for t in re.findall(r"[a-z0-9]+", chunk) if (len(t) > 2 or t in {"ai", "it", "hr", "qa", "va", "ap", "ar"}) and t not in COMP_BRAND_WORDS]
            for n in (3, 2):
                for i in range(len(tokens) - n + 1):
                    gram = tokens[i:i + n]
                    if any(w in BLOG_KW_STOPWORDS for w in gram):
                        continue
                    phrase = " ".join(gram)
                    phrase_scores[phrase] = phrase_scores.get(phrase, 0.0) + weight * (1.45 if n == 3 else 1.0)

    ranked = sorted(phrase_scores.items(), key=lambda x: (-x[1], -len(x[0])))
    selected, selected_sets = [], []
    for phrase, _ in ranked:
        ptoks = set(phrase.split())
        if any(phrase in ex or ex in phrase or len(ptoks & ex_t) >= 2 for ex, ex_t in zip(selected, selected_sets)):
            continue
        selected.append(phrase)
        selected_sets.append(ptoks)
        if len(selected) >= 4:
            break
    if not selected:
        fallback = [w for w in re.findall(r"[a-z]{3,}", slug.lower()) if w not in BLOG_KW_STOPWORDS and w not in COMP_BRAND_WORDS]
        return ", ".join(fallback[:4]) or slug.replace("-", " ")
    return ", ".join(selected)


def _scrape_blog_metadata(url: str, slug: str) -> tuple[str, str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    title_txt, h1_txt, meta_txt, h2_list = "", "", "", []
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=6) as resp:
            raw_html = resp.read(75000).decode("utf-8", errors="ignore")
        clean_html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S)
        m_t = re.search(r"<title[^>]*>(.*?)</title>", clean_html, re.I | re.S)
        if m_t:
            title_txt = re.split(r"\s+[|\-–—]\s+", re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m_t.group(1)))).strip())[0].strip()
        m_h1 = re.search(r"<h1[^>]*>(.*?)</h1>", clean_html, re.I | re.S)
        if m_h1:
            h1_txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m_h1.group(1)))).strip()
        m_desc = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', clean_html, re.I | re.S) or re.search(r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']', clean_html, re.I | re.S)
        if m_desc:
            meta_txt = re.sub(r"\s+", " ", html.unescape(m_desc.group(1))).strip()
        for h2_m in re.findall(r"<h2[^>]*>(.*?)</h2>", clean_html, re.I | re.S)[:8]:
            h2_c = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", h2_m))).strip()
            if 4 <= len(h2_c) <= 120:
                h2_list.append(h2_c)
    except Exception:
        pass
    headline = h1_txt if len(h1_txt) >= 8 else (title_txt if len(title_txt) >= 8 else slug.replace("-", " ").title())
    return headline, extract_blog_keywords(slug, title_txt, h1_txt, meta_txt, h2_list)


@st.cache_data(ttl=86400, show_spinner=False)
def scan_competitor_new_blogs_auto(lookback_days: int = 45) -> tuple[pd.DataFrame, list[str]]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)

    def scan_one(comp_tuple):
        cname, domain, _ = comp_tuple
        found, parsed_any = [], False
        for sm_url in [f"https://{domain}/sitemap_index.xml", f"https://{domain}/sitemap.xml", f"https://www.{domain}/sitemap_index.xml", f"https://www.{domain}/sitemap.xml"]:
            try:
                with urllib.request.urlopen(urllib.request.Request(sm_url, headers=headers), timeout=7) as resp:
                    root = ET.fromstring(resp.read())
                parsed_any = True
                child_locs = [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text and el.text.strip().lower().endswith(".xml")]
                sub_roots = []
                if child_locs:
                    blog_children = [u for u in child_locs if any(k in u.lower() for k in ["post", "blog", "article", "insight", "news", "resource"])]
                    for sub_u in (blog_children[:4] if blog_children else child_locs[:3]):
                        try:
                            with urllib.request.urlopen(urllib.request.Request(sub_u, headers=headers), timeout=7) as s_resp:
                                sub_roots.append(ET.fromstring(s_resp.read()))
                        except Exception:
                            continue
                else:
                    sub_roots = [root]

                for sroot in sub_roots:
                    for url_node in sroot.iter():
                        if not url_node.tag.endswith("url"):
                            continue
                        loc_txt, mod_txt = None, None
                        for ch in url_node:
                            if ch.tag.endswith("loc") and ch.text:
                                loc_txt = ch.text.strip()
                            elif ch.tag.endswith("lastmod") and ch.text:
                                mod_txt = ch.text.strip()[:10]
                        if not loc_txt or not mod_txt or not is_competitor_blog_url(loc_txt):
                            continue
                        try:
                            if datetime.strptime(mod_txt, "%Y-%m-%d").date() >= cutoff_date:
                                slug = [p for p in urlparse(loc_txt).path.split("/") if p][-1]
                                found.append({"Competitor": cname, "Published": mod_txt, "slug": slug, "Blog URL": loc_txt})
                        except ValueError:
                            continue
                        if len(found) >= 15:
                            break
                    if len(found) >= 15:
                        break
                break
            except Exception:
                continue
        return found, (f"{cname}: {len(found)} blogs" if parsed_any else f"{cname}: blocked")

    all_found, notes = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for f_rows, n_txt in ex.map(scan_one, COMPETITORS_REGISTRY):
            all_found.extend(f_rows)
            notes.append(n_txt)
    if not all_found:
        return pd.DataFrame(), notes

    out_df = pd.DataFrame(all_found).drop_duplicates(subset=["Blog URL"]).sort_values("Published", ascending=False).head(45).reset_index(drop=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        meta_results = list(ex.map(lambda p: _scrape_blog_metadata(p[0], p[1]), zip(out_df["Blog URL"], out_df["slug"])))
    out_df["Blog Headline"] = [m[0] for m in meta_results]
    out_df["Main Target Keywords"] = [m[1] for m in meta_results]
    out_df = out_df[~out_df["Blog Headline"].str.contains(OFF_TOPIC_NOISE_RE, regex=True, na=False)].reset_index(drop=True)
    if not out_df.empty:
        out_df["Vertical"] = out_df.apply(lambda r: classify_sector(f"{r['Blog Headline']} {r['Main Target Keywords']} {r['slug']}"), axis=1)
    return out_df, notes


def _serpapi_get(params: dict, api_key: str) -> dict:
    q_params = {**params, "api_key": api_key, "output": "json"}
    url = f"https://serpapi.com/search.json?{urllib.parse.urlencode(q_params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ConnextTrendRadar/3.4)"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


@st.cache_data(ttl=604800, show_spinner=False)
def run_on_demand_us_ads_transparency_scan(
    api_key: str,
    keywords_tuple: tuple[str, ...],
    scan_all_12_transparency: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Button-triggered only (never runs on load, cached 7 days)."""
    trans_rows, paid_rows, errors = [], [], []
    credits_used = 0

    if scan_all_12_transparency:
        def check_trans(comp_tuple):
            cname, domain, _ = comp_tuple
            tdata = _serpapi_get(
                {"engine": "google_ads_transparency_center", "text": domain, "region": "2840", "platform": "SEARCH", "num": 40},
                api_key,
            )
            creatives = tdata.get("ad_creatives") or []
            tot_res = int((tdata.get("search_information") or {}).get("total_results") or len(creatives))
            sample_adv = creatives[0].get("advertiser") if creatives else ""
            last_shown = creatives[0].get("last_shown") if creatives else ""
            return {
                "Competitor": cname,
                "Domain": domain,
                "US Google Search Ads": f"Active ({tot_res}+ creatives)" if (tot_res > 0 or len(creatives) > 0) else "No Active US Search Ads Found",
                "Active US Creatives": tot_res,
                "Verified Legal Payer": sample_adv or "—",
                "Last Shown": str(last_shown)[:10] if last_shown else "—",
                "Google Ads Transparency (US)": f"https://adstransparency.google.com/?region=US&domain={domain}",
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(check_trans, c): c[0] for c in COMPETITORS_REGISTRY}
            for fut in concurrent.futures.as_completed(futs):
                credits_used += 1
                try:
                    trans_rows.append(fut.result())
                except Exception as exc:
                    errors.append(str(exc))

    if keywords_tuple:
        def check_kw_ads(kw: str) -> list[dict]:
            data = _serpapi_get({"engine": "google", "q": kw, "location": "United States", "hl": "en", "gl": "us", "num": 10}, api_key)
            if "error" in data:
                raise RuntimeError(data["error"])
            hits = []
            for ad in data.get("ads", []):
                ad_link = str(ad.get("link") or ad.get("displayed_link") or "")
                matched = "Other Advertiser"
                for cname, domain, pat in COMPETITORS_REGISTRY:
                    if domain in ad_link.lower() or re.search(pat, str(ad.get("title") or ""), re.I):
                        matched = cname
                        break
                if "connextglobal.com" in ad_link.lower():
                    matched = "Connext Global (You)"
                hits.append({
                    "Trending Keyword": kw,
                    "Advertiser": matched,
                    "Tracked 12 Rival?": "Yes (Tracked Rival)" if matched not in ("Other Advertiser", "Connext Global (You)") else matched,
                    "Ad Slot": str(ad.get("block_position") or "Top").title(),
                    "Ad Headline": str(ad.get("title") or "").strip(),
                    "Ad Description": str(ad.get("description") or "").strip(),
                    "Landing Page": ad_link,
                })
            return hits

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(check_kw_ads, kw): kw for kw in keywords_tuple}
            for fut in concurrent.futures.as_completed(futs):
                credits_used += 1
                try:
                    paid_rows.extend(fut.result())
                except Exception as exc:
                    errors.append(str(exc))

    t_df = pd.DataFrame(trans_rows).sort_values("Active US Creatives", ascending=False).reset_index(drop=True) if trans_rows else pd.DataFrame()
    p_df = pd.DataFrame(paid_rows).drop_duplicates(subset=["Trending Keyword", "Advertiser", "Ad Headline"]).reset_index(drop=True) if paid_rows else pd.DataFrame()
    ts_str = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    if errors and t_df.empty and p_df.empty:
        return t_df, p_df, f"SerpApi Error: {errors[0]}"
    return t_df, p_df, f"Completed US Google Ads Transparency scan ({credits_used} SerpApi credits used • cached 7 days • {ts_str})"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_google_trends_b2b(seed_queries: tuple[str, ...], geo: str = "US", timeframe: str = "today 1-m", strict_b2b: bool = True):
    interest_rows, rising_rows, notes = [], [], []
    if HAS_TRENDSPY and seed_queries:
        try:
            tr = TrendsPyClient()
            batch_kws = list(seed_queries[:5])
            iot_df = tr.interest_over_time(batch_kws, geo=geo, timeframe=timeframe)
            if iot_df is not None and not iot_df.empty:
                iot_reset = iot_df.reset_index()
                date_col = iot_reset.columns[0]
                for kw in batch_kws:
                    if kw in iot_reset.columns:
                        for _, r in iot_reset.iterrows():
                            interest_rows.append({"date": pd.to_datetime(r[date_col]), "keyword": kw, "interest_score": int(r[kw])})
                notes.append("Relative Interest loaded via free Google Trends")
            for kw in batch_kws[:3]:
                try:
                    rq_dict = tr.related_queries(kw, geo=geo, timeframe=timeframe)
                    if isinstance(rq_dict, dict):
                        rising_df = rq_dict.get("rising")
                        if isinstance(rising_df, pd.DataFrame) and not rising_df.empty:
                            q_c = "query" if "query" in rising_df.columns else rising_df.columns[0]
                            v_c = "value" if "value" in rising_df.columns else rising_df.columns[-1]
                            for _, rr in rising_df.head(10).iterrows():
                                rq_clean = str(rr[q_c]).lower().strip()
                                if rq_clean and not STRICT_NON_INDUSTRY_REGEX.search(rq_clean) and is_relevant_to_connext(rq_clean, strict=strict_b2b):
                                    rising_rows.append({"Seed Topic": kw, "Rising B2B Query": rq_clean, "Surge": str(rr[v_c])})
                except Exception:
                    continue
        except Exception:
            notes.append("Google Trends chart rate-limited on cloud IP — B2B Industry Feed below remains active.")
    return pd.DataFrame(interest_rows), (pd.DataFrame(rising_rows).drop_duplicates(subset=["Rising B2B Query"]).reset_index(drop=True) if rising_rows else pd.DataFrame()), notes


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_b2b_industry_trending_feed(vertical_key: str) -> pd.DataFrame:
    query = INDUSTRY_NEWS_FEEDS.get(vertical_key, INDUSTRY_NEWS_FEEDS["All B2B Outsourcing & Staffing"])
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query, safe='')}&hl=en-US&gl=US&ceid=US:en"
    rows = []
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as resp:
            root = ET.fromstring(resp.read())
        for item in root.iter("item"):
            raw_title = (item.findtext("title") or "").strip()
            clean_title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title
            src_el = item.find("source")
            if clean_title:
                rows.append({
                    "Date": (item.findtext("pubDate") or "").strip()[:16],
                    "Trending B2B Headline": clean_title,
                    "Vertical": classify_sector(clean_title),
                    "Source": src_el.text.strip() if src_el is not None and src_el.text else "Industry News",
                    "Article Link": (item.findtext("link") or "").strip(),
                })
            if len(rows) >= 20:
                break
    except Exception:
        pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_microsoft_ad_library_all() -> pd.DataFrame:
    base = "https://adlibrary.api.bingads.microsoft.com/api/v1"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    def query_one(comp_tuple):
        cname, _, _ = comp_tuple
        found = []
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{base}/Advertisers?searchText={urllib.parse.quote(cname)}&top=2", headers=headers), timeout=8) as r1:
                advs = json.loads(r1.read().decode("utf-8"))
            adv_list = advs if isinstance(advs, list) else advs.get("value", [])
            if not adv_list:
                return found
            adv = adv_list[0]
            adv_id = adv.get("AdvertiserId") or adv.get("advertiserId")
            adv_name = adv.get("AdvertiserName") or adv.get("advertiserName") or cname
            if not adv_id:
                return found
            with urllib.request.urlopen(urllib.request.Request(f"{base}/Ads?advertiserId={adv_id}&top=6", headers=headers), timeout=8) as r2:
                ads = json.loads(r2.read().decode("utf-8"))
            for a in (ads if isinstance(ads, list) else ads.get("value", []))[:6]:
                found.append({
                    "Competitor": cname, "Advertiser": adv_name,
                    "Ad Headline": str(a.get("Title") or a.get("title") or "—"),
                    "Active Dates": f"{str(a.get('StartDate') or a.get('startDate') or '')[:10]} → {str(a.get('EndDate') or a.get('endDate') or '')[:10]}",
                    "Ad Library Link": f"https://adlibrary.ads.microsoft.com/advertiser/{adv_id}",
                })
        except Exception:
            pass
        return found

    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(query_one, COMPETITORS_REGISTRY):
            all_rows.extend(res)
    return pd.DataFrame(all_rows)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_bing_webmaster_queries(api_key: str, site_url: str, brand_regex: str, exclude_local_ph: bool, custom_negatives: str, strict_b2b: bool):
    url = f"https://ssl.bing.com/webmaster/api.svc/json/GetQueryStats?siteUrl={urllib.parse.quote(site_url.strip(), safe='')}&apikey={urllib.parse.quote(api_key.strip(), safe='')}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=18) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return pd.DataFrame(), f"Bing Webmaster API Error: {exc}"

    brand_pat = re.compile(brand_regex, re.I) if brand_regex.strip() else None
    custom_pat = re.compile(custom_negatives, re.I) if custom_negatives.strip() else None
    rows = []
    for item in payload.get("d", []):
        m = re.search(r"/Date\((-?\d+)", str(item.get("Date", "")))
        if not m:
            continue
        dt = datetime.fromtimestamp(int(m.group(1)) / 1000.0, tz=timezone.utc).replace(tzinfo=None)
        q_clean = str(item.get("Query") or "").lower().strip()
        if not q_clean or (brand_pat and brand_pat.search(q_clean)) or STRICT_NON_INDUSTRY_REGEX.search(q_clean):
            continue
        if exclude_local_ph and LOCAL_PH_AND_STUDENT_REGEX.search(q_clean):
            continue
        if custom_pat and custom_pat.search(q_clean):
            continue
        if strict_b2b and not is_relevant_to_connext(q_clean, strict=True):
            continue
        raw_pos = float(item.get("AvgImpressionPosition") or 0.0)
        rows.append({"date": dt, "query": q_clean, "impressions": int(item.get("Impressions") or 0), "clicks": int(item.get("Clicks") or 0), "position": round(raw_pos / 10.0, 1) if raw_pos > 0 else 0.0})

    if not rows:
        return pd.DataFrame(), "No non-branded B2B Bing queries found."
    df = pd.DataFrame(rows)
    all_dates = sorted(df["date"].unique())
    recent_dates = all_dates[-2:] if len(all_dates) >= 2 else all_dates
    prior_dates = all_dates[-6:-2] if len(all_dates) >= 6 else all_dates[:-2]
    rec_df = df[df["date"].isin(recent_dates)].copy()
    rec_df["wpos"] = rec_df["position"] * rec_df["impressions"]
    agg = rec_df.groupby("query", as_index=False).agg(bing_impr_recent=("impressions", "sum"), bing_clicks=("clicks", "sum"), wpos=("wpos", "sum"))
    agg["bing_rank"] = (agg["wpos"] / np.maximum(agg["bing_impr_recent"], 1)).round(1)
    if prior_dates:
        pri_df = df[df["date"].isin(prior_dates)].groupby("query", as_index=False).agg(bing_impr_prior=("impressions", "sum"))
        agg = agg.merge(pri_df, on="query", how="left").fillna({"bing_impr_prior": 0})
        scale = len(recent_dates) / max(len(prior_dates), 1)
        agg["growth_pct"] = (((agg["bing_impr_recent"] / np.maximum(agg["bing_impr_prior"] * scale, 1.0)) - 1.0) * 100).round(0).astype(int)
    else:
        agg["growth_pct"] = 0
    agg["vertical"] = agg["query"].apply(classify_sector)
    agg["bing_rank_label"] = agg["bing_rank"].apply(human_rank_label)
    return agg.sort_values(["bing_impr_recent", "growth_pct"], ascending=False).reset_index(drop=True), f"Synced {len(agg):,} B2B Bing queries"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_google_and_bing_suggestions(seed_topic: str, strict_b2b: bool = True) -> pd.DataFrame:
    rows, seen = [], set()
    for q_str in [seed_topic, f"how to {seed_topic}", f"cost of {seed_topic}", f"best {seed_topic}", f"{seed_topic} vs ", f"{seed_topic} pricing"]:
        try:
            url = f"https://suggestqueries.google.com/complete/search?client=firefox&hl=en&gl=us&q={urllib.parse.quote(q_str)}"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(data, list) and len(data) >= 2:
                for s in data[1]:
                    sc = str(s).lower().strip()
                    if sc and sc not in seen and not STRICT_NON_INDUSTRY_REGEX.search(sc) and is_relevant_to_connext(sc, strict=strict_b2b):
                        seen.add(sc)
                        rows.append({"Search Suggestion": sc, "Engine": "Google", "Intent": "Commercial / Pricing" if COMMERCIAL_MODIFIERS.search(sc) else "Question / Guide"})
        except Exception:
            continue
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://api.bing.com/osjson.aspx?query={urllib.parse.quote(seed_topic)}", headers={"User-Agent": "Mozilla/5.0"}), timeout=6) as resp:
            b_data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        if isinstance(b_data, list) and len(b_data) >= 2:
            for s in b_data[1]:
                sc = str(s).lower().strip()
                if sc and sc not in seen and not STRICT_NON_INDUSTRY_REGEX.search(sc) and is_relevant_to_connext(sc, strict=strict_b2b):
                    seen.add(sc)
                    rows.append({"Search Suggestion": sc, "Engine": "Bing", "Intent": "Commercial / Pricing" if COMMERCIAL_MODIFIERS.search(sc) else "Question / Guide"})
    except Exception:
        pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_reddit_b2b_discussions(seed_topic: str) -> pd.DataFrame:
    try:
        url = f"https://www.reddit.com/search.json?q={urllib.parse.quote(f'{seed_topic} outsourcing OR offshore OR agency')}&sort=relevance&t=year&limit=10"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ConnextTrendRadar/3.4"}), timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        rows = []
        for child in (data.get("data") or {}).get("children", []):
            p = child.get("data") or {}
            title = str(p.get("title") or "").strip()
            if title:
                rows.append({
                    "Community": f"r/{p.get('subreddit', '')}", "Buyer Question / Discussion": title,
                    "Comments": int(p.get("num_comments") or 0),
                    "Date": datetime.fromtimestamp(float(p.get("created_utc") or 0), tz=timezone.utc).strftime("%Y-%m-%d"),
                    "Thread": f"https://www.reddit.com{p.get('permalink', '')}",
                })
        return pd.DataFrame(rows).sort_values("Comments", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ==============================================================================
# CLUSTERING, SURGE + CONTENT DECAY ENGINE & INTERNAL LINKING ARCHITECT
# ==============================================================================
@st.cache_data(show_spinner=False)
def cluster_queries(df: pd.DataFrame, distance_threshold: float) -> pd.DataFrame:
    non_brand = df[~df["is_branded"]].copy()
    if non_brand.empty:
        return non_brand
    non_brand["norm_page"] = non_brand["page"].apply(normalize_url)
    non_brand["path"] = non_brand["norm_page"].apply(url_to_path)
    top_queries = non_brand.groupby("query", as_index=False).agg(total_impressions=("impressions", "sum")).sort_values("total_impressions", ascending=False).head(3500).reset_index(drop=True)
    if len(top_queries) < 2:
        non_brand["topic"] = non_brand["query"]
        non_brand["word_count"] = non_brand["query"].str.split().str.len()
        non_brand["is_long_tail"] = non_brand["word_count"] >= 4
        non_brand["is_ai_prompt"] = (non_brand["word_count"] >= 7) | non_brand["query"].str.contains(AI_PROMPT_REGEX, na=False)
        return non_brand

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
    mat = vec.fit_transform(top_queries["query"]).astype(np.float32).toarray()
    valid_mask = np.linalg.norm(mat, axis=1) > 1e-9
    c_ids = np.arange(-1, -len(top_queries) - 1, -1, dtype=int)
    if int(np.sum(valid_mask)) >= 2:
        c_ids[valid_mask] = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance_threshold).fit_predict(mat[valid_mask])
    top_queries["cluster_id"] = c_ids
    topic_names = top_queries.sort_values("total_impressions", ascending=False).groupby("cluster_id")["query"].first().to_dict()
    top_queries["topic"] = top_queries["cluster_id"].map(topic_names)

    merged = non_brand.merge(top_queries[["query", "topic"]], on="query", how="inner")
    merged["word_count"] = merged["query"].str.split().str.len()
    merged["is_long_tail"] = merged["word_count"] >= 4
    merged["is_ai_prompt"] = (merged["word_count"] >= 7) | merged["query"].str.contains(AI_PROMPT_REGEX, na=False)
    return merged


def analyze_trends_and_decay(
    non_brand_df: pd.DataFrame,
    live_set: set[str],
    live_df: pd.DataFrame,
    live_vec: TfidfVectorizer,
    live_mat,
    min_daily_imp: int,
    min_up_days: int,
    min_norm_slope: float,
    min_lift: float,
    strict_b2b: bool = True,
):
    """Uses 100% real GSC clicks, impressions, CTR, and position—zero invented metrics."""
    if non_brand_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    all_dates = sorted(non_brand_df["date"].unique())
    n_days = len(all_dates)
    if n_days < 14:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    today_dates, week_dates = all_dates[-3:], all_dates[-7:]
    prior_14_dates = all_dates[-21:-7] if n_days >= 21 else all_dates[:-7]
    month_dates = all_dates[-min(30, n_days):]
    month_comp = min(21, n_days // 2)
    month_recent, month_prior = all_dates[-month_comp:], (all_dates[-(month_comp * 2):-month_comp] if n_days >= month_comp * 2 else all_dates[:-month_comp])

    df = non_brand_df.copy()
    if "clicks" not in df.columns:
        df["clicks"] = 0
    df["weighted_pos"] = df["position"] * df["impressions"]
    daily_topic = df.groupby(["topic", "date"], as_index=False).agg(
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        weighted_pos=("weighted_pos", "sum"),
    )

    recent_df, prior_df, month_df = df[df["date"].isin(week_dates)], df[df["date"].isin(prior_14_dates)], df[df["date"].isin(month_dates)]
    top_q_map = recent_df.groupby(["topic", "query"], as_index=False)["impressions"].sum().sort_values("impressions", ascending=False).groupby("topic")["query"].apply(list).to_dict()
    lt_q_map = recent_df[recent_df["is_long_tail"]].groupby(["topic", "query"], as_index=False)["impressions"].sum().sort_values("impressions", ascending=False).groupby("topic")["query"].apply(list).to_dict()

    topic_q_imp_map = {}
    for t_name, grp in recent_df.groupby(["topic", "query"], as_index=False)["impressions"].sum().sort_values("impressions", ascending=False).groupby("topic"):
        topic_q_imp_map[t_name] = list(zip(grp["query"].tolist(), grp["impressions"].tolist()))

    pages_agg = recent_df.groupby(["topic", "norm_page", "path"], as_index=False).agg(
        page_clicks_7d=("clicks", "sum"),
        page_imp_7d=("impressions", "sum"),
        page_weighted_pos=("weighted_pos", "sum"),
        queries_on_page=("query", "nunique"),
    ).sort_values(["topic", "page_imp_7d"], ascending=[True, False])
    pages_agg["page_ctr_%"] = ((pages_agg["page_clicks_7d"] / np.maximum(pages_agg["page_imp_7d"], 1)) * 100).round(2)
    pages_agg["page_org_pos"] = (pages_agg["page_weighted_pos"] / np.maximum(pages_agg["page_imp_7d"], 1)).round(1)
    pages_agg["page_role"] = pages_agg["path"].apply(lambda p: classify_page_role(p, live_set))
    pages_agg["is_live_url"] = pages_agg["path"].isin(live_set)
    pages_agg["is_money_url"] = pages_agg["page_role"].isin(["Industry Page", "Team Solution Page", "Service Model Page", "Location Hub"])

    prior_pages_agg = prior_df.groupby(["topic", "norm_page"], as_index=False)["impressions"].sum().sort_values(["topic", "impressions"], ascending=[True, False])
    prior_page_map = prior_pages_agg.groupby("topic")["norm_page"].first().to_dict()

    top_page_map = pages_agg.groupby("topic")["norm_page"].first().to_dict()
    top_page_role_map = pages_agg.groupby("topic")["page_role"].first().to_dict()
    has_money_map = pages_agg.groupby("topic")["is_money_url"].any().to_dict()

    live_gsc_urls_by_topic = (
        pages_agg[pages_agg["is_live_url"]].groupby("topic")["norm_page"].apply(list).to_dict()
    )

    topic_7d_totals = pages_agg.groupby("topic")["page_imp_7d"].transform("sum").replace(0, 1)
    pages_agg["share_ratio"] = pages_agg["page_imp_7d"] / topic_7d_totals
    significant_page_cnt_map = pages_agg[pages_agg["share_ratio"] >= 0.15].groupby("topic")["norm_page"].nunique().to_dict()
    page_cnt_map = pages_agg.groupby("topic")["norm_page"].nunique().to_dict()

    surge_results, decay_results = [], []

    for topic, group in daily_topic.groupby("topic"):
        if strict_b2b and not is_relevant_to_connext(topic, strict=True):
            continue

        series = group.set_index("date")["impressions"].reindex(all_dates, fill_value=0)
        arr_today, arr_week, arr_prior14 = series.loc[today_dates].values, series.loc[week_dates].values, series.loc[prior_14_dates].values
        arr_m_rec, arr_m_base = series.loc[month_recent].values, series.loc[month_prior].values

        m_today, m_week, m_prior = float(np.mean(arr_today)), float(np.mean(arr_week)), max(float(np.mean(arr_prior14)), 1.0)
        m_mrec, m_mbase = float(np.mean(arr_m_rec)), max(float(np.mean(arr_m_base)), 1.0)

        rec_grp = group[group["date"].isin(week_dates)]
        pri_grp = group[group["date"].isin(prior_14_dates)]
        mon_grp = group[group["date"].isin(month_dates)]

        imp_7d = int(rec_grp["impressions"].sum())
        clicks_7d = int(rec_grp["clicks"].sum())
        imp_30d = int(mon_grp["impressions"].sum())
        clicks_30d = int(mon_grp["clicks"].sum())
        real_ctr_30d = round((clicks_30d / max(imp_30d, 1)) * 100, 2)

        avg_pos_now = float(rec_grp["weighted_pos"].sum() / max(imp_7d, 1)) if imp_7d > 0 else 0.0
        avg_pos_prior = float(pri_grp["weighted_pos"].sum() / max(int(pri_grp["impressions"].sum()), 1)) if not pri_grp.empty else avg_pos_now
        rank_slip = round(avg_pos_now - avg_pos_prior, 1)

        drop_pct_7d = round(((m_week / m_prior) - 1.0) * 100, 1)
        if m_prior >= max(min_daily_imp, 15) and (drop_pct_7d <= -22.0 or (rank_slip >= 2.5 and avg_pos_prior <= 15.0)):
            gsc_url_d = top_page_map.get(topic) or prior_page_map.get(topic, "")
            decay_results.append({
                "Topic": topic,
                "Vertical": classify_sector(f"{topic} {gsc_url_d}"),
                "Page Type": classify_page_role(url_to_path(gsc_url_d), live_set) if gsc_url_d else "Unknown",
                "7d Impr.": imp_7d,
                "7d Clicks": clicks_7d,
                "Impr. Change": f"{drop_pct_7d:.0f}%",
                "Prior Rank": f"#{avg_pos_prior:.1f}",
                "Current Rank": f"#{avg_pos_now:.1f}" if avg_pos_now > 0 else "Dropped",
                "Rank Change": f"{rank_slip:+.1f}" if avg_pos_now > 0 else "Lost",
                "Affected Page": gsc_url_d,
                "Recommended Refresh": "Update H1/H2s, refresh stats & add internal links" if avg_pos_prior <= 10.0 else "Expand page depth & add supporting blog links",
            })

        if max(m_today, m_week, m_mrec) < min_daily_imp:
            continue
        lift_today = m_today / m_prior
        is_today = bool(m_today >= min_daily_imp and lift_today >= (min_lift + 0.15) and arr_today[-1] > arr_today[0])
        up_days = int(np.sum(np.diff(arr_week) > 0))
        norm_slope_w = float(np.polyfit(np.arange(len(arr_week)), arr_week, 1)[0]) / max(m_week, 1.0)
        is_week = bool(m_today >= min_daily_imp and up_days >= min_up_days and norm_slope_w >= min_norm_slope and lift_today >= min_lift)
        lift_month = m_mrec / m_mbase
        norm_slope_m = float(np.polyfit(np.arange(len(arr_m_rec)), arr_m_rec, 1)[0]) / max(m_mrec, 1.0)
        is_month = bool(m_mrec >= (min_daily_imp * 0.8) and lift_month >= max(1.25, min_lift - 0.15) and norm_slope_m >= (min_norm_slope * 0.35))
        if not (is_today or is_week or is_month):
            continue

        q_list = top_q_map.get(topic, [topic])
        lt_list = lt_q_map.get(topic, [])
        joined_q = ", ".join(q_list[:6])
        q_imp_pairs = topic_q_imp_map.get(topic, [(topic, imp_7d)])
        funnel, funnel_reason = classify_cluster_funnel(topic, q_imp_pairs, top_page_role_map.get(topic, ""))
        has_comm = bool(funnel == "BOFU — Ready to Buy")

        u_pages = int(page_cnt_map.get(topic, 1))
        sig_pages = int(significant_page_cnt_map.get(topic, 1))
        gsc_url = top_page_map.get(topic) or prior_page_map.get(topic, "")
        gsc_role = classify_page_role(url_to_path(gsc_url), live_set) if gsc_url else "Not Ranking"
        sector = classify_sector(f"{joined_q} {gsc_url}")
        site_chk = check_topic_against_sitemap(
            topic, joined_q, sector, live_set, live_df, live_vec, live_mat,
            gsc_live_urls=live_gsc_urls_by_topic.get(topic, []),
        )
        # ALWAYS a verified commercial Service/Industry/Team/Location page (NEVER a blog!)
        rec_lp = site_chk["best_money_url"]

        if has_money_map.get(topic, False):
            cov_status = "Service Page Ranking"
        elif site_chk["already_exists_in_sitemap"]:
            cov_status = f"Covered ({site_chk['closest_existing_role']})"
        else:
            cov_status = "New Topic Gap"

        if has_comm and avg_pos_now > 5.0:
            action = "Launch Paid Ads"
        elif site_chk["already_exists_in_sitemap"] or (4.0 <= avg_pos_now <= 15.0):
            action = "Update Existing Page"
        elif not site_chk["already_exists_in_sitemap"]:
            action = "Write New Page"
        else:
            action = "Defend Top 3 Rank"

        prio_score = min(100, int(
            (30 if funnel.startswith("BOFU") else (20 if funnel.startswith("MOFU") else 10))
            + min(30, int(np.log1p(imp_7d) * 4.5))
            + min(25, int(max(lift_today - 1.0, 0) * 18))
            + (15 if 4.0 <= avg_pos_now <= 15.0 else (10 if avg_pos_now > 15.0 else 5))
        ))

        surge_results.append({
            "topic": topic, "sector": sector, "funnel": funnel, "funnel_reason": funnel_reason,
            "priority_score": prio_score,
            "is_today": is_today, "is_week": is_week, "is_month": is_month,
            "is_striking_distance": bool(4.0 <= avg_pos_now <= 15.0),
            "is_cannibalized": bool(sig_pages >= 2),
            "team_action": action,
            "growth_pct_3d": round((lift_today - 1.0) * 100),
            "growth_pct_week": round(((m_week / m_prior) - 1.0) * 100),
            "month_growth_pct": round((lift_month - 1.0) * 100),
            "impressions_7d": imp_7d, "clicks_7d": clicks_7d,
            "impressions_30d": imp_30d, "clicks_30d": clicks_30d,
            "real_ctr_30d_%": real_ctr_30d,
            "sparkline_7d": [int(v) for v in arr_week],
            "connext_org_pos": round(avg_pos_now, 1), "google_rank_label": human_rank_label(avg_pos_now),
            "commercial_intent": has_comm, "coverage_status": cov_status,
            "already_exists_in_sitemap": site_chk["already_exists_in_sitemap"],
            "closest_existing_url": site_chk["closest_existing_url"],
            "closest_existing_role": site_chk["closest_existing_role"],
            "closest_overlap_pct": site_chk["closest_overlap_pct"],
            "existing_sitemap_matches": site_chk["existing_sitemap_matches"],
            "ranking_pages_count": u_pages,
            "primary_page": gsc_url, "primary_page_role": gsc_role,
            "recommended_money_page": rec_lp, "best_money_role": site_chk["best_money_role"],
            "matching_money_pages": site_chk["top_money_pages"], "matching_content_pages": site_chk["top_content_pages"],
            "queries_list": q_list, "long_tail_list": lt_list,
            "total_queries": len(q_list), "long_tail_count": len(lt_list), "top_rising_queries": joined_q,
        })

    decay_df = pd.DataFrame(decay_results).sort_values("7d Impr.", ascending=False).reset_index(drop=True) if decay_results else pd.DataFrame()
    if not surge_results:
        return pd.DataFrame(), decay_df, daily_topic, pd.DataFrame(), pd.DataFrame()

    trends_df = pd.DataFrame(surge_results).sort_values(["priority_score", "impressions_7d"], ascending=False).reset_index(drop=True)

    q_recent = recent_df.groupby(["topic", "query", "is_long_tail", "is_ai_prompt"], as_index=False).agg(
        clicks_7d=("clicks", "sum"), imp_7d=("impressions", "sum"), weighted_pos=("weighted_pos", "sum"), top_page=("norm_page", "first")
    )
    q_mon_agg = month_df.groupby("query", as_index=False).agg(clicks_30d=("clicks", "sum"), imp_30d=("impressions", "sum"))
    q_merged = q_recent.merge(q_mon_agg, on="query", how="left").merge(prior_df.groupby("query", as_index=False).agg(imp_base14=("impressions", "sum")), on="query", how="left").fillna({"clicks_30d": 0, "imp_30d": 0, "imp_base14": 0})
    q_merged["clicks_30d"] = q_merged["clicks_30d"].astype(int)
    q_merged["imp_30d"] = q_merged["imp_30d"].astype(int)
    q_merged["real_ctr_%"] = ((q_merged["clicks_30d"] / np.maximum(q_merged["imp_30d"], 1)) * 100).round(2)
    q_merged["daily_7d"] = (q_merged["imp_7d"] / 7.0).round(1)
    q_merged["query_growth_pct"] = (((q_merged["daily_7d"] / np.maximum(q_merged["imp_base14"] / max(len(prior_14_dates), 1), 0.5)) - 1.0) * 100).round(0).astype(int)
    q_merged["connext_org_pos"] = (q_merged["weighted_pos"] / np.maximum(q_merged["imp_7d"], 1)).round(1)
    q_merged["sector"] = q_merged.apply(lambda r: classify_sector(f"{r['query']} {r['top_page']}"), axis=1)
    q_merged["is_commercial"] = q_merged["query"].apply(lambda q: classify_single_query_funnel(q)[0] == "BOFU — Ready to Buy")
    q_merged = q_merged.sort_values(["imp_7d", "imp_30d"], ascending=False)

    pages_merged = pages_agg[pages_agg["topic"].isin(trends_df["topic"])].copy()
    tot_map = trends_df.set_index("topic")["impressions_7d"].to_dict()
    pages_merged["topic_imp_share_%"] = ((pages_merged["page_imp_7d"] / pages_merged["topic"].map(tot_map).replace(0, 1)) * 100).round(1)
    return trends_df, decay_df, daily_topic, q_merged, pages_merged


def build_internal_linking_plan(trends_df: pd.DataFrame, all_pages_gsc_df: pd.DataFrame) -> pd.DataFrame:
    """Strictly maps live Editorial Blogs (both /blog/ and root-level) -> verified Commercial Service Pages."""
    page_imp_map = all_pages_gsc_df.groupby("norm_page")["impressions"].sum().to_dict() if not all_pages_gsc_df.empty else {}
    links = []
    for _, r in trends_df.iterrows():
        target_money = r["recommended_money_page"]
        if not target_money or r["connext_org_pos"] <= 3.0:
            continue
        candidate_blogs = list(r["matching_content_pages"][:3])
        if "Blog" in r["primary_page_role"] and r["primary_page"]:
            candidate_blogs.insert(0, {"full_url": r["primary_page"], "role": r["primary_page_role"]})
        for c_page in candidate_blogs:
            src_url = c_page["full_url"]
            if src_url == target_money:
                continue
            links.append({
                "Source Blog Post (Edit This Post)": src_url,
                "Source Type": c_page["role"],
                "Source GSC Impressions": int(page_imp_map.get(src_url, 0)),
                "Anchor Text to Add": r["topic"],
                "Target Service / Industry Page": target_money,
                "Target Page Type": r["best_money_role"],
                "Current Topic Rank": r["google_rank_label"],
                "Vertical": r["sector"],
            })
    if not links:
        return pd.DataFrame()
    return pd.DataFrame(links).drop_duplicates(subset=["Source Blog Post (Edit This Post)", "Target Service / Industry Page"]).sort_values("Source GSC Impressions", ascending=False).reset_index(drop=True)


def build_real_ads_tables(trends_df: pd.DataFrame, queries_df: pd.DataFrame, kp_lookup: dict):
    """Builds Google/Bing Ads planning tables using 100% real GSC data + real Keyword Planner CSV (when uploaded)."""
    ads_topics, rec_kws = [], []
    lp_map = trends_df.set_index("topic")["recommended_money_page"].to_dict()
    prio_rank = {"P0 — Launch Today": 0, "P1 — Search Ad Group": 1, "P2 — Defend Top 5": 2, "P2 — Long-Tail": 3}

    for _, r in trends_df.iterrows():
        pos, comm = r["connext_org_pos"], r["commercial_intent"]
        prio = "P0 — Launch Today" if (r["is_today"] and comm and pos > 5.0) else ("P1 — Search Ad Group" if (comm and pos > 5.0) else ("P2 — Defend Top 5" if comm else "P2 — Long-Tail"))
        ads_topics.append({
            "_prio_sort": prio_rank[prio],
            "Priority": prio,
            "Topic": r["topic"],
            "Funnel": r["funnel"],
            "Vertical": r["sector"],
            "Google Rank": r["google_rank_label"],
            "7d Impr. (GSC)": r["impressions_7d"],
            "30d Impr. (GSC)": r["impressions_30d"],
            "30d Clicks (GSC)": r["clicks_30d"],
            "Real CTR (%)": r["real_ctr_30d_%"],
            "Target Service Page": r["recommended_money_page"],
        })

    for _, q in queries_df.iterrows():
        kw, is_lt, is_comm = q["query"], q["is_long_tail"], q["is_commercial"]
        kp = kp_lookup.get(kw.lower(), {})
        kp_vol = kp.get("kp_volume", 0)
        kp_low = kp.get("kp_low_bid", 0.0)
        kp_high = kp.get("kp_high_bid", 0.0)
        kp_comp = kp.get("kp_competition", "—")
        mtype = '"Phrase" + [Exact]' if (is_comm and not is_lt) else ("[Exact Match]" if is_lt else '"Phrase Match"')

        row_dict = {
            "Keyword": kw,
            "7d Impr. (GSC)": int(q["imp_7d"]),
            "30d Impr. (GSC)": int(q["imp_30d"]),
            "30d Clicks (GSC)": int(q["clicks_30d"]),
            "Real CTR (%)": float(q["real_ctr_%"]),
            "Growth": f"+{int(q['query_growth_pct'])}%" if q["query_growth_pct"] >= 0 else f"{int(q['query_growth_pct'])}%",
            "Google Rank": human_rank_label(q["connext_org_pos"]),
            "Match Type": mtype,
            "Topic": q["topic"],
            "Target Service Page": lp_map.get(q["topic"], "https://connextglobal.com/our-service-model"),
        }
        if kp_lookup:
            row_dict["KP Monthly Searches"] = kp_vol if kp_vol > 0 else "—"
            row_dict["KP Top Bid Range (Real)"] = f"${kp_low:.2f} – ${kp_high:.2f}" if kp_high > 0 else "—"
            row_dict["KP Competition"] = kp_comp
        rec_kws.append(row_dict)

    ads_df = pd.DataFrame(ads_topics).sort_values(["_prio_sort", "30d Impr. (GSC)"], ascending=[True, False]).drop(columns=["_prio_sort"]).reset_index(drop=True)
    kws_df = pd.DataFrame(rec_kws).sort_values(["7d Impr. (GSC)", "30d Impr. (GSC)"], ascending=False).reset_index(drop=True) if rec_kws else pd.DataFrame()
    return ads_df, kws_df


def build_content_tables(trends_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleanly splits surging topics into three unambiguous buckets:
      1. new_ideas: Topics NOT covered anywhere in connextglobal.com/sitemap.xml -> Write a brand-new blog post.
      2. blog_updates: Topics where an existing Blog (/blog/ or root-level blog) is already live -> Refresh & link to Service Page.
      3. service_updates: Topics where a commercial Service/Industry/Team page is already ranking -> Optimize on-page & FAQ schema.
    """
    new_ideas, blog_updates, service_updates = [], [], []
    for _, r in trends_df.iterrows():
        if not r["already_exists_in_sitemap"] and r["coverage_status"] == "New Topic Gap":
            slug_base = re.sub(r"[^a-z0-9]+", "-", r["topic"].lower()).strip("-")
            new_ideas.append({
                "Priority": "P1 — High Demand Gap" if r["impressions_7d"] >= 100 else "P2 — Emerging Topic",
                "New Blog Topic to Write": r["topic"],
                "Vertical": r["sector"],
                "Funnel Stage": r["funnel"],
                "7d Impr.": r["impressions_7d"],
                "30d Impr.": r["impressions_30d"],
                "Suggested Blog URL": f"/blog/{slug_base}-guide",
                "Service Page to Link To": r["recommended_money_page"],
                "Subheadings / Queries to Cover": r["top_rising_queries"],
            })
        elif "Blog" in r["closest_existing_role"] or r["closest_existing_role"] in {"Case Study", "White Paper", "One-Pager", "E-Book"}:
            prio = "P1 — Quick Win (Rank #4–#15)" if r["is_striking_distance"] else "P2 — Keep Fresh"
            blog_updates.append({
                "Priority": prio,
                "Topic": r["topic"],
                "Vertical": r["sector"],
                "7d Impr.": r["impressions_7d"],
                "Google Rank": r["google_rank_label"],
                "Blog Type": r["closest_existing_role"],
                "Existing Blog Post to Refresh": r["closest_existing_url"],
                "Service Page to Link To": r["recommended_money_page"],
                "How to Improve This Blog": f"Add H2s for ({r['top_rising_queries'][:55]}...), add FAQ schema & link to Service Page.",
            })
        else:
            prio = "P1 — Quick Win (Rank #4–#15)" if r["is_striking_distance"] else "P2 — Defend Rank"
            service_updates.append({
                "Priority": prio,
                "Topic": r["topic"],
                "Vertical": r["sector"],
                "7d Impr.": r["impressions_7d"],
                "Google Rank": r["google_rank_label"],
                "Page Type": r["closest_existing_role"],
                "Service / Industry Page": r["closest_existing_url"],
                "Optimization Note": f"Add FAQPage JSON-LD schema & answer rising buyer queries ({r['top_rising_queries'][:55]}...).",
            })

    new_df = pd.DataFrame(new_ideas).sort_values("7d Impr.", ascending=False).reset_index(drop=True) if new_ideas else pd.DataFrame()
    blog_df = pd.DataFrame(blog_updates).sort_values("7d Impr.", ascending=False).reset_index(drop=True) if blog_updates else pd.DataFrame()
    svc_df = pd.DataFrame(service_updates).sort_values("7d Impr.", ascending=False).reset_index(drop=True) if service_updates else pd.DataFrame()
    return new_df, blog_df, svc_df


# ==============================================================================
# LOAD SITEMAP & STREAMLINED SIDEBAR
# ==============================================================================
sitemap_paths, sitemap_source_label = fetch_live_sitemap_paths()
live_set, live_df, live_vec, live_mat = _build_index_from_paths(sitemap_paths)

st.sidebar.image(CNXT_LOGO_WHITE, width=175)
st.sidebar.markdown("**Enterprise Search Radar**")
st.sidebar.caption(f"100% Real Data • {sitemap_source_label}")
if st.sidebar.button("Lock / Log Out"):
    st.session_state["cnxt_authenticated"] = False
    st.rerun()
st.sidebar.divider()

has_secrets = "gcp_service_account" in st.secrets and "gsc_config" in st.secrets
data_source = st.sidebar.radio(
    "Search Console Source",
    ["Live GSC API", "Upload GSC CSV / ZIP"],
    index=0,
    help="Keep 'Live GSC API' selected for automatic live data. Initial sync takes 30–60 seconds.",
)
strict_b2b_filter = st.sidebar.checkbox(
    "Strict Connext B2B Relevance Filter",
    value=True,
    help="Blocks unrelated queries (holidays, weather, translations, personal tax/loans, generic tutorials) so only Connext B2B topics appear. Updates instantly.",
)
exclude_local_ph = st.sidebar.checkbox(
    "Filter Out Local PH / Student Noise",
    value=True,
    help="Removes local PH city directory and student/university searches so only B2B buyer demand appears.",
)
exclude_glossary = st.sidebar.checkbox(
    "Hide Glossary / 'Meaning' Queries",
    value=False,
    help="Turn on to hide definition/meaning lookups (like 'rebadging meaning') and focus strictly on MOFU/BOFU buyer topics. Updates instantly.",
)
custom_negatives = st.sidebar.text_input(
    "Extra Excluded Words",
    placeholder="intern|sample|pdf",
    help="Optional: separate extra words to hide using a vertical pipe | (job-seeker words are already removed automatically).",
)

base_raw_df, harvested_neg_df = pd.DataFrame(), pd.DataFrame()
if data_source == "Live GSC API":
    if has_secrets:
        site_url = st.secrets["gsc_config"].get("site_url", DEFAULT_SITE_URL)
        brand_regex = st.secrets["gsc_config"].get("brand_regex", DEFAULT_BRAND_REGEX)
        lookback = st.sidebar.slider("GSC History Window (Days)", 28, 90, 45, help="45 days is the fastest balance between short-term spikes and monthly trends.")
        if st.sidebar.button("Refresh Live GSC Data", type="primary"):
            fetch_gsc_api_data.clear()
            fetch_live_sitemap_paths.clear()
        with st.spinner("Loading live Google Search Console data & checking connextglobal.com/sitemap.xml... This normally takes 30 to 60 seconds. Please keep this tab open."):
            try:
                base_raw_df, harvested_neg_df = fetch_gsc_api_data(dict(st.secrets["gcp_service_account"]), site_url, lookback, brand_regex, custom_negatives, exclude_local_ph)
            except Exception as err:
                st.error(f"GSC API Error: {err}")
                st.stop()
    else:
        uploaded_json = st.sidebar.file_uploader("Service Account JSON", type=["json"])
        site_url = st.sidebar.text_input("GSC Property", value=DEFAULT_SITE_URL)
        brand_regex = DEFAULT_BRAND_REGEX
        lookback = st.sidebar.slider("GSC History Window (Days)", 28, 90, 45)
        if uploaded_json:
            with st.spinner("Loading live Google Search Console data (30–60 seconds)..."):
                base_raw_df, harvested_neg_df = fetch_gsc_api_data(json.load(uploaded_json), site_url, lookback, brand_regex, custom_negatives, exclude_local_ph)
else:
    uploaded_csv = st.sidebar.file_uploader("Upload GSC Web CSV / ZIP", type=["csv", "zip"])
    brand_regex = DEFAULT_BRAND_REGEX
    if uploaded_csv:
        try:
            base_raw_df, harvested_neg_df = parse_gsc_web_upload(uploaded_csv, brand_regex, custom_negatives, exclude_local_ph)
        except Exception as err:
            st.sidebar.error(str(err))

# Apply fast in-memory relevance & glossary filters (instant toggle without re-fetching GSC API!)
raw_df = base_raw_df.copy()
if not raw_df.empty:
    if strict_b2b_filter:
        raw_df = raw_df[raw_df["is_branded"] | raw_df["query"].apply(lambda q: is_relevant_to_connext(q, strict=True))].copy()
    if exclude_glossary:
        raw_df = raw_df[~raw_df["query"].str.contains(TOFU_DEF_RE, regex=True, na=False)].copy()

st.sidebar.divider()
preset_mode = st.sidebar.selectbox(
    "Trend Sensitivity",
    ["Broad (All Verticals)", "Balanced", "Strict Breakouts"],
    index=0,
    help="'Broad' shows rising demand across all Connext verticals. 'Strict Breakouts' shows only major volume spikes.",
)
def_imp, def_up, def_lift, def_slope = (50, 4, 1.40, 10) if preset_mode.startswith("Strict") else ((30, 4, 1.30, 8) if preset_mode.startswith("Balanced") else (15, 3, 1.18, 4))

with st.sidebar.expander("Trend Threshold Settings", expanded=False):
    min_daily_imp = st.number_input("Min Avg Daily Impr.", 5, 5000, def_imp, 5)
    min_up_days = st.slider("Weekly Upward Days", 2, 6, def_up)
    min_lift = st.slider("Min Growth Multiplier", 1.05, 3.0, def_lift, 0.05)
    min_slope_pct = st.slider("Min Daily Slope (%)", 1, 40, def_slope, 1)
    cluster_dist = st.slider("Topic Grouping Breadth", 0.30, 0.75, 0.55, 0.05)

has_bing_secret = "bing" in st.secrets and "api_key" in st.secrets["bing"] and bool(str(st.secrets["bing"]["api_key"]).strip())
has_serpapi_secret = "serpapi" in st.secrets and "api_key" in st.secrets["serpapi"] and bool(str(st.secrets["serpapi"]["api_key"]).strip())

# ==============================================================================
# HEADER, HERO BANNER & TOP QUICK-START GUIDE
# ==============================================================================
h_col1, h_col2 = st.columns([1.1, 4])
with h_col1:
    st.image(CNXT_LOGO_COLOR, width=180)
with h_col2:
    if not raw_df.empty:
        st.caption(f"Live GSC Window: **{raw_df['date'].min().strftime('%b %d')} – {raw_df['date'].max().strftime('%b %d, %Y')}** ({raw_df['date'].nunique()} complete days)  •  {sitemap_source_label}  •  100% Real Verified Data")

st.markdown(
    """<div class="cnxt-hero">
<div class="cnxt-eyebrow">CONNEXT GLOBAL ENTERPRISE SEARCH &amp; TOPIC INTELLIGENCE</div>
<h1>Search, Content &amp; Competitor <span>Trend Radar</span></h1>
<p>100% verified Google Search Console Impressions, Clicks, CTR, and Rank—strictly filtered to Connext B2B topics, paired with Content Decay alerts, New Content &amp; Internal Linking plans, Competitor US Ads Transparency, and Competitor Blogs.</p>
</div>""",
    unsafe_allow_html=True,
)

with st.expander("New here? 1-Minute Quick Start (How to Use This Tool & What to Expect)", expanded=False):
    qs1, qs2, qs3 = st.columns(3)
    with qs1:
        st.markdown(
            """**1. Start With the Filter Bar Below**
- Pick a **Business Vertical** (e.g., *Healthcare & RCM*, *Finance & Accounting*) or leave on **All Verticals**.
- Pick a **Funnel Stage** (*BOFU — Ready to Buy*, *MOFU — Evaluating*, or *TOFU — Awareness*).
- Every chart and table across Tabs 1–4 updates instantly."""
        )
    with qs2:
        st.markdown(
            """**2. Pick the Tab for Your Role**
- **Leadership / Managers:** *Tab 1 (Executive Overview)* → copy the Teams Summary or download the Excel report.
- **Content / SEO Team:** *Tab 2 (Decay & On-Page Audit)* + *Tab 4 (New Content & Linking)* + *Tab 6 (Competitor Blogs)*.
- **Paid Ads Team:** *Tab 3 (Google & Bing Ads Builder)* + *Tab 5 (Competitor US Ads Transparency)*."""
        )
    with qs3:
        st.markdown(
            """**3. What to Expect (Speed & Real Data)**
- Initial load takes **30–60 seconds** (live Google Search Console sync, cached for 1 hour).
- **Zero guessed numbers:** All impressions, clicks, CTRs, and ranks are real GSC metrics. Keyword Planner bids appear only when you upload a real Keyword Planner CSV.
- **Tabs 5 & 6** run external scans **only when you click their button** (~10–20s)."""
        )

if raw_df.empty:
    st.info("Connect your Google Search Console API or upload a GSC CSV in the sidebar to begin.")
    st.stop()

with st.spinner("Grouping B2B queries into topics, classifying buyer funnel stages & mapping verified Service Pages..."):
    clustered_df = cluster_queries(raw_df, cluster_dist)
    trends_df, decay_df, daily_topic_df, all_queries_df, pages_df = analyze_trends_and_decay(
        clustered_df, live_set, live_df, live_vec, live_mat,
        min_daily_imp, min_up_days, min_slope_pct / 100.0, min_lift,
        strict_b2b=strict_b2b_filter,
    )

if trends_df.empty:
    st.warning("No surging B2B topics matched your sensitivity filter. Select **Broad (All Verticals)** in the sidebar.")
    st.stop()

# ==============================================================================
# CLEAN GLOBAL FILTER BAR
# ==============================================================================
v_counts = trends_df["sector"].value_counts().to_dict()
v_opts = ["All Verticals"] + [f"{v} ({v_counts.get(v, 0)})" for v in ALL_CONNEXT_VERTICALS if v_counts.get(v, 0) > 0]
a_counts = trends_df["team_action"].value_counts().to_dict()
a_opts = [
    "All Actions",
    f"Launch Paid Ads ({a_counts.get('Launch Paid Ads', 0)})",
    f"Update Existing Page ({a_counts.get('Update Existing Page', 0)})",
    f"Write New Page ({a_counts.get('Write New Page', 0)})",
    f"Striking Distance Quick Wins #4–#15 ({int(trends_df['is_striking_distance'].sum())})",
]

fc1, fc2, fc3, fc4 = st.columns([1.4, 1.4, 1.0, 1.2])
sel_vert = fc1.selectbox("Business Vertical", v_opts, index=0)
sel_act = fc2.selectbox("Recommended Next Step", a_opts, index=0)
sel_funnel = fc3.selectbox("Funnel Stage", ["All Funnel Stages", "BOFU — Ready to Buy", "MOFU — Evaluating", "TOFU — Awareness"])
search_box = fc4.text_input("Filter by Keyword or URL", placeholder="e.g., medical billing, cpa...")

f_trends = trends_df.copy()
if sel_vert != "All Verticals":
    f_trends = f_trends[f_trends["sector"] == sel_vert.rsplit(" (", 1)[0]]
if sel_act != "All Actions":
    f_trends = f_trends[f_trends["is_striking_distance"]] if sel_act.startswith("Striking Distance") else f_trends[f_trends["team_action"] == sel_act.rsplit(" (", 1)[0]]
if sel_funnel != "All Funnel Stages":
    f_trends = f_trends[f_trends["funnel"] == sel_funnel]
if search_box.strip():
    pat = re.escape(search_box.strip())
    f_trends = f_trends[f_trends["topic"].str.contains(pat, case=False, na=False) | f_trends["top_rising_queries"].str.contains(pat, case=False, na=False) | f_trends["primary_page"].str.contains(pat, case=False, na=False)]
f_trends = f_trends.reset_index(drop=True)

if f_trends.empty:
    st.info("No topics match those exact filters right now. Reset **Business Vertical**, **Recommended Next Step**, or **Funnel Stage** above to see all active trends.")
    st.stop()

f_queries = all_queries_df[all_queries_df["topic"].isin(f_trends["topic"])].copy()
f_pages = pages_df[pages_df["topic"].isin(f_trends["topic"])].copy()
f_decay = decay_df[decay_df["Vertical"] == sel_vert.rsplit(" (", 1)[0]].copy() if (sel_vert != "All Verticals" and not decay_df.empty) else decay_df.copy()

kp_lookup_dict = st.session_state.get("kp_lookup_dict", {})
ads_topics_df, rec_kws_df = build_real_ads_tables(f_trends, f_queries, kp_lookup_dict)
truly_new_df, existing_blogs_df, existing_services_df = build_content_tables(f_trends)
internal_links_df = build_internal_linking_plan(f_trends, clustered_df)

# --- TOP KPI ROW (100% REAL GSC DATA) ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Surging B2B Topics", len(f_trends), help="Connext-relevant B2B topic groups gaining search visibility on Google right now.")
k2.metric("Content Decay Alerts", len(f_decay), help="Topics where real 7d GSC impressions dropped >= 22% or Google rank slipped >= 2.5 spots.")
k3.metric("7d Impressions (Real GSC)", f"{int(f_trends['impressions_7d'].sum()):,}", help="Actual Google Search impressions for these surging topics in the last 7 days.")
k4.metric("7d Clicks (Real GSC)", f"{int(f_trends['clicks_7d'].sum()):,}", help="Actual organic clicks from Google Search in the last 7 days.")
k5.metric("Avg Google Rank (Real GSC)", f"#{f_trends['connext_org_pos'].mean():.1f}", help="Impression-weighted average Google organic position across surging topics.")

# --- 7 ENTERPRISE TABS ---
t_over, t_pages, t_ads, t_content, t_ads_trans, t_comp_blogs, t_guide = st.tabs([
    "1. Executive Overview",
    f"2. Pages, Decay & Audit ({len(f_decay)} Decay)",
    f"3. Google & Bing Ads Builder ({len(rec_kws_df)})",
    f"4. New Content & Linking ({len(truly_new_df)} New)",
    "5. Competitor US Ads Transparency (12)",
    "6. Competitor Blogs & Trends ($0)",
    "7. User Guide & Glossary",
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
with t_over:
    render_tab_guide(
        "Executive Overview",
        "Shows Connext-relevant B2B topics growing fastest on Google using 100% real GSC Impressions, Clicks, CTR, and Rank, plus each cluster's Top Keywords, TOFU/MOFU/BOFU funnel stage, Currently Ranking URL, and verified Target Service Page.",
        "Ranked bar charts, a 0–100 Priority Score per topic, daily trend curves, and a multi-sheet Excel report (.xlsx).",
        "Start weekly team meetings here: review the top surging topics, copy the 1-Click Teams Executive Summary at the bottom, or export the Excel workbook.",
    )

    top_bar1, top_bar2 = st.columns([2.5, 1])
    with top_bar1:
        horizon = st.radio("Time Horizon:", ["All Surging", "Today (3-Day Spike)", "This Week (7-Day)", "This Month (30-Day)"], horizontal=True)
    with top_bar2:
        exec_summary_df = pd.DataFrame([
            {"Metric": "Report Date (UTC)", "Value": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
            {"Metric": "Surging B2B Topics", "Value": len(f_trends)},
            {"Metric": "Content Decay Alerts", "Value": len(f_decay)},
            {"Metric": "Real 7d Impressions (GSC)", "Value": int(f_trends["impressions_7d"].sum())},
            {"Metric": "Real 7d Clicks (GSC)", "Value": int(f_trends["clicks_7d"].sum())},
            {"Metric": "Avg Google Rank (GSC)", "Value": round(float(f_trends["connext_org_pos"].mean()), 1)},
        ])
        excel_bytes = build_enterprise_excel_bytes({
            "Executive Summary": exec_summary_df,
            "Surging Topics": f_trends[["topic", "top_rising_queries", "sector", "funnel", "funnel_reason", "priority_score", "team_action", "impressions_7d", "clicks_7d", "impressions_30d", "clicks_30d", "real_ctr_30d_%", "google_rank_label", "primary_page", "primary_page_role", "recommended_money_page"]],
            "Content Decay Alerts": f_decay,
            "Ads Topic Queue": ads_topics_df,
            "Suggested Keywords": rec_kws_df.head(500),
            "New Blog Topics": truly_new_df,
            "Existing Blog Refreshes": existing_blogs_df,
            "Service Page Optimizations": existing_services_df,
            "Internal Linking": internal_links_df,
        })
        st.download_button(
            "Download Full Executive Report (.xlsx)",
            excel_bytes,
            file_name=f"connext_enterprise_search_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_exec_xlsx",
        )

    vdf = f_trends.copy()
    g_col, g_lbl = "growth_pct_3d", "3d Surge"
    if "Today" in horizon:
        vdf = vdf[vdf["is_today"]]
    elif "This Week" in horizon:
        vdf = vdf[vdf["is_week"]]
        g_col, g_lbl = "growth_pct_week", "7d Growth"
    elif "This Month" in horizon:
        vdf = vdf[vdf["is_month"]]
        g_col, g_lbl = "month_growth_pct", "30d Growth"

    if vdf.empty:
        st.info(f"No topics flagged under **{horizon}**.")
    else:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("#### Top Surging Topics by 7-Day Impressions")
            fig_t = px.bar(
                vdf.head(10), x="impressions_7d", y="topic", orientation="h", color="team_action", text=g_col,
                color_discrete_map={"Launch Paid Ads": CNXT_COLORS["sun"], "Update Existing Page": CNXT_COLORS["blue"], "Write New Page": CNXT_COLORS["twilight"], "Defend Top 3 Rank": CNXT_COLORS["teal"]},
                custom_data=["google_rank_label", "clicks_7d", "funnel"],
                labels={"impressions_7d": "7-Day Impressions", "topic": "", "team_action": ""},
            )
            fig_t.update_traces(texttemplate="+%{text}%", textposition="outside", hovertemplate="<b>%{y}</b><br>7d Impr: <b>%{x:,}</b><br>7d Clicks: <b>%{customdata[1]:,}</b><br>Rank: <b>%{customdata[0]}</b><br>Funnel: %{customdata[2]}<extra></extra>")
            fig_t.update_layout(margin=dict(l=5, r=45, t=10, b=45), yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_plotly_light(fig_t, height=370, legend_bottom=True), use_container_width=True)

        with c2:
            st.markdown("#### Impressions by Vertical")
            sec_df = vdf.groupby("sector", as_index=False).agg(topics=("topic", "count"), impr=("impressions_7d", "sum")).sort_values("impr", ascending=True)
            fig_s = px.bar(sec_df, x="impr", y="sector", orientation="h", text="topics", color_discrete_sequence=[CNXT_COLORS["twilight"]], labels={"impr": "7-Day Impressions", "sector": ""})
            fig_s.update_traces(texttemplate="%{text} topics", textposition="outside")
            fig_s.update_layout(margin=dict(l=5, r=55, t=10, b=45))
            st.plotly_chart(style_plotly_light(fig_s, height=370), use_container_width=True)

        h_c1, h_c2 = st.columns([1.3, 1])
        with h_c1:
            st.markdown("#### Brand vs. Non-Brand Search Demand Health")
            bnb = raw_df.groupby(["date", "is_branded"], as_index=False)["impressions"].sum()
            bnb["Segment"] = np.where(bnb["is_branded"], "Branded (Connext Awareness)", "Non-Branded (B2B Acquisition)")
            fig_bnb = px.area(bnb, x="date", y="impressions", color="Segment", color_discrete_map={"Non-Branded (B2B Acquisition)": CNXT_COLORS["blue"], "Branded (Connext Awareness)": CNXT_COLORS["sun"]}, labels={"date": "", "impressions": "Daily Impressions"})
            fig_bnb.update_layout(margin=dict(l=10, r=10, t=10, b=40))
            st.plotly_chart(style_plotly_light(fig_bnb, height=260, legend_bottom=True), use_container_width=True)

        with h_c2:
            st.markdown("#### Surging Topics by Buyer Funnel Stage")
            f_agg = vdf.groupby("funnel", as_index=False).agg(topics=("topic", "count"), impr=("impressions_7d", "sum"))
            fig_f = px.pie(f_agg, names="funnel", values="impr", hole=0.52, color="funnel", color_discrete_map={"BOFU — Ready to Buy": CNXT_COLORS["sun"], "MOFU — Evaluating": CNXT_COLORS["blue"], "TOFU — Awareness": CNXT_COLORS["teal"]})
            fig_f.update_layout(margin=dict(l=10, r=10, t=10, b=35))
            st.plotly_chart(style_plotly_light(fig_f, height=260, legend_bottom=True), use_container_width=True)

        st.markdown("#### Daily Search Trend Inspector")
        sel_t = st.selectbox("Select a topic to inspect its daily curve:", vdf["topic"].tolist(), key="sel_ov_topic")
        sr = vdf[vdf["topic"] == sel_t].iloc[0]
        st.markdown(
            f"""<div class="cnxt-callout">
<strong>{sr['topic'].title()}</strong> ({sr['sector']} • {sr['funnel']} — <em>{sr['funnel_reason']}</em>) — Priority <strong>{sr['priority_score']}/100</strong> • <strong>{sr['impressions_7d']:,}</strong> 7d impr. &amp; <strong>{sr['clicks_7d']:,}</strong> 7d clicks (<strong>+{sr[g_col]}%</strong> {g_lbl}) • Rank: <strong>{sr['google_rank_label']}</strong><br/>
Currently Ranking ({sr['primary_page_role']}): <code>{sr['primary_page']}</code> • Target Service Page ({sr['best_money_role']}): <code>{sr['recommended_money_page']}</code>
</div>""",
            unsafe_allow_html=True,
        )
        ts = daily_topic_df[daily_topic_df["topic"] == sel_t].sort_values("date").copy()
        ts["avg7"] = ts["impressions"].rolling(7, min_periods=1).mean().round(0)
        fig_l = go.Figure()
        fig_l.add_trace(go.Bar(x=ts["date"], y=ts["impressions"], name="Daily Impressions", marker_color="rgba(0,90,172,0.28)"))
        fig_l.add_trace(go.Scatter(x=ts["date"], y=ts["avg7"], mode="lines+markers", name="7-Day Trendline", line=dict(color=CNXT_COLORS["sun"], width=3), marker=dict(color=CNXT_COLORS["twilight"], size=6)))
        fig_l.update_layout(margin=dict(l=10, r=10, t=10, b=40))
        st.plotly_chart(style_plotly_light(fig_l, height=260, legend_bottom=True), use_container_width=True)

        st.markdown(f"#### All Surging Topics ({len(vdf)})")
        ov_tbl = vdf[["priority_score", "topic", "top_rising_queries", "sector", "funnel", "funnel_reason", "team_action", "sparkline_7d", "impressions_7d", "clicks_7d", "real_ctr_30d_%", g_col, "google_rank_label", "primary_page", "primary_page_role", "recommended_money_page"]].copy()
        ov_tbl["growth_lbl"] = ov_tbl[g_col].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
        st.dataframe(
            ov_tbl[["priority_score", "topic", "top_rising_queries", "sector", "funnel", "funnel_reason", "team_action", "sparkline_7d", "impressions_7d", "clicks_7d", "real_ctr_30d_%", "growth_lbl", "google_rank_label", "primary_page", "primary_page_role", "recommended_money_page"]],
            use_container_width=True, hide_index=True, key="tbl_overview",
            column_config={
                "priority_score": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%d"),
                "topic": "Topic",
                "top_rising_queries": st.column_config.TextColumn("Top Keywords in Cluster", width="medium"),
                "sector": "Vertical",
                "funnel": "Funnel Stage",
                "funnel_reason": "Why This Funnel",
                "team_action": "Next Step",
                "sparkline_7d": st.column_config.LineChartColumn("7d Trend", width="small"),
                "impressions_7d": st.column_config.NumberColumn("7d Impr."),
                "clicks_7d": st.column_config.NumberColumn("7d Clicks"),
                "real_ctr_30d_%": st.column_config.NumberColumn("Real CTR", format="%.2f%%"),
                "growth_lbl": g_lbl,
                "google_rank_label": "Google Rank",
                "primary_page": st.column_config.LinkColumn("Currently Ranking URL"),
                "primary_page_role": "Ranking Page Type",
                "recommended_money_page": st.column_config.LinkColumn("Target Service Page"),
            },
        )

        with st.expander("1-Click Teams Executive Summary (Copy & Paste)", expanded=False):
            lines = [f"{i+1}. **{r['topic'].title()}** ({r['sector']}, {r['funnel']}) — {r['impressions_7d']:,} 7d impr., {r['clicks_7d']:,} clicks (+{r[g_col]}%), Rank {r['google_rank_label']} → **{r['team_action']}**" for i, r in vdf.head(5).iterrows()]
            decay_note = f"\n• Content Decay Alerts Requiring Refresh: **{len(f_decay)}** topics" if not f_decay.empty else ""
            st.code(f"Connext Search Trend Summary ({datetime.now(timezone.utc).strftime('%b %d, %Y')}){decay_note}\n\n" + "\n".join(lines), language="markdown")

# ==============================================================================
# TAB 2: PAGES, CONTENT DECAY, SITE STRUCTURE & LIVE ON-PAGE AUDIT
# ==============================================================================
with t_pages:
    render_tab_guide(
        "Pages, Decay & Live On-Page Audit",
        "Shows (1) topics losing real GSC impressions or rank (Content Decay), (2) which Connext URLs rank with their accurate Page Type (including root-level blogs without /blog/), and (3) a live On-Page SEO & AI-Readiness Grader.",
        "Actionable decay alerts, exact URLs ranking on Google, a full sitemap structure breakdown, and a 0–100 HTML audit checklist when you click 'Run Live On-Page Audit'.",
        "Fix Decay pages on Page 1 first to protect existing organic traffic. Then select any topic, run the Live On-Page Audit on its URL, and send any 'Fix/Missing' items to Content or Webflow dev.",
    )

    with st.expander(f"Connext Live Website Structure Breakdown ({len(live_df):,} Sitemap URLs)", expanded=False):
        role_counts = live_df["role"].value_counts().reset_index()
        role_counts.columns = ["Classified Page Role", "Live URL Count"]
        st.caption("Root-level blog posts without a `/blog/` folder are automatically separated as `Blog / Guide (Root URL)` so they are never treated as Service Pages.")
        st.dataframe(role_counts, use_container_width=True, hide_index=True, key="tbl_site_roles")

    if not f_decay.empty:
        st.markdown(f"##### Content Decay Early Warning ({len(f_decay)} Slipping Topics)")
        st.dataframe(f_decay, use_container_width=True, hide_index=True, key="tbl_decay", column_config={"Affected Page": st.column_config.LinkColumn("Affected Page")})
    else:
        st.success("No major content decay alerts detected in this vertical over the last 7 days.")

    cann_df = f_trends[f_trends["is_cannibalized"]]
    if not cann_df.empty:
        with st.expander(f"Multi-URL Split Check: {len(cann_df)} Topics Have 2+ URLs Each Holding >= 15% Share", expanded=False):
            st.dataframe(cann_df[["topic", "sector", "ranking_pages_count", "google_rank_label", "impressions_7d", "clicks_7d", "primary_page", "primary_page_role", "recommended_money_page"]], use_container_width=True, hide_index=True, key="tbl_cann", column_config={"primary_page": st.column_config.LinkColumn("Top Ranking URL"), "recommended_money_page": st.column_config.LinkColumn("Target Service Page")})

    st.divider()
    sel_c = st.selectbox("Inspect Ranking Pages, Exact Queries & Run Live On-Page Audit:", f_trends["topic"].tolist(), key="sel_pages_topic")
    tm = f_trends[f_trends["topic"] == sel_c].iloc[0]

    p1, p2 = st.columns([1.25, 1])
    with p1:
        st.markdown("**Connext Pages Currently Ranking on Google (Real GSC Metrics)**")
        tp = f_pages[f_pages["topic"] == sel_c].sort_values("page_imp_7d", ascending=False).copy()
        tp["rank_lbl"] = tp["page_org_pos"].apply(human_rank_label)
        st.dataframe(tp[["norm_page", "page_role", "page_imp_7d", "page_clicks_7d", "page_ctr_%", "rank_lbl"]], use_container_width=True, hide_index=True, key="tbl_tp", column_config={"norm_page": st.column_config.LinkColumn("Ranking URL"), "page_role": "Page Type", "page_imp_7d": "7d Impr.", "page_clicks_7d": "7d Clicks", "page_ctr_%": st.column_config.NumberColumn("CTR", format="%.2f%%"), "rank_lbl": "Rank"})
    with p2:
        st.markdown("**Live Matching Pages in `sitemap.xml`**")
        recs, seen = [], set()
        for lbl, src in [("Verified Service Page", [{"full_url": tm["recommended_money_page"], "role": tm["best_money_role"]}]), ("Covers Topic", tm["existing_sitemap_matches"]), ("Related Blog / Guide", tm["matching_content_pages"])]:
            for item in src:
                if item["full_url"] not in seen:
                    recs.append({"URL": item["full_url"], "Page Type": item["role"], "Role": lbl})
                    seen.add(item["full_url"])
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True, key="tbl_sm_m", column_config={"URL": st.column_config.LinkColumn("Live Connext URL")})

    st.markdown("**Live On-Page SEO & AI-Readiness Grader ($0 On-Demand)**")
    audit_col1, audit_col2 = st.columns([2.2, 1])
    safe_key = re.sub(r"[^a-z0-9]+", "_", sel_c.lower())[:32]
    audit_url = audit_col1.text_input("URL to Audit:", value=tm["primary_page"] or tm["recommended_money_page"], key=f"in_audit_url_{safe_key}")
    if audit_col2.button("Run Live On-Page Audit", type="primary", key="btn_run_audit"):
        with st.spinner(f"Auditing {audit_url} (~5 seconds)..."):
            st.session_state["onpage_audit_res"] = run_live_onpage_seo_audit(audit_url.strip(), tuple(tm["queries_list"][:3]))
    if "onpage_audit_res" in st.session_state:
        ares = st.session_state["onpage_audit_res"]
        if "error" in ares:
            st.warning(ares["error"])
        else:
            st.caption(f"Audited URL: **{ares.get('url', audit_url)}**")
            st.metric("On-Page SEO & AI-Readiness Score", f"{ares['score']} / 100")
            st.dataframe(ares["checks"], use_container_width=True, hide_index=True, key="tbl_onpage_checks")

    st.markdown("**Exact Search Queries & AI Prompts (Real GSC Data)**")
    tqs = f_queries[f_queries["topic"] == sel_c].sort_values("imp_7d", ascending=False).copy()
    tqs["growth"] = tqs["query_growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
    tqs["rank"] = tqs["connext_org_pos"].apply(human_rank_label)
    st.dataframe(tqs[["query", "is_ai_prompt", "imp_7d", "clicks_7d", "imp_30d", "clicks_30d", "real_ctr_%", "growth", "rank", "top_page"]], use_container_width=True, hide_index=True, key="tbl_tqs", column_config={"is_ai_prompt": st.column_config.CheckboxColumn("AI Prompt"), "real_ctr_%": st.column_config.NumberColumn("CTR", format="%.2f%%"), "top_page": st.column_config.LinkColumn("Ranking URL")})

# ==============================================================================
# TAB 3: GOOGLE & BING ADS BUILDER (100% REAL GSC + OPTIONAL REAL KEYWORD PLANNER)
# ==============================================================================
with t_ads:
    render_tab_guide(
        "Google & Bing Ads Builder",
        "Turns Connext's real GSC search demand into ready-to-launch Google & Bing Ads keyword lists routed to verified commercial Service Pages, real wasted non-B2B negative keywords, day-of-week schedules, and RSA ad copy—with zero invented metrics.",
        "Shows real 7d & 30d GSC Impressions, Clicks, CTR, and Rank. If you upload a Google Keyword Planner CSV below, it also displays Google's real Avg. Monthly Searches and Top-of-Page Bid Ranges ($).",
        "1) Keep 'Show BOFU Buyer Topics Only' checked to focus on high-intent buyer searches. 2) Optionally export the .txt list into Google Keyword Planner and upload the CSV back here for real bid ranges. 3) Copy Exact/Phrase keywords + Real GSC Negatives into Google/Bing Ads.",
    )

    with st.expander("Optional: Upload Free Google Keyword Planner CSV for Real Monthly Volume & Top-of-Page Bid Ranges ($)", expanded=bool(kp_lookup_dict)):
        st.caption("Download the clean .txt keyword list below → paste into Google Ads → Keyword Planner → 'Get search volume and forecasts' → export CSV and upload here.")
        kp_up = st.file_uploader("Upload Google Keyword Planner CSV / XLSX", type=["csv", "xlsx", "xls"], key="inline_kp_upload")
        if kp_up:
            try:
                parsed_kp = parse_keyword_planner_csv(kp_up)
                st.session_state["kp_lookup_dict"] = parsed_kp
                st.success(f"Loaded {len(parsed_kp):,} real Keyword Planner rows!")
                st.rerun()
            except Exception as err:
                st.error(str(err))

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("BOFU Buyer Topics", int((f_trends["funnel"] == "BOFU — Ready to Buy").sum()))
    o2.metric("Real 30d Impressions (GSC)", f"{int(f_trends['impressions_30d'].sum()):,}")
    o3.metric("Real 30d Clicks (GSC)", f"{int(f_trends['clicks_30d'].sum()):,}")
    o4.metric("Keyword Planner Rows Loaded", f"{len(kp_lookup_dict):,}" if kp_lookup_dict else "Not Uploaded")

    only_bofu_ads = st.checkbox("Show BOFU Ready-to-Buy Topics Only (Recommended for Paid Search)", value=True, key="chk_only_bofu_ads")
    view_ads_topics = ads_topics_df[ads_topics_df["Funnel"] == "BOFU — Ready to Buy"].copy() if only_bofu_ads else ads_topics_df.copy()

    st.markdown(f"#### Priority Ad Topics ({len(view_ads_topics)})")
    if view_ads_topics.empty:
        st.info("No BOFU topics in this narrow filter. Uncheck 'Show BOFU Ready-to-Buy Topics Only' above to view MOFU topics.")
    else:
        st.dataframe(
            view_ads_topics, use_container_width=True, hide_index=True, key="tbl_ads_t",
            column_config={
                "Real CTR (%)": st.column_config.NumberColumn("Real CTR", format="%.2f%%"),
                "Target Service Page": st.column_config.LinkColumn("Target Service Page"),
            },
        )

    st.divider()
    ad_e1, ad_e2 = st.columns([1.2, 1])
    with ad_e1:
        st.markdown("#### Day-of-Week B2B Search Rhythm (Real GSC Impressions)")
        dow_df = clustered_df.groupby("date", as_index=False)["impressions"].sum()
        dow_df["Day"] = dow_df["date"].dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_agg = dow_df.groupby("Day", as_index=False)["impressions"].mean().round(0)
        dow_agg["Day"] = pd.Categorical(dow_agg["Day"], categories=dow_order, ordered=True)
        dow_agg = dow_agg.sort_values("Day")
        fig_dow = px.bar(dow_agg, x="Day", y="impressions", text="impressions", color_discrete_sequence=[CNXT_COLORS["blue"]], labels={"impressions": "Avg Daily B2B Impr.", "Day": ""})
        fig_dow.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_dow.update_layout(margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(style_plotly_light(fig_dow, height=290), use_container_width=True)

    with ad_e2:
        st.markdown("#### Live Negative Keyword Harvester (Real GSC Waste)")
        if not harvested_neg_df.empty:
            st.caption("Actual job-seeker & non-B2B searches seen on connextglobal.com—block these in Google & Bing Ads.")
            st.dataframe(harvested_neg_df.head(15), use_container_width=True, hide_index=True, key="tbl_neg_harv", height=245)
        else:
            st.info("No wasted non-B2B queries captured in this date range.")

    st.divider()
    st.markdown(f"#### Suggested Paid Keywords ({len(rec_kws_df)})")
    if not rec_kws_df.empty:
        kf1, kf2 = st.columns([2, 1])
        avail_ad_topics = sorted(view_ads_topics["Topic"].unique().tolist()) if not view_ads_topics.empty else sorted(rec_kws_df["Topic"].unique().tolist())
        kw_topics = kf1.multiselect("Filter by Topic:", sorted(rec_kws_df["Topic"].unique().tolist()), default=avail_ad_topics, key="ms_kw_t")
        min_k_imp = kf2.number_input("Min 7-Day Impressions:", 1, 10000, 5, 5, key="num_min_kw")
        scoped_kws = rec_kws_df[(rec_kws_df["Topic"].isin(kw_topics)) & (rec_kws_df["7d Impr. (GSC)"] >= min_k_imp)]
        st.dataframe(
            scoped_kws, use_container_width=True, hide_index=True, key="tbl_ads_k",
            column_config={
                "Real CTR (%)": st.column_config.NumberColumn("Real CTR", format="%.2f%%"),
                "Target Service Page": st.column_config.LinkColumn("Target Service Page"),
            },
        )

        d1, d2 = st.columns(2)
        d1.download_button("Download Paid Keywords CSV", scoped_kws.to_csv(index=False).encode("utf-8"), file_name="connext_paid_keywords.csv", mime="text/csv", key="dl_pk")
        d2.download_button("Export Clean List for Google Keyword Planner (.txt)", "\n".join(scoped_kws["Keyword"].head(500).tolist()).encode("utf-8"), file_name="keywords_for_keyword_planner.txt", mime="text/plain", key="dl_kp")

        with st.expander("Ad Group Copy-Paste Builder (Exact, Phrase, Real GSC Negatives & RSA Headlines)", expanded=False):
            builder_topics = view_ads_topics["Topic"].tolist() if not view_ads_topics.empty else ads_topics_df["Topic"].tolist()
            sel_ad = st.selectbox("Pick Topic:", builder_topics, key="sel_ad_grp")
            ad_row = f_trends[f_trends["topic"] == sel_ad].iloc[0]
            kws = rec_kws_df[rec_kws_df["Topic"] == sel_ad]["Keyword"].tolist() or ad_row["queries_list"]
            st.caption(f"Verified Commercial Landing Page ({ad_row['best_money_role']}): **{ad_row['recommended_money_page']}**")
            ce, cp, cn = st.columns(3)
            ce.markdown("**[Exact Match]**")
            ce.code("\n".join(f"[{q}]" for q in kws), language="text")
            cp.markdown('**"Phrase Match"**')
            cp.code("\n".join(f'"{q}"' for q in kws), language="text")
            cn.markdown("**B2B Negatives (Base + Real GSC Waste)**")
            base_negs = ["-jobs", "-careers", "-hiring", "-salary", "-internship", "-glassdoor", "-indeed", "-free", "-template"]
            if not harvested_neg_df.empty:
                col0 = harvested_neg_df.columns[0]
                base_negs.extend([f'"{q}"' for q in harvested_neg_df[col0].head(10).tolist()])
            cn.code("\n".join(base_negs), language="text")
            hl, desc = build_rsa_assets(sel_ad)
            ch, cd = st.columns(2)
            ch.markdown("**Headlines (≤ 30 chars)**")
            for cat, h in hl:
                ch.text(f"[{cat}] {h} ({len(h)}/30)")
            cd.markdown("**Descriptions (≤ 90 chars)**")
            for d in desc:
                cd.text(f"{d} ({len(d)}/90)")

# ==============================================================================
# TAB 4: NEW CONTENT PLAN, BLOG REFRESHES, SERVICE PAGES & INTERNAL LINKING
# ==============================================================================
with t_content:
    render_tab_guide(
        "New Content Plan & Internal Linking",
        "Separates your content roadmap into 4 clear buckets: (1) Brand-new blog posts to write for topics missing from Connext, (2) Existing blog posts (both /blog/ and root-level blogs) to refresh, (3) Commercial Service Pages to optimize, and (4) Contextual internal links from blogs into our Service Pages.",
        "Every topic is verified against both live GSC ranking URLs and connextglobal.com/sitemap.xml so writers never create duplicate blog posts.",
        "1) Assign Section 1 topics as new blog posts. 2) Refresh existing blogs in Section 2. 3) In Section 4, open each Source Blog in Webflow and add a link pointing to the Target Service Page.",
    )

    st.markdown(f"#### 1. New Content to Write — True Content Gaps ({len(truly_new_df)})")
    st.markdown(
        """<div class="cnxt-callout">
<strong>Why Write These:</strong> Buyers are searching for these B2B topics on Google, and neither a blog post nor a dedicated subpage in <code>connextglobal.com/sitemap.xml</code> covers them yet. Write a new article and link readers to the verified <strong>Service Page to Link To</strong>.
</div>""",
        unsafe_allow_html=True,
    )
    if truly_new_df.empty:
        st.success("All currently surging B2B topics already have a live page in connextglobal.com/sitemap.xml!")
    else:
        st.dataframe(
            truly_new_df, use_container_width=True, hide_index=True, key="tbl_new_c",
            column_config={"Service Page to Link To": st.column_config.LinkColumn("Service Page to Link To")},
        )
        st.download_button("Download New Blog Topics CSV", truly_new_df.to_csv(index=False).encode("utf-8"), file_name="connext_new_blog_topics.csv", mime="text/csv", key="dl_new_c")

    st.divider()
    st.markdown(f"#### 2. Existing Blog Posts to Refresh & Expand ({len(existing_blogs_df)})")
    st.markdown(
        """<div class="cnxt-callout">
<strong>Why Refresh Existing Blogs:</strong> Connext already has a live blog post (either under <code>/blog/</code> or as a root-level blog URL) covering these topics. Expand these articles with new H2s, add FAQ schema, and link them to the matching <strong>Service Page to Link To</strong>.
</div>""",
        unsafe_allow_html=True,
    )
    if existing_blogs_df.empty:
        st.info("No existing blog post refreshes queued under current filters.")
    else:
        st.dataframe(
            existing_blogs_df, use_container_width=True, hide_index=True, key="tbl_ex_blogs",
            column_config={
                "Existing Blog Post to Refresh": st.column_config.LinkColumn("Existing Blog Post"),
                "Service Page to Link To": st.column_config.LinkColumn("Service Page to Link To"),
            },
        )
        st.download_button("Download Existing Blog Refreshes CSV", existing_blogs_df.to_csv(index=False).encode("utf-8"), file_name="connext_existing_blog_refreshes.csv", mime="text/csv", key="dl_ex_blogs")

    st.divider()
    st.markdown(f"#### 3. Commercial Service / Industry Pages to Optimize ({len(existing_services_df)})")
    if existing_services_df.empty:
        st.info("No Service/Industry page optimizations queued under current filters.")
    else:
        st.dataframe(
            existing_services_df, use_container_width=True, hide_index=True, key="tbl_ex_svcs",
            column_config={"Service / Industry Page": st.column_config.LinkColumn("Service / Industry Page")},
        )

    st.divider()
    st.markdown(f"#### 4. Internal Linking Opportunities — Connect Blogs → Verified Service Pages ({len(internal_links_df)})")
    st.markdown(
        """<div class="cnxt-callout">
<strong>How This Works (No Jargon):</strong> Adding a hyperlink inside an educational blog post pointing to our commercial <strong>Target Service / Industry Page</strong> passes ranking strength to the Service Page and guides readers toward requesting pricing.
</div>""",
        unsafe_allow_html=True,
    )
    if internal_links_df.empty:
        st.info("No internal link additions needed under current filters.")
    else:
        st.dataframe(
            internal_links_df, use_container_width=True, hide_index=True, key="tbl_int_links",
            column_config={
                "Source Blog Post (Edit This Post)": st.column_config.LinkColumn("Source Blog Post"),
                "Target Service / Industry Page": st.column_config.LinkColumn("Target Service Page"),
            },
        )
        st.download_button("Download Internal Linking Plan CSV", internal_links_df.to_csv(index=False).encode("utf-8"), file_name="connext_internal_linking_plan.csv", mime="text/csv", key="dl_int_links")

    st.divider()
    st.markdown("#### 5. 1-Click Writer Brief & Webflow `FAQPage` JSON-LD Generator")
    sel_b = st.selectbox("Select Topic to Generate Writer Brief:", f_trends["topic"].tolist(), key="sel_brief")
    br = f_trends[f_trends["topic"] == sel_b].iloc[0]
    lt_b = "\n".join(f"  - `{q}`" for q in (br["long_tail_list"] or br["queries_list"]))
    is_new_post = (not br["already_exists_in_sitemap"]) and (br["coverage_status"] == "New Topic Gap")
    bc1, bc2 = st.columns([1.1, 1])
    with bc1:
        st.markdown(
            f"""**Writer Brief: {br['topic'].title()}** ({br['sector']} • {br['funnel']})
- **Deliverable:** {'Write a New Blog Post (Topic not yet in sitemap)' if is_new_post else 'Refresh Existing ' + br['closest_existing_role'] + ': `' + br['closest_existing_url'] + '`'}
- **Current Rank & Demand:** {br['google_rank_label']} • **{br['impressions_7d']:,}** 7d impr. • **{br['clicks_7d']:,}** 7d clicks
- **Verified Commercial Service Page to Link To ({br['best_money_role']}):** `{br['recommended_money_page']}`
- **Exact Rising Searches to Answer (Use as H2s / FAQs):**
{lt_b}"""
        )
    with bc2:
        st.markdown("**Ready-to-Paste `FAQPage` JSON-LD Schema (For Webflow Custom Code)**")
        st.code(build_faq_jsonld(br["topic"], br["queries_list"]), language="html")

# ==============================================================================
# TAB 5: DEDICATED COMPETITOR US ADS TRANSPARENCY TAB (ALL 12 COMPETITORS)
# ==============================================================================
with t_ads_trans:
    render_tab_guide(
        "Competitor US Ads Transparency (12 Rivals)",
        "Dedicated view of all 12 Connext BPO competitors (Solvo, Genius, Microsourcing, Emapta, Outsourced, OP360, Lean Solutions, TaskUs, TOA Global, Cloudstaff, Near, BairesDev) across Google Ads Transparency Center (US Region) and Microsoft Ad Library.",
        "Section 1 gives instant 1-click links for every competitor ($0, zero API credits). Section 2 runs an optional button-triggered US Google Ads Transparency & Live US Ad Auction scan using your existing SerpApi key (cached 7 days).",
        "Click 'Open US Ads ↗' next to any competitor to inspect every Search, Display, and YouTube ad they are currently running in the United States, study their offers, and differentiate Connext's ad copy.",
    )

    st.markdown("#### 1. Direct US Google Ads Transparency Center & Microsoft Ad Library Directory ($0 Free)")
    st.markdown(
        """<div class="cnxt-callout">
<strong>How to Inspect Any Competitor's US Ads ($0):</strong> Click <strong>Open US Ads ↗</strong> next to any competitor below to open their verified advertiser hub filtered to the <strong>United States</strong> (using their bare domain so all agency and subsidiary accounts appear), or click <strong>Search MS Ads ↗</strong> to inspect Microsoft/Bing ads.
</div>""",
        unsafe_allow_html=True,
    )
    dir_df = pd.DataFrame([
        {
            "Competitor": cname,
            "Domain": dom,
            "Google Ads Transparency (US)": f"https://adstransparency.google.com/?region=US&domain={dom}",
            "Microsoft Ad Library": "https://adlibrary.ads.microsoft.com/",
        }
        for cname, dom, _ in COMPETITORS_REGISTRY
    ])
    st.dataframe(
        dir_df, use_container_width=True, hide_index=True, key="tbl_ad_dir_main",
        column_config={
            "Google Ads Transparency (US)": st.column_config.LinkColumn("Google Ads Transparency (US)", display_text="Open US Ads ↗"),
            "Microsoft Ad Library": st.column_config.LinkColumn("Microsoft Ad Library", display_text="Search MS Ads ↗"),
        },
    )

    st.divider()
    st.markdown("#### 2. Optional Automated US Google Ads Transparency & Live US Ad Auction Scan")
    if has_serpapi_secret:
        st.caption("Never runs automatically—uses your existing SerpApi key ONLY when you click the button below (cached 7 days to protect your 1,000 monthly credits).")
        sc1, sc2 = st.columns([1.4, 1])
        num_kw_ads = sc1.slider("Top BOFU Surging Keywords to Check for Live US Google Ads:", 0, 10, 5, step=1, key="sl_us_ad_kws")
        chk_12_trans = sc2.checkbox("Check All 12 Competitors in Google Ads Transparency (+12 credits)", value=True, key="chk_12_trans")
        est_c = (12 if chk_12_trans else 0) + num_kw_ads

        b_ad1, b_ad2 = st.columns([2, 1])
        with b_ad1:
            if st.button(f"Run On-Demand US Google Ads Transparency Scan ({est_c} credits, cached 7d)", type="primary", key="btn_run_us_ads"):
                bofu_kws = f_queries[f_queries["is_commercial"]]["query"].head(num_kw_ads).tolist() if num_kw_ads > 0 else []
                with st.spinner("Scanning Google Ads Transparency Center (US) & live US Google Ads..."):
                    t_df, p_df, s_msg = run_on_demand_us_ads_transparency_scan(
                        str(st.secrets["serpapi"]["api_key"]).strip(),
                        tuple(bofu_kws),
                        scan_all_12_transparency=chk_12_trans,
                    )
                    st.session_state["us_trans_df"] = t_df
                    st.session_state["us_paid_df"] = p_df
                    st.session_state["us_ads_msg"] = s_msg
        with b_ad2:
            if st.button("Clear 7-Day Ad Cache", key="btn_clr_us_ads"):
                run_on_demand_us_ads_transparency_scan.clear()
                st.session_state.pop("us_trans_df", None)
                st.session_state.pop("us_paid_df", None)
                st.session_state.pop("us_ads_msg", None)

        if "us_ads_msg" in st.session_state:
            st.success(st.session_state["us_ads_msg"])
        if "us_trans_df" in st.session_state and not st.session_state["us_trans_df"].empty:
            st.markdown("**Competitor US Google Search Ads Status (Ads Transparency Center)**")
            st.dataframe(st.session_state["us_trans_df"], use_container_width=True, hide_index=True, key="tbl_us_trans", column_config={"Google Ads Transparency (US)": st.column_config.LinkColumn("View US Ads", display_text="Open US Ads ↗")})
        if "us_paid_df" in st.session_state and not st.session_state["us_paid_df"].empty:
            st.markdown("**Live US Google Paid Search Ads Detected on Trending Keywords**")
            st.dataframe(st.session_state["us_paid_df"], use_container_width=True, hide_index=True, key="tbl_us_paid", column_config={"Landing Page": st.column_config.LinkColumn("Ad Landing Page")})
    else:
        st.info("To run automated US Google Ads Transparency checks inside this tab, add `[serpapi]` `api_key` in Streamlit Secrets (never runs automatically—only when you click the button).")

    st.divider()
    st.markdown("#### 3. Official Free Microsoft Ad Library API Scan (EU/EEA Spillover Creatives — $0)")
    if st.button("Scan 12 Competitors via Free Microsoft Ad Library API ($0)", key="btn_ms_api"):
        with st.spinner("Querying Microsoft Ad Library API (~10s)..."):
            st.session_state["ms_api_df"] = fetch_free_microsoft_ad_library_all()
    if "ms_api_df" in st.session_state:
        ms_df = st.session_state["ms_api_df"]
        if ms_df.empty:
            st.info("No EU/EEA Microsoft Ads returned (use the direct US Ad Library links in Section 1 above for US-only campaigns).")
        else:
            st.dataframe(ms_df, use_container_width=True, hide_index=True, key="tbl_ms_ads", column_config={"Ad Library Link": st.column_config.LinkColumn("Open Ad", display_text="View Ad ↗")})

# ==============================================================================
# TAB 6: ON-DEMAND COMPETITOR NEW BLOGS & B2B INDUSTRY TRENDS ($0)
# ==============================================================================
with t_comp_blogs:
    render_tab_guide(
        "Competitor New Blogs & Industry Trends ($0)",
        "Scans our 12 competitors' sitemaps strictly for newly published B2B blog posts + extracts their Main Target Keywords, plus provides industry-filtered Google Trends, Bing Search, and Reddit buyer research.",
        "Runs on demand (~10–20s per click, $0 cost) so the main dashboard loads quickly.",
        "Filter Competitor Blogs to 'Content Gaps Only' to spot topics rivals are writing about that Connext lacks, and use the Question & Reddit Finder to uncover real buyer pain points.",
    )

    r_sub1, r_sub2, r_sub3, r_sub4 = st.tabs([
        "A. Competitor New Blogs & Keywords ($0)",
        "B. B2B Google Trends & Industry Feed ($0)",
        "C. Bing Search Performance ($0)",
        "D. Question & Reddit Pain-Point Finder ($0)",
    ])

    with r_sub1:
        b_top1, b_top2, b_top3 = st.columns([2.2, 1.1, 0.9])
        blog_days = b_top1.slider("Competitor Blogs Published / Updated Within (Days):", 14, 90, 45, key="sl_blog_days")
        if b_top2.button("Scan 12 Competitors' New Blogs Now", type="primary", key="btn_scan_blogs"):
            with st.spinner("Scanning all 12 competitor sitemaps strictly for new B2B blogs & extracting main keywords (~15–20s)..."):
                bdf, bnotes = scan_competitor_new_blogs_auto(lookback_days=blog_days)
                if not bdf.empty:
                    cov_l, cls_l, svc_l = [], [], []
                    for _, row in bdf.iterrows():
                        chk = check_topic_against_sitemap(f"{row['Blog Headline']} {row['Main Target Keywords']}", "", row["Vertical"], live_set, live_df, live_vec, live_mat)
                        cov_l.append("Connext Covers" if chk["already_exists_in_sitemap"] else "Content Gap")
                        cls_l.append(chk["closest_existing_url"])
                        svc_l.append(chk["best_money_url"])
                    bdf["Connext Status"] = cov_l
                    bdf["Connext Closest Page"] = cls_l
                    bdf["Connext Service Page"] = svc_l
                st.session_state["comp_blogs_df"] = bdf
                st.session_state["comp_blog_notes"] = bnotes
        if b_top3.button("Clear Cache", key="btn_clr_blogs"):
            scan_competitor_new_blogs_auto.clear()
            st.session_state.pop("comp_blogs_df", None)

        if "comp_blog_notes" in st.session_state:
            st.caption("Blog feed status: " + " • ".join(st.session_state["comp_blog_notes"]))

        comp_blogs_df = st.session_state.get("comp_blogs_df", pd.DataFrame())
        if comp_blogs_df.empty:
            st.info("Click **Scan 12 Competitors' New Blogs Now** above to pull strictly new B2B editorial blog posts and their extracted target keywords (cached 24 hours, $0).")
        else:
            bf1, bf2, bf3 = st.columns([1.2, 1.2, 1])
            b_comps = bf1.multiselect("Competitor:", sorted(comp_blogs_df["Competitor"].unique().tolist()), default=sorted(comp_blogs_df["Competitor"].unique().tolist()), key="ms_b_comp")
            b_verts = bf2.multiselect("Vertical:", sorted(comp_blogs_df["Vertical"].unique().tolist()), default=sorted(comp_blogs_df["Vertical"].unique().tolist()), key="ms_b_vert")
            b_gap = bf3.selectbox("Connext Coverage:", ["All New Blogs", "Content Gaps Only", "Already Covered"], key="sel_b_gap")

            bview = comp_blogs_df[(comp_blogs_df["Competitor"].isin(b_comps)) & (comp_blogs_df["Vertical"].isin(b_verts))].copy()
            if b_gap == "Content Gaps Only":
                bview = bview[bview["Connext Status"] == "Content Gap"]
            elif b_gap == "Already Covered":
                bview = bview[bview["Connext Status"] == "Connext Covers"]

            bk1, bk2, bk3 = st.columns(3)
            bk1.metric("New Competitor Blogs", len(bview))
            bk2.metric("Topic Gaps Missing on Connext", int((bview["Connext Status"] == "Content Gap").sum()))
            bk3.metric("Active Rivals Publishing", bview["Competitor"].nunique())

            st.dataframe(
                bview[["Competitor", "Published", "Blog Headline", "Main Target Keywords", "Vertical", "Connext Status", "Connext Service Page", "Blog URL"]],
                use_container_width=True, hide_index=True, key="tbl_comp_blogs",
                column_config={
                    "Blog Headline": st.column_config.TextColumn("Blog Headline", width="large"),
                    "Main Target Keywords": st.column_config.TextColumn("Main Target Keywords", width="medium"),
                    "Connext Service Page": st.column_config.LinkColumn("Matching Connext Service Page"),
                    "Blog URL": st.column_config.LinkColumn("Read Competitor Blog"),
                },
            )
            st.download_button("Download Competitor New Blogs & Keywords CSV", bview[["Competitor", "Published", "Blog Headline", "Main Target Keywords", "Vertical", "Connext Status", "Connext Service Page", "Blog URL"]].to_csv(index=False).encode("utf-8"), file_name="connext_competitor_new_blogs_keywords.csv", mime="text/csv", key="dl_comp_blogs")

    with r_sub2:
        tc1, tc2, tc3 = st.columns([2.2, 0.9, 0.9])
        gt_seeds = tc1.text_input("Compare up to 5 B2B Topics in Google Trends:", value="medical billing outsourcing, offshore accounting, virtual assistant outsourcing, nearshore software developers, bpo outsourcing", key="gt_in")
        gt_geo = tc2.selectbox("Region:", ["US", "CA", "AU", "GB"], index=0, key="gt_geo")
        gt_tf = tc3.selectbox("Window:", ["today 1-m", "today 3-m", "today 12-m"], index=0, key="gt_tf")

        news_vert = st.selectbox("B2B Industry News Feed (Last 14 Days):", list(INDUSTRY_NEWS_FEEDS.keys()), index=0, key="sel_news_vert")
        if st.button("Load B2B Google Trends & Industry Headlines", type="primary", key="btn_load_gt"):
            seed_tup = tuple(s.strip().lower() for s in gt_seeds.split(",") if s.strip())[:5]
            with st.spinner("Fetching B2B Google Trends & industry headlines (~10s)..."):
                iot_d, ris_d, g_n = fetch_free_google_trends_b2b(seed_tup, geo=gt_geo, timeframe=gt_tf, strict_b2b=strict_b2b_filter)
                if not ris_d.empty:
                    sm_s, sm_l = [], []
                    for _, rr in ris_d.iterrows():
                        vert_r = classify_sector(rr["Rising B2B Query"])
                        chk = check_topic_against_sitemap(rr["Rising B2B Query"], "", vert_r, live_set, live_df, live_vec, live_mat)
                        sm_s.append("Already Covered" if chk["already_exists_in_sitemap"] else "New Topic Gap")
                        sm_l.append(chk["best_money_url"])
                    ris_d["Connext Status"] = sm_s
                    ris_d["Target Service Page"] = sm_l
                st.session_state["gt_iot_df"] = iot_d
                st.session_state["gt_ris_df"] = ris_d
                st.session_state["gt_notes"] = g_n
                st.session_state["b2b_news_df"] = fetch_b2b_industry_trending_feed(news_vert)

        if "gt_notes" in st.session_state and st.session_state["gt_notes"]:
            st.caption(" • ".join(st.session_state["gt_notes"]))
        iot_df = st.session_state.get("gt_iot_df", pd.DataFrame())
        if not iot_df.empty:
            fig_gt = px.line(iot_df, x="date", y="interest_score", color="keyword", color_discrete_sequence=[CNXT_COLORS["blue"], CNXT_COLORS["sun"], CNXT_COLORS["teal"], CNXT_COLORS["twilight"], CNXT_COLORS["red"]], labels={"date": "", "interest_score": "Search Interest (0–100)", "keyword": ""})
            fig_gt.update_layout(margin=dict(l=10, r=10, t=10, b=40))
            st.plotly_chart(style_plotly_light(fig_gt, height=310, legend_bottom=True), use_container_width=True)

        ris_df = st.session_state.get("gt_ris_df", pd.DataFrame())
        if not ris_df.empty:
            st.markdown("**Rising B2B Queries (Google Trends)**")
            st.dataframe(ris_df, use_container_width=True, hide_index=True, key="tbl_gt_r", column_config={"Target Service Page": st.column_config.LinkColumn("Target Service Page")})

        b2b_news_df = st.session_state.get("b2b_news_df", pd.DataFrame())
        if not b2b_news_df.empty:
            st.markdown("**Live B2B Industry Trend Headlines (Strictly Industry-Filtered)**")
            st.dataframe(b2b_news_df, use_container_width=True, hide_index=True, key="tbl_b2b_news", column_config={"Article Link": st.column_config.LinkColumn("Read Article")})

    with r_sub3:
        if has_bing_secret:
            if st.button("Load / Refresh Bing Webmaster Queries", type="primary", key="btn_ref_bing"):
                with st.spinner("Syncing Bing Webmaster Tools (~8s)..."):
                    bdf, b_stat = fetch_free_bing_webmaster_queries(str(st.secrets["bing"]["api_key"]), st.secrets["bing"].get("site_url", "https://connextglobal.com/"), brand_regex, exclude_local_ph, custom_negatives, strict_b2b_filter)
                    st.session_state["bing_wmt_df"] = bdf
                    st.session_state["bing_wmt_stat"] = b_stat
            if "bing_wmt_stat" in st.session_state:
                st.caption(st.session_state["bing_wmt_stat"])
            bing_df = st.session_state.get("bing_wmt_df", pd.DataFrame())
            if not bing_df.empty:
                bing_df["Growth"] = bing_df["growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
                st.dataframe(bing_df[["query", "vertical", "bing_impr_recent", "bing_clicks", "Growth", "bing_rank_label"]], use_container_width=True, hide_index=True, key="tbl_bing", column_config={"query": "Bing Query", "vertical": "Vertical", "bing_impr_recent": "Recent Impr.", "bing_clicks": "Clicks", "bing_rank_label": "Bing Rank"})
        else:
            st.markdown("""<div class="cnxt-callout"><strong>Connect Free Bing Webmaster Tools ($0):</strong> Go to <strong>bing.com/webmasters → Settings → API Access → Generate API Key</strong>, then add <code>[bing]</code> with <code>api_key</code> and <code>site_url = "https://connextglobal.com/"</code> in Streamlit Secrets.</div>""", unsafe_allow_html=True)

    with r_sub4:
        rc1, rc2, rc3 = st.columns([1.2, 1.4, 0.8])
        picked_seed = rc1.selectbox("Pick Surging GSC Topic:", f_trends["topic"].tolist(), key="sel_res_seed")
        custom_seed = rc2.text_input("Or Type Custom B2B Topic:", placeholder="e.g., dental billing outsourcing", key="in_res_seed")
        active_seed = custom_seed.strip() or picked_seed
        if rc3.button("Mine Questions & Reddit", type="primary", key="btn_mine_res"):
            with st.spinner(f"Mining Google/Bing Autocomplete & Reddit for '{active_seed}' (~10s)..."):
                s_df = fetch_free_google_and_bing_suggestions(active_seed, strict_b2b=strict_b2b_filter)
                if not s_df.empty:
                    s_df["Connext Status"] = ["Already Covered" if check_topic_against_sitemap(q, "", classify_sector(q), live_set, live_df, live_vec, live_mat)["already_exists_in_sitemap"] else "New Gap" for q in s_df["Search Suggestion"]]
                st.session_state["res_sugg_df"] = s_df
                st.session_state["res_red_df"] = fetch_free_reddit_b2b_discussions(active_seed)
                st.session_state["res_seed_lbl"] = active_seed

        if "res_seed_lbl" in st.session_state:
            col_sugg, col_red = st.columns(2)
            with col_sugg:
                st.markdown(f"**Google & Bing Autocomplete for `{st.session_state['res_seed_lbl']}`**")
                sdf = st.session_state.get("res_sugg_df", pd.DataFrame())
                if sdf.empty:
                    st.info("No B2B autocomplete suggestions returned.")
                else:
                    st.dataframe(sdf, use_container_width=True, hide_index=True, key="tbl_sugg")
            with col_red:
                st.markdown(f"**Reddit B2B Buyer Discussions for `{st.session_state['res_seed_lbl']}`**")
                rdf = st.session_state.get("res_red_df", pd.DataFrame())
                if rdf.empty:
                    st.info("No recent Reddit threads matched—try a shorter 2-word topic.")
                else:
                    st.dataframe(rdf, use_container_width=True, hide_index=True, key="tbl_reddit", column_config={"Thread": st.column_config.LinkColumn("Open Thread")})

# ==============================================================================
# TAB 7: COMPLETE USER GUIDE, PLAIN-ENGLISH GLOSSARY & TEAM PLAYBOOKS
# ==============================================================================
with t_guide:
    st.markdown("### Complete User Guide for Non-SEO Reviewers")
    st.markdown('<p class="cnxt-sec-sub">Everything you need to know about how this radar works, what each real metric means, and how to turn insights into leads and traffic.</p>', unsafe_allow_html=True)

    g_row1_c1, g_row1_c2 = st.columns(2)
    with g_row1_c1:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>1. What to Expect (Speed &amp; 100% Real Data)</h3>
<ul>
<li><strong>Initial Login Load (30–60 seconds):</strong> After unlocking, the app pulls up to 45–90 days of live Google Search Console rows, filters out job-seeker and off-topic searches, groups queries into Connext B2B topics, and indexes <code>connextglobal.com/sitemap.xml</code>.</li>
<li><strong>Zero Invented Numbers:</strong> Every impression, click, CTR, and Google rank comes directly from Google Search Console. Keyword Planner search volumes and bid ranges appear only when you upload an actual Keyword Planner CSV.</li>
<li><strong>Root-Level Blogs vs. Service Pages:</strong> Migrated blog posts published at root URLs without <code>/blog/</code> are automatically labeled <em>Blog / Guide (Root URL)</em> so Target Service Page links always point to true commercial Industry, Team, Service, or Location pages.</li>
<li><strong>On-Demand Tabs (Tabs 5 &amp; 6):</strong> Competitor US Ads Transparency Scans, Competitor Blog Scans, On-Page Audits, Google Trends, Bing, and Reddit run only when you click their button (typically 10–20 seconds).</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )
    with g_row1_c2:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>2. Step-by-Step: How to Use the Tool in 5 Minutes</h3>
<ul>
<li><strong>Step 1 — Narrow Your Focus:</strong> Use the top filter bar to pick a <strong>Business Vertical</strong> (e.g., <em>Healthcare &amp; RCM</em>) or a <strong>Funnel Stage</strong> (e.g., <em>BOFU — Ready to Buy</em>).</li>
<li><strong>Step 2 — Spot Priorities (Tab 1):</strong> Sort by <strong>Priority (0–100)</strong> or <strong>7d Impr.</strong> Click any topic in the Trend Inspector to verify whether demand is steadily growing or a one-day spike.</li>
<li><strong>Step 3 — Protect Existing Rankings (Tab 2):</strong> Review <strong>Content Decay Alerts</strong> and run the <strong>Live On-Page Audit</strong> on slipping Page-1 URLs.</li>
<li><strong>Step 4 — Take Action (Tabs 3 &amp; 4):</strong> Export ready-to-bid keywords &amp; negative keywords for Paid Ads, or grab New Blog Topics, Existing Blog Refreshes, Internal Links, and FAQ Schema for Content.</li>
<li><strong>Step 5 — Check Competitors &amp; US Ads (Tabs 5 &amp; 6):</strong> Open <strong>Tab 5</strong> for Competitor US Ads Transparency and <strong>Tab 6</strong> for Competitor New Blogs.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### 3. Plain-English Metrics Glossary (No SEO Jargon)")
    glossary_df = pd.DataFrame([
        {"Metric / Term": "Impressions (7d / 30d)", "Plain-English Meaning": "Real Google Search Console count of how many times a connextglobal.com link appeared on a searcher's screen."},
        {"Metric / Term": "Clicks (7d / 30d) & Real CTR (%)", "Plain-English Meaning": "Actual organic clicks from Google Search and the exact Click-Through Rate (Clicks ÷ Impressions) recorded by Google Search Console."},
        {"Metric / Term": "Google Rank (Top 3 / Page 1 / Page 2)", "Plain-English Meaning": "Where Connext appears organically (unpaid). Top 3 (#1–3) gets the highest CTR; #4–#15 ('Striking Distance') are quick wins; Page 2+ (#11+) needs Ads or a page refresh."},
        {"Metric / Term": "3d / 7d / 30d Growth (%)", "Plain-English Meaning": "How much faster daily impressions grew compared to the prior 2-week baseline. Example: +100% means daily visibility doubled."},
        {"Metric / Term": "Priority Score (0–100)", "Plain-English Meaning": "Combines buyer intent (BOFU/MOFU/TOFU), real search volume, surge speed, and rank gap into a single score so you know what matters most first."},
        {"Metric / Term": "Funnel Stage (BOFU / MOFU / TOFU)", "Plain-English Meaning": "BOFU = Ready to buy ('pricing', 'hire', 'outsourcing company'); MOFU = Comparing options ('best', 'vs', 'dedicated team', 'employee rebadging'); TOFU = Early awareness/definitions ('what is', 'meaning', 'trends')."},
        {"Metric / Term": "Content Decay Alert", "Plain-English Meaning": "Early warning for topics where Connext lost >=22% of real impressions or slipped >=2.5 rank spots. Refreshing these pages protects existing organic traffic."},
        {"Metric / Term": "Blog / Guide (Root URL)", "Plain-English Meaning": "Editorial articles on connextglobal.com that live directly at the root domain without '/blog/' in the URL—kept strictly separate from commercial Service Pages."},
        {"Metric / Term": "Internal Linking (Blogs → Service Pages)", "Plain-English Meaning": "Adding a hyperlink inside an educational blog post pointing to a commercial Service/Industry page so Google ranks the Service Page higher and readers can request pricing."},
    ])
    st.dataframe(glossary_df, use_container_width=True, hide_index=True, key="tbl_glossary")

    st.markdown("### 4. Role-Based Weekly Playbooks (How to Use the Results)")
    pb1, pb2, pb3 = st.columns(3)
    with pb1:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>Content &amp; SEO Team</h3>
<ul>
<li><strong>Monday (10 min):</strong> Open <strong>Tab 2</strong> → check <em>Content Decay Alerts</em> and run the <em>Live On-Page Audit</em> on any slipping Page-1 URL.</li>
<li><strong>Tuesday (15 min):</strong> Open <strong>Tab 4</strong> → pick new topics from <em>1. New Content to Write</em>, refresh blogs in <em>2. Existing Blog Posts to Refresh</em>, add the <em>4. Internal Links</em> in Webflow, and paste the <em>FAQPage JSON-LD</em> schema.</li>
<li><strong>Thursday (10 min):</strong> Open <strong>Tab 6</strong> → scan our 12 competitors' new blogs, filter to <em>Content Gaps Only</em>, and add the best topics to the editorial calendar.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )
    with pb2:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>Paid Media (Google &amp; Bing Ads)</h3>
<ul>
<li><strong>Find High-ROI Campaigns:</strong> Set <em>Funnel Stage</em> to <strong>BOFU — Ready to Buy</strong> and open <strong>Tab 3</strong>. Focus on <em>P0 / P1</em> topics where Connext ranks below #5 organically.</li>
<li><strong>Verify Real CPC Bids ($0):</strong> Click <em>Export Clean List for Google Keyword Planner (.txt)</em>, paste into Google Keyword Planner, download the forecast CSV, and upload it back into Tab 3.</li>
<li><strong>Cut Wasted Spend:</strong> Copy the <em>Real GSC Negative Keywords</em> and check the <em>Day-of-Week Chart</em> to weight ad budgets toward peak Mon–Fri B2B search days.</li>
<li><strong>Spy on Rival US Ads:</strong> Open <strong>Tab 5 (Competitor US Ads Transparency)</strong> for direct US Google Ads Transparency links and optional US auction checks.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )
    with pb3:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>Leadership &amp; Marketing Managers</h3>
<ul>
<li><strong>Weekly Pulse Check:</strong> Open <strong>Tab 1 (Executive Overview)</strong> to see total surging B2B topics, real 7d impressions &amp; clicks, and Content Decay count.</li>
<li><strong>Brand vs. Acquisition Health:</strong> Check the <em>Brand vs. Non-Brand</em> area chart to verify new-buyer search demand is growing alongside brand awareness.</li>
<li><strong>Share in 10 Seconds:</strong> Expand <em>1-Click Teams Executive Summary</em> to copy a clean bulleted update into Microsoft Teams, or click <em>Download Full Executive Report (.xlsx)</em> for leadership decks.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

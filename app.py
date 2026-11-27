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


def inject_connext_brand_css():
    """Hardened light-surface styling that prevents dark-mode text/input clashes on login & dashboard."""
    css = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Arimo:wght@500;600;700&family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
:root {
    --cnxt-blue: #005AAC; --cnxt-sun: #E58B25; --cnxt-yellow: #F1C916;
    --cnxt-twilight: #071776; --cnxt-midnight: #071739; --cnxt-teal: #0F818A;
    --cnxt-teal-blue: #90CBDB; --cnxt-bg: #F7F8FB; --cnxt-paper: #FFFFFF;
    --cnxt-ink: #0F172A; --cnxt-ash: #64748B; --cnxt-line: #E2E8F0;
    --font-display: 'Arimo', Arial, sans-serif;
    --font-body: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}
html, body, .stApp { font-family: var(--font-body); background-color: var(--cnxt-bg) !important; color: var(--cnxt-ink) !important; }
header[data-testid="stHeader"] { background-color: var(--cnxt-bg) !important; border-bottom: 1px solid var(--cnxt-line); }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1440px; }
h1, h2, h3, h4 { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; letter-spacing: -0.015em; }

/* Main Canvas Form & Input Contrast */
.stApp [data-testid="stForm"] {
    background-color: #FFFFFF !important; border: 1px solid var(--cnxt-line) !important;
    border-radius: 12px !important; padding: 22px 24px !important;
    box-shadow: 0 4px 16px rgba(7, 23, 57, 0.06) !important;
}
.stApp .main label, .stApp [data-testid="stWidgetLabel"] p { color: #1E293B !important; font-weight: 500 !important; }
.stApp .main input, .stApp .main textarea {
    background-color: #FFFFFF !important; color: #0F172A !important;
    border: 1px solid #CBD5E1 !important; border-radius: 8px !important;
}
.stApp .main [data-baseweb="select"] > div {
    background-color: #FFFFFF !important; color: #0F172A !important;
    border: 1px solid #CBD5E1 !important; border-radius: 8px !important;
}
.stApp .main [data-baseweb="select"] span, .stApp .main [data-baseweb="select"] input { color: #0F172A !important; }
[data-baseweb="popover"], [data-baseweb="menu"] { background-color: #FFFFFF !important; color: #0F172A !important; }
[data-baseweb="menu"] li { color: #0F172A !important; }
.stApp .main [data-testid="stRadio"] label p, .stApp .main [data-testid="stCheckbox"] label p { color: #1E293B !important; }
.stApp .main [data-testid="stExpander"] {
    background-color: #FFFFFF !important; border: 1px solid var(--cnxt-line) !important; border-radius: 10px !important;
}
.stApp .main [data-testid="stExpander"] summary p { color: var(--cnxt-twilight) !important; font-weight: 600 !important; }
.stApp .main [data-testid="stCodeBlock"] pre { background-color: #0F172A !important; color: #F8FAFC !important; border-radius: 8px !important; }

/* Primary & Form Submit Buttons */
.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button,
.stDownloadButton > button {
    background-color: var(--cnxt-sun) !important; color: #071739 !important;
    font-family: var(--font-body) !important; font-weight: 700 !important;
    border: none !important; border-radius: 999px !important; padding: 0.52rem 1.35rem !important;
    box-shadow: 0 4px 12px rgba(229, 139, 37, 0.30) !important;
}
.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
.stDownloadButton > button:hover {
    background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important;
}
.stButton > button:not([kind="primary"]) {
    background-color: #FFFFFF !important; color: var(--cnxt-twilight) !important;
    border: 1px solid #CBD5E1 !important; border-radius: 999px !important; font-weight: 600 !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--cnxt-twilight) !important; background-color: #F1F5F9 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background-color: var(--cnxt-midnight) !important; border-right: 1px solid rgba(144, 203, 219, 0.16); }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #F8FAFC !important; }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small { color: var(--cnxt-teal-blue) !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea { background-color: #FFFFFF !important; color: #071739 !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #071739 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #071739 !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] { background-color: rgba(255,255,255,0.06) !important; border: 1px solid rgba(144,203,219,0.22) !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p { color: #F1C916 !important; }
section[data-testid="stSidebar"] .stButton > button { background-color: rgba(255,255,255,0.10) !important; color: var(--cnxt-teal-blue) !important; border: 1px solid rgba(144,203,219,0.32) !important; border-radius: 999px !important; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background-color: var(--cnxt-sun) !important; color: var(--cnxt-midnight) !important; border: none !important; font-weight: 700 !important; }

/* Hero Banner */
.cnxt-hero {
    background: linear-gradient(135deg, #071776 0%, #071739 100%);
    color: #FFFFFF; padding: 22px 28px; border-radius: 12px; margin-bottom: 16px;
    border: 1px solid rgba(144,203,219,0.2); box-shadow: 0 6px 20px rgba(7,23,57,0.08);
}
.cnxt-eyebrow { font-family: var(--font-mono); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.16em; color: #F1C916; margin-bottom: 5px; font-weight: 600; }
.cnxt-hero h1 { font-size: clamp(1.35rem, 2.2vw, 1.9rem) !important; color: #FFFFFF !important; margin: 0 0 6px 0 !important; }
.cnxt-hero h1 span { color: #E58B25; }
.cnxt-hero p { color: #CBD5E1 !important; font-size: 0.93rem; margin: 0; max-width: 900px; line-height: 1.5; }

/* Metric Cards */
div[data-testid="stMetric"] {
    background-color: var(--cnxt-paper); border: 1px solid var(--cnxt-line);
    border-top: 3px solid var(--cnxt-blue); border-radius: 10px; padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetric"] label { font-family: var(--font-mono) !important; font-size: 0.70rem !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; color: var(--cnxt-ash) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; font-weight: 700 !important; font-size: 1.6rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid var(--cnxt-line); margin-bottom: 14px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-display); font-weight: 600; font-size: 0.89rem;
    color: #334155 !important; background-color: #FFFFFF !important; border-radius: 8px 8px 0 0;
    padding: 10px 14px; border: 1px solid var(--cnxt-line); border-bottom: none;
}
.stTabs [aria-selected="true"] { background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important; border-color: var(--cnxt-twilight) !important; }

/* Callout & Info Cards */
.cnxt-callout {
    background-color: #FFFFFF; border: 1px solid var(--cnxt-line);
    border-left: 4px solid var(--cnxt-sun); border-radius: 8px;
    padding: 14px 18px; margin: 8px 0 14px 0; color: #1E293B; font-size: 0.92rem; line-height: 1.55;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}
.cnxt-callout strong { color: var(--cnxt-twilight); }
.cnxt-info-card {
    background-color: #FFFFFF; border: 1px solid var(--cnxt-line); border-radius: 12px;
    padding: 20px 22px; margin-bottom: 14px; color: #1E293B; font-size: 0.92rem; line-height: 1.6;
    box-shadow: 0 2px 10px rgba(7, 23, 57, 0.04);
}
.cnxt-info-card h3 { margin-top: 0 !important; margin-bottom: 10px !important; font-size: 1.08rem !important; }
.cnxt-info-card ul { margin: 6px 0 10px 18px; padding: 0; }
.cnxt-info-card li { margin-bottom: 6px; }
.cnxt-timing-box {
    background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #E58B25;
    border-radius: 8px; padding: 12px 16px; margin: 12px 0; color: #78350F; font-size: 0.9rem; line-height: 1.5;
}
.cnxt-sec-sub { color: #64748B !important; font-size: 0.9rem; margin-top: -8px; margin-bottom: 12px; }
span[data-baseweb="tag"] { background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important; }
</style>"""
    st.markdown(css, unsafe_allow_html=True)


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

    st.markdown("<style>section[data-testid='stSidebar']{display:none !important;}</style>", unsafe_allow_html=True)

    st.image(CNXT_LOGO_COLOR, width=190)
    st.markdown(
        """<div class="cnxt-hero" style="margin-top: 12px; margin-bottom: 20px;">
<div class="cnxt-eyebrow">CONNEXT GLOBAL INTERNAL MARKETING &amp; GROWTH PLATFORM</div>
<h1>Connext Search &amp; Topic <span>Trend Radar</span></h1>
<p>A plain-English intelligence tool that turns live Google Search Console data into actionable priorities for Paid Ads, Content Creation, SEO updates, and Competitor Tracking—no SEO background required.</p>
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
Right after you click <strong>Unlock Enterprise Radar</strong>, the app connects live to Google Search Console, filters out job-seeker noise, groups thousands of searches into B2B topics, and checks every topic against <code>connextglobal.com/sitemap.xml</code>. <strong>Please keep the tab open and do not refresh while it loads.</strong>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """<div class="cnxt-info-card" style="padding: 14px 18px; font-size: 0.86rem;">
<strong>Data Security &amp; Access Guidelines:</strong>
<ul>
<li><strong>Read-Only Connection:</strong> Nothing you click inside this dashboard can edit Google Search Console, Google Ads, or the live website.</li>
<li><strong>Internal Use Only:</strong> Do not share the app link or password outside Connext Global. Click <em>Lock / Log Out</em> in the sidebar on shared screens.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    with col_guide:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>What This Tool Does (In Plain English)</h3>
<ul>
<li><strong>Finds Rising Buyer Demand:</strong> Spots B2B outsourcing topics growing right now on Google (3-day spikes, 7-day surges, and 30-day growth) across all Connext verticals.</li>
<li><strong>Prevents Duplicate Content:</strong> Automatically cross-checks every idea against <code>connextglobal.com/sitemap.xml</code> so we update existing pages instead of creating duplicates.</li>
<li><strong>Catches Slipping Pages Early:</strong> Flags <em>Content Decay</em> when a Connext topic loses search visibility or drops in Google rank.</li>
<li><strong>Builds Paid Ads &amp; ROI Cases:</strong> Estimates missed monthly clicks and equivalent ad value, plus ready-to-paste keywords, negative keywords, and headlines.</li>
<li><strong>Tracks 12 Competitors &amp; Market Trends ($0):</strong> Scans new competitor blog posts (extracting their main target keywords) and pulls B2B Google Trends, Bing, and Reddit pain points on demand.</li>
</ul>
<h3>How Each Team Uses This Data</h3>
<ul>
<li><strong>Content &amp; SEO Team:</strong> Use <em>Tab 2 (Decay &amp; On-Page Audit)</em> to rescue slipping pages, <em>Tab 4</em> for writer briefs, internal links, and FAQ schema, and <em>Tab 5</em> to see what competitors just published.</li>
<li><strong>Paid Media (Google &amp; Bing Ads):</strong> Filter by <em>BOFU — Ready to Buy</em>, open <em>Tab 3</em> to grab high-intent keywords, copy real GSC negative keywords, and use the Day-of-Week chart for ad scheduling.</li>
<li><strong>Leadership &amp; Strategy:</strong> Check <em>Tab 1 (Executive Overview)</em> for Brand vs. Non-Brand health, copy the <em>1-Click Teams Summary</em>, or download the <em>Full Excel Report (.xlsx)</em>.</li>
</ul>
<h3>What to Expect Once Inside</h3>
<ul>
<li><strong>Tabs 1–4</strong> load automatically after the initial 30–60s sync (cached for 1 hour so filtering is instant).</li>
<li><strong>Tabs 5 &amp; 6</strong> (Competitor Blogs, Trends, Reddit) run <strong>only when you click their button</strong> (~10–20s) to keep the main dashboard fast.</li>
<li>Open <strong>Tab 7 (User Guide &amp; Glossary)</strong> anytime for plain-English definitions of every metric and step-by-step workflows.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    st.stop()


require_connext_login()

# ==============================================================================
# CONNEXT SITE ARCHITECTURE & TAXONOMY
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

DEFAULT_SITE_URL = "sc-domain:connextglobal.com"
DEFAULT_BRAND_REGEX = r"\b(connext|connext\s*global(\s*solutions)?|connextglobal)\b"

STRICT_NON_INDUSTRY_REGEX = re.compile(
    r"\b("
    r"job|jobs|career|careers|hiring|hire me|salary|salaries|compensation|wage|pay rate|"
    r"apply|application|applicant|interview|resume|cv|cover letter|internship|intern|interns|"
    r"vacancy|vacancies|opening|openings|recruiter|recruiting|work from home|wfh|part time|"
    r"glassdoor|indeed|jobstreet|mynimo|kalibrr|ziprecruiter|monster|linkedin login|"
    r"employee|employees|staff portal|login|sign in|payslip|payroll login|benefits|reviews|"
    r"address|phone number|contact number|directions|map|free template|definition|wikipedia"
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

BOFU_REGEX = re.compile(r"\b(cost|pricing|rates|quote|hire|agency|company|companies|provider|partner|firm|services|outsourcing|outsource|staffing)\b", re.I)
MOFU_REGEX = re.compile(r"\b(best|top|vs|versus|compare|comparison|solutions|dedicated|offshore|nearshore|remote team|bpo)\b", re.I)
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
# UTILITY HELPERS
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


def classify_page_role(path: str, live_paths_set: set[str]) -> str:
    if path not in live_paths_set:
        return "Old / Redirected URL"
    if path == "/":
        return "Homepage"
    if path.startswith("/industries/") or path == "/hipaa-compliant-virtual-medical-assistants":
        return "Industry Page"
    if path.startswith("/teams/"):
        return "Team Solution Page"
    if path.startswith("/services/") or path in {"/our-service-model", "/our-differentiator", "/compare-by-model", "/connext-pricing", "/trust-compliance", "/roles-we-fill"}:
        return "Service Model Page"
    if path.startswith("/locations"):
        return "Location Hub"
    if path.startswith("/tools"):
        return "Interactive Tool"
    if path.startswith("/case-studies/"):
        return "Case Study"
    if path.startswith("/white-paper/"):
        return "White Paper"
    if path.startswith("/one-pager/"):
        return "One-Pager"
    if path.startswith("/e-books/"):
        return "E-Book"
    if path.startswith("/blog/") or path.startswith("/general-outsourcing") or path in {"/bpo-glossary", "/insights/faq", "/insights"}:
        return "Blog / Guide"
    return "Core Brand Page"


def classify_funnel_stage(text: str) -> str:
    if BOFU_REGEX.search(text):
        return "BOFU — Ready to Buy"
    if MOFU_REGEX.search(text):
        return "MOFU — Evaluating"
    return "TOFU — Research"


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


def expected_ctr_by_rank(pos: float) -> float:
    if pos <= 0:
        return 0.0
    if pos <= 1.5:
        return 0.285
    if pos <= 2.5:
        return 0.155
    if pos <= 3.5:
        return 0.105
    if pos <= 5.5:
        return 0.065
    if pos <= 10.5:
        return 0.028
    if pos <= 20.5:
        return 0.008
    return 0.002


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
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ConnextTrendRadar/3.0)"}
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
    live_df["is_money_page"] = live_df["role"].isin(["Industry Page", "Team Solution Page", "Service Model Page", "Location Hub", "Interactive Tool"])
    live_df["is_content_page"] = live_df["role"].isin(["Blog / Guide", "Case Study", "White Paper", "One-Pager", "E-Book"])
    live_df["slug_text"] = (
        live_df["path"]
        .str.replace(r"^/(industries|teams|services|locations|tools|blog|case-studies|white-paper|one-pager|e-books)/", "", regex=True)
        .str.replace(r"[/\-_]", " ", regex=True)
        .str.strip()
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
    mat = vec.fit_transform(live_df["slug_text"].fillna(""))
    return live_set, live_df, vec, mat


def check_topic_against_sitemap(topic: str, queries_sample: str, live_set: set[str], live_df: pd.DataFrame, vec: TfidfVectorizer, mat) -> dict:
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
            scored.loc[scored["path"] == target_path, "sim"] += 0.35
            break

    money_df = scored[scored["is_money_page"]].sort_values("sim", ascending=False)
    content_df = scored[scored["is_content_page"]].sort_values(["raw_sim", "token_overlap"], ascending=False)
    best_money_row = money_df.iloc[0] if not money_df.empty else scored.iloc[0]

    existing_matches = scored[(scored["path"] != "/") & ((scored["raw_sim"] >= 0.28) | (scored["token_overlap"] >= 0.67))].sort_values(["token_overlap", "raw_sim"], ascending=False)
    already_exists = len(existing_matches) > 0
    if already_exists:
        closest_url = existing_matches.iloc[0]["full_url"]
        closest_pct = int(round(max(float(existing_matches.iloc[0]["raw_sim"]), float(existing_matches.iloc[0]["token_overlap"])) * 100))
    elif not content_df.empty:
        closest_url = content_df.iloc[0]["full_url"]
        closest_pct = int(round(max(float(content_df.iloc[0]["raw_sim"]), float(content_df.iloc[0]["token_overlap"])) * 100))
    else:
        closest_url, closest_pct = best_money_row["full_url"], 0

    return {
        "best_money_url": best_money_row["full_url"],
        "best_money_role": best_money_row["role"],
        "has_strong_money_match": bool(boosted_path is not None or float(best_money_row["sim"]) >= 0.22),
        "already_exists_in_sitemap": already_exists,
        "closest_existing_url": closest_url,
        "closest_overlap_pct": min(100, closest_pct),
        "existing_sitemap_matches": existing_matches.head(3)[["full_url", "role"]].to_dict("records"),
        "top_money_pages": money_df.head(3)[["full_url", "role", "sim"]].to_dict("records"),
        "top_content_pages": content_df[content_df["raw_sim"] >= 0.08].head(5)[["full_url", "role", "raw_sim"]].to_dict("records"),
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

    return {"score": score, "checks": pd.DataFrame(checks)}


# ==============================================================================
# KEYWORD PLANNER ($0 CSV PARSER) & GSC LOADERS
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
    high_c = next((c for c in df.columns if "high range" in c or "cpc" in c), None)
    if not kw_c:
        raise ValueError("Missing 'Keyword' column.")

    out = {}
    for _, r in df.iterrows():
        k = str(r[kw_c]).lower().strip()
        if not k or k == "nan":
            continue
        vol = int(pd.to_numeric(r[vol_c], errors="coerce") or 0) if vol_c else 0
        low_b = float(pd.to_numeric(r[low_c], errors="coerce") or 0.0) if low_c else 0.0
        high_b = float(pd.to_numeric(r[high_c], errors="coerce") or 0.0) if high_c else low_b
        avg_cpc = round((low_b + high_b) / 2.0, 2) if (low_b > 0 and high_b > 0) else round(max(low_b, high_b), 2)
        out[k] = {"kp_volume": vol, "kp_cpc": avg_cpc, "kp_low_bid": round(low_b, 2), "kp_high_bid": round(high_b, 2)}
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
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0).astype(int)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0)
    df = df[(df["query"] != "") & (df["impressions"] > 0)].copy()

    noise_mask = df["query"].str.contains(STRICT_NON_INDUSTRY_REGEX, regex=True, na=False)
    if exclude_local_ph:
        noise_mask = noise_mask | df["query"].str.contains(LOCAL_PH_AND_STUDENT_REGEX, regex=True, na=False)
    if custom_negatives.strip():
        noise_mask = noise_mask | df["query"].str.contains(custom_negatives, regex=True, na=False)

    neg_df = df[noise_mask].groupby("query", as_index=False)["impressions"].sum().sort_values("impressions", ascending=False).head(100)
    neg_df.columns = ["Wasted Non-B2B Query", "Wasted GSC Impr."]
    clean_df = df[~noise_mask].copy()
    clean_df["is_branded"] = clean_df["query"].str.contains(brand_regex, regex=True, na=False) if brand_regex else False
    return clean_df, neg_df


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
            siteUrl=site_url,
            body={"startDate": start_dt.isoformat(), "endDate": end_dt.isoformat(), "dimensions": ["date", "query", "page"], "dataState": "all", "rowLimit": row_limit, "startRow": start_row},
        ).execute()
        if first_incomplete is None:
            first_incomplete = resp.get("metadata", {}).get("firstIncompleteDate", "None")
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
            is_noise = bool(STRICT_NON_INDUSTRY_REGEX.search(clean_q)) or (exclude_local_ph and bool(LOCAL_PH_AND_STUDENT_REGEX.search(clean_q))) or (custom_pat and bool(custom_pat.search(clean_q)))
            if is_noise:
                neg_counter[clean_q] = neg_counter.get(clean_q, 0) + int(r["impressions"])
                continue
            rows.append({
                "date": date_str, "query": clean_q, "page": normalize_url(page),
                "impressions": int(r["impressions"]), "position": float(r["position"]),
                "is_branded": bool(brand_pat.search(clean_q)) if brand_pat else False,
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
# ON-DEMAND COMPETITOR NEW BLOGS + TARGET KEYWORD EXTRACTOR ($0)
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
    out_df["Vertical"] = out_df.apply(lambda r: classify_sector(f"{r['Blog Headline']} {r['Main Target Keywords']} {r['slug']}"), axis=1)
    return out_df, notes


# ==============================================================================
# ON-DEMAND FREE DISCOVERY (TRENDS, INDUSTRY NEWS, BING, SUGGEST & REDDIT)
# ==============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_google_trends_b2b(seed_queries: tuple[str, ...], geo: str = "US", timeframe: str = "today 1-m"):
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
                                if rq_clean and not STRICT_NON_INDUSTRY_REGEX.search(rq_clean):
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
def fetch_free_bing_webmaster_queries(api_key: str, site_url: str, brand_regex: str, exclude_local_ph: bool, custom_negatives: str):
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
def fetch_free_google_and_bing_suggestions(seed_topic: str) -> pd.DataFrame:
    rows, seen = [], set()
    for q_str in [seed_topic, f"how to {seed_topic}", f"cost of {seed_topic}", f"best {seed_topic}", f"{seed_topic} vs ", f"{seed_topic} pricing"]:
        try:
            url = f"https://suggestqueries.google.com/complete/search?client=firefox&hl=en&gl=us&q={urllib.parse.quote(q_str)}"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(data, list) and len(data) >= 2:
                for s in data[1]:
                    sc = str(s).lower().strip()
                    if sc and sc not in seen and not STRICT_NON_INDUSTRY_REGEX.search(sc):
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
                if sc and sc not in seen and not STRICT_NON_INDUSTRY_REGEX.search(sc):
                    seen.add(sc)
                    rows.append({"Search Suggestion": sc, "Engine": "Bing", "Intent": "Commercial / Pricing" if COMMERCIAL_MODIFIERS.search(sc) else "Question / Guide"})
    except Exception:
        pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_reddit_b2b_discussions(seed_topic: str) -> pd.DataFrame:
    try:
        url = f"https://www.reddit.com/search.json?q={urllib.parse.quote(f'{seed_topic} outsourcing OR offshore OR agency')}&sort=relevance&t=year&limit=10"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ConnextTrendRadar/3.0"}), timeout=10) as resp:
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


def analyze_trends_and_decay(non_brand_df: pd.DataFrame, live_set: set[str], live_df: pd.DataFrame, live_vec: TfidfVectorizer, live_mat, min_daily_imp: int, min_up_days: int, min_norm_slope: float, min_lift: float):
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
    df["weighted_pos"] = df["position"] * df["impressions"]
    daily_topic = df.groupby(["topic", "date"], as_index=False).agg(impressions=("impressions", "sum"), weighted_pos=("weighted_pos", "sum"))

    recent_df, prior_df, month_df = df[df["date"].isin(week_dates)], df[df["date"].isin(prior_14_dates)], df[df["date"].isin(month_dates)]
    top_q_map = recent_df.groupby(["topic", "query"], as_index=False)["impressions"].sum().sort_values("impressions", ascending=False).groupby("topic")["query"].apply(list).to_dict()
    lt_q_map = recent_df[recent_df["is_long_tail"]].groupby(["topic", "query"], as_index=False)["impressions"].sum().sort_values("impressions", ascending=False).groupby("topic")["query"].apply(list).to_dict()

    pages_agg = recent_df.groupby(["topic", "norm_page", "path"], as_index=False).agg(page_imp_7d=("impressions", "sum"), page_weighted_pos=("weighted_pos", "sum"), queries_on_page=("query", "nunique")).sort_values(["topic", "page_imp_7d"], ascending=[True, False])
    pages_agg["page_org_pos"] = (pages_agg["page_weighted_pos"] / np.maximum(pages_agg["page_imp_7d"], 1)).round(1)
    pages_agg["page_role"] = pages_agg["path"].apply(lambda p: classify_page_role(p, live_set))
    pages_agg["is_live_url"] = pages_agg["path"].isin(live_set)
    pages_agg["is_money_url"] = pages_agg["page_role"].isin(["Industry Page", "Team Solution Page", "Service Model Page", "Location Hub", "Interactive Tool"])

    top_page_map = pages_agg.groupby("topic")["norm_page"].first().to_dict()
    has_money_map = pages_agg.groupby("topic")["is_money_url"].any().to_dict()
    has_legacy_map = (~pages_agg["is_live_url"]).groupby(pages_agg["topic"]).any().to_dict()
    page_cnt_map = pages_agg.groupby("topic")["norm_page"].nunique().to_dict()

    surge_results, decay_results = [], []

    for topic, group in daily_topic.groupby("topic"):
        series = group.set_index("date")["impressions"].reindex(all_dates, fill_value=0)
        arr_today, arr_week, arr_prior14 = series.loc[today_dates].values, series.loc[week_dates].values, series.loc[prior_14_dates].values
        arr_m_rec, arr_m_base = series.loc[month_recent].values, series.loc[month_prior].values

        m_today, m_week, m_prior = float(np.mean(arr_today)), float(np.mean(arr_week)), max(float(np.mean(arr_prior14)), 1.0)
        m_mrec, m_mbase = float(np.mean(arr_m_rec)), max(float(np.mean(arr_m_base)), 1.0)

        rec_grp = group[group["date"].isin(week_dates)]
        pri_grp = group[group["date"].isin(prior_14_dates)]
        imp_7d = int(rec_grp["impressions"].sum())
        imp_30d = int(group[group["date"].isin(month_dates)]["impressions"].sum())
        avg_pos_now = float(rec_grp["weighted_pos"].sum() / max(imp_7d, 1)) if imp_7d > 0 else 0.0
        avg_pos_prior = float(pri_grp["weighted_pos"].sum() / max(int(pri_grp["impressions"].sum()), 1)) if not pri_grp.empty else avg_pos_now
        rank_slip = round(avg_pos_now - avg_pos_prior, 1)

        drop_pct_7d = round(((m_week / m_prior) - 1.0) * 100, 1)
        if m_prior >= max(min_daily_imp, 15) and (drop_pct_7d <= -22.0 or (rank_slip >= 2.5 and avg_pos_prior <= 15.0)):
            gsc_url_d = top_page_map.get(topic, "")
            decay_results.append({
                "Topic": topic,
                "Vertical": classify_sector(f"{topic} {gsc_url_d}"),
                "7d Impr.": imp_7d,
                "Impr. Drop": f"{drop_pct_7d:.0f}%",
                "Prior Rank": f"#{avg_pos_prior:.1f}",
                "Current Rank": f"#{avg_pos_now:.1f}" if avg_pos_now > 0 else "Dropped",
                "Rank Slip": f"{rank_slip:+.1f}" if avg_pos_now > 0 else "Lost",
                "Affected Page": gsc_url_d,
                "Rescue Action": "Refresh H1/H2s, update stats & add internal links immediately" if avg_pos_prior <= 10.0 else "Audit competing SERP pages & consolidate thin content",
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
        has_comm = bool(COMMERCIAL_MODIFIERS.search(joined_q))
        funnel = classify_funnel_stage(joined_q)
        u_pages = int(page_cnt_map.get(topic, 1))
        gsc_url = top_page_map.get(topic, "")
        sector = classify_sector(f"{joined_q} {gsc_url}")
        site_chk = check_topic_against_sitemap(topic, joined_q, live_set, live_df, live_vec, live_mat)
        rec_lp = site_chk["best_money_url"] if site_chk["has_strong_money_match"] else gsc_url

        if has_legacy_map.get(topic, False):
            cov_status = "Old URL Redirect Check"
        elif has_money_map.get(topic, False):
            cov_status = "Service Page Ranking"
        elif site_chk["already_exists_in_sitemap"]:
            cov_status = "Already in Sitemap"
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

        cur_clicks = round(imp_30d * expected_ctr_by_rank(avg_pos_now), 1)
        top3_clicks = round(imp_30d * expected_ctr_by_rank(2.5), 1)
        missed_clicks = max(0.0, round(top3_clicks - cur_clicks, 1))

        prio_score = min(100, int(
            (30 if funnel.startswith("BOFU") else (20 if funnel.startswith("MOFU") else 10))
            + min(30, int(np.log1p(imp_7d) * 4.5))
            + min(25, int(max(lift_today - 1.0, 0) * 18))
            + (15 if 4.0 <= avg_pos_now <= 15.0 else (10 if avg_pos_now > 15.0 else 5))
        ))

        surge_results.append({
            "topic": topic, "sector": sector, "funnel": funnel, "priority_score": prio_score,
            "is_today": is_today, "is_week": is_week, "is_month": is_month,
            "is_striking_distance": bool(4.0 <= avg_pos_now <= 15.0), "is_cannibalized": bool(u_pages >= 2),
            "team_action": action,
            "growth_pct_3d": round((lift_today - 1.0) * 100),
            "growth_pct_week": round(((m_week / m_prior) - 1.0) * 100),
            "month_growth_pct": round((lift_month - 1.0) * 100),
            "impressions_7d": imp_7d, "impressions_30d": imp_30d,
            "sparkline_7d": [int(v) for v in arr_week],
            "connext_org_pos": round(avg_pos_now, 1), "google_rank_label": human_rank_label(avg_pos_now),
            "est_cur_clicks_30d": cur_clicks, "est_top3_clicks_30d": top3_clicks, "est_missed_clicks_30d": missed_clicks,
            "commercial_intent": has_comm, "coverage_status": cov_status,
            "already_exists_in_sitemap": site_chk["already_exists_in_sitemap"],
            "closest_existing_url": site_chk["closest_existing_url"],
            "closest_overlap_pct": site_chk["closest_overlap_pct"],
            "existing_sitemap_matches": site_chk["existing_sitemap_matches"],
            "ranking_pages_count": u_pages, "primary_page": gsc_url, "recommended_money_page": rec_lp,
            "matching_money_pages": site_chk["top_money_pages"], "matching_content_pages": site_chk["top_content_pages"],
            "queries_list": q_list, "long_tail_list": lt_list,
            "total_queries": len(q_list), "long_tail_count": len(lt_list), "top_rising_queries": joined_q,
        })

    decay_df = pd.DataFrame(decay_results).sort_values("7d Impr.", ascending=False).reset_index(drop=True) if decay_results else pd.DataFrame()
    if not surge_results:
        return pd.DataFrame(), decay_df, daily_topic, pd.DataFrame(), pd.DataFrame()

    trends_df = pd.DataFrame(surge_results).sort_values(["priority_score", "impressions_7d"], ascending=False).reset_index(drop=True)

    q_recent = recent_df.groupby(["topic", "query", "is_long_tail", "is_ai_prompt"], as_index=False).agg(imp_7d=("impressions", "sum"), weighted_pos=("weighted_pos", "sum"), top_page=("norm_page", "first"))
    q_merged = q_recent.merge(month_df.groupby("query", as_index=False).agg(imp_30d=("impressions", "sum")), on="query", how="left").merge(prior_df.groupby("query", as_index=False).agg(imp_base14=("impressions", "sum")), on="query", how="left").fillna({"imp_30d": 0, "imp_base14": 0})
    q_merged["imp_30d"] = q_merged["imp_30d"].astype(int)
    q_merged["daily_7d"] = (q_merged["imp_7d"] / 7.0).round(1)
    q_merged["query_growth_pct"] = (((q_merged["daily_7d"] / np.maximum(q_merged["imp_base14"] / max(len(prior_14_dates), 1), 0.5)) - 1.0) * 100).round(0).astype(int)
    q_merged["connext_org_pos"] = (q_merged["weighted_pos"] / np.maximum(q_merged["imp_7d"], 1)).round(1)
    q_merged["sector"] = q_merged.apply(lambda r: classify_sector(f"{r['query']} {r['top_page']}"), axis=1)
    q_merged["is_commercial"] = q_merged["query"].str.contains(COMMERCIAL_MODIFIERS, na=False)
    q_merged = q_merged.sort_values(["imp_7d", "imp_30d"], ascending=False)

    pages_merged = pages_agg[pages_agg["topic"].isin(trends_df["topic"])].copy()
    tot_map = trends_df.set_index("topic")["impressions_7d"].to_dict()
    pages_merged["topic_imp_share_%"] = ((pages_merged["page_imp_7d"] / pages_merged["topic"].map(tot_map).replace(0, 1)) * 100).round(1)
    return trends_df, decay_df, daily_topic, q_merged, pages_merged


def build_internal_linking_plan(trends_df: pd.DataFrame, all_pages_gsc_df: pd.DataFrame) -> pd.DataFrame:
    page_imp_map = all_pages_gsc_df.groupby("norm_page")["impressions"].sum().to_dict() if not all_pages_gsc_df.empty else {}
    links = []
    for _, r in trends_df.iterrows():
        target_money = r["recommended_money_page"]
        if not target_money or r["connext_org_pos"] <= 3.0:
            continue
        for c_page in r["matching_content_pages"][:3]:
            src_url = c_page["full_url"]
            if src_url == target_money:
                continue
            links.append({
                "Source Blog / Guide (Add Link Here)": src_url,
                "Source Page Type": c_page["role"],
                "Source GSC Impr.": int(page_imp_map.get(src_url, 0)),
                "Suggested Anchor Text": r["topic"],
                "Target Service / Industry Page": target_money,
                "Target Topic Rank": r["google_rank_label"],
                "Vertical": r["sector"],
            })
    if not links:
        return pd.DataFrame()
    return pd.DataFrame(links).drop_duplicates(subset=["Source Blog / Guide (Add Link Here)", "Target Service / Industry Page"]).sort_values("Source GSC Impr.", ascending=False).reset_index(drop=True)


def build_ads_and_roi_tables(trends_df: pd.DataFrame, queries_df: pd.DataFrame, kp_lookup: dict, default_cpc: float, lead_rate: float, deal_ltv: float):
    ads_topics, rec_kws = [], []
    lp_map = trends_df.set_index("topic")["recommended_money_page"].to_dict()

    for _, r in trends_df.iterrows():
        pos, comm = r["connext_org_pos"], r["commercial_intent"]
        kp = kp_lookup.get(r["topic"].lower(), {})
        cpc = kp.get("kp_cpc", default_cpc) if kp.get("kp_cpc", 0) > 0 else default_cpc
        missed = r["est_missed_clicks_30d"]
        lost_ad_val = round(missed * cpc, 0)
        pipeline_opp = round(missed * lead_rate * deal_ltv, 0)
        prio = "P0 — Launch Today" if (r["is_today"] and comm and pos > 5.0) else ("P1 — Search Ad Group" if (comm and pos > 5.0) else ("P2 — Defend Top 5" if comm else "P2 — Long-Tail"))
        ads_topics.append({
            "Priority": prio, "Topic": r["topic"], "Funnel": r["funnel"], "Vertical": r["sector"],
            "Google Rank": r["google_rank_label"], "30d Impr.": r["impressions_30d"],
            "Missed Clicks/Mo": missed, "CPC ($)": cpc,
            "Lost Traffic Value/Mo": lost_ad_val, "Pipeline Opp./Mo": pipeline_opp,
            "Landing Page": r["recommended_money_page"],
        })

    for _, q in queries_df.iterrows():
        kw, is_lt, is_comm = q["query"], q["is_long_tail"], q["is_commercial"]
        kp = kp_lookup.get(kw.lower(), {})
        cpc = kp.get("kp_cpc", default_cpc) if kp.get("kp_cpc", 0) > 0 else default_cpc
        q_missed = max(0.0, round(q["imp_30d"] * (expected_ctr_by_rank(2.5) - expected_ctr_by_rank(q["connext_org_pos"])), 1))
        mtype = '"Phrase" + [Exact]' if (is_comm and not is_lt) else ("[Exact Match]" if is_lt else '"Phrase Match"')
        rec_kws.append({
            "Keyword": kw, "7d Impr.": int(q["imp_7d"]), "30d Impr.": int(q["imp_30d"]),
            "KP Monthly Vol.": kp.get("kp_volume", "—") if kp.get("kp_volume", 0) > 0 else "—",
            "Growth": f"+{int(q['query_growth_pct'])}%" if q["query_growth_pct"] >= 0 else f"{int(q['query_growth_pct'])}%",
            "Google Rank": human_rank_label(q["connext_org_pos"]),
            "Missed Clicks/Mo": q_missed, "CPC ($)": cpc, "Lost Value/Mo": round(q_missed * cpc, 0),
            "Match Type": mtype, "Topic": q["topic"], "Landing Page": lp_map.get(q["topic"], q["top_page"]),
        })

    return (
        pd.DataFrame(ads_topics).sort_values("Lost Traffic Value/Mo", ascending=False),
        pd.DataFrame(rec_kws).sort_values(["Lost Value/Mo", "7d Impr."], ascending=False) if rec_kws else pd.DataFrame(),
    )


def build_content_tables(trends_df: pd.DataFrame):
    updates, new_ideas = [], []
    for _, r in trends_df.iterrows():
        if r["coverage_status"] == "Old URL Redirect Check":
            prio, act = "P0 — Fix Redirect", f"Verify {r['primary_page']} redirects to {r['recommended_money_page']}."
        elif r["is_cannibalized"] and r["ranking_pages_count"] >= 2:
            prio, act = "P1 — Merge Competing URLs", f"{r['ranking_pages_count']} URLs split rank (#{r['connext_org_pos']}). Consolidate into {r['closest_existing_url']}."
        elif r["already_exists_in_sitemap"]:
            prio, act = "P1 — Update Existing Page", f"Already in sitemap ({r['closest_overlap_pct']}% match). Add FAQ schema & link to service page."
        else:
            prio, act = "P2 — Protect Top Rank", f"Ranks #{r['connext_org_pos']}. Keep stats & FAQs fresh."

        updates.append({
            "Priority": prio, "Topic": r["topic"], "Vertical": r["sector"],
            "7d Impr.": r["impressions_7d"], "Google Rank": r["google_rank_label"],
            "Existing Page": r["closest_existing_url"], "Action": act,
        })
        if not r["already_exists_in_sitemap"]:
            slug_base = re.sub(r"[^a-z0-9]+", "-", r["topic"].lower()).strip("-")
            new_ideas.append({
                "New Topic Gap": r["topic"], "Vertical": r["sector"],
                "7d Impr.": r["impressions_7d"], "30d Impr.": r["impressions_30d"],
                "Suggested URL": f"/blog/{slug_base}-guide",
                "Service Page to Link": r["recommended_money_page"],
                "Rising Queries to Cover": r["top_rising_queries"],
            })
    return pd.DataFrame(updates), (pd.DataFrame(new_ideas).sort_values("7d Impr.", ascending=False) if new_ideas else pd.DataFrame())


# ==============================================================================
# LOAD SITEMAP & STREAMLINED SIDEBAR
# ==============================================================================
sitemap_paths, sitemap_source_label = fetch_live_sitemap_paths()
live_set, live_df, live_vec, live_mat = _build_index_from_paths(sitemap_paths)

st.sidebar.image(CNXT_LOGO_WHITE, width=175)
st.sidebar.markdown("**Enterprise Search Radar**")
st.sidebar.caption(f"$0 Paid APIs • {sitemap_source_label}")
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
exclude_local_ph = st.sidebar.checkbox(
    "Filter Out Local PH / Student Noise",
    value=True,
    help="Removes local PH city directory and student/university searches so only B2B buyer demand appears.",
)
custom_negatives = st.sidebar.text_input(
    "Extra Excluded Words",
    placeholder="intern|sample|pdf",
    help="Optional: separate extra words to hide using a vertical pipe | (job-seeker words are already removed automatically).",
)

raw_df, harvested_neg_df = pd.DataFrame(), pd.DataFrame()
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
                raw_df, harvested_neg_df = fetch_gsc_api_data(dict(st.secrets["gcp_service_account"]), site_url, lookback, brand_regex, custom_negatives, exclude_local_ph)
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
                raw_df, harvested_neg_df = fetch_gsc_api_data(json.load(uploaded_json), site_url, lookback, brand_regex, custom_negatives, exclude_local_ph)
else:
    uploaded_csv = st.sidebar.file_uploader("Upload GSC Web CSV / ZIP", type=["csv", "zip"])
    brand_regex = DEFAULT_BRAND_REGEX
    if uploaded_csv:
        try:
            raw_df, harvested_neg_df = parse_gsc_web_upload(uploaded_csv, brand_regex, custom_negatives, exclude_local_ph)
        except Exception as err:
            st.sidebar.error(str(err))

st.sidebar.divider()
preset_mode = st.sidebar.selectbox(
    "Trend Sensitivity",
    ["Broad (All Verticals)", "Balanced", "Strict Breakouts"],
    index=0,
    help="'Broad' shows rising demand across all Connext verticals. 'Strict Breakouts' shows only major volume spikes.",
)
def_imp, def_up, def_lift, def_slope = (50, 4, 1.40, 10) if preset_mode.startswith("Strict") else ((30, 4, 1.30, 8) if preset_mode.startswith("Balanced") else (15, 3, 1.18, 4))

with st.sidebar.expander("ROI & Threshold Settings", expanded=False):
    default_cpc_est = st.number_input("Fallback B2B CPC ($)", 1.0, 100.0, 7.50, 0.50, help="Used to estimate Lost Traffic Value when a Keyword Planner CSV is not uploaded.")
    lead_conv_pct = st.slider("Lead Conv. Rate (%)", 1.0, 15.0, 4.0, 0.5, help="Estimated % of website visitors who submit a B2B lead form.")
    avg_deal_ltv = st.number_input("Avg Client Value ($)", 1000, 250000, 18000, 1000, help="Estimated B2B pipeline value per closed client.")
    min_daily_imp = st.number_input("Min Avg Daily Impr.", 5, 5000, def_imp, 5)
    min_up_days = st.slider("Weekly Upward Days", 2, 6, def_up)
    min_lift = st.slider("Min Growth Multiplier", 1.05, 3.0, def_lift, 0.05)
    min_slope_pct = st.slider("Min Daily Slope (%)", 1, 40, def_slope, 1)
    cluster_dist = st.slider("Topic Grouping Breadth", 0.30, 0.75, 0.55, 0.05)

has_bing_secret = "bing" in st.secrets and "api_key" in st.secrets["bing"] and bool(str(st.secrets["bing"]["api_key"]).strip())

# ==============================================================================
# HEADER, HERO BANNER & TOP QUICK-START GUIDE
# ==============================================================================
h_col1, h_col2 = st.columns([1.1, 4])
with h_col1:
    st.image(CNXT_LOGO_COLOR, width=180)
with h_col2:
    if not raw_df.empty:
        st.caption(f"Live GSC Window: **{raw_df['date'].min().strftime('%b %d')} – {raw_df['date'].max().strftime('%b %d, %Y')}** ({raw_df['date'].nunique()} complete days)  •  {sitemap_source_label}  •  100% Free Mode")

st.markdown(
    """<div class="cnxt-hero">
<div class="cnxt-eyebrow">CONNEXT GLOBAL ENTERPRISE SEARCH &amp; TOPIC INTELLIGENCE</div>
<h1>Search, Content &amp; Competitor <span>Trend Radar</span></h1>
<p>Real-time Google Search momentum, content decay early warnings, Brand vs. Non-Brand health, ROI opportunity, internal linking plans, live on-page audits, and on-demand competitor blog &amp; trend research.</p>
</div>""",
    unsafe_allow_html=True,
)

with st.expander("New here? 1-Minute Quick Start (How to Use This Tool & What to Expect)", expanded=False):
    qs1, qs2, qs3 = st.columns(3)
    with qs1:
        st.markdown(
            """**1. Start With the Filter Bar Below**
- Pick a **Business Vertical** (e.g., *Healthcare & RCM*, *Finance & Accounting*) or leave on **All Verticals**.
- Pick a **Recommended Next Step** (*Launch Paid Ads*, *Update Existing Page*, or *Striking Distance Quick Wins #4–#15*).
- Every chart and table across Tabs 1–4 updates instantly."""
        )
    with qs2:
        st.markdown(
            """**2. Pick the Tab for Your Role**
- **Leadership / Managers:** *Tab 1 (Executive Overview)* → copy the Teams Summary or download the Excel report.
- **Content / SEO Team:** *Tab 2 (Decay & On-Page Audit)* + *Tab 4 (Content, Linking & FAQs)* + *Tab 5 (Competitor Blogs)*.
- **Paid Ads Team:** *Tab 3 (Paid Ads, ROI & Negatives)* + *Tab 5 (US Ad Libraries)*."""
        )
    with qs3:
        st.markdown(
            """**3. What to Expect (Speed & Data)**
- Initial load takes **30–60 seconds** (live Google Search Console sync, cached for 1 hour).
- **Tabs 5 & 6** run **on-demand when you click their button** (~10–20s) so the dashboard stays fast.
- Need definitions? Open **Tab 7 (User Guide & Glossary)** for plain-English explanations of every metric."""
        )

if raw_df.empty:
    st.info("Connect your Google Search Console API or upload a GSC CSV in the sidebar to begin.")
    st.stop()

with st.spinner("Grouping queries into topics, checking sitemap coverage & calculating ROI opportunity..."):
    clustered_df = cluster_queries(raw_df, cluster_dist)
    trends_df, decay_df, daily_topic_df, all_queries_df, pages_df = analyze_trends_and_decay(
        clustered_df, live_set, live_df, live_vec, live_mat,
        min_daily_imp, min_up_days, min_slope_pct / 100.0, min_lift,
    )

if trends_df.empty:
    st.warning("No surging topics matched your sensitivity filter. Select **Broad (All Verticals)** in the sidebar.")
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
sel_funnel = fc3.selectbox("Funnel Stage", ["All Funnel Stages", "BOFU — Ready to Buy", "MOFU — Evaluating", "TOFU — Research"])
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
ads_topics_df, rec_kws_df = build_ads_and_roi_tables(f_trends, f_queries, kp_lookup_dict, default_cpc_est, lead_conv_pct / 100.0, float(avg_deal_ltv))
existing_content_df, truly_new_df = build_content_tables(f_trends)
internal_links_df = build_internal_linking_plan(f_trends, clustered_df)

# --- TOP KPI ROW ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Surging Topics", len(f_trends), help="Number of B2B topic groups gaining search visibility on Google right now.")
k2.metric("Content Decay Alerts", len(f_decay), help="Topics where 7-day impressions dropped >= 22% or Google rank slipped >= 2.5 spots.")
k3.metric("Weekly Impressions", f"{int(f_trends['impressions_7d'].sum()):,}", help="Total times connextglobal.com appeared on Google for these surging topics in the last 7 days.")
k4.metric("Est. Missed Clicks/Mo", f"{f_trends['est_missed_clicks_30d'].sum():,.0f}", help="Extra monthly clicks Connext could win by ranking in Google's Top 3 for these topics.")
k5.metric("Lost Traffic Value/Mo", f"${ads_topics_df['Lost Traffic Value/Mo'].sum():,.0f}", help="Estimated Google Ads spend required to buy those missed organic clicks.")

# --- 7 ENTERPRISE TABS (INCLUDING COMPLETE USER GUIDE & GLOSSARY) ---
t_over, t_pages, t_ads, t_content, t_comp, t_trends, t_guide = st.tabs([
    "1. Executive Overview",
    f"2. Pages, Decay & Audit ({len(f_decay)} Decay)",
    f"3. Paid Ads & ROI ({len(rec_kws_df)})",
    f"4. Content & Linking ({len(truly_new_df)} New)",
    "5. Competitor Blogs & Ads",
    "6. Trends, Bing & Research",
    "7. User Guide & Glossary",
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
with t_over:
    render_tab_guide(
        "Executive Overview",
        "A high-level radar of which B2B topics are growing fastest on Google, how demand splits across Connext verticals, and whether growth is coming from Brand awareness or Non-Brand acquisition.",
        "Ranked bar charts, a 0–100 Priority Score per topic, daily trend curves, and a ready-to-download multi-sheet Excel report (.xlsx).",
        "Start weekly team meetings here: review the top 5 orange/blue bars, copy the 1-Click Teams Executive Summary at the bottom, or export the Excel workbook to share with stakeholders.",
    )

    top_bar1, top_bar2 = st.columns([2.5, 1])
    with top_bar1:
        horizon = st.radio("Time Horizon:", ["All Surging", "Today (3-Day Spike)", "This Week (7-Day)", "This Month (30-Day)"], horizontal=True)
    with top_bar2:
        exec_summary_df = pd.DataFrame([
            {"Metric": "Report Date (UTC)", "Value": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
            {"Metric": "Surging B2B Topics", "Value": len(f_trends)},
            {"Metric": "Content Decay Alerts", "Value": len(f_decay)},
            {"Metric": "Weekly Surging Impressions", "Value": int(f_trends["impressions_7d"].sum())},
            {"Metric": "Avg Google Rank", "Value": round(float(f_trends["connext_org_pos"].mean()), 1)},
            {"Metric": "Est. Missed Clicks / Month", "Value": round(float(f_trends["est_missed_clicks_30d"].sum()), 0)},
            {"Metric": "Lost Traffic Ad Value / Month ($)", "Value": round(float(ads_topics_df["Lost Traffic Value/Mo"].sum()), 0)},
        ])
        excel_bytes = build_enterprise_excel_bytes({
            "Executive Summary": exec_summary_df,
            "Surging Topics": f_trends[["topic", "sector", "funnel", "priority_score", "team_action", "impressions_7d", "impressions_30d", "google_rank_label", "est_missed_clicks_30d", "recommended_money_page"]],
            "Content Decay Alerts": f_decay,
            "Paid Ads & ROI": ads_topics_df,
            "Suggested Keywords": rec_kws_df.head(500),
            "Content Plan": existing_content_df,
            "New Content Gaps": truly_new_df,
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
                custom_data=["google_rank_label", "est_missed_clicks_30d", "funnel"],
                labels={"impressions_7d": "7-Day Impressions", "topic": "", "team_action": ""},
            )
            fig_t.update_traces(texttemplate="+%{text}%", textposition="outside", hovertemplate="<b>%{y}</b><br>7d Impr: <b>%{x:,}</b><br>Rank: <b>%{customdata[0]}</b><br>Funnel: %{customdata[2]}<br>Missed Clicks/Mo: <b>%{customdata[1]:,.0f}</b><extra></extra>")
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
            fig_f = px.pie(f_agg, names="funnel", values="impr", hole=0.52, color="funnel", color_discrete_map={"BOFU — Ready to Buy": CNXT_COLORS["sun"], "MOFU — Evaluating": CNXT_COLORS["blue"], "TOFU — Research": CNXT_COLORS["teal"]})
            fig_f.update_layout(margin=dict(l=10, r=10, t=10, b=35))
            st.plotly_chart(style_plotly_light(fig_f, height=260, legend_bottom=True), use_container_width=True)

        st.markdown("#### Daily Search Trend Inspector")
        sel_t = st.selectbox("Select a topic to inspect its daily curve:", vdf["topic"].tolist(), key="sel_ov_topic")
        sr = vdf[vdf["topic"] == sel_t].iloc[0]
        st.markdown(
            f"""<div class="cnxt-callout">
<strong>{sr['topic'].title()}</strong> ({sr['sector']} • {sr['funnel']}) — Priority <strong>{sr['priority_score']}/100</strong> • <strong>{sr['impressions_7d']:,}</strong> weekly impressions (<strong>+{sr[g_col]}%</strong> {g_lbl}) • Currently <strong>{sr['google_rank_label']}</strong> • Est. <strong>{sr['est_missed_clicks_30d']:,.0f} missed clicks/mo</strong> • Next Step: <strong>{sr['team_action']}</strong> → <code>{sr['recommended_money_page']}</code>
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
        ov_tbl = vdf[["priority_score", "topic", "sector", "funnel", "team_action", "sparkline_7d", "impressions_7d", g_col, "google_rank_label", "est_missed_clicks_30d", "recommended_money_page"]].copy()
        ov_tbl["growth_lbl"] = ov_tbl[g_col].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
        st.dataframe(
            ov_tbl[["priority_score", "topic", "sector", "funnel", "team_action", "sparkline_7d", "impressions_7d", "growth_lbl", "google_rank_label", "est_missed_clicks_30d", "recommended_money_page"]],
            use_container_width=True, hide_index=True, key="tbl_overview",
            column_config={
                "priority_score": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%d"),
                "topic": "Topic", "sector": "Vertical", "funnel": "Funnel", "team_action": "Next Step",
                "sparkline_7d": st.column_config.LineChartColumn("7d Trend", width="small"),
                "impressions_7d": st.column_config.NumberColumn("7d Impr."), "growth_lbl": g_lbl,
                "google_rank_label": "Google Rank", "est_missed_clicks_30d": st.column_config.NumberColumn("Missed Clicks/Mo", format="%.0f"),
                "recommended_money_page": st.column_config.LinkColumn("Target Service Page"),
            },
        )

        with st.expander("1-Click Teams Executive Summary (Copy & Paste)", expanded=False):
            lines = [f"{i+1}. **{r['topic'].title()}** ({r['sector']}, {r['funnel']}) — {r['impressions_7d']:,} weekly impr. (+{r[g_col]}%), Rank {r['google_rank_label']}, ~{r['est_missed_clicks_30d']:,.0f} missed clicks/mo → **{r['team_action']}**" for i, r in vdf.head(5).iterrows()]
            decay_note = f"\n• Content Decay Alerts Requiring Refresh: **{len(f_decay)}** topics" if not f_decay.empty else ""
            st.code(f"Connext Search Trend Summary ({datetime.now(timezone.utc).strftime('%b %d, %Y')}){decay_note}\n\n" + "\n".join(lines), language="markdown")

# ==============================================================================
# TAB 2: PAGES, CONTENT DECAY, CANNIBALIZATION & LIVE ON-PAGE AUDIT
# ==============================================================================
with t_pages:
    render_tab_guide(
        "Pages, Decay & Live On-Page Audit",
        "Shows (1) topics losing visibility/rank (Content Decay), (2) topics where 2+ Connext pages compete against each other (Cannibalization), (3) exact user searches, and (4) a live On-Page SEO Grader.",
        "Actionable decay alerts, exact URLs ranking on Google, and a 0–100 SEO/AI-readiness checklist when you click 'Run Live On-Page Audit'.",
        "Fix Decay pages on Page 1 first to protect current leads. Then select any topic, run the Live On-Page Audit on its URL, and send any 'Fix/Missing' items (Title, H1, FAQ schema) to Content or Webflow dev.",
    )

    if not f_decay.empty:
        st.markdown(f"##### Content Decay Early Warning ({len(f_decay)} Slipping Topics)")
        st.dataframe(f_decay, use_container_width=True, hide_index=True, key="tbl_decay", column_config={"Affected Page": st.column_config.LinkColumn("Affected Page")})
    else:
        st.success("No major content decay alerts detected in this vertical over the last 7 days.")

    cann_df = f_trends[f_trends["is_cannibalized"]]
    if not cann_df.empty:
        with st.expander(f"Cannibalization Check: {len(cann_df)} Surging Topics Have 2+ Competing Connext URLs", expanded=False):
            st.dataframe(cann_df[["topic", "sector", "ranking_pages_count", "google_rank_label", "impressions_7d", "primary_page", "closest_existing_url"]], use_container_width=True, hide_index=True, key="tbl_cann", column_config={"primary_page": st.column_config.LinkColumn("Top Ranking URL"), "closest_existing_url": st.column_config.LinkColumn("Main Sitemap URL")})

    st.divider()
    sel_c = st.selectbox("Inspect Ranking Pages, Exact Queries & Run Live On-Page Audit:", f_trends["topic"].tolist(), key="sel_pages_topic")
    tm = f_trends[f_trends["topic"] == sel_c].iloc[0]

    p1, p2 = st.columns([1.2, 1])
    with p1:
        st.markdown("**Connext Pages Currently Ranking on Google**")
        tp = f_pages[f_pages["topic"] == sel_c].sort_values("page_imp_7d", ascending=False).copy()
        tp["rank_lbl"] = tp["page_org_pos"].apply(human_rank_label)
        st.dataframe(tp[["norm_page", "page_role", "page_imp_7d", "topic_imp_share_%", "rank_lbl"]], use_container_width=True, hide_index=True, key="tbl_tp", column_config={"norm_page": st.column_config.LinkColumn("Ranking URL"), "page_imp_7d": "7d Impr.", "topic_imp_share_%": st.column_config.NumberColumn("Share", format="%.0f%%"), "rank_lbl": "Rank"})
    with p2:
        st.markdown("**Live Matching Pages in `sitemap.xml`**")
        recs, seen = [], set()
        for lbl, src in [("Covers Topic", tm["existing_sitemap_matches"]), ("Service Page", tm["matching_money_pages"]), ("Related Guide", tm["matching_content_pages"])]:
            for item in src:
                if item["full_url"] not in seen:
                    recs.append({"URL": item["full_url"], "Type": item["role"], "Role": lbl})
                    seen.add(item["full_url"])
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True, key="tbl_sm_m", column_config={"URL": st.column_config.LinkColumn("Live Connext URL")})

    st.markdown("**Live On-Page SEO & AI-Readiness Grader ($0 On-Demand)**")
    audit_col1, audit_col2 = st.columns([2.2, 1])
    audit_url = audit_col1.text_input("URL to Audit:", value=tm["primary_page"] or tm["recommended_money_page"], key="in_audit_url")
    if audit_col2.button("Run Live On-Page Audit", type="primary", key="btn_run_audit"):
        with st.spinner(f"Auditing {audit_url} (~5 seconds)..."):
            st.session_state["onpage_audit_res"] = run_live_onpage_seo_audit(audit_url.strip(), tuple(tm["queries_list"][:3]))
    if "onpage_audit_res" in st.session_state:
        ares = st.session_state["onpage_audit_res"]
        if "error" in ares:
            st.warning(ares["error"])
        else:
            st.metric("On-Page SEO & AI-Readiness Score", f"{ares['score']} / 100")
            st.dataframe(ares["checks"], use_container_width=True, hide_index=True, key="tbl_onpage_checks")

    st.markdown("**Exact Search Queries & AI Prompts (7+ Words)**")
    tqs = f_queries[f_queries["topic"] == sel_c].sort_values("imp_7d", ascending=False).copy()
    tqs["growth"] = tqs["query_growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
    tqs["rank"] = tqs["connext_org_pos"].apply(human_rank_label)
    st.dataframe(tqs[["query", "is_ai_prompt", "imp_7d", "imp_30d", "growth", "rank", "top_page"]], use_container_width=True, hide_index=True, key="tbl_tqs", column_config={"is_ai_prompt": st.column_config.CheckboxColumn("AI Prompt"), "top_page": st.column_config.LinkColumn("Ranking URL")})

# ==============================================================================
# TAB 3: PAID ADS, ROI, DAY-OF-WEEK SCHEDULER & LIVE NEGATIVE HARVESTER
# ==============================================================================
with t_ads:
    render_tab_guide(
        "Paid Ads, ROI & Negative Keywords",
        "Shows which rising topics make the best Google & Bing Ads campaigns, how much traffic/pipeline value we miss below Top 3, peak B2B search days, and real wasted non-B2B searches.",
        "Every suggested keyword comes with real 7-day & 30-day organic impressions, estimated CPC/Lost Value, Exact/Phrase lists, Negative Keywords, and 30/90-char RSA ad copy.",
        "1) Focus on 'P0' and 'BOFU' topics where Connext ranks below #5. 2) Export the .txt list into Google Keyword Planner if you want exact Google CPC bids and upload the CSV back here. 3) Copy Exact/Phrase keywords + Real GSC Negatives straight into Google/Bing Ads.",
    )

    with st.expander("Optional: Upload Free Google Keyword Planner CSV to Lock In Exact Volume & CPC Bids", expanded=False):
        kp_up = st.file_uploader("Upload CSV/XLSX exported from Google Keyword Planner ('Get search volume and forecasts')", type=["csv", "xlsx", "xls"], key="inline_kp_upload")
        if kp_up:
            try:
                parsed_kp = parse_keyword_planner_csv(kp_up)
                st.session_state["kp_lookup_dict"] = parsed_kp
                st.success(f"Synced {len(parsed_kp):,} Keyword Planner metrics! Recalculating ROI...")
                st.rerun()
            except Exception as err:
                st.error(str(err))

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Current Est. Clicks/Mo", f"{f_trends['est_cur_clicks_30d'].sum():,.0f}")
    o2.metric("Top-3 Potential Clicks/Mo", f"{f_trends['est_top3_clicks_30d'].sum():,.0f}")
    o3.metric("Lost Ad Value/Mo", f"${ads_topics_df['Lost Traffic Value/Mo'].sum():,.0f}")
    o4.metric("Est. Pipeline Opp./Mo", f"${ads_topics_df['Pipeline Opp./Mo'].sum():,.0f}")

    st.dataframe(
        ads_topics_df, use_container_width=True, hide_index=True, key="tbl_ads_t",
        column_config={
            "CPC ($)": st.column_config.NumberColumn("CPC ($)", format="$%.2f"),
            "Lost Traffic Value/Mo": st.column_config.NumberColumn("Lost Value/Mo", format="$%,.0f"),
            "Pipeline Opp./Mo": st.column_config.NumberColumn("Pipeline Opp./Mo", format="$%,.0f"),
            "Landing Page": st.column_config.LinkColumn("Landing Page"),
        },
    )

    st.divider()
    ad_e1, ad_e2 = st.columns([1.2, 1])
    with ad_e1:
        st.markdown("#### Day-of-Week B2B Search Rhythm (Ad Scheduling Guide)")
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
        kw_topics = kf1.multiselect("Filter by Topic:", sorted(rec_kws_df["Topic"].unique().tolist()), default=sorted(rec_kws_df["Topic"].unique().tolist()), key="ms_kw_t")
        min_k_imp = kf2.number_input("Min 7-Day Impressions:", 1, 10000, 5, 5, key="num_min_kw")
        scoped_kws = rec_kws_df[(rec_kws_df["Topic"].isin(kw_topics)) & (rec_kws_df["7d Impr."] >= min_k_imp)]
        st.dataframe(scoped_kws[["Keyword", "7d Impr.", "30d Impr.", "KP Monthly Vol.", "Growth", "Google Rank", "Missed Clicks/Mo", "CPC ($)", "Lost Value/Mo", "Match Type", "Landing Page"]], use_container_width=True, hide_index=True, key="tbl_ads_k", column_config={"CPC ($)": st.column_config.NumberColumn("CPC", format="$%.2f"), "Lost Value/Mo": st.column_config.NumberColumn("Lost $/Mo", format="$%,.0f"), "Landing Page": st.column_config.LinkColumn("Landing Page")})

        d1, d2 = st.columns(2)
        d1.download_button("Download Keywords & Opportunity CSV", scoped_kws.to_csv(index=False).encode("utf-8"), file_name="connext_paid_keywords.csv", mime="text/csv", key="dl_pk")
        d2.download_button("Export Clean List for Google Keyword Planner (.txt)", "\n".join(scoped_kws["Keyword"].head(500).tolist()).encode("utf-8"), file_name="keywords_for_keyword_planner.txt", mime="text/plain", key="dl_kp")

        with st.expander("Ad Group Copy-Paste Builder (Exact, Phrase, Real GSC Negatives & RSA Headlines)", expanded=False):
            sel_ad = st.selectbox("Pick Topic:", ads_topics_df["Topic"].tolist(), key="sel_ad_grp")
            ad_row = f_trends[f_trends["topic"] == sel_ad].iloc[0]
            kws = rec_kws_df[rec_kws_df["Topic"] == sel_ad]["Keyword"].tolist() or ad_row["queries_list"]
            st.caption(f"Target Landing Page: **{ad_row['recommended_money_page']}**")
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
# TAB 4: CONTENT PLAN, INTERNAL LINKING ARCHITECT & FAQ SCHEMA
# ==============================================================================
with t_content:
    render_tab_guide(
        "Content Plan, Internal Linking & FAQ Schema",
        "Separates surging topics into (1) existing pages to update, (2) internal links to add from high-traffic blogs to Service Pages, (3) genuinely new content gaps, and (4) Writer Briefs + Webflow FAQ Schema.",
        "Zero duplicate content suggestions—every topic is verified against connextglobal.com/sitemap.xml before recommending a new article.",
        "Assign Section 1 updates to writers first (fastest ROI), add the Section 2 internal links in Webflow to boost Page-2 Service Pages, and copy-paste the Section 4 FAQ JSON-LD into Webflow Page Settings → Custom Code.",
    )

    st.markdown("#### 1. Update Existing Sitemap Pages (Prevent Duplicate Content)")
    st.dataframe(existing_content_df, use_container_width=True, hide_index=True, key="tbl_ex_c", column_config={"Existing Page": st.column_config.LinkColumn("Existing Sitemap Page")})
    st.download_button("Download Existing Page Updates CSV", existing_content_df.to_csv(index=False).encode("utf-8"), file_name="connext_content_updates.csv", mime="text/csv", key="dl_ex_c")

    st.divider()
    st.markdown(f"#### 2. Automated Internal Linking Architect ({len(internal_links_df)} High-Impact Links)")
    if internal_links_df.empty:
        st.info("No internal link recommendations needed under current filters.")
    else:
        st.dataframe(
            internal_links_df, use_container_width=True, hide_index=True, key="tbl_int_links",
            column_config={"Source Blog / Guide (Add Link Here)": st.column_config.LinkColumn("Source Blog"), "Target Service / Industry Page": st.column_config.LinkColumn("Target Money Page")},
        )
        st.download_button("Download Internal Linking Plan CSV", internal_links_df.to_csv(index=False).encode("utf-8"), file_name="connext_internal_linking_plan.csv", mime="text/csv", key="dl_int_links")

    st.divider()
    st.markdown(f"#### 3. Truly New Content Ideas — Verified Not in Sitemap ({len(truly_new_df)})")
    if truly_new_df.empty:
        st.success("All currently surging topics already have a matching page in connextglobal.com/sitemap.xml!")
    else:
        st.dataframe(truly_new_df, use_container_width=True, hide_index=True, key="tbl_new_c", column_config={"Service Page to Link": st.column_config.LinkColumn("Service Page")})

    st.divider()
    st.markdown("#### 4. Writer Brief & Webflow `FAQPage` JSON-LD Generator")
    sel_b = st.selectbox("Select Topic:", existing_content_df["Topic"].tolist(), key="sel_brief")
    br = f_trends[f_trends["topic"] == sel_b].iloc[0]
    lt_b = "\n".join(f"  - `{q}`" for q in (br["long_tail_list"] or br["queries_list"]))
    bc1, bc2 = st.columns([1.1, 1])
    with bc1:
        st.markdown(
            f"""**Brief: {br['topic'].title()}** ({br['sector']} • {br['funnel']})
- **Action:** {'Update `' + br['closest_existing_url'] + '`' if br['already_exists_in_sitemap'] else 'Create New Article (Not in Sitemap)'}
- **Rank & Opportunity:** {br['google_rank_label']} • **{br['impressions_7d']:,}** 7d impr. • ~**{br['est_missed_clicks_30d']:,.0f}** missed clicks/mo
- **Main Service Page to Link:** `{br['recommended_money_page']}`
- **Exact Rising Queries to Answer:**
{lt_b}"""
        )
    with bc2:
        st.markdown("**Ready-to-Paste `FAQPage` JSON-LD Schema**")
        st.code(build_faq_jsonld(br["topic"], br["queries_list"]), language="html")

# ==============================================================================
# TAB 5: ON-DEMAND COMPETITOR NEW BLOGS (+ KEYWORDS) & FREE ADS LIBRARIES
# ==============================================================================
with t_comp:
    render_tab_guide(
        "Competitor New Blogs & US Ad Libraries",
        "Monitors our 12 direct BPO competitors (Solvo, Genius, Microsourcing, Emapta, Outsourced, OP360, Lean Solutions, TaskUs, TOA Global, Cloudstaff, Near, BairesDev) strictly for newly published blog articles and provides 1-click links to their live US Google & Microsoft ads.",
        "Clicking 'Scan 12 Competitors' takes ~15–20 seconds and returns each rival's newest blog headline, auto-extracted Main Target Keywords, and whether Connext already covers that topic. Note: a few sites block automated bots ('blocked')—this is normal.",
        "Filter to 'Content Gaps Only' to spot topics rivals are aggressively writing about that Connext doesn't cover yet. Use Sub-Tab 2 to inspect competitor US ad headlines and offers before launching campaigns.",
    )

    c_sub1, c_sub2 = st.tabs([
        "New Competitor Blogs & Main Keywords ($0)",
        "US Google Ads Transparency & Microsoft Ad Library ($0)",
    ])

    with c_sub1:
        b_top1, b_top2, b_top3 = st.columns([2.2, 1.1, 0.9])
        blog_days = b_top1.slider("Competitor Blogs Published / Updated Within (Days):", 14, 90, 45, key="sl_blog_days")
        if b_top2.button("Scan 12 Competitors' New Blogs Now", type="primary", key="btn_scan_blogs"):
            with st.spinner("Scanning all 12 competitor sitemaps strictly for new blogs & extracting main keywords (~15–20s)..."):
                bdf, bnotes = scan_competitor_new_blogs_auto(lookback_days=blog_days)
                if not bdf.empty:
                    cov_l, cls_l = [], []
                    for _, row in bdf.iterrows():
                        chk = check_topic_against_sitemap(f"{row['Blog Headline']} {row['Main Target Keywords']}", "", live_set, live_df, live_vec, live_mat)
                        cov_l.append("Connext Covers" if chk["already_exists_in_sitemap"] else "Content Gap")
                        cls_l.append(chk["closest_existing_url"])
                    bdf["Connext Status"] = cov_l
                    bdf["Connext Closest Page"] = cls_l
                st.session_state["comp_blogs_df"] = bdf
                st.session_state["comp_blog_notes"] = bnotes
        if b_top3.button("Clear Cache", key="btn_clr_blogs"):
            scan_competitor_new_blogs_auto.clear()
            st.session_state.pop("comp_blogs_df", None)

        if "comp_blog_notes" in st.session_state:
            st.caption("Blog feed status: " + " • ".join(st.session_state["comp_blog_notes"]))

        comp_blogs_df = st.session_state.get("comp_blogs_df", pd.DataFrame())
        if comp_blogs_df.empty:
            st.info("Click **Scan 12 Competitors' New Blogs Now** above to pull strictly new editorial blog posts and their extracted target keywords (cached 24 hours).")
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
                bview[["Competitor", "Published", "Blog Headline", "Main Target Keywords", "Vertical", "Connext Status", "Connext Closest Page", "Blog URL"]],
                use_container_width=True, hide_index=True, key="tbl_comp_blogs",
                column_config={
                    "Blog Headline": st.column_config.TextColumn("Blog Headline", width="large"),
                    "Main Target Keywords": st.column_config.TextColumn("Main Target Keywords", width="medium"),
                    "Connext Closest Page": st.column_config.LinkColumn("Connext Closest Page"),
                    "Blog URL": st.column_config.LinkColumn("Read Competitor Blog"),
                },
            )
            st.download_button("Download Competitor New Blogs & Keywords CSV", bview[["Competitor", "Published", "Blog Headline", "Main Target Keywords", "Vertical", "Connext Status", "Connext Closest Page", "Blog URL"]].to_csv(index=False).encode("utf-8"), file_name="connext_competitor_new_blogs_keywords.csv", mime="text/csv", key="dl_comp_blogs")

    with c_sub2:
        st.markdown("""<div class="cnxt-callout"><strong>Free US Ad Transparency Lookup:</strong> Click any competitor's bare-domain link below to see their active US Google Search/Display ads, or open Microsoft Ad Library and search their brand name. Zero API credits used.</div>""", unsafe_allow_html=True)
        dir_df = pd.DataFrame([
            {"Competitor": cname, "Domain": dom, "Google Ads Transparency (US)": f"https://adstransparency.google.com/?region=US&domain={dom}", "Microsoft Ad Library": "https://adlibrary.ads.microsoft.com/"}
            for cname, dom, _ in COMPETITORS_REGISTRY
        ])
        st.dataframe(dir_df, use_container_width=True, hide_index=True, key="tbl_ad_dir", column_config={"Google Ads Transparency (US)": st.column_config.LinkColumn("Open Google Ads Transparency (US)"), "Microsoft Ad Library": st.column_config.LinkColumn("Open Microsoft Ad Library")})

        st.divider()
        if st.button("Scan 12 Competitors via Free Microsoft Ad Library API (EU/EEA Spillover)", key="btn_ms_api"):
            with st.spinner("Querying Microsoft Ad Library API (~10s)..."):
                st.session_state["ms_api_df"] = fetch_free_microsoft_ad_library_all()
        if "ms_api_df" in st.session_state:
            ms_df = st.session_state["ms_api_df"]
            if ms_df.empty:
                st.info("No EU/EEA Microsoft Ads returned (use the direct US Ad Library links above for US-only campaigns).")
            else:
                st.dataframe(ms_df, use_container_width=True, hide_index=True, key="tbl_ms_ads", column_config={"Ad Library Link": st.column_config.LinkColumn("Open Ad")})

# ==============================================================================
# TAB 6: ON-DEMAND INDUSTRY-FILTERED TRENDS, BING & TOPIC RESEARCH ($0)
# ==============================================================================
with t_trends:
    render_tab_guide(
        "Industry Trends, Bing & Buyer Research",
        "External $0 research feeds strictly scoped to Connext's B2B industry: Google Trends curves + B2B industry news headlines, Bing Webmaster performance (also powers Copilot/ChatGPT Search), and Google/Bing Autocomplete + Reddit buyer questions.",
        "Runs on demand (~10 seconds per click). Google Trends may occasionally show a brief cloud rate-limit notice, while the B2B Industry Feed, Autocomplete, and Reddit remain available.",
        "Use Sub-Tab 1 to spot emerging industry shifts before they hit Search Console. Use Sub-Tab 3 to pull real customer questions and objections from Google, Bing, and Reddit for blog H2s and ad hooks.",
    )

    r_sub1, r_sub2, r_sub3 = st.tabs(["B2B Google Trends & Industry Feed", "Bing Search Performance", "Question & Reddit Pain-Point Finder"])

    with r_sub1:
        tc1, tc2, tc3 = st.columns([2.2, 0.9, 0.9])
        gt_seeds = tc1.text_input("Compare up to 5 B2B Topics in Google Trends:", value="medical billing outsourcing, offshore accounting, virtual assistant outsourcing, nearshore software developers, bpo outsourcing", key="gt_in")
        gt_geo = tc2.selectbox("Region:", ["US", "CA", "AU", "GB"], index=0, key="gt_geo")
        gt_tf = tc3.selectbox("Window:", ["today 1-m", "today 3-m", "today 12-m"], index=0, key="gt_tf")

        news_vert = st.selectbox("B2B Industry News Feed (Last 14 Days):", list(INDUSTRY_NEWS_FEEDS.keys()), index=0, key="sel_news_vert")
        if st.button("Load B2B Google Trends & Industry Headlines", type="primary", key="btn_load_gt"):
            seed_tup = tuple(s.strip().lower() for s in gt_seeds.split(",") if s.strip())[:5]
            with st.spinner("Fetching B2B Google Trends & industry headlines (~10s)..."):
                iot_d, ris_d, g_n = fetch_free_google_trends_b2b(seed_tup, geo=gt_geo, timeframe=gt_tf)
                if not ris_d.empty:
                    sm_s, sm_l = [], []
                    for _, rr in ris_d.iterrows():
                        chk = check_topic_against_sitemap(rr["Rising B2B Query"], "", live_set, live_df, live_vec, live_mat)
                        sm_s.append("Already Covered" if chk["already_exists_in_sitemap"] else "New Topic Gap")
                        sm_l.append(chk["best_money_url"] if chk["has_strong_money_match"] else chk["closest_existing_url"])
                    ris_d["Connext Status"] = sm_s
                    ris_d["Target Page"] = sm_l
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
            st.dataframe(ris_df, use_container_width=True, hide_index=True, key="tbl_gt_r", column_config={"Target Page": st.column_config.LinkColumn("Target Page")})

        b2b_news_df = st.session_state.get("b2b_news_df", pd.DataFrame())
        if not b2b_news_df.empty:
            st.markdown("**Live B2B Industry Trend Headlines (Strictly Industry-Filtered)**")
            st.dataframe(b2b_news_df, use_container_width=True, hide_index=True, key="tbl_b2b_news", column_config={"Article Link": st.column_config.LinkColumn("Read Article")})

    with r_sub2:
        if has_bing_secret:
            if st.button("Load / Refresh Bing Webmaster Queries", type="primary", key="btn_ref_bing"):
                with st.spinner("Syncing Bing Webmaster Tools (~8s)..."):
                    bdf, b_stat = fetch_free_bing_webmaster_queries(str(st.secrets["bing"]["api_key"]), st.secrets["bing"].get("site_url", "https://connextglobal.com/"), brand_regex, exclude_local_ph, custom_negatives)
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

    with r_sub3:
        rc1, rc2, rc3 = st.columns([1.2, 1.4, 0.8])
        picked_seed = rc1.selectbox("Pick Surging GSC Topic:", f_trends["topic"].tolist(), key="sel_res_seed")
        custom_seed = rc2.text_input("Or Type Custom B2B Topic:", placeholder="e.g., dental billing outsourcing", key="in_res_seed")
        active_seed = custom_seed.strip() or picked_seed
        if rc3.button("Mine Questions & Reddit", type="primary", key="btn_mine_res"):
            with st.spinner(f"Mining Google/Bing Autocomplete & Reddit for '{active_seed}' (~10s)..."):
                s_df = fetch_free_google_and_bing_suggestions(active_seed)
                if not s_df.empty:
                    s_df["Connext Status"] = ["Already Covered" if check_topic_against_sitemap(q, "", live_set, live_df, live_vec, live_mat)["already_exists_in_sitemap"] else "New Gap" for q in s_df["Search Suggestion"]]
                st.session_state["res_sugg_df"] = s_df
                st.session_state["res_red_df"] = fetch_free_reddit_b2b_discussions(active_seed)
                st.session_state["res_seed_lbl"] = active_seed

        if "res_seed_lbl" in st.session_state:
            col_sugg, col_red = st.columns(2)
            with col_sugg:
                st.markdown(f"**Google & Bing Autocomplete for `{st.session_state['res_seed_lbl']}`**")
                sdf = st.session_state.get("res_sugg_df", pd.DataFrame())
                if sdf.empty:
                    st.info("No autocomplete suggestions returned.")
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
    st.markdown('<p class="cnxt-sec-sub">Everything you need to know about how this radar works, what each number means, and how to turn insights into leads and traffic.</p>', unsafe_allow_html=True)

    g_row1_c1, g_row1_c2 = st.columns(2)
    with g_row1_c1:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>1. What to Expect (Speed, Data &amp; Freshness)</h3>
<ul>
<li><strong>Initial Login Load (30–60 seconds):</strong> After unlocking, the app pulls up to 45–90 days of live Google Search Console rows, filters out job-seeker/applicant searches, groups queries into topics, and indexes <code>connextglobal.com/sitemap.xml</code>.</li>
<li><strong>Instant Filtering After Load:</strong> Once loaded, data is cached for 1 hour so switching verticals, funnel stages, or tabs happens instantly.</li>
<li><strong>On-Demand Tabs (Tabs 5 &amp; 6):</strong> Competitor Blog Scans, On-Page Audits, Google Trends, Bing, and Reddit run only when you click their button (typically 10–20 seconds) so the main dashboard stays fast.</li>
<li><strong>2-Day Google Reporting Lag:</strong> Google Search Console data normally finalizes 48 hours behind today. This tool automatically trims partial/incomplete days so charts never show fake sudden drops.</li>
<li><strong>100% Free &amp; Read-Only:</strong> Nothing in this tool spends paid API credits or modifies Connext's live website, Google Ads, or Search Console settings.</li>
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
<li><strong>Step 2 — Spot Priorities (Tab 1):</strong> Sort by <strong>Priority (0–100)</strong> or <strong>Missed Clicks/Mo</strong>. Click any topic in the Trend Inspector to verify whether demand is steadily growing or a one-day spike.</li>
<li><strong>Step 3 — Protect Existing Rankings (Tab 2):</strong> Review <strong>Content Decay Alerts</strong> and run the <strong>Live On-Page Audit</strong> on slipping Page-1 URLs.</li>
<li><strong>Step 4 — Take Action (Tabs 3 &amp; 4):</strong> Export ready-to-bid keywords &amp; negative keywords for Paid Ads, or grab Writer Briefs, Internal Links, and FAQ Schema for Content.</li>
<li><strong>Step 5 — Check Competitors &amp; Market (Tabs 5 &amp; 6):</strong> Scan our 12 rivals' newest blogs for missing topics and check live US Ads Transparency links.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### 3. Plain-English Metrics Glossary (No SEO Jargon)")
    glossary_df = pd.DataFrame([
        {"Metric / Term": "Impressions (7d / 30d)", "Plain-English Meaning": "How many times a connextglobal.com page appeared on someone's Google screen. Rising impressions = growing buyer demand, even before they click."},
        {"Metric / Term": "Google Rank (Top 3 / Page 1 / Page 2)", "Plain-English Meaning": "Where Connext appears organically (unpaid). Top 3 (#1–3) gets the most clicks; #4–#15 ('Striking Distance') are quick wins; Page 2+ (#11+) rarely gets organic clicks without Ads or a page update."},
        {"Metric / Term": "3d / 7d / 30d Growth (%)", "Plain-English Meaning": "How much faster daily impressions grew compared to the prior 2-week baseline. Example: +100% means daily visibility doubled."},
        {"Metric / Term": "Priority Score (0–100)", "Plain-English Meaning": "Combines buyer intent (BOFU/MOFU/TOFU), search volume, surge speed, and rank gap into a single score so you know what matters most first."},
        {"Metric / Term": "Funnel Stage (BOFU / MOFU / TOFU)", "Plain-English Meaning": "BOFU = Ready to buy ('pricing', 'hire', 'outsourcing company'); MOFU = Comparing options ('best', 'vs', 'dedicated team'); TOFU = Early research ('how to', 'what is')."},
        {"Metric / Term": "Content Decay Alert", "Plain-English Meaning": "Early warning for topics where Connext lost >=22% of impressions or slipped >=2.5 rank spots. Refreshing these pages protects existing organic leads."},
        {"Metric / Term": "Missed Clicks/Mo & Lost Traffic Value ($)", "Plain-English Meaning": "Estimated extra monthly clicks we would win by moving a topic into Google's Top 3, multiplied by CPC to show what those clicks would cost in Google Ads."},
        {"Metric / Term": "Sitemap Check (Already in Sitemap vs. New Gap)", "Plain-English Meaning": "Checks connextglobal.com/sitemap.xml automatically. If 'Already in Sitemap', update that existing URL instead of writing a duplicate blog post."},
        {"Metric / Term": "Cannibalization (2+ URLs)", "Plain-English Meaning": "When two or more Connext pages try to rank for the same topic and split Google's authority. Consolidate links into one main page."},
        {"Metric / Term": "AI Prompts (7+ Words)", "Plain-English Meaning": "Long, question-style searches (e.g., 'how much does offshore medical billing cost in the philippines') that frequently trigger Google AI Overviews & AI Mode."},
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
<li><strong>Tuesday (15 min):</strong> Open <strong>Tab 4</strong> → pick 2–3 high-priority topics, copy the <em>Writer Brief</em>, add the suggested <em>Internal Links</em> in Webflow, and paste the <em>FAQPage JSON-LD</em> schema.</li>
<li><strong>Thursday (10 min):</strong> Open <strong>Tab 5</strong> → scan our 12 competitors' new blogs, filter to <em>Content Gaps Only</em>, and add the best topics to the editorial calendar.</li>
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
<li><strong>Verify Exact CPC Bids ($0):</strong> Click <em>Export Clean List for Google Keyword Planner (.txt)</em>, paste into Google Keyword Planner, download the forecast CSV, and upload it back into Tab 3.</li>
<li><strong>Cut Wasted Spend:</strong> Copy the <em>Real GSC Negative Keywords</em> and check the <em>Day-of-Week Chart</em> to weight ad budgets toward peak Mon–Fri B2B search days.</li>
<li><strong>Spy on Rival Ads:</strong> Open <strong>Tab 5 → Sub-Tab 2</strong> for direct US Google Ads Transparency links.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )
    with pb3:
        st.markdown(
            """<div class="cnxt-info-card">
<h3>Leadership &amp; Marketing Managers</h3>
<ul>
<li><strong>Weekly Pulse Check:</strong> Open <strong>Tab 1 (Executive Overview)</strong> to see total surging topics, Content Decay count, and monthly missed traffic value.</li>
<li><strong>Brand vs. Acquisition Health:</strong> Check the <em>Brand vs. Non-Brand</em> area chart to verify new-buyer search demand is growing alongside brand awareness.</li>
<li><strong>Share in 10 Seconds:</strong> Expand <em>1-Click Teams Executive Summary</em> to copy a clean bulleted update into Microsoft Teams, or click <em>Download Full Executive Report (.xlsx)</em> for leadership decks.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    with st.expander("5. Quick Troubleshooting & FAQ", expanded=False):
        st.markdown(
            """
- **"Why did the app say 'No topics match those filters'?"** — You likely combined a narrow vertical with a narrow action or funnel stage. Reset **Business Vertical**, **Recommended Next Step**, and **Funnel Stage** back to *All*, or switch **Trend Sensitivity** in the sidebar to **Broad (All Verticals)**.
- **"Why do some competitors show `blocked` in Tab 5?"** — A few competitor websites use Cloudflare/bot protection on their `sitemap.xml`. The remaining competitors still scan cleanly, and you can still check any blocked rival's US ads via the direct Ads Transparency links.
- **"Why did Google Trends say 'rate-limited'?"** — Free Google Trends occasionally throttles shared cloud servers. Wait a few minutes and try again, or use the **Live B2B Industry News Feed** and **Google/Bing Autocomplete** in Tab 6, which do not rate-limit.
- **"How do I get fresh data?"** — Click **Refresh Live GSC Data** in the left sidebar (takes 30–60 seconds).
            """
        )

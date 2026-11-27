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
    "red": "#FC4301",
    "twilight": "#071776",
    "midnight": "#071739",
    "teal": "#0F818A",
    "teal_blue": "#90CBDB",
    "cream": "#F4F1E2",
    "bg": "#F7F8FB",
    "paper": "#FFFFFF",
    "ash": "#64748B",
    "line": "#E2E8F0",
    "line_strong": "#CBD5E1",
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

COMPETITOR_BRAND_REGEX = re.compile("|".join(pat for _, _, pat in COMPETITORS_REGISTRY), re.I)

# Strict Competitor Blog URL Filters (Only editorial blog/article/insight posts)
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

# Strictly B2B Industry-Relevant News/Trend Queries (zero sports/celebrity macro noise)
INDUSTRY_NEWS_FEEDS = {
    "All B2B Outsourcing & Staffing": '("BPO outsourcing" OR "offshore staffing" OR "nearshore staffing" OR "staff augmentation") when:14d',
    "Healthcare & RCM": '("medical billing outsourcing" OR "revenue cycle management" OR "prior authorization" OR "healthcare BPO") when:14d',
    "Finance & Accounting": '("offshore accounting" OR "outsourced accounting" OR "CPA shortage" OR "finance outsourcing") when:14d',
    "Tech, IT & AI Staffing": '("nearshore software developers" OR "AI staffing" OR "IT staff augmentation" OR "offshore developers") when:14d',
    "Customer Experience (CX)": '("customer support outsourcing" OR "AI customer service" OR "contact center BPO" OR "virtual assistant") when:14d',
}

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Connext Global | Search & Topic Radar",
    page_icon=CNXT_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_connext_brand_css():
    css = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Arimo:wght@500;600;700&family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
:root {
    --cnxt-blue: #005AAC; --cnxt-sun: #E58B25; --cnxt-yellow: #F1C916;
    --cnxt-twilight: #071776; --cnxt-midnight: #071739; --cnxt-teal: #0F818A;
    --cnxt-teal-blue: #90CBDB; --cnxt-cream: #F4F1E2; --cnxt-bg: #F7F8FB;
    --cnxt-paper: #FFFFFF; --cnxt-ash: #64748B; --cnxt-line: #E2E8F0;
    --font-display: 'Arimo', Arial, sans-serif;
    --font-body: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}
html, body, [class*="css"] { font-family: var(--font-body); color: #0F172A; }
.stApp { background-color: var(--cnxt-bg); }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1440px; }
h1, h2, h3, h4 { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; letter-spacing: -0.015em; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: var(--cnxt-midnight) !important; border-right: 1px solid rgba(144, 203, 219, 0.16); }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #F8FAFC !important; }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small { color: var(--cnxt-teal-blue) !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea { background-color: #FFFFFF !important; color: #071739 !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #071739 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #071739 !important; }
section[data-testid="stSidebar"] .stButton > button { background-color: rgba(255,255,255,0.10) !important; color: var(--cnxt-teal-blue) !important; border: 1px solid rgba(144,203,219,0.32) !important; border-radius: 999px !important; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background-color: var(--cnxt-sun) !important; color: var(--cnxt-midnight) !important; border: none !important; font-weight: 600 !important; }

/* Buttons */
.stButton > button[kind="primary"], .stDownloadButton > button {
    background-color: var(--cnxt-sun) !important; color: var(--cnxt-midnight) !important;
    font-weight: 600 !important; border: none !important; border-radius: 999px !important;
    padding: 0.48rem 1.25rem !important; box-shadow: 0 4px 12px rgba(229,139,37,0.28) !important;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
    background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important;
}

/* Hero Banner */
.cnxt-hero {
    background: linear-gradient(135deg, #071776 0%, #071739 100%);
    color: #FFFFFF; padding: 22px 28px; border-radius: 12px; margin-bottom: 18px;
    border: 1px solid rgba(144,203,219,0.2); box-shadow: 0 6px 20px rgba(7,23,57,0.08);
}
.cnxt-eyebrow { font-family: var(--font-mono); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.16em; color: #F1C916; margin-bottom: 5px; font-weight: 600; }
.cnxt-hero h1 { font-size: clamp(1.35rem, 2.2vw, 1.9rem) !important; color: #FFFFFF !important; margin: 0 0 6px 0 !important; }
.cnxt-hero h1 span { color: #E58B25; }
.cnxt-hero p { color: #CBD5E1; font-size: 0.93rem; margin: 0; max-width: 880px; line-height: 1.5; }

/* Metric Cards */
div[data-testid="stMetric"] {
    background-color: var(--cnxt-paper); border: 1px solid var(--cnxt-line);
    border-top: 3px solid var(--cnxt-blue); border-radius: 10px; padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetric"] label { font-family: var(--font-mono) !important; font-size: 0.70rem !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; color: var(--cnxt-ash) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; font-weight: 700 !important; font-size: 1.65rem !important; }

/* Clean Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid var(--cnxt-line); margin-bottom: 14px; }
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-display); font-weight: 600; font-size: 0.92rem;
    color: #334155; background-color: #FFFFFF; border-radius: 8px 8px 0 0;
    padding: 10px 18px; border: 1px solid var(--cnxt-line); border-bottom: none;
}
.stTabs [aria-selected="true"] { background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important; border-color: var(--cnxt-twilight) !important; }

/* Callout Card */
.cnxt-callout {
    background-color: #FFFFFF; border: 1px solid var(--cnxt-line);
    border-left: 4px solid var(--cnxt-sun); border-radius: 8px;
    padding: 14px 18px; margin: 8px 0 14px 0; color: #1E293B; font-size: 0.92rem; line-height: 1.55;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}
.cnxt-callout strong { color: var(--cnxt-twilight); }

/* Section Subtitle */
.cnxt-sec-sub { color: #64748B; font-size: 0.9rem; margin-top: -8px; margin-bottom: 12px; }
span[data-baseweb="tag"] { background-color: var(--cnxt-twilight) !important; color: #FFFFFF !important; }
</style>"""
    st.markdown(css, unsafe_allow_html=True)


inject_connext_brand_css()


def require_connext_login():
    if st.session_state.get("cnxt_authenticated"):
        return

    st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 1.25, 1])
    with center_col:
        st.markdown("<div style='height: 44px;'></div>", unsafe_allow_html=True)
        st.image(CNXT_LOGO_COLOR, width=185)
        st.markdown(
            """<div class="cnxt-hero" style="margin-top: 14px;">
<div class="cnxt-eyebrow">INTERNAL TEAM ACCESS</div>
<h1>Connext Search &amp; <span>Topic Radar</span></h1>
<p>Sign in with your Connext team password to view live Google Search trends, Keyword Planner opportunity forecasts, competitor blog topics, and Bing/Trends research.</p>
</div>""",
            unsafe_allow_html=True,
        )
        if "app_password" not in st.secrets:
            st.error("`app_password` is not set in Streamlit Secrets. Add it in App Settings → Secrets.")
            st.stop()
        with st.form("cnxt_login_form"):
            entered_pwd = st.text_input("Team Password", type="password", placeholder="Enter team password...")
            if st.form_submit_button("Unlock Radar", type="primary", use_container_width=True):
                if hmac.compare_digest(entered_pwd, str(st.secrets["app_password"])):
                    st.session_state["cnxt_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")
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

COMMERCIAL_MODIFIERS = re.compile(
    r"\b("
    r"outsourcing|outsource|staffing|offshore|nearshore|dedicated|bpo|agency|company|firm|"
    r"services|solutions|cost|pricing|rates|quote|remote team|vs|versus|best|top|partner|provider|hire"
    r")\b",
    re.I,
)

AI_PROMPT_REGEX = re.compile(
    r"^(how|what|why|when|where|which|can|does|is|are|should|compare|difference between|pros and cons|steps to|best way to)\b",
    re.I,
)

ALL_CONNEXT_VERTICALS = [
    "Healthcare & RCM",
    "Finance, Accounting & Tax",
    "Tech, IT & AI Staffing",
    "Customer Experience (CX) & VAs",
    "Back Office & Operations Support",
    "Real Estate, Mortgage & Title/Escrow",
    "Insurance & Claims",
    "Legal & Compliance",
    "Fintech & Private Equity",
    "Engineering & Construction",
    "Sales, Marketing & E-Commerce",
    "Hospitality, Education & HR",
    "General B2B Outsourcing & Staffing",
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
    if u.endswith("/") and len(u) > len("https://connextglobal.com/"):
        u = u.rstrip("/")
    return u


def url_to_path(url: str) -> str:
    norm = normalize_url(url)
    path = urlparse(norm).path
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_sitemap_paths() -> tuple[list[str], str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ConnextTrendRadar/2.0)"}
    try:
        req = urllib.request.Request(SITEMAP_XML_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
        locs = [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text]
        child_sitemaps = [u for u in locs if u.lower().endswith(".xml")]
        if child_sitemaps:
            all_urls = []
            for sm_url in child_sitemaps[:10]:
                with urllib.request.urlopen(urllib.request.Request(sm_url, headers=headers), timeout=10) as c_resp:
                    c_root = ET.fromstring(c_resp.read())
                all_urls.extend([el.text.strip() for el in c_root.iter() if el.tag.endswith("loc") and el.text])
            locs = all_urls
        paths = sorted({url_to_path(u) for u in locs if "connextglobal.com" in u.lower()})
        if len(paths) >= 50:
            return paths, f"Live sitemap.xml ({len(paths):,} URLs)"
    except Exception:
        pass

    try:
        req = urllib.request.Request(SITEMAP_HTML_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
        hrefs = re.findall(r'href=["\']((?:https?://(?:www\.)?connextglobal\.com)?/[^"\'#?]*)["\']', html_text, re.I)
        paths = sorted({url_to_path(h) for h in hrefs})
        if len(paths) >= 50:
            return paths, f"Live /sitemap ({len(paths):,} URLs)"
    except Exception:
        pass

    fallback = sorted(set(CORE_LIVE_PATHS))
    return fallback, f"Core Pages Fallback ({len(fallback)} URLs)"


def _build_index_from_paths(all_paths: list[str]) -> tuple[set[str], pd.DataFrame, TfidfVectorizer, object]:
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
    comp_c = next((c for c in df.columns if c == "competition"), None)
    if not kw_c:
        raise ValueError("Missing 'Keyword' column in Keyword Planner export.")

    out = {}
    for _, r in df.iterrows():
        k = str(r[kw_c]).lower().strip()
        if not k or k == "nan":
            continue
        vol = int(pd.to_numeric(r[vol_c], errors="coerce") or 0) if vol_c else 0
        low_b = float(pd.to_numeric(r[low_c], errors="coerce") or 0.0) if low_c else 0.0
        high_b = float(pd.to_numeric(r[high_c], errors="coerce") or 0.0) if high_c else low_b
        avg_cpc = round((low_b + high_b) / 2.0, 2) if (low_b > 0 and high_b > 0) else round(max(low_b, high_b), 2)
        out[k] = {
            "kp_volume": vol,
            "kp_cpc": avg_cpc,
            "kp_low_bid": round(low_b, 2),
            "kp_high_bid": round(high_b, 2),
            "kp_competition": str(r[comp_c]).strip() if comp_c and pd.notna(r[comp_c]) else "Medium",
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


def parse_gsc_web_upload(uploaded_file, brand_regex: str, custom_negatives: str, exclude_local_ph: bool) -> pd.DataFrame:
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
    df = df[~df["query"].str.contains(STRICT_NON_INDUSTRY_REGEX, regex=True, na=False)]
    if exclude_local_ph:
        df = df[~df["query"].str.contains(LOCAL_PH_AND_STUDENT_REGEX, regex=True, na=False)]
    if custom_negatives.strip():
        df = df[~df["query"].str.contains(custom_negatives, regex=True, na=False)]
    df["is_branded"] = df["query"].str.contains(brand_regex, regex=True, na=False) if brand_regex else False
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gsc_api_data(creds_dict: dict, site_url: str, lookback_days: int, brand_regex: str, custom_exclude_regex: str, exclude_local_ph: bool) -> pd.DataFrame:
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/webmasters.readonly"])
    service = build("searchconsole", "v1", credentials=creds)
    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=lookback_days)

    rows, start_row, row_limit, max_rows = [], 0, 25000, 250000
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
            if not clean_q or STRICT_NON_INDUSTRY_REGEX.search(clean_q):
                continue
            if exclude_local_ph and LOCAL_PH_AND_STUDENT_REGEX.search(clean_q):
                continue
            if custom_pat and custom_pat.search(clean_q):
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
    return df


# ==============================================================================
# COMPETITOR NEW BLOGS ONLY + MAIN TARGET KEYWORD EXTRACTOR ($0 FREE)
# ==============================================================================
def is_competitor_blog_url(url: str) -> bool:
    path = urlparse(url).path.strip()
    if not path or path == "/" or BLOG_EXCLUDE_PATH_RE.search(path):
        return False
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return False
    if BLOG_INCLUDE_PATH_RE.search(path):
        slug = parts[-1]
        return len(slug) >= 5 and not slug.isdigit()
    return False


def extract_blog_keywords(slug: str, title: str, h1: str, meta_desc: str, h2s: list[str]) -> str:
    """Extracts 3-4 clean target keywords from a competitor blog's slug, title, H1, H2s, and meta description."""
    sources = [
        (slug.replace("-", " ").replace("_", " "), 3.2),
        (title, 3.0),
        (h1, 2.8),
        (" | ".join(h2s[:8]), 2.0),
        (meta_desc, 1.4),
    ]
    phrase_scores = {}
    for text, weight in sources:
        if not text:
            continue
        for chunk in re.split(r"[:|–—\-?,.;!()\[\]]+", text.lower()):
            tokens = [
                t for t in re.findall(r"[a-z0-9]+", chunk)
                if (len(t) > 2 or t in {"ai", "it", "hr", "qa", "va", "ap", "ar"}) and t not in COMP_BRAND_WORDS
            ]
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
    """Scrapes live blog title, H1, meta description, and H2s to return (Clean Headline, Main Keywords)."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    title_txt, h1_txt, meta_txt, h2_list = "", "", "", []
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw_html = resp.read(75000).decode("utf-8", errors="ignore")
        # Strip scripts/styles for clean heading extraction
        clean_html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S)

        m_t = re.search(r"<title[^>]*>(.*?)</title>", clean_html, re.I | re.S)
        if m_t:
            t_raw = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m_t.group(1)))).strip()
            title_txt = re.split(r"\s+[|\-–—]\s+", t_raw)[0].strip()

        m_h1 = re.search(r"<h1[^>]*>(.*?)</h1>", clean_html, re.I | re.S)
        if m_h1:
            h1_txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m_h1.group(1)))).strip()

        m_desc = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', clean_html, re.I | re.S)
        if not m_desc:
            m_desc = re.search(r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']', clean_html, re.I | re.S)
        if m_desc:
            meta_txt = re.sub(r"\s+", " ", html.unescape(m_desc.group(1))).strip()

        for h2_m in re.findall(r"<h2[^>]*>(.*?)</h2>", clean_html, re.I | re.S)[:8]:
            h2_c = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", h2_m))).strip()
            if 4 <= len(h2_c) <= 120:
                h2_list.append(h2_c)
    except Exception:
        pass

    headline = h1_txt if len(h1_txt) >= 8 else (title_txt if len(title_txt) >= 8 else slug.replace("-", " ").title())
    main_kws = extract_blog_keywords(slug, title_txt, h1_txt, meta_txt, h2_list)
    return headline, main_kws


@st.cache_data(ttl=86400, show_spinner=False)
def scan_competitor_new_blogs_auto(lookback_days: int = 45) -> tuple[pd.DataFrame, list[str]]:
    """Scans all 12 competitors' sitemaps strictly for NEW BLOG ARTICLES and extracts their main keywords."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)

    def scan_one(comp_tuple: tuple[str, str, str]) -> tuple[list[dict], str]:
        cname, domain, _ = comp_tuple
        found = []
        parsed_any = False
        for sm_url in [f"https://{domain}/sitemap_index.xml", f"https://{domain}/sitemap.xml", f"https://www.{domain}/sitemap_index.xml", f"https://www.{domain}/sitemap.xml"]:
            try:
                with urllib.request.urlopen(urllib.request.Request(sm_url, headers=headers), timeout=7) as resp:
                    root = ET.fromstring(resp.read())
                parsed_any = True
                child_locs = [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text and el.text.strip().lower().endswith(".xml")]
                sub_roots = []
                if child_locs:
                    # Strictly prioritize blog / post / article / insight child sitemaps
                    blog_children = [u for u in child_locs if any(k in u.lower() for k in ["post", "blog", "article", "insight", "news", "resource"])]
                    target_children = blog_children[:4] if blog_children else child_locs[:3]
                    for sub_u in target_children:
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
                            mod_dt = datetime.strptime(mod_txt, "%Y-%m-%d").date()
                        except ValueError:
                            continue
                        if mod_dt >= cutoff_date:
                            slug = [p for p in urlparse(loc_txt).path.split("/") if p][-1]
                            found.append({
                                "Competitor": cname,
                                "Published": mod_txt,
                                "slug": slug,
                                "Blog URL": loc_txt,
                            })
                            if len(found) >= 15:
                                break
                    if len(found) >= 15:
                        break
                break
            except Exception:
                continue
        return found, (f"{cname}: {len(found)} blogs" if parsed_any else f"{cname}: blocked")

    all_found, notes = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for f_rows, n_txt in executor.map(scan_one, COMPETITORS_REGISTRY):
            all_found.extend(f_rows)
            notes.append(n_txt)

    if not all_found:
        return pd.DataFrame(), notes

    out_df = pd.DataFrame(all_found).drop_duplicates(subset=["Blog URL"]).sort_values("Published", ascending=False).head(45).reset_index(drop=True)

    # Scrape headlines & main target keywords in parallel
    def enrich_row(pair):
        return _scrape_blog_metadata(pair[0], pair[1])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        meta_results = list(executor.map(enrich_row, zip(out_df["Blog URL"], out_df["slug"])))

    out_df["Blog Headline"] = [m[0] for m in meta_results]
    out_df["Main Target Keywords"] = [m[1] for m in meta_results]
    out_df["Vertical"] = out_df.apply(lambda r: classify_sector(f"{r['Blog Headline']} {r['Main Target Keywords']} {r['slug']}"), axis=1)
    return out_df, notes


# ==============================================================================
# 100% FREE DISCOVERY: INDUSTRY TRENDS, B2B NEWS RSS, BING, SUGGEST & REDDIT
# ==============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_google_trends_b2b(seed_queries: tuple[str, ...], geo: str = "US", timeframe: str = "today 1-m") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """100% Free B2B-Scoped Google Trends (no unrelated macro news)."""
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
                notes.append("Relative Interest loaded live via free Google Trends")

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
            notes.append("Google Trends chart temporarily rate-limited on cloud IP — Live B2B Industry Trends below remain active.")

    return (
        pd.DataFrame(interest_rows),
        pd.DataFrame(rising_rows).drop_duplicates(subset=["Rising B2B Query"]).reset_index(drop=True) if rising_rows else pd.DataFrame(),
        notes,
    )


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_b2b_industry_trending_feed(vertical_key: str) -> pd.DataFrame:
    """100% Free Industry-Filtered Google News RSS (Strictly BPO/Outsourcing/Connext verticals — zero sports/celebrity noise)."""
    query = INDUSTRY_NEWS_FEEDS.get(vertical_key, INDUSTRY_NEWS_FEEDS["All B2B Outsourcing & Staffing"])
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query, safe='')}&hl=en-US&gl=US&ceid=US:en"
    rows = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ConnextTrendRadar/2.0)"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            root = ET.fromstring(resp.read())
        for item in root.iter("item"):
            raw_title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()[:16]
            src_el = item.find("source")
            publisher = src_el.text.strip() if src_el is not None and src_el.text else "Industry News"
            clean_title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title
            if clean_title:
                rows.append({
                    "Date": pub,
                    "Trending B2B Headline": clean_title,
                    "Vertical": classify_sector(clean_title),
                    "Source": publisher,
                    "Article Link": link,
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
                    "Competitor": cname,
                    "Advertiser": adv_name,
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
def fetch_free_bing_webmaster_queries(api_key: str, site_url: str, brand_regex: str, exclude_local_ph: bool, custom_negatives: str) -> tuple[pd.DataFrame, str]:
    url = f"https://ssl.bing.com/webmaster/api.svc/json/GetQueryStats?siteUrl={urllib.parse.quote(site_url.strip(), safe='')}&apikey={urllib.parse.quote(api_key.strip(), safe='')}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as resp:
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
    """Pulls free live autocomplete questions & modifiers from both Google Suggest and Bing Suggest."""
    rows, seen = [], set()
    g_patterns = [seed_topic, f"how to {seed_topic}", f"cost of {seed_topic}", f"best {seed_topic}", f"{seed_topic} vs ", f"{seed_topic} pricing"]
    for q_str in g_patterns:
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
        b_url = f"https://api.bing.com/osjson.aspx?query={urllib.parse.quote(seed_topic)}"
        with urllib.request.urlopen(urllib.request.Request(b_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=6) as resp:
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
        q_enc = urllib.parse.quote(f"{seed_topic} outsourcing OR offshore OR agency")
        url = f"https://www.reddit.com/search.json?q={q_enc}&sort=relevance&t=year&limit=10"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ConnextTrendRadar/2.0"}), timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        rows = []
        for child in (data.get("data") or {}).get("children", []):
            p = child.get("data") or {}
            title = str(p.get("title") or "").strip()
            if title:
                rows.append({
                    "Community": f"r/{p.get('subreddit', '')}",
                    "Buyer Question / Discussion": title,
                    "Comments": int(p.get("num_comments") or 0),
                    "Date": datetime.fromtimestamp(float(p.get("created_utc") or 0), tz=timezone.utc).strftime("%Y-%m-%d"),
                    "Thread": f"https://www.reddit.com{p.get('permalink', '')}",
                })
        return pd.DataFrame(rows).sort_values("Comments", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ==============================================================================
# CLUSTERING & MULTI-HORIZON TREND SCORING
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


def analyze_multi_horizon_trends(non_brand_df: pd.DataFrame, live_set: set[str], live_df: pd.DataFrame, live_vec: TfidfVectorizer, live_mat, min_daily_imp: int, min_up_days: int, min_norm_slope: float, min_lift: float):
    if non_brand_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    all_dates = sorted(non_brand_df["date"].unique())
    n_days = len(all_dates)
    if n_days < 14:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

    results = []
    for topic, group in daily_topic.groupby("topic"):
        series = group.set_index("date")["impressions"].reindex(all_dates, fill_value=0)
        arr_today, arr_week, arr_prior14 = series.loc[today_dates].values, series.loc[week_dates].values, series.loc[prior_14_dates].values
        arr_m_rec, arr_m_base = series.loc[month_recent].values, series.loc[month_prior].values

        m_today, m_week, m_prior = float(np.mean(arr_today)), float(np.mean(arr_week)), max(float(np.mean(arr_prior14)), 1.0)
        m_mrec, m_mbase = float(np.mean(arr_m_rec)), max(float(np.mean(arr_m_base)), 1.0)
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

        rec_grp = group[group["date"].isin(week_dates)]
        imp_7d = int(rec_grp["impressions"].sum())
        imp_30d = int(group[group["date"].isin(month_dates)]["impressions"].sum())
        avg_pos = float(rec_grp["weighted_pos"].sum() / max(imp_7d, 1))

        q_list = top_q_map.get(topic, [topic])
        lt_list = lt_q_map.get(topic, [])
        joined_q = ", ".join(q_list[:6])
        has_comm = bool(COMMERCIAL_MODIFIERS.search(joined_q))
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

        if has_comm and avg_pos > 5.0:
            action = "Launch Paid Ads"
        elif site_chk["already_exists_in_sitemap"] or (4.0 <= avg_pos <= 15.0):
            action = "Update Existing Page"
        elif not site_chk["already_exists_in_sitemap"]:
            action = "Write New Page"
        else:
            action = "Defend Top 3 Rank"

        cur_clicks = round(imp_30d * expected_ctr_by_rank(avg_pos), 1)
        top3_clicks = round(imp_30d * expected_ctr_by_rank(2.5), 1)
        missed_clicks = max(0.0, round(top3_clicks - cur_clicks, 1))
        ads_fit = min(100, (30 if avg_pos > 8 else (20 if avg_pos > 4 else 5)) + (35 if has_comm else 10) + min(35, int((lift_today - 1.0) * 18)))

        results.append({
            "topic": topic, "sector": sector,
            "is_today": is_today, "is_week": is_week, "is_month": is_month,
            "is_striking_distance": bool(4.0 <= avg_pos <= 15.0), "is_cannibalized": bool(u_pages >= 2),
            "team_action": action, "ads_fit_score": ads_fit,
            "growth_pct_3d": round((lift_today - 1.0) * 100),
            "growth_pct_week": round(((m_week / m_prior) - 1.0) * 100),
            "month_growth_pct": round((lift_month - 1.0) * 100),
            "impressions_7d": imp_7d, "impressions_30d": imp_30d,
            "sparkline_7d": [int(v) for v in arr_week],
            "connext_org_pos": round(avg_pos, 1), "google_rank_label": human_rank_label(avg_pos),
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

    if not results:
        return pd.DataFrame(), daily_topic, pd.DataFrame(), pd.DataFrame()
    trends_df = pd.DataFrame(results).sort_values("impressions_7d", ascending=False).reset_index(drop=True)

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
    return trends_df, daily_topic, q_merged, pages_merged


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
            "Priority": prio, "Topic": r["topic"], "Vertical": r["sector"],
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
st.sidebar.markdown("**Search & Topic Radar**")
st.sidebar.caption(f"100% Free Mode • {sitemap_source_label}")
if st.sidebar.button("Lock / Log Out"):
    st.session_state["cnxt_authenticated"] = False
    st.rerun()
st.sidebar.divider()

has_secrets = "gcp_service_account" in st.secrets and "gsc_config" in st.secrets
data_source = st.sidebar.radio("Search Console Source", ["Live GSC API", "Upload GSC CSV / ZIP"], index=0)
exclude_local_ph = st.sidebar.checkbox("Filter Out Local PH / Student Noise", value=True)
custom_negatives = st.sidebar.text_input("Extra Excluded Words", placeholder="intern|sample|pdf")

raw_df = pd.DataFrame()
if data_source == "Live GSC API":
    if has_secrets:
        site_url = st.secrets["gsc_config"].get("site_url", DEFAULT_SITE_URL)
        brand_regex = st.secrets["gsc_config"].get("brand_regex", DEFAULT_BRAND_REGEX)
        lookback = st.sidebar.slider("GSC History Window (Days)", 28, 90, 45)
        if st.sidebar.button("Refresh Live Data", type="primary"):
            fetch_gsc_api_data.clear()
            fetch_live_sitemap_paths.clear()
        with st.spinner("Syncing Google Search Console..."):
            try:
                raw_df = fetch_gsc_api_data(dict(st.secrets["gcp_service_account"]), site_url, lookback, brand_regex, custom_negatives, exclude_local_ph)
            except Exception as err:
                st.error(f"GSC API Error: {err}")
                st.stop()
    else:
        uploaded_json = st.sidebar.file_uploader("Service Account JSON", type=["json"])
        site_url = st.sidebar.text_input("GSC Property", value=DEFAULT_SITE_URL)
        brand_regex = DEFAULT_BRAND_REGEX
        lookback = st.sidebar.slider("GSC History Window (Days)", 28, 90, 45)
        if uploaded_json:
            raw_df = fetch_gsc_api_data(json.load(uploaded_json), site_url, lookback, brand_regex, custom_negatives, exclude_local_ph)
else:
    uploaded_csv = st.sidebar.file_uploader("Upload GSC Web CSV / ZIP", type=["csv", "zip"])
    brand_regex = DEFAULT_BRAND_REGEX
    if uploaded_csv:
        try:
            raw_df = parse_gsc_web_upload(uploaded_csv, brand_regex, custom_negatives, exclude_local_ph)
        except Exception as err:
            st.sidebar.error(str(err))

st.sidebar.divider()
preset_mode = st.sidebar.selectbox("Trend Sensitivity", ["Broad (All Verticals)", "Balanced", "Strict Breakouts"], index=0)
def_imp, def_up, def_lift, def_slope = (50, 4, 1.40, 10) if preset_mode.startswith("Strict") else ((30, 4, 1.30, 8) if preset_mode.startswith("Balanced") else (15, 3, 1.18, 4))

with st.sidebar.expander("Advanced Settings & ROI Assumptions", expanded=False):
    default_cpc_est = st.number_input("Fallback B2B CPC ($)", 1.0, 100.0, 7.50, 0.50)
    lead_conv_pct = st.slider("Lead Conv. Rate (%)", 1.0, 15.0, 4.0, 0.5)
    avg_deal_ltv = st.number_input("Avg Client Value ($)", 1000, 250000, 18000, 1000)
    min_daily_imp = st.number_input("Min Avg Daily Impr.", 5, 5000, def_imp, 5)
    min_up_days = st.slider("Weekly Upward Days", 2, 6, def_up)
    min_lift = st.slider("Min Growth Multiplier", 1.05, 3.0, def_lift, 0.05)
    min_slope_pct = st.slider("Min Daily Slope (%)", 1, 40, def_slope, 1)
    cluster_dist = st.slider("Topic Grouping Breadth", 0.30, 0.75, 0.55, 0.05)

has_bing_secret = "bing" in st.secrets and "api_key" in st.secrets["bing"] and bool(str(st.secrets["bing"]["api_key"]).strip())

# ==============================================================================
# HEADER & HERO BANNER
# ==============================================================================
h_col1, h_col2 = st.columns([1.1, 4])
with h_col1:
    st.image(CNXT_LOGO_COLOR, width=180)
with h_col2:
    if not raw_df.empty:
        st.caption(f"Live GSC Window: **{raw_df['date'].min().strftime('%b %d')} – {raw_df['date'].max().strftime('%b %d, %Y')}** ({raw_df['date'].nunique()} days)  •  {sitemap_source_label}  •  $0 Credit-Free Mode")

st.markdown(
    """<div class="cnxt-hero">
<div class="cnxt-eyebrow">CONNEXT GLOBAL GROWTH &amp; TOPIC RESEARCH</div>
<h1>Search, Content &amp; Competitor <span>Trend Radar</span></h1>
<p>Plain-English search demand from Google Search Console, Keyword Planner traffic &amp; ROI opportunity, new competitor blog topics with extracted target keywords, and industry-filtered Google &amp; Bing trends.</p>
</div>""",
    unsafe_allow_html=True,
)

if raw_df.empty:
    st.info("Connect your Google Search Console API or upload a GSC CSV in the sidebar to begin.")
    st.stop()

with st.spinner("Analyzing live B2B search demand..."):
    clustered_df = cluster_queries(raw_df, cluster_dist)
    trends_df, daily_topic_df, all_queries_df, pages_df = analyze_multi_horizon_trends(
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
sel_intent = fc3.selectbox("Buyer Intent", ["All Searches", "Commercial / Ready-to-Buy", "Research / How-To"])
search_box = fc4.text_input("Filter by Keyword or URL", placeholder="e.g., medical billing, cpa...")

f_trends = trends_df.copy()
if sel_vert != "All Verticals":
    f_trends = f_trends[f_trends["sector"] == sel_vert.rsplit(" (", 1)[0]]
if sel_act != "All Actions":
    if sel_act.startswith("Striking Distance"):
        f_trends = f_trends[f_trends["is_striking_distance"]]
    else:
        f_trends = f_trends[f_trends["team_action"] == sel_act.rsplit(" (", 1)[0]]
if sel_intent == "Commercial / Ready-to-Buy":
    f_trends = f_trends[f_trends["commercial_intent"]]
elif sel_intent == "Research / How-To":
    f_trends = f_trends[~f_trends["commercial_intent"]]
if search_box.strip():
    pat = re.escape(search_box.strip())
    f_trends = f_trends[f_trends["topic"].str.contains(pat, case=False, na=False) | f_trends["top_rising_queries"].str.contains(pat, case=False, na=False) | f_trends["primary_page"].str.contains(pat, case=False, na=False)]
f_trends = f_trends.reset_index(drop=True)

if f_trends.empty:
    st.info("No topics match those filters. Reset **Business Vertical** or **Recommended Next Step** above.")
    st.stop()

f_queries = all_queries_df[all_queries_df["topic"].isin(f_trends["topic"])].copy()
f_pages = pages_df[pages_df["topic"].isin(f_trends["topic"])].copy()

kp_lookup_dict = st.session_state.get("kp_lookup_dict", {})
ads_topics_df, rec_kws_df = build_ads_and_roi_tables(f_trends, f_queries, kp_lookup_dict, default_cpc_est, lead_conv_pct / 100.0, float(avg_deal_ltv))
existing_content_df, truly_new_df = build_content_tables(f_trends)

# --- TOP KPI ROW ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Surging Topics", len(f_trends))
k2.metric("Weekly Impressions", f"{int(f_trends['impressions_7d'].sum()):,}")
k3.metric("Avg Google Rank", f"#{f_trends['connext_org_pos'].mean():.1f}")
k4.metric("Est. Missed Clicks/Mo", f"{f_trends['est_missed_clicks_30d'].sum():,.0f}")
k5.metric("Lost Traffic Value/Mo", f"${ads_topics_df['Lost Traffic Value/Mo'].sum():,.0f}")

# --- 6 CLEAN, SCANNABLE TABS ---
t_over, t_pages, t_ads, t_content, t_comp, t_trends = st.tabs([
    "1. Overview",
    f"2. Pages & SEO ({len(f_pages)})",
    f"3. Paid Ads & ROI ({len(rec_kws_df)})",
    f"4. Content & FAQs ({len(truly_new_df)} New)",
    "5. Competitor Blogs & Ads",
    "6. Trends, Bing & Research",
])

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with t_over:
    horizon = st.radio("Time Horizon:", ["All Surging", "Today (3-Day Spike)", "This Week (7-Day)", "This Month (30-Day)"], horizontal=True)
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
                custom_data=["google_rank_label", "est_missed_clicks_30d", "sector"],
                labels={"impressions_7d": "7-Day Impressions", "topic": "", "team_action": ""},
            )
            fig_t.update_traces(texttemplate="+%{text}%", textposition="outside", hovertemplate="<b>%{y}</b><br>7d Impr: <b>%{x:,}</b><br>Rank: <b>%{customdata[0]}</b><br>Missed Clicks/Mo: <b>%{customdata[1]:,.0f}</b><extra></extra>")
            fig_t.update_layout(height=370, margin=dict(l=5, r=45, t=10, b=40), yaxis=dict(autorange="reversed"), xaxis=dict(gridcolor=CNXT_COLORS["line"]), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_t, use_container_width=True)

        with c2:
            st.markdown("#### Impressions by Connext Vertical")
            sec_df = vdf.groupby("sector", as_index=False).agg(topics=("topic", "count"), impr=("impressions_7d", "sum")).sort_values("impr", ascending=True)
            fig_s = px.bar(sec_df, x="impr", y="sector", orientation="h", text="topics", color_discrete_sequence=[CNXT_COLORS["twilight"]], labels={"impr": "7-Day Impressions", "sector": ""})
            fig_s.update_traces(texttemplate="%{text} topics", textposition="outside")
            fig_s.update_layout(height=370, margin=dict(l=5, r=55, t=10, b=40), xaxis=dict(gridcolor=CNXT_COLORS["line"]), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("#### Daily Search Trend Inspector")
        sel_t = st.selectbox("Select a topic to inspect its daily curve:", vdf["topic"].tolist(), key="sel_ov_topic")
        sr = vdf[vdf["topic"] == sel_t].iloc[0]
        st.markdown(
            f"""<div class="cnxt-callout">
<strong>{sr['topic'].title()}</strong> ({sr['sector']}) — <strong>{sr['impressions_7d']:,}</strong> weekly impressions (<strong>+{sr[g_col]}%</strong> {g_lbl}) • Currently <strong>{sr['google_rank_label']}</strong> • Est. <strong>{sr['est_missed_clicks_30d']:,.0f} missed clicks/mo</strong> • Recommended Action: <strong>{sr['team_action']}</strong> → <code>{sr['recommended_money_page']}</code>
</div>""",
            unsafe_allow_html=True,
        )
        ts = daily_topic_df[daily_topic_df["topic"] == sel_t].sort_values("date").copy()
        ts["avg7"] = ts["impressions"].rolling(7, min_periods=1).mean().round(0)
        fig_l = go.Figure()
        fig_l.add_trace(go.Bar(x=ts["date"], y=ts["impressions"], name="Daily Impressions", marker_color="rgba(0,90,172,0.28)"))
        fig_l.add_trace(go.Scatter(x=ts["date"], y=ts["avg7"], mode="lines+markers", name="7-Day Trendline", line=dict(color=CNXT_COLORS["sun"], width=3), marker=dict(color=CNXT_COLORS["twilight"], size=6)))
        fig_l.update_layout(height=270, margin=dict(l=10, r=10, t=10, b=35), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", yaxis=dict(title="Impressions", gridcolor=CNXT_COLORS["line"]), legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_l, use_container_width=True)

        st.markdown(f"#### All Surging Topics ({len(vdf)})")
        ov_tbl = vdf[["topic", "sector", "team_action", "sparkline_7d", "impressions_7d", g_col, "google_rank_label", "est_missed_clicks_30d", "recommended_money_page"]].copy()
        ov_tbl["growth_lbl"] = ov_tbl[g_col].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
        st.dataframe(
            ov_tbl[["topic", "sector", "team_action", "sparkline_7d", "impressions_7d", "growth_lbl", "google_rank_label", "est_missed_clicks_30d", "recommended_money_page"]],
            use_container_width=True, hide_index=True, key="tbl_overview",
            column_config={
                "topic": "Topic", "sector": "Vertical", "team_action": "Next Step",
                "sparkline_7d": st.column_config.LineChartColumn("7d Trend", width="small"),
                "impressions_7d": st.column_config.NumberColumn("7d Impr."), "growth_lbl": g_lbl,
                "google_rank_label": "Google Rank", "est_missed_clicks_30d": st.column_config.NumberColumn("Missed Clicks/Mo", format="%.0f"),
                "recommended_money_page": st.column_config.LinkColumn("Target Service Page"),
            },
        )

        with st.expander("1-Click Teams Executive Summary (Copy & Paste)", expanded=False):
            lines = [f"{i+1}. **{r['topic'].title()}** ({r['sector']}) — {r['impressions_7d']:,} weekly impr. (+{r[g_col]}%), Rank {r['google_rank_label']}, ~{r['est_missed_clicks_30d']:,.0f} missed clicks/mo → **{r['team_action']}**" for i, r in vdf.head(5).iterrows()]
            st.code(f"Connext Search Trend Summary ({datetime.now(timezone.utc).strftime('%b %d, %Y')})\n" + "\n".join(lines), language="markdown")

# ==============================================================================
# TAB 2: PAGES & SEO (CANNIBALIZATION + SITEMAP + QUERIES)
# ==============================================================================
with t_pages:
    st.markdown("#### Ranking Pages, Multi-URL Split Check & Search Queries")
    st.markdown('<p class="cnxt-sec-sub">Inspect which Connext URLs currently rank, catch pages competing against each other, and view exact user searches.</p>', unsafe_allow_html=True)

    cann_df = f_trends[f_trends["is_cannibalized"]]
    if not cann_df.empty:
        with st.expander(f"Cannibalization Check: {len(cann_df)} Topics Have 2+ Competing Connext URLs", expanded=False):
            st.dataframe(cann_df[["topic", "sector", "ranking_pages_count", "google_rank_label", "impressions_7d", "primary_page", "closest_existing_url"]], use_container_width=True, hide_index=True, key="tbl_cann", column_config={"primary_page": st.column_config.LinkColumn("Top Ranking URL"), "closest_existing_url": st.column_config.LinkColumn("Main Sitemap URL")})

    sel_c = st.selectbox("Choose a Topic to Inspect:", f_trends["topic"].tolist(), key="sel_pages_topic")
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

    st.markdown("**Exact Search Queries & AI Prompts (7+ Words)**")
    tqs = f_queries[f_queries["topic"] == sel_c].sort_values("imp_7d", ascending=False).copy()
    tqs["growth"] = tqs["query_growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
    tqs["rank"] = tqs["connext_org_pos"].apply(human_rank_label)
    st.dataframe(tqs[["query", "is_ai_prompt", "imp_7d", "imp_30d", "growth", "rank", "top_page"]], use_container_width=True, hide_index=True, key="tbl_tqs", column_config={"is_ai_prompt": st.column_config.CheckboxColumn("AI Prompt"), "top_page": st.column_config.LinkColumn("Ranking URL")})

# ==============================================================================
# TAB 3: PAID ADS, EXPECTED TRAFFIC & KEYWORD PLANNER ($0 CSV SYNC)
# ==============================================================================
with t_ads:
    st.markdown("#### Expected Traffic, Lost Opportunity Cost & Paid Ads Builder")
    st.markdown('<p class="cnxt-sec-sub">Estimate missed monthly clicks below Top 3 and convert surging B2B queries into Google Ads campaigns—zero API credits required.</p>', unsafe_allow_html=True)

    with st.expander("Optional: Upload Free Google Keyword Planner CSV to Lock In Exact Monthly Volume & CPC Bids", expanded=False):
        kp_up = st.file_uploader("Upload CSV/XLSX exported from Google Keyword Planner ('Get search volume and forecasts')", type=["csv", "xlsx", "xls"], key="inline_kp_upload")
        if kp_up:
            try:
                parsed_kp = parse_keyword_planner_csv(kp_up)
                st.session_state["kp_lookup_dict"] = parsed_kp
                st.success(f"Synced {len(parsed_kp):,} Keyword Planner metrics! Tables below updated.")
                ads_topics_df, rec_kws_df = build_ads_and_roi_tables(f_trends, f_queries, parsed_kp, default_cpc_est, lead_conv_pct / 100.0, float(avg_deal_ltv))
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

        with st.expander("Ad Group Copy-Paste Builder (Exact, Phrase, Negatives & Headlines)", expanded=False):
            sel_ad = st.selectbox("Pick Topic:", ads_topics_df["Topic"].tolist(), key="sel_ad_grp")
            ad_row = f_trends[f_trends["topic"] == sel_ad].iloc[0]
            kws = rec_kws_df[rec_kws_df["Topic"] == sel_ad]["Keyword"].tolist() or ad_row["queries_list"]
            st.caption(f"Target Landing Page: **{ad_row['recommended_money_page']}**")
            ce, cp, cn = st.columns(3)
            ce.markdown("**[Exact Match]**")
            ce.code("\n".join(f"[{q}]" for q in kws), language="text")
            cp.markdown('**"Phrase Match"**')
            cp.code("\n".join(f'"{q}"' for q in kws), language="text")
            cn.markdown("**B2B Negatives**")
            cn.code("-jobs\n-careers\n-hiring\n-salary\n-internship\n-glassdoor\n-indeed\n-free\n-template", language="text")
            hl, desc = build_rsa_assets(sel_ad)
            ch, cd = st.columns(2)
            ch.markdown("**Headlines (≤ 30 chars)**")
            for cat, h in hl:
                ch.text(f"[{cat}] {h} ({len(h)}/30)")
            cd.markdown("**Descriptions (≤ 90 chars)**")
            for d in desc:
                cd.text(f"{d} ({len(d)}/90)")

# ==============================================================================
# TAB 4: CONTENT PLAN & FAQ SCHEMA
# ==============================================================================
with t_content:
    st.markdown("#### 1. Update Existing Sitemap Pages (Prevent Duplicate Content)")
    st.markdown('<p class="cnxt-sec-sub">Cross-checked against live connextglobal.com/sitemap.xml so writers refresh existing pages instead of duplicating URLs.</p>', unsafe_allow_html=True)
    st.dataframe(existing_content_df, use_container_width=True, hide_index=True, key="tbl_ex_c", column_config={"Existing Page": st.column_config.LinkColumn("Existing Sitemap Page")})
    st.download_button("Download Existing Page Updates CSV", existing_content_df.to_csv(index=False).encode("utf-8"), file_name="connext_content_updates.csv", mime="text/csv", key="dl_ex_c")

    st.divider()
    st.markdown(f"#### 2. Truly New Content Ideas — Verified Not in Sitemap ({len(truly_new_df)})")
    if truly_new_df.empty:
        st.success("All currently surging topics already have a matching page in connextglobal.com/sitemap.xml!")
    else:
        st.dataframe(truly_new_df, use_container_width=True, hide_index=True, key="tbl_new_c", column_config={"Service Page to Link": st.column_config.LinkColumn("Service Page")})
        st.download_button("Download New Content Gaps CSV", truly_new_df.to_csv(index=False).encode("utf-8"), file_name="connext_new_content_ideas.csv", mime="text/csv", key="dl_new_c")

    st.divider()
    st.markdown("#### 3. Writer Brief & Webflow `FAQPage` JSON-LD Generator")
    sel_b = st.selectbox("Select Topic:", existing_content_df["Topic"].tolist(), key="sel_brief")
    br = f_trends[f_trends["topic"] == sel_b].iloc[0]
    lt_b = "\n".join(f"  - `{q}`" for q in (br["long_tail_list"] or br["queries_list"]))
    bc1, bc2 = st.columns([1.1, 1])
    with bc1:
        st.markdown(
            f"""**Brief: {br['topic'].title()}** ({br['sector']})
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
# TAB 5: COMPETITOR NEW BLOGS (STRICTLY BLOGS + KEYWORDS) & FREE ADS LIBRARIES
# ==============================================================================
with t_comp:
    st.markdown("#### Competitor New Blog Radar & Free US Ad Libraries (12 BPO Rivals)")
    st.markdown('<p class="cnxt-sec-sub">100% Free — Zero SerpApi credits used. Tracks newly published competitor blog posts with auto-extracted target keywords plus direct US Ad Transparency links.</p>', unsafe_allow_html=True)

    c_sub1, c_sub2 = st.tabs([
        "New Competitor Blogs & Main Keywords ($0)",
        "US Google Ads Transparency & Microsoft Ad Library ($0)",
    ])

    with c_sub1:
        b_top1, b_top2 = st.columns([3, 1])
        blog_days = b_top1.slider("Show Competitor Blogs Published / Updated Within (Days):", 14, 90, 45, key="sl_blog_days")
        if b_top2.button("Refresh Blog Scan", key="btn_ref_blogs"):
            scan_competitor_new_blogs_auto.clear()
            st.rerun()

        with st.spinner("Scanning all 12 competitors strictly for new blog articles & extracting main keywords..."):
            comp_blogs_df, blog_notes = scan_competitor_new_blogs_auto(lookback_days=blog_days)

        st.caption("Blog feed status: " + " • ".join(blog_notes))

        if comp_blogs_df.empty:
            st.info("No recently dated blog posts found across the 12 competitor sitemaps in this date window. Try widening the day slider above.")
        else:
            cov_list, closest_list = [], []
            for _, row in comp_blogs_df.iterrows():
                chk = check_topic_against_sitemap(f"{row['Blog Headline']} {row['Main Target Keywords']}", "", live_set, live_df, live_vec, live_mat)
                cov_list.append("Connext Covers" if chk["already_exists_in_sitemap"] else "Content Gap")
                closest_list.append(chk["closest_existing_url"])
            comp_blogs_df["Connext Status"] = cov_list
            comp_blogs_df["Connext Closest Page"] = closest_list

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
            bk1.metric("New Competitor Blogs Found", len(bview))
            bk2.metric("Topic Gaps Missing on Connext", int((bview["Connext Status"] == "Content Gap").sum()))
            bk3.metric("Competitors Publishing Blogs", bview["Competitor"].nunique())

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
            st.download_button(
                "Download Competitor New Blogs & Keywords CSV",
                bview[["Competitor", "Published", "Blog Headline", "Main Target Keywords", "Vertical", "Connext Status", "Connext Closest Page", "Blog URL"]].to_csv(index=False).encode("utf-8"),
                file_name="connext_competitor_new_blogs_keywords.csv",
                mime="text/csv",
                key="dl_comp_blogs",
            )

    with c_sub2:
        st.markdown(
            """<div class="cnxt-callout">
<strong>Free US Ad Transparency Lookup:</strong> Open any competitor's bare domain directly in <strong>Google Ads Transparency Center (US)</strong> to inspect active US Search/Display ads, or search them in <strong>Microsoft Ad Library</strong>. Zero SerpApi credits used.
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
            dir_df, use_container_width=True, hide_index=True, key="tbl_ad_dir",
            column_config={
                "Google Ads Transparency (US)": st.column_config.LinkColumn("Open Google Ads Transparency (US)"),
                "Microsoft Ad Library": st.column_config.LinkColumn("Open Microsoft Ad Library"),
            },
        )

        st.divider()
        st.markdown("**Optional: Official Free Microsoft Ad Library API Scan (EU/EEA Spillover Creatives)**")
        if st.button("Scan 12 Competitors via Free Microsoft Ad Library API", key="btn_ms_api"):
            with st.spinner("Querying Microsoft Ad Library API..."):
                st.session_state["ms_api_df"] = fetch_free_microsoft_ad_library_all()
        if "ms_api_df" in st.session_state:
            ms_df = st.session_state["ms_api_df"]
            if ms_df.empty:
                st.info("No EU/EEA Microsoft Ads returned for these competitors (use the direct US Ad Library links above for US-only campaigns).")
            else:
                st.dataframe(ms_df, use_container_width=True, hide_index=True, key="tbl_ms_ads", column_config={"Ad Library Link": st.column_config.LinkColumn("Open Ad")})

# ==============================================================================
# TAB 6: INDUSTRY-FILTERED GOOGLE TRENDS, BING & FREE TOPIC RESEARCH
# ==============================================================================
with t_trends:
    st.markdown("#### Industry-Filtered Google Trends, Bing Search & Topic Research ($0)")
    st.markdown('<p class="cnxt-sec-sub">100% scoped to B2B outsourcing, staffing, healthcare RCM, finance, and IT—zero unrelated macro news.</p>', unsafe_allow_html=True)

    r_sub1, r_sub2, r_sub3 = st.tabs([
        "B2B Google Trends & Live Industry Radar",
        "Bing Search Performance",
        "Question & Reddit Pain-Point Finder",
    ])

    with r_sub1:
        tc1, tc2, tc3 = st.columns([2.2, 0.9, 0.9])
        gt_seeds = tc1.text_input("Compare up to 5 B2B Topics in Google Trends:", value="medical billing outsourcing, offshore accounting, virtual assistant outsourcing, nearshore software developers, bpo outsourcing", key="gt_in")
        gt_geo = tc2.selectbox("Region:", ["US", "CA", "AU", "GB"], index=0, key="gt_geo")
        gt_tf = tc3.selectbox("Window:", ["today 1-m", "today 3-m", "today 12-m"], index=0, key="gt_tf")

        seed_tup = tuple(s.strip().lower() for s in gt_seeds.split(",") if s.strip())[:5]
        iot_df, rising_df, gt_notes = fetch_free_google_trends_b2b(seed_tup, geo=gt_geo, timeframe=gt_tf)
        if gt_notes:
            st.caption(" • ".join(gt_notes))

        if not iot_df.empty:
            fig_gt = px.line(iot_df, x="date", y="interest_score", color="keyword", color_discrete_sequence=[CNXT_COLORS["blue"], CNXT_COLORS["sun"], CNXT_COLORS["teal"], CNXT_COLORS["twilight"], CNXT_COLORS["red"]], labels={"date": "", "interest_score": "Search Interest (0–100)", "keyword": ""})
            fig_gt.update_layout(height=310, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", yaxis=dict(gridcolor=CNXT_COLORS["line"]), legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(fig_gt, use_container_width=True)

        if not rising_df.empty:
            st.markdown("**Rising B2B Queries (Google Trends)**")
            sm_s, sm_l = [], []
            for _, rr in rising_df.iterrows():
                chk = check_topic_against_sitemap(rr["Rising B2B Query"], "", live_set, live_df, live_vec, live_mat)
                sm_s.append("Already Covered" if chk["already_exists_in_sitemap"] else "New Topic Gap")
                sm_l.append(chk["best_money_url"] if chk["has_strong_money_match"] else chk["closest_existing_url"])
            rising_df["Connext Status"] = sm_s
            rising_df["Target Connext Page"] = sm_l
            st.dataframe(rising_df, use_container_width=True, hide_index=True, key="tbl_gt_r", column_config={"Target Connext Page": st.column_config.LinkColumn("Target Page")})

        st.divider()
        st.markdown("#### Live B2B Industry Trend Headlines (Last 14 Days — Strictly Industry-Filtered)")
        news_vert = st.selectbox("Industry Feed:", list(INDUSTRY_NEWS_FEEDS.keys()), index=0, key="sel_news_vert")
        news_df = fetch_b2b_industry_trending_feed(news_vert)
        if news_df.empty:
            st.info("No recent headlines returned right now.")
        else:
            st.dataframe(news_df, use_container_width=True, hide_index=True, key="tbl_b2b_news", column_config={"Article Link": st.column_config.LinkColumn("Read Article")})

    with r_sub2:
        if has_bing_secret:
            if st.button("Refresh Bing Webmaster Data", key="btn_ref_bing"):
                fetch_free_bing_webmaster_queries.clear()
                st.rerun()
            bing_df, b_stat = fetch_free_bing_webmaster_queries(str(st.secrets["bing"]["api_key"]), st.secrets["bing"].get("site_url", "https://connextglobal.com/"), brand_regex, exclude_local_ph, custom_negatives)
            if bing_df.empty:
                st.warning(b_stat)
            else:
                st.success(b_stat)
                bing_df["Growth"] = bing_df["growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
                st.dataframe(bing_df[["query", "vertical", "bing_impr_recent", "bing_clicks", "Growth", "bing_rank_label"]], use_container_width=True, hide_index=True, key="tbl_bing", column_config={"query": "Bing Query", "vertical": "Vertical", "bing_impr_recent": "Recent Impr.", "bing_clicks": "Clicks", "bing_rank_label": "Bing Rank"})
        else:
            st.markdown(
                """<div class="cnxt-callout">
<strong>Connect Free Bing Webmaster Tools ($0):</strong> Go to <strong>bing.com/webmasters → Settings → API Access → Generate API Key</strong>, then add <code>[bing]</code> with <code>api_key</code> and <code>site_url = "https://connextglobal.com/"</code> in Streamlit Secrets. This tracks your Bing, Microsoft Copilot, and ChatGPT Search query visibility.
</div>""",
                unsafe_allow_html=True,
            )

    with r_sub3:
        rc1, rc2 = st.columns([1.2, 1.5])
        picked_seed = rc1.selectbox("Pick from Surging GSC Topics:", f_trends["topic"].tolist(), key="sel_res_seed")
        custom_seed = rc2.text_input("Or Type Any Custom B2B Topic:", placeholder="e.g., dental billing outsourcing", key="in_res_seed")
        active_seed = custom_seed.strip() or picked_seed

        col_sugg, col_red = st.columns(2)
        with col_sugg:
            st.markdown(f"**Google & Bing Live Autocomplete for `{active_seed}`**")
            sugg_df = fetch_free_google_and_bing_suggestions(active_seed)
            if sugg_df.empty:
                st.info("No autocomplete suggestions found.")
            else:
                sugg_df["Connext Status"] = [
                    "Already Covered" if check_topic_against_sitemap(q, "", live_set, live_df, live_vec, live_mat)["already_exists_in_sitemap"] else "New Gap"
                    for q in sugg_df["Search Suggestion"]
                ]
                st.dataframe(sugg_df, use_container_width=True, hide_index=True, key="tbl_sugg")
        with col_red:
            st.markdown(f"**Reddit B2B Buyer Discussions for `{active_seed}`**")
            rd_df = fetch_free_reddit_b2b_discussions(active_seed)
            if rd_df.empty:
                st.info("No recent Reddit threads matched this exact phrase—try a shorter 2-word topic.")
            else:
                st.dataframe(rd_df, use_container_width=True, hide_index=True, key="tbl_reddit", column_config={"Thread": st.column_config.LinkColumn("Open Thread")})

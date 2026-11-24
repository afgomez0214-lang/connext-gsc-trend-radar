import hmac
import json
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONNEXT GLOBAL CORPORATE BRAND TOKENS ---
CNXT_COLORS = {
    "blue": "#005AAC",
    "sun": "#E58B25",
    "yellow": "#F1C916",
    "red": "#FC4301",
    "twilight": "#071776",
    "midnight": "#071739",
    "teal": "#0F818A",
    "teal_blue": "#90CBDB",
    "sage": "#7E9E7F",
    "cream": "#F4F1E2",
    "ink": "#101720",
    "paper": "#FFFFFF",
    "ash": "#6B7280",
    "line": "#E5E2D2",
    "line_strong": "#C7C2A8",
}

CNXT_CHART_PALETTE = [
    "#005AAC",  # Connext Blue
    "#E58B25",  # Sun Orange
    "#0F818A",  # Teal
    "#071776",  # Twilight Navy
    "#7E9E7F",  # Sage
    "#FC4301",  # Flame Red
    "#F1C916",  # Gold Yellow
    "#90CBDB",  # Teal Blue
    "#071739",  # Midnight
]

CNXT_LOGO_COLOR = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6a72a1e48c2cd52f0e5f2578_CONNEXT%20MAIN%20LOGO%20COLOR.png"
CNXT_LOGO_WHITE = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6ad6f9596637ac5988a5bf28_CONNEXT%20MAIN%20LOGO%20WHITE%20V.png"
CNXT_FAVICON = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6a965e51559ff3d4d6cad70f_Connext_Icons_Connext%20Logo-lg.png"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Connext Global | Search & AI Visibility Trend Radar",
    page_icon=CNXT_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_connext_brand_css():
    """Applies Connext Global Webflow typography, color tokens, hero banner, and card styles."""
    css = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Arimo:wght@400;500;600;700&family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root { --cnxt-blue: #005AAC; --cnxt-sun: #E58B25; --cnxt-yellow: #F1C916; --cnxt-red: #FC4301; --cnxt-twilight: #071776; --cnxt-midnight: #071739; --cnxt-teal: #0F818A; --cnxt-teal-blue: #90CBDB; --cnxt-sage: #7E9E7F; --cnxt-cream: #F4F1E2; --cnxt-paper: #FFFFFF; --cnxt-ash: #6B7280; --cnxt-line: #E5E2D2; --cnxt-line-strong: #C7C2A8; --font-display: 'Arimo', Arial, sans-serif; --font-body: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif; --font-mono: 'JetBrains Mono', ui-monospace, monospace; }
html, body, [class*="css"] { font-family: var(--font-body); color: var(--cnxt-midnight); }
h1, h2, h3, h4 { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; letter-spacing: -0.015em; }
.stApp { background-color: var(--cnxt-paper); }
section[data-testid="stSidebar"] { background-color: var(--cnxt-midnight) !important; border-right: 1px solid rgba(144, 203, 219, 0.18); }
section[data-testid="stSidebar"] * { color: var(--cnxt-paper); }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small { color: var(--cnxt-teal-blue) !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--cnxt-paper) !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea { background-color: rgba(255, 255, 255, 0.08) !important; border: 1px solid rgba(144, 203, 219, 0.35) !important; color: var(--cnxt-paper) !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background-color: rgba(255, 255, 255, 0.06) !important; border: 1px dashed var(--cnxt-teal-blue) !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background-color: var(--cnxt-blue) !important; color: var(--cnxt-paper) !important; border: none !important; }
section[data-testid="stSidebar"] .stButton > button { background-color: rgba(255, 255, 255, 0.10) !important; color: var(--cnxt-teal-blue) !important; border: 1px solid rgba(144, 203, 219, 0.35) !important; border-radius: 999px !important; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background-color: var(--cnxt-sun) !important; color: var(--cnxt-midnight) !important; border: none !important; }
.stButton > button[kind="primary"], .stDownloadButton > button { background-color: var(--cnxt-sun) !important; color: var(--cnxt-midnight) !important; font-family: var(--font-body) !important; font-weight: 600 !important; border: none !important; border-radius: 999px !important; padding: 0.55rem 1.4rem !important; box-shadow: 0 6px 18px -8px rgba(229, 139, 37, 0.65) !important; transition: all 0.2s ease !important; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { background-color: var(--cnxt-twilight) !important; color: var(--cnxt-cream) !important; transform: translateY(-1px); }
.cnxt-hero-banner { background: linear-gradient(135deg, #071776 0%, #071739 100%); color: #FFFFFF; padding: 28px 34px; border-radius: 10px; margin-bottom: 20px; position: relative; overflow: hidden; border: 1px solid rgba(144, 203, 219, 0.22); }
.cnxt-hero-banner::after { content: ""; position: absolute; top: -90px; right: -90px; width: 280px; height: 280px; background: radial-gradient(circle, rgba(241, 201, 22, 0.22), transparent 65%); pointer-events: none; }
.cnxt-eyebrow { font-family: var(--font-mono); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.18em; color: #F1C916; margin-bottom: 8px; font-weight: 600; }
.cnxt-hero-title { font-family: var(--font-display); font-size: clamp(1.55rem, 2.6vw, 2.15rem); font-weight: 700; color: #FFFFFF; line-height: 1.18; margin: 0 0 8px 0; }
.cnxt-hero-title span { color: #E58B25; }
.cnxt-hero-sub { color: #90CBDB; font-size: 0.94rem; margin: 0; max-width: 880px; }
div[data-testid="stMetric"] { background-color: var(--cnxt-paper); border: 1px solid var(--cnxt-line); border-top: 3px solid var(--cnxt-blue); border-radius: 8px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(7, 23, 57, 0.04); }
div[data-testid="stMetric"] label { font-family: var(--font-mono) !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; color: var(--cnxt-ash) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid var(--cnxt-line); }
.stTabs [data-baseweb="tab"] { font-family: var(--font-display); font-weight: 600; color: var(--cnxt-midnight); background-color: var(--cnxt-cream); border-radius: 6px 6px 0 0; padding: 10px 18px; border: 1px solid var(--cnxt-line); border-bottom: none; }
.stTabs [aria-selected="true"] { background-color: var(--cnxt-twilight) !important; color: var(--cnxt-paper) !important; border-color: var(--cnxt-twilight) !important; }
.cnxt-cream-panel { background-color: var(--cnxt-cream); border: 1px solid var(--cnxt-line-strong); border-left: 4px solid var(--cnxt-sun); border-radius: 6px; padding: 16px 20px; margin: 12px 0; color: var(--cnxt-midnight); }
.cnxt-cream-panel strong { color: var(--cnxt-twilight); }
span[data-baseweb="tag"] { background-color: var(--cnxt-twilight) !important; color: var(--cnxt-paper) !important; }
</style>"""
    st.markdown(css, unsafe_allow_html=True)


inject_connext_brand_css()


def require_connext_login():
    """Blocks access to the app until a valid team password from st.secrets is entered."""
    if st.session_state.get("cnxt_authenticated"):
        return

    st.markdown(
        "<style>section[data-testid='stSidebar']{display:none;}</style>",
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.35, 1])
    with center_col:
        st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
        st.image(CNXT_LOGO_COLOR, width=190)
        st.markdown(
            """
            <div class="cnxt-hero-banner" style="margin-top: 14px; padding: 24px 28px;">
                <div class="cnxt-eyebrow">INTERNAL TEAM ACCESS</div>
                <h2 style="color:#FFFFFF !important; margin:0 0 6px 0; font-size:1.5rem;">
                    Connext Search &amp; AI <span style="color:#E58B25;">Trend Radar</span>
                </h2>
                <p class="cnxt-hero-sub" style="font-size:0.88rem;">
                    Sign in with your Connext team password to access Search Console trend data, page mappings, and Paid Ads keyword queues.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "app_password" not in st.secrets:
            st.error("`app_password` is not configured in Streamlit Secrets yet. Add it in App Settings → Secrets.")
            st.stop()

        with st.form("cnxt_login_form"):
            entered_pwd = st.text_input("Team Password", type="password", placeholder="Enter password...")
            submitted = st.form_submit_button("Unlock Trend Radar", type="primary", use_container_width=True)

            if submitted:
                if hmac.compare_digest(entered_pwd, str(st.secrets["app_password"])):
                    st.session_state["cnxt_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")

    st.stop()


require_connext_login()

DEFAULT_SITE_URL = "sc-domain:connextglobal.com"
DEFAULT_BRAND_REGEX = r"\b(connext|connext\s*global(\s*solutions)?|connextglobal)\b"

# Strict filter removing job seekers, applicants, employee portals, and non-B2B noise
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

# Commercial / Transactional B2B Outsourcing Modifiers
COMMERCIAL_MODIFIERS = re.compile(
    r"\b("
    r"outsourcing|outsource|staffing|offshore|nearshore|dedicated|bpo|agency|company|firm|"
    r"services|solutions|cost|pricing|rates|quote|remote team|vs|versus|best|top|partner|provider|hire"
    r")\b",
    re.I,
)

# Conversational / AI Mode Prompt Signals (7+ words or multi-part question/comparison prompts)
AI_PROMPT_REGEX = re.compile(
    r"^(how|what|why|when|where|which|can|does|is|are|should|compare|difference between|pros and cons|steps to|best way to)\b",
    re.I,
)

# Complete Connext Global B2B Vertical Taxonomy
SECTOR_PATTERNS = [
    (
        "Healthcare & RCM",
        re.compile(
            r"\b(medical|rcm|revenue cycle|healthcare|health care|billing|coding|icd|cpt|patient|"
            r"prior auth|authorization|credentialing|clinic|hospital|dental|hipaa|ehr|emr|pharmacy)\b",
            re.I,
        ),
    ),
    (
        "Finance, Accounting & Tax",
        re.compile(
            r"\b(accounting|accountant|bookkeeping|bookkeeper|accounts payable|accounts receivable|"
            r"cpa|controller|cfo|tax|audit|auditing|payroll|finance|financial|reconciliation|ap|ar)\b",
            re.I,
        ),
    ),
    (
        "Tech, IT & Engineering",
        re.compile(
            r"\b(software|developer|developers|engineer|engineering|it support|helpdesk|help desk|"
            r"qa|quality assurance|testing|data|cloud|devops|cybersecurity|tech|programmer|ui|ux|web)\b",
            re.I,
        ),
    ),
    (
        "Customer Experience (CX) & Support",
        re.compile(
            r"\b(customer service|customer care|customer support|client support|client success|"
            r"call center|contact center|omnichannel|live chat|tier 1|tier 2|technical support)\b",
            re.I,
        ),
    ),
    (
        "Back Office & Data Operations",
        re.compile(
            r"\b(back office|back-office|admin|administrative|data entry|document processing|"
            r"order processing|catalog|content moderation|virtual assistant|transcription|indexing)\b",
            re.I,
        ),
    ),
    (
        "Legal, Insurance & Compliance",
        re.compile(
            r"\b(legal|paralegal|attorney|law firm|contract|compliance|underwriting|insurance|"
            r"claims|policy|risk management|title search|escrow|regulatory)\b",
            re.I,
        ),
    ),
    (
        "Sales, Marketing & Creative",
        re.compile(
            r"\b(sdr|bdr|sales development|lead generation|marketing|seo|paid media|ppc|graphic design|"
            r"video editing|copywriting|social media|hubspot|salesforce|crm|ecommerce|e-commerce)\b",
            re.I,
        ),
    ),
    (
        "Logistics, Supply Chain & Real Estate",
        re.compile(
            r"\b(logistics|freight|dispatch|supply chain|procurement|inventory|warehouse|shipping|"
            r"real estate|property management|mortgage|leasing|construction|estimating)\b",
            re.I,
        ),
    ),
]


def normalize_url(url: str) -> str:
    """Normalizes URLs so GSC Web API URLs and GSC Generative AI Report URLs join reliably."""
    u = str(url).strip().lower()
    u = u.replace("http://", "https://").replace("https://www.", "https://")
    if u.endswith("/") and len(u) > len("https://connextglobal.com/"):
        u = u.rstrip("/")
    return u


def classify_sector(text: str) -> str:
    for sector_name, pattern in SECTOR_PATTERNS:
        if pattern.search(text):
            return sector_name
    return "General B2B Outsourcing & Staffing"


def rank_tier_label(pos: float) -> str:
    if pos <= 3.0:
        return "Top 3 (#1–3)"
    if pos <= 10.0:
        return "Page 1 (#4–10)"
    if pos <= 20.0:
        return "Page 2 (#11–20)"
    return "Deep (#21+)"


def build_rsa_assets(topic: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Generates Google Ads Headlines (<=30 chars) & Descriptions (<=90 chars) aligned with Connext Global brand voice."""
    clean_kw = re.sub(r"\s+", " ", topic.strip()).title()
    words = clean_kw.split()

    short_kw = ""
    for w in words:
        candidate = f"{short_kw} {w}".strip()
        if len(candidate) <= 22:
            short_kw = candidate
        else:
            break
    if not short_kw:
        short_kw = clean_kw[:22].strip()

    raw_headlines = [
        ("Keyword", short_kw[:30]),
        ("Keyword", f"Outsource {short_kw}"),
        ("Keyword", f"Dedicated {short_kw}"),
        ("Value", "Build Your Dedicated Team"),
        ("Value", "Global Scale, Local Touch"),
        ("Value", "Custom Offshore Staffing"),
        ("Value", "Seamless Team Integration"),
        ("CTA", "Scale Your Team Today"),
        ("CTA", "Book a Staffing Consultation"),
        ("CTA", "Request Custom Team Pricing"),
    ]

    valid_headlines = []
    seen_h = set()
    for category, h in raw_headlines:
        h_clean = h.strip()
        if len(h_clean) <= 30 and h_clean.lower() not in seen_h:
            valid_headlines.append((category, h_clean))
            seen_h.add(h_clean.lower())

    raw_descriptions = [
        f"Extend your team with dedicated {short_kw.lower()} talent built around your workflows.",
        "Overcome talent shortages and high overhead with transparent, custom offshore staffing.",
        "People-centric global teams that integrate seamlessly with your culture and operations.",
        "Scale faster with full operational support, high retention, and dedicated management.",
    ]
    valid_descriptions = [d[:90].strip() for d in raw_descriptions if len(d) <= 90]
    return valid_headlines, valid_descriptions


# --- DEMO DATA GENERATORS (WEB SEARCH + GSC GENERATIVE AI REPORT) ---
@st.cache_data
def generate_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates 45 days of Web Search data + GSC Generative AI (AI Overviews / AI Mode) page data."""
    np.random.seed(42)
    end_date = datetime.utcnow().date() - timedelta(days=2)
    dates = [end_date - timedelta(days=i) for i in range(44, -1, -1)]

    topics = [
        ("outsourced medical billing services", "/solutions/healthcare-rcm", 65, 80, [85, 90, 100, 125, 160, 210, 275, 360, 460, 580], 10.2, 8.1, False),
        ("how to outsource prior authorization in healthcare without losing control", "/blog/outsource-prior-authorization", 35, 45, [50, 55, 62, 80, 105, 140, 185, 240, 310, 390], 2.1, 1.4, False),
        ("what is the average cost of revenue cycle management outsourcing", "/blog/rcm-outsourcing-pricing", 30, 38, [42, 46, 52, 68, 92, 125, 165, 215, 275, 340], 2.4, 1.6, False),
        ("connext medical billing outsourcing", "/solutions/healthcare-rcm", 15, 18, [20, 22, 24, 26, 30, 35, 42, 48, 55, 62], 1.4, 1.2, True),
        ("offshore accounts payable outsourcing", "/solutions/finance-accounting", 55, 95, [110, 120, 135, 155, 185, 220, 260, 310, 365, 425], 6.8, 5.1, False),
        ("how do cpa firms scale tax season with offshore bookkeepers", "/blog/offshore-bookkeeping-cpa", 30, 55, [65, 72, 80, 95, 115, 140, 170, 205, 245, 290], 1.9, 1.3, False),
        ("outsource accounts payable process checklist", "/solutions/finance-accounting", 22, 35, [40, 46, 52, 64, 78, 95, 115, 138, 165, 195], 9.1, 7.2, False),
        ("hire dedicated qa automation engineers philippines", "/solutions/tech-it", 40, 42, [45, 44, 48, 52, 60, 75, 110, 195, 290, 410], 14.5, 13.8, False),
        ("offshore software development team staffing", "/solutions/tech-it", 50, 54, [55, 58, 60, 65, 78, 95, 130, 210, 300, 395], 13.2, 12.1, False),
        ("offshore software development team vs freelancer pros and cons", "/blog/offshore-dev-team-guide", 20, 22, [24, 25, 26, 28, 34, 42, 58, 90, 130, 175], 2.2, 1.5, False),
        ("b2b omnichannel customer support outsourcing", "/solutions/customer-experience", 45, 110, [130, 140, 150, 160, 175, 190, 205, 220, 240, 260], 6.2, 4.6, False),
        ("24 7 technical support call center philippines", "/solutions/customer-experience", 35, 85, [95, 102, 110, 118, 128, 140, 152, 165, 180, 195], 7.1, 5.5, False),
        ("insurance claims processing outsourcing services", "/solutions/back-office", 35, 42, [45, 50, 58, 72, 95, 125, 160, 200, 250, 305], 10.4, 8.9, False),
        ("remote paralegal support for law firms", "/blog/offshore-paralegal-staffing", 25, 30, [34, 38, 45, 58, 76, 98, 125, 158, 195, 240], 12.6, 10.8, False),
        ("outsourced b2b sdr lead generation team", "/solutions/sales-marketing", 40, 48, [52, 58, 66, 82, 105, 135, 170, 215, 265, 320], 9.2, 7.6, False),
        ("offshore freight dispatch and logistics support", "/solutions/logistics", 30, 72, [82, 88, 95, 102, 112, 122, 134, 146, 160, 175], 7.8, 6.1, False),
        ("dedicated offshore staffing partner philippines", "/how-it-works", 180, 210, [225, 235, 250, 275, 310, 355, 410, 475, 550, 635], 2.3, 1.8, False),
        ("connext global offshore staffing", "/how-it-works", 50, 55, [58, 60, 64, 68, 75, 82, 90, 98, 108, 120], 1.1, 1.0, True),
        ("virtual assistant outsourcing overview", "/blog/virtual-assistant-guide", 260, 265, [260, 270, 255, 265, 260, 275, 260, 255, 270, 265], 4.4, 4.5, False),
    ]

    web_rows = []
    for idx, d in enumerate(dates):
        weekend_factor = 0.85 if d.weekday() in (5, 6) else 1.0
        progress = idx / max(len(dates) - 1, 1)
        for query, page, early_b, mid_b, curve_10d, start_pos, end_pos, is_brand in topics:
            if idx < 20:
                base_imp = early_b * np.random.uniform(0.88, 1.12)
            elif idx < 35:
                base_imp = mid_b * np.random.uniform(0.90, 1.10)
            else:
                base_imp = curve_10d[idx - 35]

            imp = int(max(10, base_imp * weekend_factor * np.random.uniform(0.95, 1.05)))
            interp_pos = start_pos + (end_pos - start_pos) * progress
            pos = round(max(1.0, interp_pos + np.random.uniform(-0.3, 0.3)), 1)
            web_rows.append({
                "date": pd.to_datetime(d),
                "query": query,
                "page": f"https://connextglobal.com{page}",
                "impressions": imp,
                "position": pos,
                "is_branded": is_brand,
            })

    ai_pages_spec = [
        ("/blog/outsource-prior-authorization", 25, 35, [42, 48, 58, 75, 100, 135, 180, 235, 305, 385]),
        ("/blog/rcm-outsourcing-pricing", 20, 30, [36, 42, 50, 65, 88, 120, 160, 210, 270, 340]),
        ("/blog/offshore-bookkeeping-cpa", 22, 45, [55, 64, 75, 90, 112, 138, 170, 208, 250, 300]),
        ("/blog/offshore-dev-team-guide", 15, 18, [20, 22, 24, 28, 35, 45, 62, 95, 140, 190]),
        ("/solutions/healthcare-rcm", 30, 38, [44, 50, 58, 72, 92, 118, 150, 190, 235, 290]),
        ("/solutions/finance-accounting", 28, 52, [62, 70, 80, 95, 115, 140, 170, 205, 245, 290]),
        ("/how-it-works", 60, 75, [85, 92, 102, 115, 132, 155, 185, 220, 260, 305]),
        ("/blog/virtual-assistant-guide", 90, 92, [88, 94, 90, 92, 89, 95, 91, 90, 93, 91]),
    ]

    ai_rows = []
    for idx, d in enumerate(dates):
        weekend_factor = 0.86 if d.weekday() in (5, 6) else 1.0
        for page, early_b, mid_b, curve_10d in ai_pages_spec:
            if idx < 20:
                base_imp = early_b * np.random.uniform(0.88, 1.12)
            elif idx < 35:
                base_imp = mid_b * np.random.uniform(0.90, 1.10)
            else:
                base_imp = curve_10d[idx - 35]
            ai_imp = int(max(5, base_imp * weekend_factor * np.random.uniform(0.95, 1.05)))
            ai_rows.append({
                "date": pd.to_datetime(d),
                "page": f"https://connextglobal.com{page}",
                "ai_impressions": ai_imp,
            })

    return pd.DataFrame(web_rows), pd.DataFrame(ai_rows)


# --- PARSE GSC GENERATIVE AI PERFORMANCE REPORT CSV ---
def parse_gsc_ai_report_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Parses exports from Google Search Console's 'Generative AI' performance report."""
    df = pd.read_csv(uploaded_file)
    orig_cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in orig_cols}

    page_col = None
    for candidate in ["top pages", "page", "pages", "url", "landing page"]:
        if candidate in lower_map:
            page_col = lower_map[candidate]
            break
    if page_col is None:
        raise ValueError("Could not find a 'Page' or 'Top pages' column in the Generative AI CSV.")

    if "date" in lower_map:
        date_col = lower_map["date"]
        imp_col = lower_map.get("impressions") or lower_map.get("ai impressions")
        if imp_col is None:
            num_cols = [c for c in orig_cols if pd.api.types.is_numeric_dtype(df[c])]
            imp_col = num_cols[0] if num_cols else None
        if imp_col is None:
            raise ValueError("Could not find an Impressions column in the Generative AI CSV.")

        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col]),
            "page": df[page_col].astype(str).apply(normalize_url),
            "ai_impressions": pd.to_numeric(df[imp_col], errors="coerce").fillna(0).astype(int),
        })
        return out, "timeseries"

    imp_cols = [c for c in orig_cols if "impression" in c.lower() and "diff" not in c.lower()]
    if len(imp_cols) >= 2:
        out = pd.DataFrame({
            "page": df[page_col].astype(str).apply(normalize_url),
            "ai_imp_current": pd.to_numeric(df[imp_cols[0]], errors="coerce").fillna(0).astype(int),
            "ai_imp_previous": pd.to_numeric(df[imp_cols[1]], errors="coerce").fillna(0).astype(int),
        })
        return out, "comparison"
    elif len(imp_cols) == 1:
        out = pd.DataFrame({
            "page": df[page_col].astype(str).apply(normalize_url),
            "ai_imp_current": pd.to_numeric(df[imp_cols[0]], errors="coerce").fillna(0).astype(int),
            "ai_imp_previous": 0,
        })
        return out, "snapshot"

    raise ValueError("Unrecognized GSC Generative AI CSV format. Export the 'Pages' table from GSC -> Performance -> Generative AI.")


# --- GSC API FETCHING (IMPRESSIONS & POSITION ONLY) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gsc_api_data(
    creds_dict: dict,
    site_url: str,
    lookback_days: int,
    brand_regex: str,
    custom_exclude_regex: str,
) -> tuple[pd.DataFrame, str, int]:
    """Pulls daily query+page impressions & organic position, tagging branded vs non-branded."""
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
    )
    service = build("searchconsole", "v1", credentials=creds)

    end_dt = datetime.utcnow().date()
    start_dt = end_dt - timedelta(days=lookback_days)

    rows = []
    excluded_count = 0
    start_row = 0
    row_limit = 25000
    first_incomplete = None

    brand_pattern = re.compile(brand_regex, re.I) if brand_regex.strip() else None
    custom_pattern = re.compile(custom_exclude_regex, re.I) if custom_exclude_regex.strip() else None

    while True:
        body = {
            "startDate": start_dt.isoformat(),
            "endDate": end_dt.isoformat(),
            "dimensions": ["date", "query", "page"],
            "dataState": "all",
            "rowLimit": row_limit,
            "startRow": start_row,
        }

        response = service.searchanalytics().query(siteUrl=site_url, body=body).execute()

        if first_incomplete is None:
            first_incomplete = response.get("metadata", {}).get("firstIncompleteDate", "None")

        batch = response.get("rows", [])
        if not batch:
            break

        for r in batch:
            date_str, query, page = r["keys"]
            if first_incomplete != "None" and date_str >= first_incomplete:
                continue
            if STRICT_NON_INDUSTRY_REGEX.search(query):
                excluded_count += 1
                continue
            if custom_pattern and custom_pattern.search(query):
                excluded_count += 1
                continue

            is_brand = bool(brand_pattern.search(query)) if brand_pattern else False
            rows.append({
                "date": date_str,
                "query": query.lower().strip(),
                "page": page,
                "impressions": r["impressions"],
                "position": r["position"],
                "is_branded": is_brand,
            })

        if len(batch) < row_limit:
            break
        start_row += row_limit

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df, str(first_incomplete), excluded_count


# --- CLUSTERING & MULTI-HORIZON TREND SCORING ---
@st.cache_data(show_spinner=False)
def cluster_queries(df: pd.DataFrame, distance_threshold: float) -> pd.DataFrame:
    """Clusters top non-branded B2B queries into semantic topic groups and tags long-tail & AI prompt terms."""
    non_brand = df[~df["is_branded"]].copy()
    non_brand["norm_page"] = non_brand["page"].apply(normalize_url)

    query_stats = (
        non_brand.groupby("query", as_index=False)
        .agg(total_impressions=("impressions", "sum"))
        .sort_values("total_impressions", ascending=False)
    )

    top_queries = query_stats.head(3500).copy()
    if len(top_queries) < 2:
        non_brand["topic"] = non_brand["query"]
        non_brand["word_count"] = non_brand["query"].str.split().str.len()
        non_brand["is_long_tail"] = non_brand["word_count"] >= 4
        non_brand["is_ai_prompt"] = (non_brand["word_count"] >= 7) | non_brand["query"].str.contains(AI_PROMPT_REGEX, na=False)
        return non_brand

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(top_queries["query"]).astype(np.float32).toarray()

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    top_queries["cluster_id"] = clustering.fit_predict(tfidf_matrix)

    topic_names = (
        top_queries.sort_values("total_impressions", ascending=False)
        .groupby("cluster_id")["query"]
        .first()
        .to_dict()
    )
    top_queries["topic"] = top_queries["cluster_id"].map(topic_names)

    merged = non_brand.merge(top_queries[["query", "topic"]], on="query", how="inner")
    merged["word_count"] = merged["query"].str.split().str.len()
    merged["is_long_tail"] = merged["word_count"] >= 4
    merged["is_ai_prompt"] = (merged["word_count"] >= 7) | merged["query"].str.contains(AI_PROMPT_REGEX, na=False)
    return merged


def analyze_multi_horizon_trends(
    non_brand_df: pd.DataFrame,
    full_df: pd.DataFrame,
    min_daily_imp: int,
    min_up_days: int,
    min_norm_slope: float,
    min_lift: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyzes impressions and Connext organic position across Today, This Week, and This Month."""
    all_dates = sorted(non_brand_df["date"].unique())
    n_days = len(all_dates)
    if n_days < 14:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    today_dates = all_dates[-3:]
    week_dates = all_dates[-7:]
    prior_14_dates = all_dates[-21:-7] if n_days >= 21 else all_dates[:-7]

    month_len = min(21, n_days // 2)
    month_recent_dates = all_dates[-month_len:]
    month_prior_dates = all_dates[-(month_len * 2):-month_len] if n_days >= month_len * 2 else all_dates[:-month_len]

    df = non_brand_df.copy()
    df["weighted_pos"] = df["position"] * df["impressions"]

    daily_topic = (
        df.groupby(["topic", "date"], as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            weighted_pos=("weighted_pos", "sum"),
        )
    )
    daily_topic["organic_position"] = (daily_topic["weighted_pos"] / np.maximum(daily_topic["impressions"], 1)).round(1)

    recent_df = df[df["date"].isin(week_dates)]
    prior_df = df[df["date"].isin(prior_14_dates)]

    brand_recent = full_df[(full_df["is_branded"]) & (full_df["date"].isin(week_dates))].copy()
    if not brand_recent.empty:
        brand_recent["weighted_pos"] = brand_recent["position"] * brand_recent["impressions"]

    top_queries_map = (
        recent_df.groupby(["topic", "query"], as_index=False)["impressions"].sum()
        .sort_values("impressions", ascending=False)
        .groupby("topic")["query"]
        .apply(list)
        .to_dict()
    )
    long_tail_map = (
        recent_df[recent_df["is_long_tail"]]
        .groupby(["topic", "query"], as_index=False)["impressions"].sum()
        .sort_values("impressions", ascending=False)
        .groupby("topic")["query"]
        .apply(list)
        .to_dict()
    )

    pages_agg = (
        recent_df.groupby(["topic", "page", "norm_page"], as_index=False)
        .agg(
            page_imp_7d=("impressions", "sum"),
            page_weighted_pos=("weighted_pos", "sum"),
            queries_on_page=("query", "nunique"),
        )
        .sort_values(["topic", "page_imp_7d"], ascending=[True, False])
    )
    pages_agg["page_org_pos"] = (pages_agg["page_weighted_pos"] / np.maximum(pages_agg["page_imp_7d"], 1)).round(1)

    top_page_map = pages_agg.groupby("topic")["page"].first().to_dict()
    all_pages_map = {
        t: " | ".join(f"{r['page']} (#{r['page_org_pos']})" for _, r in g.iterrows())
        for t, g in pages_agg.groupby("topic")
    }
    page_count_map = pages_agg.groupby("topic")["page"].nunique().to_dict()

    results = []
    for topic, group in daily_topic.groupby("topic"):
        series = group.set_index("date")["impressions"].reindex(all_dates, fill_value=0)

        arr_today = series.loc[today_dates].values
        arr_week = series.loc[week_dates].values
        arr_prior14 = series.loc[prior_14_dates].values
        arr_month_rec = series.loc[month_recent_dates].values
        arr_month_base = series.loc[month_prior_dates].values

        mean_today = float(np.mean(arr_today))
        mean_week = float(np.mean(arr_week))
        mean_prior14 = max(float(np.mean(arr_prior14)), 1.0)
        mean_month_rec = float(np.mean(arr_month_rec))
        mean_month_base = max(float(np.mean(arr_month_base)), 1.0)

        if max(mean_today, mean_week, mean_month_rec) < min_daily_imp:
            continue

        lift_today = mean_today / mean_prior14
        latest_day_up = arr_today[-1] > arr_today[0]
        is_trending_today = bool(mean_today >= min_daily_imp and lift_today >= (min_lift + 0.15) and latest_day_up)

        diffs_week = np.diff(arr_week)
        up_days_week = int(np.sum(diffs_week > 0))
        slope_week = float(np.polyfit(np.arange(len(arr_week)), arr_week, 1)[0])
        norm_slope_week = slope_week / max(mean_week, 1.0)
        lift_week = mean_today / mean_prior14
        is_trending_week = bool(
            mean_today >= min_daily_imp
            and up_days_week >= min_up_days
            and norm_slope_week >= min_norm_slope
            and lift_week >= min_lift
        )

        lift_month = mean_month_rec / mean_month_base
        slope_month = float(np.polyfit(np.arange(len(arr_month_rec)), arr_month_rec, 1)[0])
        norm_slope_month = slope_month / max(mean_month_rec, 1.0)
        is_trending_month = bool(
            mean_month_rec >= (min_daily_imp * 0.8)
            and lift_month >= max(1.25, min_lift - 0.15)
            and norm_slope_month >= (min_norm_slope * 0.35)
        )

        if not (is_trending_today or is_trending_week or is_trending_month):
            continue

        rec_grp = group[group["date"].isin(week_dates)]
        tot_imp_7d = int(rec_grp["impressions"].sum())
        tot_imp_month = int(group[group["date"].isin(month_recent_dates)]["impressions"].sum())

        avg_pos_7d = float(rec_grp["weighted_pos"].sum() / max(tot_imp_7d, 1))
        pri_grp = prior_df[prior_df["topic"] == topic]
        if not pri_grp.empty and pri_grp["impressions"].sum() > 0:
            avg_pos_prior = float(pri_grp["weighted_pos"].sum() / pri_grp["impressions"].sum())
            pos_delta = round(avg_pos_prior - avg_pos_7d, 1)
        else:
            pos_delta = 0.0

        topic_keywords = [w for w in topic.split() if len(w) > 3]
        brand_pos_label = "No Brand Queries"
        if not brand_recent.empty and topic_keywords:
            kw_pat = re.compile("|".join(re.escape(w) for w in topic_keywords), re.I)
            matched_brand = brand_recent[brand_recent["query"].str.contains(kw_pat, na=False)]
            if not matched_brand.empty and matched_brand["impressions"].sum() > 0:
                b_pos = matched_brand["weighted_pos"].sum() / matched_brand["impressions"].sum()
                b_imp = int(matched_brand["impressions"].sum())
                brand_pos_label = f"#{b_pos:.1f} ({b_imp:,} imp)"

        queries_list = top_queries_map.get(topic, [topic])
        lt_list = long_tail_map.get(topic, [])
        joined_queries = ", ".join(queries_list[:6])
        has_commercial = bool(COMMERCIAL_MODIFIERS.search(joined_queries))
        unique_pages = int(page_count_map.get(topic, 1))
        sector = classify_sector(joined_queries)

        velocity_score = round(
            max(norm_slope_week * 100, norm_slope_month * 150) * np.log1p(max(lift_today, lift_month)) * (max(up_days_week, 3) / 6),
            1,
        )

        horizons = []
        if is_trending_today:
            horizons.append("Today")
        if is_trending_week:
            horizons.append("This Week")
        if is_trending_month:
            horizons.append("This Month")

        rank_gap_bonus = 30 if avg_pos_7d > 8 else (20 if avg_pos_7d > 4 else 5)
        intent_bonus = 35 if has_commercial else 10
        surge_bonus = min(35, int((lift_today - 1.0) * 18))
        ads_fit_score = min(100, rank_gap_bonus + intent_bonus + surge_bonus)

        results.append({
            "topic": topic,
            "sector": sector,
            "horizons": " | ".join(horizons),
            "is_today": is_trending_today,
            "is_week": is_trending_week,
            "is_month": is_trending_month,
            "velocity_score": velocity_score,
            "ads_fit_score": ads_fit_score,
            "up_days_of_6": up_days_week,
            "daily_growth_%": round(norm_slope_week * 100, 1),
            "today_lift_x": round(lift_today, 2),
            "month_lift_x": round(lift_month, 2),
            "impressions_7d": tot_imp_7d,
            "impressions_month": tot_imp_month,
            "sparkline_7d": [int(v) for v in arr_week],
            "connext_org_pos": round(avg_pos_7d, 1),
            "rank_change": pos_delta,
            "rank_tier": rank_tier_label(avg_pos_7d),
            "brand_org_pos": brand_pos_label,
            "commercial_intent": has_commercial,
            "intent_label": "Commercial / B2B" if has_commercial else "Informational / Research",
            "ranking_pages_count": unique_pages,
            "primary_page": top_page_map.get(topic, ""),
            "all_related_pages": all_pages_map.get(topic, ""),
            "queries_list": queries_list,
            "long_tail_list": lt_list,
            "total_queries": len(queries_list),
            "long_tail_count": len(lt_list),
            "top_rising_queries": joined_queries,
        })

    if not results:
        return pd.DataFrame(), daily_topic, pd.DataFrame(), pd.DataFrame()

    trends_df = pd.DataFrame(results).sort_values("velocity_score", ascending=False).reset_index(drop=True)

    q_recent = (
        recent_df.groupby(["topic", "query", "is_long_tail", "is_ai_prompt", "word_count"], as_index=False)
        .agg(
            imp_7d=("impressions", "sum"),
            weighted_pos=("weighted_pos", "sum"),
            top_page=("page", "first"),
            norm_page=("norm_page", "first"),
        )
    )
    q_base = prior_df.groupby("query", as_index=False).agg(imp_base14=("impressions", "sum"))
    q_merged = q_recent.merge(q_base, on="query", how="left").fillna({"imp_base14": 0})
    q_merged["daily_7d"] = q_merged["imp_7d"] / 7.0
    q_merged["daily_base"] = np.maximum(q_merged["imp_base14"] / max(len(prior_14_dates), 1), 0.5)
    q_merged["query_lift_x"] = (q_merged["daily_7d"] / q_merged["daily_base"]).round(2)
    q_merged["connext_org_pos"] = (q_merged["weighted_pos"] / np.maximum(q_merged["imp_7d"], 1)).round(1)
    q_merged["sector"] = q_merged["query"].apply(classify_sector)
    q_merged["is_commercial"] = q_merged["query"].str.contains(COMMERCIAL_MODIFIERS, na=False)
    q_merged = q_merged.sort_values(["imp_7d", "query_lift_x"], ascending=False)

    pages_merged = pages_agg[pages_agg["topic"].isin(trends_df["topic"])].copy()
    topic_tot_map = trends_df.set_index("topic")["impressions_7d"].to_dict()
    pages_merged["topic_imp_share_%"] = (
        (pages_merged["page_imp_7d"] / pages_merged["topic"].map(topic_tot_map).replace(0, 1)) * 100
    ).round(1)

    return trends_df, daily_topic, q_merged, pages_merged


# --- AI SEARCH (GENERATIVE AI REPORT + PAGE-TO-QUERY CONNECTOR) ---
def analyze_ai_pages_and_queries(
    ai_df: pd.DataFrame,
    ai_mode: str,
    non_brand_web_df: pd.DataFrame,
    all_queries_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Connects trending pages from GSC's Generative AI report with trending queries & clusters."""
    web = non_brand_web_df.copy()
    web["norm_page"] = web["page"].apply(normalize_url)
    web["weighted_pos"] = web["position"] * web["impressions"]

    all_dates = sorted(web["date"].unique())
    week_dates = all_dates[-7:]
    prior_14_dates = all_dates[-21:-7] if len(all_dates) >= 21 else all_dates[:-7]

    web_7d = (
        web[web["date"].isin(week_dates)]
        .groupby("norm_page", as_index=False)
        .agg(
            display_url=("page", "first"),
            web_imp_7d=("impressions", "sum"),
            web_wpos=("weighted_pos", "sum"),
        )
    )
    web_7d["web_org_pos"] = (web_7d["web_wpos"] / np.maximum(web_7d["web_imp_7d"], 1)).round(1)

    page_queries_grp = all_queries_df.sort_values(["query_lift_x", "imp_7d"], ascending=[False, False])
    connected_qs_map = {
        p: [f"{r['query']} ({r['query_lift_x']}x, #{r['connext_org_pos']})" for _, r in g.head(6).iterrows()]
        for p, g in page_queries_grp.groupby("norm_page")
    }
    raw_qs_map = page_queries_grp.groupby("norm_page")["query"].apply(list).to_dict()
    prompt_qs_map = (
        page_queries_grp[page_queries_grp["is_ai_prompt"]]
        .groupby("norm_page")["query"]
        .apply(list)
        .to_dict()
    )
    clusters_map = (
        page_queries_grp.groupby("norm_page")["topic"]
        .apply(lambda s: ", ".join(dict.fromkeys(s.tolist()[:3])))
        .to_dict()
    )

    ai_daily_series = pd.DataFrame()
    page_records = []

    if ai_df is not None and not ai_df.empty:
        if ai_mode == "timeseries":
            ai_ts = ai_df.copy()
            ai_ts["norm_page"] = ai_ts["page"].apply(normalize_url)
            ai_dates = sorted(ai_ts["date"].unique())
            ai_7d_dates = ai_dates[-7:]
            ai_base_dates = ai_dates[-21:-7] if len(ai_dates) >= 21 else ai_dates[:-7]

            ai_daily_series = ai_ts.groupby(["norm_page", "date"], as_index=False)["ai_impressions"].sum()

            for norm_p, grp in ai_daily_series.groupby("norm_page"):
                s = grp.set_index("date")["ai_impressions"].reindex(ai_dates, fill_value=0)
                rec_7 = s.loc[ai_7d_dates].values
                base_14 = s.loc[ai_base_dates].values if len(ai_base_dates) > 0 else s.values

                imp_7d = int(np.sum(rec_7))
                mean_3d = float(np.mean(rec_7[-3:]))
                mean_base = max(float(np.mean(base_14)), 1.0)
                ai_lift = round(mean_3d / mean_base, 2)
                ai_slope = float(np.polyfit(np.arange(len(rec_7)), rec_7, 1)[0])
                ai_growth_pct = round((ai_slope / max(float(np.mean(rec_7)), 1.0)) * 100, 1)

                page_records.append({
                    "norm_page": norm_p,
                    "ai_imp_7d": imp_7d,
                    "ai_lift_x": ai_lift,
                    "ai_daily_growth_%": ai_growth_pct,
                    "ai_sparkline": [int(v) for v in rec_7],
                    "data_source_tag": "GSC Generative AI Report",
                })
        else:
            for _, row in ai_df.iterrows():
                norm_p = row["page"]
                curr = int(row["ai_imp_current"])
                prev = int(row["ai_imp_previous"])
                lift = round(curr / max(prev, 1), 2) if prev > 0 else 1.0
                growth_pct = round(((curr - prev) / max(prev, 1)) * 100, 1) if prev > 0 else 0.0
                page_records.append({
                    "norm_page": norm_p,
                    "ai_imp_7d": curr,
                    "ai_lift_x": lift,
                    "ai_daily_growth_%": growth_pct,
                    "ai_sparkline": [prev, curr] if prev > 0 else [curr],
                    "data_source_tag": "GSC Generative AI CSV Export",
                })
    else:
        ai_proxy_df = web[
            (web["date"].isin(week_dates + prior_14_dates))
            & ((web["is_ai_prompt"]) | (web["position"] <= 2.5))
        ].copy()
        for norm_p, grp in ai_proxy_df.groupby("norm_page"):
            imp_7 = int(grp[grp["date"].isin(week_dates)]["impressions"].sum())
            imp_base = int(grp[grp["date"].isin(prior_14_dates)]["impressions"].sum())
            if imp_7 < 30:
                continue
            daily_7 = imp_7 / 7.0
            daily_b = max(imp_base / max(len(prior_14_dates), 1), 1.0)
            lift = round(daily_7 / daily_b, 2)
            page_records.append({
                "norm_page": norm_p,
                "ai_imp_7d": imp_7,
                "ai_lift_x": lift,
                "ai_daily_growth_%": round((lift - 1.0) * 100, 1),
                "ai_sparkline": [int(daily_b), int(daily_7)],
                "data_source_tag": "API AI-Signal Proxy (Prompts + Pos #1)",
            })

    if not page_records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ai_pages_df = pd.DataFrame(page_records).merge(web_7d, on="norm_page", how="left")
    ai_pages_df["display_url"] = ai_pages_df["display_url"].fillna(ai_pages_df["norm_page"])
    ai_pages_df["web_imp_7d"] = ai_pages_df["web_imp_7d"].fillna(0).astype(int)
    ai_pages_df["web_org_pos"] = ai_pages_df["web_org_pos"].fillna(0.0)

    ai_pages_df["ai_share_of_page_%"] = np.where(
        ai_pages_df["web_imp_7d"] > 0,
        np.minimum(100.0, (ai_pages_df["ai_imp_7d"] / ai_pages_df["web_imp_7d"]) * 100).round(1),
        0.0,
    )

    ai_pages_df["connected_clusters"] = ai_pages_df["norm_page"].map(clusters_map).fillna("Unclustered / Long-Tail")
    ai_pages_df["connected_trending_queries"] = ai_pages_df["norm_page"].map(
        lambda p: " | ".join(connected_qs_map.get(p, [])[:4]) or "No query match in current window"
    )
    ai_pages_df["ai_prompt_queries"] = ai_pages_df["norm_page"].map(
        lambda p: " | ".join(prompt_qs_map.get(p, [])[:3]) or "Short/Mid-tail queries"
    )
    ai_pages_df["sector"] = ai_pages_df.apply(
        lambda r: classify_sector(f"{r['display_url']} {r['connected_clusters']} {r['connected_trending_queries']}"),
        axis=1,
    )

    def ai_page_action(r) -> str:
        qs = " ".join(raw_qs_map.get(r["norm_page"], []))
        is_comm = bool(COMMERCIAL_MODIFIERS.search(qs))
        if is_comm and r["ai_lift_x"] >= 1.3:
            return "Ads + AEO: AI Overviews are absorbing commercial impressions—run Exact Search/AI Max ads on connected queries & add comparison tables."
        if len(prompt_qs_map.get(r["norm_page"], [])) > 0:
            return "AEO Prompt Upgrade: Add a 50-word direct answer block + FAQPage schema for the connected 7+ word AI Mode questions."
        return "Defend AI Citation: Keep statistics, structured lists, and entity schema fresh to retain AI Overview placement."

    ai_pages_df["recommended_action"] = ai_pages_df.apply(ai_page_action, axis=1)
    ai_pages_df = ai_pages_df.sort_values(["ai_lift_x", "ai_imp_7d"], ascending=False).reset_index(drop=True)

    bridge_df = page_queries_grp[page_queries_grp["norm_page"].isin(ai_pages_df["norm_page"])].copy()
    bridge_df = bridge_df.merge(
        ai_pages_df[["norm_page", "display_url", "ai_imp_7d", "ai_lift_x", "ai_share_of_page_%"]],
        on="norm_page",
        how="inner",
    )
    bridge_df = bridge_df.sort_values(["ai_lift_x", "imp_7d"], ascending=[False, False]).reset_index(drop=True)

    return ai_pages_df, ai_daily_series, bridge_df


def build_ads_recommendations(trends_df: pd.DataFrame, queries_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates trending topics for Paid Ads and outputs tiered keyword recommendations."""
    ads_topics = []
    rec_keywords = []

    for _, r in trends_df.iterrows():
        pos = r["connext_org_pos"]
        comm = r["commercial_intent"]
        pages = r["ranking_pages_count"]
        fit = r["ads_fit_score"]

        if r["is_today"] and comm and pos > 5.0:
            campaign_play = "P0 Immediate Search Launch (24h)"
            rationale = f"Demand surged {r['today_lift_x']}x in 48–72h with commercial intent while Connext ranks #{pos} organically."
        elif comm and pos > 5.0:
            campaign_play = "P1 High-Intent Search Ad Group"
            rationale = f"Sustained {r['daily_growth_%']}%/day impression growth on B2B buyer queries where organic rank is #{pos}."
        elif pages >= 2 and pos > 8.0:
            campaign_play = "P1 Dedicated Landing Page + Search"
            rationale = f"Organic impressions split across {pages} URLs (#{pos} avg rank). Consolidate paid traffic on a dedicated solution LP."
        elif comm and pos <= 5.0:
            campaign_play = "P2 SERP & AI Overview Defense"
            rationale = f"Connext ranks #{pos} organically on a surging commercial topic ({r['impressions_7d']:,} 7d impr.). Pair Paid Search with organic/AI citations."
        else:
            campaign_play = "P2 Low-CPC Long-Tail / Retargeting"
            rationale = f"Informational surge ({r['today_lift_x']}x lift). Bid on high-intent long-tail modifiers or retarget visitors from {r['primary_page']}."

        ads_topics.append({
            "Campaign Priority": campaign_play,
            "Ad Fit Score": fit,
            "Vertical": r["sector"],
            "Trending Topic": r["topic"],
            "Horizon": r["horizons"],
            "7d Impressions": r["impressions_7d"],
            "Surge Lift": f"{r['today_lift_x']}x",
            "Connext Org. Rank": pos,
            "Rank Trend": f"{r['rank_change']:+.1f}" if r["rank_change"] != 0 else "0.0",
            "Recommended LP": r["primary_page"],
            "Why Run Ads Now": rationale,
        })

    top_ad_topics = set(trends_df[trends_df["ads_fit_score"] >= 45]["topic"])
    target_q_df = queries_df[queries_df["topic"].isin(top_ad_topics)].copy()

    for _, q in target_q_df.iterrows():
        kw = q["query"]
        is_lt = q["is_long_tail"]
        is_comm = q["is_commercial"]
        q_pos = q["connext_org_pos"]

        if is_comm and not is_lt:
            kw_tier = "Core Commercial Head Term"
            match_type = '"Phrase Match" + [Exact Match]'
            bid_strategy = "Target Top of Page (High Intent)"
        elif is_comm and is_lt:
            kw_tier = "High-Intent Long-Tail (Low CPC)"
            match_type = "[Exact Match]"
            bid_strategy = "Aggressive Exact Bid (High Conversion Fit)"
        elif is_lt:
            kw_tier = "Problem-Aware Long-Tail / AI Prompt"
            match_type = "[Exact Match] (Test Cap)"
            bid_strategy = "Low-CPC Discovery / AI Max Signal"
        else:
            kw_tier = "Broad Category Modifier"
            match_type = '"Phrase Match" (Strict Negatives)'
            bid_strategy = "Moderate Bid + Job-Seeker Exclusions"

        rec_keywords.append({
            "Recommended Keyword": kw,
            "Keyword Tier": kw_tier,
            "Recommended Match Type": match_type,
            "Vertical": q["sector"],
            "Parent Topic Cluster": q["topic"],
            "7d Impressions": int(q["imp_7d"]),
            "Surge Lift": f"{q['query_lift_x']}x",
            "Connext Org. Rank": q_pos,
            "Recommended Landing Page": q["top_page"],
            "Bidding Guidance": bid_strategy,
        })

    ads_topics_df = pd.DataFrame(ads_topics).sort_values(["Ad Fit Score", "7d Impressions"], ascending=False)
    rec_kws_df = pd.DataFrame(rec_keywords).sort_values(["7d Impressions", "Surge Lift"], ascending=False) if rec_keywords else pd.DataFrame()
    return ads_topics_df, rec_kws_df


def build_content_queue(trends_df: pd.DataFrame) -> pd.DataFrame:
    """Builds Content Team sprint actions using Impressions, Organic Rank & Related Pages."""
    content_rows = []
    for _, r in trends_df.iterrows():
        pos = r["connext_org_pos"]
        pages = r["ranking_pages_count"]

        if (r["is_today"] or r["is_week"]) and 4.0 <= pos <= 15.0:
            sprint = "P1 - 48h Content Refresh (Push to Top 3)"
            action = f"Update {r['primary_page']} with H2s/FAQs for rising long-tail queries and add 3 internal links to break Top 3."
        elif pages >= 2:
            sprint = "P1 - URL Consolidation & Canonical Fix"
            action = f"Impressions split across {pages} pages ({r['all_related_pages']}). Consolidate into {r['primary_page']} or clarify intent."
        elif r["is_month"] and not r["is_week"]:
            sprint = "P1 - Monthly Pillar / Solution Page"
            action = "Sustained 30-day category growth; build comprehensive Webflow commercial solution page + FAQPage schema."
        elif pos > 15.0:
            sprint = "P1 - New Content Brief (Rank Gap)"
            action = f"Connext currently ranks #{pos}. Publish a dedicated commercial/educational asset targeting this surging cluster."
        else:
            sprint = "P2 - Defend Top-3 & AI Overview Citation"
            action = f"Connext ranks #{pos}. Add direct-answer FAQ blocks for emerging 7+ word prompts to hold Top 3 & AI citations."

        content_rows.append({
            "Sprint Priority": sprint,
            "Vertical": r["sector"],
            "Trending Topic": r["topic"],
            "Horizon": r["horizons"],
            "7d Impressions": r["impressions_7d"],
            "Surge Lift": f"{r['today_lift_x']}x",
            "Connext Org. Rank": pos,
            "Rank Trend": f"{r['rank_change']:+.1f}" if r["rank_change"] != 0 else "0.0",
            "Related Pages (#)": pages,
            "Primary Target Page": r["primary_page"],
            "All Ranking Pages": r["all_related_pages"],
            "Action Plan": action,
        })

    return pd.DataFrame(content_rows)


# --- SIDEBAR CONTROLS (MIDNIGHT NAVY WITH OFFICIAL WHITE CONNEXT LOGO) ---
st.sidebar.image(CNXT_LOGO_WHITE, width=185)
st.sidebar.markdown("**Search & AI Visibility Radar**")
st.sidebar.caption("Property: `connextglobal.com` • Impressions & Organic Rank")
if st.sidebar.button("Lock / Log Out"):
    st.session_state["cnxt_authenticated"] = False
    st.rerun()
st.sidebar.divider()

data_source = st.sidebar.radio(
    "1. Primary Web Search Data Source",
    ["Demo Dataset (connextglobal.com)", "Live Google Search Console API", "Upload GSC Web CSV"],
)

custom_negatives = st.sidebar.text_input(
    "Extra Negative Keywords (Pipe-separated)",
    placeholder="intern|internship|sample|pdf",
    help="Job-seeker & non-industry queries are blocked automatically.",
)

raw_df = pd.DataFrame()
ai_raw_df = pd.DataFrame()
ai_csv_mode = "timeseries"
incomplete_date_note = None
excluded_rows_count = 0

if data_source == "Demo Dataset (connextglobal.com)":
    raw_df, ai_raw_df = generate_demo_data()
    st.sidebar.success("Loaded 45-day Web + Generative AI sample data.")

elif data_source == "Live Google Search Console API":
    has_secrets = "gcp_service_account" in st.secrets and "gsc_config" in st.secrets
    if has_secrets:
        site_url = st.sidebar.text_input(
            "GSC Property URL",
            value=st.secrets["gsc_config"].get("site_url", DEFAULT_SITE_URL),
        )
        brand_regex = st.sidebar.text_input(
            "Brand Regex (for Brand Rank tracking)",
            value=st.secrets["gsc_config"].get("brand_regex", DEFAULT_BRAND_REGEX),
        )
        lookback = st.sidebar.slider("Lookback Days", 28, 90, 45)
        if st.sidebar.button("Fetch Latest GSC Data", type="primary") or "gsc_live_df" in st.session_state:
            with st.spinner("Querying GSC impressions & organic position..."):
                creds_dict = dict(st.secrets["gcp_service_account"])
                raw_df, incomplete_date_note, excluded_rows_count = fetch_gsc_api_data(
                    creds_dict, site_url, lookback, brand_regex, custom_negatives
                )
                st.session_state["gsc_live_df"] = raw_df
    else:
        st.sidebar.warning("`st.secrets` not configured. Upload Service Account JSON below:")
        uploaded_json = st.sidebar.file_uploader("Service Account JSON", type=["json"])
        site_url = st.sidebar.text_input("GSC Property URL", value=DEFAULT_SITE_URL)
        brand_regex = st.sidebar.text_input("Brand Regex", value=DEFAULT_BRAND_REGEX)
        lookback = st.sidebar.slider("Lookback Days", 28, 90, 45)
        if uploaded_json and site_url:
            creds_dict = json.load(uploaded_json)
            with st.spinner("Querying GSC impressions & organic position..."):
                raw_df, incomplete_date_note, excluded_rows_count = fetch_gsc_api_data(
                    creds_dict, site_url, lookback, brand_regex, custom_negatives
                )

elif data_source == "Upload GSC Web CSV":
    st.sidebar.markdown("Columns: `date, query, page, impressions, position`")
    uploaded_csv = st.sidebar.file_uploader("Upload Web Search Analytics CSV", type=["csv"])
    brand_regex = st.sidebar.text_input("Brand Regex", value=DEFAULT_BRAND_REGEX)
    if uploaded_csv:
        raw_df = pd.read_csv(uploaded_csv)
        raw_df.columns = [c.lower().strip() for c in raw_df.columns]
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df["query"] = raw_df["query"].astype(str).str.lower().str.strip()
        before_len = len(raw_df)
        raw_df = raw_df[~raw_df["query"].str.contains(STRICT_NON_INDUSTRY_REGEX, regex=True, na=False)]
        if custom_negatives.strip():
            raw_df = raw_df[~raw_df["query"].str.contains(custom_negatives, regex=True, na=False)]
        excluded_rows_count = before_len - len(raw_df)
        raw_df["is_branded"] = raw_df["query"].str.contains(brand_regex, regex=True, na=False) if brand_regex else False

if data_source != "Demo Dataset (connextglobal.com)":
    st.sidebar.divider()
    st.sidebar.subheader("2. GSC Generative AI Report (Optional)")
    st.sidebar.caption(
        "Export **Pages** from GSC → *Performance → Generative AI* and upload here to connect AI Overviews & AI Mode pages to your trending queries."
    )
    uploaded_ai_csv = st.sidebar.file_uploader("Upload Generative AI Report CSV", type=["csv"], key="ai_csv_uploader")
    if uploaded_ai_csv:
        try:
            ai_raw_df, ai_csv_mode = parse_gsc_ai_report_csv(uploaded_ai_csv)
            st.sidebar.success(f"Loaded {len(ai_raw_df):,} Generative AI rows ({ai_csv_mode}).")
        except Exception as exc:
            st.sidebar.error(f"AI CSV Error: {exc}")

st.sidebar.divider()
st.sidebar.subheader("3. Trend Velocity Sensitivity")
min_daily_imp = st.sidebar.number_input("Min Avg Daily Impressions", min_value=10, max_value=10000, value=60, step=10)
min_up_days = st.sidebar.slider("Weekly Min Upward Days (of 6)", min_value=3, max_value=6, value=4)
min_lift = st.sidebar.slider("Min Baseline Surge Multiplier", min_value=1.1, max_value=3.0, value=1.35, step=0.05)
min_slope_pct = st.sidebar.slider("Min Daily Growth Slope (%)", min_value=5, max_value=40, value=10, step=1)
cluster_dist = st.sidebar.slider("Topic Clustering Breadth", min_value=0.30, max_value=0.75, value=0.55, step=0.05)


# --- CONNEXT BRANDED HEADER & HERO BANNER ---
hdr_logo, hdr_meta = st.columns([1.2, 4])
with hdr_logo:
    st.image(CNXT_LOGO_COLOR, width=195)
with hdr_meta:
    meta_notes = ["Impressions & Organic Position Only", "Non-Branded B2B Demand", "Job-Seeker Noise Filtered"]
    if incomplete_date_note and incomplete_date_note != "None":
        meta_notes.append(f"Incomplete GSC Data Trimmed ≥ {incomplete_date_note}")
    if excluded_rows_count > 0:
        meta_notes.append(f"{excluded_rows_count:,} Non-Industry Rows Excluded")
    st.caption(" • ".join(meta_notes))

st.markdown(
    """
    <div class="cnxt-hero-banner">
        <div class="cnxt-eyebrow">CONNEXT GLOBAL GROWTH &amp; SEARCH INTELLIGENCE</div>
        <h1 class="cnxt-hero-title">Search &amp; AI Visibility <span>Trend Radar</span></h1>
        <p class="cnxt-hero-sub">
            Real-time detection of surging B2B outsourcing topics across Google Search, AI Overviews, and AI Mode—mapped directly to Connext’s organic rankings, landing pages, and Paid Search keyword opportunities.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if raw_df.empty:
    st.info("Select a data source in the sidebar or click **Fetch Latest GSC Data** to analyze `connextglobal.com` trends.")
    st.stop()

with st.spinner("Clustering B2B queries, tracking Connext organic position & linking AI Search pages..."):
    clustered_df = cluster_queries(raw_df, cluster_dist)
    trends_df, daily_topic_df, all_queries_df, pages_df = analyze_multi_horizon_trends(
        clustered_df,
        raw_df,
        min_daily_imp=min_daily_imp,
        min_up_days=min_up_days,
        min_norm_slope=min_slope_pct / 100.0,
        min_lift=min_lift,
    )
    ai_pages_df, ai_daily_series, ai_bridge_df = analyze_ai_pages_and_queries(
        ai_raw_df, ai_csv_mode, clustered_df, all_queries_df
    )

if trends_df.empty:
    st.warning("No B2B topics matched your current impression velocity thresholds. Lower **Min Baseline Surge Multiplier** or **Min Avg Daily Impressions** in the sidebar.")
    st.stop()

# --- GLOBAL VERTICAL & ORGANIC RANK TIER FILTERS ---
f1, f2, f3 = st.columns([2, 1.1, 1.1])
with f1:
    available_sectors = sorted(trends_df["sector"].unique().tolist())
    selected_sectors = st.multiselect("Filter by Business Vertical:", available_sectors, default=available_sectors)
with f2:
    rank_tiers = sorted(trends_df["rank_tier"].unique().tolist())
    selected_tiers = st.multiselect("Connext Organic Rank Tier:", rank_tiers, default=rank_tiers)
with f3:
    intent_filter = st.selectbox("Search Intent:", ["All B2B Intents", "Commercial / B2B Only", "Informational / Research Only"])

filtered_trends = trends_df[
    (trends_df["sector"].isin(selected_sectors)) & (trends_df["rank_tier"].isin(selected_tiers))
].copy()

if intent_filter == "Commercial / B2B Only":
    filtered_trends = filtered_trends[filtered_trends["commercial_intent"]]
elif intent_filter == "Informational / Research Only":
    filtered_trends = filtered_trends[~filtered_trends["commercial_intent"]]

filtered_trends = filtered_trends.reset_index(drop=True)
if filtered_trends.empty:
    st.info("No trending topics match the selected filters.")
    st.stop()

filtered_queries = all_queries_df[all_queries_df["topic"].isin(filtered_trends["topic"])].copy()
filtered_pages = pages_df[pages_df["topic"].isin(filtered_trends["topic"])].copy()
filtered_ai_pages = ai_pages_df[ai_pages_df["sector"].isin(selected_sectors)].copy() if not ai_pages_df.empty else pd.DataFrame()
filtered_ai_bridge = ai_bridge_df[ai_bridge_df["norm_page"].isin(filtered_ai_pages["norm_page"])].copy() if not ai_bridge_df.empty else pd.DataFrame()

ads_topics_df, rec_kws_df = build_ads_recommendations(filtered_trends, filtered_queries)
content_df = build_content_queue(filtered_trends)

# --- TOP KPI METRICS ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trending Today (48–72h)", int(filtered_trends["is_today"].sum()))
k2.metric("Trending This Week (5–7d)", int(filtered_trends["is_week"].sum()))
k3.metric("Trending This Month (14–30d)", int(filtered_trends["is_month"].sum()))
k4.metric("Surging AI Pages (AIO/AI Mode)", len(filtered_ai_pages))
k5.metric("Avg Connext Organic Rank", f"#{filtered_trends['connext_org_pos'].mean():.1f}")

# --- MAIN NAVIGATION TABS ---
tab_horizons, tab_ai_search, tab_pages_clusters, tab_ads_strategy, tab_content = st.tabs([
    "Today / Week / Month Radar",
    f"AI Search (AIO & AI Mode Pages: {len(filtered_ai_pages)})",
    f"Related Content & Long-Tail ({len(filtered_pages)} URLs)",
    f"Paid Ads & Recommended Keywords ({len(rec_kws_df)} KWs)",
    f"Content Action Queue ({len(content_df)})",
])

# ==============================================================================
# TAB 1: MULTI-HORIZON RADAR (TODAY, THIS WEEK, THIS MONTH)
# ==============================================================================
with tab_horizons:
    horizon_choice = st.radio(
        "Select Time Horizon:",
        [
            "All Surging Topics",
            "Trending Today (48–72h Breakout)",
            "Trending This Week (5–7d Sustained)",
            "Trending This Month (14–30d Macro)",
        ],
        horizontal=True,
    )

    view_df = filtered_trends.copy()
    if "Today" in horizon_choice:
        view_df = view_df[view_df["is_today"]]
    elif "This Week" in horizon_choice:
        view_df = view_df[view_df["is_week"]]
    elif "This Month" in horizon_choice:
        view_df = view_df[view_df["is_month"]]

    if view_df.empty:
        st.info(f"No topics currently flagged under **{horizon_choice}**.")
    else:
        g1, g2 = st.columns([1, 1.15])

        with g1:
            st.subheader("Surging Impressions by Business Vertical")
            sector_agg = (
                view_df.groupby("sector", as_index=False)
                .agg(
                    topics=("topic", "count"),
                    impressions_7d=("impressions_7d", "sum"),
                    avg_rank=("connext_org_pos", "mean"),
                )
                .sort_values("impressions_7d", ascending=True)
            )
            fig_sector = px.bar(
                sector_agg,
                x="impressions_7d",
                y="sector",
                orientation="h",
                color="avg_rank",
                text="topics",
                color_continuous_scale=[[0.0, CNXT_COLORS["twilight"]], [0.5, CNXT_COLORS["blue"]], [1.0, CNXT_COLORS["teal_blue"]]],
                labels={
                    "impressions_7d": "7-Day Impressions",
                    "sector": "Vertical",
                    "avg_rank": "Avg Org. Rank",
                },
            )
            fig_sector.update_traces(texttemplate="%{text} topics", textposition="outside")
            fig_sector.update_layout(
                height=400,
                margin=dict(l=10, r=35, t=20, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
            )
            st.plotly_chart(fig_sector, use_container_width=True)

        with g2:
            st.subheader("Connext Organic Position vs. Impression Surge")
            fig_scatter = px.scatter(
                view_df,
                x="connext_org_pos",
                y="today_lift_x",
                size="impressions_7d",
                color="sector",
                color_discrete_sequence=CNXT_CHART_PALETTE,
                hover_name="topic",
                hover_data=["horizons", "primary_page", "rank_change", "brand_org_pos"],
                labels={
                    "connext_org_pos": "Connext Organic Position (Right = #1)",
                    "today_lift_x": "Short-Term Impression Lift (x)",
                    "sector": "Vertical",
                },
                size_max=45,
            )
            fig_scatter.update_xaxes(autorange="reversed", gridcolor=CNXT_COLORS["line"])
            fig_scatter.update_yaxes(gridcolor=CNXT_COLORS["line"])
            fig_scatter.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Daily Impressions & Connext Organic Position Trajectory")
        selected_topic = st.selectbox("Inspect Topic Curve:", view_df["topic"].tolist(), key="horizon_topic_select")
        topic_series = daily_topic_df[daily_topic_df["topic"] == selected_topic].sort_values("date").copy()
        topic_series["rolling_7d_imp"] = topic_series["impressions"].rolling(7, min_periods=1).mean()

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=topic_series["date"],
            y=topic_series["impressions"],
            mode="lines+markers",
            name="Daily Impressions",
            line=dict(color=CNXT_COLORS["blue"], width=3),
            marker=dict(color=CNXT_COLORS["twilight"], size=6),
        ))
        fig_line.add_trace(go.Scatter(
            x=topic_series["date"],
            y=topic_series["rolling_7d_imp"],
            mode="lines",
            name="7-Day Rolling Avg Impr.",
            line=dict(color=CNXT_COLORS["teal"], width=2.2, dash="dash"),
        ))
        fig_line.add_trace(go.Scatter(
            x=topic_series["date"],
            y=topic_series["organic_position"],
            mode="lines+markers",
            name="Connext Organic Position",
            line=dict(color=CNXT_COLORS["sun"], width=2.8),
            marker=dict(color=CNXT_COLORS["red"], size=6),
            yaxis="y2",
        ))
        fig_line.update_layout(
            height=370,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
            yaxis=dict(title="Impressions", gridcolor=CNXT_COLORS["line"]),
            yaxis2=dict(title="Organic Position (#1 = Top)", overlaying="y", side="right", autorange="reversed"),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader(f"Trending Topics Overview ({len(view_df)})")
        overview_cols = [
            "topic", "sector", "horizons", "sparkline_7d", "impressions_7d",
            "impressions_month", "today_lift_x", "month_lift_x", "daily_growth_%",
            "connext_org_pos", "rank_change", "brand_org_pos",
            "ranking_pages_count", "primary_page", "intent_label",
        ]
        st.dataframe(
            view_df[overview_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "sparkline_7d": st.column_config.LineChartColumn("7d Impr. Trend", width="small"),
                "connext_org_pos": st.column_config.NumberColumn("Connext Org. Rank", format="#%.1f"),
                "rank_change": st.column_config.NumberColumn("Rank Δ (vs 14d)", format="%+.1f"),
                "today_lift_x": st.column_config.NumberColumn("3d Surge", format="%.2fx"),
                "month_lift_x": st.column_config.NumberColumn("Monthly Surge", format="%.2fx"),
                "primary_page": st.column_config.LinkColumn("Primary Ranking Page"),
            },
        )

# ==============================================================================
# TAB 2: AI SEARCH (GSC GENERATIVE AI REPORT + PAGE-TO-QUERY CONNECTOR)
# ==============================================================================
with tab_ai_search:
    st.subheader("Generative AI Search Radar (AI Overviews & AI Mode)")
    st.markdown(
        """
        <div class="cnxt-cream-panel">
            <strong>How GSC AI Attribution Works:</strong> Google Search Console’s <em>Generative AI</em> performance report tracks <strong>AI Impressions by Page</strong> without exposing underlying queries. Below, every trending AI page on <code>connextglobal.com</code> is automatically joined by URL to the <strong>trending keywords, 7+ word AI Mode prompts, and topic clusters</strong> ranking on that exact page.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_ai_pages.empty:
        st.info("No Generative AI page trends matched your current filters. Upload your GSC *Generative AI -> Pages* CSV in the sidebar or lower your sensitivity thresholds.")
    else:
        source_used = filtered_ai_pages["data_source_tag"].iloc[0]
        st.caption(f"Active AI Attribution Source: **{source_used}**")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Trending AI Pages", len(filtered_ai_pages))
        a2.metric("7d AI Search Impressions", f"{int(filtered_ai_pages['ai_imp_7d'].sum()):,}")
        a3.metric("Avg AI Surge Lift", f"{filtered_ai_pages['ai_lift_x'].mean():.2f}x")
        a4.metric("Connected Rising Queries", len(filtered_ai_bridge))

        ai_col1, ai_col2 = st.columns([1.15, 1])

        with ai_col1:
            st.markdown("**1. AI Search Impressions vs. Standard Web Impressions by Page**")
            top_ai_bar = filtered_ai_pages.head(10).copy()
            top_ai_bar["short_path"] = top_ai_bar["display_url"].str.replace("https://connextglobal.com", "", regex=False)
            top_ai_bar["non_ai_est"] = np.maximum(0, top_ai_bar["web_imp_7d"] - top_ai_bar["ai_imp_7d"])

            fig_ai_stack = go.Figure()
            fig_ai_stack.add_trace(go.Bar(
                y=top_ai_bar["short_path"],
                x=top_ai_bar["ai_imp_7d"],
                name="AI Overviews / AI Mode Impr.",
                orientation="h",
                marker_color=CNXT_COLORS["twilight"],
            ))
            fig_ai_stack.add_trace(go.Bar(
                y=top_ai_bar["short_path"],
                x=top_ai_bar["non_ai_est"],
                name="Standard Organic Impr.",
                orientation="h",
                marker_color=CNXT_COLORS["teal_blue"],
            ))
            fig_ai_stack.update_layout(
                barmode="stack",
                height=390,
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(title="7-Day Impressions", gridcolor=CNXT_COLORS["line"]),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_ai_stack, use_container_width=True)

        with ai_col2:
            st.markdown("**2. Daily AI Search Impression Curve**")
            selected_ai_url = st.selectbox(
                "Select AI-Surging Page:",
                filtered_ai_pages["display_url"].tolist(),
                key="ai_page_curve_select",
            )
            sel_norm = normalize_url(selected_ai_url)

            if not ai_daily_series.empty and sel_norm in set(ai_daily_series["norm_page"]):
                p_ts = ai_daily_series[ai_daily_series["norm_page"] == sel_norm].sort_values("date")
                fig_ai_line = px.area(
                    p_ts,
                    x="date",
                    y="ai_impressions",
                    labels={"date": "Date", "ai_impressions": "AI Search Impressions"},
                    color_discrete_sequence=[CNXT_COLORS["twilight"]],
                )
            else:
                p_web_ts = (
                    clustered_df[clustered_df["norm_page"] == sel_norm]
                    .groupby("date", as_index=False)["impressions"]
                    .sum()
                    .sort_values("date")
                )
                fig_ai_line = px.area(
                    p_web_ts,
                    x="date",
                    y="impressions",
                    labels={"date": "Date", "impressions": "Page Search Impressions"},
                    color_discrete_sequence=[CNXT_COLORS["twilight"]],
                )
            fig_ai_line.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
            )
            st.plotly_chart(fig_ai_line, use_container_width=True)

        st.markdown("### 3. Trending AI Pages Connected to Topic Clusters & Rising Queries")
        ai_display_cols = [
            "display_url", "sector", "ai_sparkline", "ai_imp_7d", "ai_lift_x",
            "ai_share_of_page_%", "web_org_pos", "connected_clusters",
            "connected_trending_queries", "ai_prompt_queries", "recommended_action",
        ]
        st.dataframe(
            filtered_ai_pages[ai_display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "display_url": st.column_config.LinkColumn("Trending AI Page"),
                "ai_sparkline": st.column_config.LineChartColumn("AI Impr. Trend", width="small"),
                "ai_imp_7d": st.column_config.NumberColumn("AI Impr. (7d)"),
                "ai_lift_x": st.column_config.NumberColumn("AI Surge", format="%.2fx"),
                "ai_share_of_page_%": st.column_config.NumberColumn("AI Share %", format="%.1f%%"),
                "web_org_pos": st.column_config.NumberColumn("Connext Org. Rank", format="#%.1f"),
            },
        )

        st.divider()
        st.markdown("### 4. Granular AI Page ↔ Trending Keyword Bridge")
        st.caption("Every rising query and conversational prompt mapped to the AI-surging page it ranks on.")
        if filtered_ai_bridge.empty:
            st.info("No query-level matches found for the selected AI pages.")
        else:
            bridge_cols = [
                "display_url", "ai_imp_7d", "ai_lift_x", "topic",
                "query", "is_ai_prompt", "is_long_tail", "imp_7d", "query_lift_x", "connext_org_pos",
            ]
            st.dataframe(
                filtered_ai_bridge[bridge_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "display_url": st.column_config.LinkColumn("AI Surging Page"),
                    "ai_imp_7d": st.column_config.NumberColumn("Page AI Impr."),
                    "ai_lift_x": st.column_config.NumberColumn("Page AI Lift", format="%.2fx"),
                    "is_ai_prompt": st.column_config.CheckboxColumn("AI Prompt (7+w)"),
                    "is_long_tail": st.column_config.CheckboxColumn("Long-Tail"),
                    "imp_7d": st.column_config.NumberColumn("Query 7d Impr."),
                    "query_lift_x": st.column_config.NumberColumn("Query Lift", format="%.2fx"),
                    "connext_org_pos": st.column_config.NumberColumn("Org. Rank", format="#%.1f"),
                },
            )
            st.download_button(
                "Download AI Pages & Connected Keywords CSV",
                filtered_ai_bridge[bridge_cols].to_csv(index=False).encode("utf-8"),
                file_name="connext_ai_pages_connected_keywords.csv",
                mime="text/csv",
            )

# ==============================================================================
# TAB 3: RELATED CONTENT (PAGES), CLUSTERS & LONG-TAIL QUERIES
# ==============================================================================
with tab_pages_clusters:
    st.subheader("Related Content on `connextglobal.com` & Cluster Query Breakdown")

    inspect_cluster = st.selectbox(
        "Select a Trending Topic Cluster to Inspect All Ranking Pages & Long-Tail Queries:",
        filtered_trends["topic"].tolist(),
        key="cluster_detail_select",
    )
    topic_meta = filtered_trends[filtered_trends["topic"] == inspect_cluster].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Connext Organic Position", f"#{topic_meta['connext_org_pos']}", delta=f"{topic_meta['rank_change']:+.1f} vs prior 14d")
    m2.metric("Brand Search Position", topic_meta["brand_org_pos"])
    m3.metric("Ranking Pages Count", topic_meta["ranking_pages_count"])
    m4.metric("Long-Tail Queries (4+ words)", f"{topic_meta['long_tail_count']} / {topic_meta['total_queries']}")

    col_p, col_q = st.columns([1, 1.15])

    with col_p:
        st.markdown("**1. Related Content (`connextglobal.com` Pages Ranking for This Topic)**")
        topic_pages = filtered_pages[filtered_pages["topic"] == inspect_cluster].sort_values("page_imp_7d", ascending=False)
        if len(topic_pages) > 1:
            st.warning(f"Multiple URLs ({len(topic_pages)}) are splitting impressions for this cluster. Check for content overlap.")
        st.dataframe(
            topic_pages[["page", "page_imp_7d", "topic_imp_share_%", "page_org_pos", "queries_on_page"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "page": st.column_config.LinkColumn("Connext URL"),
                "page_imp_7d": st.column_config.NumberColumn("7d Impr."),
                "topic_imp_share_%": st.column_config.NumberColumn("Share %", format="%.1f%%"),
                "page_org_pos": st.column_config.NumberColumn("Page Org. Rank", format="#%.1f"),
            },
        )

    with col_q:
        st.markdown("**2. Member Queries & Long-Tail Terms in This Cluster**")
        topic_qs = filtered_queries[filtered_queries["topic"] == inspect_cluster].sort_values("imp_7d", ascending=False)
        st.dataframe(
            topic_qs[["query", "is_long_tail", "is_ai_prompt", "imp_7d", "query_lift_x", "connext_org_pos", "top_page"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "is_long_tail": st.column_config.CheckboxColumn("Long-Tail"),
                "is_ai_prompt": st.column_config.CheckboxColumn("AI Prompt"),
                "imp_7d": st.column_config.NumberColumn("7d Impr."),
                "query_lift_x": st.column_config.NumberColumn("Lift", format="%.2fx"),
                "connext_org_pos": st.column_config.NumberColumn("Org. Rank", format="#%.1f"),
            },
        )

    st.divider()
    st.subheader("Head Terms vs. Long-Tail Impression Volume Across Verticals")
    lt_split = filtered_queries.copy()
    lt_split["Query Structure"] = np.where(lt_split["is_long_tail"], "Long-Tail (4+ Words)", "Head / Mid-Tail (1–3 Words)")
    lt_agg = lt_split.groupby(["sector", "Query Structure"], as_index=False)["imp_7d"].sum()

    fig_lt_bar = px.bar(
        lt_agg,
        x="sector",
        y="imp_7d",
        color="Query Structure",
        barmode="group",
        color_discrete_map={"Long-Tail (4+ Words)": CNXT_COLORS["teal"], "Head / Mid-Tail (1–3 Words)": CNXT_COLORS["blue"]},
        labels={"sector": "Business Vertical", "imp_7d": "7-Day Impressions"},
    )
    fig_lt_bar.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
    )
    st.plotly_chart(fig_lt_bar, use_container_width=True)

# ==============================================================================
# TAB 4: PAID ADS OPPORTUNITIES & RECOMMENDED KEYWORDS SECTION
# ==============================================================================
with tab_ads_strategy:
    st.subheader("Trending Topics Recommended for Paid Ads & Keyword Strategy")
    st.markdown(
        "Topics below are scored by **Ad Fit Score (0–100)**—combining **impression surge velocity**, "
        "**B2B commercial intent**, and **Connext’s organic rank gap**."
    )

    p0_p1_ads = ads_topics_df[ads_topics_df["Ad Fit Score"] >= 60]
    high_intent_lt = rec_kws_df[rec_kws_df["Keyword Tier"] == "High-Intent Long-Tail (Low CPC)"] if not rec_kws_df.empty else pd.DataFrame()

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"""
            <div class="cnxt-cream-panel">
                <strong>Immediate Search Launch ({len(p0_p1_ads)} topics)</strong><br/>
                {", ".join(f"<code>{t}</code>" for t in p0_p1_ads["Trending Topic"].head(3)) if not p0_p1_ads.empty else "No high-gap commercial surges currently."}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
            <div class="cnxt-cream-panel">
                <strong>High-Intent Long-Tail Keywords ({len(high_intent_lt)})</strong><br/>
                {", ".join(f"<code>[{k}]</code>" for k in high_intent_lt["Recommended Keyword"].head(3)) if not high_intent_lt.empty else "Lower sensitivity thresholds to surface long-tail terms."}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
            <div class="cnxt-cream-panel">
                <strong>Connext B2B Bidding Guardrail</strong><br/>
                Apply <code>-jobs</code>, <code>-careers</code>, <code>-hiring</code>, <code>-salary</code>, <code>-glassdoor</code>, and <code>-template</code> negatives before launch.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 1. Trending Topics Evaluated for Paid Campaigns")
    st.dataframe(
        ads_topics_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ad Fit Score": st.column_config.ProgressColumn("Ad Fit Score", min_value=0, max_value=100, format="%d/100"),
            "Connext Org. Rank": st.column_config.NumberColumn("Connext Org. Rank", format="#%.1f"),
            "Recommended LP": st.column_config.LinkColumn("Recommended LP"),
        },
    )

    st.divider()
    st.markdown("### 2. Recommended Keywords to Bid On (Exact, Phrase & Long-Tail)")
    if rec_kws_df.empty:
        st.info("No keywords met the Ad Fit threshold under current filters.")
    else:
        kw_topic_filter = st.multiselect(
            "Filter Recommended Keywords by Topic Cluster:",
            options=sorted(rec_kws_df["Parent Topic Cluster"].unique().tolist()),
            default=sorted(rec_kws_df["Parent Topic Cluster"].unique().tolist()),
        )
        scoped_kws = rec_kws_df[rec_kws_df["Parent Topic Cluster"].isin(kw_topic_filter)]

        st.dataframe(
            scoped_kws,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Connext Org. Rank": st.column_config.NumberColumn("Connext Org. Rank", format="#%.1f"),
                "Recommended Landing Page": st.column_config.LinkColumn("Landing Page"),
            },
        )
        st.download_button(
            "Download Recommended Paid Keywords CSV",
            scoped_kws.to_csv(index=False).encode("utf-8"),
            file_name="connext_recommended_paid_keywords.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("### 3. Ad Group Builder: Copy-Paste Keywords & Brand-Aligned RSA Copy")
        selected_ad_topic = st.selectbox("Select Trending Topic to Build Ad Group:", ads_topics_df["Trending Topic"].tolist())
        ad_row = filtered_trends[filtered_trends["topic"] == selected_ad_topic].iloc[0]
        kw_list = ad_row["queries_list"]

        c_ex, c_ph, c_ng = st.columns(3)
        with c_ex:
            st.markdown("**[Exact Match] Keywords**")
            st.code("\n".join(f"[{q}]" for q in kw_list), language="text")
        with c_ph:
            st.markdown('**"Phrase Match" Keywords**')
            st.code("\n".join(f'"{q}"' for q in kw_list), language="text")
        with c_ng:
            st.markdown("**B2B Negative Keywords**")
            st.code(
                "-jobs\n-careers\n-hiring\n-salary\n-internship\n-glassdoor\n-indeed\n-jobstreet\n-free\n-template",
                language="text",
            )

        headlines, descriptions = build_rsa_assets(selected_ad_topic)
        ch, cd = st.columns(2)
        with ch:
            st.markdown("**RSA Headlines (≤ 30 chars)**")
            for cat, h in headlines:
                st.text(f"[{cat}] {h} ({len(h)}/30)")
        with cd:
            st.markdown("**RSA Descriptions (≤ 90 chars)**")
            for d in descriptions:
                st.text(f"{d} ({len(d)}/90)")

# ==============================================================================
# TAB 5: CONTENT ACTION QUEUE & BRIEF GENERATOR
# ==============================================================================
with tab_content:
    st.subheader("Content Team Action Queue (Based on Impressions & Organic Rank)")
    st.dataframe(
        content_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Connext Org. Rank": st.column_config.NumberColumn("Connext Org. Rank", format="#%.1f"),
            "Primary Target Page": st.column_config.LinkColumn("Primary Target Page"),
        },
    )
    st.download_button(
        "Download Content Team Sprint CSV",
        content_df.to_csv(index=False).encode("utf-8"),
        file_name="connext_content_action_queue.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Rapid Content Refresh / New Page Brief")
    brief_topic = st.selectbox("Select Topic for Content Brief:", content_df["Trending Topic"].tolist())
    b_row = filtered_trends[filtered_trends["topic"] == brief_topic].iloc[0]

    lt_bullets = (
        "\n".join(f"  - `{q}`" for q in b_row["long_tail_list"])
        if b_row["long_tail_list"]
        else "\n".join(f"  - `{q}`" for q in b_row["queries_list"])
    )

    brief_md = f"""### Content Execution Brief: {b_row['topic'].title()} ({b_row['sector']})
- **Trend Horizon:** `{b_row['horizons']}`
- **Connext Organic Position:** **#{b_row['connext_org_pos']}** (`{b_row['rank_change']:+.1f}` vs prior 14d) | **Brand Search Position:** `{b_row['brand_org_pos']}`
- **Primary Ranking Page:** `{b_row['primary_page']}`
- **All Related Pages ({b_row['ranking_pages_count']}):** `{b_row['all_related_pages']}`
- **Impression Momentum:** **{b_row['impressions_7d']:,}** 7d impressions (**{b_row['today_lift_x']}x** short-term lift | **{b_row['month_lift_x']}x** monthly lift)
- **Rising Long-Tail Queries to Cover (H2s / FAQs):**
{lt_bullets}
"""
    st.markdown(brief_md)

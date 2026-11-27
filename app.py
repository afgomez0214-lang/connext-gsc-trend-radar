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

CNXT_LOGO_COLOR = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6a72a1e48c2cd52f0e5f2578_CONNEXT%20MAIN%20LOGO%20COLOR.png"
CNXT_LOGO_WHITE = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6ad6f9596637ac5988a5bf28_CONNEXT%20MAIN%20LOGO%20WHITE%20V.png"
CNXT_FAVICON = "https://cdn.prod.website-files.com/6a5abaee04ee2fd8ff22a3ec/6a965e51559ff3d4d6cad70f_Connext_Icons_Connext%20Logo-lg.png"

SITEMAP_XML_URL = "https://connextglobal.com/sitemap.xml"
SITEMAP_HTML_URL = "https://connextglobal.com/sitemap"

# --- CONNEXT 12 TRACKED COMPETITORS ---
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

COMPETITOR_BRAND_REGEX = re.compile(
    "|".join(pat for _, _, pat in COMPETITORS_REGISTRY),
    re.I,
)

TRENDS_SEED_TOPICS = [
    ("Healthcare & RCM", "medical billing outsourcing"),
    ("Finance, Accounting & Tax", "offshore accounting"),
    ("Customer Experience (CX) & VAs", "virtual assistant outsourcing"),
    ("Tech, IT & AI Staffing", "nearshore software developers"),
    ("General B2B Outsourcing & Staffing", "bpo outsourcing"),
]

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Connext Global | Search, Trends, Bing & Competitor Radar",
    page_icon=CNXT_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_connext_brand_css():
    css = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Arimo:wght@400;500;600;700&family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root { --cnxt-blue: #005AAC; --cnxt-sun: #E58B25; --cnxt-yellow: #F1C916; --cnxt-red: #FC4301; --cnxt-twilight: #071776; --cnxt-midnight: #071739; --cnxt-teal: #0F818A; --cnxt-teal-blue: #90CBDB; --cnxt-sage: #7E9E7F; --cnxt-cream: #F4F1E2; --cnxt-paper: #FFFFFF; --cnxt-ash: #6B7280; --cnxt-line: #E5E2D2; --cnxt-line-strong: #C7C2A8; --font-display: 'Arimo', Arial, sans-serif; --font-body: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif; --font-mono: 'JetBrains Mono', ui-monospace, monospace; }
html, body, [class*="css"] { font-family: var(--font-body); color: var(--cnxt-midnight); }
h1, h2, h3, h4 { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; letter-spacing: -0.015em; }
.stApp { background-color: var(--cnxt-paper); }
section[data-testid="stSidebar"] { background-color: var(--cnxt-midnight) !important; border-right: 1px solid rgba(144, 203, 219, 0.18); }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: var(--cnxt-paper) !important; }
section[data-testid="stSidebar"] code { background-color: rgba(144, 203, 219, 0.18) !important; color: #F1C916 !important; padding: 2px 6px; border-radius: 4px; }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small { color: var(--cnxt-teal-blue) !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--cnxt-paper) !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea { background-color: #FFFFFF !important; border: 1px solid #90CBDB !important; color: #071739 !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #071739 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #071739 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background-color: rgba(255, 255, 255, 0.06) !important; border: 1px dashed var(--cnxt-teal-blue) !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background-color: var(--cnxt-blue) !important; color: var(--cnxt-paper) !important; border: none !important; }
section[data-testid="stSidebar"] .stButton > button { background-color: rgba(255, 255, 255, 0.10) !important; color: var(--cnxt-teal-blue) !important; border: 1px solid rgba(144, 203, 219, 0.35) !important; border-radius: 999px !important; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background-color: var(--cnxt-sun) !important; color: var(--cnxt-midnight) !important; border: none !important; font-weight: 600 !important; }
.stButton > button[kind="primary"], .stDownloadButton > button { background-color: var(--cnxt-sun) !important; color: var(--cnxt-midnight) !important; font-family: var(--font-body) !important; font-weight: 600 !important; border: none !important; border-radius: 999px !important; padding: 0.55rem 1.4rem !important; box-shadow: 0 6px 18px -8px rgba(229, 139, 37, 0.65) !important; transition: all 0.2s ease !important; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { background-color: var(--cnxt-twilight) !important; color: var(--cnxt-cream) !important; transform: translateY(-1px); }
.cnxt-hero-banner { background: linear-gradient(135deg, #071776 0%, #071739 100%); color: #FFFFFF; padding: 24px 30px; border-radius: 10px; margin-bottom: 16px; position: relative; overflow: hidden; border: 1px solid rgba(144, 203, 219, 0.22); }
.cnxt-hero-banner::after { content: ""; position: absolute; top: -90px; right: -90px; width: 280px; height: 280px; background: radial-gradient(circle, rgba(241, 201, 22, 0.22), transparent 65%); pointer-events: none; }
.cnxt-eyebrow { font-family: var(--font-mono); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.18em; color: #F1C916; margin-bottom: 6px; font-weight: 600; }
.cnxt-hero-title { font-family: var(--font-display); font-size: clamp(1.45rem, 2.4vw, 2.05rem); font-weight: 700; color: #FFFFFF; line-height: 1.18; margin: 0 0 8px 0; }
.cnxt-hero-title span { color: #E58B25; }
.cnxt-hero-sub { color: #90CBDB; font-size: 0.94rem; margin: 0; max-width: 920px; }
div[data-testid="stMetric"] { background-color: var(--cnxt-paper); border: 1px solid var(--cnxt-line); border-top: 3px solid var(--cnxt-blue); border-radius: 8px; padding: 14px 16px; box-shadow: 0 2px 10px rgba(7, 23, 57, 0.04); }
div[data-testid="stMetric"] label { font-family: var(--font-mono) !important; font-size: 0.71rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; color: var(--cnxt-ash) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--cnxt-twilight) !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid var(--cnxt-line); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { font-family: var(--font-display); font-weight: 600; color: var(--cnxt-midnight); background-color: var(--cnxt-cream); border-radius: 6px 6px 0 0; padding: 10px 14px; border: 1px solid var(--cnxt-line); border-bottom: none; }
.stTabs [aria-selected="true"] { background-color: var(--cnxt-twilight) !important; color: var(--cnxt-paper) !important; border-color: var(--cnxt-twilight) !important; }
.cnxt-cream-panel { background-color: var(--cnxt-cream); border: 1px solid var(--cnxt-line-strong); border-left: 4px solid var(--cnxt-sun); border-radius: 6px; padding: 14px 18px; margin: 10px 0; color: var(--cnxt-midnight); font-size: 0.93rem; line-height: 1.55; }
.cnxt-cream-panel strong { color: var(--cnxt-twilight); }
span[data-baseweb="tag"] { background-color: var(--cnxt-twilight) !important; color: var(--cnxt-paper) !important; }
</style>"""
    st.markdown(css, unsafe_allow_html=True)


inject_connext_brand_css()


def require_connext_login():
    if st.session_state.get("cnxt_authenticated"):
        return

    st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 1.35, 1])
    with center_col:
        st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
        st.image(CNXT_LOGO_COLOR, width=190)
        st.markdown(
            """<div class="cnxt-hero-banner" style="margin-top: 14px; padding: 24px 28px;">
<div class="cnxt-eyebrow">INTERNAL TEAM ACCESS</div>
<h2 style="color:#FFFFFF !important; margin:0 0 6px 0; font-size:1.5rem;">Connext Search, AI &amp; <span style="color:#E58B25;">Trend Radar</span></h2>
<p class="cnxt-hero-sub" style="font-size:0.88rem;">Sign in with your Connext team password to view Google Search Console, Bing Search, Google Trends, and Competitor intelligence.</p>
</div>""",
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

# ==============================================================================
# CONNEXT CORE COMMERCIAL PAGES (FALLBACK IF LIVE SITEMAP.XML IS UNREACHABLE)
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
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path


def classify_page_role(path: str, live_paths_set: set[str]) -> str:
    if path not in live_paths_set:
        return "Old / Redirected URL"
    if path == "/":
        return "Homepage"
    if path.startswith("/industries/") or path == "/hipaa-compliant-virtual-medical-assistants":
        return "Industry Page"
    if path.startswith("/teams/"):
        return "Team Solution Page"
    if path.startswith("/services/") or path in {
        "/our-service-model", "/our-differentiator", "/compare-by-model",
        "/connext-pricing", "/trust-compliance", "/roles-we-fill",
    }:
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
        return "Not in Top 20"
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_sitemap_paths() -> tuple[list[str], str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ConnextTrendRadar/1.0)"}

    try:
        req = urllib.request.Request(SITEMAP_XML_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
        locs = [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text]
        child_sitemaps = [u for u in locs if u.lower().endswith(".xml")]
        if child_sitemaps:
            all_urls = []
            for sm_url in child_sitemaps[:10]:
                c_req = urllib.request.Request(sm_url, headers=headers)
                with urllib.request.urlopen(c_req, timeout=10) as c_resp:
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
    live_df["is_money_page"] = live_df["role"].isin([
        "Industry Page", "Team Solution Page", "Service Model Page", "Location Hub", "Interactive Tool"
    ])
    live_df["is_content_page"] = live_df["role"].isin([
        "Blog / Guide", "Case Study", "White Paper", "One-Pager", "E-Book"
    ])
    live_df["slug_text"] = (
        live_df["path"]
        .str.replace(r"^/(industries|teams|services|locations|tools|blog|case-studies|white-paper|one-pager|e-books)/", "", regex=True)
        .str.replace(r"[/\-_]", " ", regex=True)
        .str.strip()
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
    mat = vec.fit_transform(live_df["slug_text"].fillna(""))
    return live_set, live_df, vec, mat


def check_topic_against_sitemap(
    topic: str,
    queries_sample: str,
    live_set: set[str],
    live_df: pd.DataFrame,
    vec: TfidfVectorizer,
    mat,
) -> dict:
    combined_text = f"{topic} {queries_sample}".strip()
    q_vec = vec.transform([combined_text])
    sims = cosine_similarity(q_vec, mat).flatten()

    scored = live_df.copy()
    scored["raw_sim"] = sims
    scored["sim"] = sims

    topic_tokens = {
        w for w in re.findall(r"[a-z0-9]+", topic.lower())
        if len(w) > 2 and w not in GENERIC_BRAND_STOPWORDS
    }

    def specific_token_overlap(slug_txt: str) -> float:
        if not topic_tokens:
            return 0.0
        slug_tokens = set(re.findall(r"[a-z0-9]+", str(slug_txt).lower()))
        return len(topic_tokens & slug_tokens) / len(topic_tokens)

    scored["token_overlap"] = scored["slug_text"].apply(specific_token_overlap)

    boosted_path = None
    for pat, target_path in MONEY_PAGE_BOOST_RULES:
        if pat.search(combined_text) and target_path in live_set:
            boosted_path = target_path
            scored.loc[scored["path"] == target_path, "sim"] += 0.35
            break

    money_df = scored[scored["is_money_page"]].sort_values("sim", ascending=False)
    content_df = scored[scored["is_content_page"]].sort_values(["raw_sim", "token_overlap"], ascending=False)

    best_money_row = money_df.iloc[0] if not money_df.empty else scored.iloc[0]
    best_money_url = best_money_row["full_url"]
    best_money_role = best_money_row["role"]
    best_money_score = float(best_money_row["sim"])

    existing_matches = scored[
        (scored["path"] != "/")
        & ((scored["raw_sim"] >= 0.28) | (scored["token_overlap"] >= 0.67))
    ].sort_values(["token_overlap", "raw_sim"], ascending=False)

    already_exists = len(existing_matches) > 0
    if already_exists:
        closest_url = existing_matches.iloc[0]["full_url"]
        closest_role = existing_matches.iloc[0]["role"]
        closest_pct = int(round(max(float(existing_matches.iloc[0]["raw_sim"]), float(existing_matches.iloc[0]["token_overlap"])) * 100))
    elif not content_df.empty:
        closest_url = content_df.iloc[0]["full_url"]
        closest_role = content_df.iloc[0]["role"]
        closest_pct = int(round(max(float(content_df.iloc[0]["raw_sim"]), float(content_df.iloc[0]["token_overlap"])) * 100))
    else:
        closest_url, closest_role, closest_pct = best_money_url, best_money_role, 0

    return {
        "best_money_url": best_money_url,
        "best_money_role": best_money_role,
        "best_money_score": round(best_money_score, 2),
        "has_strong_money_match": bool(boosted_path is not None or best_money_score >= 0.22),
        "already_exists_in_sitemap": already_exists,
        "closest_existing_url": closest_url,
        "closest_existing_role": closest_role,
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
    valid_h, seen_h = [], set()
    for cat, h in raw_headlines:
        hc = h.strip()
        if len(hc) <= 30 and hc.lower() not in seen_h:
            valid_h.append((cat, hc))
            seen_h.add(hc.lower())

    raw_desc = [
        f"Extend your team with dedicated {short_kw.lower()} talent built around your workflows.",
        "Overcome talent shortages and high overhead with transparent, custom offshore staffing.",
        "People-centric global teams that integrate seamlessly with your culture and operations.",
        "Scale faster with full operational support, high retention, and dedicated management.",
    ]
    return valid_h, [d[:90].strip() for d in raw_desc if len(d) <= 90]


# --- GSC EXPORT & API LOADERS ---
def _read_csv_or_zip(uploaded_file, preferred_names: list[str]) -> pd.DataFrame:
    fname = uploaded_file.name.lower()
    if fname.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            if not csv_names:
                raise ValueError("No CSV files found inside the uploaded ZIP archive.")
            chosen = csv_names[0]
            for pref in preferred_names:
                matches = [n for n in csv_names if pref in n.lower()]
                if matches:
                    chosen = matches[0]
                    break
            with zf.open(chosen) as f:
                return pd.read_csv(io.BytesIO(f.read()))
    return pd.read_csv(uploaded_file)


def parse_gsc_web_upload(uploaded_file, brand_regex: str, custom_negatives: str, exclude_local_ph: bool) -> tuple[pd.DataFrame, int]:
    df = _read_csv_or_zip(uploaded_file, preferred_names=["quer", "page", "chart", "date"])
    df.columns = [c.lower().strip() for c in df.columns]
    if "top queries" in df.columns and "query" not in df.columns:
        df = df.rename(columns={"top queries": "query"})
    if "top pages" in df.columns and "page" not in df.columns:
        df = df.rename(columns={"top pages": "page"})
    missing = {"date", "query", "page", "impressions", "position"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Export with Date + Query + Page dimensions.")
    df["date"] = pd.to_datetime(df["date"])
    df["query"] = df["query"].astype(str).str.lower().str.strip()
    df["page"] = df["page"].astype(str).apply(normalize_url)
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0).astype(int)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0)
    df = df[(df["query"] != "") & (df["impressions"] > 0)].copy()

    before_len = len(df)
    df = df[~df["query"].str.contains(STRICT_NON_INDUSTRY_REGEX, regex=True, na=False)]
    if exclude_local_ph:
        df = df[~df["query"].str.contains(LOCAL_PH_AND_STUDENT_REGEX, regex=True, na=False)]
    if custom_negatives.strip():
        df = df[~df["query"].str.contains(custom_negatives, regex=True, na=False)]
    excluded = before_len - len(df)
    df["is_branded"] = df["query"].str.contains(brand_regex, regex=True, na=False) if brand_regex else False
    return df, excluded


def parse_gsc_ai_report_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    df = _read_csv_or_zip(uploaded_file, preferred_names=["page", "date", "chart"])
    orig_cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in orig_cols}
    page_col = next((lower_map[c] for c in ["top pages", "page", "pages", "url", "landing page"] if c in lower_map), None)
    if page_col is None:
        raise ValueError("Could not find a 'Page' or 'Top pages' column in the Generative AI export.")
    if "date" in lower_map:
        imp_col = lower_map.get("impressions") or lower_map.get("ai impressions")
        if imp_col is None:
            num_cols = [c for c in orig_cols if pd.api.types.is_numeric_dtype(df[c])]
            imp_col = num_cols[0] if num_cols else None
        if imp_col is None:
            raise ValueError("Could not find an Impressions column.")
        return pd.DataFrame({
            "date": pd.to_datetime(df[lower_map["date"]]),
            "page": df[page_col].astype(str).apply(normalize_url),
            "ai_impressions": pd.to_numeric(df[imp_col], errors="coerce").fillna(0).astype(int),
        }), "timeseries"

    imp_cols = [c for c in orig_cols if "impression" in c.lower() and "diff" not in c.lower()]
    if len(imp_cols) >= 2:
        return pd.DataFrame({
            "page": df[page_col].astype(str).apply(normalize_url),
            "ai_imp_current": pd.to_numeric(df[imp_cols[0]], errors="coerce").fillna(0).astype(int),
            "ai_imp_previous": pd.to_numeric(df[imp_cols[1]], errors="coerce").fillna(0).astype(int),
        }), "comparison"
    if len(imp_cols) == 1:
        return pd.DataFrame({
            "page": df[page_col].astype(str).apply(normalize_url),
            "ai_imp_current": pd.to_numeric(df[imp_cols[0]], errors="coerce").fillna(0).astype(int),
            "ai_imp_previous": 0,
        }), "snapshot"
    raise ValueError("Unrecognized GSC Generative AI format.")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gsc_api_data(
    creds_dict: dict,
    site_url: str,
    lookback_days: int,
    brand_regex: str,
    custom_exclude_regex: str,
    exclude_local_ph: bool,
) -> tuple[pd.DataFrame, str, int, int]:
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
    )
    service = build("searchconsole", "v1", credentials=creds)
    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=lookback_days)

    rows, raw_total, excluded = [], 0, 0
    start_row, row_limit, max_rows = 0, 25000, 250000
    first_incomplete = None

    brand_pattern = re.compile(brand_regex, re.I) if brand_regex.strip() else None
    custom_pattern = re.compile(custom_exclude_regex, re.I) if custom_exclude_regex.strip() else None

    while start_row < max_rows:
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
            if r["impressions"] <= 0:
                continue
            clean_q = query.lower().strip()
            if not clean_q:
                continue
            raw_total += 1
            if STRICT_NON_INDUSTRY_REGEX.search(clean_q):
                excluded += 1
                continue
            if exclude_local_ph and LOCAL_PH_AND_STUDENT_REGEX.search(clean_q):
                excluded += 1
                continue
            if custom_pattern and custom_pattern.search(clean_q):
                excluded += 1
                continue
            rows.append({
                "date": date_str,
                "query": clean_q,
                "page": normalize_url(page),
                "impressions": int(r["impressions"]),
                "position": float(r["position"]),
                "is_branded": bool(brand_pattern.search(clean_q)) if brand_pattern else False,
            })
        if len(batch) < row_limit:
            break
        start_row += row_limit

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df, str(first_incomplete), excluded, raw_total


# ==============================================================================
# 100% FREE GOOGLE TRENDS (TRENDSPY + OFFICIAL RSS) & FREE BING SEARCH RADAR
# ==============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_google_trends_data(
    seed_queries: tuple[str, ...],
    geo: str = "US",
    timeframe: str = "today 1-m",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    100% Free Google Trends integration (zero API keys, zero SerpApi credits):
      1. trendspy -> Interest Over Time (0-100) + Rising Related Queries
      2. Official Google Trends Daily RSS (https://trends.google.com/trending/rss)
    """
    interest_rows = []
    rising_rows = []
    notes = []

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
                            interest_rows.append({
                                "date": pd.to_datetime(r[date_col]),
                                "keyword": kw,
                                "interest_score": int(r[kw]),
                            })
                notes.append("Interest Over Time loaded via free trendspy")

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
                                    rising_rows.append({
                                        "Seed Topic": kw,
                                        "Rising Query": rq_clean,
                                        "Growth / Surge": str(rr[v_c]),
                                    })
                except Exception:
                    continue
        except Exception:
            notes.append("Google Trends interactive endpoint temporarily rate-limited on cloud IP; RSS feed loaded below")

    rss_rows = []
    try:
        rss_url = f"https://trends.google.com/trending/rss?geo={urllib.parse.quote(geo)}"
        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
        for item in root.iter():
            if not item.tag.endswith("item"):
                continue
            title_txt, traffic_txt, pub_txt = "", "", ""
            for ch in item:
                if ch.tag.endswith("title") and ch.text:
                    title_txt = ch.text.strip()
                elif ch.tag.endswith("approx_traffic") and ch.text:
                    traffic_txt = ch.text.strip()
                elif ch.tag.endswith("pubDate") and ch.text:
                    pub_txt = ch.text.strip()[:22]
            if title_txt:
                rss_rows.append({
                    "Trending Search": title_txt,
                    "Approx. Search Volume": traffic_txt or "Trending",
                    "Published": pub_txt,
                })
    except Exception:
        pass

    iot_out = pd.DataFrame(interest_rows)
    rising_out = pd.DataFrame(rising_rows).drop_duplicates(subset=["Rising Query"]).reset_index(drop=True) if rising_rows else pd.DataFrame()
    rss_out = pd.DataFrame(rss_rows)
    return iot_out, rising_out, rss_out, notes


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_bing_webmaster_queries(
    api_key: str,
    site_url: str,
    brand_regex: str,
    exclude_local_ph: bool,
    custom_negatives: str,
) -> tuple[pd.DataFrame, str]:
    """
    100% Free Official Bing Webmaster Tools API (GetQueryStats):
    Pulls weekly Bing Search impressions, clicks, and average position for connextglobal.com.
    """
    enc_site = urllib.parse.quote(site_url.strip(), safe="")
    enc_key = urllib.parse.quote(api_key.strip(), safe="")
    url = f"https://ssl.bing.com/webmaster/api.svc/json/GetQueryStats?siteUrl={enc_site}&apikey={enc_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
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
        if not q_clean:
            continue
        if brand_pat and brand_pat.search(q_clean):
            continue
        if STRICT_NON_INDUSTRY_REGEX.search(q_clean):
            continue
        if exclude_local_ph and LOCAL_PH_AND_STUDENT_REGEX.search(q_clean):
            continue
        if custom_pat and custom_pat.search(q_clean):
            continue

        raw_pos = float(item.get("AvgImpressionPosition") or 0.0)
        rows.append({
            "date": dt,
            "query": q_clean,
            "impressions": int(item.get("Impressions") or 0),
            "clicks": int(item.get("Clicks") or 0),
            "position": round(raw_pos / 10.0, 1) if raw_pos > 0 else 0.0,
        })

    if not rows:
        return pd.DataFrame(), "Connected to Bing Webmaster Tools, but no non-branded B2B queries were returned."

    df = pd.DataFrame(rows)
    all_dates = sorted(df["date"].unique())
    recent_dates = all_dates[-2:] if len(all_dates) >= 2 else all_dates
    prior_dates = all_dates[-6:-2] if len(all_dates) >= 6 else all_dates[:-2]

    rec_df = df[df["date"].isin(recent_dates)].copy()
    rec_df["wpos"] = rec_df["position"] * rec_df["impressions"]
    agg_rec = rec_df.groupby("query", as_index=False).agg(
        bing_impr_recent=("impressions", "sum"),
        bing_clicks=("clicks", "sum"),
        wpos=("wpos", "sum"),
    )
    agg_rec["bing_rank"] = (agg_rec["wpos"] / np.maximum(agg_rec["bing_impr_recent"], 1)).round(1)

    if prior_dates:
        pri_df = df[df["date"].isin(prior_dates)].groupby("query", as_index=False).agg(bing_impr_prior=("impressions", "sum"))
        agg_rec = agg_rec.merge(pri_df, on="query", how="left").fillna({"bing_impr_prior": 0})
        scale = len(recent_dates) / max(len(prior_dates), 1)
        base_scaled = np.maximum(agg_rec["bing_impr_prior"] * scale, 1.0)
        agg_rec["growth_pct"] = (((agg_rec["bing_impr_recent"] / base_scaled) - 1.0) * 100).round(0).astype(int)
    else:
        agg_rec["growth_pct"] = 0

    agg_rec["vertical"] = agg_rec["query"].apply(classify_sector)
    agg_rec["bing_rank_label"] = agg_rec["bing_rank"].apply(human_rank_label)
    agg_rec = agg_rec.sort_values(["bing_impr_recent", "growth_pct"], ascending=False).reset_index(drop=True)
    return agg_rec, f"Auto-synced {len(agg_rec):,} non-branded B2B queries from Bing Webmaster Tools"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_free_bing_suggestions(seed_queries: tuple[str, ...]) -> pd.DataFrame:
    """100% Free Bing Search Autocomplete / Related Searches (zero API keys required)."""
    out = []
    for sq in seed_queries[:8]:
        try:
            url = f"https://api.bing.com/osjson.aspx?query={urllib.parse.quote(sq)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and len(data) >= 2:
                for sugg in data[1]:
                    s_clean = str(sugg).lower().strip()
                    if s_clean and s_clean != sq.lower().strip() and not STRICT_NON_INDUSTRY_REGEX.search(s_clean):
                        out.append({
                            "Seed Keyword": sq,
                            "Bing Live Search Suggestion": s_clean,
                            "Vertical": classify_sector(s_clean),
                        })
        except Exception:
            continue
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).drop_duplicates(subset=["Bing Live Search Suggestion"]).reset_index(drop=True)


# ==============================================================================
# AUTOMATED COMPETITOR INTELLIGENCE (SERPAPI + FREE SITEMAPS)
# ==============================================================================
def _serpapi_get(params: dict, api_key: str) -> dict:
    q_params = {**params, "api_key": api_key, "output": "json"}
    url = f"https://serpapi.com/search.json?{urllib.parse.urlencode(q_params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ConnextTrendRadar/1.0)"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


@st.cache_data(ttl=604800, show_spinner=False)
def run_serpapi_competitor_and_trends_scan(
    api_key: str,
    keywords_tuple: tuple[str, ...],
    include_google_trends: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    serp_rows, trends_rows, errors = [], [], []
    credits_used = 0

    def check_one_keyword(kw: str) -> list[dict]:
        data = _serpapi_get(
            {"engine": "google", "q": kw, "location": "United States", "hl": "en", "gl": "us", "num": 20},
            api_key,
        )
        if "error" in data:
            raise RuntimeError(data["error"])
        connext_pos = 0
        comp_hits = []
        for item in data.get("organic_results", []):
            link = str(item.get("link") or "")
            link_low = link.lower()
            pos = int(item.get("position") or 0)
            if "connextglobal.com" in link_low and connext_pos == 0:
                connext_pos = pos
            for cname, domain, _ in COMPETITORS_REGISTRY:
                if domain in link_low:
                    comp_hits.append({
                        "keyword": kw,
                        "competitor": cname,
                        "comp_google_pos": pos,
                        "comp_title": str(item.get("title") or "").strip(),
                        "comp_url": link,
                    })
        for h in comp_hits:
            h["connext_live_pos"] = connext_pos
        return comp_hits

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        fut_map = {executor.submit(check_one_keyword, kw): kw for kw in keywords_tuple}
        for fut in concurrent.futures.as_completed(fut_map):
            credits_used += 1
            try:
                serp_rows.extend(fut.result())
            except Exception as exc:
                errors.append(str(exc))

    if include_google_trends:
        for vertical_name, seed_q in TRENDS_SEED_TOPICS:
            try:
                credits_used += 1
                tdata = _serpapi_get(
                    {"engine": "google_trends", "q": seed_q, "data_type": "RELATED_QUERIES", "date": "today 1-m", "geo": "US"},
                    api_key,
                )
                rising_list = (tdata.get("related_queries") or {}).get("rising") or []
                for r_item in rising_list[:10]:
                    rq = str(r_item.get("query") or "").lower().strip()
                    if not rq or COMPETITOR_BRAND_REGEX.search(rq) or STRICT_NON_INDUSTRY_REGEX.search(rq):
                        continue
                    trends_rows.append({
                        "Vertical": vertical_name,
                        "Seed Topic": seed_q,
                        "Rising Query (Google Trends US)": rq,
                        "Trend Surge": str(r_item.get("value") or "Rising"),
                        "Surge Score": int(r_item.get("extracted_value") or 0),
                    })
            except Exception as exc:
                errors.append(f"Trends ({seed_q}): {exc}")

    serp_df = pd.DataFrame(serp_rows)
    if not serp_df.empty:
        serp_df = serp_df.drop_duplicates(subset=["keyword", "competitor"]).sort_values(["comp_google_pos", "keyword"]).reset_index(drop=True)

    trends_out_df = pd.DataFrame(trends_rows)
    if not trends_out_df.empty:
        trends_out_df = trends_out_df.drop_duplicates(subset=["Rising Query (Google Trends US)"]).sort_values("Surge Score", ascending=False).reset_index(drop=True)

    sync_ts = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    if errors and serp_df.empty and trends_out_df.empty:
        return serp_df, trends_out_df, f"SerpApi Error: {errors[0]}"
    return serp_df, trends_out_df, f"Synced via your SerpApi account ({credits_used} searches used • cached 7 days • {sync_ts})"


def _fetch_page_headline(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            chunk = resp.read(35000).decode("utf-8", errors="ignore")
        m_title = re.search(r"<title[^>]*>(.*?)</title>", chunk, re.I | re.S)
        if m_title:
            clean_t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m_title.group(1)))).strip()
            clean_t = re.split(r"\s+[|\-–—]\s+", clean_t)[0].strip()
            if len(clean_t) >= 5:
                return clean_t
    except Exception:
        pass
    return ""


@st.cache_data(ttl=86400, show_spinner=False)
def scan_competitor_sitemaps_auto(lookback_days: int = 45) -> tuple[pd.DataFrame, list[str]]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)

    def scan_one(comp_tuple: tuple[str, str, str]) -> tuple[list[dict], str]:
        cname, domain, _ = comp_tuple
        found = []
        sitemap_candidates = [
            f"https://{domain}/sitemap_index.xml",
            f"https://{domain}/sitemap.xml",
            f"https://www.{domain}/sitemap_index.xml",
            f"https://www.{domain}/sitemap.xml",
        ]
        parsed_any = False
        for sm_url in sitemap_candidates:
            try:
                req = urllib.request.Request(sm_url, headers=headers)
                with urllib.request.urlopen(req, timeout=7) as resp:
                    root = ET.fromstring(resp.read())
                parsed_any = True
                child_locs = [
                    el.text.strip()
                    for el in root.iter()
                    if el.tag.endswith("loc") and el.text and el.text.strip().lower().endswith(".xml")
                ]
                sub_roots = []
                if child_locs:
                    prioritized = sorted(
                        child_locs,
                        key=lambda u: (0 if any(k in u.lower() for k in ["post", "blog", "article", "page", "insight", "resource"]) else 1),
                    )[:4]
                    for sub_u in prioritized:
                        try:
                            s_req = urllib.request.Request(sub_u, headers=headers)
                            with urllib.request.urlopen(s_req, timeout=7) as s_resp:
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
                        if not loc_txt or not mod_txt:
                            continue
                        try:
                            mod_dt = datetime.strptime(mod_txt, "%Y-%m-%d").date()
                        except ValueError:
                            continue
                        if mod_dt >= cutoff_date:
                            path_str = urlparse(loc_txt).path.strip("/")
                            if not path_str or any(bad in path_str.lower() for bad in ["career", "job", "privacy", "terms", "author/", "tag/", "category/"]):
                                continue
                            slug_topic = re.sub(r"[-_/]+", " ", path_str.split("/")[-1]).strip()
                            if len(slug_topic) < 4:
                                continue
                            found.append({
                                "Competitor": cname,
                                "Published / Updated": mod_txt,
                                "Slug Topic": slug_topic,
                                "Vertical": classify_sector(f"{slug_topic} {path_str}"),
                                "Competitor URL": loc_txt,
                            })
                            if len(found) >= 25:
                                break
                    if len(found) >= 25:
                        break
                break
            except Exception:
                continue

        note = f"{cname}: {len(found)} new URLs" if parsed_any else f"{cname}: blocked"
        return found, note

    all_found, notes = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for f_rows, n_txt in executor.map(scan_one, COMPETITORS_REGISTRY):
            all_found.extend(f_rows)
            notes.append(n_txt)

    if not all_found:
        return pd.DataFrame(), notes

    out_df = pd.DataFrame(all_found).drop_duplicates(subset=["Competitor URL"]).sort_values("Published / Updated", ascending=False).reset_index(drop=True)
    top_urls = out_df["Competitor URL"].head(35).tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        titles = list(executor.map(_fetch_page_headline, top_urls))
    title_map = dict(zip(top_urls, titles))
    out_df["Live Page Headline"] = out_df.apply(
        lambda r: title_map.get(r["Competitor URL"]) or r["Slug Topic"].title(), axis=1
    )
    return out_df, notes


# --- CLUSTERING & MULTI-HORIZON TREND SCORING ---
@st.cache_data(show_spinner=False)
def cluster_queries(df: pd.DataFrame, distance_threshold: float) -> pd.DataFrame:
    non_brand = df[~df["is_branded"]].copy()
    if non_brand.empty:
        return non_brand
    non_brand["norm_page"] = non_brand["page"].apply(normalize_url)
    non_brand["path"] = non_brand["norm_page"].apply(url_to_path)

    query_stats = (
        non_brand.groupby("query", as_index=False)
        .agg(total_impressions=("impressions", "sum"))
        .sort_values("total_impressions", ascending=False)
    )
    top_queries = query_stats.head(3500).copy().reset_index(drop=True)
    if len(top_queries) < 2:
        non_brand["topic"] = non_brand["query"]
        non_brand["word_count"] = non_brand["query"].str.split().str.len()
        non_brand["is_long_tail"] = non_brand["word_count"] >= 4
        non_brand["is_ai_prompt"] = (non_brand["word_count"] >= 7) | non_brand["query"].str.contains(AI_PROMPT_REGEX, na=False)
        return non_brand

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(top_queries["query"]).astype(np.float32).toarray()
    row_norms = np.linalg.norm(tfidf_matrix, axis=1)
    valid_mask = row_norms > 1e-9

    cluster_ids = np.arange(-1, -len(top_queries) - 1, -1, dtype=int)
    if int(np.sum(valid_mask)) >= 2:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance_threshold
        )
        cluster_ids[valid_mask] = clustering.fit_predict(tfidf_matrix[valid_mask])
    top_queries["cluster_id"] = cluster_ids

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
    live_set: set[str],
    live_df: pd.DataFrame,
    live_vec: TfidfVectorizer,
    live_mat,
    min_daily_imp: int,
    min_up_days: int,
    min_norm_slope: float,
    min_lift: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if non_brand_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_dates = sorted(non_brand_df["date"].unique())
    n_days = len(all_dates)
    if n_days < 14:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    today_dates = all_dates[-3:]
    week_dates = all_dates[-7:]
    prior_14_dates = all_dates[-21:-7] if n_days >= 21 else all_dates[:-7]

    month_len = min(30, n_days)
    month_dates = all_dates[-month_len:]
    month_comp_len = min(21, n_days // 2)
    month_recent_dates = all_dates[-month_comp_len:]
    month_prior_dates = all_dates[-(month_comp_len * 2):-month_comp_len] if n_days >= month_comp_len * 2 else all_dates[:-month_comp_len]

    df = non_brand_df.copy()
    df["weighted_pos"] = df["position"] * df["impressions"]

    daily_topic = (
        df.groupby(["topic", "date"], as_index=False)
        .agg(impressions=("impressions", "sum"), weighted_pos=("weighted_pos", "sum"))
    )
    daily_topic["organic_position"] = (daily_topic["weighted_pos"] / np.maximum(daily_topic["impressions"], 1)).round(1)

    recent_df = df[df["date"].isin(week_dates)]
    prior_df = df[df["date"].isin(prior_14_dates)]
    month_df = df[df["date"].isin(month_dates)]

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
        recent_df.groupby(["topic", "norm_page", "path"], as_index=False)
        .agg(page_imp_7d=("impressions", "sum"), page_weighted_pos=("weighted_pos", "sum"), queries_on_page=("query", "nunique"))
        .sort_values(["topic", "page_imp_7d"], ascending=[True, False])
    )
    pages_agg["page_org_pos"] = (pages_agg["page_weighted_pos"] / np.maximum(pages_agg["page_imp_7d"], 1)).round(1)
    pages_agg["page_role"] = pages_agg["path"].apply(lambda p: classify_page_role(p, live_set))
    pages_agg["is_live_url"] = pages_agg["path"].isin(live_set)
    pages_agg["is_money_url"] = pages_agg["page_role"].isin([
        "Industry Page", "Team Solution Page", "Service Model Page", "Location Hub", "Interactive Tool"
    ])

    top_page_map = pages_agg.groupby("topic")["norm_page"].first().to_dict()
    top_page_role_map = pages_agg.groupby("topic")["page_role"].first().to_dict()
    has_money_ranking_map = pages_agg.groupby("topic")["is_money_url"].any().to_dict()
    has_legacy_url_map = (~pages_agg["is_live_url"]).groupby(pages_agg["topic"]).any().to_dict()
    page_count_map = pages_agg.groupby("topic")["norm_page"].nunique().to_dict()

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
        is_trending_today = bool(mean_today >= min_daily_imp and lift_today >= (min_lift + 0.15) and arr_today[-1] > arr_today[0])

        up_days_week = int(np.sum(np.diff(arr_week) > 0))
        slope_week = float(np.polyfit(np.arange(len(arr_week)), arr_week, 1)[0])
        norm_slope_week = slope_week / max(mean_week, 1.0)
        is_trending_week = bool(
            mean_today >= min_daily_imp and up_days_week >= min_up_days and norm_slope_week >= min_norm_slope and lift_today >= min_lift
        )

        lift_month = mean_month_rec / mean_month_base
        slope_month = float(np.polyfit(np.arange(len(arr_month_rec)), arr_month_rec, 1)[0])
        norm_slope_month = slope_month / max(mean_month_rec, 1.0)
        is_trending_month = bool(
            mean_month_rec >= (min_daily_imp * 0.8) and lift_month >= max(1.25, min_lift - 0.15) and norm_slope_month >= (min_norm_slope * 0.35)
        )

        if not (is_trending_today or is_trending_week or is_trending_month):
            continue

        rec_grp = group[group["date"].isin(week_dates)]
        tot_imp_7d = int(rec_grp["impressions"].sum())
        tot_imp_30d = int(group[group["date"].isin(month_dates)]["impressions"].sum())
        avg_pos_7d = float(rec_grp["weighted_pos"].sum() / max(tot_imp_7d, 1))

        queries_list = top_queries_map.get(topic, [topic])
        lt_list = long_tail_map.get(topic, [])
        joined_queries = ", ".join(queries_list[:6])
        has_commercial = bool(COMMERCIAL_MODIFIERS.search(joined_queries))
        unique_pages = int(page_count_map.get(topic, 1))

        gsc_top_url = top_page_map.get(topic, "")
        gsc_top_role = top_page_role_map.get(topic, "Blog / Guide")
        has_money_ranking = bool(has_money_ranking_map.get(topic, False))
        has_legacy_url = bool(has_legacy_url_map.get(topic, False))

        sector = classify_sector(f"{joined_queries} {gsc_top_url}")
        site_check = check_topic_against_sitemap(topic, joined_queries, live_set, live_df, live_vec, live_mat)
        recommended_lp = site_check["best_money_url"] if site_check["has_strong_money_match"] else gsc_top_url

        if has_legacy_url:
            coverage_status = "Old URL Redirect Check"
        elif has_money_ranking:
            coverage_status = "Service/Industry Page Ranking"
        elif site_check["already_exists_in_sitemap"]:
            coverage_status = "Already Covered in Sitemap"
        else:
            coverage_status = "Not in Sitemap (New Topic)"

        if has_commercial and avg_pos_7d > 5.0:
            team_action = "Launch Paid Ads"
        elif site_check["already_exists_in_sitemap"] or (4.0 <= avg_pos_7d <= 15.0):
            team_action = "Update Existing Page"
        elif not site_check["already_exists_in_sitemap"]:
            team_action = "Write New Page (Not in Sitemap)"
        else:
            team_action = "Defend Top 3 Rank"

        growth_pct_3d = round((lift_today - 1.0) * 100)
        growth_pct_week = round(((mean_week / mean_prior14) - 1.0) * 100)
        month_growth_pct = round((lift_month - 1.0) * 100)

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
            "team_action": team_action,
            "ads_fit_score": ads_fit_score,
            "daily_growth_%": round(norm_slope_week * 100, 1),
            "growth_pct_3d": growth_pct_3d,
            "growth_pct_week": growth_pct_week,
            "month_growth_pct": month_growth_pct,
            "impressions_7d": tot_imp_7d,
            "impressions_30d": tot_imp_30d,
            "daily_avg_7d": round(tot_imp_7d / 7.0, 1),
            "sparkline_7d": [int(v) for v in arr_week],
            "connext_org_pos": round(avg_pos_7d, 1),
            "google_rank_label": human_rank_label(avg_pos_7d),
            "commercial_intent": has_commercial,
            "intent_label": "Ready-to-Buy / Commercial" if has_commercial else "Research / How-To",
            "coverage_status": coverage_status,
            "already_exists_in_sitemap": site_check["already_exists_in_sitemap"],
            "closest_existing_url": site_check["closest_existing_url"],
            "closest_existing_role": site_check["closest_existing_role"],
            "closest_overlap_pct": site_check["closest_overlap_pct"],
            "existing_sitemap_matches": site_check["existing_sitemap_matches"],
            "ranking_pages_count": unique_pages,
            "primary_page": gsc_top_url,
            "primary_page_role": gsc_top_role,
            "recommended_money_page": recommended_lp,
            "best_money_role": site_check["best_money_role"],
            "matching_money_pages": site_check["top_money_pages"],
            "matching_content_pages": site_check["top_content_pages"],
            "queries_list": queries_list,
            "long_tail_list": lt_list,
            "total_queries": len(queries_list),
            "long_tail_count": len(lt_list),
            "top_rising_queries": joined_queries,
        })

    if not results:
        return pd.DataFrame(), daily_topic, pd.DataFrame(), pd.DataFrame()

    trends_df = pd.DataFrame(results).sort_values("impressions_7d", ascending=False).reset_index(drop=True)

    q_recent = (
        recent_df.groupby(["topic", "query", "is_long_tail", "is_ai_prompt", "word_count"], as_index=False)
        .agg(imp_7d=("impressions", "sum"), weighted_pos=("weighted_pos", "sum"), top_page=("norm_page", "first"), top_path=("path", "first"))
    )
    q_month = month_df.groupby("query", as_index=False).agg(imp_30d=("impressions", "sum"))
    q_base = prior_df.groupby("query", as_index=False).agg(imp_base14=("impressions", "sum"))

    q_merged = (
        q_recent.merge(q_month, on="query", how="left")
        .merge(q_base, on="query", how="left")
        .fillna({"imp_30d": 0, "imp_base14": 0})
    )
    q_merged["imp_30d"] = q_merged["imp_30d"].astype(int)
    q_merged["daily_7d"] = (q_merged["imp_7d"] / 7.0).round(1)
    q_merged["daily_base"] = np.maximum(q_merged["imp_base14"] / max(len(prior_14_dates), 1), 0.5)
    q_merged["query_lift_x"] = (q_merged["daily_7d"] / q_merged["daily_base"]).round(2)
    q_merged["query_growth_pct"] = ((q_merged["query_lift_x"] - 1.0) * 100).round(0).astype(int)
    q_merged["connext_org_pos"] = (q_merged["weighted_pos"] / np.maximum(q_merged["imp_7d"], 1)).round(1)
    q_merged["sector"] = q_merged.apply(lambda r: classify_sector(f"{r['query']} {r['top_page']}"), axis=1)
    q_merged["is_commercial"] = q_merged["query"].str.contains(COMMERCIAL_MODIFIERS, na=False)
    q_merged["page_role"] = q_merged["top_path"].apply(lambda p: classify_page_role(p, live_set))
    q_merged = q_merged.sort_values(["imp_7d", "imp_30d"], ascending=False)

    pages_merged = pages_agg[pages_agg["topic"].isin(trends_df["topic"])].copy()
    topic_tot_map = trends_df.set_index("topic")["impressions_7d"].to_dict()
    pages_merged["topic_imp_share_%"] = (
        (pages_merged["page_imp_7d"] / pages_merged["topic"].map(topic_tot_map).replace(0, 1)) * 100
    ).round(1)

    return trends_df, daily_topic, q_merged, pages_merged


def analyze_ai_search_verified(
    ai_df: pd.DataFrame,
    ai_mode: str,
    non_brand_web_df: pd.DataFrame,
    all_queries_df: pd.DataFrame,
    live_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if non_brand_web_df.empty or all_queries_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    web = non_brand_web_df.copy()
    web["norm_page"] = web["page"].apply(normalize_url)
    web["weighted_pos"] = web["position"] * web["impressions"]
    week_dates = sorted(web["date"].unique())[-7:]

    web_7d = (
        web[web["date"].isin(week_dates)]
        .groupby("norm_page", as_index=False)
        .agg(display_url=("norm_page", "first"), web_imp_7d=("impressions", "sum"), web_wpos=("weighted_pos", "sum"))
    )
    web_7d["web_org_pos"] = (web_7d["web_wpos"] / np.maximum(web_7d["web_imp_7d"], 1)).round(1)

    page_queries_grp = all_queries_df.sort_values(["imp_7d", "query_lift_x"], ascending=[False, False])
    connected_qs_map = {
        p: [f"{r['query']} ({r['imp_7d']:,} impr., #{r['connext_org_pos']})" for _, r in g.head(6).iterrows()]
        for p, g in page_queries_grp.groupby("top_page")
    }
    clusters_map = page_queries_grp.groupby("top_page")["topic"].apply(lambda s: ", ".join(dict.fromkeys(s.tolist()[:3]))).to_dict()

    ai_pages_df = pd.DataFrame()
    ai_daily_series = pd.DataFrame()

    if ai_df is not None and not ai_df.empty:
        page_records = []
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
                page_records.append({
                    "norm_page": norm_p,
                    "ai_imp_7d": int(np.sum(rec_7)),
                    "ai_lift_x": round(float(np.mean(rec_7[-3:])) / max(float(np.mean(base_14)), 1.0), 2),
                    "ai_sparkline": [int(v) for v in rec_7],
                })
        else:
            for _, row in ai_df.iterrows():
                curr, prev = int(row["ai_imp_current"]), int(row["ai_imp_previous"])
                page_records.append({
                    "norm_page": row["page"],
                    "ai_imp_7d": curr,
                    "ai_lift_x": round(curr / max(prev, 1), 2) if prev > 0 else 1.0,
                    "ai_sparkline": [prev, curr] if prev > 0 else [curr],
                })
        if page_records:
            ai_pages_df = pd.DataFrame(page_records).merge(web_7d, on="norm_page", how="left")
            ai_pages_df["display_url"] = ai_pages_df["display_url"].fillna(ai_pages_df["norm_page"])
            ai_pages_df["page_role"] = ai_pages_df["norm_page"].apply(lambda u: classify_page_role(url_to_path(u), live_set))
            ai_pages_df["web_imp_7d"] = ai_pages_df["web_imp_7d"].fillna(0).astype(int)
            ai_pages_df["connected_clusters"] = ai_pages_df["norm_page"].map(clusters_map).fillna("No Web Cluster Match")
            ai_pages_df["connected_trending_queries"] = ai_pages_df["norm_page"].map(
                lambda p: " | ".join(connected_qs_map.get(p, [])[:4]) or "No query match"
            )
            ai_pages_df["sector"] = ai_pages_df.apply(
                lambda r: classify_sector(f"{r['display_url']} {r['connected_clusters']} {r['connected_trending_queries']}"),
                axis=1,
            )
            ai_pages_df = ai_pages_df.sort_values(["ai_lift_x", "ai_imp_7d"], ascending=False).reset_index(drop=True)

    prompts_df = all_queries_df[all_queries_df["is_ai_prompt"]].sort_values(["imp_7d", "query_lift_x"], ascending=[False, False]).reset_index(drop=True)
    return ai_pages_df, ai_daily_series, prompts_df


def build_ads_recommendations(trends_df: pd.DataFrame, queries_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ads_topics, rec_keywords = [], []
    money_page_by_topic = trends_df.set_index("topic")["recommended_money_page"].to_dict()

    for _, r in trends_df.iterrows():
        pos, comm, lp_url = r["connext_org_pos"], r["commercial_intent"], r["recommended_money_page"]
        if r["is_today"] and comm and pos > 5.0:
            prio = "P0 — Launch Ads Today (24h)"
            why = f"Volume jumped +{r['growth_pct_3d']}% ({r['impressions_7d']:,} organic 7d impr.) while Connext ranks #{pos}. Route ads to {lp_url}."
        elif comm and pos > 5.0:
            prio = "P1 — High-Intent Search Ad Group"
            why = f"Growing buyer topic ({r['impressions_7d']:,} organic 7d impr.) where Connext ranks #{pos}. Route ads to {lp_url}."
        elif comm and pos <= 5.0:
            prio = "P2 — Defend Top-5 Google Spot"
            why = f"Connext ranks #{pos} organically ({r['impressions_7d']:,} 7d impr.). Pair Paid Search with organic."
        else:
            prio = "P2 — Long-Tail / Retargeting"
            why = f"Research surge (+{r['growth_pct_3d']}%, {r['impressions_7d']:,} 7d impr.). Bid on specific buyer modifiers."

        ads_topics.append({
            "Priority": prio,
            "Ad Fit Score": r["ads_fit_score"],
            "Vertical": r["sector"],
            "Trending Topic": r["topic"],
            "Organic 7d Impr.": r["impressions_7d"],
            "Organic 30d Impr.": r["impressions_30d"],
            "Daily Avg Impr.": r["daily_avg_7d"],
            "Growth (vs 2w)": f"+{r['growth_pct_3d']}%",
            "Current Google Rank": r["google_rank_label"],
            "Best Landing Page": lp_url,
            "Why Run Ads": why,
        })

    for _, q in queries_df.iterrows():
        kw, is_lt, is_comm = q["query"], q["is_long_tail"], q["is_commercial"]
        q_growth = int(q["query_growth_pct"])
        if is_comm and not is_lt:
            ktier, mtype, tip = "Core Buyer Keyword", '"Phrase" + [Exact]', "Bid for Top of Page"
        elif is_comm and is_lt:
            ktier, mtype, tip = "Specific Long-Tail Buyer (Lower CPC)", "[Exact Match]", "High-Intent Exact Bid"
        elif is_lt:
            ktier, mtype, tip = "Question / Research Long-Tail", "[Exact Match] (Low Cap)", "Test with Low CPC Cap"
        else:
            ktier, mtype, tip = "Broad Category Term", '"Phrase Match"', "Moderate Bid + Job Negatives"

        rec_keywords.append({
            "Suggested Keyword": kw,
            "Organic 7d Impr.": int(q["imp_7d"]),
            "Organic 30d Impr.": int(q["imp_30d"]),
            "Daily Avg Impr.": float(q["daily_7d"]),
            "Growth (vs 2w)": f"+{q_growth}%" if q_growth >= 0 else f"{q_growth}%",
            "Current Google Rank": human_rank_label(q["connext_org_pos"]),
            "Keyword Type": ktier,
            "Match Type": mtype,
            "Topic Cluster": q["topic"],
            "Vertical": q["sector"],
            "Recommended Landing Page": money_page_by_topic.get(q["topic"], q["top_page"]),
            "Bidding Tip": tip,
        })

    ads_df = pd.DataFrame(ads_topics).sort_values(["Ad Fit Score", "Organic 7d Impr."], ascending=False)
    kws_df = pd.DataFrame(rec_keywords).sort_values(["Organic 7d Impr.", "Organic 30d Impr."], ascending=False) if rec_keywords else pd.DataFrame()
    return ads_df, kws_df


def build_content_and_new_topics(trends_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_rows, truly_new_rows = [], []
    for _, r in trends_df.iterrows():
        pos, money_lp, sitemap_url = r["connext_org_pos"], r["recommended_money_page"], r["closest_existing_url"]
        if r["coverage_status"] == "Old URL Redirect Check":
            prio, act = "P0 — Fix Old URL Redirect", f"Confirm old URL ({r['primary_page']}) redirects to {money_lp}."
        elif r["already_exists_in_sitemap"]:
            prio = "P1 — Update Existing Sitemap Page (No Duplicate)"
            act = f"Page {sitemap_url} already covers this ({r['closest_overlap_pct']}% match). Update headings/FAQs and link to {money_lp}."
        else:
            prio, act = "P2 — Protect Ranking", f"Connext ranks #{pos}. Keep FAQs and statistics updated."

        existing_rows.append({
            "Priority": prio,
            "Sitemap Status": "Already in Sitemap.xml" if r["already_exists_in_sitemap"] else "Service Page Exists",
            "Vertical": r["sector"],
            "Trending Topic": r["topic"],
            "7d Impressions": r["impressions_7d"],
            "Growth (vs 2w)": f"+{r['growth_pct_3d']}%",
            "Current Google Rank": r["google_rank_label"],
            "Existing Sitemap Page": sitemap_url,
            "Main Service Page": money_lp,
            "Recommended Action": act,
        })

        if not r["already_exists_in_sitemap"]:
            slug_base = re.sub(r"[^a-z0-9]+", "-", r["topic"].lower()).strip("-")
            truly_new_rows.append({
                "Recommended Asset": "New Guide / Subpage (Not in Sitemap)",
                "Vertical": r["sector"],
                "New Topic Gap": r["topic"],
                "7d Impressions": r["impressions_7d"],
                "30d Impressions": r["impressions_30d"],
                "Growth (vs 2w)": f"+{r['growth_pct_3d']}%",
                "Draft URL Slug (Not Live Yet)": f"/blog/{slug_base}-guide",
                "Closest Existing Sitemap Page": sitemap_url,
                "Similarity": f"{r['closest_overlap_pct']}% (Low — Safe)",
                "Service Page to Link To": money_lp,
                "Rising Searches to Cover": r["top_rising_queries"],
            })

    return pd.DataFrame(existing_rows), (
        pd.DataFrame(truly_new_rows).sort_values(["7d Impressions", "30d Impressions"], ascending=False)
        if truly_new_rows else pd.DataFrame()
    )


# --- LOAD LIVE CONNEXT SITEMAP ---
sitemap_paths, sitemap_source_label = fetch_live_sitemap_paths()
live_set, live_df, live_vec, live_mat = _build_index_from_paths(sitemap_paths)

# --- SIDEBAR CONTROLS ---
st.sidebar.image(CNXT_LOGO_WHITE, width=185)
st.sidebar.markdown("**Automated Search & Competitor Radar**")
st.sidebar.caption(f"`connextglobal.com` • {sitemap_source_label}")
if st.sidebar.button("Lock / Log Out"):
    st.session_state["cnxt_authenticated"] = False
    st.rerun()
st.sidebar.divider()

has_secrets = "gcp_service_account" in st.secrets and "gsc_config" in st.secrets
data_source = st.sidebar.radio(
    "1. Search Console Data Source",
    ["Live Google Search Console API", "Upload GSC Web Export (.csv / .zip)"],
    index=0,
)
exclude_local_ph = st.sidebar.checkbox("Exclude Local PH City & School Searches", value=True)
custom_negatives = st.sidebar.text_input("Extra Words to Exclude (Pipe-separated)", placeholder="intern|internship|sample|pdf")

raw_df, ai_raw_df = pd.DataFrame(), pd.DataFrame()
ai_csv_mode, incomplete_date_note = "timeseries", None
excluded_rows_count, raw_total_count = 0, 0

if data_source == "Live Google Search Console API":
    if has_secrets:
        site_url = st.sidebar.text_input("GSC Property URL", value=st.secrets["gsc_config"].get("site_url", DEFAULT_SITE_URL))
        brand_regex = st.sidebar.text_input("Brand Terms Regex", value=st.secrets["gsc_config"].get("brand_regex", DEFAULT_BRAND_REGEX))
        lookback = st.sidebar.slider("Days of History to Analyze", 28, 90, 45)
        if st.sidebar.button("Refresh Live GSC Data", type="primary"):
            fetch_gsc_api_data.clear()
        with st.spinner("Loading live Google Search Console data for connextglobal.com..."):
            try:
                raw_df, incomplete_date_note, excluded_rows_count, raw_total_count = fetch_gsc_api_data(
                    dict(st.secrets["gcp_service_account"]), site_url, lookback, brand_regex, custom_negatives, exclude_local_ph
                )
            except Exception as api_err:
                st.error(f"Google Search Console API Error: {api_err}")
                st.stop()
    else:
        uploaded_json = st.sidebar.file_uploader("Service Account JSON", type=["json"])
        site_url = st.sidebar.text_input("GSC Property URL", value=DEFAULT_SITE_URL)
        brand_regex = st.sidebar.text_input("Brand Terms Regex", value=DEFAULT_BRAND_REGEX)
        lookback = st.sidebar.slider("Days of History to Analyze", 28, 90, 45)
        if uploaded_json and site_url:
            raw_df, incomplete_date_note, excluded_rows_count, raw_total_count = fetch_gsc_api_data(
                json.load(uploaded_json), site_url, lookback, brand_regex, custom_negatives, exclude_local_ph
            )
else:
    uploaded_csv = st.sidebar.file_uploader("Upload GSC Web CSV or ZIP", type=["csv", "zip"])
    brand_regex = st.sidebar.text_input("Brand Terms Regex", value=DEFAULT_BRAND_REGEX)
    if uploaded_csv:
        try:
            raw_df, excluded_rows_count = parse_gsc_web_upload(uploaded_csv, brand_regex, custom_negatives, exclude_local_ph)
            raw_total_count = len(raw_df) + excluded_rows_count
        except Exception as err:
            st.sidebar.error(f"Upload Error: {err}")

st.sidebar.divider()
st.sidebar.subheader("2. Free Feeds & Sync Status")
if st.sidebar.button("Re-Sync connextglobal.com/sitemap.xml"):
    fetch_live_sitemap_paths.clear()
    st.rerun()

has_bing_secret = "bing" in st.secrets and "api_key" in st.secrets["bing"] and bool(str(st.secrets["bing"]["api_key"]).strip())
has_serpapi_secret = "serpapi" in st.secrets and "api_key" in st.secrets["serpapi"] and bool(str(st.secrets["serpapi"]["api_key"]).strip())

st.sidebar.caption(
    f"Free Google Trends: **Active ($0)**  •  "
    f"Bing Webmaster: **{'Connected' if has_bing_secret else 'Suggestions Only ($0)'}**"
)

uploaded_ai_csv = st.sidebar.file_uploader("Optional: GSC Generative AI (.csv / .zip)", type=["csv", "zip"], key="ai_csv_uploader")
if uploaded_ai_csv:
    try:
        ai_raw_df, ai_csv_mode = parse_gsc_ai_report_csv(uploaded_ai_csv)
        st.sidebar.success(f"Loaded {len(ai_raw_df):,} AI rows.")
    except Exception as exc:
        st.sidebar.error(f"AI Export Error: {exc}")

st.sidebar.divider()
st.sidebar.subheader("3. Trend Sensitivity")
preset_mode = st.sidebar.selectbox(
    "Sensitivity Preset",
    ["Broad & Inclusive (Shows All Verticals)", "Balanced (Moderate Surges)", "Strict (Major Breakouts Only)"],
    index=0,
)
if preset_mode == "Strict (Major Breakouts Only)":
    def_imp, def_up, def_lift, def_slope = 50, 4, 1.40, 10
elif preset_mode == "Balanced (Moderate Surges)":
    def_imp, def_up, def_lift, def_slope = 30, 4, 1.30, 8
else:
    def_imp, def_up, def_lift, def_slope = 15, 3, 1.18, 4

with st.sidebar.expander("Advanced Thresholds", expanded=False):
    min_daily_imp = st.number_input("Min Avg Daily Impressions", min_value=5, max_value=10000, value=def_imp, step=5)
    min_up_days = st.slider("Weekly Upward Days (out of 6)", min_value=2, max_value=6, value=def_up)
    min_lift = st.slider("Min Growth Multiplier (vs 2w avg)", min_value=1.05, max_value=3.0, value=def_lift, step=0.05)
    min_slope_pct = st.slider("Min Daily Trend Slope (%)", min_value=1, max_value=40, value=def_slope, step=1)
    cluster_dist = st.slider("Keyword Grouping Breadth", min_value=0.30, max_value=0.75, value=0.55, step=0.05)

# --- HEADER & HERO BANNER ---
hdr_logo, hdr_meta = st.columns([1.2, 4])
with hdr_logo:
    st.image(CNXT_LOGO_COLOR, width=195)
with hdr_meta:
    if not raw_df.empty:
        date_badge = f"Verified GSC Window: {raw_df['date'].min().strftime('%b %d, %Y')} – {raw_df['date'].max().strftime('%b %d, %Y')} ({raw_df['date'].nunique()} days)"
    else:
        date_badge = "Awaiting GSC Data"
    st.caption(f"{date_badge}  •  {sitemap_source_label}  •  Free Google Trends & Bing Integrated")

st.markdown(
    """<div class="cnxt-hero-banner">
<div class="cnxt-eyebrow">CONNEXT GLOBAL AUTOMATED SEARCH &amp; COMPETITIVE INTELLIGENCE</div>
<h1 class="cnxt-hero-title">Search, AI, Bing &amp; <span>Trend Radar</span></h1>
<p class="cnxt-hero-sub">Track Google Search Console momentum, free Google Trends curves &amp; rising searches, Bing Search performance, live <strong>sitemap.xml</strong> anti-duplicate checks, and all 12 BPO competitors in one place.</p>
</div>""",
    unsafe_allow_html=True,
)

if raw_df.empty:
    st.info("Configure GSC credentials or upload a Search Console export in the sidebar to load live Connext data.")
    st.stop()

with st.spinner("Analyzing live GSC trends & checking connextglobal.com/sitemap.xml..."):
    clustered_df = cluster_queries(raw_df, cluster_dist)
    trends_df, daily_topic_df, all_queries_df, pages_df = analyze_multi_horizon_trends(
        clustered_df, live_set, live_df, live_vec, live_mat,
        min_daily_imp, min_up_days, min_slope_pct / 100.0, min_lift,
    )
    ai_pages_df, ai_daily_series, ai_prompts_df = analyze_ai_search_verified(
        ai_raw_df, ai_csv_mode, clustered_df, all_queries_df, live_set
    )

if trends_df.empty:
    st.warning("No topics matched your sensitivity settings. Select **Broad & Inclusive** in the sidebar.")
    st.stop()

# --- FILTER BAR ---
vertical_counts = trends_df["sector"].value_counts().to_dict()
vertical_options = ["All Business Verticals"] + [f"{v} ({vertical_counts.get(v, 0)} topics)" for v in ALL_CONNEXT_VERTICALS]
action_counts = trends_df["team_action"].value_counts().to_dict()
action_options = [
    "All Recommended Actions",
    f"Launch Paid Ads ({action_counts.get('Launch Paid Ads', 0)})",
    f"Update Existing Page ({action_counts.get('Update Existing Page', 0)})",
    f"Write New Page (Not in Sitemap) ({action_counts.get('Write New Page (Not in Sitemap)', 0)})",
    f"Defend Top 3 Rank ({action_counts.get('Defend Top 3 Rank', 0)})",
]

f1, f2, f3, f4 = st.columns([1.45, 1.45, 1.05, 1.15])
with f1:
    selected_vertical_label = st.selectbox("Business Vertical:", vertical_options, index=0)
with f2:
    selected_action_label = st.selectbox("Recommended Next Step:", action_options, index=0)
with f3:
    intent_filter = st.selectbox("Search Type:", ["All Searches", "Ready-to-Buy / Commercial", "Research / How-To"])
with f4:
    quick_search = st.text_input("Search Topic, Keyword, or URL:", placeholder="e.g., medical billing, cpa, python...")

filtered_trends = trends_df.copy()
if selected_vertical_label != "All Business Verticals":
    filtered_trends = filtered_trends[filtered_trends["sector"] == selected_vertical_label.rsplit(" (", 1)[0]]
if selected_action_label != "All Recommended Actions":
    filtered_trends = filtered_trends[filtered_trends["team_action"] == selected_action_label.rsplit(" (", 1)[0]]
if intent_filter == "Ready-to-Buy / Commercial":
    filtered_trends = filtered_trends[filtered_trends["commercial_intent"]]
elif intent_filter == "Research / How-To":
    filtered_trends = filtered_trends[~filtered_trends["commercial_intent"]]
if quick_search.strip():
    qs_pat = re.escape(quick_search.strip())
    filtered_trends = filtered_trends[
        filtered_trends["topic"].str.contains(qs_pat, case=False, na=False)
        | filtered_trends["top_rising_queries"].str.contains(qs_pat, case=False, na=False)
        | filtered_trends["primary_page"].str.contains(qs_pat, case=False, na=False)
        | filtered_trends["closest_existing_url"].str.contains(qs_pat, case=False, na=False)
    ]
filtered_trends = filtered_trends.reset_index(drop=True)

if filtered_trends.empty:
    st.info("No surging topics match those filters. Reset **Business Vertical** or **Recommended Next Step** above.")
    st.stop()

filtered_queries = all_queries_df[all_queries_df["topic"].isin(filtered_trends["topic"])].copy()
filtered_pages = pages_df[pages_df["topic"].isin(filtered_trends["topic"])].copy()
filtered_ai_pages = (
    ai_pages_df[ai_pages_df["sector"] == selected_vertical_label.rsplit(" (", 1)[0]].copy()
    if selected_vertical_label != "All Business Verticals" and not ai_pages_df.empty
    else (ai_pages_df.copy() if not ai_pages_df.empty else pd.DataFrame())
)
filtered_ai_prompts = ai_prompts_df[ai_prompts_df["topic"].isin(filtered_trends["topic"])].copy() if not ai_prompts_df.empty else pd.DataFrame()

ads_topics_df, rec_kws_df = build_ads_recommendations(filtered_trends, filtered_queries)
existing_content_df, truly_new_df = build_content_and_new_topics(filtered_trends)

# --- TOP KPI METRICS ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Surging Today (3d)", int(filtered_trends["is_today"].sum()))
k2.metric("Surging This Week (7d)", int(filtered_trends["is_week"].sum()))
k3.metric("Growing This Month (30d)", int(filtered_trends["is_month"].sum()))
k4.metric("Weekly Impressions", f"{int(filtered_trends['impressions_7d'].sum()):,}")
k5.metric("Avg Google Rank", f"#{filtered_trends['connext_org_pos'].mean():.1f}")

# --- 7 MAIN NAVIGATION TABS ---
ai_tab_count = len(filtered_ai_pages) if not filtered_ai_pages.empty else len(filtered_ai_prompts)
tab_horizons, tab_pages_clusters, tab_ai_search, tab_ads_strategy, tab_content, tab_competitors, tab_free_trends_bing = st.tabs([
    "Trending Topics",
    f"Pages & Queries ({len(filtered_pages)})",
    f"AI Search ({ai_tab_count})",
    f"Paid Ads & Keywords ({len(rec_kws_df)})",
    f"Content Plan ({len(truly_new_df)} New)",
    "Competitor SEO (12)",
    "Free Google Trends & Bing ($0)",
])

# ==============================================================================
# TAB 1: TRENDING TOPICS OVERVIEW
# ==============================================================================
with tab_horizons:
    horizon_choice = st.radio(
        "Timeframe:",
        ["All Surging Topics", "Today (3-Day Spike)", "This Week (7-Day Surge)", "This Month (30-Day Growth)"],
        horizontal=True,
    )
    view_df = filtered_trends.copy()
    growth_col, growth_col_label = "growth_pct_3d", "3-Day Surge (vs 2w)"
    if "Today" in horizon_choice:
        view_df = view_df[view_df["is_today"]]
    elif "This Week" in horizon_choice:
        view_df = view_df[view_df["is_week"]]
        growth_col, growth_col_label = "growth_pct_week", "7-Day Growth (vs 2w)"
    elif "This Month" in horizon_choice:
        view_df = view_df[view_df["is_month"]]
        growth_col, growth_col_label = "month_growth_pct", "30-Day Growth"

    if view_df.empty:
        st.info(f"No topics flagged under **{horizon_choice}**.")
    else:
        g1, g2 = st.columns([1.15, 1])
        with g1:
            st.subheader("Top Surging Topics by Weekly Impressions")
            top_bars = view_df.head(12).copy()
            fig_topics = px.bar(
                top_bars, x="impressions_7d", y="topic", orientation="h", color="team_action", text=growth_col,
                color_discrete_map={
                    "Launch Paid Ads": CNXT_COLORS["sun"],
                    "Update Existing Page": CNXT_COLORS["blue"],
                    "Write New Page (Not in Sitemap)": CNXT_COLORS["twilight"],
                    "Defend Top 3 Rank": CNXT_COLORS["teal"],
                },
                custom_data=["sector", "google_rank_label", growth_col, "closest_existing_url"],
                labels={"impressions_7d": "7-Day Impressions", "topic": "", "team_action": ""},
            )
            fig_topics.update_traces(
                texttemplate="+%{text}%", textposition="outside",
                hovertemplate="<b>%{y}</b><br>7d Impr: <b>%{x:,}</b><br>Growth: <b>+%{customdata[2]}%</b><br>Rank: <b>%{customdata[1]}</b><br>Vertical: %{customdata[0]}<extra></extra>",
            )
            fig_topics.update_layout(height=420, margin=dict(l=10, r=45, t=10, b=50), yaxis=dict(autorange="reversed"), xaxis=dict(gridcolor=CNXT_COLORS["line"]), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig_topics, use_container_width=True)

        with g2:
            st.subheader("Weekly Impressions by Business Vertical")
            sec_agg = view_df.groupby("sector", as_index=False).agg(topics=("topic", "count"), impressions_7d=("impressions_7d", "sum")).sort_values("impressions_7d", ascending=True)
            fig_sec = px.bar(sec_agg, x="impressions_7d", y="sector", orientation="h", text="topics", color_discrete_sequence=[CNXT_COLORS["twilight"]], labels={"impressions_7d": "7-Day Impressions", "sector": ""})
            fig_sec.update_traces(texttemplate="%{text} topics", textposition="outside")
            fig_sec.update_layout(height=420, margin=dict(l=10, r=55, t=10, b=50), xaxis=dict(gridcolor=CNXT_COLORS["line"]), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sec, use_container_width=True)

        st.divider()
        selected_topic = st.selectbox("Inspect Daily Search Trend:", view_df["topic"].tolist(), key="horizon_topic_select")
        sel_row = view_df[view_df["topic"] == selected_topic].iloc[0]
        st.markdown(
            f"""<div class="cnxt-cream-panel">
<strong>"{sel_row['topic'].title()}":</strong> <strong>{sel_row['impressions_7d']:,}</strong> 7-day impr. / <strong>{sel_row['impressions_30d']:,}</strong> 30-day impr. (<strong>+{sel_row[growth_col]}%</strong> {growth_col_label}) • Rank: <strong>{sel_row['google_rank_label']}</strong> • Next Step: <strong>{sel_row['team_action']}</strong> → <code>{sel_row['recommended_money_page']}</code>
</div>""",
            unsafe_allow_html=True,
        )

        ts = daily_topic_df[daily_topic_df["topic"] == selected_topic].sort_values("date").copy()
        ts["rolling_7d"] = ts["impressions"].rolling(7, min_periods=1).mean().round(0)
        fig_line = go.Figure()
        fig_line.add_trace(go.Bar(x=ts["date"], y=ts["impressions"], name="Daily Impressions", marker_color="rgba(0,90,172,0.30)"))
        fig_line.add_trace(go.Scatter(x=ts["date"], y=ts["rolling_7d"], mode="lines+markers", name="7-Day Avg Trend", line=dict(color=CNXT_COLORS["sun"], width=3), marker=dict(color=CNXT_COLORS["twilight"], size=6)))
        fig_line.update_layout(height=310, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="Impressions", gridcolor=CNXT_COLORS["line"]), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_line, use_container_width=True)

        ov = view_df[["topic", "sector", "team_action", "coverage_status", "sparkline_7d", "impressions_7d", "impressions_30d", growth_col, "google_rank_label", "closest_existing_url", "recommended_money_page"]].copy()
        ov["growth_str"] = ov[growth_col].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
        st.dataframe(
            ov[["topic", "sector", "team_action", "coverage_status", "sparkline_7d", "impressions_7d", "impressions_30d", "growth_str", "google_rank_label", "closest_existing_url", "recommended_money_page"]],
            use_container_width=True, hide_index=True, key="tbl_overview",
            column_config={
                "topic": "Trending Topic", "sector": "Vertical", "team_action": "Recommended Action", "coverage_status": "Sitemap Check",
                "sparkline_7d": st.column_config.LineChartColumn("7d Curve", width="small"),
                "impressions_7d": st.column_config.NumberColumn("7d Impr."), "impressions_30d": st.column_config.NumberColumn("30d Impr."),
                "growth_str": growth_col_label, "google_rank_label": "Google Rank",
                "closest_existing_url": st.column_config.LinkColumn("Existing Sitemap Page"),
                "recommended_money_page": st.column_config.LinkColumn("Best Service Page"),
            },
        )

# ==============================================================================
# TAB 2: PAGES ON TOPICS & LONG-TAIL QUERIES
# ==============================================================================
with tab_pages_clusters:
    st.subheader("Connext Pages & Exact Searches per Topic")
    inspect_cluster = st.selectbox("Select Topic Cluster:", filtered_trends["topic"].tolist(), key="cluster_detail_select")
    topic_meta = filtered_trends[filtered_trends["topic"] == inspect_cluster].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Google Rank", topic_meta["google_rank_label"])
    m2.metric("Sitemap Status", topic_meta["coverage_status"])
    m3.metric("Pages Ranking", topic_meta["ranking_pages_count"])
    m4.metric("Long-Tail (4+w)", f"{topic_meta['long_tail_count']} / {topic_meta['total_queries']}")

    col_p, col_live = st.columns([1.15, 1])
    with col_p:
        st.markdown("**1. Connext Pages Currently Ranking on Google**")
        tp = filtered_pages[filtered_pages["topic"] == inspect_cluster].sort_values("page_imp_7d", ascending=False).copy()
        tp["rank_text"] = tp["page_org_pos"].apply(human_rank_label)
        st.dataframe(
            tp[["norm_page", "page_role", "page_imp_7d", "topic_imp_share_%", "rank_text", "queries_on_page"]],
            use_container_width=True, hide_index=True, key="tbl_topic_pages",
            column_config={"norm_page": st.column_config.LinkColumn("Ranking URL"), "page_imp_7d": st.column_config.NumberColumn("7d Impr."), "topic_imp_share_%": st.column_config.NumberColumn("Share %", format="%.1f%%")},
        )
    with col_live:
        st.markdown("**2. Matching Live Pages Already in `sitemap.xml`**")
        recs, seen = [], set()
        for label, src in [("Already Covers Topic", topic_meta["existing_sitemap_matches"]), ("Commercial Service Page", topic_meta["matching_money_pages"]), ("Related Sitemap Content", topic_meta["matching_content_pages"])]:
            for item in src:
                if item["full_url"] not in seen:
                    recs.append({"Live Sitemap URL": item["full_url"], "Page Type": item["role"], "Status": label})
                    seen.add(item["full_url"])
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True, key="tbl_sitemap_matches", column_config={"Live Sitemap URL": st.column_config.LinkColumn("Live Sitemap URL")})

    st.markdown("**3. Exact Searches People Typed (With Real 7d & 30d Impressions)**")
    tqs = filtered_queries[filtered_queries["topic"] == inspect_cluster].sort_values("imp_7d", ascending=False).copy()
    tqs["growth_str"] = tqs["query_growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
    tqs["rank_str"] = tqs["connext_org_pos"].apply(human_rank_label)
    st.dataframe(
        tqs[["query", "is_long_tail", "imp_7d", "imp_30d", "daily_7d", "growth_str", "rank_str", "top_page"]],
        use_container_width=True, hide_index=True, key="tbl_topic_queries",
        column_config={"is_long_tail": st.column_config.CheckboxColumn("Long-Tail"), "top_page": st.column_config.LinkColumn("Ranking URL")},
    )

# ==============================================================================
# TAB 3: AI SEARCH & CONVERSATIONAL PROMPTS
# ==============================================================================
with tab_ai_search:
    st.subheader("AI Overviews & AI Mode Visibility")
    if not filtered_ai_pages.empty:
        st.dataframe(
            filtered_ai_pages[["display_url", "page_role", "sector", "ai_sparkline", "ai_imp_7d", "web_imp_7d", "connected_clusters", "connected_trending_queries"]],
            use_container_width=True, hide_index=True, key="tbl_ai_pages",
            column_config={"display_url": st.column_config.LinkColumn("AI Page"), "ai_sparkline": st.column_config.LineChartColumn("AI Trend", width="small")},
        )
    else:
        st.markdown(
            """<div class="cnxt-cream-panel">
Showing live 7+ word conversational prompts from Google Search Console Web data. (Optional: upload a GSC Generative AI CSV/ZIP in the sidebar to isolate AI-only page counts.)
</div>""",
            unsafe_allow_html=True,
        )

    if not filtered_ai_prompts.empty:
        pd_ai = filtered_ai_prompts.copy()
        pd_ai["growth_str"] = pd_ai["query_growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
        pd_ai["rank_str"] = pd_ai["connext_org_pos"].apply(human_rank_label)
        st.dataframe(
            pd_ai[["query", "sector", "topic", "imp_7d", "imp_30d", "growth_str", "rank_str", "top_page"]],
            use_container_width=True, hide_index=True, key="tbl_ai_prompts",
            column_config={"top_page": st.column_config.LinkColumn("Ranking Connext Page")},
        )
        st.download_button("Download AI Prompts CSV", pd_ai[["query", "sector", "topic", "imp_7d", "imp_30d", "growth_str", "rank_str", "top_page"]].to_csv(index=False).encode("utf-8"), file_name="connext_ai_prompts.csv", mime="text/csv", key="dl_ai_prompts")

# ==============================================================================
# TAB 4: PAID ADS & SUGGESTED KEYWORDS
# ==============================================================================
with tab_ads_strategy:
    st.subheader("Google Ads Opportunities & Suggested Keywords")
    st.caption("Note: Impression columns show real organic Google Search Console impressions for connextglobal.com.")

    st.dataframe(
        ads_topics_df, use_container_width=True, hide_index=True, key="tbl_ads_topics",
        column_config={"Ad Fit Score": st.column_config.ProgressColumn("Ad Fit Score", min_value=0, max_value=100, format="%d/100"), "Best Landing Page": st.column_config.LinkColumn("Best Landing Page")},
    )

    st.divider()
    st.markdown(f"### Suggested Keywords ({len(rec_kws_df)} Keywords with Real Impressions)")
    if not rec_kws_df.empty:
        c_f1, c_f2 = st.columns([2, 1])
        with c_f1:
            kw_topic_filter = st.multiselect("Filter by Topic:", sorted(rec_kws_df["Topic Cluster"].unique().tolist()), default=sorted(rec_kws_df["Topic Cluster"].unique().tolist()))
        with c_f2:
            min_kw_imp = st.number_input("Min Organic 7d Impressions:", min_value=1, max_value=10000, value=5, step=5)
        scoped_kws = rec_kws_df[(rec_kws_df["Topic Cluster"].isin(kw_topic_filter)) & (rec_kws_df["Organic 7d Impr."] >= min_kw_imp)]
        st.dataframe(scoped_kws, use_container_width=True, hide_index=True, key="tbl_ads_kws", column_config={"Recommended Landing Page": st.column_config.LinkColumn("Recommended Landing Page")})
        st.download_button("Download Suggested Paid Keywords CSV", scoped_kws.to_csv(index=False).encode("utf-8"), file_name="connext_paid_keywords.csv", mime="text/csv", key="dl_ads_kws")

        st.divider()
        st.markdown("### Ad Group Builder (Exact, Phrase, Negatives & RSA Copy)")
        sel_ad_topic = st.selectbox("Select Topic for Ad Group:", ads_topics_df["Trending Topic"].tolist(), key="sel_ad_group")
        ad_row = filtered_trends[filtered_trends["topic"] == sel_ad_topic].iloc[0]
        t_kw_df = rec_kws_df[rec_kws_df["Topic Cluster"] == sel_ad_topic].sort_values("Organic 7d Impr.", ascending=False)
        st.caption(f"Landing Page: **{ad_row['recommended_money_page']}** • Topic Organic Volume: **{ad_row['impressions_7d']:,}** (7d) / **{ad_row['impressions_30d']:,}** (30d)")
        st.dataframe(t_kw_df[["Suggested Keyword", "Organic 7d Impr.", "Organic 30d Impr.", "Daily Avg Impr.", "Growth (vs 2w)", "Current Google Rank", "Match Type"]], use_container_width=True, hide_index=True, key="tbl_adgroup_kws")

        kw_list = t_kw_df["Suggested Keyword"].tolist() if not t_kw_df.empty else ad_row["queries_list"]
        ce, cp, cn = st.columns(3)
        ce.markdown("**[Exact Match]**")
        ce.code("\n".join(f"[{q}]" for q in kw_list), language="text")
        cp.markdown('**"Phrase Match"**')
        cp.code("\n".join(f'"{q}"' for q in kw_list), language="text")
        cn.markdown("**B2B Negatives**")
        cn.code("-jobs\n-careers\n-hiring\n-salary\n-internship\n-glassdoor\n-indeed\n-jobstreet\n-free\n-template", language="text")

        hl, desc = build_rsa_assets(sel_ad_topic)
        ch, cd = st.columns(2)
        ch.markdown("**RSA Headlines (≤ 30 chars)**")
        for cat, h in hl:
            ch.text(f"[{cat}] {h} ({len(h)}/30)")
        cd.markdown("**RSA Descriptions (≤ 90 chars)**")
        for d in desc:
            cd.text(f"{d} ({len(d)}/90)")

# ==============================================================================
# TAB 5: CONTENT PLAN & ANTI-DUPLICATE SITEMAP CHECK
# ==============================================================================
with tab_content:
    st.subheader("1. Update Existing Sitemap Pages (Already Published — Do Not Duplicate)")
    st.dataframe(
        existing_content_df, use_container_width=True, hide_index=True, key="tbl_existing_content",
        column_config={"Existing Sitemap Page": st.column_config.LinkColumn("Existing Sitemap Page"), "Main Service Page": st.column_config.LinkColumn("Main Service Page")},
    )
    st.download_button("Download Existing Page Updates CSV", existing_content_df.to_csv(index=False).encode("utf-8"), file_name="connext_existing_updates.csv", mime="text/csv", key="dl_existing")

    st.divider()
    st.subheader(f"2. Truly New Content Ideas — Verified Not in Sitemap ({len(truly_new_df)})")
    if truly_new_df.empty:
        st.success("Every currently surging topic already has a matching page in connextglobal.com/sitemap.xml!")
    else:
        st.dataframe(
            truly_new_df, use_container_width=True, hide_index=True, key="tbl_truly_new",
            column_config={"Closest Existing Sitemap Page": st.column_config.LinkColumn("Closest Existing Page"), "Service Page to Link To": st.column_config.LinkColumn("Service Page")},
        )
        st.download_button("Download New Content Gaps CSV", truly_new_df.to_csv(index=False).encode("utf-8"), file_name="connext_new_content_gaps.csv", mime="text/csv", key="dl_new_gaps")

    st.divider()
    st.subheader("3. One-Click Writer Brief")
    brief_topic = st.selectbox("Select Topic:", existing_content_df["Trending Topic"].tolist(), key="sel_brief")
    b_row = filtered_trends[filtered_trends["topic"] == brief_topic].iloc[0]
    lt_bullets = "\n".join(f"  - `{q}`" for q in (b_row["long_tail_list"] or b_row["queries_list"]))
    st.markdown(
        f"""### Writer Brief: {b_row['topic'].title()} ({b_row['sector']})
- **Action:** {'Update Existing Page: `' + b_row['closest_existing_url'] + '`' if b_row['already_exists_in_sitemap'] else 'Create New Page (Not in Sitemap)'}
- **Current Google Rank:** **{b_row['google_rank_label']}** | **Main Service Page:** `{b_row['recommended_money_page']}`
- **Organic Volume:** **{b_row['impressions_7d']:,}** (7d) / **{b_row['impressions_30d']:,}** (30d)
- **Exact Rising Searches to Cover (H2s / FAQs):**
{lt_bullets}
"""
    )

# ==============================================================================
# TAB 6: 100% AUTOMATED COMPETITOR SEO (SERPAPI + SITEMAPS)
# ==============================================================================
with tab_competitors:
    st.subheader("Automated Competitor SEO & Growing Queries (12 BPO Rivals)")
    comp_sub1, comp_sub2 = st.tabs([
        "A. Automated Competitor Sitemap & New Content Radar ($0 Free)",
        "B. Automated Competitor Google Ranks (Your SerpApi)",
    ])

    with comp_sub1:
        col_sm_hdr, col_sm_btn = st.columns([3, 1])
        with col_sm_hdr:
            st.markdown("Automatically scans all 12 competitors' `sitemap.xml` feeds and live page `<title>` headlines (**100% free, zero SerpApi credits used**).")
        with col_sm_btn:
            if st.button("Refresh Sitemap Scan Now", key="btn_refresh_comp_sm"):
                scan_competitor_sitemaps_auto.clear()
                st.rerun()

        with st.spinner("Scanning live competitor sitemaps & page titles across all 12 rivals..."):
            comp_sm_df, sm_notes = scan_competitor_sitemaps_auto(lookback_days=45)

        st.caption("Live feed status: " + " • ".join(sm_notes))
        if not comp_sm_df.empty:
            coverage_labels, closest_urls = [], []
            for _, row in comp_sm_df.iterrows():
                chk = check_topic_against_sitemap(f"{row['Slug Topic']} {row['Live Page Headline']}", "", live_set, live_df, live_vec, live_mat)
                coverage_labels.append("Connext Already Covers" if chk["already_exists_in_sitemap"] else "Gap — Connext Missing")
                closest_urls.append(chk["closest_existing_url"])
            comp_sm_df["Connext Coverage"] = coverage_labels
            comp_sm_df["Connext Closest Page"] = closest_urls

            st.dataframe(
                comp_sm_df[["Competitor", "Published / Updated", "Live Page Headline", "Vertical", "Connext Coverage", "Connext Closest Page", "Competitor URL"]],
                use_container_width=True, hide_index=True, key="tbl_comp_sitemaps_auto",
                column_config={"Competitor URL": st.column_config.LinkColumn("Competitor New Page"), "Connext Closest Page": st.column_config.LinkColumn("Connext Closest Page")},
            )

    with comp_sub2:
        if not has_serpapi_secret:
            st.info("Add `[serpapi]` `api_key` in Streamlit Secrets to run live competitor SERP checks using your existing SerpApi plan.")
        else:
            num_kws_to_check = st.slider("Top Surging Keywords to Check:", 5, 25, 15, step=5, key="serp_kw_slider")
            auto_seed_kws = filtered_queries[filtered_queries["is_commercial"]]["query"].head(num_kws_to_check).tolist()
            if st.button(f"Run / Load Weekly SerpApi Competitor Check ({len(auto_seed_kws)} searches, cached 7d)", type="primary", key="btn_run_serpapi_scan"):
                st.session_state["serpapi_scan_ready"] = True
            if st.session_state.get("serpapi_scan_ready"):
                serp_comp_df, _, serp_status = run_serpapi_competitor_and_trends_scan(
                    str(st.secrets["serpapi"]["api_key"]).strip(), tuple(auto_seed_kws), include_google_trends=False
                )
                st.success(serp_status)
                if not serp_comp_df.empty:
                    serp_comp_df["Vertical"] = serp_comp_df["keyword"].apply(classify_sector)
                    serp_comp_df["Connext Live Rank"] = serp_comp_df["connext_live_pos"].apply(human_rank_label)
                    st.dataframe(serp_comp_df, use_container_width=True, hide_index=True, key="tbl_serpapi_ranks", column_config={"comp_url": st.column_config.LinkColumn("Competitor URL")})

# ==============================================================================
# TAB 7: 100% FREE GOOGLE TRENDS & BING SEARCH RADAR ($0 — ZERO CREDITS)
# ==============================================================================
with tab_free_trends_bing:
    st.subheader("Free Google Trends & Bing Search Radar ($0 Extra Cost)")
    st.markdown(
        """<div class="cnxt-cream-panel">
<strong>100% Free Market &amp; Bing Intelligence:</strong> Uses <code>trendspy</code> + Google's official daily RSS feed for free Google Trends curves &amp; rising searches, paired with Microsoft's free Bing Webmaster Tools API &amp; Bing Live Search Suggestions (important for Microsoft Copilot &amp; ChatGPT Search citations).
</div>""",
        unsafe_allow_html=True,
    )

    free_sub1, free_sub2 = st.tabs([
        "A. Free Google Trends Explorer & Rising Searches ($0)",
        "B. Free Bing Search Performance & Live Suggestions ($0)",
    ])

    # --- SUB-TAB A: 100% FREE GOOGLE TRENDS ---
    with free_sub1:
        gt_c1, gt_c2, gt_c3 = st.columns([2.2, 1, 1])
        with gt_c1:
            default_gt_seeds = "medical billing outsourcing, offshore accounting, virtual assistant outsourcing, nearshore software developers, bpo outsourcing"
            gt_seeds_input = st.text_input(
                "Compare up to 5 Topics in Google Trends (comma-separated):",
                value=default_gt_seeds,
                key="free_gt_seeds_input",
            )
        with gt_c2:
            gt_geo = st.selectbox("Country / Region:", ["US", "CA", "AU", "GB", "PH", ""], index=0, key="free_gt_geo")
        with gt_c3:
            gt_timeframe = st.selectbox("Time Window:", ["today 1-m", "today 3-m", "today 12-m", "now 7-d"], index=0, key="free_gt_tf")

        seed_tuple = tuple(s.strip().lower() for s in gt_seeds_input.split(",") if s.strip())[:5]
        if st.button("Refresh Free Google Trends Now", key="btn_refresh_free_gt"):
            fetch_free_google_trends_data.clear()
            st.rerun()

        with st.spinner("Fetching free Google Trends data..."):
            iot_df, free_rising_df, rss_daily_df, gt_notes = fetch_free_google_trends_data(seed_tuple, geo=gt_geo, timeframe=gt_timeframe)

        if gt_notes:
            st.caption(" • ".join(gt_notes))

        if not iot_df.empty:
            st.markdown("### 1. Google Trends Relative Interest Over Time (0–100 Scale)")
            fig_gt = px.line(
                iot_df,
                x="date",
                y="interest_score",
                color="keyword",
                color_discrete_sequence=[CNXT_COLORS["blue"], CNXT_COLORS["sun"], CNXT_COLORS["teal"], CNXT_COLORS["twilight"], CNXT_COLORS["red"]],
                labels={"date": "", "interest_score": "Relative Search Interest (0–100)", "keyword": "Topic"},
            )
            fig_gt.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=45),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor=CNXT_COLORS["line"]),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_gt, use_container_width=True)

        if not free_rising_df.empty:
            st.markdown("### 2. Rising & Breakout Related Searches (Cross-Checked Against Connext Sitemap)")
            sm_stat, sm_link = [], []
            for _, r in free_rising_df.iterrows():
                chk = check_topic_against_sitemap(r["Rising Query"], "", live_set, live_df, live_vec, live_mat)
                sm_stat.append("Already in Sitemap" if chk["already_exists_in_sitemap"] else "New Gap (Not in Sitemap)")
                sm_link.append(chk["best_money_url"] if chk["has_strong_money_match"] else chk["closest_existing_url"])
            free_rising_df["Connext Sitemap Status"] = sm_stat
            free_rising_df["Recommended Connext Page"] = sm_link

            st.dataframe(
                free_rising_df,
                use_container_width=True,
                hide_index=True,
                key="tbl_free_gt_rising",
                column_config={"Recommended Connext Page": st.column_config.LinkColumn("Recommended Connext Page")},
            )
            st.download_button(
                "Download Free Google Trends Rising Queries CSV",
                free_rising_df.to_csv(index=False).encode("utf-8"),
                file_name="connext_free_google_trends_rising.csv",
                mime="text/csv",
                key="dl_free_gt_rising",
            )

        if not rss_daily_df.empty:
            with st.expander(f"3. Today's Macro Trending Searches RSS Feed ({gt_geo or 'Global'}) — {len(rss_daily_df)} Topics", expanded=iot_df.empty):
                st.dataframe(rss_daily_df, use_container_width=True, hide_index=True, key="tbl_free_gt_rss")

    # --- SUB-TAB B: 100% FREE BING WEBMASTER TOOLS & BING LIVE SUGGESTIONS ---
    with free_sub2:
        if has_bing_secret:
            b_site = st.secrets["bing"].get("site_url", "https://connextglobal.com/")
            if st.button("Refresh Bing Webmaster Data", key="btn_refresh_bing"):
                fetch_free_bing_webmaster_queries.clear()
                st.rerun()
            with st.spinner("Loading live Bing Search queries from Bing Webmaster Tools..."):
                bing_df, bing_status = fetch_free_bing_webmaster_queries(
                    str(st.secrets["bing"]["api_key"]), b_site, brand_regex, exclude_local_ph, custom_negatives
                )
            if bing_df.empty:
                st.warning(bing_status)
            else:
                st.success(bing_status)
                bk1, bk2, bk3, bk4 = st.columns(4)
                bk1.metric("Bing Non-Branded Queries", f"{len(bing_df):,}")
                bk2.metric("Recent Bing Impressions", f"{int(bing_df['bing_impr_recent'].sum()):,}")
                bk3.metric("Recent Bing Clicks", f"{int(bing_df['bing_clicks'].sum()):,}")
                bk4.metric("Avg Bing Rank", f"#{bing_df['bing_rank'].mean():.1f}")

                bing_df["Growth"] = bing_df["growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
                st.markdown("### Connext Bing Search Queries & Momentum (Also Powers Copilot / ChatGPT Search)")
                st.dataframe(
                    bing_df[["query", "vertical", "bing_impr_recent", "bing_clicks", "Growth", "bing_rank_label"]],
                    use_container_width=True,
                    hide_index=True,
                    key="tbl_bing_wmt",
                    column_config={
                        "query": "Bing Search Query",
                        "vertical": "Vertical",
                        "bing_impr_recent": st.column_config.NumberColumn("Recent Bing Impr."),
                        "bing_clicks": st.column_config.NumberColumn("Bing Clicks"),
                        "Growth": "Growth vs Prior Period",
                        "bing_rank_label": "Bing Rank",
                    },
                )
                st.download_button(
                    "Download Bing Search Trends CSV",
                    bing_df[["query", "vertical", "bing_impr_recent", "bing_clicks", "Growth", "bing_rank_label"]].to_csv(index=False).encode("utf-8"),
                    file_name="connext_bing_search_trends.csv",
                    mime="text/csv",
                    key="dl_bing_wmt",
                )
        else:
            st.info(
                "**Want Connext's live Bing Search Impressions & Rankings ($0 Free)?**\n"
                "1. Open [bing.com/webmasters](https://www.bing.com/webmasters) → `connextglobal.com` → **Settings (gear icon) → API Access → API Key → Generate**.\n"
                "2. In **Streamlit Cloud → Settings → Secrets**, add:\n"
                "```toml\n[bing]\napi_key = \"your_free_bing_webmaster_api_key\"\nsite_url = \"[https://connextglobal.com/](https://connextglobal.com/)\"\n```"
            )

        st.divider()
        st.markdown("### Live Bing Search Autocomplete & Related Queries Explorer ($0 — No API Key Required)")
        bing_seed_input = st.text_input(
            "Enter B2B Seed Terms to Pull Live Bing Search Suggestions (comma-separated):",
            value="medical billing outsourcing, offshore accounting, virtual assistant, nearshore developers, bpo company",
            key="free_bing_sugg_input",
        )
        bing_seed_tuple = tuple(s.strip() for s in bing_seed_input.split(",") if s.strip())[:8]
        bing_sugg_df = fetch_free_bing_suggestions(bing_seed_tuple)
        if not bing_sugg_df.empty:
            b_stat, b_link = [], []
            for _, br in bing_sugg_df.iterrows():
                chk = check_topic_against_sitemap(br["Bing Live Search Suggestion"], "", live_set, live_df, live_vec, live_mat)
                b_stat.append("Already in Sitemap" if chk["already_exists_in_sitemap"] else "New Gap (Not in Sitemap)")
                b_link.append(chk["best_money_url"] if chk["has_strong_money_match"] else chk["closest_existing_url"])
            bing_sugg_df["Connext Sitemap Status"] = b_stat
            bing_sugg_df["Recommended Connext Page"] = b_link

            st.dataframe(
                bing_sugg_df,
                use_container_width=True,
                hide_index=True,
                key="tbl_bing_suggestions",
                column_config={"Recommended Connext Page": st.column_config.LinkColumn("Recommended Connext Page")},
            )
            st.download_button(
                "Download Bing Live Search Suggestions CSV",
                bing_sugg_df.to_csv(index=False).encode("utf-8"),
                file_name="connext_bing_live_suggestions.csv",
                mime="text/csv",
                key="dl_bing_suggestions",
            )

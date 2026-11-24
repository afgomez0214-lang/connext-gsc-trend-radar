import base64
import hmac
import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from datetime import datetime, timedelta
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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Connext Global | Google Search & AI Trend Radar",
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
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid var(--cnxt-line); }
.stTabs [data-baseweb="tab"] { font-family: var(--font-display); font-weight: 600; color: var(--cnxt-midnight); background-color: var(--cnxt-cream); border-radius: 6px 6px 0 0; padding: 10px 18px; border: 1px solid var(--cnxt-line); border-bottom: none; }
.stTabs [aria-selected="true"] { background-color: var(--cnxt-twilight) !important; color: var(--cnxt-paper) !important; border-color: var(--cnxt-twilight) !important; }
.cnxt-cream-panel { background-color: var(--cnxt-cream); border: 1px solid var(--cnxt-line-strong); border-left: 4px solid var(--cnxt-sun); border-radius: 6px; padding: 14px 18px; margin: 10px 0; color: var(--cnxt-midnight); font-size: 0.93rem; line-height: 1.55; }
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
            """<div class="cnxt-hero-banner" style="margin-top: 14px; padding: 24px 28px;">
<div class="cnxt-eyebrow">INTERNAL TEAM ACCESS</div>
<h2 style="color:#FFFFFF !important; margin:0 0 6px 0; font-size:1.5rem;">Connext Search &amp; AI <span style="color:#E58B25;">Trend Radar</span></h2>
<p class="cnxt-hero-sub" style="font-size:0.88rem;">Sign in with your Connext team password to view live Google search demand, existing sitemap coverage, and recommended Google Ads keywords.</p>
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
# CONNEXT GLOBAL LIVE SITE ARCHITECTURE (FALLBACK + LIVE SITEMAP.XML SYNC)
# ==============================================================================
CORE_LIVE_PATHS = [
    "/",
    "/about-us",
    "/acceptable-use-policy",
    "/bpo-glossary",
    "/careers",
    "/compare-by-model",
    "/connext-in-the-news",
    "/connext-pricing",
    "/contact",
    "/general-outsourcing",
    "/general-outsourcing/do-what-you-do-best-and-outsource-the-rest",
    "/general-outsourcing/offshore-captive-model",
    "/hipaa-compliant-virtual-medical-assistants",
    "/insights",
    "/insights/faq",
    "/our-differentiator",
    "/our-service-model",
    "/partnerships",
    "/privacy-policy",
    "/roles-we-fill",
    "/search",
    "/sitemap",
    "/terms-conditions",
    "/thank-you",
    "/thank-you-consultation",
    "/thank-you-content",
    "/trust-compliance",
    # --- Industries (26) ---
    "/industries/construction",
    "/industries/education",
    "/industries/engineering",
    "/industries/engineering/engineering-outsourcing-solutions",
    "/industries/financial-services",
    "/industries/fintech",
    "/industries/healthcare",
    "/industries/healthcare/dental-billing-outsourcing",
    "/industries/healthcare/healthcare-software",
    "/industries/healthcare/medical-billing",
    "/industries/healthcare/medical-coding",
    "/industries/healthcare/outsource-claims-management",
    "/industries/healthcare/outsource-credentialing",
    "/industries/healthcare/outsource-prior-authorization",
    "/industries/healthcare/remote-monitoring",
    "/industries/healthcare/revenue-cycle-management",
    "/industries/healthcare/virtual-nursing",
    "/industries/hospitality",
    "/industries/human-resources",
    "/industries/insurance",
    "/industries/insurance/insurance-outsourcing",
    "/industries/legal",
    "/industries/private-equity",
    "/industries/real-estate",
    "/industries/real-estate/mortgage",
    "/industries/real-estate/title-escrow",
    # --- Teams (23) ---
    "/teams/ai-staffing",
    "/teams/back-office/payroll-outsourcing-solutions",
    "/teams/business-operation-support",
    "/teams/customer-support",
    "/teams/customer-support/call-center-outsourcing",
    "/teams/customer-support/customer-success",
    "/teams/customer-support/technical-support-outsourcing",
    "/teams/customer-support/virtual-assistants",
    "/teams/customer-support/zendesk-staffing",
    "/teams/finance-and-accounting-outsourcing-services",
    "/teams/financial-services/accounting",
    "/teams/financial-services/accounts-softwares",
    "/teams/financial-services/debt-collection-outsourcing",
    "/teams/financial-services/hire-certified-public-accountants-with-connext",
    "/teams/financial-services/hire-expert-controllers-with-connext",
    "/teams/it-development",
    "/teams/it-development/machine-learing",
    "/teams/it-development/python-staffing",
    "/teams/it-development/servicenow-staffing",
    "/teams/marketing",
    "/teams/marketing/e-commerce-outsourcing",
    "/teams/marketing/outsource-digital-marketing",
    "/teams/operation-support/payroll-outsourcing",
    # --- Services (8) ---
    "/services/co-management",
    "/services/compliance",
    "/services/contractor-transitions",
    "/services/employer-of-record",
    "/services/office-space",
    "/services/process-it-support",
    "/services/recruiting",
    "/services/retention",
    # --- Locations (6) ---
    "/locations",
    "/locations/outsource-to-colombia",
    "/locations/outsource-to-india",
    "/locations/outsource-to-mexico",
    "/locations/outsource-to-the-philippines",
    "/locations/outsource-to-united-states",
    # --- Tools (8) ---
    "/tools",
    "/tools/co-management-fit",
    "/tools/country-explorer",
    "/tools/hiring-timeline-estimator",
    "/tools/pricing-calculator",
    "/tools/roi-calculator",
    "/tools/salary-benchmark",
    "/tools/team-readiness",
    # --- E-Books (5) ---
    "/e-books/asia-pacific-is-redefining-bpo-ebook",
    "/e-books/bpo-trends-2025-ebook",
    "/e-books/outsourcing-with-connext-ebook",
    "/e-books/redefining-growth-through-global-talent",
    "/e-books/your-first-offshore-team-guide",
    # --- Case Studies (42) ---
    "/case-studies/augmenting-a-revenue-cycle-management-companys-workforce-with-medical-billing-specialists-in-the-philippines",
    "/case-studies/behavioral-health-rcm-case-study",
    "/case-studies/boosting-sales-pipeline-and-revenue-for-a-debt-relief-services-provider",
    "/case-studies/building-a-cross-functional-support-team-for-a-pos-software-company-with-connext",
    "/case-studies/building-a-custom-offshore-accounting-team-for-a-retail-grocery-chain",
    "/case-studies/building-a-dedicated-offshore-credentialing-team-to-support-business-growth",
    "/case-studies/building-a-multilingual-patient-scheduling-team-for-a-vision-clinic-holding-company",
    "/case-studies/building-a-responsive-call-center-and-patient-engagement-team-for-a-leading-eye-care-provider",
    "/case-studies/building-a-scalable-operations-team-for-a-public-biotechnology-company",
    "/case-studies/building-custom-support-teams-for-an-it-managed-service-provider-holding-company",
    "/case-studies/building-industry-best-property-management-service-offshore",
    "/case-studies/cost-and-process-improvement-for-title-and-escrow",
    "/case-studies/creating-a-24-7-customer-support-team-for-venture-backed-e-commerce-startup",
    "/case-studies/creating-a-collaborative-and-people-first-outsourcing-experience-for-a-u-s-based-it-services-company",
    "/case-studies/creating-a-cross-functional-support-team-for-a-regional-bank",
    "/case-studies/developing-a-central-scheduling-system-for-a-nationwide-healthcare-system",
    "/case-studies/direct-booking-solution-for-an-entertainment-firm",
    "/case-studies/empowering-a-u-s-based-university-with-scalable-operational-support-and-a-trusted-offshore-team",
    "/case-studies/hawaii-title-and-escrow-services",
    "/case-studies/how-a-global-fintech-platform-scaled-from-10-to-100-offshore-professionals-in-under-a-year",
    "/case-studies/how-connext-global-solutions-unlocks-client-growth-by-providing-accounting-support",
    "/case-studies/it-offshore-service-desk-case-study",
    "/case-studies/it-staff-augmentation-for-payment-services-company",
    "/case-studies/multi-skilled-associate-integration-for-a-pharmacy-group",
    "/case-studies/offshore-back-office-support-food-distributors",
    "/case-studies/ofthalmology-group",
    "/case-studies/process-automation-financial-services",
    "/case-studies/recruitment-services-for-expense-management-software-company",
    "/case-studies/scaling-a-fintech-innovator-with-offshore-operations-support",
    "/case-studies/scaling-a-leading-multifamily-flooring-company-with-offshore-accounting-support",
    "/case-studies/scaling-a-records-retrieval-company-with-a-custom-offshore-team",
    "/case-studies/scaling-a-us-based-mortgage-team-with-connext",
    "/case-studies/scaling-an-ai-driven-digital-health-startup-with-a-hybrid-human-and-ai-operations-model",
    "/case-studies/scaling-operational-support-for-a-global-investment-management-firm",
    "/case-studies/scaling-patient-scheduling-and-support-for-a-growing-gastroenterology-practice",
    "/case-studies/scaling-portfolio-companies",
    "/case-studies/staff-augmentation-for-accounting-management",
    "/case-studies/streamlining-business-processes-with-a-dedicated-offshore-team",
    "/case-studies/strengthening-back-office-operations-for-a-growing-health-benefits-platform",
    "/case-studies/strengthening-healthcare-operations-with-scalable-offshore-teams",
    "/case-studies/supporting-healthcare-automation-with-scalable-staffing-for-a-healthtech-startup",
    "/case-studies/transitioning-a-dental-software-company-to-connext-for-efficiency",
    # --- One-Pagers (25) ---
    "/one-pager/5-signs-hawaii-team-needs-broader-talent-strategy",
    "/one-pager/ai-alone-is-not-enough-to-fix-your-payroll-this-will",
    "/one-pager/brand-voice-governance-outsourced-hospitality-teams",
    "/one-pager/build-high-performing-teams",
    "/one-pager/build-offshore-teams-that-work-like-onshore",
    "/one-pager/build-operate-transfer-offshore-healthcare-back-office",
    "/one-pager/comparing-four-major-offshore-workforce-models",
    "/one-pager/cx-operations-infrastructure-reliability",
    "/one-pager/data-residency-vs-data-access-security-reviewer-map",
    "/one-pager/deal-research-desk-raci-offshore-analyst",
    "/one-pager/eor-only-vs-eor-plus-co-management",
    "/one-pager/finance-accounting-team-real-results",
    "/one-pager/human-in-the-loop-ai-customer-service-guide",
    "/one-pager/offshore-claims-support-co-managed-model",
    "/one-pager/offshore-clinical-support-roles-guide",
    "/one-pager/offshore-staffing-contract-checklist",
    "/one-pager/one-standard-for-every-offshore-partner-you-keep",
    "/one-pager/reg-sp-service-provider-oversight-checklist-offshore-ria-teams",
    "/one-pager/revenue-cycle-offshore-team-guide",
    "/one-pager/shrinkage-vs-attrition-bpo-metric",
    "/one-pager/signed-agreement-to-operational-team-90-days",
    "/one-pager/staff-augmentation-that-puts-you-in-control",
    "/one-pager/the-connext-take-on-global-capability-centers",
    "/one-pager/why-colombia-belongs-at-top-of-nearshore-shortlist",
    "/one-pager/why-teams-fail-with-outsourcing-co-management-guide",
    # --- White Papers (61) ---
    "/white-paper/5-things-impacting-the-rise-of-business-processes-outsourcing-in-healthcare",
    "/white-paper/accountants-as-business-translators",
    "/white-paper/accounting-shortage-in-the-united-states-addressing-the-gap-through-offshore-staffing",
    "/white-paper/adopting-a-digital-transformation-mindset",
    "/white-paper/ai-enabled-worker-classification-compliance-global-hiring",
    "/white-paper/ai-human-solutions",
    "/white-paper/an-introduction-to-co-management",
    "/white-paper/bpo-and-the-evolution-of-the-modern-back-office",
    "/white-paper/building-ai-augmented-operations",
    "/white-paper/business-process-outsourcing-and-the-philippines",
    "/white-paper/cfo-guide-rcm-fa-staffing-decisions",
    "/white-paper/client-engagement-using-real-time-video-the-new-normal",
    "/white-paper/confidence-gap-survey-2025-report",
    "/white-paper/connext-recruiting-vs-traditional-headhunters-us",
    "/white-paper/creating-a-positive-spiral-in-your-business-a-case-for-outsourcing",
    "/white-paper/enterprise-leaders-guide-large-scale-offshore-deployment",
    "/white-paper/fintech-leaders-guide-secure-offshore-staffing",
    "/white-paper/guide-a-practical-guide-for-healthcare-finance-leaders",
    "/white-paper/guide-to-offshore-outsourcing-for-it-managed-service-providers",
    "/white-paper/guide-to-outsourcing-for-small-businesses",
    "/white-paper/healthcare-ai-customer-support-deployment-playbook",
    "/white-paper/healthcare-executives-guide-build-operate-transfer-rcm",
    "/white-paper/healthcare-leaders-guide-secure-visible-offshore-rcm",
    "/white-paper/how-india-powers-the-future-of-work",
    "/white-paper/ifrs-18-readiness-2027-finance-leaders-guide",
    "/white-paper/insurance-claims-outsourcing-governance-guide",
    "/white-paper/leveraging-offshore-outsourcing-to-augment-ai-and-automation-solutions",
    "/white-paper/loan-automation",
    "/white-paper/looking-for-specialized-talent-start-with-india",
    "/white-paper/managing-foreign-exchange-risk-offshore-staffing",
    "/white-paper/maximizing-healthcare-operations-the-colombian-advantage",
    "/white-paper/nearshore-delivery-latin-america-location-talent-control",
    "/white-paper/offshore-accounting-job-roles-tasks-and-salary-data",
    "/white-paper/offshore-growth-toolkit",
    "/white-paper/offshore-staffing-healthcare-finance-whitepaper",
    "/white-paper/offshore-staffing-rcm-companies-whitepaper",
    "/white-paper/portfolio-offshore-operating-standard-private-equity",
    "/white-paper/practical-guide-outsourced-accounting-us-companies",
    "/white-paper/rebadging-done-right-the-people-first-approach-to-transitioning-teams",
    "/white-paper/scale-offshore-teams-without-losing-control",
    "/white-paper/scaling-offshore-healthcare-operations",
    "/white-paper/staff-augmentation-growth-operational-agility",
    "/white-paper/startup-offshore-talent-strategy",
    "/white-paper/strategic-workforce-solutions",
    "/white-paper/technology-outsourcing-pricing-models-contracts",
    "/white-paper/the-2026-rcm-staffing-benchmark-report",
    "/white-paper/the-2026-state-of-customer-service-operations",
    "/white-paper/the-bpo-switch-framework",
    "/white-paper/the-evolution-of-outsourcing-in-eastern-europe-from-call-centers-to-strategic-hubs",
    "/white-paper/the-future-of-ai-in-colombia-what-u-s-businesses-need-to-know",
    "/white-paper/the-global-hiring-playbook",
    "/white-paper/the-offshore-staffing-rfp-kit",
    "/white-paper/the-role-of-bpos-in-business-continuity",
    "/white-paper/the-true-cost-of-onshore-vs-offshore-talent",
    "/white-paper/three-pillars-of-managing-a-remote-workforce",
    "/white-paper/title-and-escrow-operationsyour-path-to-automation-is-through-the-philippines",
    "/white-paper/top-10-ways-offshoring-is-helping-healthcare-providers-address-workforce-shortages",
    "/white-paper/top-5-mistakes-when-choosing-a-bpo-partner",
    "/white-paper/understanding-work-setup-preferences-of-offshore-accounting-staff",
    "/white-paper/what-to-look-for-in-a-business-process-outsourcing-provider",
    "/white-paper/why-ai-human-teams-outperform-traditional-bpo",
]

BLOG_PATHS_ZLIB = (
    "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"
)

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

# Optional filter for local PH city directory & student/school searches
LOCAL_PH_AND_STUDENT_REGEX = re.compile(
    r"\b("
    r"school|schools|university|universities|college|colleges|student|students|tuition|courses|admission|strand|"
    r"baguio|clark|bgc|bonifacio global city|makati|ortigas|quezon city|alabang|iloilo|davao|cebu|pampanga|taguig|"
    r"list of bpo|top bpo companies in|call center companies in"
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

# Conversational / AI Mode Prompt Signals (7+ words or question/comparison prompts)
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

# Connext Global Real Vertical & Team Taxonomy (matches both queries and URL paths)
SECTOR_PATTERNS = [
    (
        "Healthcare & RCM",
        re.compile(
            r"(/industries/healthcare|hipaa-compliant-virtual-medical|\b(medical|rcm|revenue cycle|healthcare|health care|billing|coding|icd|cpt|patient|"
            r"prior auth|authorization|credentialing|clinic|hospital|dental|hipaa|ehr|emr|pharmacy|nursing|remote monitoring)\b)",
            re.I,
        ),
    ),
    (
        "Finance, Accounting & Tax",
        re.compile(
            r"(/teams/finance|/teams/financial-services|payroll-outsourcing|\b(accounting|accountant|bookkeeping|bookkeeper|accounts payable|accounts receivable|"
            r"cpa|controller|cfo|tax|audit|auditing|payroll|finance|financial|reconciliation|debt collection|ap|ar)\b)",
            re.I,
        ),
    ),
    (
        "Tech, IT & AI Staffing",
        re.compile(
            r"(/teams/it-development|/teams/ai-staffing|\b(software|developer|developers|engineer|engineering|it support|helpdesk|help desk|"
            r"qa|quality assurance|testing|data|cloud|devops|cybersecurity|tech|programmer|python|servicenow|machine learning|ai staffing|ai)\b)",
            re.I,
        ),
    ),
    (
        "Customer Experience (CX) & VAs",
        re.compile(
            r"(/teams/customer-support|\b(customer service|customer care|customer support|client support|client success|"
            r"call center|contact center|omnichannel|live chat|tier 1|tier 2|technical support|zendesk|virtual assistant)\b)",
            re.I,
        ),
    ),
    (
        "Back Office & Operations Support",
        re.compile(
            r"(/teams/business-operation-support|/teams/back-office|\b(back office|back-office|admin|administrative|data entry|data conversion|document processing|"
            r"order processing|catalog|content moderation|transcription|indexing|operations support|shared services|procurement)\b)",
            re.I,
        ),
    ),
    (
        "Real Estate, Mortgage & Title/Escrow",
        re.compile(
            r"(/industries/real-estate|\b(real estate|property management|mortgage|title|escrow|leasing|appraisal|loan processing)\b)",
            re.I,
        ),
    ),
    (
        "Insurance & Claims",
        re.compile(
            r"(/industries/insurance|\b(insurance|underwriting|claims|policy|actuarial|adjuster)\b)",
            re.I,
        ),
    ),
    (
        "Legal & Compliance",
        re.compile(
            r"(/industries/legal|\b(legal|paralegal|attorney|law firm|contract|compliance|regulatory|litigation)\b)",
            re.I,
        ),
    ),
    (
        "Fintech & Private Equity",
        re.compile(
            r"(/industries/fintech|/industries/private-equity|\b(fintech|private equity|portfolio company|portfolio companies|investment|wealth management|payment)\b)",
            re.I,
        ),
    ),
    (
        "Engineering & Construction",
        re.compile(
            r"(/industries/engineering|/industries/construction|\b(civil engineering|structural|cad|bim|construction|estimating|architectural|architecture)\b)",
            re.I,
        ),
    ),
    (
        "Sales, Marketing & E-Commerce",
        re.compile(
            r"(/teams/marketing|\b(sdr|bdr|sales development|lead generation|marketing|digital marketing|seo|paid media|ppc|graphic design|"
            r"video editing|copywriting|social media|hubspot|salesforce|crm|ecommerce|e-commerce)\b)",
            re.I,
        ),
    ),
    (
        "Hospitality, Education & HR",
        re.compile(
            r"(/industries/hospitality|/industries/education|/industries/human-resources|\b(hospitality|hotel|travel|education|university|higher ed|human resources|hr outsourcing|recruiting)\b)",
            re.I,
        ),
    ),
]

# Direct topic-to-money-page boost rules for Connext's verified live Webflow URLs
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
    """Normalizes full URLs to canonical https://connextglobal.com/... format."""
    u = str(url).strip().lower()
    if u.startswith("/"):
        u = f"https://connextglobal.com{u}"
    u = u.replace("http://", "https://").replace("https://www.", "https://")
    u = u.split("?")[0].split("#")[0]
    if u.endswith("/") and len(u) > len("https://connextglobal.com/"):
        u = u.rstrip("/")
    return u


def url_to_path(url: str) -> str:
    """Extracts clean path (e.g., '/industries/healthcare') from a normalized URL."""
    norm = normalize_url(url)
    path = urlparse(norm).path
    if not path:
        return "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path


def classify_page_role(path: str, live_paths_set: set[str]) -> str:
    """Classifies a path into Connext Global's real Webflow site structure."""
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
    """Converts numeric Google position into plain English for non-SEO readers."""
    if pos <= 3.0:
        return f"Top 3 on Google (#{pos:.1f})"
    if pos <= 10.0:
        return f"Page 1 (#{pos:.1f})"
    if pos <= 20.0:
        return f"Page 2 (#{pos:.1f})"
    return f"Page 3+ (#{pos:.1f})"


def _get_embedded_paths() -> list[str]:
    blog_paths = zlib.decompress(base64.b64decode(BLOG_PATHS_ZLIB)).decode("utf-8").split("\n")
    return sorted(set(CORE_LIVE_PATHS) | set(blog_paths))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_sitemap_paths() -> tuple[list[str], str]:
    """
    Fetches live URLs directly from https://connextglobal.com/sitemap.xml
    (with fallback to https://connextglobal.com/sitemap and the embedded 1,301-page snapshot).
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ConnextTrendRadar/1.0)"}

    # 1. Try https://connextglobal.com/sitemap.xml
    try:
        req = urllib.request.Request(SITEMAP_XML_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        locs = [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text]

        # Handle sitemapindex if child sitemaps exist
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
            return paths, f"Live Sitemap.xml ({len(paths):,} URLs)"
    except Exception:
        pass

    # 2. Fallback: try https://connextglobal.com/sitemap HTML page
    try:
        req = urllib.request.Request(SITEMAP_HTML_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
        hrefs = re.findall(r'href=["\']((?:https?://(?:www\.)?connextglobal\.com)?/[^"\'#?]*)["\']', html_text, re.I)
        paths = sorted({url_to_path(h) for h in hrefs})
        if len(paths) >= 50:
            return paths, f"Live /sitemap Page ({len(paths):,} URLs)"
    except Exception:
        pass

    # 3. Fallback: embedded Nov 20, 2026 verified 1,301-page sitemap snapshot
    embedded = _get_embedded_paths()
    return embedded, f"Verified Sitemap Snapshot ({len(embedded):,} URLs)"


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


def parse_uploaded_live_pages(uploaded_file) -> list[str]:
    """Parses an optional user-uploaded Excel/CSV of live connextglobal.com URLs."""
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, header=None)
    else:
        df = pd.read_csv(uploaded_file, header=None)
    urls = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    paths = sorted({url_to_path(u) for u in urls if "connextglobal.com" in u.lower() or u.startswith("/")})
    return paths


def check_topic_against_sitemap(
    topic: str,
    queries_sample: str,
    live_set: set[str],
    live_df: pd.DataFrame,
    vec: TfidfVectorizer,
    mat,
) -> dict:
    """
    Checks a trending topic against ALL pages in connextglobal.com/sitemap.xml to:
      1. Identify the best existing commercial Service/Industry page.
      2. Detect whether an article/guide covering this topic ALREADY exists in the sitemap
         (preventing duplicate content suggestions!).
    """
    combined_text = f"{topic} {queries_sample}"
    q_vec = vec.transform([combined_text])
    sims = cosine_similarity(q_vec, mat).flatten()

    scored = live_df.copy()
    scored["raw_sim"] = sims
    scored["sim"] = sims

    # Also compute specific keyword overlap (ignoring generic words like 'outsourcing', 'bpo', 'services')
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
            scored.loc[scored["path"] == target_path, "sim"] += 0.45
            break

    money_df = scored[scored["is_money_page"]].sort_values("sim", ascending=False)
    content_df = scored[scored["is_content_page"]].sort_values(["raw_sim", "token_overlap"], ascending=False)

    best_money_row = money_df.iloc[0]
    best_money_url = best_money_row["full_url"]
    best_money_role = best_money_row["role"]
    best_money_score = float(best_money_row["sim"])

    # Check if ANY existing page in sitemap.xml already covers this exact topic
    existing_matches = scored[
        (scored["path"] != "/")
        & ((scored["raw_sim"] >= 0.28) | (scored["token_overlap"] >= 0.67))
    ].sort_values(["token_overlap", "raw_sim"], ascending=False)

    already_exists_in_sitemap = len(existing_matches) > 0
    closest_existing_url = existing_matches.iloc[0]["full_url"] if already_exists_in_sitemap else (
        content_df.iloc[0]["full_url"] if not content_df.empty else best_money_url
    )
    closest_existing_role = existing_matches.iloc[0]["role"] if already_exists_in_sitemap else (
        content_df.iloc[0]["role"] if not content_df.empty else best_money_role
    )
    closest_overlap_pct = int(round(
        max(
            float(existing_matches.iloc[0]["raw_sim"]) if already_exists_in_sitemap else float(content_df.iloc[0]["raw_sim"]),
            float(existing_matches.iloc[0]["token_overlap"]) if already_exists_in_sitemap else float(content_df.iloc[0]["token_overlap"]),
        ) * 100
    ))

    top_money_pages = money_df.head(3)[["full_url", "role", "sim"]].to_dict("records")
    top_content_pages = content_df[content_df["raw_sim"] >= 0.08].head(5)[["full_url", "role", "raw_sim"]].to_dict("records")

    return {
        "best_money_url": best_money_url,
        "best_money_role": best_money_role,
        "best_money_score": round(best_money_score, 2),
        "has_strong_money_match": bool(boosted_path is not None or best_money_score >= 0.22),
        "already_exists_in_sitemap": already_exists_in_sitemap,
        "closest_existing_url": closest_existing_url,
        "closest_existing_role": closest_existing_role,
        "closest_overlap_pct": min(100, closest_overlap_pct),
        "existing_sitemap_matches": existing_matches.head(3)[["full_url", "role"]].to_dict("records"),
        "top_money_pages": top_money_pages,
        "top_content_pages": top_content_pages,
    }


def classify_sector(text: str) -> str:
    for sector_name, pattern in SECTOR_PATTERNS:
        if pattern.search(text):
            return sector_name
    return "General B2B Outsourcing & Staffing"


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


# --- PARSE GSC CSV OR ZIP EXPORTS (WEB & GENERATIVE AI) ---
def _read_csv_or_zip(uploaded_file, preferred_names: list[str]) -> pd.DataFrame:
    """Reads a CSV directly or extracts the matching CSV inside a Google Search Console .zip export."""
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
    """Parses a GSC Web Search Analytics CSV or ZIP export with exact column validation."""
    df = _read_csv_or_zip(uploaded_file, preferred_names=["quer", "page", "chart", "date"])
    df.columns = [c.lower().strip() for c in df.columns]

    rename_map = {}
    if "top queries" in df.columns and "query" not in df.columns:
        rename_map["top queries"] = "query"
    if "top pages" in df.columns and "page" not in df.columns:
        rename_map["top pages"] = "page"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"date", "query", "page", "impressions", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Uploaded GSC file is missing columns: {sorted(missing)}. "
            "Export with Date + Query + Page dimensions (e.g., via Looker Studio or Search Analytics for Sheets)."
        )

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
    excluded_count = before_len - len(df)
    df["is_branded"] = df["query"].str.contains(brand_regex, regex=True, na=False) if brand_regex else False
    return df, excluded_count


def parse_gsc_ai_report_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Parses CSV or ZIP exports from Google Search Console's 'Generative AI' performance report."""
    df = _read_csv_or_zip(uploaded_file, preferred_names=["page", "date", "chart"])
    orig_cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in orig_cols}

    page_col = None
    for candidate in ["top pages", "page", "pages", "url", "landing page"]:
        if candidate in lower_map:
            page_col = lower_map[candidate]
            break
    if page_col is None:
        raise ValueError("Could not find a 'Page' or 'Top pages' column in the Generative AI export.")

    if "date" in lower_map:
        date_col = lower_map["date"]
        imp_col = lower_map.get("impressions") or lower_map.get("ai impressions")
        if imp_col is None:
            num_cols = [c for c in orig_cols if pd.api.types.is_numeric_dtype(df[c])]
            imp_col = num_cols[0] if num_cols else None
        if imp_col is None:
            raise ValueError("Could not find an Impressions column in the Generative AI export.")

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

    raise ValueError("Unrecognized GSC Generative AI format. Export the 'Pages' table from GSC -> Performance -> Generative AI.")


# --- GSC API FETCHING (LIVE DATA — IMPRESSIONS & POSITION ONLY) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gsc_api_data(
    creds_dict: dict,
    site_url: str,
    lookback_days: int,
    brand_regex: str,
    custom_exclude_regex: str,
    exclude_local_ph: bool,
) -> tuple[pd.DataFrame, str, int, int]:
    """Pulls live daily query+page impressions & organic position from Google Search Console."""
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
    )
    service = build("searchconsole", "v1", credentials=creds)

    end_dt = datetime.utcnow().date()
    start_dt = end_dt - timedelta(days=lookback_days)

    rows = []
    raw_total_count = 0
    excluded_count = 0
    start_row = 0
    row_limit = 25000
    max_rows = 250000
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

            raw_total_count += 1
            if STRICT_NON_INDUSTRY_REGEX.search(clean_q):
                excluded_count += 1
                continue
            if exclude_local_ph and LOCAL_PH_AND_STUDENT_REGEX.search(clean_q):
                excluded_count += 1
                continue
            if custom_pattern and custom_pattern.search(clean_q):
                excluded_count += 1
                continue

            is_brand = bool(brand_pattern.search(clean_q)) if brand_pattern else False
            rows.append({
                "date": date_str,
                "query": clean_q,
                "page": normalize_url(page),
                "impressions": int(r["impressions"]),
                "position": float(r["position"]),
                "is_branded": is_brand,
            })

        if len(batch) < row_limit:
            break
        start_row += row_limit

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df, str(first_incomplete), excluded_count, raw_total_count


# --- CLUSTERING & MULTI-HORIZON TREND SCORING (ZERO-VECTOR SAFE) ---
@st.cache_data(show_spinner=False)
def cluster_queries(df: pd.DataFrame, distance_threshold: float) -> pd.DataFrame:
    """Clusters top non-branded B2B queries into semantic topic groups safely (handles zero-norm vectors)."""
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
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
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
    full_df: pd.DataFrame,
    live_set: set[str],
    live_df: pd.DataFrame,
    live_vec: TfidfVectorizer,
    live_mat,
    min_daily_imp: int,
    min_up_days: int,
    min_norm_slope: float,
    min_lift: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyzes impressions, Connext Google rank, sitemap coverage, and query-level metrics."""
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
        .agg(
            impressions=("impressions", "sum"),
            weighted_pos=("weighted_pos", "sum"),
        )
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
        .agg(
            page_imp_7d=("impressions", "sum"),
            page_weighted_pos=("weighted_pos", "sum"),
            queries_on_page=("query", "nunique"),
        )
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
        tot_imp_30d = int(group[group["date"].isin(month_dates)]["impressions"].sum())

        avg_pos_7d = float(rec_grp["weighted_pos"].sum() / max(tot_imp_7d, 1))
        pri_grp = prior_df[prior_df["topic"] == topic]
        if not pri_grp.empty and pri_grp["impressions"].sum() > 0:
            avg_pos_prior = float(pri_grp["weighted_pos"].sum() / pri_grp["impressions"].sum())
            pos_delta = round(avg_pos_prior - avg_pos_7d, 1)
        else:
            pos_delta = 0.0

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

        # Plain-English Primary Team Action
        if has_commercial and avg_pos_7d > 5.0:
            team_action = "Launch Paid Ads"
        elif site_check["already_exists_in_sitemap"] or (4.0 <= avg_pos_7d <= 15.0):
            team_action = "Update Existing Page"
        elif not site_check["already_exists_in_sitemap"]:
            team_action = "Write New Page (Not in Sitemap)"
        else:
            team_action = "Defend Top 3 Rank"

        growth_pct_vs_base = round((lift_today - 1.0) * 100)
        month_growth_pct = round((lift_month - 1.0) * 100)

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
            "team_action": team_action,
            "velocity_score": velocity_score,
            "ads_fit_score": ads_fit_score,
            "up_days_of_6": up_days_week,
            "daily_growth_%": round(norm_slope_week * 100, 1),
            "today_lift_x": round(lift_today, 2),
            "growth_pct_3d": growth_pct_vs_base,
            "month_lift_x": round(lift_month, 2),
            "month_growth_pct": month_growth_pct,
            "impressions_7d": tot_imp_7d,
            "impressions_30d": tot_imp_30d,
            "daily_avg_7d": round(tot_imp_7d / 7.0, 1),
            "sparkline_7d": [int(v) for v in arr_week],
            "connext_org_pos": round(avg_pos_7d, 1),
            "google_rank_label": human_rank_label(avg_pos_7d),
            "rank_change": pos_delta,
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

    # Query-Level Metrics (with BOTH 7-day and 30-day real GSC impressions!)
    q_recent = (
        recent_df.groupby(["topic", "query", "is_long_tail", "is_ai_prompt", "word_count"], as_index=False)
        .agg(
            imp_7d=("impressions", "sum"),
            weighted_pos=("weighted_pos", "sum"),
            top_page=("norm_page", "first"),
            top_path=("path", "first"),
        )
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


# --- AI SEARCH (VERIFIED GSC GENERATIVE AI REPORT + REAL WEB PROMPTS) ---
def analyze_ai_search_verified(
    ai_df: pd.DataFrame,
    ai_mode: str,
    non_brand_web_df: pd.DataFrame,
    all_queries_df: pd.DataFrame,
    live_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strictly uses verified data without inventing AI impression counts."""
    if non_brand_web_df.empty or all_queries_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    web = non_brand_web_df.copy()
    web["norm_page"] = web["page"].apply(normalize_url)
    web["weighted_pos"] = web["position"] * web["impressions"]

    all_dates = sorted(web["date"].unique())
    week_dates = all_dates[-7:]

    web_7d = (
        web[web["date"].isin(week_dates)]
        .groupby("norm_page", as_index=False)
        .agg(
            display_url=("norm_page", "first"),
            web_imp_7d=("impressions", "sum"),
            web_wpos=("weighted_pos", "sum"),
        )
    )
    web_7d["web_org_pos"] = (web_7d["web_wpos"] / np.maximum(web_7d["web_imp_7d"], 1)).round(1)

    page_queries_grp = all_queries_df.sort_values(["imp_7d", "query_lift_x"], ascending=[False, False])
    connected_qs_map = {
        p: [f"{r['query']} ({r['imp_7d']:,} impr., #{r['connext_org_pos']})" for _, r in g.head(6).iterrows()]
        for p, g in page_queries_grp.groupby("top_page")
    }
    raw_qs_map = page_queries_grp.groupby("top_page")["query"].apply(list).to_dict()
    prompt_qs_map = (
        page_queries_grp[page_queries_grp["is_ai_prompt"]]
        .groupby("top_page")["query"]
        .apply(list)
        .to_dict()
    )
    clusters_map = (
        page_queries_grp.groupby("top_page")["topic"]
        .apply(lambda s: ", ".join(dict.fromkeys(s.tolist()[:3])))
        .to_dict()
    )

    ai_pages_df = pd.DataFrame()
    ai_daily_series = pd.DataFrame()
    bridge_df = pd.DataFrame()

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

                imp_7d = int(np.sum(rec_7))
                mean_3d = float(np.mean(rec_7[-3:]))
                mean_base = max(float(np.mean(base_14)), 1.0)
                ai_lift = round(mean_3d / mean_base, 2)

                page_records.append({
                    "norm_page": norm_p,
                    "ai_imp_7d": imp_7d,
                    "ai_lift_x": ai_lift,
                    "ai_sparkline": [int(v) for v in rec_7],
                })
        else:
            for _, row in ai_df.iterrows():
                norm_p = row["page"]
                curr = int(row["ai_imp_current"])
                prev = int(row["ai_imp_previous"])
                lift = round(curr / max(prev, 1), 2) if prev > 0 else 1.0
                page_records.append({
                    "norm_page": norm_p,
                    "ai_imp_7d": curr,
                    "ai_lift_x": lift,
                    "ai_sparkline": [prev, curr] if prev > 0 else [curr],
                })

        if page_records:
            ai_pages_df = pd.DataFrame(page_records).merge(web_7d, on="norm_page", how="left")
            ai_pages_df["display_url"] = ai_pages_df["display_url"].fillna(ai_pages_df["norm_page"])
            ai_pages_df["page_role"] = ai_pages_df["norm_page"].apply(lambda u: classify_page_role(url_to_path(u), live_set))
            ai_pages_df["web_imp_7d"] = ai_pages_df["web_imp_7d"].fillna(0).astype(int)
            ai_pages_df["web_org_pos"] = ai_pages_df["web_org_pos"].fillna(0.0)
            ai_pages_df["connected_clusters"] = ai_pages_df["norm_page"].map(clusters_map).fillna("No Web Cluster Match")
            ai_pages_df["connected_trending_queries"] = ai_pages_df["norm_page"].map(
                lambda p: " | ".join(connected_qs_map.get(p, [])[:4]) or "No query match in window"
            )
            ai_pages_df["ai_prompt_queries"] = ai_pages_df["norm_page"].map(
                lambda p: " | ".join(prompt_qs_map.get(p, [])[:3]) or "None"
            )
            ai_pages_df["sector"] = ai_pages_df.apply(
                lambda r: classify_sector(f"{r['display_url']} {r['connected_clusters']} {r['connected_trending_queries']}"),
                axis=1,
            )

            def ai_page_action(r) -> str:
                qs = " ".join(raw_qs_map.get(r["norm_page"], []))
                is_comm = bool(COMMERCIAL_MODIFIERS.search(qs))
                if is_comm and r["ai_lift_x"] >= 1.3:
                    return "Run Google Ads on connected buyer keywords & add pricing/comparison tables."
                if len(prompt_qs_map.get(r["norm_page"], [])) > 0:
                    return "Add a 50-word direct answer summary + FAQ section for the connected 7+ word questions."
                return "Keep statistics and bulleted answers fresh to hold AI Overview visibility."

            ai_pages_df["recommended_action"] = ai_pages_df.apply(ai_page_action, axis=1)
            ai_pages_df = ai_pages_df.sort_values(["ai_lift_x", "ai_imp_7d"], ascending=False).reset_index(drop=True)

            bridge_df = page_queries_grp[page_queries_grp["top_page"].isin(ai_pages_df["norm_page"])].copy()
            bridge_df = bridge_df.merge(
                ai_pages_df[["norm_page", "display_url", "ai_imp_7d", "ai_lift_x"]],
                left_on="top_page",
                right_on="norm_page",
                how="inner",
            )
            bridge_df = bridge_df.sort_values(["ai_lift_x", "imp_7d"], ascending=[False, False]).reset_index(drop=True)

    prompts_df = all_queries_df[all_queries_df["is_ai_prompt"]].copy()
    prompts_df = prompts_df.sort_values(["imp_7d", "query_lift_x"], ascending=[False, False]).reset_index(drop=True)

    return ai_pages_df, ai_daily_series, bridge_df, prompts_df


def build_ads_recommendations(trends_df: pd.DataFrame, queries_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates trending topics for Google Ads and outputs ALL suggested keywords
    with real 7-day impressions, 30-day impressions, daily averages, growth %, and Google rank.
    """
    ads_topics = []
    rec_keywords = []

    money_page_by_topic = trends_df.set_index("topic")["recommended_money_page"].to_dict()

    for _, r in trends_df.iterrows():
        pos = r["connext_org_pos"]
        comm = r["commercial_intent"]
        pages = r["ranking_pages_count"]
        fit = r["ads_fit_score"]
        lp_url = r["recommended_money_page"]

        if r["is_today"] and comm and pos > 5.0:
            campaign_play = "P0 — Launch Ads Today (24h)"
            rationale = f"Search volume jumped +{r['growth_pct_3d']}% in 3 days ({r['impressions_7d']:,} weekly impr.) while Connext sits at Rank #{pos}. Run Exact/Phrase ads to {lp_url}."
        elif comm and pos > 5.0:
            campaign_play = "P1 — High-Intent Search Ad Group"
            rationale = f"Steady +{r['daily_growth_%']}% daily growth on buyer keywords ({r['impressions_7d']:,} weekly impr.) where Connext ranks #{pos}. Route ads to {lp_url}."
        elif r["coverage_status"] == "Already Covered in Sitemap" and pos > 5.0:
            campaign_play = "P1 — Route Traffic to Service Page"
            rationale = f"Connext ranks #{pos} organically ({r['impressions_7d']:,} weekly impr.). Run Google Ads directly to {lp_url} to capture immediate clicks."
        elif pages >= 2 and pos > 8.0:
            campaign_play = "P1 — Consolidate onto Landing Page"
            rationale = f"Organic impressions ({r['impressions_7d']:,}/wk) are split across {pages} URLs (#{pos} avg rank). Focus ad spend on {lp_url}."
        elif comm and pos <= 5.0:
            campaign_play = "P2 — Defend Top-5 Google Spot"
            rationale = f"Connext already ranks #{pos} organically ({r['impressions_7d']:,} weekly impr.). Pair Paid Search with organic to own the top of Google."
        else:
            campaign_play = "P2 — Long-Tail / Retargeting"
            rationale = f"Research-stage surge (+{r['growth_pct_3d']}%, {r['impressions_7d']:,} weekly impr.). Bid on specific long-tail buyer terms or retarget visitors to {lp_url}."

        ads_topics.append({
            "Priority": campaign_play,
            "Ad Fit Score": fit,
            "Vertical": r["sector"],
            "Trending Topic": r["topic"],
            "7d Impressions": r["impressions_7d"],
            "30d Impressions": r["impressions_30d"],
            "Daily Avg Impr.": r["daily_avg_7d"],
            "Growth (vs 2w)": f"+{r['growth_pct_3d']}%",
            "Current Google Rank": r["google_rank_label"],
            "Best Landing Page": lp_url,
            "Why Run Ads": rationale,
        })

    # Include ALL keywords from filtered surging topics so no keyword is hidden without impression data
    for _, q in queries_df.iterrows():
        kw = q["query"]
        is_lt = q["is_long_tail"]
        is_comm = q["is_commercial"]
        q_pos = q["connext_org_pos"]
        rec_lp = money_page_by_topic.get(q["topic"], q["top_page"])
        q_growth = int(q["query_growth_pct"])
        growth_str = f"+{q_growth}%" if q_growth >= 0 else f"{q_growth}%"

        if is_comm and not is_lt:
            kw_tier = "Core Buyer Keyword"
            match_type = '"Phrase" + [Exact]'
            bid_strategy = "Bid for Top of Page (High Buyer Intent)"
        elif is_comm and is_lt:
            kw_tier = "Specific Long-Tail Buyer (Lower CPC)"
            match_type = "[Exact Match]"
            bid_strategy = "High-Intent Exact Bid"
        elif is_lt:
            kw_tier = "Question / Research Long-Tail"
            match_type = "[Exact Match] (Low Bid Cap)"
            bid_strategy = "Test with Low CPC Cap"
        else:
            kw_tier = "Broad Category Term"
            match_type = '"Phrase Match" (Strict Negatives)'
            bid_strategy = "Moderate Bid + Exclude Job Seekers"

        rec_keywords.append({
            "Suggested Keyword": kw,
            "7d Impressions": int(q["imp_7d"]),
            "30d Impressions": int(q["imp_30d"]),
            "Daily Avg Impr.": float(q["daily_7d"]),
            "Growth (vs 2w)": growth_str,
            "Current Google Rank": human_rank_label(q_pos),
            "Keyword Type": kw_tier,
            "Match Type": match_type,
            "Topic Cluster": q["topic"],
            "Vertical": q["sector"],
            "Recommended Landing Page": rec_lp,
            "Bidding Tip": bid_strategy,
        })

    ads_topics_df = pd.DataFrame(ads_topics).sort_values(["Ad Fit Score", "7d Impressions"], ascending=False)
    rec_kws_df = pd.DataFrame(rec_keywords).sort_values(["7d Impressions", "30d Impressions"], ascending=False) if rec_keywords else pd.DataFrame()
    return ads_topics_df, rec_kws_df


def build_content_and_new_topics(trends_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates content recommendations into:
      1. existing_updates_df: Topics where a page ALREADY EXISTS in connextglobal.com/sitemap.xml (no duplicate creation!)
      2. truly_new_df: Topics NOT found in connextglobal.com/sitemap.xml (genuine new content opportunities)
    """
    existing_rows = []
    truly_new_rows = []

    for _, r in trends_df.iterrows():
        pos = r["connext_org_pos"]
        pages = r["ranking_pages_count"]
        cov = r["coverage_status"]
        money_lp = r["recommended_money_page"]
        gsc_url = r["primary_page"]
        sitemap_url = r["closest_existing_url"]
        overlap_pct = r["closest_overlap_pct"]

        if cov == "Old URL Redirect Check":
            priority = "P0 — Fix Old URL Redirect"
            action = f"Google is still showing an older URL ({gsc_url}). Confirm it redirects to {money_lp}."
        elif r["already_exists_in_sitemap"]:
            priority = "P1 — Update Existing Sitemap Page (Do Not Duplicate)"
            if gsc_url != sitemap_url:
                action = (
                    f"Connext already has a live page covering this topic ({sitemap_url}, {overlap_pct}% match). "
                    f"Update that existing page with the rising search phrases and link it with {money_lp} instead of writing a duplicate article."
                )
            else:
                action = (
                    f"Live page {sitemap_url} already exists and ranks #{pos}. "
                    f"Refresh its headings & FAQ section with the rising long-tail searches to push into Google's Top 3."
                )
        elif pages >= 2:
            priority = "P1 — Consolidate Competing Pages"
            action = f"{pages} Connext pages are splitting impressions. Strengthen {money_lp} as the primary page."
        else:
            priority = "P2 — Protect Top-3 Ranking"
            action = f"Connext ranks #{pos} on {gsc_url}. Keep statistics and FAQ answers updated."

        existing_rows.append({
            "Priority": priority,
            "Sitemap Status": "Already in Sitemap.xml" if r["already_exists_in_sitemap"] else "Service Page Exists",
            "Vertical": r["sector"],
            "Trending Topic": r["topic"],
            "7d Impressions": r["impressions_7d"],
            "Growth (vs 2w)": f"+{r['growth_pct_3d']}%",
            "Current Google Rank": r["google_rank_label"],
            "Existing Sitemap Page": sitemap_url,
            "Main Service Page": money_lp,
            "Recommended Action": action,
        })

        # ONLY suggest creating a new page when NO matching page exists in connextglobal.com/sitemap.xml!
        if not r["already_exists_in_sitemap"]:
            slug_base = re.sub(r"[^a-z0-9]+", "-", r["topic"].lower()).strip("-")
            truly_new_rows.append({
                "Recommended New Page Type": "New Blog Guide / Commercial Subpage (Verified Not in Sitemap)",
                "Vertical": r["sector"],
                "New Topic Gap": r["topic"],
                "7d Impressions": r["impressions_7d"],
                "30d Impressions": r["impressions_30d"],
                "Growth (vs 2w)": f"+{r['growth_pct_3d']}%",
                "Suggested New URL Slug": f"/blog/{slug_base}-guide (Not Live Yet)",
                "Closest Existing Sitemap Page": sitemap_url,
                "Similarity to Existing": f"{overlap_pct}% (Low — Safe to Create)",
                "Service Page to Link To": money_lp,
                "Real Rising Searches to Cover": r["top_rising_queries"],
            })

    existing_df = pd.DataFrame(existing_rows)
    truly_new_df = pd.DataFrame(truly_new_rows).sort_values(["7d Impressions", "30d Impressions"], ascending=False) if truly_new_rows else pd.DataFrame()
    return existing_df, truly_new_df


# --- LOAD LIVE SITEMAP (HTTPS://CONNEXTGLOBAL.COM/SITEMAP.XML) ---
sitemap_paths, sitemap_source_label = fetch_live_sitemap_paths()
default_live_set, default_live_df, default_live_vec, default_live_mat = _build_index_from_paths(sitemap_paths)

# --- SIDEBAR CONTROLS (MIDNIGHT NAVY WITH OFFICIAL WHITE CONNEXT LOGO) ---
st.sidebar.image(CNXT_LOGO_WHITE, width=185)
st.sidebar.markdown("**Search & AI Visibility Radar**")
st.sidebar.caption(f"Property: `connextglobal.com` • {sitemap_source_label}")
if st.sidebar.button("Lock / Log Out"):
    st.session_state["cnxt_authenticated"] = False
    st.rerun()
st.sidebar.divider()

has_secrets = "gcp_service_account" in st.secrets and "gsc_config" in st.secrets
source_options = ["Live Google Search Console API", "Upload GSC Web Export (.csv / .zip)"]
data_source = st.sidebar.radio(
    "1. Search Console Data Source",
    source_options,
    index=0,
)

exclude_local_ph = st.sidebar.checkbox(
    "Exclude Local PH City & School Searches",
    value=True,
    help="Removes local PH directory/student searches (e.g., 'bpo in baguio', 'nursing school') so only B2B buyer demand is shown.",
)

custom_negatives = st.sidebar.text_input(
    "Extra Words to Exclude (Pipe-separated)",
    placeholder="intern|internship|sample|pdf",
    help="Job-seeker & recruiting terms are already excluded automatically.",
)

raw_df = pd.DataFrame()
ai_raw_df = pd.DataFrame()
ai_csv_mode = "timeseries"
incomplete_date_note = None
excluded_rows_count = 0
raw_total_count = 0

if data_source == "Live Google Search Console API":
    if has_secrets:
        site_url = st.sidebar.text_input(
            "GSC Property URL",
            value=st.secrets["gsc_config"].get("site_url", DEFAULT_SITE_URL),
        )
        brand_regex = st.sidebar.text_input(
            "Brand Terms Regex",
            value=st.secrets["gsc_config"].get("brand_regex", DEFAULT_BRAND_REGEX),
        )
        lookback = st.sidebar.slider("Days of History to Analyze", 28, 90, 45)
        if st.sidebar.button("Refresh Live GSC Data", type="primary"):
            fetch_gsc_api_data.clear()
        with st.spinner("Loading live Google Search Console data for connextglobal.com..."):
            try:
                creds_dict = dict(st.secrets["gcp_service_account"])
                raw_df, incomplete_date_note, excluded_rows_count, raw_total_count = fetch_gsc_api_data(
                    creds_dict, site_url, lookback, brand_regex, custom_negatives, exclude_local_ph
                )
            except Exception as api_err:
                st.error(f"Google Search Console API Error: {api_err}")
                st.stop()
    else:
        st.sidebar.warning("`st.secrets` not configured. Upload Service Account JSON below:")
        uploaded_json = st.sidebar.file_uploader("Service Account JSON", type=["json"])
        site_url = st.sidebar.text_input("GSC Property URL", value=DEFAULT_SITE_URL)
        brand_regex = st.sidebar.text_input("Brand Terms Regex", value=DEFAULT_BRAND_REGEX)
        lookback = st.sidebar.slider("Days of History to Analyze", 28, 90, 45)
        if uploaded_json and site_url:
            creds_dict = json.load(uploaded_json)
            with st.spinner("Loading live Google Search Console data for connextglobal.com..."):
                raw_df, incomplete_date_note, excluded_rows_count, raw_total_count = fetch_gsc_api_data(
                    creds_dict, site_url, lookback, brand_regex, custom_negatives, exclude_local_ph
                )

elif data_source == "Upload GSC Web Export (.csv / .zip)":
    st.sidebar.markdown("Required columns: `date, query, page, impressions, position`")
    uploaded_csv = st.sidebar.file_uploader("Upload Web Search Analytics CSV or ZIP", type=["csv", "zip"])
    brand_regex = st.sidebar.text_input("Brand Terms Regex", value=DEFAULT_BRAND_REGEX)
    if uploaded_csv:
        try:
            raw_df, excluded_rows_count = parse_gsc_web_upload(
                uploaded_csv, brand_regex, custom_negatives, exclude_local_ph
            )
            raw_total_count = len(raw_df) + excluded_rows_count
            st.sidebar.success(f"Loaded {len(raw_df):,} verified GSC rows.")
        except Exception as upload_err:
            st.sidebar.error(f"Upload Error: {upload_err}")

st.sidebar.divider()
st.sidebar.subheader("2. Sitemap & AI Report Sync")
if st.sidebar.button("Re-Sync connextglobal.com/sitemap.xml"):
    fetch_live_sitemap_paths.clear()
    st.rerun()

uploaded_ai_csv = st.sidebar.file_uploader(
    "GSC Generative AI Export (.csv / .zip)",
    type=["csv", "zip"],
    key="ai_csv_uploader",
    help="In GSC, go to Performance -> Generative AI -> Export -> Download CSV and upload the .zip or Pages.csv here.",
)
if uploaded_ai_csv:
    try:
        ai_raw_df, ai_csv_mode = parse_gsc_ai_report_csv(uploaded_ai_csv)
        st.sidebar.success(f"Loaded {len(ai_raw_df):,} Generative AI rows ({ai_csv_mode}).")
    except Exception as exc:
        st.sidebar.error(f"AI Export Error: {exc}")

uploaded_live_sheet = st.sidebar.file_uploader(
    "Manual Sitemap Override (.xlsx / .csv)",
    type=["xlsx", "xls", "csv"],
    key="live_pages_uploader",
    help="Optional: upload a custom URL spreadsheet to override connextglobal.com/sitemap.xml.",
)
if uploaded_live_sheet:
    try:
        custom_paths = parse_uploaded_live_pages(uploaded_live_sheet)
        if custom_paths:
            live_set, live_df, live_vec, live_mat = _build_index_from_paths(custom_paths)
            sitemap_source_label = f"Uploaded File ({len(live_set):,} URLs)"
            st.sidebar.success(f"Loaded {len(live_set):,} custom URLs.")
        else:
            live_set, live_df, live_vec, live_mat = default_live_set, default_live_df, default_live_vec, default_live_mat
    except Exception as exc:
        st.sidebar.error(f"Upload Error: {exc}")
        live_set, live_df, live_vec, live_mat = default_live_set, default_live_df, default_live_vec, default_live_mat
else:
    live_set, live_df, live_vec, live_mat = default_live_set, default_live_df, default_live_vec, default_live_mat

st.sidebar.divider()
st.sidebar.subheader("3. Trend Sensitivity")
preset_mode = st.sidebar.selectbox(
    "How sensitive should trend detection be?",
    [
        "Broad & Inclusive (Shows All Verticals)",
        "Balanced (Moderate Surges)",
        "Strict (Major Breakouts Only)",
    ],
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


# --- CONNEXT BRANDED HEADER & HERO BANNER ---
hdr_logo, hdr_meta = st.columns([1.2, 4])
with hdr_logo:
    st.image(CNXT_LOGO_COLOR, width=195)
with hdr_meta:
    if not raw_df.empty:
        d_min = raw_df["date"].min().strftime("%b %d, %Y")
        d_max = raw_df["date"].max().strftime("%b %d, %Y")
        n_completed_days = raw_df["date"].nunique()
        date_badge = f"Verified GSC Window: {d_min} – {d_max} ({n_completed_days} complete days)"
    else:
        date_badge = "Awaiting GSC Data"
    st.caption(f"{date_badge}  •  Synced with {sitemap_source_label}  •  Impressions & Google Rank Only")

st.markdown(
    """<div class="cnxt-hero-banner">
<div class="cnxt-eyebrow">CONNEXT GLOBAL GROWTH &amp; SEARCH INTELLIGENCE</div>
<h1 class="cnxt-hero-title">Search &amp; AI Visibility <span>Trend Radar</span></h1>
<p class="cnxt-hero-sub">Track rising B2B outsourcing topics on Google right now—checked against <strong>connextglobal.com/sitemap.xml</strong> so Paid Ads gets real search volume per keyword and Content only creates pages we don’t already have.</p>
</div>""",
    unsafe_allow_html=True,
)

# --- 30-SECOND PLAIN-ENGLISH GUIDE FOR NON-SEO REVIEWERS ---
with st.expander("New here? 30-Second Guide for Non-SEO Reviewers", expanded=False):
    st.markdown(
        """
- **What are "Impressions"?** Every time someone searches Google and sees a `connextglobal.com` link on their screen, that counts as **1 impression**. Rising impressions mean growing customer demand—even before they click.
- **What is "Growth (vs 2w)"?** How much faster people are searching for a topic over the last 3 days compared to the previous 2 weeks (e.g., `+85%` means nearly double the daily search visibility).
- **What is "Google Rank"?** Where Connext currently shows up in organic (unpaid) Google results: **Top 3 (#1–3)** is the top of Page 1, **Page 1 (#4–10)** is visible without scrolling to Page 2, and **Page 2+ (#11+)** means most searchers won't see us unless we run Google Ads or update our page.
- **How to use each tab:**
  1. **Trending Topics Overview:** See which topics and business verticals are growing fastest right now.
  2. **Pages & Long-Tail Queries:** See which Connext pages currently show up on Google and the exact phrases people type.
  3. **AI Search & Prompts:** See question-style searches (7+ words) that trigger Google AI Overviews & AI Mode.
  4. **Paid Ads & Keywords:** Get ready-to-bid Google Ads keywords with **real 7-day and 30-day impression counts** and matching landing pages.
  5. **Content Plan & New Topics:** Automatically separates **pages that already exist in `sitemap.xml`** (update them, don't duplicate!) from **genuinely new topic gaps**.
        """
    )

if raw_df.empty:
    st.info("Configure your GSC credentials or upload a Search Console export in the sidebar to begin analyzing `connextglobal.com`.")
    st.stop()

with st.spinner("Grouping live B2B search queries & cross-checking connextglobal.com/sitemap.xml..."):
    clustered_df = cluster_queries(raw_df, cluster_dist)
    trends_df, daily_topic_df, all_queries_df, pages_df = analyze_multi_horizon_trends(
        clustered_df,
        raw_df,
        live_set=live_set,
        live_df=live_df,
        live_vec=live_vec,
        live_mat=live_mat,
        min_daily_imp=min_daily_imp,
        min_up_days=min_up_days,
        min_norm_slope=min_slope_pct / 100.0,
        min_lift=min_lift,
    )
    ai_pages_df, ai_daily_series, ai_bridge_df, ai_prompts_df = analyze_ai_search_verified(
        ai_raw_df, ai_csv_mode, clustered_df, all_queries_df, live_set
    )

# --- DATA VERIFICATION SUMMARY EXPANDER ---
with st.expander("Data Verification & GSC Extraction Summary (100% Live Data)", expanded=False):
    aud1, aud2, aud3, aud4, aud5 = st.columns(5)
    aud1.metric("Raw GSC Rows Fetched", f"{raw_total_count:,}")
    aud2.metric("Job/Noise Rows Filtered", f"{excluded_rows_count:,}")
    aud3.metric("Non-Branded B2B Rows", f"{len(clustered_df):,}")
    aud4.metric("Unique Non-Branded Queries", f"{clustered_df['query'].nunique():,}" if not clustered_df.empty else "0")
    aud5.metric("Live Sitemap URLs Checked", f"{len(live_set):,}")

    if incomplete_date_note and incomplete_date_note != "None":
        st.caption(
            f"Google Search Console reports incomplete data starting `{incomplete_date_note}`. Dates on or after `{incomplete_date_note}` are automatically excluded so daily trends are never distorted by partial-day numbers."
        )

if trends_df.empty:
    st.warning("No topics matched your current sensitivity settings. Select **Broad & Inclusive** in the sidebar.")
    st.stop()

# --- CLEAN, EASY-TO-UNDERSTAND FILTER BAR ---
vertical_counts = trends_df["sector"].value_counts().to_dict()
vertical_options = ["All Business Verticals"] + [
    f"{v} ({vertical_counts.get(v, 0)} topics)" for v in ALL_CONNEXT_VERTICALS
]

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
    chosen_vertical = selected_vertical_label.rsplit(" (", 1)[0]
    filtered_trends = filtered_trends[filtered_trends["sector"] == chosen_vertical]

if selected_action_label != "All Recommended Actions":
    chosen_action = selected_action_label.rsplit(" (", 1)[0]
    filtered_trends = filtered_trends[filtered_trends["team_action"] == chosen_action]

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
        | filtered_trends["recommended_money_page"].str.contains(qs_pat, case=False, na=False)
        | filtered_trends["closest_existing_url"].str.contains(qs_pat, case=False, na=False)
    ]

filtered_trends = filtered_trends.reset_index(drop=True)
if filtered_trends.empty:
    st.info("No surging topics in this specific combination right now. Select **All Business Verticals** or **All Recommended Actions** above to see all active trends.")
    st.stop()

filtered_queries = all_queries_df[all_queries_df["topic"].isin(filtered_trends["topic"])].copy()
filtered_pages = pages_df[pages_df["topic"].isin(filtered_trends["topic"])].copy()

if selected_vertical_label != "All Business Verticals" and not ai_pages_df.empty:
    chosen_vertical = selected_vertical_label.rsplit(" (", 1)[0]
    filtered_ai_pages = ai_pages_df[ai_pages_df["sector"] == chosen_vertical].copy()
else:
    filtered_ai_pages = ai_pages_df.copy() if not ai_pages_df.empty else pd.DataFrame()

filtered_ai_bridge = ai_bridge_df[ai_bridge_df["norm_page"].isin(filtered_ai_pages["norm_page"])].copy() if not ai_bridge_df.empty else pd.DataFrame()
filtered_ai_prompts = ai_prompts_df[ai_prompts_df["topic"].isin(filtered_trends["topic"])].copy() if not ai_prompts_df.empty else pd.DataFrame()

ads_topics_df, rec_kws_df = build_ads_recommendations(filtered_trends, filtered_queries)
existing_content_df, truly_new_df = build_content_and_new_topics(filtered_trends)

# --- TOP KPI METRICS (PLAIN ENGLISH) ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Surging Today (Last 3 Days)",
    int(filtered_trends["is_today"].sum()),
    help="Topics with a sharp impression spike over the last 3 completed days compared to the prior 2 weeks.",
)
k2.metric(
    "Surging This Week (7 Days)",
    int(filtered_trends["is_week"].sum()),
    help="Topics growing consistently day-over-day across the past week.",
)
k3.metric(
    "Growing This Month (30 Days)",
    int(filtered_trends["is_month"].sum()),
    help="Topics showing sustained month-over-month impression expansion.",
)
k4.metric(
    "Weekly Search Impressions",
    f"{int(filtered_trends['impressions_7d'].sum()):,}",
    help="Total times connextglobal.com appeared on Google for these surging topics in the last 7 days.",
)
k5.metric(
    "Avg Google Rank",
    f"#{filtered_trends['connext_org_pos'].mean():.1f}",
    help="Average organic search position for connextglobal.com across these topics (#1–10 = Page 1).",
)

# --- MAIN NAVIGATION TABS ---
ai_tab_count = len(filtered_ai_pages) if not filtered_ai_pages.empty else len(filtered_ai_prompts)
tab_horizons, tab_pages_clusters, tab_ai_search, tab_ads_strategy, tab_content = st.tabs([
    "Trending Topics Overview",
    f"Pages & Long-Tail Queries ({len(filtered_pages)} URLs)",
    f"AI Search & Prompts ({ai_tab_count})",
    f"Paid Ads & Keywords ({len(rec_kws_df)} KWs)",
    f"Content Plan & New Topics ({len(truly_new_df)} Truly New)",
])

# ==============================================================================
# TAB 1: TRENDING TOPICS OVERVIEW
# ==============================================================================
with tab_horizons:
    horizon_choice = st.radio(
        "Timeframe:",
        [
            "All Surging Topics",
            "Today (Last 3 Days Spike)",
            "This Week (7-Day Surge)",
            "This Month (30-Day Growth)",
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
        g1, g2 = st.columns([1.15, 1])

        with g1:
            st.subheader("Top Surging Topics by Weekly Search Impressions")
            top_bars = view_df.head(12).copy()
            fig_topics = px.bar(
                top_bars,
                x="impressions_7d",
                y="topic",
                orientation="h",
                color="team_action",
                text="growth_pct_3d",
                color_discrete_map={
                    "Launch Paid Ads": CNXT_COLORS["sun"],
                    "Update Existing Page": CNXT_COLORS["blue"],
                    "Write New Page (Not in Sitemap)": CNXT_COLORS["twilight"],
                    "Defend Top 3 Rank": CNXT_COLORS["teal"],
                },
                custom_data=["sector", "google_rank_label", "growth_pct_3d", "closest_existing_url"],
                labels={
                    "impressions_7d": "7-Day Search Impressions",
                    "topic": "",
                    "team_action": "Recommended Action",
                },
            )
            fig_topics.update_traces(
                texttemplate="+%{text}%",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "7-Day Impressions: <b>%{x:,}</b><br>"
                    "Growth vs. Prior 2 Weeks: <b>+%{customdata[2]}%</b><br>"
                    "Google Rank: <b>%{customdata[1]}</b><br>"
                    "Vertical: %{customdata[0]}<br>"
                    "Closest Sitemap Page: %{customdata[3]}<extra></extra>"
                ),
            )
            fig_topics.update_layout(
                height=430,
                margin=dict(l=10, r=45, t=10, b=50),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(gridcolor=CNXT_COLORS["line"]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
                legend=dict(orientation="h", y=-0.18, title=""),
            )
            st.plotly_chart(fig_topics, use_container_width=True)

        with g2:
            st.subheader("Search Impressions by Business Vertical")
            sector_agg = (
                view_df.groupby("sector", as_index=False)
                .agg(
                    topics=("topic", "count"),
                    impressions_7d=("impressions_7d", "sum"),
                )
                .sort_values("impressions_7d", ascending=True)
            )
            fig_sector = px.bar(
                sector_agg,
                x="impressions_7d",
                y="sector",
                orientation="h",
                text="topics",
                color_discrete_sequence=[CNXT_COLORS["twilight"]],
                custom_data=["topics"],
                labels={
                    "impressions_7d": "7-Day Search Impressions",
                    "sector": "",
                },
            )
            fig_sector.update_traces(
                texttemplate="%{text} topics",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "7-Day Impressions: <b>%{x:,}</b><br>"
                    "Surging Topics: <b>%{customdata[0]}</b><extra></extra>"
                ),
            )
            fig_sector.update_layout(
                height=430,
                margin=dict(l=10, r=55, t=10, b=50),
                xaxis=dict(gridcolor=CNXT_COLORS["line"]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
            )
            st.plotly_chart(fig_sector, use_container_width=True)

        st.divider()
        st.subheader("Inspect a Topic's Daily Search Trend")
        selected_topic = st.selectbox("Choose a topic to see its daily impression history & sitemap check:", view_df["topic"].tolist(), key="horizon_topic_select")
        sel_row = view_df[view_df["topic"] == selected_topic].iloc[0]

        sitemap_note = (
            f"<strong>Already in Sitemap.xml:</strong> <code>{sel_row['closest_existing_url']}</code> ({sel_row['closest_existing_role']}, {sel_row['closest_overlap_pct']}% match) — update this page rather than creating a duplicate."
            if sel_row["already_exists_in_sitemap"]
            else f"<strong>Not in Sitemap.xml yet:</strong> Closest existing page is <code>{sel_row['closest_existing_url']}</code> ({sel_row['closest_overlap_pct']}% overlap) — safe to create a dedicated page."
        )

        st.markdown(
            f"""<div class="cnxt-cream-panel">
<strong>Plain-English Summary for "{sel_row['topic'].title()}":</strong><br/>
• <strong>Search Volume &amp; Momentum:</strong> Appeared <strong>{sel_row['impressions_7d']:,} times</strong> on Google in the last 7 days and <strong>{sel_row['impressions_30d']:,} times</strong> in the last 30 days (<strong>+{sel_row['growth_pct_3d']}%</strong> vs. prior 2 weeks).<br/>
• <strong>Current Google Rank:</strong> <strong>{sel_row['google_rank_label']}</strong> on <code>{sel_row['primary_page']}</code>.<br/>
• {sitemap_note}<br/>
• <strong>Recommended Next Step ({sel_row['team_action']}):</strong> Route Google Ads &amp; internal links to <code>{sel_row['recommended_money_page']}</code> using rising searches like <em>{sel_row['top_rising_queries']}</em>.
</div>""",
            unsafe_allow_html=True,
        )

        topic_series = daily_topic_df[daily_topic_df["topic"] == selected_topic].sort_values("date").copy()
        topic_series["rolling_7d_imp"] = topic_series["impressions"].rolling(7, min_periods=1).mean().round(0)

        fig_line = go.Figure()
        fig_line.add_trace(go.Bar(
            x=topic_series["date"],
            y=topic_series["impressions"],
            name="Daily Impressions",
            marker_color="rgba(0, 90, 172, 0.30)",
            hovertemplate="Date: %{x|%b %d, %Y}<br>Daily Impressions: <b>%{y:,}</b><extra></extra>",
        ))
        fig_line.add_trace(go.Scatter(
            x=topic_series["date"],
            y=topic_series["rolling_7d_imp"],
            mode="lines+markers",
            name="7-Day Average Trend",
            line=dict(color=CNXT_COLORS["sun"], width=3),
            marker=dict(color=CNXT_COLORS["twilight"], size=6),
            hovertemplate="Date: %{x|%b %d, %Y}<br>7-Day Avg Impressions: <b>%{y:,.0f}</b><extra></extra>",
        ))
        fig_line.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=15, b=45),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
            yaxis=dict(title="Google Search Impressions", gridcolor=CNXT_COLORS["line"]),
            xaxis=dict(title=""),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader(f"All Surging Topics ({len(view_df)})")
        overview_table = view_df[[
            "topic", "sector", "team_action", "coverage_status", "sparkline_7d",
            "impressions_7d", "impressions_30d", "growth_pct_3d",
            "google_rank_label", "closest_existing_url", "recommended_money_page",
        ]].copy()
        overview_table["growth_label"] = overview_table["growth_pct_3d"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
        overview_table = overview_table[[
            "topic", "sector", "team_action", "coverage_status", "sparkline_7d",
            "impressions_7d", "impressions_30d", "growth_label",
            "google_rank_label", "closest_existing_url", "recommended_money_page",
        ]]
        st.dataframe(
            overview_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "topic": "Trending Topic",
                "sector": "Business Vertical",
                "team_action": "Recommended Action",
                "coverage_status": "Sitemap Check",
                "sparkline_7d": st.column_config.LineChartColumn("7-Day Curve", width="small"),
                "impressions_7d": st.column_config.NumberColumn("7d Impressions"),
                "impressions_30d": st.column_config.NumberColumn("30d Impressions"),
                "growth_label": "Growth (vs 2w)",
                "google_rank_label": "Current Google Rank",
                "closest_existing_url": st.column_config.LinkColumn("Existing Sitemap Page"),
                "recommended_money_page": st.column_config.LinkColumn("Best Service Page"),
            },
        )

# ==============================================================================
# TAB 2: PAGES ON TOPICS & LONG-TAIL QUERIES
# ==============================================================================
with tab_pages_clusters:
    st.subheader("Which Connext Pages Rank for Each Trending Topic & Long-Tail Search")
    st.markdown(
        """<div class="cnxt-cream-panel">
<strong>How to Use This Tab:</strong> Pick any trending topic below to see (1) which <code>connextglobal.com</code> pages currently appear on Google for it, (2) which live pages from <code>sitemap.xml</code> already cover it, and (3) the exact 7-day and 30-day search volume of every phrase people typed.
</div>""",
        unsafe_allow_html=True,
    )

    sec_col1, sec_col2 = st.columns([1, 1.1])
    with sec_col1:
        st.markdown("**Weekly Impressions by Website Section**")
        role_agg = (
            filtered_pages.groupby("page_role", as_index=False)
            .agg(impressions=("page_imp_7d", "sum"), urls=("norm_page", "nunique"))
            .sort_values("impressions", ascending=True)
        )
        fig_roles = px.bar(
            role_agg,
            x="impressions",
            y="page_role",
            orientation="h",
            text="urls",
            color_discrete_sequence=[CNXT_COLORS["twilight"]],
            labels={"impressions": "7-Day Impressions", "page_role": ""},
        )
        fig_roles.update_traces(texttemplate="%{text} pages", textposition="outside")
        fig_roles.update_layout(
            height=330,
            margin=dict(l=10, r=45, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
        )
        st.plotly_chart(fig_roles, use_container_width=True)

    with sec_col2:
        st.markdown("**Short Searches vs. Detailed Long-Tail Searches (4+ Words)**")
        lt_split = filtered_queries.copy()
        lt_split["Search Length"] = np.where(lt_split["is_long_tail"], "Long-Tail (4+ Words)", "Short / Mid-Tail (1–3 Words)")
        lt_agg = lt_split.groupby(["sector", "Search Length"], as_index=False)["imp_7d"].sum()
        fig_lt_bar = px.bar(
            lt_agg,
            x="sector",
            y="imp_7d",
            color="Search Length",
            barmode="group",
            color_discrete_map={"Long-Tail (4+ Words)": CNXT_COLORS["teal"], "Short / Mid-Tail (1–3 Words)": CNXT_COLORS["blue"]},
            labels={"sector": "", "imp_7d": "7-Day Impressions"},
        )
        fig_lt_bar.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=10, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
            legend=dict(orientation="h", y=-0.22, title=""),
        )
        st.plotly_chart(fig_lt_bar, use_container_width=True)

    st.divider()
    inspect_cluster = st.selectbox(
        "Select a Trending Topic to Inspect Its Pages & Exact Searches:",
        filtered_trends["topic"].tolist(),
        key="cluster_detail_select",
    )
    topic_meta = filtered_trends[filtered_trends["topic"] == inspect_cluster].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Connext Google Rank", topic_meta["google_rank_label"])
    m2.metric("Sitemap Status", topic_meta["coverage_status"])
    m3.metric("Connext Pages Ranking", topic_meta["ranking_pages_count"])
    m4.metric("Long-Tail Searches (4+ words)", f"{topic_meta['long_tail_count']} of {topic_meta['total_queries']}")

    col_p, col_live = st.columns([1.15, 1])

    with col_p:
        st.markdown("**1. Connext Pages Currently Ranking on Google**")
        topic_pages = filtered_pages[filtered_pages["topic"] == inspect_cluster].sort_values("page_imp_7d", ascending=False).copy()
        topic_pages["rank_text"] = topic_pages["page_org_pos"].apply(human_rank_label)
        st.dataframe(
            topic_pages[["norm_page", "page_role", "page_imp_7d", "topic_imp_share_%", "rank_text", "queries_on_page"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "norm_page": st.column_config.LinkColumn("Ranking Connext URL"),
                "page_role": "Page Type",
                "page_imp_7d": st.column_config.NumberColumn("7d Impressions"),
                "topic_imp_share_%": st.column_config.NumberColumn("Share %", format="%.1f%%"),
                "rank_text": "Google Rank",
                "queries_on_page": "Keywords",
            },
        )

    with col_live:
        st.markdown("**2. Matching Live Pages Already in `sitemap.xml`**")
        matched_records = []
        seen_urls = set()
        for item in topic_meta["existing_sitemap_matches"]:
            if item["full_url"] not in seen_urls:
                matched_records.append({
                    "Live Connext URL": item["full_url"],
                    "Page Type": item["role"],
                    "Status": "Already Covers Topic in Sitemap",
                })
                seen_urls.add(item["full_url"])
        for item in topic_meta["matching_money_pages"]:
            if item["full_url"] not in seen_urls:
                matched_records.append({
                    "Live Connext URL": item["full_url"],
                    "Page Type": item["role"],
                    "Status": "Best Commercial Service Page",
                })
                seen_urls.add(item["full_url"])
        for item in topic_meta["matching_content_pages"]:
            if item["full_url"] not in seen_urls:
                matched_records.append({
                    "Live Connext URL": item["full_url"],
                    "Page Type": item["role"],
                    "Status": "Related Content in Sitemap",
                })
                seen_urls.add(item["full_url"])
        matched_live_df = pd.DataFrame(matched_records)
        st.dataframe(
            matched_live_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Live Connext URL": st.column_config.LinkColumn("Live Sitemap URL"),
            },
        )

    st.markdown("**3. Exact Searches People Typed for This Topic (With Real Impressions)**")
    topic_qs = filtered_queries[filtered_queries["topic"] == inspect_cluster].sort_values("imp_7d", ascending=False).copy()
    topic_qs["growth_str"] = topic_qs["query_growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
    topic_qs["rank_str"] = topic_qs["connext_org_pos"].apply(human_rank_label)
    st.dataframe(
        topic_qs[["query", "is_long_tail", "imp_7d", "imp_30d", "daily_7d", "growth_str", "rank_str", "top_page"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "query": "Search Query",
            "is_long_tail": st.column_config.CheckboxColumn("Long-Tail (4+w)"),
            "imp_7d": st.column_config.NumberColumn("7d Impressions"),
            "imp_30d": st.column_config.NumberColumn("30d Impressions"),
            "daily_7d": st.column_config.NumberColumn("Daily Avg", format="%.1f"),
            "growth_str": "Growth (vs 2w)",
            "rank_str": "Google Rank",
            "top_page": st.column_config.LinkColumn("Ranking Connext URL"),
        },
    )

# ==============================================================================
# TAB 3: AI SEARCH (VERIFIED GSC GENERATIVE AI REPORT + REAL WEB PROMPTS)
# ==============================================================================
with tab_ai_search:
    st.subheader("AI Overviews & AI Mode Visibility")

    if not filtered_ai_pages.empty:
        st.markdown(
            """<div class="cnxt-cream-panel">
<strong>Verified GSC Generative AI Report Connected:</strong> Showing exact page-level AI Overview &amp; AI Mode impressions from your uploaded GSC Generative AI export, connected by URL to the live Web Search queries ranking on each page.
</div>""",
            unsafe_allow_html=True,
        )

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Pages in GSC AI Report", len(filtered_ai_pages))
        a2.metric("Verified AI Impressions", f"{int(filtered_ai_pages['ai_imp_7d'].sum()):,}")
        a3.metric("Avg AI Growth", f"+{round((filtered_ai_pages['ai_lift_x'].mean()-1)*100)}%")
        a4.metric("Connected Web Queries", len(filtered_ai_bridge))

        ai_col1, ai_col2 = st.columns([1.15, 1])

        with ai_col1:
            st.markdown("**1. Top Pages by Verified Generative AI Impressions**")
            top_ai_bar = filtered_ai_pages.head(10).copy()
            top_ai_bar["short_path"] = top_ai_bar["display_url"].str.replace("https://connextglobal.com", "", regex=False)

            fig_ai_bar = px.bar(
                top_ai_bar,
                x="ai_imp_7d",
                y="short_path",
                orientation="h",
                color_discrete_sequence=[CNXT_COLORS["twilight"]],
                labels={"ai_imp_7d": "Verified AI Impressions", "short_path": ""},
            )
            fig_ai_bar.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_ai_bar, use_container_width=True)

        with ai_col2:
            st.markdown("**2. AI Impression Trend Curve**")
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
                    labels={"date": "", "ai_impressions": "Verified AI Impressions"},
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
                    labels={"date": "", "impressions": "Web Search Impressions (URL)"},
                    color_discrete_sequence=[CNXT_COLORS["blue"]],
                )
            fig_ai_line.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto, sans-serif", color=CNXT_COLORS["midnight"]),
            )
            st.plotly_chart(fig_ai_line, use_container_width=True)

        st.markdown("### 3. Verified AI Pages Connected to Rising Searches")
        ai_display_cols = [
            "display_url", "page_role", "sector", "ai_sparkline", "ai_imp_7d",
            "web_imp_7d", "connected_clusters", "connected_trending_queries", "recommended_action",
        ]
        st.dataframe(
            filtered_ai_pages[ai_display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "display_url": st.column_config.LinkColumn("Trending AI Page"),
                "page_role": "Page Type",
                "sector": "Vertical",
                "ai_sparkline": st.column_config.LineChartColumn("AI Trend", width="small"),
                "ai_imp_7d": st.column_config.NumberColumn("Verified AI Impr."),
                "web_imp_7d": st.column_config.NumberColumn("Web 7d Impr."),
                "connected_clusters": "Topic Clusters",
                "connected_trending_queries": "Connected Rising Searches",
                "recommended_action": "Recommended Action",
            },
        )
    else:
        st.markdown(
            """<div class="cnxt-cream-panel">
<strong>How to View Isolated AI Overview / AI Mode Page Numbers:</strong> Google’s Search Console API combines AI impressions with standard Web impressions. To see exact AI-only page counts here, open <strong>Google Search Console → Performance → Generative AI</strong>, click <strong>Export → Download CSV</strong>, and upload the <code>.zip</code> or <code>Pages.csv</code> in <strong>Section 2 of the left sidebar</strong>.<br/><br/>
Below are your <strong>100% live GSC conversational question &amp; prompt searches (7+ words)</strong>—these are the exact queries that trigger Google AI Overviews and AI Mode answers.
</div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Conversational & Question Searches Triggering AI Answers (7+ Words)")
    if filtered_ai_prompts.empty:
        st.info("No 7+ word or question-style searches matched your current filters.")
    else:
        p_disp = filtered_ai_prompts.copy()
        p_disp["growth_str"] = p_disp["query_growth_pct"].apply(lambda x: f"+{x}%" if x >= 0 else f"{x}%")
        p_disp["rank_str"] = p_disp["connext_org_pos"].apply(human_rank_label)
        st.dataframe(
            p_disp[["query", "sector", "topic", "imp_7d", "imp_30d", "growth_str", "rank_str", "top_page"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "query": "Conversational Search / Prompt",
                "sector": "Vertical",
                "topic": "Parent Topic",
                "imp_7d": st.column_config.NumberColumn("7d Impressions"),
                "imp_30d": st.column_config.NumberColumn("30d Impressions"),
                "growth_str": "Growth (vs 2w)",
                "rank_str": "Google Rank",
                "top_page": st.column_config.LinkColumn("Ranking Connext Page"),
            },
        )
        st.download_button(
            "Download Conversational AI Prompts CSV",
            p_disp[["query", "sector", "topic", "imp_7d", "imp_30d", "growth_str", "rank_str", "top_page"]].to_csv(index=False).encode("utf-8"),
            file_name="connext_conversational_ai_prompts.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 4: PAID ADS OPPORTUNITIES & SUGGESTED KEYWORDS (WITH REAL IMPRESSIONS)
# ==============================================================================
with tab_ads_strategy:
    st.subheader("Trending Topics & Suggested Keywords for Google Ads (With Real Search Volume)")
    st.markdown(
        "Every topic and suggested keyword below includes its **real Google Search Console impressions (7-day, 30-day, and daily average)**, **2-week growth %**, and **current organic Google rank** so your Paid Media team can prioritize high-volume buyer terms."
    )

    p0_p1_ads = ads_topics_df[ads_topics_df["Ad Fit Score"] >= 60]
    high_intent_lt = rec_kws_df[rec_kws_df["Keyword Type"] == "Specific Long-Tail Buyer (Lower CPC)"] if not rec_kws_df.empty else pd.DataFrame()

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"""<div class="cnxt-cream-panel">
<strong>Top Paid Search Opportunities ({len(p0_p1_ads)})</strong><br/>
{", ".join(f"<code>{t}</code> ({imp:,} 7d impr.)" for t, imp in zip(p0_p1_ads["Trending Topic"].head(2), p0_p1_ads["7d Impressions"].head(2))) if not p0_p1_ads.empty else "No high-gap commercial surges currently."}
</div>""",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""<div class="cnxt-cream-panel">
<strong>Top Lower-CPC Buyer Keywords ({len(high_intent_lt)})</strong><br/>
{", ".join(f"<code>[{k}]</code> ({imp:,} 7d impr.)" for k, imp in zip(high_intent_lt["Suggested Keyword"].head(2), high_intent_lt["7d Impressions"].head(2))) if not high_intent_lt.empty else "Switch to Broad sensitivity in sidebar to surface more long-tail terms."}
</div>""",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """<div class="cnxt-cream-panel">
<strong>B2B Negative Keyword Guardrail</strong><br/>
Always add <code>-jobs</code>, <code>-careers</code>, <code>-hiring</code>, <code>-salary</code>, <code>-glassdoor</code>, and <code>-template</code> negatives before launching.
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### 1. Trending Topics Evaluated for Google Ads")
    st.dataframe(
        ads_topics_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ad Fit Score": st.column_config.ProgressColumn("Ad Fit Score", min_value=0, max_value=100, format="%d/100"),
            "7d Impressions": st.column_config.NumberColumn("7d Impressions"),
            "30d Impressions": st.column_config.NumberColumn("30d Impressions"),
            "Daily Avg Impr.": st.column_config.NumberColumn("Daily Avg Impr.", format="%.1f"),
            "Best Landing Page": st.column_config.LinkColumn("Best Landing Page"),
        },
    )

    st.divider()
    st.markdown(f"### 2. All Suggested Keywords to Bid On ({len(rec_kws_df)} Keywords with Real Impressions)")
    if rec_kws_df.empty:
        st.info("No keywords matched the current filters.")
    else:
        f_kw1, f_kw2 = st.columns([2, 1])
        with f_kw1:
            kw_topic_filter = st.multiselect(
                "Filter Keywords by Topic:",
                options=sorted(rec_kws_df["Topic Cluster"].unique().tolist()),
                default=sorted(rec_kws_df["Topic Cluster"].unique().tolist()),
            )
        with f_kw2:
            min_kw_imp = st.number_input("Min 7-Day Impressions per Keyword:", min_value=1, max_value=10000, value=5, step=5)

        scoped_kws = rec_kws_df[
            (rec_kws_df["Topic Cluster"].isin(kw_topic_filter))
            & (rec_kws_df["7d Impressions"] >= min_kw_imp)
        ]

        st.dataframe(
            scoped_kws,
            use_container_width=True,
            hide_index=True,
            column_config={
                "7d Impressions": st.column_config.NumberColumn("7d Impressions"),
                "30d Impressions": st.column_config.NumberColumn("30d Impressions"),
                "Daily Avg Impr.": st.column_config.NumberColumn("Daily Avg Impr.", format="%.1f"),
                "Recommended Landing Page": st.column_config.LinkColumn("Recommended Landing Page"),
            },
        )
        st.download_button(
            "Download Suggested Keywords & Impressions CSV",
            scoped_kws.to_csv(index=False).encode("utf-8"),
            file_name="connext_suggested_paid_keywords_with_impressions.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("### 3. Ad Group Builder: Keyword Volume Breakdown & Copy-Paste Lists")
        selected_ad_topic = st.selectbox("Select Trending Topic to Build Ad Group:", ads_topics_df["Trending Topic"].tolist())
        ad_row = filtered_trends[filtered_trends["topic"] == selected_ad_topic].iloc[0]
        topic_kw_df = rec_kws_df[rec_kws_df["Topic Cluster"] == selected_ad_topic].sort_values("7d Impressions", ascending=False)

        st.caption(
            f"Recommended Landing Page: **{ad_row['recommended_money_page']}**  •  "
            f"Topic Total: **{ad_row['impressions_7d']:,}** 7d impressions / **{ad_row['impressions_30d']:,}** 30d impressions"
        )

        # Show exact impression volume for every keyword in this selected ad group
        st.dataframe(
            topic_kw_df[["Suggested Keyword", "7d Impressions", "30d Impressions", "Daily Avg Impr.", "Growth (vs 2w)", "Current Google Rank", "Match Type"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "7d Impressions": st.column_config.NumberColumn("7d Impressions"),
                "30d Impressions": st.column_config.NumberColumn("30d Impressions"),
                "Daily Avg Impr.": st.column_config.NumberColumn("Daily Avg Impr.", format="%.1f"),
            },
        )

        kw_list = topic_kw_df["Suggested Keyword"].tolist() if not topic_kw_df.empty else ad_row["queries_list"]
        c_ex, c_ph, c_ng = st.columns(3)
        with c_ex:
            st.markdown("**[Exact Match] Keywords (Ready to Paste)**")
            st.code("\n".join(f"[{q}]" for q in kw_list), language="text")
        with c_ph:
            st.markdown('**"Phrase Match" Keywords (Ready to Paste)**')
            st.code("\n".join(f'"{q}"' for q in kw_list), language="text")
        with c_ng:
            st.markdown("**Negative Keywords (Block Job Seekers)**")
            st.code(
                "-jobs\n-careers\n-hiring\n-salary\n-internship\n-glassdoor\n-indeed\n-jobstreet\n-free\n-template",
                language="text",
            )

        headlines, descriptions = build_rsa_assets(selected_ad_topic)
        ch, cd = st.columns(2)
        with ch:
            st.markdown("**Google Ads Headlines (≤ 30 chars)**")
            for cat, h in headlines:
                st.text(f"[{cat}] {h} ({len(h)}/30)")
        with cd:
            st.markdown("**Google Ads Descriptions (≤ 90 chars)**")
            for d in descriptions:
                st.text(f"{d} ({len(d)}/90)")

# ==============================================================================
# TAB 5: CONTENT PLAN & NEW TOPICS (CHECKED AGAINST CONNEXTGLOBAL.COM/SITEMAP.XML)
# ==============================================================================
with tab_content:
    st.subheader("1. Update Existing Sitemap Pages (Already Published on `connextglobal.com`)")
    st.markdown(
        """<div class="cnxt-cream-panel">
<strong>Anti-Duplicate Sitemap Protection:</strong> Every trending topic below was checked against <code>https://connextglobal.com/sitemap.xml</code>. When Connext already has a live page on that topic, it appears here so the Content team <strong>updates the existing page</strong> instead of creating a duplicate URL.
</div>""",
        unsafe_allow_html=True,
    )
    st.dataframe(
        existing_content_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "7d Impressions": st.column_config.NumberColumn("7d Impressions"),
            "Existing Sitemap Page": st.column_config.LinkColumn("Existing Sitemap Page"),
            "Main Service Page": st.column_config.LinkColumn("Main Service Page"),
        },
    )
    st.download_button(
        "Download Existing Page Updates CSV",
        existing_content_df.to_csv(index=False).encode("utf-8"),
        file_name="connext_existing_sitemap_page_updates.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader(f"2. Truly New Content Ideas — Verified Not in Sitemap ({len(truly_new_df)})")
    st.markdown(
        "Only topics with **no matching page in `connextglobal.com/sitemap.xml`** appear here. The closest existing sitemap page is shown side-by-side so you can confirm it is a genuine content gap."
    )
    if truly_new_df.empty:
        st.success("Every currently surging topic already has a matching page in `connextglobal.com/sitemap.xml`! Focus on updating the existing pages in Section 1 above.")
    else:
        st.dataframe(
            truly_new_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "7d Impressions": st.column_config.NumberColumn("7d Impressions"),
                "30d Impressions": st.column_config.NumberColumn("30d Impressions"),
                "Closest Existing Sitemap Page": st.column_config.LinkColumn("Closest Existing Sitemap Page"),
                "Service Page to Link To": st.column_config.LinkColumn("Service Page to Link To"),
            },
        )
        st.download_button(
            "Download Truly New Content Roadmap CSV",
            truly_new_df.to_csv(index=False).encode("utf-8"),
            file_name="connext_truly_new_content_gaps.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("3. One-Click Writer Brief")
    brief_topic = st.selectbox("Select Topic to Generate a Writer Brief:", existing_content_df["Trending Topic"].tolist())
    b_row = filtered_trends[filtered_trends["topic"] == brief_topic].iloc[0]

    lt_bullets = (
        "\n".join(f"  - `{q}`" for q in b_row["long_tail_list"])
        if b_row["long_tail_list"]
        else "\n".join(f"  - `{q}`" for q in b_row["queries_list"])
    )
    supporting_links = "\n".join(
        f"  - `{item['full_url']}` ({item['role']})" for item in b_row["matching_content_pages"]
    ) or "  - None matched in sitemap"

    action_headline = (
        f"Update Existing Live Page: `{b_row['closest_existing_url']}`"
        if b_row["already_exists_in_sitemap"]
        else "Create New Page (Verified Not in Sitemap)"
    )

    brief_md = f"""### Writer Brief: {b_row['topic'].title()} ({b_row['sector']})
- **Action:** {action_headline}
- **Current Google Rank:** **{b_row['google_rank_label']}**
- **Main Service Page to Link To:** `{b_row['recommended_money_page']}` ({b_row['best_money_role']})
- **Search Volume:** **{b_row['impressions_7d']:,}** 7-day impressions / **{b_row['impressions_30d']:,}** 30-day impressions (**+{b_row['growth_pct_3d']}%** vs. prior 2 weeks)
- **Related Live Pages in Sitemap to Interlink:**
{supporting_links}
- **Exact Search Questions & Phrases to Cover (H2s / FAQs):**
{lt_bullets}
"""
    st.markdown(brief_md)

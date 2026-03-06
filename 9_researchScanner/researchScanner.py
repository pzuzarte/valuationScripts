#!/usr/bin/env python3
"""
researchScanner.py — Emerging Technology Research Scanner
=========================================================
Scans public APIs to surface early signals of secular technology shifts
relevant to health-tech, electronic-tech, and tech-services investors.

Data sources (all free, no API key required):
  · openFDA          — Recent NDA/BLA drug approvals
  · ClinicalTrials   — Phase 2/3 trials (active, completed, positive results)
  · arXiv            — Trending AI, ML, quantum, biotech research papers
  · SEC EDGAR        — Recent S-1 IPO filings
  · USPTO PatentsView— Recent technology patents

Usage:
    python researchScanner.py [--days 60]
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import webbrowser
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
TODAY      = datetime.now(timezone.utc)
TIMEOUT    = 20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "researchData")
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "ValuationSuite/2.1 Research Scanner (educational)",
    "Accept":     "application/json",
}

# arXiv categories — tech-investor focus
ARXIV_CATS = [
    ("cs.AI",           "Artificial Intelligence"),
    ("cs.LG",           "Machine Learning"),
    ("cs.AR",           "Hardware Architecture"),
    ("cs.NE",           "Neural Computing"),
    ("cs.ET",           "Emerging Technologies"),
    ("eess.SP",         "Signal Processing"),
    ("quant-ph",        "Quantum Computing"),
    ("q-bio.BM",        "Biomolecules"),
    ("q-bio.GN",        "Genomics"),
    ("cond-mat.mes-hall", "Nanoscale Materials"),
]

CAT_LABEL = {c: l for c, l in ARXIV_CATS}

# USPTO CPC codes — technology areas to watch
CPC_AREAS = {
    "G06N":  "AI / Machine Learning",
    "H01L":  "Semiconductors",
    "H04W":  "Wireless / 5G",
    "H04L":  "Networking / Internet",
    "A61B":  "Medical Devices",
    "A61K":  "Pharma / Drug Delivery",
    "A61P":  "Therapeutic Agents",
    "G16H":  "Health Informatics",
    "G06F":  "Computing Systems",
    "B82Y":  "Nanotechnology",
}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url, timeout=TIMEOUT, extra_headers=None):
    h   = {**HEADERS, **(extra_headers or {})}
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), None
    except Exception as exc:
        return None, str(exc)


def _get_json(url, timeout=TIMEOUT, extra_headers=None):
    data, err = _get(url, timeout=timeout, extra_headers=extra_headers)
    if err:
        return None, err
    try:
        return json.loads(data), None
    except Exception as exc:
        return None, f"JSON parse: {exc}"


def _post_json(url, payload, timeout=TIMEOUT):
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=body, method="POST",
        headers={**HEADERS, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), None
    except Exception as exc:
        return None, str(exc)


def _ds(dt):
    """Date → YYYY-MM-DD string."""
    return dt.strftime("%Y-%m-%d")


def _safe(text, maxlen=130):
    if not text:
        return ""
    text = str(text).strip().replace("\n", " ").replace("\r", "")
    return (text[:maxlen] + "…") if len(text) > maxlen else text


def _fmt_date(raw, fmt_in="%Y-%m-%d", fmt_out="%b %d, %Y"):
    try:
        return datetime.strptime(raw, fmt_in).strftime(fmt_out)
    except Exception:
        return raw or "—"


# ── 1. FDA — Recent Drug Approvals ───────────────────────────────────────────

def fetch_fda_approvals(days=90):
    """Recent NDA / BLA approvals from openFDA drugsfda endpoint."""
    results = []
    start   = (TODAY - timedelta(days=days)).strftime("%Y%m%d")

    url = (
        "https://api.fda.gov/drug/drugsfda.json?"
        f"search=submissions.submission_status:AP"
        f"+AND+submissions.submission_status_date:[{start}+TO+29991231]"
        f"&sort=submissions.submission_status_date:desc"
        f"&limit=30"
    )
    data, err = _get_json(url)
    if err or not data:
        print(f"  ⚠  FDA API: {err}")
        return results

    for item in data.get("results", []):
        app_num  = item.get("application_number", "—")
        sponsor  = item.get("sponsor_name", "—") or "—"
        openfda  = item.get("openfda", {}) or {}
        brand    = (openfda.get("brand_name")   or ["—"])[0]
        generic  = (openfda.get("generic_name") or [""])[0].lower()
        pharm    = (openfda.get("pharm_class_epc") or [""])[0]

        ap_subs  = sorted(
            [s for s in (item.get("submissions") or []) if s.get("submission_status") == "AP"],
            key=lambda s: s.get("submission_status_date", ""), reverse=True,
        )
        if not ap_subs:
            continue
        latest      = ap_subs[0]
        date_raw    = latest.get("submission_status_date", "")
        if date_raw and date_raw < start:
            continue

        date_str    = _fmt_date(date_raw, "%Y%m%d", "%b %d, %Y") if date_raw else "—"
        sub_type    = latest.get("submission_type", "—")           # ORIG, SUPPL, BLA, …
        priority    = (latest.get("review_priority") or "standard").title()
        designation = latest.get("application_docs") and "Yes" or "—"

        results.append({
            "app_number": app_num,
            "brand":      brand.title(),
            "generic":    generic,
            "sponsor":    sponsor.title()[:40],
            "pharm":      _safe(pharm, 60),
            "type":       sub_type,
            "date":       date_str,
            "date_raw":   date_raw,
            "priority":   priority,
        })

    results.sort(key=lambda x: x["date_raw"], reverse=True)
    return results[:25]


# ── 2. Clinical Trials — Phase 2/3 ───────────────────────────────────────────

def fetch_clinical_trials(days=90):
    """Phase 2/3 interventional trials recently completed or active-not-recruiting.

    ClinicalTrials.gov API v2 note: phase filtering uses aggFilters, NOT filter.phase
    (filter.phase returns 400). Commas must NOT be percent-encoded in aggFilters.
    """
    results = []
    cutoff  = TODAY - timedelta(days=days)

    # Build URL manually — aggFilters commas must stay literal (not %2C)
    fields = ",".join([
        "NCTId", "BriefTitle", "OverallStatus", "Phase",
        "LeadSponsorName", "LeadSponsorClass",
        "StartDate", "CompletionDate", "LastUpdatePostDate",
        "Condition", "InterventionType", "InterventionName",
    ])
    url = (
        "https://clinicaltrials.gov/api/v2/studies?"
        "filter.overallStatus=COMPLETED,ACTIVE_NOT_RECRUITING"
        "&aggFilters=studyType:int,phase:2,phase:3"
        "&sort=LastUpdatePostDate:desc"
        "&pageSize=30"
        f"&fields={fields}"
    )
    data, err = _get_json(url, extra_headers={"Accept": "application/json"})
    if err or not data:
        print(f"  ⚠  ClinicalTrials API: {err}")
        return results

    for study in data.get("studies", []):
        proto   = study.get("protocolSection", {}) or {}
        id_m    = proto.get("identificationModule",        {}) or {}
        stat_m  = proto.get("statusModule",                {}) or {}
        spon_m  = proto.get("sponsorCollaboratorsModule",  {}) or {}
        cond_m  = proto.get("conditionsModule",            {}) or {}
        int_m   = proto.get("armsInterventionsModule",     {}) or {}
        out_m   = proto.get("outcomesModule",              {}) or {}

        nct_id   = id_m.get("nctId", "—")
        title    = _safe(id_m.get("briefTitle", "—"), 100)
        status   = (stat_m.get("overallStatus") or "—").replace("_", " ").title()
        phase    = (stat_m.get("phase") or "—").replace("_", " ")

        last_upd = (stat_m.get("lastUpdatePostDateStruct") or {}).get("date", "")
        try:
            upd_dt = datetime.strptime(last_upd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if upd_dt < cutoff:
                continue
        except Exception:
            pass

        completion = (stat_m.get("completionDateStruct") or {}).get("date", "—")
        sponsor    = (spon_m.get("leadSponsor") or {}).get("name", "—")
        spon_class = (spon_m.get("leadSponsor") or {}).get("class", "")     # INDUSTRY / NIH / OTHER

        conditions   = (cond_m.get("conditions") or [])[:3]
        cond_str     = ", ".join(conditions) or "—"

        interventions = (int_m.get("interventions") or [])
        types         = list({i.get("type", "") for i in interventions if i.get("type")})[:2]
        int_str       = ", ".join(t.title() for t in types) or "—"

        primary_out = _safe(
            ((out_m.get("primaryOutcomes") or [{}])[0]).get("measure", ""), 80
        )

        results.append({
            "nct_id":      nct_id,
            "title":       title,
            "status":      status,
            "phase":       phase,
            "sponsor":     sponsor[:50],
            "spon_class":  spon_class,
            "conditions":  cond_str,
            "intervention":int_str,
            "completion":  completion or "—",
            "last_updated":last_upd or "—",
            "primary_out": primary_out,
            "url":         f"https://clinicaltrials.gov/study/{nct_id}",
        })

    return results[:25]


# ── 3. arXiv — Research Papers ───────────────────────────────────────────────

def fetch_arxiv_papers(days=30):
    """Recent arXiv papers in AI, ML, quantum, biotech."""
    results = []
    cutoff  = TODAY - timedelta(days=days)

    cat_q = "+OR+".join(f"cat:{c}" for c, _ in ARXIV_CATS[:8])
    url   = (
        "https://export.arxiv.org/api/query?"
        f"search_query=({cat_q})"
        "&sortBy=submittedDate&sortOrder=descending"
        "&max_results=50&start=0"
    )
    raw, err = _get(url, extra_headers={"Accept": "application/atom+xml"})
    if err or not raw:
        print(f"  ⚠  arXiv API: {err}")
        return results

    try:
        ns   = {"atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(raw)

        for entry in root.findall("atom:entry", ns):
            pub_str = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
            try:
                pub_dt = datetime.strptime(pub_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
            except Exception:
                pass

            title   = _safe(entry.findtext("atom:title",   default="", namespaces=ns).strip(), 120)
            summary = _safe(entry.findtext("atom:summary", default="", namespaces=ns).strip(), 220)
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            link    = link_el.get("href", "") if link_el is not None else ""

            authors_el = entry.findall("atom:author/atom:name", ns)
            authors    = [a.text for a in authors_el[:3] if a.text]
            author_str = ", ".join(authors) + (" et al." if len(authors_el) > 3 else "")

            pri_cat = entry.find("arxiv:primary_category", ns)
            cat_code  = (pri_cat.get("term", "") if pri_cat is not None else "").split(".")[0] + (
                "." + pri_cat.get("term", "").split(".")[1] if "." in (pri_cat.get("term","") if pri_cat is not None else "") else ""
            ) if pri_cat is not None else ""
            cat_label = CAT_LABEL.get(cat_code, cat_code)

            results.append({
                "title":     title,
                "authors":   author_str or "—",
                "category":  cat_label,
                "cat_code":  cat_code,
                "published": pub_str,
                "summary":   summary,
                "url":       link,
            })
    except Exception as exc:
        print(f"  ⚠  arXiv parse: {exc}")

    return results[:30]


# ── 4. SEC EDGAR — S-1 IPO Filings ───────────────────────────────────────────

# SIC codes relevant to tech, health-tech, biotech, medtech
_TECH_SICS = {
    "7371", "7372", "7373", "7374", "7375", "7376", "7377", "7379",  # Software / IT services
    "3674", "3669", "3577", "3672", "3679", "3661", "3663", "3812",  # Electronic hardware
    "3841", "3845", "3826", "3827", "3829",                           # Medical instruments
    "2836", "2835", "2833", "2830",                                   # Pharma / biologics
    "8731", "8099", "8000", "8011",                                   # R&D / health services
    "3559", "3825",                                                   # Industrial instruments
}


def _parse_display_name(raw):
    """Extract company name from EDGAR display_names entry.
    Format: 'Company Name  (TICKER)  (CIK 0001234567)'
    or:     'Company Name  (CIK 0001234567)'
    """
    if not raw:
        return "—"
    # Strip CIK suffix: ' (CIK 0001234567)'
    import re
    name = re.sub(r'\s*\(CIK\s+\d+\)', '', str(raw)).strip()
    # Strip ticker suffix: ' (TICK)'
    name = re.sub(r'\s*\([A-Z]{1,5}\)\s*$', '', name).strip()
    return name or "—"


def fetch_ipo_pipeline(days=90):
    """Recent S-1 registration statements from SEC EDGAR, filtered for tech/biotech."""
    results = []
    start   = _ds(TODAY - timedelta(days=days))

    url = (
        "https://efts.sec.gov/LATEST/search-index?"
        + urllib.parse.urlencode({
            "forms":     "S-1",
            "dateRange": "custom",
            "startdt":   start,
        })
    )
    data, err = _get_json(url)
    if err or not data:
        print(f"  ⚠  EDGAR API: {err}")
        return results

    for hit in (data.get("hits", {}).get("hits", [])) or []:
        src       = hit.get("_source", {}) or {}
        form_type = src.get("form", "") or src.get("file_type", "S-1")

        # Skip amendments (S-1/A) — only show original filings
        if form_type == "S-1/A":
            continue

        # Parse company name from display_names list
        display   = src.get("display_names") or []
        raw_name  = display[0] if display else ""
        company   = _safe(_parse_display_name(raw_name), 60)

        file_date = src.get("file_date", "—")
        sics      = src.get("sics") or []
        sic_str   = sics[0] if sics else ""
        location  = (src.get("biz_locations") or [""])[0]

        # Filter: only tech / biotech / health-tech SIC codes
        if sic_str and sic_str not in _TECH_SICS:
            continue

        # Build EDGAR URL from CIK
        ciks = src.get("ciks") or []
        cik  = (ciks[0] or "").lstrip("0") if ciks else ""
        try:
            if cik:
                edgar_url = (
                    f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                    f"&CIK={urllib.parse.quote(cik, safe='')}"
                    f"&type=S-1&dateb=&owner=include&count=10"
                )
            else:
                edgar_url = (
                    f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                    f"&company={urllib.parse.quote(company, safe='')}"
                    f"&type=S-1&dateb=&owner=include&count=10"
                )
        except Exception:
            edgar_url = "https://www.sec.gov/cgi-bin/browse-edgar?type=S-1"

        results.append({
            "company":   company,
            "form":      form_type,
            "sic":       sic_str,
            "location":  location,
            "filed":     _fmt_date(file_date),
            "filed_raw": file_date,
            "url":       edgar_url,
        })

    results.sort(key=lambda x: x["filed_raw"], reverse=True)
    return results[:30]


# ── 5. NIH Research Grants ────────────────────────────────────────────────────
# (PatentsView API v1 now requires a free API key registration at
#  patentsview.org/apis/  — add your key to the PATENTSVIEW_KEY constant
#  above when available; until then this section shows NIH grant data.)

PATENTSVIEW_KEY = ""   # optional: paste your free PatentsView API key here

NIH_TECH_TERMS = [
    "artificial intelligence",
    "machine learning",
    "CRISPR gene editing",
    "mRNA therapy",
    "CAR-T cell therapy",
    "digital health",
    "medical device",
    "immunotherapy",
    "nanotechnology",
    "quantum computing",
]


def fetch_patents(days=60):
    """
    Tries PatentsView API first (requires free API key).
    Falls back to NIH research grants as a tech-investment signal proxy.
    """
    # ── PatentsView (if key is configured) ──────────────────────────────────
    if PATENTSVIEW_KEY:
        return _fetch_patentsview(days)

    # ── NIH Reporter fallback ────────────────────────────────────────────────
    return _fetch_nih_grants(days)


def _fetch_patentsview(days):
    """Fetch recent tech patents via PatentsView API v1 (requires API key)."""
    results = []
    start   = _ds(TODAY - timedelta(days=days))
    query   = {"_and": [
        {"_gte": {"patent_date": start}},
        {"_or": [{"_begins": {"cpcs.cpc_group_id": code}} for code in CPC_AREAS]},
    ]}
    fields  = ["patent_number", "patent_title", "patent_date",
               "assignees.assignee_organization", "cpcs.cpc_group_id"]
    url = (
        "https://search.patentsview.org/api/v1/patent/?"
        + urllib.parse.urlencode({
            "q": json.dumps(query), "f": json.dumps(fields),
            "s": json.dumps([{"patent_date": "desc"}]), "per_page": "30",
        })
    )
    data, err = _get_json(url, extra_headers={"X-Api-Key": PATENTSVIEW_KEY})
    if err or not data:
        print(f"  ⚠  PatentsView: {err}")
        return results
    for patent in (data.get("patents") or []):
        num      = patent.get("patent_number", "—")
        titl     = _safe(patent.get("patent_title", "—"), 110)
        date     = patent.get("patent_date", "—")
        assignees = patent.get("assignees") or []
        assignee = (assignees[0].get("assignee_organization") or "Individual") if assignees else "Individual"
        cpcs     = patent.get("cpcs") or []
        cpc_id   = (cpcs[0].get("cpc_group_id") or "") if cpcs else ""
        area     = next((lbl for code, lbl in CPC_AREAS.items() if cpc_id.startswith(code[:4])), "Technology")
        results.append({"number": num, "title": titl, "date": _fmt_date(date),
                        "date_raw": date, "assignee": (_safe(assignee, 45) or "Individual").title(),
                        "area": area, "cpc": cpc_id, "source": "patent",
                        "url": f"https://patents.google.com/patent/US{num}"})
    return results[:25]


def _fetch_nih_grants(days=60):
    """NIH research grants as a proxy for emerging tech investment signals."""
    results = []
    current_year = TODAY.year
    fiscal_years = list({current_year - 1, current_year})
    seen_ids     = set()

    for term in NIH_TECH_TERMS[:6]:        # limit to top 6 terms to avoid rate limits
        payload = {
            "criteria": {
                "fiscal_years": fiscal_years,
                "activity_codes": ["R01", "R35", "P01", "U01", "R21", "DP2"],
                "advanced_text_search": {
                    "operator": "and",
                    "search_field": "all",
                    "search_text": term,
                },
            },
            "include_fields": [
                "ProjectNum", "ProjectTitle", "OrgName", "PIName",
                "FiscalYear", "AwardAmount", "AbstractText",
            ],
            "offset": 0, "limit": 5,
            "sort_field": "award_amount", "sort_order": "desc",
        }
        data, err = _post_json("https://api.reporter.nih.gov/v2/projects/search", payload)
        if err or not data:
            continue
        for proj in (data.get("results") or []):
            pnum   = proj.get("project_num", "")
            amount = proj.get("award_amount") or 0
            if amount < 200_000 or pnum in seen_ids:
                continue
            seen_ids.add(pnum)
            pi_raw = proj.get("pi_name") or ""
            if isinstance(pi_raw, list):
                pi_raw = pi_raw[0] if pi_raw else ""
            results.append({
                "number":   pnum,
                "title":    _safe(proj.get("project_title", "—"), 110),
                "date":     str(proj.get("fiscal_year", "—")),
                "date_raw": str(proj.get("fiscal_year", "0")),
                "assignee": _safe(proj.get("org_name", "—"), 45),
                "area":     term.title(),
                "cpc":      f"${amount:,.0f}",
                "source":   "grant",
                "url":      f"https://reporter.nih.gov/project-details/{pnum}" if pnum else
                            "https://reporter.nih.gov/",
            })

    results.sort(key=lambda x: x.get("date_raw", ""), reverse=True)
    return results[:25]


# ── HTML builder ─────────────────────────────────────────────────────────────

def _pill(text, color="#4f8ef7"):
    return (f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
            f'border-radius:4px;padding:2px 8px;font-size:11px;white-space:nowrap">'
            f'{text}</span>')


def _priority_pill(p):
    c = "#f0a500" if "Priority" in p else "#8b949e"
    return _pill(p, c)


def _status_pill(s):
    if "Completed" in s:
        return _pill(s, "#00c896")
    if "Active" in s:
        return _pill(s, "#f0a500")
    return _pill(s, "#8b949e")


def _cat_pill(cat):
    CAT_COLORS = {
        "Artificial Intelligence": "#4f8ef7",
        "Machine Learning":        "#4f8ef7",
        "Hardware Architecture":   "#a78bfa",
        "Neural Computing":        "#4f8ef7",
        "Quantum Computing":       "#c084fc",
        "Signal Processing":       "#34d399",
        "Biomolecules":            "#f87171",
        "Genomics":                "#f87171",
        "Nanoscale Materials":     "#fbbf24",
        "Emerging Technologies":   "#00c896",
    }
    c = CAT_COLORS.get(cat, "#8b949e")
    return _pill(cat, c)


def _area_pill(area):
    AREA_COLORS = {
        "AI / Machine Learning":  "#4f8ef7",
        "Semiconductors":          "#a78bfa",
        "Wireless / 5G":           "#34d399",
        "Networking / Internet":   "#00c896",
        "Medical Devices":         "#f87171",
        "Pharma / Drug Delivery":  "#f87171",
        "Therapeutic Agents":      "#f87171",
        "Health Informatics":      "#f59e0b",
        "Computing Systems":       "#4f8ef7",
        "Nanotechnology":          "#fbbf24",
    }
    c = AREA_COLORS.get(area, "#8b949e")
    return _pill(area, c)


def _th(text, align="left"):
    return f'<th style="text-align:{align};padding:10px 12px;white-space:nowrap">{text}</th>'


def _td(content, align="left", extra=""):
    return f'<td style="text-align:{align};padding:9px 12px;{extra}">{content}</td>'


def _ext_link(url, label):
    return (f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#4f8ef7;text-decoration:none;font-size:12px">'
            f'{label} ↗</a>')


def _table_wrap(headers, rows, empty_msg="No data available for this period."):
    if not rows:
        return f'<p style="color:#8b949e;padding:20px">{empty_msg}</p>'
    thead = "<tr>" + "".join(headers) + "</tr>"
    tbody = "\n".join("<tr>" + r + "</tr>" for r in rows)
    return (
        '<div style="overflow-x:auto">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        f'<thead style="background:#161b22;color:#8b949e;font-size:11px;'
        f'letter-spacing:.05em;font-weight:700;text-transform:uppercase">'
        f'{thead}</thead>'
        f'<tbody style="color:#e6edf3">{tbody}</tbody>'
        '</table></div>'
    )


# ── Section builders ─────────────────────────────────────────────────────────

def _fda_tab(fda):
    headers = [
        _th("Drug (Brand)"), _th("Generic / Class"), _th("Sponsor"),
        _th("App #"), _th("Type"), _th("Priority", "center"),
        _th("Approved", "center"),
    ]
    rows = []
    for r in fda:
        link = (f'<a href="https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?'
                f'event=overview.process&ApplNo={r["app_number"].replace("NDA","").replace("BLA","").zfill(6)}" '
                f'target="_blank" rel="noopener noreferrer" style="color:#4f8ef7;text-decoration:none">'
                f'<b>{r["brand"]}</b></a>')
        generic_cell = f'<span style="color:#8b949e;font-size:12px">{r["generic"]}</span>'
        rows.append(
            _td(link) +
            _td(generic_cell) +
            _td(r["sponsor"], extra="color:#8b949e;font-size:12px") +
            _td(r["app_number"], extra="color:#8b949e;font-size:12px;font-family:monospace") +
            _td(r["type"],   extra="color:#8b949e;font-size:12px") +
            _td(_priority_pill(r["priority"]), "center") +
            _td(r["date"],  "center", "color:#8b949e;font-size:12px;white-space:nowrap")
        )
    return (
        '<div style="margin-bottom:16px;color:#8b949e;font-size:13px">'
        'Recent NDA &amp; BLA drug approvals from the FDA. '
        '<b style="color:#e6edf3">Priority Review</b> = FDA designated accelerated review (6-month vs standard 12-month).</div>'
        + _table_wrap(headers, rows, "No FDA approvals found for this period.")
    )


def _trials_tab(trials):
    headers = [
        _th("Trial Title"), _th("Sponsor"), _th("Conditions"),
        _th("Phase", "center"), _th("Intervention", "center"),
        _th("Status", "center"), _th("Completion", "center"),
    ]
    rows = []
    for r in trials:
        title_link = (
            f'<a href="{r["url"]}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#e6edf3;text-decoration:none;font-size:12px">'
            f'{r["title"]}</a>'
            + (f'<br><span style="color:#8b949e;font-size:11px">{r["nct_id"]}</span>'
               if r["nct_id"] != "—" else "")
        )
        industry_badge = (
            ' <span style="background:#4f8ef722;color:#4f8ef7;border:1px solid #4f8ef744;'
            'border-radius:3px;padding:1px 5px;font-size:10px">Industry</span>'
            if r["spon_class"] == "INDUSTRY" else ""
        )
        rows.append(
            _td(title_link) +
            _td(f'{r["sponsor"]}{industry_badge}', extra="font-size:12px;color:#8b949e") +
            _td(f'<span style="font-size:12px;color:#8b949e">{r["conditions"]}</span>') +
            _td(_pill(r["phase"].replace("Phase ", "Ph "), "#a78bfa"), "center") +
            _td(f'<span style="font-size:12px;color:#8b949e">{r["intervention"]}</span>', "center") +
            _td(_status_pill(r["status"]), "center") +
            _td(r["completion"], "center", "color:#8b949e;font-size:12px;white-space:nowrap")
        )
    return (
        '<div style="margin-bottom:16px;color:#8b949e;font-size:13px">'
        'Phase 2/3 interventional trials that are active (not yet recruiting) or recently completed. '
        '<b style="color:#e6edf3">Industry-sponsored</b> trials are most relevant for investors.</div>'
        + _table_wrap(headers, rows, "No clinical trials found for this period.")
    )


def _papers_tab(papers):
    if not papers:
        return '<p style="color:#8b949e;padding:20px">No papers found for this period.</p>'
    cards = []
    for p in papers:
        date_fmt = _fmt_date(p["published"], "%Y-%m-%d", "%b %d")
        cards.append(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
            f'padding:16px;display:flex;flex-direction:column;gap:8px;min-height:160px">'
            f'<div style="display:flex;align-items:flex-start;gap:8px;justify-content:space-between">'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap">{_cat_pill(p["category"])}</div>'
            f'<span style="color:#8b949e;font-size:11px;white-space:nowrap">{date_fmt}</span>'
            f'</div>'
            f'<a href="{p["url"]}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#e6edf3;text-decoration:none;font-size:13px;font-weight:600;line-height:1.4">'
            f'{p["title"]}</a>'
            f'<p style="color:#8b949e;font-size:12px;line-height:1.5;margin:0">{p["summary"]}</p>'
            f'<span style="color:#6b7194;font-size:11px">{p["authors"]}</span>'
            f'</div>'
        )
    return (
        '<div style="margin-bottom:16px;color:#8b949e;font-size:13px">'
        'Latest preprints in AI, ML, quantum computing, and biotech from arXiv — '
        'often 2–5 years ahead of commercial applications.</div>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px">'
        + "\n".join(cards) + '</div>'
    )


def _ipo_tab(ipos):
    headers = [
        _th("Company"), _th("Location"), _th("SIC", "center"),
        _th("Filed", "center"), _th("Link", "center"),
    ]
    rows = []
    for r in ipos:
        company_cell = f'<b style="color:#e6edf3">{r["company"]}</b>'
        rows.append(
            _td(company_cell) +
            _td(r.get("location", "—"), extra="color:#8b949e;font-size:12px") +
            _td(f'<span style="font-family:monospace;font-size:11px;color:#6b7194">{r.get("sic","—")}</span>', "center") +
            _td(r["filed"], "center", "color:#8b949e;font-size:12px;white-space:nowrap") +
            _td(_ext_link(r["url"], "EDGAR"), "center")
        )
    return (
        '<div style="margin-bottom:16px;color:#8b949e;font-size:13px">'
        'Original S-1 IPO registration statements filed with the SEC — filtered for tech, biotech, '
        'and health-tech companies by SIC code. Most S-1s are filed 3–12 months before listing.</div>'
        + _table_wrap(headers, rows, "No tech/biotech S-1 filings found for this period.")
    )


def _patents_tab(patents):
    is_grants = patents and patents[0].get("source") == "grant"

    if is_grants:
        headers = [
            _th("Project Title"), _th("Institution"), _th("Focus Area", "center"),
            _th("Award", "right"), _th("FY", "center"), _th("Link", "center"),
        ]
        rows = []
        for r in patents:
            title_cell = f'<span style="font-size:12px;color:#e6edf3">{r["title"]}</span>'
            rows.append(
                _td(title_cell) +
                _td(r["assignee"], extra="color:#8b949e;font-size:12px") +
                _td(_area_pill(r["area"]), "center") +
                _td(r["cpc"], "right", "color:#00c896;font-size:12px;font-family:monospace") +
                _td(r["date"], "center", "color:#8b949e;font-size:12px") +
                _td(_ext_link(r["url"], "NIH Reporter"), "center")
            )
        key_note = (
            '<div style="margin-bottom:16px;color:#8b949e;font-size:13px">'
            'NIH research grant awards by technology focus area — a strong leading indicator of '
            'where biomedical and tech-adjacent science is being funded. '
            'Showing top awards by dollar amount. '
            '<span style="color:#f0a500">Patent data via PatentsView requires a free API key — '
            'add it to the <code>PATENTSVIEW_KEY</code> constant in researchScanner.py to enable.</span>'
            '</div>'
        )
        return key_note + _table_wrap(headers, rows, "No NIH grant data found.")
    else:
        headers = [
            _th("Patent Title"), _th("Assignee"), _th("Tech Area", "center"),
            _th("CPC", "center"), _th("Granted", "center"), _th("Link", "center"),
        ]
        rows = []
        for r in patents:
            title_cell = f'<span style="font-size:12px;color:#e6edf3">{r["title"]}</span>'
            rows.append(
                _td(title_cell) +
                _td(r["assignee"], extra="color:#8b949e;font-size:12px") +
                _td(_area_pill(r["area"]), "center") +
                _td(f'<span style="font-family:monospace;font-size:11px;color:#6b7194">{r["cpc"]}</span>', "center") +
                _td(r["date"], "center", "color:#8b949e;font-size:12px;white-space:nowrap") +
                _td(_ext_link(r["url"], "Google Patents"), "center")
            )
        return (
            '<div style="margin-bottom:16px;color:#8b949e;font-size:13px">'
            'Recent USPTO utility patents in AI/ML, semiconductors, wireless, medical devices, and biotech. '
            'High-volume patent activity by a company often precedes commercial announcements.</div>'
            + _table_wrap(headers, rows, "No patents found for this period.")
        )


def _overview_tab(fda, trials, papers, ipos, patents, days):
    def stat_card(icon, count, label, sub):
        return (
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
            f'padding:20px 24px;display:flex;flex-direction:column;gap:6px">'
            f'<div style="font-size:26px">{icon}</div>'
            f'<div style="font-size:32px;font-weight:800;color:#e6edf3">{count}</div>'
            f'<div style="font-size:14px;font-weight:600;color:#e6edf3">{label}</div>'
            f'<div style="font-size:12px;color:#8b949e">{sub}</div>'
            f'</div>'
        )

    stat_grid = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:28px">'
        + stat_card("💊", len(fda),     "FDA Approvals",   f"NDA/BLA, last {days} days")
        + stat_card("🧬", len(trials),  "Clinical Trials", f"Phase 2/3 active/completed")
        + stat_card("📄", len(papers),  "Research Papers", f"arXiv preprints, last {days} days")
        + stat_card("📋", len(ipos),    "S-1 Filings",     f"IPO pipeline, last {days} days")
        + stat_card("🔒", len(patents), "Tech Patents",    f"USPTO grants, last {days} days")
        + '</div>'
    )

    def signal_section(title, icon, items, fmt_fn, max_items=5):
        if not items:
            return f'<div style="color:#8b949e;font-size:13px">No {title.lower()} data.</div>'
        html = (
            f'<h3 style="font-size:14px;font-weight:700;color:#e6edf3;margin:0 0 12px">'
            f'{icon} {title}</h3>'
            '<div style="display:flex;flex-direction:column;gap:8px">'
        )
        for item in items[:max_items]:
            html += fmt_fn(item)
        html += '</div>'
        return html

    def fmt_fda(r):
        return (
            f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
            f'border-bottom:1px solid #30363d">'
            f'{_priority_pill(r["priority"])}'
            f'<span style="color:#e6edf3;font-size:13px;font-weight:600">{r["brand"]}</span>'
            f'<span style="color:#8b949e;font-size:12px">— {r["sponsor"]}</span>'
            f'<span style="color:#8b949e;font-size:12px;margin-left:auto">{r["date"]}</span>'
            f'</div>'
        )

    def fmt_trial(r):
        return (
            f'<div style="padding:8px 0;border-bottom:1px solid #30363d">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
            f'{_status_pill(r["status"])}'
            f'<span style="color:#e6edf3;font-size:13px">{r["title"]}</span>'
            f'</div>'
            f'<span style="color:#8b949e;font-size:12px">{r["sponsor"]} · {r["conditions"]}</span>'
            f'</div>'
        )

    def fmt_paper(r):
        return (
            f'<div style="padding:8px 0;border-bottom:1px solid #30363d">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
            f'{_cat_pill(r["category"])}'
            f'<span style="color:#e6edf3;font-size:13px">'
            f'<a href="{r["url"]}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#e6edf3;text-decoration:none">{r["title"]}</a></span>'
            f'</div>'
            f'<span style="color:#8b949e;font-size:12px">{r["authors"]}</span>'
            f'</div>'
        )

    def fmt_ipo(r):
        return (
            f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
            f'border-bottom:1px solid #30363d">'
            f'<span style="color:#e6edf3;font-size:13px;font-weight:600">{r["company"]}</span>'
            f'<span style="color:#8b949e;font-size:12px">filed {r["filed"]}</span>'
            f'<a href="{r["url"]}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#4f8ef7;font-size:12px;margin-left:auto">EDGAR ↗</a>'
            f'</div>'
        )

    def fmt_patent(r):
        return (
            f'<div style="padding:8px 0;border-bottom:1px solid #30363d">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
            f'{_area_pill(r["area"])}'
            f'<span style="color:#e6edf3;font-size:13px">'
            f'<a href="{r["url"]}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#e6edf3;text-decoration:none">{r["title"]}</a></span>'
            f'</div>'
            f'<span style="color:#8b949e;font-size:12px">{r["assignee"]} · {r["date"]}</span>'
            f'</div>'
        )

    two_col = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">'
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px">'
        f'{signal_section("Recent FDA Approvals", "💊", fda, fmt_fda)}'
        f'</div>'
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px">'
        f'{signal_section("Phase 3 Trials", "🧬", trials, fmt_trial)}'
        f'</div>'
        '</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">'
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px">'
        f'{signal_section("Trending Research", "📄", papers, fmt_paper)}'
        f'</div>'
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px">'
        f'{signal_section("IPO Pipeline", "📋", ipos, fmt_ipo)}'
        f'</div>'
        '</div>'
        '<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:20px">'
        f'{signal_section("Patent Activity", "🔒", patents, fmt_patent, 6)}'
        '</div>'
    )

    return stat_grid + two_col


# ── Full HTML ─────────────────────────────────────────────────────────────────

def build_html(fda, trials, papers, ipos, patents, days):
    gen_date = datetime.now().strftime("%B %d, %Y  %H:%M")
    ts       = datetime.now().strftime("%Y_%m_%d")

    is_grants   = patents and patents[0].get("source") == "grant"
    patent_label = "NIH Grants" if is_grants else "Patent Watch"

    TABS = [
        ("overview",  "Overview",         _overview_tab(fda, trials, papers, ipos, patents, days)),
        ("fda",       "FDA Pipeline",      _fda_tab(fda)),
        ("trials",    "Clinical Trials",   _trials_tab(trials)),
        ("papers",    "Research Papers",   _papers_tab(papers)),
        ("ipo",       "IPO Pipeline",      _ipo_tab(ipos)),
        ("patents",   patent_label,        _patents_tab(patents)),
    ]

    BADGE_COUNTS = {
        "overview": len(fda) + len(trials) + len(papers) + len(ipos) + len(patents),
        "fda":      len(fda),
        "trials":   len(trials),
        "papers":   len(papers),
        "ipo":      len(ipos),
        "patents":  len(patents),
    }

    def tab_btn(tid, label):
        badge = BADGE_COUNTS.get(tid, 0)
        badge_html = (
            f' <span class="tab-badge" id="badge-{tid}">{badge}</span>'
            if badge > 0 else ""
        )
        return (
            f'<button class="tab-btn" id="tb-{tid}" onclick="showTab(\'{tid}\')">'
            f'{label}{badge_html}</button>'
        )

    def tab_content(tid, content):
        return (
            f'<div class="tab-pane" id="tp-{tid}" style="display:none">'
            f'{content}'
            f'</div>'
        )

    tab_btns    = "\n".join(tab_btn(tid, label)   for tid, label, _ in TABS)
    tab_panes   = "\n".join(tab_content(tid, html) for tid, _, html  in TABS)
    first_tab   = TABS[0][0]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Scanner — {gen_date}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:     #0e1117; --panel:  #161b22; --card:   #1c2128;
    --border: #30363d; --accent: #4f8ef7; --green:  #00c896;
    --muted:  #8b949e; --text:   #e6edf3; --red:    #f85149;
    --orange: #f0a500; --radius: 8px;
  }}
  html, body {{ min-height:100%; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .topbar {{
    background:var(--panel); border-bottom:1px solid var(--border);
    padding:0 28px; height:54px; display:flex; align-items:center; gap:14px;
    position:sticky; top:0; z-index:100;
  }}
  .topbar h1 {{ font-size:17px; font-weight:700; flex:1; }}
  .topbar .meta {{ font-size:12px; color:var(--muted); }}
  .badge-lookback {{
    background:#00c89622; color:#00c896; border:1px solid #00c89644;
    border-radius:4px; padding:2px 10px; font-size:12px; font-weight:600;
  }}
  .main {{ max-width:1400px; margin:0 auto; padding:24px 28px 40px; }}
  /* Tabs */
  .tab-bar {{
    display:flex; gap:4px; border-bottom:1px solid var(--border);
    margin-bottom:24px; overflow-x:auto; padding-bottom:0;
  }}
  .tab-btn {{
    background:transparent; border:none; color:var(--muted);
    font-size:13px; font-weight:600; padding:10px 18px 11px;
    cursor:pointer; border-bottom:2px solid transparent;
    transition:color .15s,border-color .15s; white-space:nowrap;
    display:flex; align-items:center; gap:6px;
  }}
  .tab-btn:hover {{ color:var(--text); }}
  .tab-btn.active {{ color:var(--green); border-bottom-color:var(--green); }}
  .tab-badge {{
    background:var(--green); color:#000; font-size:10px; font-weight:800;
    border-radius:10px; padding:1px 6px; min-width:18px; text-align:center;
  }}
  /* Table zebra */
  tbody tr:nth-child(even) {{ background:#12171d; }}
  tbody tr:hover {{ background:#1c2128; }}
  tbody td {{ border-bottom:1px solid #1e2530; vertical-align:middle; }}
  thead th {{ border-bottom:1px solid var(--border); }}
  /* Responsive grid fallback */
  @media (max-width:800px) {{
    .two-col {{ grid-template-columns:1fr !important; }}
  }}
</style>
</head>
<body>

<div class="topbar">
  <h1>🔭&nbsp; Emerging Technology Research Scanner</h1>
  <span class="badge-lookback">Last {days} days</span>
  <span class="meta">Generated {gen_date}</span>
</div>

<div class="main">
  <div class="tab-bar">
    {tab_btns}
  </div>
  {tab_panes}
</div>

<script>
function showTab(id) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const pane = document.getElementById('tp-' + id);
  const btn  = document.getElementById('tb-' + id);
  if (pane) pane.style.display = 'block';
  if (btn)  btn.classList.add('active');
}}
showTab('{first_tab}');
</script>
</body>
</html>""", ts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emerging Technology Research Scanner")
    parser.add_argument("--days", type=int, default=60,
                        help="Lookback window in days (default: 60)")
    args = parser.parse_args()

    days = max(7, min(365, args.days))

    print(f"\n  Emerging Technology Research Scanner")
    print(f"  Lookback: {days} days  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  " + "─" * 50)

    results = {}

    tasks = {
        "fda":     (fetch_fda_approvals,  (days,)),
        "trials":  (fetch_clinical_trials, (days,)),
        "papers":  (fetch_arxiv_papers,   (days,)),
        "ipos":    (fetch_ipo_pipeline,   (days,)),
        "patents": (fetch_patents,         (days,)),
    }

    print("  Fetching data in parallel…")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn, *a): key for key, (fn, a) in tasks.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                results[key] = fut.result()
                n = len(results[key])
                label = {
                    "fda":     "FDA approvals",
                    "trials":  "clinical trials",
                    "papers":  "arXiv papers",
                    "ipos":    "S-1 filings",
                    "patents": "patents",
                }[key]
                print(f"    ✓  {label}: {n} results")
            except Exception as exc:
                results[key] = []
                print(f"    ✗  {key}: {exc}")

    elapsed = time.time() - t0
    print(f"  Fetch complete in {elapsed:.1f}s")
    print("  " + "─" * 50)

    fda     = results.get("fda",     [])
    trials  = results.get("trials",  [])
    papers  = results.get("papers",  [])
    ipos    = results.get("ipos",    [])
    patents = results.get("patents", [])

    print("  Building HTML report…")
    html_str, ts = build_html(fda, trials, papers, ipos, patents, days)

    fname = f"{ts}_research.html"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"  Saved → {fpath}")
    webbrowser.open(f"file://{fpath}")
    print("  ✓  Done\n")


if __name__ == "__main__":
    main()

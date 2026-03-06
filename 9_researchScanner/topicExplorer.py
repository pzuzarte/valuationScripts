#!/usr/bin/env python3
"""
topicExplorer.py — Dynamic Research Topic Explorer
====================================================
Opens an interactive browser UI where the user selects specific technology
topics, then fetches live data from arXiv, NIH, and ClinicalTrials filtered
to those exact topics.

Requires the ValuationSuite Flask server (app.py) to be running for data
fetching. The HTML UI is generated locally and makes AJAX calls to Flask.

Usage:
    python topicExplorer.py [--days 30]
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

# ── Topic taxonomy ────────────────────────────────────────────────────────────
# Each topic defines its data-source mappings:
#   arxiv_cats  : arXiv category codes to include
#   arxiv_kw    : keywords searched in title+abstract (any match)
#   nih_terms   : NIH Reporter search terms (each fetched separately)
#   ct_query    : ClinicalTrials.gov free-text query (intervention or condition)
#   color       : accent color for UI pill

TOPICS = {
    # ── Artificial Intelligence ───────────────────────────────────────────────
    "llm": {
        "label":      "Large Language Models",
        "category":   "Artificial Intelligence",
        "icon":       "🧠",
        "arxiv_cats": ["cs.CL", "cs.AI", "cs.LG"],
        "arxiv_kw":   ["large language model", "LLM", "GPT", "transformer", "foundation model"],
        "nih_terms":  ["large language model", "natural language processing clinical"],
        "ct_query":   "artificial intelligence language",
        "color":      "#4f8ef7",
    },
    "computer_vision": {
        "label":      "Computer Vision",
        "category":   "Artificial Intelligence",
        "icon":       "👁",
        "arxiv_cats": ["cs.CV", "cs.AI"],
        "arxiv_kw":   ["image recognition", "object detection", "vision transformer", "diffusion model"],
        "nih_terms":  ["computer vision medical imaging", "deep learning image analysis"],
        "ct_query":   "computer vision imaging artificial intelligence",
        "color":      "#4f8ef7",
    },
    "robotics": {
        "label":      "Robotics & Automation",
        "category":   "Artificial Intelligence",
        "icon":       "🤖",
        "arxiv_cats": ["cs.RO", "cs.AI", "cs.SY"],
        "arxiv_kw":   ["robot", "autonomous system", "reinforcement learning manipulation"],
        "nih_terms":  ["surgical robotics", "robotic assisted surgery"],
        "ct_query":   "robotic surgery automation",
        "color":      "#4f8ef7",
    },
    "ai_chips": {
        "label":      "AI Chips & Hardware",
        "category":   "Artificial Intelligence",
        "icon":       "⚡",
        "arxiv_cats": ["cs.AR", "eess.SP", "cs.ET"],
        "arxiv_kw":   ["neural processing unit", "NPU", "accelerator", "neuromorphic", "in-memory computing"],
        "nih_terms":  ["neuromorphic computing", "brain inspired computing"],
        "ct_query":   "",
        "color":      "#4f8ef7",
    },

    # ── Biotechnology ─────────────────────────────────────────────────────────
    "gene_editing": {
        "label":      "Gene Editing (CRISPR)",
        "category":   "Biotechnology",
        "icon":       "✂️",
        "arxiv_cats": ["q-bio.GN", "q-bio.BM"],
        "arxiv_kw":   ["CRISPR", "gene editing", "base editing", "prime editing", "Cas9"],
        "nih_terms":  ["CRISPR gene editing", "genome editing CRISPR"],
        "ct_query":   "CRISPR gene editing",
        "color":      "#f87171",
    },
    "mrna": {
        "label":      "mRNA Therapies",
        "category":   "Biotechnology",
        "icon":       "💉",
        "arxiv_cats": ["q-bio.BM", "q-bio.MN"],
        "arxiv_kw":   ["mRNA", "messenger RNA", "lipid nanoparticle", "mRNA vaccine"],
        "nih_terms":  ["mRNA therapy", "messenger RNA vaccine therapeutic"],
        "ct_query":   "mRNA vaccine therapy",
        "color":      "#f87171",
    },
    "car_t": {
        "label":      "CAR-T & Cell Therapies",
        "category":   "Biotechnology",
        "icon":       "🔬",
        "arxiv_cats": ["q-bio.CB", "q-bio.TO"],
        "arxiv_kw":   ["CAR-T", "chimeric antigen receptor", "cell therapy", "TCR therapy", "NK cell"],
        "nih_terms":  ["CAR-T cell therapy", "chimeric antigen receptor cancer"],
        "ct_query":   "CAR-T cell therapy chimeric antigen",
        "color":      "#f87171",
    },
    "synthetic_bio": {
        "label":      "Synthetic Biology",
        "category":   "Biotechnology",
        "icon":       "🧫",
        "arxiv_cats": ["q-bio.SC", "q-bio.BM", "q-bio.GN"],
        "arxiv_kw":   ["synthetic biology", "metabolic engineering", "protein design", "directed evolution"],
        "nih_terms":  ["synthetic biology", "metabolic engineering biosynthesis"],
        "ct_query":   "synthetic biology engineered",
        "color":      "#f87171",
    },
    "protein_engineering": {
        "label":      "Protein Engineering & AI",
        "category":   "Biotechnology",
        "icon":       "🧬",
        "arxiv_cats": ["q-bio.BM", "cs.LG"],
        "arxiv_kw":   ["protein structure", "AlphaFold", "protein design", "protein language model"],
        "nih_terms":  ["protein engineering therapeutic", "AI protein structure prediction"],
        "ct_query":   "protein engineering",
        "color":      "#f87171",
    },

    # ── Health Technology ─────────────────────────────────────────────────────
    "digital_health": {
        "label":      "Digital Health & Wearables",
        "category":   "Health Technology",
        "icon":       "⌚",
        "arxiv_cats": ["cs.HC", "eess.SP", "cs.LG"],
        "arxiv_kw":   ["wearable", "digital health", "remote monitoring", "health sensor", "mHealth"],
        "nih_terms":  ["digital health wearable monitoring", "remote patient monitoring"],
        "ct_query":   "digital health wearable monitoring device",
        "color":      "#f59e0b",
    },
    "medical_devices": {
        "label":      "Medical Devices & Implants",
        "category":   "Health Technology",
        "icon":       "🏥",
        "arxiv_cats": ["eess.SP", "cs.RO"],
        "arxiv_kw":   ["medical device", "implantable", "brain computer interface", "neural implant", "cochlear"],
        "nih_terms":  ["implantable medical device", "neural interface brain computer"],
        "ct_query":   "implant medical device neural",
        "color":      "#f59e0b",
    },
    "ai_diagnostics": {
        "label":      "AI Diagnostics & Imaging",
        "category":   "Health Technology",
        "icon":       "🩺",
        "arxiv_cats": ["cs.CV", "eess.IV", "cs.LG"],
        "arxiv_kw":   ["medical image", "radiology AI", "pathology deep learning", "diagnostic AI", "liquid biopsy"],
        "nih_terms":  ["AI diagnostic imaging", "deep learning radiology pathology"],
        "ct_query":   "artificial intelligence diagnostic imaging screening",
        "color":      "#f59e0b",
    },
    "immuno_oncology": {
        "label":      "Immuno-Oncology",
        "category":   "Health Technology",
        "icon":       "🎯",
        "arxiv_cats": ["q-bio.TO", "q-bio.CB"],
        "arxiv_kw":   ["immunotherapy", "checkpoint inhibitor", "PD-1", "tumor microenvironment", "antibody drug conjugate"],
        "nih_terms":  ["immunotherapy cancer checkpoint", "PD-1 PD-L1 cancer"],
        "ct_query":   "immunotherapy checkpoint inhibitor cancer",
        "color":      "#f59e0b",
    },

    # ── Semiconductors & Computing ────────────────────────────────────────────
    "semiconductors": {
        "label":      "Advanced Semiconductors",
        "category":   "Semiconductors & Computing",
        "icon":       "💾",
        "arxiv_cats": ["cond-mat.mes-hall", "cond-mat.mtrl-sci", "cs.AR"],
        "arxiv_kw":   ["semiconductor", "transistor", "2D material", "silicon photonics", "GAA", "FinFET", "3nm"],
        "nih_terms":  ["semiconductor nanotechnology biomedical"],
        "ct_query":   "",
        "color":      "#a78bfa",
    },
    "quantum": {
        "label":      "Quantum Computing",
        "category":   "Semiconductors & Computing",
        "icon":       "⚛️",
        "arxiv_cats": ["quant-ph", "cs.ET"],
        "arxiv_kw":   ["quantum computing", "qubit", "quantum error correction", "quantum advantage", "quantum algorithm"],
        "nih_terms":  ["quantum computing biosimulation", "quantum sensing biomedical"],
        "ct_query":   "quantum",
        "color":      "#a78bfa",
    },
    "photonics": {
        "label":      "Photonics & Optical",
        "category":   "Semiconductors & Computing",
        "icon":       "💡",
        "arxiv_cats": ["physics.optics", "eess.SP", "physics.app-ph"],
        "arxiv_kw":   ["photonics", "optical computing", "LiDAR", "laser", "silicon photonics"],
        "nih_terms":  ["optical imaging photonics biomedical", "photodynamic therapy"],
        "ct_query":   "photodynamic optical therapy",
        "color":      "#a78bfa",
    },
    "edge_computing": {
        "label":      "Edge Computing & IoT",
        "category":   "Semiconductors & Computing",
        "icon":       "🌐",
        "arxiv_cats": ["cs.DC", "cs.NI", "cs.AR"],
        "arxiv_kw":   ["edge computing", "IoT", "federated learning", "TinyML", "on-device inference"],
        "nih_terms":  ["edge computing healthcare federated learning"],
        "ct_query":   "remote monitoring IoT device",
        "color":      "#a78bfa",
    },

    # ── Communications & Infrastructure ──────────────────────────────────────
    "wireless": {
        "label":      "5G / 6G Wireless",
        "category":   "Communications & Infrastructure",
        "icon":       "📡",
        "arxiv_cats": ["eess.SP", "cs.NI", "cs.IT"],
        "arxiv_kw":   ["5G", "6G", "mmWave", "massive MIMO", "beamforming", "reconfigurable intelligent surface"],
        "nih_terms":  ["wireless health monitoring 5G"],
        "ct_query":   "",
        "color":      "#34d399",
    },
    "cybersecurity": {
        "label":      "Cybersecurity",
        "category":   "Communications & Infrastructure",
        "icon":       "🔐",
        "arxiv_cats": ["cs.CR", "cs.AI"],
        "arxiv_kw":   ["adversarial attack", "differential privacy", "homomorphic encryption", "zero trust", "intrusion detection"],
        "nih_terms":  ["cybersecurity health data privacy", "secure health information"],
        "ct_query":   "",
        "color":      "#34d399",
    },
    "space_tech": {
        "label":      "Satellite & Space Tech",
        "category":   "Communications & Infrastructure",
        "icon":       "🛸",
        "arxiv_cats": ["eess.SP", "cs.NI", "physics.space-ph"],
        "arxiv_kw":   ["LEO satellite", "starlink", "satellite communication", "space computing", "CubeSat"],
        "nih_terms":  ["space medicine microgravity health"],
        "ct_query":   "space microgravity",
        "color":      "#34d399",
    },

    # ── Energy & Cleantech ────────────────────────────────────────────────────
    "batteries": {
        "label":      "Battery & Energy Storage",
        "category":   "Energy & Cleantech",
        "icon":       "🔋",
        "arxiv_cats": ["cond-mat.mtrl-sci", "physics.chem-ph"],
        "arxiv_kw":   ["solid state battery", "lithium", "energy storage", "sodium ion", "anode cathode"],
        "nih_terms":  ["battery energy storage biomedical implant"],
        "ct_query":   "",
        "color":      "#00c896",
    },
    "nuclear_fusion": {
        "label":      "Nuclear Fusion",
        "category":   "Energy & Cleantech",
        "icon":       "☀️",
        "arxiv_cats": ["physics.plasm-ph"],
        "arxiv_kw":   ["nuclear fusion", "tokamak", "inertial confinement", "plasma confinement", "fusion reactor"],
        "nih_terms":  ["fusion energy"],
        "ct_query":   "",
        "color":      "#00c896",
    },
    "nanotechnology": {
        "label":      "Nanotechnology",
        "category":   "Energy & Cleantech",
        "icon":       "🔩",
        "arxiv_cats": ["cond-mat.mes-hall", "cond-mat.mtrl-sci"],
        "arxiv_kw":   ["nanoparticle", "nanotechnology", "graphene", "carbon nanotube", "nano drug delivery"],
        "nih_terms":  ["nanotechnology drug delivery cancer", "nanoparticle therapeutic"],
        "ct_query":   "nanoparticle drug delivery",
        "color":      "#00c896",
    },
}

# Category display order
CATEGORY_ORDER = [
    "Artificial Intelligence",
    "Biotechnology",
    "Health Technology",
    "Semiconductors & Computing",
    "Communications & Infrastructure",
    "Energy & Cleantech",
]

CATEGORY_ICONS = {
    "Artificial Intelligence":         "🧠",
    "Biotechnology":                    "🧬",
    "Health Technology":                "🏥",
    "Semiconductors & Computing":       "💾",
    "Communications & Infrastructure":  "📡",
    "Energy & Cleantech":               "⚡",
}


# ── Data fetchers (called by Flask endpoint) ──────────────────────────────────

TODAY   = datetime.now(timezone.utc)
TIMEOUT = 18

_HDRS = {
    "User-Agent": "ValuationSuite/2.1 TopicExplorer (educational)",
    "Accept":     "application/json",
}


def _get_json(url, extra=None):
    h   = {**_HDRS, **(extra or {})}
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)


def _post_json(url, payload):
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=body, method="POST",
                                  headers={**_HDRS, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)


def _safe(t, n=120):
    if not t: return ""
    t = str(t).strip().replace("\n", " ")
    return (t[:n] + "…") if len(t) > n else t


# ── arXiv ─────────────────────────────────────────────────────────────────────

def _fetch_arxiv_for_topics(topic_keys, days):
    """Fetch arXiv papers matching any of the selected topics."""
    cats    = set()
    kws     = []
    cutoff  = TODAY - timedelta(days=days)

    for k in topic_keys:
        t = TOPICS.get(k, {})
        cats.update(t.get("arxiv_cats", []))
        kws.extend(t.get("arxiv_kw",   []))

    if not cats:
        return []

    # Build query using space-separated boolean operators (arXiv API style).
    # Use urllib.parse.urlencode so parentheses, quotes and spaces are
    # percent-encoded correctly — avoids 400 Bad Request from raw '+OR+' syntax.
    cat_q = " OR ".join(f"cat:{c}" for c in sorted(cats))
    if kws:
        kw_parts = " OR ".join(
            f'ti:"{w}" OR abs:"{w}"'
            for w in kws[:6]           # limit keywords to avoid URL length issues
        )
        query = f"({cat_q}) AND ({kw_parts})"
    else:
        query = f"({cat_q})"

    params = urllib.parse.urlencode({
        "search_query": query,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
        "max_results":  "40",
        "start":        "0",
    })
    url = f"https://export.arxiv.org/api/query?{params}"

    try:
        req = urllib.request.Request(url, headers={**_HDRS, "Accept": "application/atom+xml"})
        raw = urllib.request.urlopen(req, timeout=TIMEOUT).read()
    except Exception:
        return []
    if not raw:
        return []

    results = []
    try:
        ns   = {"atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(raw)

        for entry in root.findall("atom:entry", ns):
            pub = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
            try:
                if datetime.strptime(pub, "%Y-%m-%d").replace(tzinfo=timezone.utc) < cutoff:
                    continue
            except Exception:
                pass

            title   = _safe(entry.findtext("atom:title",   default="", namespaces=ns).strip(), 120)
            summary = _safe(entry.findtext("atom:summary", default="", namespaces=ns).strip(), 220)
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            link    = link_el.get("href", "") if link_el is not None else ""
            authors = [a.text for a in entry.findall("atom:author/atom:name", ns)[:3] if a.text]
            author_str = ", ".join(authors) + (" et al." if len(authors) > 3 else "")

            pri = entry.find("arxiv:primary_category", ns)
            cat_code  = pri.get("term", "") if pri is not None else ""
            # Map primary category back to a topic label.
            # First try exact cat match; then fall back to the first selected topic label.
            matched = [k for k in topic_keys
                       if cat_code in TOPICS.get(k, {}).get("arxiv_cats", [])]
            if matched:
                topic_label = TOPICS[matched[0]]["label"]
            elif topic_keys:
                topic_label = TOPICS[topic_keys[0]]["label"]
            else:
                topic_label = cat_code

            results.append({
                "source":  "arxiv",
                "topic":   topic_label,
                "title":   title,
                "authors": author_str or "—",
                "date":    pub,
                "summary": summary,
                "url":     link,
                "cat":     cat_code,
            })
    except Exception as exc:
        pass

    return results[:30]


# ── NIH Grants ────────────────────────────────────────────────────────────────

def _fetch_nih_for_topics(topic_keys, days):
    """Fetch NIH grants matching the selected topics."""
    current_year = TODAY.year
    fiscal_years = list({current_year - 1, current_year})
    seen         = set()
    results      = []

    # Collect all NIH terms with their topic label
    term_topic = {}
    for k in topic_keys:
        t = TOPICS.get(k, {})
        for term in t.get("nih_terms", []):
            term_topic[term] = t["label"]

    for term, topic_label in list(term_topic.items())[:8]:    # max 8 to avoid rate limits
        payload = {
            "criteria": {
                "fiscal_years": fiscal_years,
                "activity_codes": ["R01", "R35", "U01", "R21", "P01", "DP2"],
                "advanced_text_search": {
                    "operator": "and", "search_field": "all", "search_text": term,
                },
            },
            "include_fields": ["ProjectNum", "ProjectTitle", "OrgName",
                               "PIName", "FiscalYear", "AwardAmount", "AbstractText"],
            "offset": 0, "limit": 4,
            "sort_field": "award_amount", "sort_order": "desc",
        }
        data, _ = _post_json("https://api.reporter.nih.gov/v2/projects/search", payload)
        if not data:
            continue
        for proj in (data.get("results") or []):
            pnum   = proj.get("project_num", "")
            amount = proj.get("award_amount") or 0
            if amount < 200_000 or pnum in seen:
                continue
            seen.add(pnum)
            results.append({
                "source":  "nih",
                "topic":   topic_label,
                "title":   _safe(proj.get("project_title", "—"), 110),
                "org":     _safe(proj.get("org_name", "—"), 50),
                "amount":  amount,
                "year":    proj.get("fiscal_year", "—"),
                "url":     f"https://reporter.nih.gov/project-details/{pnum}" if pnum else
                           "https://reporter.nih.gov/",
            })

    results.sort(key=lambda x: x["amount"], reverse=True)
    return results[:20]


# ── Clinical Trials ───────────────────────────────────────────────────────────

def _fetch_trials_for_topics(topic_keys, days):
    """Fetch Phase 2/3 trials matching the selected topics."""
    cutoff  = TODAY - timedelta(days=days)
    results = []
    seen    = set()

    queries = {}
    for k in topic_keys:
        t = TOPICS.get(k, {})
        q = t.get("ct_query", "").strip()
        if q:
            queries[q] = t["label"]

    for query, topic_label in list(queries.items())[:6]:
        fields = "NCTId,BriefTitle,OverallStatus,Phase,LeadSponsorName,LeadSponsorClass,CompletionDate,LastUpdatePostDate,Condition"
        url = (
            "https://clinicaltrials.gov/api/v2/studies?"
            f"query.term={urllib.parse.quote(query)}"
            "&filter.overallStatus=COMPLETED,ACTIVE_NOT_RECRUITING"
            "&aggFilters=studyType:int,phase:2,phase:3"
            "&sort=LastUpdatePostDate:desc"
            "&pageSize=10"
            f"&fields={fields}"
        )
        data, _ = _get_json(url, extra={"Accept": "application/json"})
        if not data:
            continue

        for study in (data.get("studies") or []):
            proto   = study.get("protocolSection", {}) or {}
            id_m    = proto.get("identificationModule",       {}) or {}
            stat_m  = proto.get("statusModule",               {}) or {}
            spon_m  = proto.get("sponsorCollaboratorsModule", {}) or {}
            cond_m  = proto.get("conditionsModule",           {}) or {}

            nct_id  = id_m.get("nctId", "")
            if not nct_id or nct_id in seen:
                continue

            last_upd = (stat_m.get("lastUpdatePostDateStruct") or {}).get("date", "")
            try:
                if datetime.strptime(last_upd, "%Y-%m-%d").replace(tzinfo=timezone.utc) < cutoff:
                    continue
            except Exception:
                pass

            seen.add(nct_id)
            status     = (stat_m.get("overallStatus") or "—").replace("_", " ").title()
            completion = (stat_m.get("completionDateStruct") or {}).get("date", "—")
            sponsor    = (spon_m.get("leadSponsor") or {}).get("name", "—")
            spon_class = (spon_m.get("leadSponsor") or {}).get("class", "")
            conditions = (cond_m.get("conditions") or [])[:2]

            results.append({
                "source":     "trials",
                "topic":      topic_label,
                "nct_id":     nct_id,
                "title":      _safe(id_m.get("briefTitle", "—"), 100),
                "status":     status,
                "sponsor":    sponsor[:50],
                "industry":   spon_class == "INDUSTRY",
                "conditions": ", ".join(conditions) or "—",
                "completion": completion,
                "url":        f"https://clinicaltrials.gov/study/{nct_id}",
            })

    results.sort(key=lambda x: x["completion"], reverse=True)
    return results[:25]


# ── Main fetch entry point (called by Flask /api/research-topics) ─────────────

def fetch_topics(topic_keys, days=30, sources=None):
    """
    Fetch data for the given topic keys from selected sources.
    Returns a dict: { "arxiv": [...], "nih": [...], "trials": [...] }
    Called by the Flask /api/research-topics endpoint.
    """
    if sources is None:
        sources = ["arxiv", "nih", "trials"]

    topic_keys = [k for k in topic_keys if k in TOPICS]
    if not topic_keys:
        return {"arxiv": [], "nih": [], "trials": [], "error": "No valid topics selected."}

    out = {}
    tasks = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        if "arxiv"  in sources: tasks["arxiv"]  = pool.submit(_fetch_arxiv_for_topics,  topic_keys, days)
        if "nih"    in sources: tasks["nih"]    = pool.submit(_fetch_nih_for_topics,    topic_keys, days)
        if "trials" in sources: tasks["trials"] = pool.submit(_fetch_trials_for_topics, topic_keys, days)

        for key, fut in tasks.items():
            try:
                out[key] = fut.result()
            except Exception as exc:
                out[key] = []

    out.setdefault("arxiv",  [])
    out.setdefault("nih",    [])
    out.setdefault("trials", [])
    return out


# ── HTML generator ────────────────────────────────────────────────────────────

def build_html():
    """Generate the interactive Topic Explorer HTML page."""

    # Build topic data for JavaScript
    topics_js   = json.dumps(TOPICS,          ensure_ascii=False, indent=2)
    cat_order_js = json.dumps(CATEGORY_ORDER, ensure_ascii=False)
    cat_icons_js = json.dumps(CATEGORY_ICONS, ensure_ascii=False)

    # Build category → topic_keys map
    cat_topics = {}
    for key, t in TOPICS.items():
        cat_topics.setdefault(t["category"], []).append(key)

    # Generate sidebar checkboxes HTML
    sidebar_html = ""
    for cat in CATEGORY_ORDER:
        keys = cat_topics.get(cat, [])
        if not keys:
            continue
        icon = CATEGORY_ICONS.get(cat, "")
        sidebar_html += f"""
        <div class="cat-group">
          <div class="cat-header">
            <span>{icon} {cat}</span>
            <button class="select-all-btn" onclick="toggleCategory('{cat}')">All</button>
          </div>"""
        for key in keys:
            t = TOPICS[key]
            sidebar_html += f"""
          <label class="topic-item">
            <input type="checkbox" class="topic-cb" value="{key}" data-cat="{cat}">
            <span class="topic-icon">{t['icon']}</span>
            <span class="topic-label">{t['label']}</span>
          </label>"""
        sidebar_html += "\n        </div>"

    gen_date = datetime.now().strftime("%B %d, %Y %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Topic Explorer</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:     #0e1117; --panel:  #161b22; --card:   #1c2128;
    --border: #30363d; --accent: #4f8ef7; --green:  #00c896;
    --muted:  #8b949e; --text:   #e6edf3; --red:    #f85149;
    --orange: #f0a500; --radius: 8px;
  }}
  html, body {{ height: 100%; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}

  /* ── Layout ── */
  .shell {{ display: flex; flex-direction: column; height: 100vh; }}
  .topbar {{ background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 0 22px; height: 52px; display: flex; align-items: center; gap: 14px;
    flex-shrink: 0; }}
  .topbar h1 {{ font-size: 16px; font-weight: 700; flex: 1; }}
  .body {{ display: flex; flex: 1; overflow: hidden; }}

  /* ── Sidebar ── */
  .sidebar {{
    width: 260px; background: var(--panel); border-right: 1px solid var(--border);
    display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden;
  }}
  .sidebar-top {{
    padding: 14px 16px 10px; border-bottom: 1px solid var(--border); flex-shrink: 0;
  }}
  .sidebar-top h2 {{ font-size: 11px; font-weight: 700; color: var(--muted);
    letter-spacing: .08em; text-transform: uppercase; margin-bottom: 10px; }}
  .controls {{ display: flex; flex-direction: column; gap: 8px; }}
  .ctrl-row {{ display: flex; align-items: center; gap: 8px; }}
  .ctrl-row label {{ font-size: 12px; color: var(--muted); white-space: nowrap; flex-shrink: 0; }}
  .ctrl-row input[type=number], .ctrl-row select {{
    flex: 1; background: var(--card); border: 1px solid var(--border); color: var(--text);
    font-size: 12px; padding: 5px 8px; border-radius: 5px; outline: none;
  }}
  .sources-row {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }}
  .src-toggle {{
    font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 4px;
    border: 1px solid var(--border); cursor: pointer; transition: all .15s;
    background: var(--card); color: var(--muted);
  }}
  .src-toggle.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  .sidebar-topics {{ flex: 1; overflow-y: auto; padding: 10px 12px; }}
  .cat-group {{ margin-bottom: 16px; }}
  .cat-header {{
    display: flex; align-items: center; justify-content: space-between;
    font-size: 11px; font-weight: 700; color: var(--muted); letter-spacing: .07em;
    text-transform: uppercase; margin-bottom: 6px; padding: 0 2px;
  }}
  .select-all-btn {{
    font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 3px;
    border: 1px solid var(--border); background: transparent; color: var(--muted);
    cursor: pointer;
  }}
  .select-all-btn:hover {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .topic-item {{
    display: flex; align-items: center; gap: 7px; padding: 6px 8px;
    border-radius: 5px; cursor: pointer; transition: background .12s;
    font-size: 13px; color: var(--muted);
  }}
  .topic-item:hover {{ background: var(--card); color: var(--text); }}
  .topic-item input {{ flex-shrink: 0; accent-color: var(--accent); cursor: pointer; }}
  .topic-item input:checked + .topic-icon + .topic-label {{ color: var(--text); }}
  .topic-icon {{ font-size: 14px; }}

  .sidebar-footer {{
    padding: 12px 16px; border-top: 1px solid var(--border); flex-shrink: 0;
  }}
  .btn-scan {{
    width: 100%; padding: 10px; background: var(--accent); color: #fff;
    border: none; border-radius: 6px; font-size: 14px; font-weight: 700;
    cursor: pointer; transition: opacity .15s;
  }}
  .btn-scan:hover {{ opacity: .88; }}
  .btn-scan:disabled {{ opacity: .4; cursor: not-allowed; }}
  .selection-count {{
    font-size: 11px; color: var(--muted); text-align: center; margin-bottom: 8px;
  }}

  /* ── Results ── */
  .results {{ flex: 1; overflow-y: auto; padding: 20px 24px; }}
  .placeholder {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100%; color: var(--muted); gap: 12px; text-align: center;
  }}
  .placeholder .big {{ font-size: 52px; }}
  .placeholder h2 {{ font-size: 18px; font-weight: 600; color: var(--text); }}
  .placeholder p {{ font-size: 13px; max-width: 400px; line-height: 1.6; }}

  /* Results layout */
  .results-header {{
    display: flex; align-items: baseline; gap: 12px; margin-bottom: 20px;
    flex-shrink: 0;
  }}
  .results-header h2 {{ font-size: 16px; font-weight: 700; }}
  .results-meta {{ font-size: 12px; color: var(--muted); }}
  .results-meta b {{ color: var(--text); }}

  .section-header {{
    font-size: 12px; font-weight: 700; color: var(--muted);
    letter-spacing: .07em; text-transform: uppercase;
    margin: 24px 0 10px; display: flex; align-items: center; gap: 8px;
  }}
  .section-count {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1px 8px; font-size: 11px; color: var(--muted);
    font-weight: 600; text-transform: none; letter-spacing: 0;
  }}

  /* Paper cards */
  .cards-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 12px;
  }}
  .paper-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 16px; display: flex; flex-direction: column; gap: 7px;
  }}
  .paper-card:hover {{ border-color: #3d444d; }}
  .card-top {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }}
  .pill {{
    display: inline-flex; align-items: center; gap: 4px;
    background: #ffffff12; border: 1px solid #ffffff1e;
    border-radius: 4px; padding: 2px 8px; font-size: 11px;
    white-space: nowrap; flex-shrink: 0;
  }}
  .card-date {{ font-size: 11px; color: var(--muted); white-space: nowrap; }}
  .card-title {{
    font-size: 13px; font-weight: 600; color: var(--text); line-height: 1.4;
    text-decoration: none;
  }}
  .card-title:hover {{ color: var(--accent); }}
  .card-summary {{ font-size: 12px; color: var(--muted); line-height: 1.5; }}
  .card-author {{ font-size: 11px; color: #6b7194; }}

  /* Table rows */
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{
    text-align: left; padding: 9px 12px; background: var(--panel);
    color: var(--muted); font-size: 11px; font-weight: 700;
    letter-spacing: .05em; text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }}
  .data-table td {{
    padding: 9px 12px; border-bottom: 1px solid #1e2530; vertical-align: middle;
  }}
  .data-table tr:nth-child(even) {{ background: #12171d; }}
  .data-table tr:hover {{ background: var(--card); }}

  /* Spinner */
  .spinner {{
    width: 36px; height: 36px; border: 3px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin .7s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

  /* Server error */
  .server-warning {{
    background: #f0a50011; border: 1px solid #f0a50033;
    border-radius: 8px; padding: 16px; margin-bottom: 20px;
    font-size: 13px; color: var(--orange);
  }}

  /* ext link */
  a.ext {{ color: var(--accent); text-decoration: none; font-size: 12px; }}
  a.ext:hover {{ text-decoration: underline; }}

  /* Status dot */
  .dot-green {{ color: #00c896; }}
  .dot-orange {{ color: #f0a500; }}

  /* Industry badge */
  .ind-badge {{
    background: #4f8ef722; color: #4f8ef7; border: 1px solid #4f8ef744;
    border-radius: 3px; padding: 1px 5px; font-size: 10px;
  }}
</style>
</head>
<body>
<div class="shell">

  <!-- Topbar -->
  <div class="topbar">
    <h1>🔭&nbsp; Research Topic Explorer</h1>
    <span style="font-size:12px;color:var(--muted)">Generated {gen_date}</span>
  </div>

  <div class="body">

    <!-- Sidebar: topic selector -->
    <div class="sidebar">
      <div class="sidebar-top">
        <h2>Settings</h2>
        <div class="controls">
          <div class="ctrl-row">
            <label>Days back</label>
            <input type="number" id="days-input" value="30" min="7" max="180" step="7">
          </div>
          <div>
            <div style="font-size:11px;font-weight:700;color:var(--muted);margin-bottom:6px">Sources</div>
            <div class="sources-row">
              <button class="src-toggle active" data-src="arxiv"  onclick="toggleSource(this)">arXiv</button>
              <button class="src-toggle active" data-src="nih"    onclick="toggleSource(this)">NIH Grants</button>
              <button class="src-toggle active" data-src="trials" onclick="toggleSource(this)">Trials</button>
            </div>
          </div>
        </div>
      </div>

      <div class="sidebar-topics">
        <div style="display:flex;align-items:center;justify-content:space-between;
                    font-size:11px;font-weight:700;color:var(--muted);
                    letter-spacing:.07em;text-transform:uppercase;margin-bottom:12px">
          <span>Topics</span>
          <div style="display:flex;gap:6px">
            <button class="select-all-btn" onclick="selectAll(true)">All</button>
            <button class="select-all-btn" onclick="selectAll(false)">None</button>
          </div>
        </div>
        {sidebar_html}
      </div>

      <div class="sidebar-footer">
        <div class="selection-count" id="sel-count">0 topics selected</div>
        <button class="btn-scan" id="btn-scan" onclick="runScan()" disabled>
          ▶&nbsp; Scan Selected Topics
        </button>
      </div>
    </div>

    <!-- Results pane -->
    <div class="results" id="results">
      <div class="placeholder" id="placeholder">
        <div class="big">🔭</div>
        <h2>Select Topics to Scan</h2>
        <p>Choose one or more research topics from the left panel,
           then click <strong>Scan Selected Topics</strong> to fetch
           live data from arXiv, NIH grants, and ClinicalTrials.gov.</p>
        <p style="margin-top:8px;color:#6b7194;font-size:12px">
          Requires the ValuationSuite server (app.py) to be running on port 5050 or 5051.
        </p>
      </div>
    </div>

  </div><!-- .body -->
</div><!-- .shell -->

<script>
const TOPICS       = {topics_js};
const CAT_ORDER    = {cat_order_js};
const CAT_ICONS    = {cat_icons_js};

let activeSources = new Set(['arxiv', 'nih', 'trials']);

// ── Source toggles ──────────────────────────────────────────────────────────
function toggleSource(btn) {{
  const src = btn.dataset.src;
  if (activeSources.has(src)) {{ activeSources.delete(src); btn.classList.remove('active'); }}
  else                         {{ activeSources.add(src);    btn.classList.add('active');    }}
}}

// ── Category select all ─────────────────────────────────────────────────────
function toggleCategory(cat) {{
  const boxes = document.querySelectorAll(`.topic-cb[data-cat="${{cat}}"]`);
  const allChecked = [...boxes].every(b => b.checked);
  boxes.forEach(b => {{ b.checked = !allChecked; }});
  updateCount();
}}

function selectAll(on) {{
  document.querySelectorAll('.topic-cb').forEach(b => {{ b.checked = on; }});
  updateCount();
}}

function updateCount() {{
  const n   = [...document.querySelectorAll('.topic-cb:checked')].length;
  document.getElementById('sel-count').textContent = n === 0 ? 'No topics selected' : `${{n}} topic${{n === 1 ? '' : 's'}} selected`;
  document.getElementById('btn-scan').disabled = n === 0;
}}

document.querySelectorAll('.topic-cb').forEach(b => b.addEventListener('change', updateCount));

// ── Server detection ────────────────────────────────────────────────────────
async function findServer() {{
  for (const port of [5050, 5051]) {{
    try {{
      const r = await fetch(`http://127.0.0.1:${{port}}/api/scripts`, {{signal: AbortSignal.timeout(2000)}});
      if (r.ok) return `http://127.0.0.1:${{port}}`;
    }} catch {{}}
  }}
  return null;
}}

// ── Scan ────────────────────────────────────────────────────────────────────
async function runScan() {{
  const selectedKeys = [...document.querySelectorAll('.topic-cb:checked')].map(b => b.value);
  if (!selectedKeys.length) return;

  const days    = parseInt(document.getElementById('days-input').value) || 30;
  const sources = [...activeSources];

  const res = document.getElementById('results');
  res.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:100%;gap:16px;color:var(--muted)">
      <div class="spinner"></div>
      <div style="font-size:14px">Scanning ${{selectedKeys.length}} topic${{selectedKeys.length>1?'s':''}}…</div>
    </div>`;

  const base = await findServer();
  if (!base) {{
    res.innerHTML = `
      <div class="server-warning">
        ⚠ &nbsp;<strong>ValuationSuite server not found.</strong><br><br>
        This tool requires the Flask server to be running. Start it by launching
        <code>ValuationSuite.app</code> or running <code>python app.py</code>
        from the project folder, then click Scan again.
      </div>`;
    return;
  }}

  let data;
  try {{
    const r = await fetch(`${{base}}/api/research-topics`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{topics: selectedKeys, days, sources}}),
    }});
    data = await r.json();
  }} catch (err) {{
    res.innerHTML = `<div class="server-warning">⚠ Fetch error: ${{err}}</div>`;
    return;
  }}

  if (data.error) {{
    res.innerHTML = `<div class="server-warning">⚠ ${{data.error}}</div>`;
    return;
  }}

  renderResults(data, selectedKeys, days);
}}

// ── Render ──────────────────────────────────────────────────────────────────
function pill(text, color) {{
  return `<span class="pill" style="color:${{color}};border-color:${{color}}44;background:${{color}}18">${{text}}</span>`;
}}

function statusDot(status) {{
  if (status.includes('Completed'))  return '<span class="dot-green">●</span>';
  if (status.includes('Active'))     return '<span class="dot-orange">●</span>';
  return '<span style="color:var(--muted)">●</span>';
}}

function renderResults(data, selectedKeys, days) {{
  const arxiv  = data.arxiv  || [];
  const nih    = data.nih    || [];
  const trials = data.trials || [];
  const total  = arxiv.length + nih.length + trials.length;

  const selectedLabels = selectedKeys.map(k => TOPICS[k]?.label || k);
  const topicList      = selectedLabels.slice(0,4).join(', ') + (selectedLabels.length > 4 ? ` +${{selectedLabels.length-4}} more` : '');

  let html = `
    <div class="results-header">
      <h2>Scan Results</h2>
      <div class="results-meta"><b>${{total}}</b> items · <b>${{selectedKeys.length}}</b> topics · last <b>${{days}}</b> days</div>
    </div>
    <div style="color:var(--muted);font-size:12px;margin-bottom:20px">${{topicList}}</div>`;

  // ── arXiv section
  if (arxiv.length) {{
    html += `<div class="section-header">📄 arXiv Research Papers <span class="section-count">${{arxiv.length}}</span></div>`;
    html += '<div class="cards-grid">';
    for (const p of arxiv) {{
      const topicColor = Object.values(TOPICS).find(t => t.label === p.topic)?.color || '#8b949e';
      html += `
        <div class="paper-card">
          <div class="card-top">
            <div style="display:flex;gap:6px;flex-wrap:wrap">${{pill(p.topic, topicColor)}}</div>
            <span class="card-date">${{p.date}}</span>
          </div>
          <a class="card-title" href="${{p.url}}" target="_blank" rel="noopener noreferrer">${{p.title}}</a>
          <div class="card-summary">${{p.summary}}</div>
          <div class="card-author">${{p.authors}}</div>
        </div>`;
    }}
    html += '</div>';
  }}

  // ── Clinical Trials section
  if (trials.length) {{
    html += `<div class="section-header">🧬 Clinical Trials (Phase 2/3) <span class="section-count">${{trials.length}}</span></div>`;
    html += `<div style="overflow-x:auto"><table class="data-table">
      <thead><tr>
        <th>Trial</th><th>Topic</th><th>Sponsor</th><th>Conditions</th>
        <th style="text-align:center">Status</th><th style="text-align:center">Completion</th>
      </tr></thead><tbody>`;
    for (const t of trials) {{
      const topicColor = Object.values(TOPICS).find(x => x.label === t.topic)?.color || '#8b949e';
      const indBadge   = t.industry ? '<span class="ind-badge">Industry</span>' : '';
      html += `<tr>
        <td><a href="${{t.url}}" target="_blank" rel="noopener noreferrer"
               style="color:var(--text);text-decoration:none;font-size:12px">${{t.title}}</a>
            <br><span style="color:var(--muted);font-size:11px">${{t.nct_id}}</span></td>
        <td>${{pill(t.topic, topicColor)}}</td>
        <td style="font-size:12px;color:var(--muted)">${{t.sponsor}} ${{indBadge}}</td>
        <td style="font-size:12px;color:var(--muted)">${{t.conditions}}</td>
        <td style="text-align:center">${{statusDot(t.status)}} <span style="font-size:11px;color:var(--muted)">${{t.status}}</span></td>
        <td style="text-align:center;font-size:12px;color:var(--muted);white-space:nowrap">${{t.completion}}</td>
      </tr>`;
    }}
    html += '</tbody></table></div>';
  }}

  // ── NIH Grants section
  if (nih.length) {{
    html += `<div class="section-header">💰 NIH Research Grants <span class="section-count">${{nih.length}}</span></div>`;
    html += `<div style="overflow-x:auto"><table class="data-table">
      <thead><tr>
        <th>Project</th><th>Topic</th><th>Institution</th>
        <th style="text-align:right">Award</th><th style="text-align:center">FY</th><th style="text-align:center">Link</th>
      </tr></thead><tbody>`;
    for (const g of nih) {{
      const topicColor = Object.values(TOPICS).find(x => x.label === g.topic)?.color || '#8b949e';
      const fmt = new Intl.NumberFormat('en-US', {{style:'currency', currency:'USD', maximumFractionDigits:0}}).format(g.amount);
      html += `<tr>
        <td style="font-size:12px">${{g.title}}</td>
        <td>${{pill(g.topic, topicColor)}}</td>
        <td style="font-size:12px;color:var(--muted)">${{g.org}}</td>
        <td style="text-align:right;font-size:12px;color:#00c896;font-family:monospace">${{fmt}}</td>
        <td style="text-align:center;font-size:12px;color:var(--muted)">${{g.year}}</td>
        <td style="text-align:center"><a class="ext" href="${{g.url}}" target="_blank">NIH ↗</a></td>
      </tr>`;
    }}
    html += '</tbody></table></div>';
  }}

  if (!total) {{
    html += `<div style="color:var(--muted);font-size:13px;padding:20px">
      No results found for the selected topics and time window. Try increasing the days or selecting different topics.
    </div>`;
  }}

  document.getElementById('results').innerHTML = html;
}}
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Research Topic Explorer")
    parser.add_argument("--days", type=int, default=30, help="Default lookback (days)")
    args = parser.parse_args()

    print(f"\n  Research Topic Explorer")
    print(f"  {len(TOPICS)} topics across {len(CATEGORY_ORDER)} categories")

    # ── Prefer the Flask-hosted version (same-origin AJAX, no CORS issues) ────
    # When launched from the ValuationSuite app, Flask is always running.
    # Opening http://127.0.0.1:<port>/topic-explorer means the page and the API
    # share the same origin, so browser CORS policy never blocks the AJAX scan.
    flask_url = None
    for port in [5050, 5051]:
        try:
            check = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/scripts", timeout=1
            )
            if check.status == 200:
                flask_url = f"http://127.0.0.1:{port}/topic-explorer"
                break
        except Exception:
            pass

    if flask_url:
        print(f"  ValuationSuite server detected — opening hosted UI…")
        print(f"  URL: {flask_url}")
        webbrowser.open(flask_url)
        print(f"  ✓  Done\n")
        return

    # ── Fallback: write HTML to disk and open as file:// ─────────────────────
    # Works for direct standalone runs; the user will see a warning in the UI
    # asking them to start the Flask server before scanning.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR    = os.path.join(SCRIPT_DIR, "researchData")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"  Flask server not found — writing static UI to disk…")
    html_str = build_html()
    fpath    = os.path.join(OUT_DIR, "topicExplorer.html")

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"  Saved → {fpath}")
    print(f"  Opening browser…")
    webbrowser.open(f"file://{fpath}")
    print(f"  ⚠  Note: start ValuationSuite (python app.py) before scanning topics.\n")


if __name__ == "__main__":
    main()

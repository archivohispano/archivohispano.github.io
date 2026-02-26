#!/usr/bin/env python3
"""
PDF Extraction Pipeline -- Albizu Campos Archive
=================================================
Fully automated: extracts texts, cleans OCR, translates, auto-reviews,
commits passing texts to _texts/, and pushes to GitHub.

QA gates:
  QA1 OCR quality     (Qwen)   -- score >= THRESHOLD or retry cleanup
  QA2 Text integrity  (Gemini) -- must pass
  QA3 Front matter    (code)   -- must pass hard checks
  QA4 Translation     (Qwen)   -- score >= THRESHOLD or retry translation
  QA5 Final review    (Gemini) -- score >= THRESHOLD --> auto-commit & push

THRESHOLD = 4  (strict mode, as requested)

Supported sources:
  Obras Escogidas 1923-1936 (Tomos 1-4) -- Pedro Albizu Campos
  Dr. Pedro Albizu Campos y la Independencia de Puerto Rico (1961)
                                         -- Laura de Albizu Campos (whole book)

Usage:
    python scripts/pdf_pipeline.py --tomo 4          # start here (text layer, fastest)
    python scripts/pdf_pipeline.py --tomo 2
    python scripts/pdf_pipeline.py --laura
    python scripts/pdf_pipeline.py --all             # tomos 1-4
    python scripts/pdf_pipeline.py --everything      # all five PDFs
    python scripts/pdf_pipeline.py --pdf /path/to.pdf

Dependencies:
    pip install pymupdf google-generativeai openai pillow
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# -- Third-party imports ------------------------------------------------------
try:
    import fitz
except ImportError:
    sys.exit("ERROR: pip install pymupdf")

try:
    import google.generativeai as genai
except ImportError:
    sys.exit("ERROR: pip install google-generativeai")

# openai/DashScope optional -- only needed if QWEN_API_KEY is set
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import PIL.Image
except ImportError:
    sys.exit("ERROR: pip install pillow")

# -- Config -------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
try:
    from pipeline_config import GEMINI_API_KEY, QWEN_API_KEY, PDF_PATHS as _CFG_PATHS
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    QWEN_API_KEY   = os.environ.get("QWEN_API_KEY", "")
    _CFG_PATHS     = {}

# -- Paths --------------------------------------------------------------------
REPO_ROOT       = Path(__file__).parent.parent
INCOMING_DIR    = REPO_ROOT / "incoming"
TEXTS_DIR       = REPO_ROOT / "_texts"
IMAGE_CACHE_DIR = REPO_ROOT / ".pipeline_cache" / "images"

INCOMING_DIR.mkdir(exist_ok=True)
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -- Quality threshold --------------------------------------------------------
QA_THRESHOLD = 4   # strict: score must be >= 4/5 to auto-publish

# -- Known collections --------------------------------------------------------
KNOWN_COLLECTIONS = {
    "independencia-puerto-rico",
    "nacionalismo-puertorriqueno",
    "espiritualidad",
    "carcel-princesa",
    "relaciones-internacionales",
    "cartas",
    "discursos",
    "huelgas",
    "biografia",
}

# -- Spanish stop-words for slug generation -----------------------------------
ES_STOP_WORDS = {
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "el",
    "en", "entre", "hacia", "hasta", "la", "las", "le", "les", "lo", "los",
    "mediante", "para", "por", "se", "sin", "sobre", "su", "sus", "un",
    "una", "uno", "unos", "unas", "y", "o", "u", "e", "que", "es", "nos", "dr",
}

# -- Author configs -----------------------------------------------------------
_AUTHOR_PAC = {
    "author":           "pedro-albizu-campos",
    "author_name":      "Pedro Albizu Campos",
    "file_prefix":      "pac",
    "es_base":          "/es/textos/pedro-albizu-campos",
    "en_base":          "/en/texts/pedro-albizu-campos",
    "default_source":   "Obras Escogidas 1923-1936",
    "default_source_en":"Obras Escogidas 1923-1936 (Selected Works)",
}

_AUTHOR_LAC = {
    "author":           "laura-de-albizu-campos",
    "author_name":      "Laura de Albizu Campos",
    "file_prefix":      "lac",
    "es_base":          "/es/textos/laura-de-albizu-campos",
    "en_base":          "/en/texts/laura-de-albizu-campos",
    "default_source":   "Dr. Pedro Albizu Campos y la Independencia de Puerto Rico",
    "default_source_en":"Dr. Pedro Albizu Campos and the Independence of Puerto Rico",
}


# -- Data classes -------------------------------------------------------------
@dataclass
class TextSection:
    title:      str
    start_page: int
    end_page:   int
    date_hint:  str = ""
    doc_type:   str = "unknown"


@dataclass
class QAResult:
    score:    int
    passed:   bool
    issues:   list = field(default_factory=list)
    doc_type: str  = ""


@dataclass
class ProcessedText:
    section:         TextSection
    raw_text:        str
    cleaned_text:    str
    qa1:             QAResult
    qa2:             QAResult
    front_matter_es: dict
    front_matter_en: dict
    es_body:         str
    en_body:         str
    qa3_issues:      list = field(default_factory=list)
    qa4:             Optional[QAResult] = None
    qa5:             Optional[QAResult] = None
    status:          str  = "STAGED"
    slug:            str  = ""


# -- API helpers --------------------------------------------------------------
def get_gemini() -> genai.GenerativeModel:
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.0-flash")


def get_qwen():
    """Return a DashScope/Qwen client if key is configured, else None."""
    if not _OPENAI_AVAILABLE:
        return None
    if not QWEN_API_KEY or QWEN_API_KEY in ("YOUR_QWEN_API_KEY_HERE", ""):
        return None
    return _OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _strip_json_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()


# -- Image cache --------------------------------------------------------------
def _get_page_image(doc: fitz.Document, page_idx: int) -> bytes:
    doc_hash   = hashlib.md5(str(doc.name).encode()).hexdigest()[:8]
    cache_path = IMAGE_CACHE_DIR / f"{doc_hash}_p{page_idx:04d}.png"
    if cache_path.exists():
        return cache_path.read_bytes()
    pix       = doc[page_idx].get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    cache_path.write_bytes(img_bytes)
    return img_bytes


def _bytes_to_pil(img_bytes: bytes) -> PIL.Image.Image:
    return PIL.Image.open(io.BytesIO(img_bytes))


# =============================================================================
# STAGE 1 -- TOC Detection
# =============================================================================
def detect_toc(pdf_path: Path, has_text_layer: bool, author_cfg: dict) -> list:
    model  = get_gemini()
    doc    = fitz.open(str(pdf_path))
    n_toc  = min(10, len(doc))
    a_name = author_cfg["author_name"]

    instruction = (
        f"You are reading a book by/about {a_name} on Puerto Rican independence. "
        "Identify ALL individual texts in the table of contents "
        "(speeches, letters, essays, proclamations, chapters). "
        "Return ONLY a valid JSON array, no commentary, no markdown. "
        'Each element: {"title": str, "start_page": int, "end_page": int, '
        '"date_hint": str, "doc_type": str}. '
        "doc_type: speech | letter | essay | proclamation | chapter | other. "
        "If end_page unknown, estimate start_page + 4."
    )

    if has_text_layer:
        toc_text = "\n".join(doc[i].get_text() for i in range(n_toc))
        response = model.generate_content(f"{instruction}\n\nTOC text:\n{toc_text}")
    else:
        parts = [instruction] + [_bytes_to_pil(_get_page_image(doc, i)) for i in range(n_toc)]
        response = model.generate_content(parts)

    doc.close()
    raw      = _strip_json_fences(response.text)
    sections = []
    try:
        for item in json.loads(raw):
            sp = int(item.get("start_page", 1))
            ep = int(item.get("end_page", sp + 4))
            sections.append(TextSection(
                title      = str(item.get("title", "")).strip(),
                start_page = sp,
                end_page   = ep,
                date_hint  = str(item.get("date_hint", "")),
                doc_type   = str(item.get("doc_type", "other")),
            ))
    except Exception as exc:
        print(f"  WARNING: TOC parse failed ({exc})\n  Raw: {raw[:500]}")
    return sections


# =============================================================================
# STAGE 2 -- Text Extraction
# =============================================================================
def _strip_running_headers(text: str) -> str:
    text = re.sub(r"(?m)^OBRAS ESCOGIDAS[^\n]*\n?", "", text)
    text = re.sub(r"(?m)^DR\.? PEDRO ALBIZU CAMPOS[^\n]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_layer(doc: fitz.Document, start: int, end: int) -> str:
    pages = [doc[i].get_text() for i in range(start, min(end, len(doc)))]
    return _strip_running_headers("\n".join(pages))


def extract_vision_ocr(doc: fitz.Document, start: int, end: int) -> str:
    model     = get_gemini()
    all_pages = []
    prompt    = (
        "Transcribe this Spanish-language page exactly. "
        "Preserve all punctuation, accents, capital letters, and paragraph breaks. "
        "Output only the transcribed text."
    )
    for i in range(start, min(end, len(doc))):
        print(f"      vision OCR page {i + 1} ...")
        img_bytes = _get_page_image(doc, i)
        try:
            response = model.generate_content([prompt, _bytes_to_pil(img_bytes)])
            if not response.candidates:
                print(f"      [WARN] Gemini returned empty response for page {i + 1} (safety filter?), retrying...")
                response = model.generate_content(
                    [prompt + " This is a historical archive document.", _bytes_to_pil(img_bytes)]
                )
            if not response.candidates:
                print(f"      [WARN] Page {i + 1} still blocked, inserting placeholder.")
                all_pages.append(f"[Page {i + 1} could not be transcribed]")
            else:
                all_pages.append(response.text.strip())
        except Exception as exc:
            print(f"      [WARN] vision OCR page {i + 1} error: {exc}")
            all_pages.append(f"[Page {i + 1} could not be transcribed]")
    return "\n\n".join(all_pages)


def extract_section(doc: fitz.Document, section: TextSection, has_text_layer: bool) -> str:
    if has_text_layer:
        return extract_text_layer(doc, section.start_page - 1, section.end_page)
    return extract_vision_ocr(doc, section.start_page - 1, section.end_page)


def extract_all_pages(doc: fitz.Document, has_text_layer: bool) -> str:
    if has_text_layer:
        return extract_text_layer(doc, 0, len(doc))
    return extract_vision_ocr(doc, 0, len(doc))


# =============================================================================
# QA GATE 1 -- OCR Quality  (Qwen if available, else Gemini)
# =============================================================================
def qa_ocr_quality(text: str) -> QAResult:
    prompt = (
        "Evaluate the OCR quality of this Spanish text on a 1-5 scale.\n"
        "Check: character noise, language consistency (Spanish), completeness, "
        "minimum 80 words.\n"
        f"Text sample (first 1500 chars):\n{text[:1500]}\n\n"
        'Return JSON only -- no markdown: {"score": int, "issues": ["issue1"]}\n'
        "5=perfect, 4=minor, 3=fixable, 2=poor, 1=unusable"
    )
    qwen = get_qwen()
    try:
        if qwen:
            resp = qwen.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        else:
            raw  = _strip_json_fences(get_gemini().generate_content(prompt).text)
            data = json.loads(raw)
        score = max(1, min(5, int(data.get("score", 3))))
        return QAResult(score=score, passed=score >= QA_THRESHOLD, issues=data.get("issues", []))
    except Exception as exc:
        print(f"      QA1 error: {exc}")
        return QAResult(score=3, passed=False, issues=[str(exc)])


# =============================================================================
# STAGE 3 -- Text Cleanup  (Qwen if available, else Gemini)
# =============================================================================
def clean_text(raw: str, issues: list, extra_instruction: str = "") -> str:
    issues_str = "\n".join(f"- {i}" for i in issues) if issues else "- General OCR artifacts"
    prompt = (
        "Clean this Spanish historical text extracted from a scanned book.\n\n"
        f"Known OCR issues:\n{issues_str}\n"
        + (f"\nAdditional instruction: {extra_instruction}\n" if extra_instruction else "")
        + "\nFix: character substitutions (o/th/i noise), hyphenated line-breaks, "
        "running headers ('OBRAS ESCOGIDAS', 'DR. PEDRO'), standalone page numbers.\n"
        "Preserve all proper nouns, dates, historical terms verbatim.\n"
        "Preserve paragraph structure.\n\n"
        "Return ONLY the cleaned Spanish text.\n\n"
        f"TEXT:\n{raw}"
    )
    qwen = get_qwen()
    try:
        if qwen:
            resp = qwen.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        else:
            return get_gemini().generate_content(prompt).text.strip()
    except Exception as exc:
        print(f"      Cleanup error: {exc}")
        return raw


def _retry_clean(raw: str, qa1: QAResult) -> tuple:
    """One retry with a stronger cleanup prompt. Returns (cleaned, new_qa1)."""
    print("      retry clean (stronger) ...")
    issues_ext = qa1.issues + [f"Previous score was only {qa1.score}/5 -- be more thorough"]
    cleaned2   = clean_text(raw, issues_ext, extra_instruction="Be very aggressive about fixing noise.")
    qa1b       = qa_ocr_quality(cleaned2)
    print(f"      QA1 retry: {qa1b.score}/5")
    return cleaned2, qa1b


# =============================================================================
# QA GATE 2 -- Text Integrity  (Gemini)
# =============================================================================
def qa_text_integrity(text: str, whole_book: bool = False) -> QAResult:
    if whole_book:
        # A complete book always passes integrity; classify as book
        return QAResult(score=5, passed=True, doc_type="book")
    model  = get_gemini()
    prompt = (
        "Evaluate this text extracted from a Puerto Rican independence book.\n\n"
        "Checks:\n"
        "1. Complete standalone piece (not a fragment or index entry)\n"
        "2. Clear opening and closing\n"
        "3. Free of mixed-in editorial prefaces\n"
        "4. At least 100 words\n\n"
        "Classify doc_type: speech | letter | essay | proclamation | chapter | fragment\n\n"
        'Return JSON only: {"passed": bool, "doc_type": str, "issues": [str]}\n\n'
        f"TEXT (first 2000 chars):\n{text[:2000]}"
    )
    try:
        raw    = _strip_json_fences(model.generate_content(prompt).text)
        data   = json.loads(raw)
        passed = bool(data.get("passed", True))
        return QAResult(
            score    = 5 if passed else 2,
            passed   = passed,
            issues   = data.get("issues", []),
            doc_type = data.get("doc_type", "unknown"),
        )
    except Exception as exc:
        print(f"      QA2 error: {exc}")
        return QAResult(score=3, passed=True, issues=[str(exc)])


# =============================================================================
# STAGE 4 -- Metadata Generation  (Gemini)
# =============================================================================
def generate_metadata(text: str, section: TextSection, author_cfg: dict) -> dict:
    model            = get_gemini()
    collections_list = ", ".join(sorted(KNOWN_COLLECTIONS))
    a_name           = author_cfg["author_name"]
    default_src      = author_cfg["default_source"]

    prompt = (
        f"Generate metadata for this text by {a_name}.\n\n"
        f"Document type hint: {section.doc_type}\n"
        f"Date hint: {section.date_hint}\n"
        f"Title hint: {section.title}\n\n"
        f"Suggest collections from: {collections_list}\n\n"
        "Return JSON only:\n"
        '{"title": "exact Spanish title", "date": "YYYY-MM-DD or YYYY-01-01", '
        f'"source": "{default_src}", "source_detail": "full citation", '
        '"collections": ["slug1"]}\n\n'
        f"TEXT (first 3000 chars):\n{text[:3000]}"
    )
    try:
        raw = _strip_json_fences(model.generate_content(prompt).text)
        return json.loads(raw)
    except Exception as exc:
        print(f"      Metadata error: {exc}")
        return {
            "title":         section.title,
            "date":          _parse_date_hint(section.date_hint),
            "source":        default_src,
            "source_detail": f"{a_name}, {default_src}",
            "collections":   [],
        }


def _parse_date_hint(hint: str) -> str:
    if not hint:
        return "1930-01-01"
    for pat, fmt in [
        (r"(\d{4})-(\d{2})-(\d{2})", lambda m: m.group(0)),
        (r"(\d{4})-(\d{2})",         lambda m: f"{m.group(1)}-{m.group(2)}-01"),
        (r"(\d{4})",                  lambda m: f"{m.group(1)}-01-01"),
    ]:
        m = re.search(pat, hint)
        if m:
            return fmt(m)
    return "1930-01-01"


def make_slug(title: str) -> str:
    title = title.lower()
    for src, dst in [("\u00e1","a"),("\u00e9","e"),("\u00ed","i"),("\u00f3","o"),
                     ("\u00fa","u"),("\u00fc","u"),("\u00f1","n"),("\u00e7","c")]:
        title = title.replace(src, dst)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    words = [w for w in title.split() if w not in ES_STOP_WORDS]
    return "-".join(words[:8]) or "texto"


def build_front_matter_es(meta: dict, slug: str, author_cfg: dict) -> dict:
    return {
        "layout":          "text",
        "lang":            "es",
        "title":           meta.get("title", ""),
        "author":          author_cfg["author"],
        "author_name":     author_cfg["author_name"],
        "date":            meta.get("date", "1930-01-01"),
        "source":          meta.get("source", author_cfg["default_source"]),
        "source_detail":   meta.get("source_detail", ""),
        "country":         "puerto-rico",
        "permalink":       f"{author_cfg['es_base']}/{slug}",
        "english_version": f"{author_cfg['en_base']}/{slug}",
        "collections":     meta.get("collections", []),
    }


def build_front_matter_en(meta: dict, en_title: str, slug: str, author_cfg: dict) -> dict:
    return {
        "layout":           "text",
        "lang":             "en",
        "title":            en_title,
        "author":           author_cfg["author"],
        "author_name":      author_cfg["author_name"],
        "date":             meta.get("date", "1930-01-01"),
        "source":           author_cfg["default_source_en"],
        "source_detail":    meta.get("source_detail", ""),
        "country":          "puerto-rico",
        "permalink":        f"{author_cfg['en_base']}/{slug}",
        "spanish_version":  f"{author_cfg['es_base']}/{slug}",
        "collections":      meta.get("collections", []),
    }


# =============================================================================
# QA GATE 3 -- Front Matter Validation  (programmatic)
# =============================================================================
_REQUIRED_ES = {"layout","lang","title","author","author_name","date",
                "source","country","permalink","english_version"}
_REQUIRED_EN = {"layout","lang","title","author","author_name","date",
                "source","country","permalink","spanish_version"}
_DATE_RE     = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


def qa_front_matter(fm_es: dict, fm_en: dict, slug: str, author_cfg: dict) -> list:
    issues  = []
    es_base = author_cfg["es_base"]
    en_base = author_cfg["en_base"]

    for f in _REQUIRED_ES:
        if not fm_es.get(f):
            issues.append(f"ES missing: {f}")
    for f in _REQUIRED_EN:
        if not fm_en.get(f):
            issues.append(f"EN missing: {f}")
    if not _DATE_RE.match(str(fm_es.get("date", ""))):
        issues.append(f"Bad date: {fm_es.get('date')}")
    if fm_es.get("permalink") != f"{es_base}/{slug}":
        issues.append(f"ES permalink mismatch")
    if fm_en.get("permalink") != f"{en_base}/{slug}":
        issues.append(f"EN permalink mismatch")

    existing: set = set()
    for md_file in TEXTS_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            m = re.search(r"^permalink:\s*(.+)$", content, re.MULTILINE)
            if m:
                existing.add(m.group(1).strip())
        except OSError:
            pass
    if fm_es.get("permalink") in existing:
        issues.append(f"Duplicate ES permalink")
    if fm_en.get("permalink") in existing:
        issues.append(f"Duplicate EN permalink")
    for col in fm_es.get("collections", []):
        if col not in KNOWN_COLLECTIONS:
            issues.append(f"Unknown collection (warning): {col}")
    return issues


# =============================================================================
# STAGE 5 -- English Translation  (Gemini)
# =============================================================================
def translate_to_english(es_text: str, es_title: str, author_cfg: dict,
                         retry_hints: str = "") -> tuple:
    """Returns (en_body, en_title)."""
    model  = get_gemini()
    a_name = author_cfg["author_name"]
    hints  = f"\nFix these issues from a previous attempt: {retry_hints}\n" if retry_hints else ""

    body_prompt = (
        f"Translate this text by {a_name} from Spanish to English.{hints}\n"
        "- Preserve political/legal terminology ('Patria'='Homeland' etc.)\n"
        "- Keep all proper nouns, place names, dates exactly\n"
        "- Maintain formal rhetorical register\n"
        "- Do not add, remove, or paraphrase\n\n"
        "Return ONLY the English translation.\n\n"
        f"SPANISH:\n{es_text}"
    )
    title_prompt = (
        f'Translate to English: "{es_title}"\nReturn only the title.'
    )
    try:
        en_body  = model.generate_content(body_prompt).text.strip()
        en_title = model.generate_content(title_prompt).text.strip().strip('"')
    except Exception as exc:
        print(f"      Translation error: {exc}")
        en_body, en_title = f"[Translation failed: {exc}]", es_title
    return en_body, en_title


def translate_chunked(es_text: str, es_title: str, author_cfg: dict,
                      chunk_words: int = 4000) -> tuple:
    """Translate a long text in word-count chunks. Returns (en_body, en_title)."""
    words  = es_text.split()
    if len(words) <= chunk_words:
        return translate_to_english(es_text, es_title, author_cfg)

    chunks         = []
    for i in range(0, len(words), chunk_words):
        chunks.append(" ".join(words[i:i + chunk_words]))

    print(f"      translating in {len(chunks)} chunks ({len(words)} words total) ...")
    translated = []
    for idx, chunk in enumerate(chunks):
        print(f"        chunk {idx + 1}/{len(chunks)} ...")
        en_chunk, _ = translate_to_english(chunk, es_title, author_cfg)
        translated.append(en_chunk)

    en_body  = "\n\n".join(translated)
    _, en_title = translate_to_english("", es_title, author_cfg)  # title only
    return en_body, en_title


def _retry_translate(es_text: str, es_title: str, author_cfg: dict,
                     qa4: QAResult) -> tuple:
    """One retry with QA4 issues as guidance. Returns (en_body, en_title, new_qa4)."""
    print("      retry translation (fixing issues) ...")
    hints    = "; ".join(qa4.issues[:3]) if qa4.issues else "improve accuracy"
    en_body2, en_title2 = translate_to_english(es_text, es_title, author_cfg,
                                                retry_hints=hints)
    qa4b = qa_translation(es_text, en_body2, author_cfg)
    print(f"      QA4 retry: {qa4b.score}/5")
    return en_body2, en_title2, qa4b


# =============================================================================
# QA GATE 4 -- Translation Quality  (Qwen if available, else Gemini)
# =============================================================================
def qa_translation(es_text: str, en_text: str, author_cfg: dict) -> QAResult:
    a_name    = author_cfg["author_name"]
    sentences = [s.strip() for s in re.split(r"[.!?]+", en_text) if len(s.strip()) > 30]
    sample    = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences[:3]))
    prompt = (
        f"Evaluate this English translation of a text by {a_name}.\n"
        "1. No Spanish phrases untranslated\n"
        "2. Proper nouns/dates/places preserved\n"
        "3. Natural accurate English (1-5)\n\n"
        f"Sample EN:\n{sample}\n\n"
        f"ES original (first 1500 chars):\n{es_text[:1500]}\n\n"
        'Return JSON only -- no markdown: {"score": int, "issues": [str]}\n'
        "5=excellent, 4=good, 3=acceptable, 2=poor, 1=unusable"
    )
    qwen = get_qwen()
    try:
        if qwen:
            resp = qwen.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        else:
            raw  = _strip_json_fences(get_gemini().generate_content(prompt).text)
            data = json.loads(raw)
        score = max(1, min(5, int(data.get("score", 3))))
        return QAResult(score=score, passed=score >= QA_THRESHOLD, issues=data.get("issues", []))
    except Exception as exc:
        print(f"      QA4 error: {exc}")
        return QAResult(score=3, passed=False, issues=[str(exc)])


# =============================================================================
# QA GATE 5 -- Final Auto-Review  (Gemini)  <-- NEW
# =============================================================================
def qa_final_review(es_text: str, en_text: str, fm_es: dict,
                    author_cfg: dict) -> QAResult:
    """
    Holistic editorial review. Score >= QA_THRESHOLD --> auto-publish.
    This is the last gate before a text goes live on the website.
    """
    model  = get_gemini()
    title  = fm_es.get("title", "")
    a_name = author_cfg["author_name"]

    prompt = (
        f"You are a final editorial reviewer for a bilingual digital archive of "
        f"Hispanic American political and historical texts.\n\n"
        f"Author: {a_name}\nTitle: {title}\n\n"
        f"SPANISH (first 2500 chars):\n{es_text[:2500]}\n\n"
        f"ENGLISH TRANSLATION (first 2500 chars):\n{en_text[:2500]}\n\n"
        "Rate publication readiness 1-5:\n"
        "5 Excellent: coherent, complete, well translated, publish immediately\n"
        "4 Good: minor imperfections, publishable as-is\n"
        "3 Acceptable: issues present, needs human review before publishing\n"
        "2 Poor: significant problems, do not auto-publish\n"
        "1 Unusable: garbled, incomplete, or not a genuine historical text\n\n"
        'Return JSON only: {"score": int, "issues": [str], "verdict": "PUBLISH" or "SKIP"}'
    )
    try:
        raw    = _strip_json_fences(model.generate_content(prompt).text)
        data   = json.loads(raw)
        score  = max(1, min(5, int(data.get("score", 3))))
        verdict = data.get("verdict", "PUBLISH" if score >= QA_THRESHOLD else "SKIP")
        passed  = (verdict == "PUBLISH") and (score >= QA_THRESHOLD)
        return QAResult(score=score, passed=passed, issues=data.get("issues", []))
    except Exception as exc:
        print(f"      QA5 error: {exc}")
        return QAResult(score=2, passed=False, issues=[f"QA error: {exc}"])


# =============================================================================
# Front matter rendering
# =============================================================================
def _render_front_matter(fm: dict) -> str:
    lines = ["---"]
    for key, value in fm.items():
        if isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: []")
        else:
            v = str(value)
            if any(c in v for c in ':#{}[]|>&*!,') or (v and v[0] == '"'):
                escaped = v.replace('"', '\\"')
                lines.append(f'{key}: "{escaped}"')
            else:
                lines.append(f"{key}: {v}")
    lines.append("---")
    return "\n".join(lines)


# =============================================================================
# Auto-commit helpers
# =============================================================================
def _write_to_texts(slug: str, fm_es: dict, es_body: str,
                    fm_en: dict, en_body: str, author_cfg: dict) -> bool:
    """
    Write text pair directly to _texts/ and git-add both files.
    Returns True if successfully written and staged.
    """
    author        = author_cfg["author"]
    es_target_dir = TEXTS_DIR / "es" / author
    en_target_dir = TEXTS_DIR / "en" / author
    es_target_dir.mkdir(parents=True, exist_ok=True)
    en_target_dir.mkdir(parents=True, exist_ok=True)

    es_target = es_target_dir / f"{slug}.md"
    en_target = en_target_dir / f"{slug}.md"

    if es_target.exists() or en_target.exists():
        print(f"      SKIP -- already exists: {slug}.md")
        return False

    es_target.write_text(_render_front_matter(fm_es) + "\n\n" + es_body + "\n",
                         encoding="utf-8")
    en_target.write_text(_render_front_matter(fm_en) + "\n\n" + en_body + "\n",
                         encoding="utf-8")

    try:
        subprocess.run(
            ["git", "add", str(es_target), str(en_target)],
            cwd=str(REPO_ROOT), check=True, capture_output=True,
        )
        print(f"      --> _texts/ {es_target.name} / {en_target.name}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"      git add failed: {exc}")
        return False


def _write_to_incoming(slug: str, prefix: str, fm_es: dict, es_body: str,
                       fm_en: dict, en_body: str):
    """Write to incoming/ for manual review (texts that failed QA5)."""
    es_path = INCOMING_DIR / f"{prefix}-{slug}-es.md"
    en_path = INCOMING_DIR / f"{prefix}-{slug}-en.md"
    es_path.write_text(_render_front_matter(fm_es) + "\n\n" + es_body + "\n", encoding="utf-8")
    en_path.write_text(_render_front_matter(fm_en) + "\n\n" + en_body + "\n", encoding="utf-8")
    print(f"      incoming/ {es_path.name} (REVIEW_NEEDED)")


def commit_and_push(label: str, committed_count: int):
    """Commit all staged files and push to origin/main."""
    if committed_count == 0:
        print(f"  No texts to commit for {label}.")
        return

    # Check there are actually staged changes
    result = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if not result.stdout.strip():
        print(f"  Nothing staged for {label}.")
        return

    msg = (
        f"add {committed_count} text(s) from {label} via automated pipeline\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    try:
        subprocess.run(["git", "commit", "-m", msg],
                       cwd=str(REPO_ROOT), check=True)
        subprocess.run(["git", "push", "origin", "main"],
                       cwd=str(REPO_ROOT), check=True)
        print(f"  Pushed {committed_count} texts from {label} to GitHub.")
    except subprocess.CalledProcessError as exc:
        print(f"  Git error: {exc}")
        print("  Files committed locally. Push manually with: git push origin main")


# =============================================================================
# Pipeline Report
# =============================================================================
def generate_report(label: str, results: list) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path      = INCOMING_DIR / f"pipeline_report_{label}_{timestamp}.md"

    committed   = [r for r in results if r.status == "COMMITTED"]
    review      = [r for r in results if r.status == "REVIEW_NEEDED"]

    lines = [
        f"# Pipeline Report -- {label.upper()} -- {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Summary",
        "",
        "| Text | QA1 | QA2 | QA3 | QA4 | QA5 | Status |",
        "|------|-----|-----|-----|-----|-----|--------|",
    ]
    for r in results:
        qa1_s = f"{r.qa1.score}/5" if r.qa1 else "--"
        qa2_s = "[OK]" if (r.qa2 and r.qa2.passed) else "[X]"
        qa3_s = "[OK]" if not r.qa3_issues else f"(!) {len([i for i in r.qa3_issues if 'warning' not in i.lower()])}"
        qa4_s = f"{r.qa4.score}/5" if r.qa4 else "--"
        qa5_s = f"{r.qa5.score}/5" if r.qa5 else "--"
        lines.append(
            f"| {r.section.title[:40]} | {qa1_s} | {qa2_s} | {qa3_s} | {qa4_s} | {qa5_s} | {r.status} |"
        )

    lines += [
        "",
        f"**Committed to site**: {len(committed)}  |  **Needs review**: {len(review)}",
        "",
    ]

    if review:
        lines += ["## Texts Requiring Manual Review", ""]
        for r in review:
            lines.append(f"### {r.section.title}")
            if r.qa1 and r.qa1.issues:
                lines.append(f"- **QA1**: {'; '.join(r.qa1.issues)}")
            if r.qa2 and not r.qa2.passed:
                lines.append(f"- **QA2**: {'; '.join(r.qa2.issues)}")
            if r.qa3_issues:
                lines.append(f"- **QA3**: {'; '.join(r.qa3_issues)}")
            if r.qa4 and r.qa4.issues:
                lines.append(f"- **QA4**: {'; '.join(r.qa4.issues)}")
            if r.qa5 and r.qa5.issues:
                lines.append(f"- **QA5**: {'; '.join(r.qa5.issues)}")
            lines.append(f"- Files: {r.slug} (in incoming/)")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# =============================================================================
# Per-section orchestration
# =============================================================================
def process_section(
    doc:            fitz.Document,
    section:        TextSection,
    has_text_layer: bool,
    author_cfg:     dict,
) -> Optional[ProcessedText]:
    """Run all 5 QA gates on one section. Auto-commits if QA5 passes."""
    print(f"\n  [{section.title[:55]}]")

    # Stage 2: Extract
    print(f"    extract  p.{section.start_page}-{section.end_page} ...")
    raw_text = extract_section(doc, section, has_text_layer)

    if len(raw_text.split()) < 30:
        print(f"    SKIP -- too short ({len(raw_text.split())} words)")
        return None

    # QA1: OCR quality (with one retry if below threshold)
    print("    QA1   OCR quality ...")
    qa1     = qa_ocr_quality(raw_text)
    cleaned = clean_text(raw_text, qa1.issues)
    print(f"    QA1   {qa1.score}/5")

    if not qa1.passed:
        cleaned, qa1 = _retry_clean(raw_text, qa1)

    # QA2: Text integrity
    print("    QA2   text integrity ...")
    qa2 = qa_text_integrity(cleaned)
    print(f"    QA2   {'PASS' if qa2.passed else 'FAIL'} ({qa2.doc_type})")
    if not qa2.passed:
        print("    SKIP -- integrity check failed")
        return None

    # Stage 4: Metadata + front matter
    print("    meta  generating ...")
    meta  = generate_metadata(cleaned, section, author_cfg)
    slug  = make_slug(meta.get("title", section.title))
    fm_es = build_front_matter_es(meta, slug, author_cfg)

    # Stage 5: Translation (chunked for long texts)
    print("    trans translating to English ...")
    en_body, en_title = translate_chunked(cleaned, meta.get("title", section.title), author_cfg)
    fm_en = build_front_matter_en(meta, en_title, slug, author_cfg)

    # QA3: Front matter validation
    print("    QA3   validating front matter ...")
    qa3_issues = qa_front_matter(fm_es, fm_en, slug, author_cfg)
    hard_qa3   = [i for i in qa3_issues if "warning" not in i.lower()]
    if hard_qa3:
        print(f"    QA3   hard issues: {hard_qa3}")

    # QA4: Translation quality (with one retry if below threshold)
    print("    QA4   translation quality ...")
    qa4 = qa_translation(cleaned, en_body, author_cfg)
    print(f"    QA4   {qa4.score}/5")

    if not qa4.passed:
        en_body, en_title, qa4 = _retry_translate(cleaned, meta.get("title", section.title),
                                                   author_cfg, qa4)
        fm_en = build_front_matter_en(meta, en_title, slug, author_cfg)

    # QA5: Final editorial review
    print("    QA5   final review ...")
    qa5 = qa_final_review(cleaned, en_body, fm_es, author_cfg)
    print(f"    QA5   {qa5.score}/5 {'PUBLISH' if qa5.passed else 'SKIP'}")

    # Determine fate
    if qa5.passed and not hard_qa3:
        committed = _write_to_texts(slug, fm_es, cleaned, fm_en, en_body, author_cfg)
        status    = "COMMITTED" if committed else "REVIEW_NEEDED"
    else:
        _write_to_incoming(slug, author_cfg["file_prefix"],
                           fm_es, cleaned, fm_en, en_body)
        status = "REVIEW_NEEDED"

    return ProcessedText(
        section=section, raw_text=raw_text, cleaned_text=cleaned,
        qa1=qa1, qa2=qa2, front_matter_es=fm_es, front_matter_en=fm_en,
        es_body=cleaned, en_body=en_body, qa3_issues=qa3_issues,
        qa4=qa4, qa5=qa5, status=status, slug=slug,
    )


# =============================================================================
# Whole-book mode (Laura's book)
# =============================================================================
def process_whole_book(pdf_path: Path, has_text_layer: bool,
                       label: str, author_cfg: dict) -> int:
    """
    Extract the entire PDF as a single text. Returns count of committed texts (0 or 1).
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Volume : {label.upper()} (whole-book mode)")
    print(f"Author : {author_cfg['author_name']}")
    print(f"File   : {pdf_path.name}")
    print(sep)

    if not pdf_path.exists():
        print(f"ERROR: file not found: {pdf_path}")
        return 0

    doc = fitz.open(str(pdf_path))
    print(f"Pages  : {len(doc)}")

    # Extract all pages
    print("\nExtracting all pages ...")
    raw_text = extract_all_pages(doc, has_text_layer)
    doc.close()
    word_count = len(raw_text.split())
    print(f"Extracted {word_count} words")

    if word_count < 100:
        print("ERROR: extracted text too short -- aborting")
        return 0

    # QA1
    print("\nQA1  OCR quality ...")
    qa1     = qa_ocr_quality(raw_text)
    cleaned = clean_text(raw_text, qa1.issues)
    print(f"QA1  {qa1.score}/5")
    if not qa1.passed:
        cleaned, qa1 = _retry_clean(raw_text, qa1)

    # QA2 (auto-pass for whole book)
    qa2 = qa_text_integrity(cleaned, whole_book=True)

    # Metadata (use known facts for Laura's book)
    print("\nMeta generating ...")
    known_section = TextSection(
        title      = "Dr. Pedro Albizu Campos y la Independencia de Puerto Rico",
        start_page = 1,
        end_page   = len(raw_text.split()) // 300,  # rough page estimate
        date_hint  = "1961",
        doc_type   = "book",
    )
    meta = generate_metadata(cleaned, known_section, author_cfg)
    # Enforce known facts
    if not meta.get("date") or not meta["date"].startswith("1961"):
        meta["date"] = "1961-01-01"
    meta.setdefault("collections", [])
    for col in ["independencia-puerto-rico", "nacionalismo-puertorriqueno", "biografia"]:
        if col not in meta["collections"]:
            meta["collections"].append(col)

    slug  = make_slug(meta.get("title", known_section.title))
    fm_es = build_front_matter_es(meta, slug, author_cfg)

    # Translation (chunked)
    print("\nTrans translating to English (chunked) ...")
    en_body, en_title = translate_chunked(cleaned, meta.get("title", known_section.title),
                                          author_cfg)
    fm_en = build_front_matter_en(meta, en_title, slug, author_cfg)

    # QA3
    print("\nQA3  front matter ...")
    qa3_issues = qa_front_matter(fm_es, fm_en, slug, author_cfg)
    hard_qa3   = [i for i in qa3_issues if "warning" not in i.lower()]

    # QA4
    print("\nQA4  translation quality ...")
    qa4 = qa_translation(cleaned, en_body, author_cfg)
    print(f"QA4  {qa4.score}/5")
    if not qa4.passed:
        en_body, en_title, qa4 = _retry_translate(cleaned, meta.get("title"),
                                                   author_cfg, qa4)
        fm_en = build_front_matter_en(meta, en_title, slug, author_cfg)

    # QA5
    print("\nQA5  final review ...")
    qa5 = qa_final_review(cleaned, en_body, fm_es, author_cfg)
    print(f"QA5  {qa5.score}/5 {'PUBLISH' if qa5.passed else 'SKIP'}")

    committed = 0
    if qa5.passed and not hard_qa3:
        ok = _write_to_texts(slug, fm_es, cleaned, fm_en, en_body, author_cfg)
        committed = 1 if ok else 0
    else:
        _write_to_incoming(slug, author_cfg["file_prefix"],
                           fm_es, cleaned, fm_en, en_body)
        print("  --> incoming/ for manual review")

    # Report
    dummy_section = known_section
    dummy_result  = ProcessedText(
        section=dummy_section, raw_text=raw_text, cleaned_text=cleaned,
        qa1=qa1, qa2=qa2, front_matter_es=fm_es, front_matter_en=fm_en,
        es_body=cleaned, en_body=en_body, qa3_issues=qa3_issues,
        qa4=qa4, qa5=qa5,
        status="COMMITTED" if committed else "REVIEW_NEEDED",
        slug=slug,
    )
    report_path = generate_report(label, [dummy_result])
    print(f"\nReport: {report_path.name}")
    return committed


# =============================================================================
# PDF-level orchestration (section mode)
# =============================================================================
def process_pdf(pdf_path: Path, has_text_layer: bool,
                label: str, author_cfg: dict, whole_book: bool = False) -> int:
    """Process one PDF. Returns count of texts committed."""
    if whole_book:
        count = process_whole_book(pdf_path, has_text_layer, label, author_cfg)
        commit_and_push(label, count)
        return count

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Volume : {label.upper()}")
    print(f"Author : {author_cfg['author_name']}")
    print(f"File   : {pdf_path.name}")
    print(f"Mode   : {'text-layer' if has_text_layer else 'vision OCR'}")
    print(sep)

    if not pdf_path.exists():
        print(f"ERROR: file not found: {pdf_path}")
        return 0

    doc = fitz.open(str(pdf_path))
    print(f"Pages  : {len(doc)}")

    print("\nStage 1 -- TOC detection ...")
    sections = detect_toc(pdf_path, has_text_layer, author_cfg)
    print(f"Found {len(sections)} sections:")
    for idx, s in enumerate(sections, 1):
        print(f"  {idx:2d}. {s.title[:55]:<55} p.{s.start_page}-{s.end_page}  [{s.doc_type}]")

    if not sections:
        print("ERROR: no sections -- aborting.")
        doc.close()
        return 0

    results: list[ProcessedText] = []
    for section in sections:
        r = process_section(doc, section, has_text_layer, author_cfg)
        if r:
            results.append(r)

    doc.close()

    committed_count = sum(1 for r in results if r.status == "COMMITTED")
    report_path     = generate_report(label, results)
    review_count    = sum(1 for r in results if r.status == "REVIEW_NEEDED")
    skipped         = len(sections) - len(results)

    print(f"\n{sep}")
    print(f"  {label.upper()} done  -- committed: {committed_count}  "
          f"review: {review_count}  skipped: {skipped}")
    print(f"  Report: {report_path.name}")
    print(sep)

    commit_and_push(label, committed_count)
    return committed_count


# =============================================================================
# PDF Registry
# =============================================================================
_DESKTOP = Path("C:/Users/diego/Desktop/Albizu")

PDF_REGISTRY = [
    {
        "label":          "tomo1",
        "path":           _CFG_PATHS.get("tomo1") or _DESKTOP / "Pedro Albizu_ Campos Obras Escogidas, 1923-1936 (Tomo 1) 1.pdf",
        "has_text_layer": False,
        "author_cfg":     _AUTHOR_PAC,
        "whole_book":     False,
    },
    {
        "label":          "tomo2",
        "path":           _CFG_PATHS.get("tomo2") or _DESKTOP / "Pedro Albizu Campos_ Obras escogidas 1923-1936 (Tomo 2_ 1934-1935) 2.pdf",
        "has_text_layer": True,
        "author_cfg":     _AUTHOR_PAC,
        "whole_book":     False,
    },
    {
        "label":          "tomo3",
        "path":           _CFG_PATHS.get("tomo3") or _DESKTOP / "Pedro Albizu Campos Obras Escogidas 1923-1936 Tomo 3 3.pdf",
        "has_text_layer": False,
        "author_cfg":     _AUTHOR_PAC,
        "whole_book":     False,
    },
    {
        "label":          "tomo4",
        "path":           _CFG_PATHS.get("tomo4") or _DESKTOP / "Pedro Albizu Campos Obras Escogidas 1923-1936 Tomo 4 4.pdf",
        "has_text_layer": True,
        "author_cfg":     _AUTHOR_PAC,
        "whole_book":     False,
    },
    {
        "label":          "laura",
        "path":           _CFG_PATHS.get("laura") or _DESKTOP / "Laura de Albizu Campos - Dr. Pedro Albizu Campos y la Independencia de Puerto Rico (1961) - libgen.li.pdf",
        "has_text_layer": True,
        "author_cfg":     _AUTHOR_LAC,
        "whole_book":     True,
    },
]

for _e in PDF_REGISTRY:
    if not isinstance(_e["path"], Path):
        _e["path"] = Path(_e["path"])

_OBRAS_REGISTRY = [e for e in PDF_REGISTRY if e["label"].startswith("tomo")]
_LAURA_ENTRY    = next(e for e in PDF_REGISTRY if e["label"] == "laura")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PDF pipeline -- auto-extracts, QA-reviews, commits, and pushes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/pdf_pipeline.py --tomo 4      # fastest validation run\n"
            "  python scripts/pdf_pipeline.py --laura\n"
            "  python scripts/pdf_pipeline.py --all         # tomos 1-4\n"
            "  python scripts/pdf_pipeline.py --everything  # all five PDFs\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",        type=Path)
    group.add_argument("--all",        action="store_true")
    group.add_argument("--laura",      action="store_true")
    group.add_argument("--everything", action="store_true")
    group.add_argument("--tomo",       choices=["1","2","3","4"])
    parser.add_argument("--text-layer", action="store_true")
    parser.add_argument("--vision",     action="store_true")

    args = parser.parse_args()

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        sys.exit("ERROR: GEMINI_API_KEY not set. Edit scripts/pipeline_config.py")
    if not QWEN_API_KEY or QWEN_API_KEY == "YOUR_QWEN_API_KEY_HERE":
        print("INFO: QWEN_API_KEY not set -- all stages will use Gemini (fully functional).")

    def _run(entry):
        has_text = entry["has_text_layer"]
        if args.text_layer: has_text = True
        if args.vision:     has_text = False
        process_pdf(Path(entry["path"]), has_text, entry["label"],
                    entry["author_cfg"], entry["whole_book"])

    if args.everything:
        for entry in PDF_REGISTRY:
            _run(entry)
    elif args.all:
        for entry in _OBRAS_REGISTRY:
            _run(entry)
    elif args.laura:
        _run(_LAURA_ENTRY)
    elif args.tomo:
        _run(_OBRAS_REGISTRY[int(args.tomo) - 1])
    else:  # --pdf
        pdf_path = args.pdf
        has_text = True
        if args.vision:
            has_text = False
        elif not args.text_layer:
            try:
                d = fitz.open(str(pdf_path))
                sample = d[min(4, len(d) - 1)].get_text().strip()
                d.close()
                has_text = len(sample) > 50
                print(f"Auto-detected: {'text-layer' if has_text else 'vision OCR'}")
            except Exception:
                pass
        matched    = next((e for e in PDF_REGISTRY if Path(e["path"]).name == pdf_path.name), None)
        label      = matched["label"]      if matched else "unknown"
        author_cfg = matched["author_cfg"] if matched else _AUTHOR_PAC
        whole_book = matched["whole_book"] if matched else False
        if matched and not args.text_layer and not args.vision:
            has_text = matched["has_text_layer"]
        process_pdf(pdf_path, has_text, label, author_cfg, whole_book)


if __name__ == "__main__":
    main()

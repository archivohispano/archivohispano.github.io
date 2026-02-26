#!/usr/bin/env python3
"""
Tomo 4 targeted extraction — missing and review-needed texts.

Tomo IV covers June–July 1936 (Editorial Claves Latinoamericanas, México D.F., 1987).

PDF page mapping (empirically measured):
  The PDF contains two-page spreads: each PDF scan contains two book pages.
  Calibration points:
    book p.19 → PDF p.11  (Conferencia en el Teatro Municipal starts)
    book p.21 → PDF p.12  (confirmed from "OBRAS ESCOGIDAS 21" header on PDF p.12)
    book p.131 → PDF p.67 (confirmed from "OBRAS ESCOGIDAS 131" header on PDF p.67)
    book p.133 → PDF p.68 (confirmed: "En la cárcel" starts at PDF p.68)

  Formula: pdf_page_1indexed = round(11 + (book_page - 19) / 2)
  Or 0-indexed: pdf_idx = pdf_page_1indexed - 1

  PDF total pages: 72

Usage:
    python scripts/tomo4_extract.py                # all sections
    python scripts/tomo4_extract.py --only 1 2 3   # specific section indices
    python scripts/tomo4_extract.py --dry-run       # preview page ranges
    python scripts/tomo4_extract.py --skip-conferencia  # skip the 24-page conference
"""

import argparse
import io
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from pipeline_config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

import fitz
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PDF_PATH = Path(r"C:\Users\diego\Desktop\Albizu\Pedro Albizu Campos Obras Escogidas 1923-1936 Tomo 4 4.pdf")
PDF_TOTAL = 72   # total pages in the PDF

REPO_ROOT = Path(r"C:\Users\diego\Documents\GitHub\archivohispano.github.io")
ES_DIR    = REPO_ROOT / "_texts" / "es" / "pedro-albizu-campos"
EN_DIR    = REPO_ROOT / "_texts" / "en" / "pedro-albizu-campos"

SOURCE_DETAIL = "Pedro Albizu Campos, Obras Escogidas 1923-1936, Tomo IV (México D.F.: Editorial Claves Latinoamericanas, 1987)"
SOURCE_DETAIL_EN = SOURCE_DETAIL  # same citation in EN front matter

# ---------------------------------------------------------------------------
# TOC for Tomo 4 (entries we need to process)
# Each entry: (index, title_es, title_en, book_start, book_end_excl, date, slug, collections, notes)
# book_end_excl: book page where the NEXT section begins (for page range calculation)
# ---------------------------------------------------------------------------
SECTIONS = [
    {
        "idx": 1,
        "title_es": "Conferencia en el Teatro Municipal",
        "title_en": "Conference at the Teatro Municipal",
        "book_start": 19,
        "book_end_excl": 43,  # next section starts at 43
        "date": "1936-05-20",
        "slug": "conferencia-teatro-municipal",
        "collections": ["independencia-puerto-rico", "nacionalismo-puertorriqueno", "discursos"],
        "notes": "Pronounced May 20, 1936 at Teatro Municipal (today Teatro Tapia). Source: La Palabra, May 25 – July 6, 1936.",
        "skip": False,
    },
    {
        "idx": 2,
        "title_es": "Pedro Albizu Campos: El discurso del Coronel Enrique De Orbeta",
        "title_en": "Pedro Albizu Campos: The Speech of Colonel Enrique De Orbeta",
        "book_start": 43,
        "book_end_excl": 51,  # next section starts at 51
        "date": "1936-06-01",
        "slug": "el-discurso-del-coronel-enrique-de-orbeta",
        "collections": ["independencia-puerto-rico", "nacionalismo-puertorriqueno"],
        "notes": "Approximate date June 1936.",
        "skip": False,
    },
    {
        "idx": 3,
        "title_es": "Pedro Albizu Campos: El absurdo de la estadidad para Puerto Rico",
        "title_en": "Pedro Albizu Campos: The Absurdity of Statehood for Puerto Rico",
        "book_start": 55,
        "book_end_excl": 61,  # next section starts at 61
        "date": "1936-06-01",
        "slug": "el-absurdo-de-la-estadidad-para-puerto-rico",
        "collections": ["independencia-puerto-rico", "nacionalismo-puertorriqueno"],
        "notes": "El Mundo, July 2, 1936.",
        "skip": False,
    },
    {
        "idx": 4,
        "title_es": "Carta a Marta Lomar (11 de junio de 1936)",
        "title_en": "Letter to Marta Lomar (June 11, 1936)",
        "book_start": 61,
        "book_end_excl": 65,  # next section starts at 65
        "date": "1936-06-11",
        "slug": "carta-marta-lomar-11-junio-1936",
        "collections": ["independencia-puerto-rico", "cartas"],
        "notes": "El Mundo, July 15, 1936.",
        "skip": False,
    },
    {
        "idx": 5,
        "title_es": "Pedro Albizu Campos: El proceso judicial contra el liderato nacionalista",
        "title_en": "Pedro Albizu Campos: The Judicial Process against the Nationalist Leadership",
        "book_start": 69,
        "book_end_excl": 73,  # next section starts at 73
        "date": "1936-07-01",
        "slug": "proceso-judicial-liderato-nacionalista",
        "collections": ["independencia-puerto-rico", "nacionalismo-puertorriqueno"],
        "notes": "Approximate date July 1936.",
        "skip": False,
    },
    {
        "idx": 6,
        "title_es": '"En la cárcel o frente a la muerte se renuevan los votos de la consagración"',
        "title_en": '"In prison or facing death, the vows of consecration are renewed"',
        "book_start": 133,
        "book_end_excl": 137,  # estimated end (notes section begins)
        "date": "1936-10-10",
        "slug": "en-la-carcel-o-frente-a-la-muerte",
        "collections": ["independencia-puerto-rico", "nacionalismo-puertorriqueno", "carcel-princesa"],
        "notes": "Written from the Cárcel de San Juan, October 10, 1936, day of the bail denial. El Mundo, July 25, 1936, p. 1 y 17.",
        "skip": False,
    },
]

# Sections already committed — skip these
COMMITTED_SLUGS = {
    "carta-barcelo",        # Carta a Antonio R. Barceló
    "carta-marta-lomar-25-junio-1936",  # Carta a Marta Lomar (25 jun)
    "ante-la-persecucion-aconsejamos-serenidad",
}


# ---------------------------------------------------------------------------
# Page offset formula
# ---------------------------------------------------------------------------
def book_to_pdf_0idx(book_page: int) -> int:
    """Convert book page (1-indexed) to PDF page (0-indexed)."""
    # Formula calibrated from empirical measurements:
    #   book p.19 → PDF p.11 (1-indexed) → idx 10
    #   book p.133 → PDF p.68 (1-indexed) → idx 67
    # Each PDF page = 2 book pages (two-page spreads)
    pdf_1idx = round(11 + (book_page - 19) / 2)
    pdf_1idx = max(1, min(PDF_TOTAL, pdf_1idx))
    return pdf_1idx - 1   # convert to 0-indexed


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
OCR_PROMPT = (
    "Transcribe all Spanish text exactly as it appears on this page. "
    "Preserve paragraph breaks, dialogue dashes, and all formatting. "
    "If the page contains a header like 'OBRAS ESCOGIDAS' followed by a page number, "
    "omit that header line. "
    "Do not translate, summarize, or add commentary. "
    "Output only the transcribed Spanish text."
)

TRANSLATE_PROMPT_TEMPLATE = (
    "Translate the following Spanish text into fluent English. "
    "Preserve all paragraph breaks, proper names, and formatting. "
    "This is a historical political text by Pedro Albizu Campos, Puerto Rican independence leader (1891-1965). "
    "Preserve any dialogue dashes (—). "
    "Output only the English translation.\n\n"
    "SPANISH TEXT:\n{text}"
)


def ocr_page(doc, page_0idx: int) -> str:
    page = doc[page_0idx]
    mat  = fitz.Matrix(2.0, 2.0)
    pix  = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img  = Image.open(io.BytesIO(img_bytes))
    try:
        resp = model.generate_content([OCR_PROMPT, img])
        if resp.candidates:
            return resp.text.strip()
        return f"[Page {page_0idx+1} blocked by safety filter]"
    except Exception as e:
        return f"[Page {page_0idx+1} error: {e}]"


def ocr_section(doc, section: dict, window: int = 1) -> str:
    """OCR a range of PDF pages for the given section, with a ±window buffer."""
    bp_start = section["book_start"]
    bp_end   = section["book_end_excl"]

    idx_start = max(0,          book_to_pdf_0idx(bp_start) - window)
    idx_end   = min(PDF_TOTAL - 1, book_to_pdf_0idx(bp_end) + window)

    print(f"  OCR: PDF pages {idx_start+1}–{idx_end+1} "
          f"(book pp.{bp_start}–{bp_end-1})", flush=True)

    pages = []
    for i in range(idx_start, idx_end + 1):
        print(f"    page {i+1} ...", flush=True)
        text = ocr_page(doc, i)
        pages.append(f"--- PDF PAGE {i+1} ---\n{text}")
        time.sleep(0.8)

    return "\n\n".join(pages)


def clean_ocr(raw: str) -> str:
    """Light clean-up: remove PDF page markers, collapse blank lines."""
    lines   = raw.splitlines()
    cleaned = [l for l in lines if not l.startswith("--- PDF PAGE ")]
    body    = "\n".join(cleaned).strip()
    body    = re.sub(r'\n{3,}', '\n\n', body)
    return body


def translate_text(es_text: str) -> str:
    prompt = TRANSLATE_PROMPT_TEMPLATE.format(text=es_text)
    print("  Translating to English ...", flush=True)
    try:
        resp = model.generate_content(prompt)
        if resp.candidates:
            return resp.text.strip()
        return "[Translation blocked by safety filter]"
    except Exception as e:
        return f"[Translation error: {e}]"


# ---------------------------------------------------------------------------
# Front matter builders
# ---------------------------------------------------------------------------
def build_es_fm(section: dict) -> str:
    slug = section["slug"]
    collections_yaml = "\n".join(f"  - {c}" for c in section["collections"])
    return (
        f"---\n"
        f"layout: text\n"
        f"lang: es\n"
        f'title: "{section["title_es"]}"\n'
        f"author: pedro-albizu-campos\n"
        f"author_name: Pedro Albizu Campos\n"
        f"date: {section['date']}\n"
        f"source: Obras Escogidas 1923-1936\n"
        f'source_detail: "{SOURCE_DETAIL}"\n'
        f"country: puerto-rico\n"
        f"permalink: /es/textos/pedro-albizu-campos/{slug}\n"
        f"english_version: /en/texts/pedro-albizu-campos/{slug}\n"
        f"collections:\n"
        f"{collections_yaml}\n"
        f"---"
    )


def build_en_fm(section: dict) -> str:
    slug = section["slug"]
    collections_yaml = "\n".join(f"  - {c}" for c in section["collections"])
    return (
        f"---\n"
        f"layout: text\n"
        f"lang: en\n"
        f'title: "{section["title_en"]}"\n'
        f"author: pedro-albizu-campos\n"
        f"author_name: Pedro Albizu Campos\n"
        f"date: {section['date']}\n"
        f"source: Obras Escogidas 1923-1936 (Selected Works)\n"
        f'source_detail: "{SOURCE_DETAIL_EN}"\n'
        f"country: puerto-rico\n"
        f"permalink: /en/texts/pedro-albizu-campos/{slug}\n"
        f"spanish_version: /es/textos/pedro-albizu-campos/{slug}\n"
        f"collections:\n"
        f"{collections_yaml}\n"
        f"---"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract missing Tomo 4 texts via Vision OCR")
    ap.add_argument("--only", type=int, nargs="+",
                    help="Only process these section indices")
    ap.add_argument("--skip-conferencia", action="store_true",
                    help="Skip section 1 (Conferencia, 24 pages)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print page ranges without calling APIs")
    args = ap.parse_args()

    to_process = [
        s for s in SECTIONS
        if not s["skip"]
        and (not args.only or s["idx"] in args.only)
        and (not args.skip_conferencia or s["idx"] != 1)
    ]

    if args.dry_run:
        print(f"\nTomo 4 — {len(to_process)} sections to process\n")
        for s in to_process:
            bp_s = s["book_start"]
            bp_e = s["book_end_excl"]
            p_s  = book_to_pdf_0idx(bp_s) + 1 - 1  # 1-indexed with buffer
            p_e  = book_to_pdf_0idx(bp_e) + 1 + 1
            print(f"  [{s['idx']}] book pp.{bp_s}-{bp_e-1}  "
                  f"-> PDF pp.{book_to_pdf_0idx(bp_s)+1}-{book_to_pdf_0idx(bp_e)+1}  "
                  f"  {s['title_es'][:55]}")
        return

    print(f"Opening PDF: {PDF_PATH}", flush=True)
    doc = fitz.open(str(PDF_PATH))
    print(f"PDF pages: {len(doc)}", flush=True)

    results = []
    for section in to_process:
        print(f"\n{'='*60}", flush=True)
        print(f"[{section['idx']}] {section['title_es']}", flush=True)

        # OCR
        raw = ocr_section(doc, section)

        # Save raw OCR to incoming/ for debugging
        raw_path = REPO_ROOT / "incoming" / f"tomo4_raw_{section['slug']}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        print(f"  Raw OCR saved: {raw_path.name}", flush=True)

        # Clean
        body_es = clean_ocr(raw)

        # Translate
        time.sleep(1.0)
        body_en = translate_text(body_es)

        # Write ES
        es_fm   = build_es_fm(section)
        es_path = ES_DIR / f"{section['slug']}.md"
        es_path.write_text(es_fm + "\n\n" + body_es + "\n", encoding="utf-8")
        print(f"  ES -> {es_path.name}", flush=True)

        # Write EN
        en_fm   = build_en_fm(section)
        en_path = EN_DIR / f"{section['slug']}.md"
        en_path.write_text(en_fm + "\n\n" + body_en + "\n", encoding="utf-8")
        print(f"  EN -> {en_path.name}", flush=True)

        results.append(section["slug"])
        time.sleep(1.0)

    doc.close()

    print(f"\n{'='*60}", flush=True)
    print(f"Done — {len(results)} section(s) written:", flush=True)
    for slug in results:
        print(f"  {slug}", flush=True)


if __name__ == "__main__":
    main()

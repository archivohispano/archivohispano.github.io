#!/usr/bin/env python3
"""
Laura de Albizu Campos — full text extraction via pymupdf4llm.

The PDF has 90 pages with a text layer on 88/90 pages (good quality throughout).
pymupdf4llm handles:
  - Soft-hyphen word joins (word\xad + newline → wordcontinuation)
  - Column detection
  - Basic noise filtering

This script adds further cleaning for the OCR artifacts specific to this PDF,
then writes a Jekyll-ready ES markdown file.

Usage:
    python scripts/laura_extract.py
    python scripts/laura_extract.py --raw-only   # stop after raw extraction
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

PDF_PATH = Path("C:/Users/diego/Desktop/Albizu") / \
    "Laura de Albizu Campos - Dr. Pedro Albizu Campos y la Independencia de Puerto Rico (1961) - libgen.li.pdf"

OUTDIR = Path("incoming")

# Raw markdown saved here for inspection / re-runs without re-extracting
RAW_OUT    = OUTDIR / "laura_raw.md"
# Final ES file (ready for manual review then moving to _texts/)
FINAL_OUT  = OUTDIR / "lac-pedro-albizu-campos-independencia-puerto-rico-es.md"

FRONT_MATTER = """\
---
layout: text
lang: es
title: "Dr. Pedro Albizu Campos y la Independencia de Puerto Rico"
author: laura-de-albizu-campos
author_name: Laura de Albizu Campos
date: 1961-01-01
source: "Dr. Pedro Albizu Campos y la Independencia de Puerto Rico"
source_detail: "Laura de Albizu Campos, Dr. Pedro Albizu Campos y la Independencia de Puerto Rico (San Juan, Puerto Rico: Partido Nacionalista de Puerto Rico, 1961)."
country: puerto-rico
permalink: /es/textos/laura-de-albizu-campos/pedro-albizu-campos-independencia-puerto-rico
english_version: /en/texts/laura-de-albizu-campos/pedro-albizu-campos-independencia-puerto-rico
collections:
  - independencia-puerto-rico
  - nacionalismo-puertorriqueno
  - biografia
---
"""

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Decorative / noise characters from scanned ornaments
_NOISE_CHARS = re.compile(r'[■●▪◆•★◄►□▲▼✦≡®©™]+')

# Stray single characters on their own line (initial caps, ornaments OCR'd as letters)
# Keeps lines that are actual Roman numerals (I II III IV V VI) since those are chapter headers
_STRAY_CHAR  = re.compile(r'(?m)^(?!(?:I{1,3}|IV|VI?I?)$)[IiflL1\|\\/*#@~^_]{1,2}\s*$')

# Standalone page numbers on their own line (1–99)
_PAGE_NUM    = re.compile(r'(?m)^\d{1,3}\s*$')

# pymupdf4llm uses "-----" as a page separator; collapse to blank line
_HR          = re.compile(r'\n-----+\n')

# Collapse 3+ consecutive blank lines to 2
_EXCESS_NL   = re.compile(r'\n{3,}')

# Soft hyphen (already handled by pymupdf4llm but belt-and-suspenders)
_SOFT_HYPHEN = re.compile(r'\xad\s*')

# Bad OCR patterns seen in this specific PDF
_BAD_WORDS   = [
    (re.compile(r'\bS5JMAK1®?\b'), ''),         # garbled "SUMARIO"
    (re.compile(r'\bnHDEPENDÉNCIA\b'), 'INDEPENDENCIA'),
    (re.compile(r'\bPUERTO RIC0\b'), 'PUERTO RICO'),
    (re.compile(r'\bWashiAgto[xn]\b'), 'Washington'),
    (re.compile(r'\bcondici[oó]n\s+do\s+formar\s+parto'), 'condición de formar parte'),
    (re.compile(r'\bostsnGament®\b'), 'extensamente'),
    (re.compile(r'\bcaracteríse?as\b'), 'características'),
]


def clean(text: str) -> str:
    """Clean OCR artifacts from pymupdf4llm output."""
    text = _SOFT_HYPHEN.sub('', text)
    text = _NOISE_CHARS.sub('', text)
    text = _STRAY_CHAR.sub('', text)
    text = _PAGE_NUM.sub('', text)
    text = _HR.sub('\n\n', text)
    for pattern, replacement in _BAD_WORDS:
        text = pattern.sub(replacement, text)
    text = _EXCESS_NL.sub('\n\n', text)
    return text.strip()


def find_content_start(text: str) -> int:
    """
    Return the index where the publishable content begins.
    We want to start from the 1961 dedication paragraph (PDF page 4),
    skipping the garbled cover pages.
    """
    markers = [
        'San Juan de Puerto Rico, a 12 de septiembre',   # dedication page
        'El Partido Nacionalista de Puerto Rico publica', # fallback
        'Hay situaciones tan absurdas',                   # first prose sentence
    ]
    for m in markers:
        idx = text.find(m)
        if idx >= 0:
            para = text.rfind('\n\n', 0, idx)
            return max(0, para)
    return 0


def install_pymupdf4llm():
    try:
        import pymupdf4llm
        return pymupdf4llm
    except ImportError:
        print("pymupdf4llm not found — installing (pip install pymupdf4llm)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pymupdf4llm", "-q"],
            stdout=subprocess.DEVNULL,
        )
        print("Installed.")
        import pymupdf4llm
        return pymupdf4llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-only", action="store_true",
                        help="Extract raw markdown and stop (no cleaning/front matter)")
    parser.add_argument("--from-raw", action="store_true",
                        help="Skip extraction, re-use existing incoming/laura_raw.md")
    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)

    # ── Step 1: extract ────────────────────────────────────────────────────────
    if args.from_raw:
        if not RAW_OUT.exists():
            sys.exit(f"Error: {RAW_OUT} not found — run without --from-raw first.")
        raw = RAW_OUT.read_text(encoding='utf-8')
        print(f"Loaded raw from {RAW_OUT} ({len(raw.split())} words)")
    else:
        m4llm = install_pymupdf4llm()
        if not PDF_PATH.exists():
            sys.exit(f"Error: PDF not found at {PDF_PATH}")
        print(f"Extracting {PDF_PATH.name} ...")
        raw = m4llm.to_markdown(str(PDF_PATH))
        RAW_OUT.write_text(raw, encoding='utf-8')
        print(f"Raw extraction: {len(raw.split())} words -> {RAW_OUT}")

    if args.raw_only:
        print("--raw-only set, stopping here.")
        return

    # ── Step 2: clean ──────────────────────────────────────────────────────────
    cleaned = clean(raw)
    start   = find_content_start(cleaned)
    body    = cleaned[start:]

    word_count = len(body.split())
    print(f"After cleaning: {word_count} words")

    if word_count < 20_000:
        print(f"WARNING: expected ~28,668 words, got {word_count}.")
        print(f"Inspect {RAW_OUT} to diagnose. Writing output anyway.")

    # ── Step 3: write final file ───────────────────────────────────────────────
    FINAL_OUT.write_text(FRONT_MATTER + "\n" + body + "\n", encoding='utf-8')
    print(f"Written: {FINAL_OUT}")
    print()
    print("Next steps:")
    print("  1. Review the file — check for remaining OCR noise and chapter breaks")
    print("  2. Optionally split into per-chapter files (6 chapters, p.9/17/25/53/75/81)")
    print("  3. Run translation pipeline for EN version")
    print("  4. Move to _texts/es/laura-de-albizu-campos/ and _texts/en/laura-de-albizu-campos/")


if __name__ == "__main__":
    main()

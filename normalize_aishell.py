#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AISHELL Transcript Normalization
--------------------------------
Input : aishell_transcript_test.txt
Output: aishell_transcript_test_normalized.txt

Normalization steps:
1. Full-width → half-width (Unicode NFKC)
2. Lowercase English
3. Remove punctuation
4. Normalize Chinese numerals → digits (simple mapping)
5. Remove spaces (character-level CER)
"""

import unicodedata
import re
import sys


# ---------- CONFIG ----------
INPUT_FILE  = "/content/data_aishell/transcript/asr_test_hyp_llm_corrected4.txt"
OUTPUT_FILE = "/content/data_aishell/transcript/asr_test_hyp_llm_corrected4_normalized.txt"


# ---------- Chinese numeral mapping (simple, deterministic) ----------
CN_NUM = {
    "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"
}


# ---------- Normalization Functions ----------

def to_halfwidth(text):
    """Convert full-width chars → half-width using Unicode NFKC."""
    return unicodedata.normalize("NFKC", text)


def lowercase_english(text):
    return text.lower()


# Keep: Chinese chars + letters + digits
PUNCT_PATTERN = r"[^\w\s\u4e00-\u9fff]"

def remove_punctuation(text):
    return re.sub(PUNCT_PATTERN, "", text)


def normalize_numbers(text):
    """Simple Chinese digit conversion (一二三 → 123)."""
    return "".join(CN_NUM.get(c, c) for c in text)


def remove_spaces(text):
    return text.replace(" ", "")


def normalize_text(text):
    text = to_halfwidth(text)
    text = lowercase_english(text)
    text = remove_punctuation(text)
    text = normalize_numbers(text)
    text = remove_spaces(text)
    return text


# ---------- Main ----------

def main():
    print("Starting AISHELL normalization...")

    total = 0
    kept = 0
    empty = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            line = line.strip()

            if not line:
                continue

            try:
                utt_id, text = line.split(maxsplit=1)
            except ValueError:
                # malformed line
                continue

            norm_text = normalize_text(text)

            if norm_text == "":
                empty += 1
                continue

            fout.write(f"{utt_id} {norm_text}\n")
            kept += 1

    print("Normalization complete.")
    print(f"Total lines read   : {total}")
    print(f"Valid lines kept   : {kept}")
    print(f"Empty removed      : {empty}")
    print(f"Output written to  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

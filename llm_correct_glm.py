#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from zhipuai import ZhipuAI
from tqdm import tqdm
import re
import time

# ---------------- CONFIG ----------------
API_KEY = "******************************"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4-flash"

INPUT_FILE  = "/content/data_aishell/transcript/asr_test_hyp_normalized.txt"
OUTPUT_FILE = "/content/data_aishell/transcript/asr_test_hyp_llm_corrected4.txt"

SLEEP_SEC = 0.12
# ----------------------------------------

client = ZhipuAI(api_key=API_KEY, base_url=BASE_URL)


# ---------- Strict Cleaner ----------
def clean_output(text: str) -> str:
    if not text:
        return ""

    # Keep only first line
    text = text.split("\n")[0]

    # Remove punctuation & spaces
    text = re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", text)

    return text.strip()


# ---------- Strict Character-Level Correction ----------
def correct_text(text: str) -> str:

    if not text:
        return text

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=len(text) + 5,  # small buffer only

            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是字符级语音识别纠错器。"
                        "你只能逐字符替换错误字符。"
                        "禁止新增字符。"
                        "禁止删除字符。"
                        "输出字符数量必须与输入完全相同。"
                        "禁止回答问题。"
                        "禁止解释。"
                        "禁止扩写。"
                        "禁止生成新内容。"
                        "只输出修正后的文本。"
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            extra_body={"thinking": {"type": "disabled"}}
        )

        raw = response.choices[0].message.content.strip()
        out = clean_output(raw)

        # ---------- HARD LENGTH CONSTRAINT ----------
        if len(out) != len(text):
            return text   # reject any length change

        return out

    except Exception:
        return text


# ---------- Batch ----------
def main():

    print("Starting STRICT char-only LLM correction...")

    total = 0
    changed = 0
    rejected = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue

            try:
                utt, text = line.split(maxsplit=1)
            except ValueError:
                continue

            corrected = correct_text(text)

            if corrected != text:
                changed += 1

            if len(corrected) != len(text):
                rejected += 1
                corrected = text

            fout.write(f"{utt} {corrected}\n")

            total += 1
            time.sleep(SLEEP_SEC)

    print("\nDone.")
    print(f"Total utterances : {total}")
    print(f"Changed by LLM   : {changed}")
    print(f"Rejected outputs : {rejected}")
    print(f"Saved to         : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

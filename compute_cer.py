#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Character Error Rate (CER) computation for AISHELL-style transcripts.

Input format (both REF and HYP):
    <utt_id> <normalized_text>

Requirements:
- Both files must already be normalized
- Character-level (no spaces inside text)
- UTF-8 encoding
"""

# ---------------- CONFIG ----------------
REF_FILE = "data_aishell/transcript/aishell_transcript_test_normalized.txt"
HYP_FILE = "/content/data_aishell/transcript/asr_test_hyp_llm_corrected4_normalized.txt"
PRINT_WORST = 20
# ----------------------------------------


# ---------- Edit Distance with S/D/I ----------
def edit_distance(ref, hyp):
    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    # Backtrack to count S/D/I
    i, j = n, m
    S = D = I = 0

    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
        else:
            if i > 0 and j > 0 and ref[i - 1] != hyp[j - 1]:
                S += 1
            i -= 1
            j -= 1

    return S, D, I


# ---------- Load transcript ----------
def load_transcript(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            utt, txt = parts
            data[utt] = txt

    return data


# ---------- Main ----------
def main():
    print("Loading transcripts...")

    ref = load_transcript(REF_FILE)
    hyp = load_transcript(HYP_FILE)

    print(f"Reference utterances : {len(ref)}")
    print(f"Hypothesis utterances: {len(hyp)}")

    total_S = total_D = total_I = total_N = 0
    missing = 0
    empty_hyp = 0
    per_utt_stats = []

    for utt, ref_text in ref.items():

        if utt not in hyp:
            missing += 1
            continue

        hyp_text = hyp[utt]

        if hyp_text == "":
            empty_hyp += 1

        S, D, I = edit_distance(ref_text, hyp_text)

        N = len(ref_text)
        total_S += S
        total_D += D
        total_I += I
        total_N += N

        cer = (S + D + I) / N if N > 0 else 0
        per_utt_stats.append((cer, utt, ref_text, hyp_text))

    if total_N == 0:
        print("ERROR: No valid reference characters.")
        return

    CER = (total_S + total_D + total_I) / total_N

    print("\n===== CER RESULT =====")
    print(f"CER: {CER * 100:.2f}%")
    print(f"S (substitutions): {total_S}")
    print(f"D (deletions)    : {total_D}")
    print(f"I (insertions)   : {total_I}")
    print(f"N (ref chars)    : {total_N}")

    print("\n===== ALIGNMENT DIAGNOSTICS =====")
    print(f"Missing hypothesis : {missing}")
    print(f"Empty hypothesis   : {empty_hyp}")

    # Worst utterances
    print(f"\n===== Worst {PRINT_WORST} Utterances =====")
    per_utt_stats.sort(reverse=True)

    for i in range(min(PRINT_WORST, len(per_utt_stats))):
        cer, utt, r, h = per_utt_stats[i]
        print(f"\n[{i+1}] {utt}  CER={cer*100:.2f}%")
        print(f"REF: {r}")
        print(f"HYP: {h}")


if __name__ == "__main__":
    main()

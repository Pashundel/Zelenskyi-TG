

import argparse
import json
import logging
import re
import time
from collections import Counter
from typing import Dict, List, Set, Tuple

import pandas as pd


DEFAULT_UK_STOPWORDS = {
    "а", "або", "але", "би", "був", "була", "були", "було", "бути", "в", "вам", "вас", "весь",
    "ви", "від", "він", "вона", "вони", "воно", "все", "всіх", "всього", "де", "для", "до", "дуже",
    "є", "ж", "же", "за", "з", "зі", "й", "і", "із", "її", "їй", "їм", "його", "йому", "їх",
    "коли", "кожен", "кожна", "кожне", "кожні", "лише", "ми", "мене", "мені", "може", "можуть",
    "мій", "моя", "моє", "мої", "на", "над", "нам", "нас", "наш", "наша", "наше", "наші", "не",
    "неї", "ним", "ними", "них", "ні", "ніж", "ну", "о", "однак", "під", "по", "поки", "потім",
    "при", "про", "сам", "сама", "саме", "самі", "свій", "свого", "свої", "себе", "собі", "та",
    "так", "також", "там", "те", "ти", "той", "то", "тому", "тут", "у", "уже", "усе", "усіх",
    "хоч", "це", "цей", "ця", "ці", "цієї", "цієї", "цім", "щоб", "що", "як", "я", "який",
    "яка", "яке", "які", "якого", "якому", "україна", "ukraine"
}

APOS_VARIANTS = {"’", "ʼ", "`", "´"}


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("prepare_lda")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def read_csv_with_fallback(path: str, preferred_encoding: str) -> pd.DataFrame:
    encodings = [preferred_encoding, "utf-8-sig", "utf-8", "cp1251", "cp1252"]
    for enc in dict.fromkeys(encodings):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode {path}")


def parse_json_list(raw_value) -> List[str]:
    if isinstance(raw_value, list):
        return [str(x) for x in raw_value]
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []
    try:
        val = json.loads(raw_value)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        return []
    return []


def load_word_list(path: str | None) -> Set[str]:
    if not path:
        return set()
    words: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip().lower()
            if not item or item.startswith("#"):
                continue
            words.add(item)
    return words


def normalize_token(token: str) -> str:
    tok = str(token).strip().lower()
    for apos in APOS_VARIANTS:
        tok = tok.replace(apos, "'")
    tok = tok.replace("\u00a0", " ").strip()
    return tok


def is_informative_surface(token: str, min_len: int) -> bool:
    if not token or len(token) < min_len:
        return False
    if token.isdigit():
        return False

    if not re.search(r"[\u0400-\u04FFA-Za-z]", token):
        return False

    if re.fullmatch(r"[-_'’ʼ`]+", token):
        return False

    return True


def keep_by_upos(upos: str, allowed_upos: Set[str]) -> bool:
    if not allowed_upos:
        return True
    return str(upos).upper() in allowed_upos


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LDA-ready tokens from corrected lemmas.")
    parser.add_argument("--input", default="posts_lemmatized_corrected.csv")
    parser.add_argument("--output", default="posts_for_lda.csv")
    parser.add_argument("--terms-output", default="lda_terms_diagnostics.csv")
    parser.add_argument("--lemmas-col", default="lemmas_json")
    parser.add_argument("--upos-col", default="upos_json")
    parser.add_argument("--tokens-final-col", default="tokens_lda_json")
    parser.add_argument("--input-encoding", default="utf-8-sig")
    parser.add_argument("--output-encoding", default="utf-8-sig")
    parser.add_argument("--base-stopwords-file", default="", help="Optional TXT with one stopword per line")
    parser.add_argument("--custom-stopwords-file", default="", help="Optional TXT with project stopwords")
    parser.add_argument("--allowed-upos", default="NOUN,PROPN,ADJ,VERB")
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--min-doc-freq", type=int, default=5)
    parser.add_argument("--max-doc-freq-ratio", type=float, default=0.8)
    parser.add_argument("--log", default="prepare_lda_log.txt")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    logger = setup_logger(args.log)
    started = time.time()

    df = read_csv_with_fallback(args.input, args.input_encoding)
    if args.lemmas_col not in df.columns or args.upos_col not in df.columns:
        raise ValueError(f"Input must contain '{args.lemmas_col}' and '{args.upos_col}'")

    allowed_upos = {item.strip().upper() for item in args.allowed_upos.split(",") if item.strip()}

    base_stopwords = set(DEFAULT_UK_STOPWORDS)
    base_stopwords.update(load_word_list(args.base_stopwords_file))
    custom_stopwords = load_word_list(args.custom_stopwords_file)
    all_stopwords = {normalize_token(w) for w in (base_stopwords | custom_stopwords)}

    logger.info("Rows loaded: %s", len(df))
    logger.info("Stopwords: base=%s custom=%s total=%s", len(base_stopwords), len(custom_stopwords), len(all_stopwords))
    logger.info("Allowed UPOS: %s", sorted(allowed_upos) if allowed_upos else "ALL")

    stage1_docs: List[List[str]] = []
    stage1_upos_docs: List[List[str]] = []

    removal_counts = Counter()
    row_count = len(df)

    for i, (raw_lemmas, raw_upos) in enumerate(zip(df[args.lemmas_col].fillna(""), df[args.upos_col].fillna("")), start=1):
        lemmas = parse_json_list(raw_lemmas)
        upos_list = parse_json_list(raw_upos)

        filtered_tokens: List[str] = []
        filtered_upos: List[str] = []

        for lemma, upos in zip(lemmas, upos_list):
            token = normalize_token(lemma)
            up = str(upos).upper().strip()

            if not is_informative_surface(token, args.min_token_len):
                removal_counts["surface"] += 1
                continue

            if not keep_by_upos(up, allowed_upos):
                removal_counts["upos"] += 1
                continue

            if token in all_stopwords:
                removal_counts["stopword"] += 1
                continue

            filtered_tokens.append(token)
            filtered_upos.append(up)

        stage1_docs.append(filtered_tokens)
        stage1_upos_docs.append(filtered_upos)

        if i % args.log_every == 0 or i == row_count:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0.0
            eta = ((row_count - i) / rate / 60) if rate else 0.0
            logger.info("Stage1 %s/%s (%.1f%%) | %.1f rows/s | ETA %.1f min", i, row_count, i / row_count * 100, rate, eta)

    doc_freq = Counter()
    term_freq = Counter()

    for tokens in stage1_docs:
        unique_terms = set(tokens)
        doc_freq.update(unique_terms)
        term_freq.update(tokens)

    max_doc_freq_abs = int(args.max_doc_freq_ratio * len(stage1_docs)) if stage1_docs else 0

    kept_terms = set()
    term_reason: Dict[str, str] = {}
    for term in term_freq.keys():
        dfreq = doc_freq[term]
        if dfreq < args.min_doc_freq:
            term_reason[term] = "rare"
            continue
        if args.max_doc_freq_ratio > 0 and dfreq > max_doc_freq_abs:
            term_reason[term] = "too_common"
            continue
        term_reason[term] = "kept"
        kept_terms.add(term)

    final_docs: List[List[str]] = []
    for tokens in stage1_docs:
        final_docs.append([tok for tok in tokens if tok in kept_terms])

    out = df.copy()
    out[args.tokens_final_col] = [json.dumps(tokens, ensure_ascii=False) for tokens in final_docs]
    out.to_csv(args.output, index=False, encoding=args.output_encoding)

    diagnostics_rows = []
    for term, tfreq in term_freq.most_common():
        dfreq = doc_freq[term]
        diagnostics_rows.append(
            {
                "term": term,
                "term_freq": tfreq,
                "doc_freq": dfreq,
                "doc_freq_ratio": round(dfreq / len(stage1_docs), 6) if stage1_docs else 0.0,
                "status": term_reason.get(term, "removed"),
                "is_custom_stopword": 1 if term in custom_stopwords else 0,
                "is_base_stopword": 1 if term in base_stopwords else 0,
            }
        )

    diag_df = pd.DataFrame(diagnostics_rows)
    diag_df.to_csv(args.terms_output, index=False, encoding=args.output_encoding)

    docs_nonempty = sum(1 for d in final_docs if d)
    logger.info("Saved LDA corpus file: %s", args.output)
    logger.info("Saved term diagnostics: %s", args.terms_output)
    logger.info("Final docs non-empty: %s/%s", docs_nonempty, len(final_docs))
    logger.info("Final vocab size: %s", len(kept_terms))
    logger.info("Removed counts by stage: %s", dict(removal_counts))
    logger.info("Done in %.2f minutes", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
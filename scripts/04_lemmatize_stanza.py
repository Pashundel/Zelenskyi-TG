import argparse
import json
import logging
import time
from typing import List, Tuple

import pandas as pd
import stanza


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("lemmatize")
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


def read_csv_with_fallback(path: str, logger: logging.Logger, preferred_encoding: str) -> pd.DataFrame:
    encodings = [preferred_encoding, "utf-8-sig", "utf-8", "cp1251", "cp1252"]
    tried = []

    for enc in dict.fromkeys(encodings):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("Loaded CSV using encoding=%s", enc)
            return df
        except UnicodeDecodeError:
            tried.append(enc)

    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode CSV with tried encodings: {tried}")


def parse_tokens(raw: str) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []

    # normal JSON path
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return [str(x) for x in value]
    except Exception:
        pass

    # fallback for malformed list-like strings
    try:
        value = eval(raw, {"__builtins__": {}}, {})  # safe-ish limited eval
        if isinstance(value, list):
            return [str(x) for x in value]
    except Exception:
        pass

    return []


def lemmatize_tokens(nlp, tokens: List[str]) -> Tuple[List[str], List[str]]:
    if not tokens:
        return [], []

    doc = nlp([tokens])  # pretokenized single sentence
    lemmas, upos_tags = [], []

    for sent in doc.sentences:
        for word in sent.words:
            lemmas.append(word.lemma if word.lemma is not None else "")
            upos_tags.append(word.upos if word.upos is not None else "")

    return lemmas, upos_tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Lemmatize + UPOS with Stanza from tokens_json")
    parser.add_argument("--input", default="posts_tokenized.csv")
    parser.add_argument("--output", default="posts_lemmatized.csv")
    parser.add_argument("--review-output", default="lemma_upos_review.csv")
    parser.add_argument("--tokens-col", default="tokens_json")
    parser.add_argument("--lang", default="uk")
    parser.add_argument("--input-encoding", default="utf-8-sig")
    parser.add_argument("--output-encoding", default="utf-8-sig")
    parser.add_argument("--log", default="lemmatization_log.txt")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    logger = setup_logger(args.log)
    started = time.time()

    df = read_csv_with_fallback(args.input, logger, args.input_encoding)
    row_count = len(df)
    logger.info("Loaded %s rows from %s", row_count, args.input)

    if args.tokens_col not in df.columns:
        raise ValueError(f"Column '{args.tokens_col}' not found in {args.input}")

    logger.info("Loading Stanza model for lang=%s", args.lang)
    stanza.download(args.lang, processors="tokenize,pos,lemma")
    nlp = stanza.Pipeline(
        lang=args.lang,
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=True,
        use_gpu=False,
        verbose=False,
    )

    lemmas_json = []
    upos_json = []
    unique_pairs = set()
    bad_rows = 0

    for i, raw in enumerate(df[args.tokens_col].fillna(""), start=1):
        tokens = parse_tokens(raw)
        if not isinstance(tokens, list):
            tokens = []

        if not tokens and isinstance(raw, str) and raw.strip():
            bad_rows += 1

        lemmas, upos_tags = lemmatize_tokens(nlp, tokens)

        for lm, up in zip(lemmas, upos_tags):
            unique_pairs.add((lm, up))

        lemmas_json.append(json.dumps(lemmas, ensure_ascii=False))
        upos_json.append(json.dumps(upos_tags, ensure_ascii=False))

        if i % args.log_every == 0 or i == row_count:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0.0
            eta_minutes = ((row_count - i) / rate / 60) if rate else 0.0
            logger.info("Progress %s/%s (%.1f%%) | %.2f rows/s | ETA %.1f min",
                        i, row_count, (i / row_count * 100) if row_count else 100.0, rate, eta_minutes)

    out = df.copy()
    out["lemmas_json"] = lemmas_json
    out["upos_json"] = upos_json
    out.to_csv(args.output, index=False, encoding=args.output_encoding)

    review_df = pd.DataFrame(sorted(unique_pairs), columns=["lemma", "upos"])
    review_df["corrected_lemma"] = ""
    review_df["corrected_upos"] = ""
    review_df.to_csv(args.review_output, index=False, encoding=args.output_encoding)

    logger.info("Rows with unparsable tokens_json: %s", bad_rows)
    logger.info("Saved lemmatized file: %s (encoding=%s)", args.output, args.output_encoding)
    logger.info("Saved review file: %s (encoding=%s)", args.review_output, args.output_encoding)
    logger.info("Done in %.2f minutes", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
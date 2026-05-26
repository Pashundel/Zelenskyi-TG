import argparse
import json
import logging
import re
import time
from typing import List

import pandas as pd


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("tokenize")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def read_csv_with_fallback(path: str, logger: logging.Logger, preferred_encoding: str) -> pd.DataFrame:
    encodings = [preferred_encoding, "utf-8-sig", "utf-8", "cp1251", "cp1252"]
    tried = []

    for encoding in dict.fromkeys(encodings):
        try:
            df = pd.read_csv(path, encoding=encoding)
            logger.info("Loaded CSV using encoding=%s", encoding)
            return df
        except UnicodeDecodeError:
            tried.append(encoding)

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Unable to decode CSV with tried encodings: {tried}",
    )


def cyrillic_score(text: str) -> int:
    return sum(1 for char in text if "\u0400" <= char <= "\u04FF")


def looks_mojibake(text: str) -> bool:
    if not text:
        return False
    suspicious_markers = ["â", "Ã", "Ð", "Ñ", "â€", "â€“", "â€”", "�", "—", "–"]
    marker_hits = sum(text.count(marker) for marker in suspicious_markers)
    return marker_hits >= 2


def repair_mojibake(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "" if text is None else text

    candidates = {text}

    for bad_enc in ("latin1", "cp1252", "mac_roman"):
        try:
            candidates.add(text.encode(bad_enc).decode("utf-8"))
        except Exception:
            pass

    first_pass = list(candidates)
    for candidate in first_pass:
        for bad_enc in ("latin1", "cp1252", "mac_roman"):
            try:
                candidates.add(candidate.encode(bad_enc).decode("utf-8"))
            except Exception:
                pass

    best = max(candidates, key=lambda value: (cyrillic_score(value), -value.count("�")))
    return best


def clean_and_repair_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\ufeff", "").replace("\u00a0", " ")
    if looks_mojibake(normalized):
        normalized = repair_mojibake(normalized)
    return normalized


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize cleaned text with mojibake repair.")
    parser.add_argument("--input", default="posts_text_cleaned.csv", help="Input CSV file")
    parser.add_argument("--output", default="posts_tokenized.csv", help="Output CSV file")
    parser.add_argument("--text-col", default="text_clean", help="Text column to tokenize")
    parser.add_argument("--input-encoding", default="utf-8-sig", help="Preferred input CSV encoding")
    parser.add_argument("--output-encoding", default="utf-8-sig", help="Output CSV encoding")
    parser.add_argument("--log", default="tokenization_log.txt", help="Log file path")
    parser.add_argument("--log-every", type=int, default=500, help="Progress interval (rows)")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    started = time.time()

    df = read_csv_with_fallback(args.input, logger, args.input_encoding)
    row_count = len(df)
    logger.info("Loaded %s rows from %s", row_count, args.input)

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.input}")

    repaired_count = 0
    tokens_json_list = []

    for index, raw_text in enumerate(df[args.text_col].fillna(""), start=1):
        source_text = raw_text if isinstance(raw_text, str) else ""
        fixed_text = clean_and_repair_text(source_text)

        if fixed_text != source_text:
            repaired_count += 1

        tokens = tokenize(fixed_text)
        tokens_json_list.append(json.dumps(tokens, ensure_ascii=False))

        if index % args.log_every == 0 or index == row_count:
            elapsed = time.time() - started
            rate = index / elapsed if elapsed else 0.0
            remaining = row_count - index
            eta_minutes = (remaining / rate) / 60 if rate else 0.0
            logger.info(
                "Progress %s/%s (%.1f%%) | %.1f rows/s | ETA %.1f min",
                index,
                row_count,
                (index / row_count * 100) if row_count else 100.0,
                rate,
                eta_minutes,
            )

    output_df = df.copy()
    output_df["tokens_json"] = tokens_json_list
    output_df.to_csv(args.output, index=False, encoding=args.output_encoding)

    logger.info("Rows repaired for mojibake: %s", repaired_count)
    logger.info("Saved tokenized output to %s (encoding=%s)", args.output, args.output_encoding)
    logger.info("Done in %.2f minutes", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
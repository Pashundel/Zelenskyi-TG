#!/usr/bin/env python3
import csv
import json
import sys
from datetime import datetime

# Analysis window (inclusive)
START = datetime.fromisoformat("2022-02-24T00:00:00")
END = datetime.fromisoformat("2025-12-31T23:59:59")


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def flatten_text(text_field):
    """
    Telegram export can store text as:
    - string
    - list of strings/dicts (rich text entities)
    """
    if isinstance(text_field, str):
        return text_field

    if isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)

    return ""


def cyrillic_score(s):
    return sum(1 for ch in s if "\u0400" <= ch <= "\u04FF")


def repair_mojibake(s):
    """
    Fix common mojibake cases where UTF-8 text was mis-decoded as cp1252/latin1.
    """
    if not isinstance(s, str) or not s:
        return s

    candidates = {s}

    # One pass
    for bad_enc in ("latin1", "cp1252"):
        try:
            candidates.add(s.encode(bad_enc).decode("utf-8"))
        except Exception:
            pass

    # Second pass (for double corruption)
    first_pass = list(candidates)
    for c in first_pass:
        for bad_enc in ("latin1", "cp1252"):
            try:
                candidates.add(c.encode(bad_enc).decode("utf-8"))
            except Exception:
                pass

    # Pick candidate with most Cyrillic characters; tie-break: fewer replacement chars
    best = max(candidates, key=lambda x: (cyrillic_score(x), -x.count("�")))
    return best


def is_empty_text_entities(val):
    # Telegram may represent empty as "", [], None, or missing
    return (val is None) or (val == "") or (isinstance(val, list) and len(val) == 0)


def is_placeholder_photo_without_text(msg, text):
    return (
        safe_int(msg.get("photo_file_size", 0)) > 0
        and msg.get("photo") == "(File not included. Change data exporting settings to download.)"
        and text.strip() == ""
        and is_empty_text_entities(msg.get("text_entities"))
    )


def is_video_without_text(msg, text):
    return (
        msg.get("media_type") == "video_file"
        and text.strip() == ""
        and is_empty_text_entities(msg.get("text_entities"))
    )


def parse_messages(data):
    messages = data.get("messages", [])

    post_rows = []
    reaction_rows = []

    seen_ids = set()
    seen_text_time = set()

    qc = {
        "messages_total": 0,
        "non_message_skipped": 0,
        "missing_or_bad_date_skipped": 0,
        "out_of_scope_date_skipped": 0,
        "photo_only_no_text_skipped": 0,
        "video_only_no_text_skipped": 0,
        "other_empty_text_skipped": 0,
        "duplicate_id_skipped": 0,
        "duplicate_text_timestamp_skipped": 0,
        "kept_posts": 0,
        "reaction_rows_written": 0
    }

    for msg in messages:
        qc["messages_total"] += 1

        if not isinstance(msg, dict) or msg.get("type") != "message":
            qc["non_message_skipped"] += 1
            continue

        date_str = msg.get("date", "")
        if not date_str:
            qc["missing_or_bad_date_skipped"] += 1
            continue

        try:
            dt = datetime.fromisoformat(date_str)
        except Exception:
            qc["missing_or_bad_date_skipped"] += 1
            continue

        if not (START <= dt <= END):
            qc["out_of_scope_date_skipped"] += 1
            continue

        raw_text = flatten_text(msg.get("text", ""))
        text = repair_mojibake(raw_text)

        # Your explicit skip rules
        if is_placeholder_photo_without_text(msg, text):
            qc["photo_only_no_text_skipped"] += 1
            continue

        if is_video_without_text(msg, text):
            qc["video_only_no_text_skipped"] += 1
            continue

        # Any other empty text -> skip
        if text.strip() == "":
            qc["other_empty_text_skipped"] += 1
            continue

        message_id = msg.get("id")

        # Duplicate by ID
        if message_id in seen_ids:
            qc["duplicate_id_skipped"] += 1
            continue
        seen_ids.add(message_id)

        # Duplicate by identical text + timestamp
        text_key = " ".join(text.split())
        text_time_key = (date_str, text_key)
        if text_time_key in seen_text_time:
            qc["duplicate_text_timestamp_skipped"] += 1
            continue
        seen_text_time.add(text_time_key)

        reactions = msg.get("reactions", [])
        if not isinstance(reactions, list):
            reactions = []

        reactions_total = sum(
            safe_int(r.get("count", 0)) for r in reactions if isinstance(r, dict)
        )

        is_forwarded = int(any(k in msg for k in (
            "forwarded_from",
            "forwarded_from_id",
            "forwarded_from_name",
            "fwd_from",
            "fwd_from_id"
        )))

        post_rows.append({
            "message_id": message_id,
            "date": date_str,
            "date_unixtime": msg.get("date_unixtime", ""),
            "edited": msg.get("edited", ""),
            "edited_unixtime": msg.get("edited_unixtime", ""),
            "from": msg.get("from", ""),
            "from_id": msg.get("from_id", ""),
            "media_type": msg.get("media_type", ""),
            "mime_type": msg.get("mime_type", ""),
            "file_name": msg.get("file_name", ""),
            "duration_seconds": msg.get("duration_seconds", ""),
            "width": msg.get("width", ""),
            "height": msg.get("height", ""),
            "text": text,
            "text_length": len(text),
            "reactions_total": reactions_total,
            "is_forwarded": is_forwarded
        })

        for r in reactions:
            if not isinstance(r, dict):
                continue
            reaction_rows.append({
                "message_id": message_id,
                "date": date_str,
                "emoji": r.get("emoji", ""),
                "reaction_type": r.get("type", ""),
                "count": safe_int(r.get("count", 0))
            })

    qc["kept_posts"] = len(post_rows)
    qc["reaction_rows_written"] = len(reaction_rows)

    return post_rows, reaction_rows, qc


def write_csv(posts_csv, reactions_csv, post_rows, reaction_rows):
    post_fields = [
        "message_id", "date", "date_unixtime", "edited", "edited_unixtime",
        "from", "from_id", "media_type", "mime_type", "file_name",
        "duration_seconds", "width", "height", "text", "text_length",
        "reactions_total", "is_forwarded"
    ]

    reaction_fields = [
        "message_id", "date", "emoji", "reaction_type", "count"
    ]

    # utf-8-sig helps Excel/Numbers on macOS read Cyrillic correctly
    with open(posts_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=post_fields)
        writer.writeheader()
        writer.writerows(post_rows)

    with open(reactions_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=reaction_fields)
        writer.writeheader()
        writer.writerows(reaction_rows)


def write_qc_log(log_path, input_json, qc):
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("=== Parse QC Log ===\n")
        logf.write(f"input_file: {input_json}\n")
        logf.write(f"analysis_window_start: {START.isoformat()}\n")
        logf.write(f"analysis_window_end: {END.isoformat()}\n\n")
        for key, value in qc.items():
            logf.write(f"{key}: {value}\n")


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 parse_telegram_json.py input.json posts_flat.csv reactions_long.csv")
        sys.exit(1)

    input_json = sys.argv[1]
    posts_csv = sys.argv[2]
    reactions_csv = sys.argv[3]

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    post_rows, reaction_rows, qc = parse_messages(data)
    write_csv(posts_csv, reactions_csv, post_rows, reaction_rows)
    write_qc_log("parse_qc_log.txt", input_json, qc)

    print("Done.")
    print(f"Posts in scope: {len(post_rows)}")
    print(f"Reaction rows: {len(reaction_rows)}")
    print("QC log: parse_qc_log.txt")
    print(f"Wrote: {posts_csv}")
    print(f"Wrote: {reactions_csv}")


if __name__ == "__main__":
    main()
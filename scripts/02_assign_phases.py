#!/usr/bin/env python3
import csv
from datetime import datetime, date
from collections import Counter

INPUT_CSV = "posts_clean.csv"
OUTPUT_CSV = "posts_clean_phased.csv"
LOG_FILE = "phase_assignment_log.txt"


def parse_date_safe(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def get_row_date(row):
    # Preferred: date_kyiv from your time-clean script
    if "date_kyiv" in row and row["date_kyiv"]:
        d = parse_date_safe(row["date_kyiv"])
        if d:
            return d

    # Fallback: first 10 chars of original date field (YYYY-MM-DD...)
    if "date" in row and row["date"]:
        d = parse_date_safe(row["date"][:10])
        if d:
            return d

    return None


def assign_phase(d):
    # Inclusive boundaries
    if date(2022, 2, 24) <= d <= date(2022, 3, 31):
        return 1
    if date(2022, 4, 1) <= d <= date(2022, 11, 11):
        return 2
    if date(2022, 11, 12) <= d <= date(2023, 7, 3):
        return 3
    if date(2023, 7, 4) <= d <= date(2023, 12, 1):
        return 4
    if date(2023, 12, 2) <= d <= date(2025, 1, 19):
        return 5
    if date(2025, 1, 20) <= d <= date(2025, 12, 31):
        return 6
    return ""  # unassigned


def main():
    total_rows = 0
    missing_date_rows = 0
    unassigned_rows = 0
    phase_counts = Counter()

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as fin, \
         open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as fout:

        reader = csv.DictReader(fin)
        fields = reader.fieldnames if reader.fieldnames else []
        if "phase" not in fields:
            fields = fields + ["phase"]

        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for row in reader:
            total_rows += 1
            d = get_row_date(row)

            if d is None:
                row["phase"] = ""
                missing_date_rows += 1
            else:
                ph = assign_phase(d)
                row["phase"] = ph
                if ph == "":
                    unassigned_rows += 1
                else:
                    phase_counts[ph] += 1

            writer.writerow(row)

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("=== Phase Assignment Log ===\n")
        log.write(f"input_csv: {INPUT_CSV}\n")
        log.write(f"output_csv: {OUTPUT_CSV}\n\n")
        log.write(f"total_rows: {total_rows}\n")
        log.write(f"missing_date_rows: {missing_date_rows}\n")
        log.write(f"unassigned_rows: {unassigned_rows}\n")
        for p in range(1, 7):
            log.write(f"phase_{p}_rows: {phase_counts.get(p, 0)}\n")

    print("Done.")
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {LOG_FILE}")


if __name__ == "__main__":
    main()
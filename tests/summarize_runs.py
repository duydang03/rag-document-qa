"""Tóm tắt các lần chạy lặp lại của evaluate_rag_quality.py.

Đọc tests/reports/top{k}_run*.json và in kết quả từng lần chạy,
kèm số lần mỗi ca không đạt PASS.

Chạy từ thư mục gốc của dự án:
    python tests/summarize_runs.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "tests" / "reports"
TOP_K_VALUES = (5, 8)


def main():
    found_any = False

    for k in TOP_K_VALUES:
        pattern = str(REPORT_DIR / f"top{k}_run*.json")
        paths = sorted(glob.glob(pattern))

        print(f"--- top_k = {k}")

        if not paths:
            print(f"  (không tìm thấy file khớp {pattern})")
            continue

        found_any = True
        flagged = Counter()
        latencies = []

        for path in paths:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)

            bad = [x["id"] for x in r["results"] if x["status"] != "PASS"]
            flagged.update(bad)
            latencies.append(r["average_latency_seconds"])

            print(
                os.path.basename(path),
                "PASS", r["pass"],
                "REVIEW", r["review"],
                "FAIL", r["fail"],
                "ERROR", r["errors"],
                "| answerable", r["answerable_pass_rate"],
                "| abstain", r["abstention_rate"],
                "| latency", r["average_latency_seconds"],
                bad,
            )

        print(
            f"Số lần không PASS theo từng ca ({len(paths)} lần chạy):",
            dict(flagged) or "không có",
        )
        print(
            "Độ trễ trung bình qua các lần chạy:",
            round(sum(latencies) / len(latencies), 3),
            "giây\n",
        )

    if not found_any:
        print(
            "Chưa có báo cáo nào. Chạy 3 lần cho mỗi top-k rồi đổi tên thành "
            "top5_run1.json, top8_run1.json, ... trong tests/reports/."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
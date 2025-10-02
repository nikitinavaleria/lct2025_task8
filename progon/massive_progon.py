import mimetypes
import csv
from pathlib import Path
import requests

CSV_COLUMNS = [
    "path_to_study",
    "study_uid",
    "series_uid",
    "probability_of_pathology",
    "pathology",
    "processing_status",
    "time_of_processing",
    "most_dangerous_pathology_type"
            ]

def main(data_dir: Path, endpoint: str):
    if not data_dir.exists():
        print(f"Папка {data_dir} не найдена")
        return

    files = [p for p in data_dir.rglob("*") if p.is_file()]
    if not files:
        print("Файлы не найдены")
        return

    print(f"Нашли {len(files)} файлов")

    results_csv = Path("results.csv")
    rows_for_csv = []

    for i, p in enumerate(files, 1):
        ctype, _ = mimetypes.guess_type(str(p))
        ctype = ctype or "application/octet-stream"

        resp = None
        try:
            with p.open("rb") as f:
                resp = requests.post(
                    endpoint,
                    files={"file": (p.name, f, ctype)},
                    timeout=120
                )

            try:
                data = resp.json()
            except Exception:
                data = {"processing_status": "Failure", "raw": resp.text}
        except Exception as e:
            data = {"processing_status": "Failure", "error": str(e)}

        status = getattr(resp, "status_code", 599)

        row_csv = {
            "path_to_study": str(p),
            "study_uid": data.get("study_uid", ""),
            "series_uid": data.get("series_uid", ""),
            "probability_of_pathology": data.get("probability_of_pathology", 0.0),
            "pathology": data.get("pathology", 0),
            "processing_status": data.get("processing_status", "Failure"),
            "time_of_processing": data.get("time_of_processing", 0.0),
            "most_dangerous_pathology_type": data.get("most_dangerous_pathology_type", ""),
        }
        rows_for_csv.append(row_csv)

        print(f"[{i}/{len(files)}] {p} -> {status}")

    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"Готово. CSV: {results_csv}")

if __name__ == "__main__":
    data_dir = Path("data")
    endpoint = "http://localhost:8000/inference/predict"

    main(data_dir, endpoint)

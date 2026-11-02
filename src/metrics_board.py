"""Build the current aggregate metric ledger without reading participant data."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = ("rmse", "mse", "mae", "msle", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim")
HIGHER = {"psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim"}
DEFAULT_ROW = {"GRAPE": "other visit, same camera, under 1 year", "UPOL": "same-sitting repeat", "UTIRIS": "same-sitting repeat"}
LABELS = {"rmse": "RMSE", "mse": "MSE", "mae": "MAE", "msle": "MSLE", "psnr": "PSNR",
          "psnr_eps": "PSNR_eps", "psnr_01": "PSNR_01", "psnr_01_eps": "PSNR_01_eps", "ssim": "SSIM"}


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def number(value):
    return "не посчитано" if value is None else f"{value:.3f}"


def extract_brats(data):
    rows = []
    for dataset, rungs in data["summary"].items():
        for rung, entry in rungs.items():
            for metric in METRICS:
                v = entry[metric]
                c = v.get("c", {})
                rows.append({"experiment": "BraTS 2D central patch", "dataset": dataset, "rung": rung,
                             "metric": metric, "direction": "higher" if metric in HIGHER else "lower",
                             "raw_median": v["median"], "raw_q1": v["iqr"][0], "raw_q3": v["iqr"][1],
                             "c_mean": c.get("mean", .5), "c_ci_low": c.get("ci95", [None, None])[0],
                             "c_ci_high": c.get("ci95", [None, None])[1], "n_pairs": entry.get("n_pairs"),
                             "n_people": entry.get("n_people"), "n_reference_fills": entry.get("n_fills"),
                             "status": "saved_descriptive_result" if c else "reference_by_construction",
                             "source": "ladder2d_brats_baseline.json"})
    return rows


def build():
    baseline = load("ladder2d_brats_baseline.json")
    if baseline is None:
        raise ValueError("Missing BraTS baseline summary")
    rows = extract_brats(baseline)
    state = load("care_iteration2_state.json")
    care = load("care_iteration2.json")
    sources = ["ladder2d_brats_baseline.json", "ladder2d.json", "care_reproduction.json",
               "care_reproduction_all_phases_posthoc.json", "care_iteration2.json", "care_iteration2_state.json"]
    stamp = datetime.now(timezone.utc).isoformat()
    lines = ["# Текущие метрики и результаты", "", f"Обновлено (UTC): {stamp}. Генерируется из сохранённых агрегатов; исходные результаты не перезаписываются.",
             "", "## Что сейчас проверяется", "",
             "| Блок | Метрики | Данные | Текущий статус |", "|---|---|---|---|",
             "| BraTS 2D | Все 9 выходов: RMSE, MSE, MAE, MSLE, PSNR, PSNR_eps, PSNR_01, PSNR_01_eps, SSIM | GRAPE, UPOL, UTIRIS; центральная область 64×64 из 224×224 | Посчитано; описательные результаты |",
             "| Признаки 2D | Пиксели P, DINOv2 D, контроль яркости N; RETFound R и сосудистые Dice/компоненты/ветвления для GRAPE | Отдельный протокол и представление | Посчитано отдельно; см. ниже |",
             "| QTIM MRI | 9 BraTS-метрик; сегментационные проверки отдельным этапом | SyN + P90 | Текущих проверенных чисел нет; прежние affine-результаты не используем |",
             "| CARE, итерация 2 | SSIM, PSNR; NSD крупных/малых органов; clDice кишечника/сосудов | 36 тестовых КТ; результаты авторов | " + (state.get("stage", "ещё не запущено") if state else "ещё не запущено") + " |",
             "| H19 на CARE | Dice, NSD 2 мм, clDice на mean/shrinkage/wrong-detail | Изображения через фиксированный сегментатор | Ещё не выполнено |", "",
             "c = доля согласованных по условиям посторонних, получивших худшую оценку, с половиной веса совпадений; 0.5 — типичный посторонний. Это калибровка, а не ещё одна метрика.",
             "", "## Все девять BraTS-выходов: собственное изображение / population mean", "",
             "Здесь указаны средние c. GRAPE: другая дата, та же камера, <1 года; UPOL/UTIRIS: повтор в той же сессии. Mean в исходном агрегате объединён по same-eye строкам; это НЕ парное сравнение с отдельной строкой GRAPE.", "",
             "| Метрика | Лучше в исходной шкале | GRAPE своё / mean | UPOL своё / mean | UTIRIS своё / mean | QTIM |", "|---|---|---|---|---|---|"]
    for metric in METRICS:
        cells = []
        for ds in DEFAULT_ROW:
            ds_data = baseline["summary"][ds]
            own, mean = [ds_data[r][metric]["c"]["mean"] for r in (DEFAULT_ROW[ds], "population mean")]
            cells.append(f"{own:.3f} / {mean:.3f}")
        lines.append(f"| {LABELS[metric]} | {'↑' if metric in HIGHER else '↓'} | " + " | ".join(cells) + " | не готово |")
    lines += ["", "RMSE/MSE и варианты PSNR могут давать одинаковые ранги в одном случае: девять выходов не означают девять независимых свидетельств.",
              "Старый текст «единственное исключение — MAE на UPOL» неполон: plain PSNR на GRAPE также выше у своего изображения (0.641), чем у объединённого mean (0.630). Это описательное сравнение разных пулов, без парного теста и без вывода о значимости.",
              "", "## Выборки и все ступени из таблицы на скриншоте", "",
              "| Набор | Ступень | Пар | Людей | c MSE | c SSIM |", "|---|---|---|---|---|---|"]
    for ds, rungs in baseline["summary"].items():
        for rung, e in rungs.items():
            if rung == "stranger (matched)":
                continue
            lines.append(f"| {ds} | {rung} | {e['n_pairs']} | {e['n_people']} | {e['mse']['c']['mean']:.3f} | {e['ssim']['c']['mean']:.3f} |")
    lines += ["", "На скриншоте нет: семи остальных выходов пакета, исходных значений, 95% интервалов c, размеров выборок, строки посторонних, shrinkage-пути, других представлений, семейных сравнений и готовых QTIM-значений. Пустая клетка не равна нулю.",
              "", "## CARE: каждая метрика отдельно", "",
              "Воспроизводится агрегация опубликованных оценок, а не генерация КТ или запуск сегментатора. Сравнение — медиана и два квартиля с таблицей 2, допуск округления 0.050001.", "",
              "| Метрика | Совпало | Не совпало | Нет покрытия | Нет конечной оценки | Статус |", "|---|---|---|---|---|---|"]
    names = {"ssim": "SSIM ↑", "psnr": "PSNR ↑", "large_nsd": "NSD крупных органов ↑", "small_nsd": "NSD малых органов ↑",
             "intestine_cldice": "clDice кишечника ↑", "vessel_cldice": "clDice сосудов ↑"}
    for metric, label in names.items():
        rr = [r for r in care.get("rows", []) if r["metric"] == metric] if care else []
        counts = {s: sum(r["status"] == s for r in rr) for s in ("matched", "mismatch", "blocked_coverage", "unavailable")}
        if not rr:
            lines.append(f"| {label} | — | — | — | — | ожидается итерация 2 |")
        else:
            status = "воспроизведено" if counts["matched"] == 18 else "частичное покрытие" if counts["blocked_coverage"] else "не воспроизведено"
            lines.append(f"| {label} | {counts['matched']}/18 | {counts['mismatch']} | {counts['blocked_coverage']} | {counts['unavailable']} | {status} |")
    lines += ["", "[CARE: текущие численные значения и сравнение с публикацией](care_iteration2_report.md). История: [13 случаев](care_reproduction_report.md), [36 случаев: прежняя остановка по покрытию](care_reproduction_all_phases_posthoc_report.md). Это не результат проверки H19."]
    ladder = load("ladder2d.json")
    if ladder:
        lines += ["", "## Другие представления: отдельный 2D-прогон", "",
                  "В каждой ячейке c своего изображения / c среднего для одной и той же ступени. Это другой оператор и область, чем BraTS-таблица выше; числа не склеиваются.", "",
                  "| Представление | GRAPE | UPOL | UTIRIS |", "|---|---|---|---|"]
        for method in ("P | primary", "D | primary", "N | primary", "R | primary", "V_dice | primary", "V_components | primary", "V_branches | primary"):
            cells = []
            for ds in DEFAULT_ROW:
                key = "same_camera_<1y" if ds == "GRAPE" else "same_eye_same_sitting"
                entry = ladder["results"][ds]["rungs"].get(method, {}).get(key)
                cells.append("не применялось" if not entry else f"{entry['lambda=1']['mean']:.3f} / " + (f"{entry['lambda=0']['mean']:.3f}" if "lambda=0" in entry else "нет mean"))
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines += ["", "Все ступени, lambda=0/0.25/0.5/0.75/1, интервалы и чувствительности сохранены в [ladder2d.json](ladder2d.json). Отсутствующая lambda не достраивается."]
    lines += ["", "## Полные исходные шкалы и интервалы: девять BraTS-выходов", "",
              "c: среднее и 95% bootstrap-интервал с группировкой по людям; исходная шкала: медиана [Q1, Q3]. Источник: [сохранённый агрегат](ladder2d_brats_baseline.json).", "",
              "| Набор | Ступень | Метрика | Исходная медиана [Q1, Q3] | c [95% интервал] | Пар / людей |", "|---|---|---|---|---|---|"]
    for r in rows:
        raw = f"{r['raw_median']:.4g} [{r['raw_q1']:.4g}, {r['raw_q3']:.4g}]"
        cal = f"{r['c_mean']:.3f}" + (f" [{r['c_ci_low']:.3f}, {r['c_ci_high']:.3f}]" if r["c_ci_low"] is not None else " (по построению)")
        n = f"{r['n_pairs']} / {r['n_people']}" if r["n_pairs"] is not None else f"{r['n_reference_fills']} reference fills; не независимые люди"
        lines.append(f"| {r['dataset']} | {r['rung']} | {LABELS[r['metric']]} | {raw} | {cal} | {n} |")
    lines += ["", "Синтетические H19, pupil и population-null результаты остаются отдельными экспериментами: [label_robustness_report.md](label_robustness_report.md), [population_null_report.md](population_null_report.md), [pupil_report.md](pupil_report.md). Они не являются реальными КТ/MRI результатами."]
    hashes = {s: hashlib.sha256((RESULTS / s).read_bytes()).hexdigest() for s in sources if (RESULTS / s).exists()}
    ledger = {"generated_at_utc": stamp, "source_sha256": hashes, "brats_rows": rows,
              "qtim_status": "unvalidated_rerun", "care_iteration2": care.get("metric_summary") if care else None}
    (RESULTS / "METRICS_STATUS.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (RESULTS / "METRICS_STATUS.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    path = RESULTS / "METRICS_STATUS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build())

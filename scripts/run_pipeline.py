from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import PROJECT_ROOT, RAW_DIR, ensure_dirs, setup_logging, utc_now

SCRIPTS = PROJECT_ROOT / "scripts"
IC_EXTRACTED = RAW_DIR / "ic_splc" / "extracted_csv" / "ic_splc_nodes.csv"


def run(script: str, args: list[str], logger) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    logger.info("RUN %s", " ".join(cmd))
    started = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = round(time.time() - started, 1)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode} after {elapsed}s")
    logger.info("OK %s (%.1fs)", script, elapsed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full AtmoGraph dataset pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Do not run the acquisition step")
    parser.add_argument("--force-download", action="store_true", help="Re-download source files")
    parser.add_argument("--allow-restricted", action="store_true", help="Process and include restricted CC BY-NC evaluation assets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("run_pipeline")
    logger.info("Pipeline start %s", utc_now())

    if not args.skip_download:
        download_args = []
        if args.force_download:
            download_args.append("--force")
        if args.allow_restricted:
            download_args.append("--include-restricted")
        run("download_datasets.py", download_args, logger)

    if not IC_EXTRACTED.exists():
        raise SystemExit(
            "IC-SPLC extracted CSVs are missing. Run the Neo4j extraction first: "
            "python3 scripts/preprocess_ic_splc.py --extract"
        )

    for script in [
        "preprocess_sampoq.py",
        "preprocess_ic_splc.py",
        "preprocess_anonnet.py",
        "preprocess_syndelay.py",
        "preprocess_smart_logistics.py",
        "preprocess_gdelt.py",
    ]:
        run(script, [], logger)

    if args.allow_restricted:
        run("preprocess_disruption_monitoring.py", ["--allow-restricted"], logger)
    else:
        logger.info("Skipping restricted CC BY-NC preprocessing (pass --allow-restricted to enable)")

    run("build_graph_dataset.py", ["--seed", str(args.seed)], logger)
    run("inspect_datasets.py", [], logger)
    run("validate_datasets.py", [], logger)
    run("generate_dataset_report.py", [], logger)

    logger.info("Pipeline complete %s", utc_now())
    print(
        "\n".join(
            [
                "Pipeline complete.",
                "Reports: datasets/reports/dataset_summary.md, data_quality_report.md, preprocessing_report.md",
                "Manifest: datasets/metadata/dataset_manifest.json",
                "Validation: datasets/reports/validation_results.json",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

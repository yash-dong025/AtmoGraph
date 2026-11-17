"""Serve the AtmoGraph dashboard and its JSON API over localhost.

Run:
    .venv/bin/python ml/reports/serve_dashboard.py
    .venv/bin/python ml/reports/serve_dashboard.py --open
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import re
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.reports.dashboard_data import (
    DEFAULT_MAX_NODES,
    DEFAULT_SEED,
    DashboardData,
    PredictionUnavailable,
)

DASHBOARD_PATH = PROJECT_ROOT / "ml" / "reports" / "atmograph_dashboard.html"
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}
PROVENANCE_COLUMNS = [
    "dataset_version",
    "seed_id",
    "hops",
    "max_nodes",
    "filter_sources",
    "filter_relationships",
    "filter_split",
    "filter_min_degree",
    "exported_at",
]
NODE_EXPORT_COLUMNS = [
    "id",
    "label",
    "source",
    "node_type",
    "degree",
    "in_degree",
    "out_degree",
    "component_id",
    "component_size",
    "industry",
    "country",
    "ticker",
    "product_category",
    "id_method",
    "mention_count",
    "embedding_available",
    "is_anonymised",
    "raw_degree",
    "depth",
    *PROVENANCE_COLUMNS,
]
EDGE_EXPORT_COLUMNS = [
    "id",
    "source",
    "target",
    "relationship",
    "confidence",
    "edge_count",
    "split",
    "source_datasets",
    "first_year",
    "last_year",
    *PROVENANCE_COLUMNS,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def query_value(params: dict[str, list[str]], name: str, default: str | None = None) -> str | None:
    values = params.get(name)
    if not values:
        return default
    return values[0]


def query_list(params: dict[str, list[str]], name: str) -> list[str]:
    result: list[str] = []
    for value in params.get(name, []):
        for item in value.split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:80] or "subgraph"


def csv_bytes(rows: list[dict], columns: list[str], context: dict[str, object]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        record = {column: row.get(column) for column in columns}
        record.update(context)
        writer.writerow({key: "" if value is None else value for key, value in record.items()})
    return buffer.getvalue().encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    dashboard: DashboardData
    server_version = "AtmoGraphDashboard/1.0"

    def _send_bytes(
        self,
        payload: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_error_json(self, message: str, status: int) -> None:
        self._send_json({"error": message, "generated_at": utc_now()}, status)

    def _send_download(self, payload: bytes, content_type: str, filename: str) -> None:
        self._send_bytes(
            payload,
            content_type,
            extra_headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    def _subgraph_from_params(self, params: dict[str, list[str]]) -> dict:
        seed = query_value(params, "seed", DEFAULT_SEED) or DEFAULT_SEED
        hops = int(query_value(params, "hops", "2") or "2")
        max_nodes = int(
            query_value(params, "max_nodes", str(DEFAULT_MAX_NODES))
            or str(DEFAULT_MAX_NODES)
        )
        min_degree = int(query_value(params, "min_degree", "0") or "0")
        split = query_value(params, "split", "all") or "all"
        sources = query_list(params, "sources")
        relationships = query_list(params, "relationships")
        return self.dashboard.subgraph(
            seed_id=seed,
            hops=hops,
            max_nodes=max_nodes,
            sources=sources,
            relationships=relationships,
            split=split,
            min_degree=min_degree,
        )

    def _serve_dashboard(self) -> None:
        if not DASHBOARD_PATH.exists():
            self._send_error_json(
                f"Dashboard HTML is missing: {DASHBOARD_PATH.relative_to(PROJECT_ROOT)}",
                HTTPStatus.NOT_FOUND,
            )
            return
        body = DASHBOARD_PATH.read_bytes()
        self._send_bytes(body, "text/html; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self.send_response(int(HTTPStatus.NO_CONTENT))
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        params = parse_qs(parsed.query, keep_blank_values=True)
        try:
            if path in {"/", "/index.html", "/atmograph_dashboard.html"}:
                self._serve_dashboard()
                return
            if path == "/favicon.ico":
                self._send_bytes(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
                return
            if path == "/api/health":
                self._send_json({"status": "ok", "generated_at": utc_now()})
                return
            if path == "/api/summary":
                self._send_json(self.dashboard.summary())
                return
            if path == "/api/analytics":
                self._send_json(self.dashboard.analytics)
                return
            if path == "/api/search":
                query = query_value(params, "q", "") or ""
                limit = int(query_value(params, "limit", "25") or "25")
                self._send_json({"query": query, "results": self.dashboard.search(query, limit)})
                return
            if path == "/api/subgraph":
                self._send_json(self._subgraph_from_params(params))
                return
            if path == "/api/export":
                payload = self._subgraph_from_params(params)
                fmt = (query_value(params, "format", "json") or "json").lower()
                context = {
                    "dataset_version": payload.get("dataset_version") or "",
                    "seed_id": payload["seed"]["id"],
                    "hops": payload["params"]["hops"],
                    "max_nodes": payload["params"]["max_nodes"],
                    "filter_sources": ",".join(payload["filters"]["sources"]),
                    "filter_relationships": ",".join(payload["filters"]["relationships"]),
                    "filter_split": payload["filters"]["split"],
                    "filter_min_degree": payload["filters"]["min_degree"],
                    "exported_at": utc_now(),
                }
                stem = f"atmograph_{safe_filename(payload['seed']['id'])}_h{context['hops']}_n{context['max_nodes']}"
                if fmt == "json":
                    body = json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                    ).encode("utf-8")
                    self._send_download(
                        body,
                        "application/json; charset=utf-8",
                        f"{stem}.json",
                    )
                elif fmt in {"nodes_csv", "node_csv", "nodes"}:
                    body = csv_bytes(payload["nodes"], NODE_EXPORT_COLUMNS, context)
                    self._send_download(
                        body,
                        "text/csv; charset=utf-8",
                        f"{stem}_nodes.csv",
                    )
                elif fmt in {"edges_csv", "edge_csv", "edges"}:
                    body = csv_bytes(payload["edges"], EDGE_EXPORT_COLUMNS, context)
                    self._send_download(
                        body,
                        "text/csv; charset=utf-8",
                        f"{stem}_edges.csv",
                    )
                else:
                    raise ValueError("format must be json, nodes_csv, or edges_csv")
                return
            if path == "/api/node_summary":
                node_id = query_value(params, "id", "") or ""
                if not node_id:
                    raise ValueError("Missing node id")
                self._send_json(self.dashboard.node_summary_by_id(node_id))
                return
            if path == "/api/node":
                node_id = query_value(params, "id", "") or ""
                if not node_id:
                    raise ValueError("Missing node id")
                self._send_json(
                    self.dashboard.node_detail(
                        node_id,
                        sources=query_list(params, "sources"),
                        relationships=query_list(params, "relationships"),
                        split=query_value(params, "split", "all") or "all",
                        min_degree=int(query_value(params, "min_degree", "0") or "0"),
                    )
                )
                return
            if path == "/api/experiment":
                self._send_json(self.dashboard.experiment)
                return
            if path == "/api/predict":
                self._send_error_json("Use POST for prediction", HTTPStatus.METHOD_NOT_ALLOWED)
                return
            self._send_error_json("Not found", HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._send_error_json(str(exc), HTTPStatus.NOT_FOUND)
        except PredictionUnavailable as exc:
            self._send_json(
                {"available": False, "error": str(exc), "generated_at": utc_now()},
                HTTPStatus.CONFLICT,
            )
        except (ValueError, TypeError) as exc:
            self._send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            self._send_error_json("Internal server error", HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path != "/api/predict":
                self._send_error_json("Not found", HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            source_id = str(payload.get("source_id", "")).strip()
            target_id = str(payload.get("target_id", "")).strip()
            if not source_id or not target_id:
                raise ValueError("source_id and target_id are required")
            self._send_json(self.dashboard.predict_pair(source_id, target_id))
        except PredictionUnavailable as exc:
            self._send_json(
                {"available": False, "error": str(exc), "generated_at": utc_now()},
                HTTPStatus.CONFLICT,
            )
        except KeyError as exc:
            self._send_error_json(str(exc), HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            self._send_error_json("Internal server error", HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {format % args}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    print("Loading graph and experiment artifacts...", file=sys.stderr)
    dashboard = DashboardData()
    DashboardHandler.dashboard = dashboard
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}/"
    print(
        json.dumps(
            {
                "url": url,
                "nodes": dashboard.num_nodes,
                "edges": dashboard.num_edges,
                "gnn_status": dashboard.experiment["gnn"]["status"],
                "baseline_available": dashboard.experiment["baseline"] is not None,
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

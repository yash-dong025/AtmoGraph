
"""
AtmoGraph - Member 1
NLP to Neo4j Entity Matching Adapter

Reads extracted entities from Member 2's NLP API,
compares them with Member 1's graph API,
and returns exact matches or fuzzy suggestions.

No database writes are performed.
"""

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch_json(url, method="GET"):
    request = Request(
        url,
        method=method,
        headers={"Accept": "application/json"}
    )

    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {body}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to {url}: {exc.reason}"
        ) from exc


def normalize_name(value):
    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def match_entity(entity, graph_nodes, threshold=0.78):
    name = entity.get("normalized_name") or entity.get("text", "")
    normalized = normalize_name(name)

    if not normalized:
        return {
            "name": name,
            "status": "unmatched",
            "matched_entity_id": None,
            "suggestions": []
        }

    # Exact ID/name matching
    exact = [
        node for node in graph_nodes
        if normalized in {
            normalize_name(node.get("name", "")),
            normalize_name(node.get("id", ""))
        }
    ]

    if len(exact) == 1:
        return {
            "name": name,
            "entity_type": entity.get("entity_type"),
            "status": "matched",
            "matched_entity_id": exact[0]["id"],
            "matched_name": exact[0].get("name"),
            "suggestions": []
        }

    if len(exact) > 1:
        return {
            "name": name,
            "entity_type": entity.get("entity_type"),
            "status": "ambiguous",
            "matched_entity_id": None,
            "suggestions": [
                {"id": n["id"], "name": n.get("name")}
                for n in exact
            ]
        }

    # Fuzzy matches are suggestions only.
    candidates = []

    for node in graph_nodes:
        score = SequenceMatcher(
            None,
            normalized,
            normalize_name(node.get("name", ""))
        ).ratio()

        if score >= threshold:
            candidates.append({
                "id": node["id"],
                "name": node.get("name"),
                "similarity": round(score, 3)
            })

    candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return {
        "name": name,
        "entity_type": entity.get("entity_type"),
        "status": "review" if candidates else "unmatched",
        "matched_entity_id": None,
        "suggestions": candidates[:3]
    }


def run(article_id, nlp_url, backend_url):
    print("\n=== AtmoGraph NLP-Graph Integration ===")

    # Read extracted entities from Member 2
    print("[1/3] Reading NLP annotations...")

    annotation = fetch_json(
        f"{nlp_url}/api/nlp/entities/{article_id}"
    )

    entities = annotation.get("entities", [])

    print(f"Article: {article_id}")
    print(f"Extracted entities: {len(entities)}")

    # Fetch Neo4j graph from Member 1
    print("[2/3] Fetching Neo4j graph...")

    graph = fetch_json(
        f"{backend_url}/api/v1/graph?limit=2000"
    )

    nodes = graph.get("nodes", [])

    print(f"Available graph nodes: {len(nodes)}")

    # Match extracted entities
    print("[3/3] Matching entities...")

    results = [
        match_entity(entity, nodes)
        for entity in entities
    ]

    summary = {
        "article_id": article_id,
        "total_extracted": len(entities),
        "matched": sum(
            item["status"] == "matched" for item in results
        ),
        "review_required": sum(
            item["status"] in {"review", "ambiguous"}
            for item in results
        ),
        "unmatched": sum(
            item["status"] == "unmatched" for item in results
        ),
        "results": results
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nMatching completed. Neo4j was not modified.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--article-id",
        default="mock-001"
    )
    parser.add_argument(
        "--nlp-url",
        default="http://127.0.0.1:8010"
    )
    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8000"
    )

    args = parser.parse_args()

    try:
        run(
            article_id=args.article_id,
            nlp_url=args.nlp_url.rstrip("/"),
            backend_url=args.backend_url.rstrip("/")
        )
    except RuntimeError as exc:
        parser.exit(1, f"ERROR: {exc}\n")

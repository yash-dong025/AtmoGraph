# Member 1 Neo4j graph schema

## Nodes
- `:Entity` with unique `id`, `name`, `kind`, `city`, `country`, `risk_score`.
- `kind` is `SUPPLIER`, `MANUFACTURER`, `PORT`, `WAREHOUSE`, or `RETAILER`.
- `:Disruption` has unique `id`, `kind`, `severity`, `description`, `observed_at`, `risk_score`.
- `:Prediction` has unique `id`, `horizon_days`, `predicted_delay_days`, `risk_score`, `model_name`, `generated_at`.

## Relationships
- `(:Entity)-[:SUPPLIES_TO]->(:Entity)`
- `(:Entity)-[:SHIPS_TO]->(:Entity)`
- `(:Entity)-[:DISTRIBUTES_TO]->(:Entity)`
- `(:Disruption)-[:AFFECTS]->(:Entity)`
- `(:Prediction)-[:PREDICTS_FOR]->(:Entity)`

Transport/business edges have numeric `transit_days`. Edges point **upstream to downstream**.

## Design notes
- Only supply-chain relationships are traversed by downstream queries.
- NLP/ML members populate disruptions/predictions; Member 1 only persists and serves them.
- `risk_score` is a demo score from 0..100, **not** a calibrated probability.
- Seed dataset is synthetic and not a validated real-world logistics network.
- Deleting an Entity removes connected relationships, including linked event/prediction edges, but not detached event/prediction nodes; manage those through an explicit retention policy for production.
- Version 1 permits business edges without a strict kind-to-kind matrix; agree on rules with the team.

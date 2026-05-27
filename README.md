# somop

Synthetic OMOP generator. Produces test datasets for the [Daphne](https://github.com/HDRUK/project-daphne-api) federated cohort discovery platform.

## Install

```bash
pip install -e .
```

## Typical workflow

```bash
# 1. Generate synthetic data files
somop generate --config configs/conditions.yaml

# 2. Spin up the full stack (Postgres + omop-lite loader + Bunny)
somop run \
  --config configs/conditions.yaml \
  --collection-id <bunny-collection-uuid> \
  --api-url http://host.docker.internal:8100/api/v1 \
  --api-username admin@example.com \
  --api-password secret
```

`somop run` is self-contained: it spins up a local Postgres container, loads the OMOP data using [omop-lite](https://github.com/Health-Informatics-UoN/omop-lite), then starts two Bunny instances — all in a single `docker-compose.run.yaml`. Ctrl+C stops and removes all containers.

---

## Commands

### `somop generate` — create synthetic data files

```bash
somop generate --config configs/conditions.yaml [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | *(required)* | Path to YAML config |
| `--out-dir` | from config | Override output directory |
| `--n-people` | from config | Override `person.n_people` |
| `--seed` | from config | Override random seed |
| `--chunk-size` | from config | Override chunk size |
| `--concepts` | — | Path to `CONCEPT.csv` to copy into the data directory |

Output files (tab-separated `.csv`) are written to the `out_dir` from the config (default: `.`):

`PERSON.csv` · `CONDITION_OCCURRENCE.csv` · `DRUG_EXPOSURE.csv` · `MEASUREMENT.csv` · `OBSERVATION.csv` · `PROCEDURE_OCCURRENCE.csv` · `SPECIMEN.csv` · `DEATH.csv` · `LOCATION.csv`

---

### `somop run` — full stack: Postgres + loader + Bunny

Generates a `docker-compose.run.yaml` containing all services and starts them. No prior `somop load` needed.

```bash
somop run --config configs/conditions.yaml \
  --collection-id <uuid> \
  --api-url http://host.docker.internal:8100/api/v1 \
  --api-username admin@example.com \
  --api-password secret
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | *(required)* | Path to YAML config |
| `--collection-id` | *(required)* | Bunny collection UUID |
| `--api-url` | *(required)* | Daphne API base URL |
| `--api-username` / `TASK_API_USERNAME` | *(required)* | Daphne API username |
| `--api-password` / `TASK_API_PASSWORD` | *(required)* | Daphne API password |
| `--db-name` | config file stem | Postgres database name |
| `--db-password` / `DB_PASSWORD` | `postgres` | Postgres password |
| `--db-url` / `DATABASE_URL` | — | Use an external Postgres instead of a container (`postgresql://user:pass@host:port/db`) |
| `--db-port` | — | Expose the containerised Postgres on this host port |
| `--drop-db` / `--no-drop-db` | drop | When using `--db-url`: drop and recreate the DB before loading |
| `--concepts` | — | Path to `CONCEPT.csv` to copy into the data directory before loading |
| `--bunny-build` | — | Path to a local Bunny build context (Dockerfile directory) instead of pulling the remote image |
| `--compose-out` | `<out_dir>/docker-compose.run.yaml` | Where to write the compose file |

**With an external Postgres** (e.g. a persistent dev DB):

```bash
somop generate --config configs/mortality_conditions.yaml --concepts ./data/CONCEPT.csv

somop run \
  --config configs/mortality_conditions.yaml \
  --collection-id <uuid> \
  --api-url http://host.docker.internal:8100/api/v1 \
  --api-username <username> \
  --api-password <password> \
  --db-url postgresql://postgres:postgres@host.docker.internal:5435/mortality_conditions \
  --bunny-build ../bunny/hutch-bunny
```

---

### `somop load` — load data into a persistent database (advanced)

Use this when you want to load data into a long-lived database independently of running Bunny — for example to inspect the schema, reload with a different dataset, or share a DB across multiple runs.

```bash
somop load --config configs/conditions.yaml [--db-url ...] [--concepts ...]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | *(required)* | Path to YAML config |
| `--db-name` | config file stem | Postgres database name |
| `--db-password` / `DB_PASSWORD` | `postgres` | Postgres password |
| `--db-url` / `DATABASE_URL` | — | Use an existing Postgres instead of a container |
| `--db-port` | — | Expose the containerised Postgres on this host port |
| `--drop-db` / `--no-drop-db` | drop | Drop and recreate the DB before loading |
| `--concepts` | — | Path to `CONCEPT.csv` to copy into the data directory |
| `--compose-out` | `<out_dir>/docker-compose.load.yaml` | Where to write the compose file |

---

## Config file structure

```yaml
seed: 42
out_dir: ./data/my_dataset
chunk_size: 100_000

person:
  n_people: 5000
  genders:
    - concept_id: 8507  # male
      p: 0.5
    - concept_id: 8532  # female
      p: 0.5
  age_dist: normal      # normal | lognormal | uniform
  age_param1: 40.0      # mean
  age_param2: 18.0      # sd
  min_age: 0.0
  max_age: 110.0

condition:
  items:
    - concept_id: 198185
      p: 0.60

drug_exposure:
  items:
    - concept_id: 1503297
      p: 0.30

measurement:
  items:
    - concept_id: 3955314
      unit_concept_id: 12310002
      p: 0.70
      dist: lognormal   # adds value_as_number; normal | lognormal | uniform
      param1: 7.0       # mu
      param2: 0.7       # sigma

observation:
  items:
    - concept_id: 4275495
      p: 0.40

procedure:
  items:
    - concept_id: 4047494
      p: 0.25

specimen:
  items:
    - concept_id: 4001225
      p: 0.20

death:
  p: 0.05
  causes:
    - concept_id: 4306655
      p: 1.0
  death_type_concept_id: 32519  # EHR

location:
  items:
    - location_id: 1
      city: London
      country_source_value: GBR
      latitude: 51.5074
      longitude: -0.1278

interactions:
  after_drug_exposure:
    condition: 1.2      # multiply condition p by 1.2 for persons with a drug exposure
    measurement: 1.5
  after_condition:
    observation: 1.3
```

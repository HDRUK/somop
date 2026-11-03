from __future__ import annotations
import os
from typing import Optional
from urllib.parse import urlparse


def _parse_db_url(url: str) -> dict:
    p = urlparse(url)
    return {
        "host": p.hostname,
        "port": p.port or 5432,
        "user": p.username or "postgres",
        "password": p.password or "postgres",
        "dbname": p.path.lstrip("/"),
    }


def build_load_compose(
    *,
    db_name: str,
    db_password: str,
    data_dir: str,
    db_port: Optional[int] = None,
    db_url: Optional[str] = None,
    drop_db: bool = True,
) -> dict:
    abs_data_dir = os.path.abspath(data_dir)
    services = {}

    if db_url:
        conn = _parse_db_url(db_url)
        db_host = conn["host"]
        db_conn_port = conn["port"]
        db_user = conn["user"]
        db_password = conn["password"]
        db_name = conn["dbname"]

        if drop_db:
            terminate = (
                f"psql -h {db_host} -p {db_conn_port} -U {db_user} -d postgres -c "
                f"\"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname='{db_name}' AND pid <> pg_backend_pid();\""
            )
            init_cmd = (
                f"{terminate} && "
                f"dropdb -h {db_host} -p {db_conn_port} -U {db_user} --if-exists {db_name} && "
                f"createdb -h {db_host} -p {db_conn_port} -U {db_user} {db_name}"
            )
        else:
            init_cmd = f"createdb -h {db_host} -p {db_conn_port} -U {db_user} {db_name} || true"

        services["init-db"] = {
            "image": "postgres:16",
            "environment": {"PGPASSWORD": db_password},
            "command": ["sh", "-c", init_cmd],
            "restart": "no",
        }

        loader = {
            "image": "ghcr.io/health-informatics-uon/omop-lite:latest",
            "depends_on": {
                "init-db": {"condition": "service_completed_successfully"},
            },
            "environment": {
                "DB_HOST": db_host,
                "DB_PORT": db_conn_port,
                "DB_USER": db_user,
                "DB_PASSWORD": db_password,
                "DB_NAME": db_name,
                "SCHEMA_NAME": "public",
                "DATA_DIR": "/data",
            },
            "volumes": [f"{abs_data_dir}:/data:ro"],
        }
    else:
        db_service = {
            "image": "postgres:16",
            "environment": {
                "POSTGRES_PASSWORD": db_password,
                "POSTGRES_USER": "postgres",
                "POSTGRES_DB": db_name,
            },
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U postgres"],
                "interval": "5s",
                "timeout": "5s",
                "retries": 10,
            },
            "volumes": ["pgdata:/var/lib/postgresql/data"],
        }
        if db_port is not None:
            db_service["ports"] = [f"{db_port}:5432"]
        services["db"] = db_service

        loader = {
            "image": "ghcr.io/health-informatics-uon/omop-lite:latest",
            "depends_on": {
                "db": {"condition": "service_healthy"},
            },
            "environment": {
                "DB_HOST": "db",
                "DB_USER": "postgres",
                "DB_PASSWORD": db_password,
                "DB_NAME": db_name,
                "SCHEMA_NAME": "public",
                "DATA_DIR": "/data",
            },
            "volumes": [f"{abs_data_dir}:/data:ro"],
        }

    services["loader"] = loader
    compose = {"name": f"somop-{db_name}", "services": services}
    if not db_url:
        compose["volumes"] = {"pgdata": None}
    return compose


def _build_dataset_services(
    *,
    prefix: str,
    db_name: str,
    db_password: str,
    data_dir: str,
    collection_id: str,
    api_url: str,
    api_username: str,
    api_password: str,
    db_port: Optional[int] = None,
    db_url: Optional[str] = None,
    drop_db: bool = True,
    bunny_build: Optional[str] = None,
    skip_load: bool = False,
    bunny_a_interval: int = 5,
    bunny_b_interval: int = 20,
) -> tuple[dict, dict]:
    """Build the full services dict for one dataset.

    prefix="" → service names are db/loader/bunny-a/bunny-b (single-dataset mode).
    prefix="foo" → names are db-foo/loader-foo/bunny-a-foo/bunny-b-foo (multi-dataset mode).

    skip_load=True omits the loader (and init-db) so Bunny starts straight against an
    existing database; for the embedded Postgres this relies on the named data volume
    surviving a previous load.

    Returns (services, volumes) where volumes is the top-level named-volume declarations.
    """
    sfx = f"-{prefix}" if prefix else ""
    abs_data_dir = os.path.abspath(data_dir)
    services = {}
    volumes = {}

    if db_url:
        conn = _parse_db_url(db_url)
        db_host = conn["host"]
        db_conn_port = conn["port"]
        db_user = conn["user"]
        db_password = conn["password"]
        db_name = conn["dbname"]

        if not skip_load:
            if drop_db:
                terminate = (
                    f"psql -h {db_host} -p {db_conn_port} -U {db_user} -d postgres -c "
                    f"\"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    f"WHERE datname='{db_name}' AND pid <> pg_backend_pid();\""
                )
                init_cmd = (
                    f"{terminate} && "
                    f"dropdb -h {db_host} -p {db_conn_port} -U {db_user} --if-exists {db_name} && "
                    f"createdb -h {db_host} -p {db_conn_port} -U {db_user} {db_name}"
                )
            else:
                init_cmd = f"createdb -h {db_host} -p {db_conn_port} -U {db_user} {db_name} || true"

            services[f"init-db{sfx}"] = {
                "image": "postgres:16",
                "environment": {"PGPASSWORD": db_password},
                "command": ["sh", "-c", init_cmd],
                "restart": "no",
            }

            services[f"loader{sfx}"] = {
                "image": "ghcr.io/health-informatics-uon/omop-lite:latest",
                "depends_on": {
                    f"init-db{sfx}": {"condition": "service_completed_successfully"},
                },
                "environment": {
                    "DB_HOST": db_host,
                    "DB_PORT": db_conn_port,
                    "DB_USER": db_user,
                    "DB_PASSWORD": db_password,
                    "DB_NAME": db_name,
                    "SCHEMA_NAME": "public",
                    "DATA_DIR": "/data",
                },
                "volumes": [f"{abs_data_dir}:/data:ro"],
            }
    else:
        db_host = f"db{sfx}"
        db_conn_port = 5432
        db_user = "postgres"
        pgdata_vol = f"pgdata{sfx}"

        db_service = {
            "image": "postgres:16",
            "environment": {
                "POSTGRES_PASSWORD": db_password,
                "POSTGRES_USER": "postgres",
                "POSTGRES_DB": db_name,
            },
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U postgres"],
                "interval": "5s",
                "timeout": "5s",
                "retries": 10,
            },
            "volumes": [f"{pgdata_vol}:/var/lib/postgresql/data"],
        }
        if db_port is not None:
            db_service["ports"] = [f"{db_port}:5432"]
        services[f"db{sfx}"] = db_service
        volumes[pgdata_vol] = None

        if not skip_load:
            services[f"loader{sfx}"] = {
                "image": "ghcr.io/health-informatics-uon/omop-lite:latest",
                "depends_on": {
                    f"db{sfx}": {"condition": "service_healthy"},
                },
                "environment": {
                    "DB_HOST": f"db{sfx}",
                    "DB_USER": "postgres",
                    "DB_PASSWORD": db_password,
                    "DB_NAME": db_name,
                    "SCHEMA_NAME": "public",
                    "DATA_DIR": "/data",
                },
                "volumes": [f"{abs_data_dir}:/data:ro"],
            }

    if skip_load:
        # No loader to wait on: depend on the embedded DB being healthy, or nothing
        # for an external DB (assumed already up and populated).
        bunny_depends = (
            None if db_url else {f"db{sfx}": {"condition": "service_healthy"}}
        )
    else:
        bunny_depends = {f"loader{sfx}": {"condition": "service_completed_successfully"}}

    polling_intervals = {"a": bunny_a_interval, "b": bunny_b_interval}
    for bunny_type in ("a", "b"):
        services[f"bunny-{bunny_type}{sfx}"] = _bunny_service(
            db_name=db_name,
            db_host=db_host,
            db_port=db_conn_port,
            db_user=db_user,
            db_password=db_password,
            api_url=api_url,
            api_username=api_username,
            api_password=api_password,
            collection_id=collection_id,
            bunny_type=bunny_type,
            bunny_build=bunny_build,
            depends_on=bunny_depends,
            polling_interval=polling_intervals[bunny_type],
        )
    return services, volumes


def build_run_compose(
    *,
    db_name: str,
    db_password: str,
    data_dir: str,
    collection_id: str,
    api_url: str,
    api_username: str,
    api_password: str,
    db_port: Optional[int] = None,
    db_url: Optional[str] = None,
    drop_db: bool = True,
    bunny_build: Optional[str] = None,
    skip_load: bool = False,
    bunny_a_interval: int = 5,
    bunny_b_interval: int = 20,
) -> dict:
    services, volumes = _build_dataset_services(
        prefix="",
        db_name=db_name,
        db_password=db_password,
        data_dir=data_dir,
        collection_id=collection_id,
        api_url=api_url,
        api_username=api_username,
        api_password=api_password,
        db_port=db_port,
        db_url=db_url,
        drop_db=drop_db,
        bunny_build=bunny_build,
        skip_load=skip_load,
        bunny_a_interval=bunny_a_interval,
        bunny_b_interval=bunny_b_interval,
    )
    compose = {"name": f"somop-{db_name}", "services": services}
    if volumes:
        compose["volumes"] = volumes
    return compose


def build_multi_compose(datasets: list) -> dict:
    """Build a single compose file for multiple datasets.

    Each item in datasets must be a dict with keys matching _build_dataset_services parameters,
    plus 'prefix' (the dataset name used for service name namespacing).
    """
    all_services = {}
    all_volumes = {}
    for d in datasets:
        services, volumes = _build_dataset_services(**d)
        all_services.update(services)
        all_volumes.update(volumes)
    compose = {"name": "somop-multi", "services": all_services}
    if all_volumes:
        compose["volumes"] = all_volumes
    return compose


def _bunny_service(
    *,
    db_name: str,
    db_host: str = "db",
    db_port: int = 5432,
    db_user: str = "postgres",
    db_password: str,
    api_url: str,
    api_username: str,
    api_password: str,
    collection_id: str,
    bunny_type: str,
    bunny_build: Optional[str] = None,
    depends_on: Optional[dict] = None,
    polling_interval: int = 5,
) -> dict:
    image_or_build = (
        {"build": {"context": os.path.abspath(bunny_build)}}
        if bunny_build
        else {
            "image": "ghcr.io/health-informatics-uon/hutch/bunny:edge",
            # Always re-pull so we run the latest edge build of Bunny.
            "pull_policy": "always",
        }
    )
    service = {
        **image_or_build,
        "environment": {
            "DATASOURCE_DB_USERNAME": db_user,
            "DATASOURCE_DB_PASSWORD": db_password,
            "DATASOURCE_DB_DATABASE": db_name,
            "DATASOURCE_DB_DRIVERNAME": "postgresql",
            "DATASOURCE_DB_SCHEMA": "public",
            "DATASOURCE_DB_HOST": db_host,
            "DATASOURCE_DB_PORT": db_port,
            "BUNNY_LOGGER_LEVEL": "DEBUG",
            "TASK_API_BASE_URL": api_url,
            "TASK_API_USERNAME": api_username,
            "TASK_API_PASSWORD": api_password,
            "TASK_API_ENFORCE_HTTPS": "false",
            "TASK_API_TYPE": bunny_type,
            "COLLECTION_ID": collection_id,
            "OMOP_DEATH_ENABLED": "true",
            "OMOP_LOCATION_ENABLED": "true",
            "OMOP_SPECIMEN_ENABLED": "true",
            "POLLING_INTERVAL": polling_interval,
        },
    }
    if depends_on:
        service["depends_on"] = depends_on
    return service

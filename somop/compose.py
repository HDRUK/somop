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


def build_compose(
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
        db_host = "db"
        db_conn_port = 5432
        db_user = "postgres"

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
    services["bunny-a"] = _bunny_service(
        db_name=db_name,
        db_host=db_host,
        db_port=db_conn_port,
        db_user=db_user,
        db_password=db_password,
        api_url=api_url,
        api_username=api_username,
        api_password=api_password,
        collection_id=collection_id,
        bunny_type="a",
        bunny_build=bunny_build,
    )
    services["bunny-b"] = _bunny_service(
        db_name=db_name,
        db_host=db_host,
        db_port=db_conn_port,
        db_user=db_user,
        db_password=db_password,
        api_url=api_url,
        api_username=api_username,
        api_password=api_password,
        collection_id=collection_id,
        bunny_type="b",
        bunny_build=bunny_build,
    )

    return {"name": f"somop-{db_name}", "services": services}


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
) -> dict:
    image_or_build = (
        {"build": {"context": os.path.abspath(bunny_build)}}
        if bunny_build
        else {"image": "ghcr.io/health-informatics-uon/hutch/bunny:edge"}
    )
    return {
        **image_or_build,
        "depends_on": {
            "loader": {"condition": "service_completed_successfully"},
        },
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
            "OMOP_SPECIMEN_ENABLED": "true",
        },
    }

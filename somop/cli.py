import os
import shutil
import subprocess
import click
import yaml
from .generate import generate as run_generate
from .compose import build_load_compose, build_run_compose


@click.group()
def main():
    pass


@main.command(help="Generate synthetic OMOP data files from a YAML config.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(dir_okay=False, readable=True, path_type=str),
    help="Path to YAML configuration.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=str),
    default=None,
    help="Override: output directory",
)
@click.option("--seed", type=int, default=None, help="Override: random seed")
@click.option("--n-people", type=int, default=None, help="Override: person.n_people")
@click.option("--chunk-size", type=int, default=None, help="Override: chunk_size")
@click.option(
    "--concepts",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to CONCEPT.csv to copy into the data directory.",
)
def generate(config_path, out_dir, seed, n_people, chunk_size, concepts):
    overrides = {}
    if out_dir is not None:
        overrides["out_dir"] = out_dir
    if seed is not None:
        overrides["seed"] = seed
    if n_people is not None or chunk_size is not None:
        overrides.setdefault("person", {})
        if n_people is not None:
            overrides["person"]["n_people"] = n_people
        if chunk_size is not None:
            overrides["chunk_size"] = chunk_size

    paths = run_generate(config=config_path, overrides=overrides or None)
    data_dir = os.path.dirname(os.path.abspath(paths["person"]))
    click.echo(f"Generated data in: {data_dir}")

    if concepts:
        dest = os.path.join(data_dir, "CONCEPT.csv")
        shutil.copy2(concepts, dest)
        click.echo(f"Copied CONCEPT.csv to: {dest}")


@main.command(help="Load generated data into Postgres using omop-lite.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(dir_okay=False, readable=True, path_type=str),
    help="Path to YAML configuration.",
)
@click.option("--db-name", default=None, help="Database name (default: config file stem).")
@click.option(
    "--db-password",
    default="postgres",
    show_default=True,
    envvar="DB_PASSWORD",
    help="Postgres password. Also reads DB_PASSWORD env var.",
)
@click.option(
    "--db-url",
    default=None,
    envvar="DATABASE_URL",
    help="Use an existing Postgres: postgresql://user:password@host:port/dbname. Also reads DATABASE_URL env var.",
)
@click.option(
    "--db-port",
    default=None,
    type=int,
    help="Expose Postgres on this host port. Omit if you don't need host access.",
)
@click.option(
    "--drop-db/--no-drop-db",
    default=True,
    show_default=True,
    help="When using --db-url: drop the database before creating it.",
)
@click.option(
    "--concepts",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to CONCEPT.csv to copy into the data directory.",
)
@click.option(
    "--compose-out",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    help="Where to write the generated docker-compose file (default: <out_dir>/docker-compose.load.yaml).",
)
def load(config_path, db_name, db_password, db_url, db_port, drop_db, concepts, compose_out):
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f) or {}
    data_dir = os.path.abspath(raw_cfg.get("out_dir", "."))

    if concepts:
        dest = os.path.join(data_dir, "CONCEPT.csv")
        shutil.copy2(concepts, dest)
        click.echo(f"Copied CONCEPT.csv to: {dest}")

    if db_name is None:
        db_name = os.path.splitext(os.path.basename(config_path))[0]

    compose_dict = build_load_compose(
        db_name=db_name,
        db_password=db_password,
        data_dir=data_dir,
        db_port=db_port,
        db_url=db_url,
        drop_db=drop_db,
    )

    if compose_out is None:
        compose_out = os.path.join(data_dir, "docker-compose.load.yaml")
    with open(compose_out, "w") as f:
        yaml.dump(compose_dict, f, default_flow_style=False, sort_keys=False)
    click.echo(f"Wrote docker-compose to: {compose_out}")

    subprocess.run(["docker", "compose", "-f", compose_out, "up", "--build"], check=True)


@main.command(help="Spin up the full stack: load data into Postgres and start Bunny. Ctrl+C stops and removes containers.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(dir_okay=False, readable=True, path_type=str),
    help="Path to YAML configuration.",
)
@click.option("--collection-id", required=True, help="Bunny collection UUID.")
@click.option(
    "--api-url",
    required=True,
    help="Daphne API base URL (e.g. http://host.docker.internal:8100/api/v1).",
)
@click.option(
    "--api-username",
    required=True,
    envvar="TASK_API_USERNAME",
    help="Daphne API username. Also reads TASK_API_USERNAME env var.",
)
@click.option(
    "--api-password",
    required=True,
    envvar="TASK_API_PASSWORD",
    help="Daphne API password. Also reads TASK_API_PASSWORD env var.",
)
@click.option("--db-name", default=None, help="Database name (default: config file stem).")
@click.option(
    "--db-password",
    default="postgres",
    show_default=True,
    envvar="DB_PASSWORD",
    help="Postgres password. Also reads DB_PASSWORD env var.",
)
@click.option(
    "--db-url",
    default=None,
    envvar="DATABASE_URL",
    help="Use an existing Postgres instead of a container: postgresql://user:password@host:port/dbname. Also reads DATABASE_URL env var.",
)
@click.option(
    "--db-port",
    default=None,
    type=int,
    help="Expose the containerised Postgres on this host port. Omit if you don't need host access.",
)
@click.option(
    "--drop-db/--no-drop-db",
    default=True,
    show_default=True,
    help="When using --db-url: drop the database before loading.",
)
@click.option(
    "--concepts",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to CONCEPT.csv to copy into the data directory before loading.",
)
@click.option(
    "--bunny-build",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    help="Path to a local Bunny build context (directory with a Dockerfile).",
)
@click.option(
    "--compose-out",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    help="Where to write the generated docker-compose file (default: <out_dir>/docker-compose.run.yaml).",
)
def run(
    config_path,
    collection_id,
    api_url,
    api_username,
    api_password,
    db_name,
    db_password,
    db_url,
    db_port,
    drop_db,
    concepts,
    bunny_build,
    compose_out,
):
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f) or {}
    data_dir = os.path.abspath(raw_cfg.get("out_dir", "."))

    if concepts:
        dest = os.path.join(data_dir, "CONCEPT.csv")
        shutil.copy2(concepts, dest)
        click.echo(f"Copied CONCEPT.csv to: {dest}")

    if db_name is None:
        db_name = os.path.splitext(os.path.basename(config_path))[0]

    compose_dict = build_run_compose(
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
    )

    if compose_out is None:
        compose_out = os.path.join(data_dir, "docker-compose.run.yaml")
    with open(compose_out, "w") as f:
        yaml.dump(compose_dict, f, default_flow_style=False, sort_keys=False)
    click.echo(f"Wrote docker-compose to: {compose_out}")

    click.echo("Cleaning up any existing containers...")
    subprocess.run(["docker", "compose", "-f", compose_out, "down"], check=True)

    click.echo(f"Starting Bunny from: {compose_out}")
    proc = subprocess.Popen(["docker", "compose", "-f", compose_out, "up", "--build"])
    try:
        proc.wait()
    except KeyboardInterrupt:
        click.echo("\nStopping...")
        subprocess.run(["docker", "compose", "-f", compose_out, "down"])
        proc.wait()


if __name__ == "__main__":
    main()

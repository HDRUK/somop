import os
import shutil
import subprocess
import click
import yaml
from .generate import generate as run_generate
from .compose import build_compose


@click.group()
def main():
    pass


@main.command(help="Generate OMOP-like mock data and write a docker-compose file.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(dir_okay=False, readable=True, path_type=str),
    help="Path to YAML configuration.",
)
@click.option("--collection-id", required=True, help="BUNNY collection UUID.")
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
    "--concepts",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to CONCEPT.csv to copy into the data directory.",
)
@click.option(
    "--compose-out",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    help="Where to write the generated docker-compose file (default: <out_dir>/docker-compose.yaml).",
)
@click.option(
    "--db-port",
    default=None,
    type=int,
    help="Expose Postgres on this host port (e.g. 5435). Omit if you don't need host access.",
)
@click.option(
    "--db-url",
    default=None,
    envvar="DATABASE_URL",
    help="Use an existing Postgres instead of creating a db container. "
         "Format: postgresql://user:password@host:port/dbname. "
         "Overrides --db-name and --db-password. Also reads DATABASE_URL env var.",
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
    "--bunny-build",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    help="Path to a local Bunny build context (directory with a Dockerfile). Uses this instead of pulling the remote image.",
)
@click.option("--force", is_flag=True, default=False, help="Regenerate data even if it already exists.")
@click.option(
    "--drop-db/--no-drop-db",
    default=True,
    show_default=True,
    help="When using --db-url: drop the database before creating it.",
)
def generate(
    config_path,
    collection_id,
    api_url,
    api_username,
    api_password,
    db_name,
    db_password,
    concepts,
    compose_out,
    db_port,
    db_url,
    out_dir,
    seed,
    n_people,
    chunk_size,
    force,
    drop_db,
    bunny_build,
):
    # 1. Build overrides for data generation
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

    # 2. Resolve out_dir from config
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f) or {}
    data_dir = os.path.abspath(overrides.get("out_dir", raw_cfg.get("out_dir", ".")))

    # 3. Generate data (skip if already present unless --force)
    if not force and os.path.exists(os.path.join(data_dir, "PERSON.csv")):
        click.echo(f"Data already exists in {data_dir}, skipping generation (use --force to regenerate).")
    else:
        paths = run_generate(config=config_path, overrides=overrides or None)
        data_dir = os.path.dirname(os.path.abspath(paths["person"]))
        click.echo(f"Generated data in: {data_dir}")

    # 4. Copy CONCEPT.csv if provided
    if concepts:
        dest = os.path.join(data_dir, "CONCEPT.csv")
        shutil.copy2(concepts, dest)
        click.echo(f"Copied CONCEPT.csv to: {dest}")

    # 5. Derive DB name from config stem if not set
    if db_name is None:
        db_name = os.path.splitext(os.path.basename(config_path))[0]

    # 6. Build and write compose file
    compose_dict = build_compose(
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
        compose_out = os.path.join(data_dir, "docker-compose.yaml")
    with open(compose_out, "w") as f:
        yaml.dump(compose_dict, f, default_flow_style=False, sort_keys=False)
    click.echo(f"Wrote docker-compose to: {compose_out}")


@main.command(help="Start the stack from a YAML config. Ctrl+C stops and removes containers.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(dir_okay=False, readable=True, path_type=str),
    help="Path to the YAML configuration used when the stack was generated.",
)
@click.option(
    "--compose-out",
    default=None,
    type=click.Path(dir_okay=False, path_type=str),
    help="Explicit path to the docker-compose file (default: <out_dir>/docker-compose.yaml).",
)
def run(config_path, compose_out):
    if compose_out is None:
        with open(config_path) as f:
            raw_cfg = yaml.safe_load(f) or {}
        out_dir = os.path.abspath(raw_cfg.get("out_dir", "."))
        compose_out = os.path.join(out_dir, "docker-compose.yaml")

    if not os.path.exists(compose_out):
        raise click.ClickException(
            f"Compose file not found: {compose_out}\nRun 'somop generate --config {config_path} ...' first."
        )

    click.echo(f"Cleaning up any existing containers...")
    subprocess.run(["docker", "compose", "-f", compose_out, "down"], check=True)

    click.echo(f"Starting stack from: {compose_out}")
    proc = subprocess.Popen(["docker", "compose", "-f", compose_out, "up", "--build"])
    try:
        proc.wait()
    except KeyboardInterrupt:
        click.echo("\nStopping stack...")
        subprocess.run(["docker", "compose", "-f", compose_out, "down"])
        proc.wait()


if __name__ == "__main__":
    main()

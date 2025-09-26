import click
from .generate import generate as run_generate


@click.command(help="Generate OMOP-like mock data from a YAML configuration.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, exists=False, readable=True, path_type=str),
    default=None,
    show_default=True,
    help="Path to YAML configuration. If omitted, defaults are used.",
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
def main(config_path, out_dir, seed, n_people, chunk_size):
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
    click.echo(f"Done. Wrote outputs to: {paths}")


if __name__ == "__main__":
    main()

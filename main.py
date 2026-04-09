
import click


@click.group()
@click.version_option(version="0.1.0", prog_name="libucks")
def cli():
    """libucks — Librarian Buckets, local AI memory server for coding agents."""


if __name__ == "__main__":
    cli()

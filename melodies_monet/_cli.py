# SPDX-License-Identifier: Apache-2.0
#
"""
melodies-monet -- MELODIES MONET CLI
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path

_LOGGING_LEVEL = os.environ.get("MM_LOGGING_LEVEL", None)
if _LOGGING_LEVEL is not None:
    import logging

    logging.basicConfig(level=_LOGGING_LEVEL.upper())

try:
    import typer
except ImportError as e:
    print(
        "The MELODIES MONET CLI requires the module 'typer'. "
        "You can install it with `conda install -c conda-forge typer` or "
        "`pip install typer`. "
        f"The error message was: {e}"
    )
    raise SystemExit(1)

DEBUG = False
INFO_COLOR = typer.colors.CYAN
ERROR_COLOR = typer.colors.BRIGHT_RED
SUCCESS_COLOR = typer.colors.GREEN

HEADER = """
------------------
| MELODIES MONET |
------------------
""".strip()


def _get_full_name(obj):
    """Get the full name of a function or type,
    including the module name if not builtin."""
    import builtins
    import inspect

    mod = inspect.getmodule(obj)
    name = obj.__qualname__
    if mod is None or mod is builtins:
        return name
    else:
        return f"{mod.__name__}.{name}"


@contextmanager
def _timer(desc=""):
    start = time.perf_counter()

    tpl = f"{desc} {{status}} in {{elapsed:.3g}} seconds"

    typer.secho(f"{desc} ...", fg=INFO_COLOR)
    try:
        yield
    except Exception as e:
        typer.secho(
            tpl.format(status="failed", elapsed=time.perf_counter() - start),
            fg=ERROR_COLOR,
        )
        typer.secho(f"Error message (type: {_get_full_name(type(e))}): {e}", fg=ERROR_COLOR)
        if DEBUG:
            raise
        else:
            typer.echo("(Use the '--debug' flag to see more info.)")
            raise typer.Exit(1)
    else:
        typer.secho(
            tpl.format(status="succeeded", elapsed=time.perf_counter() - start),
            fg=SUCCESS_COLOR,
        )


@contextmanager
def _ignore_pandas_numeric_only_futurewarning():
    """Disable pandas `numeric_only` FutureWarning"""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=(
                "The default value of numeric_only in DataFrameGroupBy.mean is deprecated. "
                "In a future version, numeric_only will default to False. "
                "Either specify numeric_only or select only columns "
                "which should be valid for the function."
            ),
        )
        yield


def _version_callback(value: bool):
    from . import __version__

    if value:
        typer.echo(f"melodies-monet {__version__}")
        # TODO: monet/monetio versions?
        raise typer.Exit()


app = typer.Typer()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version/",
        help="Print version.",
        callback=_version_callback,
        is_eager=True,
    ),
):
    """MELODIES MONET"""


@app.command()
def run(
    control: str = typer.Argument(
        ...,
        help="Path to the control file to use.",
    ),
    debug: bool = typer.Option(False, "--debug/", help="Print more messages (including full tracebacks)."),
):
    """Run MELODIES MONET as described in the control file CONTROL."""

    global DEBUG

    DEBUG = debug

    p = Path(control)
    if not p.is_file():
        typer.echo(f"Error: control file {control!r} does not exist")
        raise typer.Exit(2)

    typer.echo(HEADER)
    typer.secho(f"Using control file: {control!r}", fg=INFO_COLOR)
    typer.secho(f"with full path: {p.absolute().as_posix()}", fg=INFO_COLOR)

    with _timer("Importing the driver"):
        from melodies_monet.driver import analysis

    with _timer("Reading control file and initializing"):
        an = analysis()
        an.control = control
        an.read_control()
        if debug and not an.debug:
            typer.secho(
                f"Setting `analysis.debug` (was {an.debug}) to True since --debug used.",
                fg=INFO_COLOR,
            )
            an.debug = True

    with _timer("Opening model(s)"):
        an.open_models()

    # Note: currently MM expects having at least model and at least one obs
    # but in the future, model-to-model only might be an option
    with _timer("Opening observations(s)"):
        an.open_obs()

    with _timer("Pairing"):
        if an.read is not None:
            an.read_analysis()
        else:
            an.pair_data()

    if an.save is not None:
        with _timer("Saving paired datasets"):
            an.save_analysis()

    if an.control_dict.get("plotting") is not None:
        with (
            _timer("Plotting and saving the figures"),
            _ignore_pandas_numeric_only_futurewarning(),
        ):
            an.plotting()

    if an.control_dict.get("stats") is not None:
        with (
            _timer("Computing and saving statistics"),
            _ignore_pandas_numeric_only_futurewarning(),
        ):
            an.stats()


cli = app

_typer_click_object = typer.main.get_command(app)  # for sphinx-click in docs


if __name__ == "__main__":
    cli()

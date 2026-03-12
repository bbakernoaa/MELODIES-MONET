======================
Command Line Interface
======================

Installing :mod:`melodies_monet` provides a command line interface (CLI):
``melodies-monet``.

For example, you can use the CLI to run control files without writing
any Python code::

    melodies-monet run control.yml

**Subcommands**

* |run|_ -- run a control file

.. |run| replace:: ``run``
.. _run: #melodies-monet-run

.. click:: melodies_monet._cli:_typer_click_object
   :prog: melodies-monet
   :nested: full

Data Retrieval
--------------

Data retrieval commands have been moved to the :mod:`monetio` CLI.
Please refer to the `MONETIO CLI documentation <https://monetio.readthedocs.io/en/stable/cli.html>`_
for information on how to download and process data from various observation networks.

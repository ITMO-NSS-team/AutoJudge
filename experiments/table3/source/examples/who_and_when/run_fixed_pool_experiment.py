"""Run the controlled 9Bvm experiment; use --help for prepare/run/report commands."""

if __package__:
    from .q1.cli import main
else:
    from q1.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

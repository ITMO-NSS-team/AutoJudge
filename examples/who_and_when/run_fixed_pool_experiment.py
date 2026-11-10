"""Run the controlled fixed-pool experiment; use --help for prepare/run/report commands."""

if __package__:
    from .fixed_pool.cli import main
else:
    from fixed_pool.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

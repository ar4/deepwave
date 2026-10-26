"""Deepwave-vs-third-party comparison benchmark scripts.

Runs the same problem through deepwave and other differentiable
wave-propagation packages (e.g. Devito, j-Wave) to give honest, reproducible
comparative numbers. No results are checked in: wall-clock timings are not
portable between machines or package versions, so run the three scripts
yourself, back to back on one machine, and compare the JSON files they
write. Like `benchmarks/`, this package is development-only and not part of
the installed `deepwave` distribution.
"""

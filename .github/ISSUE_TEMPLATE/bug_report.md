---
name: Bug report
about: Report incorrect output, a crash, or unexpected behaviour
title: ""
labels: bug
assignees: ""
---

**Describe the bug**
A clear, concise description of what went wrong.

**Minimal reproduction**
A minimal, self-contained script (ideally under ~30 lines, no external data
files) that reproduces the issue. Issues without a reproduction are much
harder to act on and may be closed pending one.

```python
import torch
import deepwave

# ...
```

**Expected behaviour**
What you expected to happen instead.

**Actual behaviour**
What actually happened (error message / traceback, or a description of the
incorrect output — a plot or printed tensor values helps).

**Environment**
- Deepwave version (`python -c "import deepwave; print(deepwave.__version__)"`):
- PyTorch version:
- OS:
- CPU or GPU (and if GPU, which):
- Python version:
- Installed from PyPI wheel or built from source?

**Additional context**
Anything else relevant — e.g. whether this worked in a previous Deepwave
version, whether it only happens with a particular `accuracy`/`dtype`/
`python_backend`, etc.

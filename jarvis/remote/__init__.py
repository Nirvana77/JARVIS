"""M3: the remote-edge split — the brain (``python -m jarvis serve``) and the
audio satellite (``python -m jarvis edge``).

Nothing in here is imported by the all-in-one path, and nothing in here imports
a model: the edge installs ``numpy``, ``sounddevice`` and ``websockets`` and
nothing else. See ``PRD/milestone-3-remote-edge.md``.
"""

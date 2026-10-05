"""Refinery — verified agent trajectories in, a small distilled model out.

Stage layout (see HLD.md §3):
    taskpool -> farm -> verifier -> compiler -> trainer -> evalkit

Every stage is a resumable, append-only JSONL transform driven by a single
`Config`. No stage knows which scale it is running at (LLD.md D-001).
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0.0"

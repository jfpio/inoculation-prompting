---
trigger: always_on
---

# Python Package Management Policy
Always use `uv` (by Astral) for all Python package management and environment tasks. 
- **Do not** use `pip install`, `poetry`, or `conda`.
- **Install packages:** Use `uv add <package>` (e.g., `uv add pandas`).
- **Run scripts:** Use `uv run <script.py>` to execute in the virtual environment.
- **Initialize projects:** Use `uv init` to create the `pyproject.toml`.
- **Sync dependencies:** Use `uv sync` to update the environment.
- **Tools:** Use `uv tool run <tool>` (or `uvx`) for CLI tools like ruff or black.
"""Development-only frame selection parsed from pipeline.yaml.

The parser covers the two blocks this tool needs. It is not a general YAML loader.
"""

from __future__ import annotations


def load_run_roles(text: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    in_split = False
    section = None
    for line in text.splitlines():
        if line.startswith("split:"):
            in_split = True
            section = None
            continue
        if not in_split:
            continue
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith(":") and not stripped.startswith("-"):
            section = stripped[:-1].strip()
            continue
        if stripped.startswith("- ") and section:
            roles[stripped[2:].strip()] = section
    return roles


def load_view_frames(text: str) -> list[tuple[str, int]]:
    frames: list[tuple[str, int]] = []
    in_block = False
    run = None
    for line in text.splitlines():
        if line.startswith("view_frames:"):
            in_block = True
            run = None
            continue
        if not in_block:
            continue
        if line and not line[0].isspace() and not line.startswith("#") and not line.startswith("-"):
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- run:"):
            run = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("run:"):
            run = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("message_index:"):
            if run is None:
                raise ValueError("message_index without run")
            frames.append((run, int(stripped.split(":", 1)[1].strip())))
            run = None
    return frames


def require_development(run: str, roles: dict[str, str]) -> str:
    role = roles.get(run)
    if role != "development":
        raise ValueError(
            f"просмотр development-кадров не открывает проезд {run!r} (роль {role!r})"
        )
    return role

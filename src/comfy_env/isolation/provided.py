"""input_files(): the declared provider for input-directory dropdowns.

One function, three habitats:

- **vanilla ComfyUI** (pack used without comfy-env): returns the live file
  list -- the pack is correct un-isolated, no comfy-env vocabulary leaks
  into /object_info;
- **the metadata scan child**: same live list, but as a ProvidedList whose
  `.provider` records the arguments -- the scan detects the tag and emits a
  `volatile_inputs` entry, which is how the parent knows to re-list the
  folder on every ask;
- **the worker**: a minimal inline twin lives in _persistent_worker.py's
  comfy_env stub (the worker is text-copied and must stay self-contained,
  ADR-0006).

This file must stay stdlib-only at import (folder_paths is imported lazily
at call time) and parseable by Python 3.9.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Union


class ProvidedList(list):
    """A list that remembers where its contents came from.

    `provider` is the recipe (a plain JSON-able dict); `offset`/`span` locate
    the provided run inside THIS list. Concatenation keeps the tag with the
    run's new position: Python calls the SUBCLASS operand's reflected
    method first, so `["none"] + provided` lands in __radd__ and the offset
    shifts right by len(lhs). sorted()/list() return plain lists -- the tag
    is correctly lost when the shape is no longer the provided run.
    """

    def __new__(cls, iterable=(), provider=None, offset=0, span=None):
        self = super().__new__(cls, iterable)
        return self

    def __init__(self, iterable=(), provider=None, offset=0, span=None):
        super().__init__(iterable)
        self.provider = provider
        self.offset = int(offset)
        self.span = len(self) if span is None else int(span)

    def __add__(self, other):
        return ProvidedList(list(self) + list(other),
                            provider=self.provider,
                            offset=self.offset, span=self.span)

    def __radd__(self, other):
        return ProvidedList(list(other) + list(self),
                            provider=self.provider,
                            offset=len(list(other)) + self.offset,
                            span=self.span)


def _normalize_sources(sources) -> List[Dict[str, Any]]:
    out = []
    for s in sources:
        if isinstance(s, str):
            out.append({"dir": s, "recursive": False, "rel_to_input": False})
        else:
            d = dict(s)
            d.setdefault("dir", "")
            d.setdefault("recursive", False)
            d.setdefault("rel_to_input", False)
            out.append(d)
    return out


def _list_sources(base: str, sources: List[Dict[str, Any]],
                  exts: Optional[Sequence[str]]) -> List[str]:
    """The one walker. Same semantics as the legacy marker engine: per-source
    dir/recursive/rel_to_input, extension filter (leading dot, case-folded),
    values slash-normalized, de-duplicated, sorted."""
    ex = {str(e).lower() for e in (exts or [])}
    names, seen = [], set()
    for src in sources:
        subdir = src.get("dir", "") or ""
        root = os.path.join(base, subdir) if subdir else base
        try:
            if src.get("recursive"):
                for r, _dirs, files in os.walk(root):
                    for fn in files:
                        if ex and os.path.splitext(fn)[1].lower() not in ex:
                            continue
                        rel = os.path.relpath(
                            os.path.join(r, fn),
                            base if src.get("rel_to_input") else root)
                        v = rel.replace(os.sep, "/")
                        if v not in seen:
                            seen.add(v)
                            names.append(v)
            else:
                for fn in os.listdir(root):
                    if not os.path.isfile(os.path.join(root, fn)):
                        continue
                    if ex and os.path.splitext(fn)[1].lower() not in ex:
                        continue
                    v = (os.path.join(subdir, fn).replace(os.sep, "/")
                         if src.get("rel_to_input") and subdir else fn)
                    if v not in seen:
                        seen.add(v)
                        names.append(v)
        except Exception:
            continue
    names.sort()
    return names


def input_files(sources: Union[str, Sequence],
                exts: Optional[Sequence[str]] = None,
                placeholder: Optional[str] = None) -> ProvidedList:
    """List input-directory files for a combo, and declare the recipe.

    sources: a dir string, or a sequence of dir strings / dicts
        ({"dir": "3d", "recursive": True, "rel_to_input": True}).
    exts: extension allow-list (with leading dots); None = everything.
    placeholder: single entry shown when nothing matches (keeps saved
        workflows loadable instead of an empty combo).
    """
    if isinstance(sources, str):
        sources = [sources]
    norm = _normalize_sources(sources)
    provider = {
        "kind": "input_dir",
        "sources": norm,
        "exts": [str(e).lower() for e in (exts or [])],
        "placeholder": placeholder,
    }
    try:
        import folder_paths  # ComfyUI core -- present in all three habitats
        base = folder_paths.get_input_directory()
        names = _list_sources(base, norm, exts)
    except Exception:
        names = []
    if not names and placeholder is not None:
        names = [placeholder]
    return ProvidedList(names, provider=provider, offset=0, span=len(names))

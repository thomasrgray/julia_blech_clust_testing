"""Resolve animal IDs and the confirmed legacy stimulus naming convention."""
import re
from pathlib import Path


def animal_for_session(session, mapping):
    """Prefer explicit mappings, otherwise require one TG ID across the path."""
    path = Path(session)
    for key in (str(path), path.name):
        if key in mapping:
            value = mapping[key]
            if value is None or not str(value).strip():
                raise ValueError(f"Empty animal ID for {key}")
            return str(value)
    matches = {match.upper() for part in path.parts for match in
               re.findall(r"(?:^|_)(TG\d+)(?=_|$)", part, re.I)}
    if len(matches) != 1:
        raise ValueError(f"Cannot determine one animal for {session}; supply --animal-map")
    return matches.pop()


def canonicalize_stimulus_names(names):
    """Translate waterIA/BA to retro and accompanying plain IA/BA to ortho.

    Only the presence of waterIA or waterBA activates the legacy convention
    for that odor. Already canonical IA/BA/oIA/oBA sessions are unchanged.
    Mapping is simultaneous; no trial, index, or source metadata is changed.
    """
    names = list(names)
    aliases = {}
    for odor in ("IA", "BA"):
        if "water" + odor in names:
            aliases["water" + odor] = odor
            if odor in names:
                aliases[odor] = "o" + odor
    result = [aliases.get(name, name) for name in names]
    sources = {}
    for original, target in zip(names, result):
        sources.setdefault(target, set()).add(original)
    collisions = {target: sorted(values) for target, values in sources.items()
                  if len(values) > 1}
    if collisions:
        raise ValueError(f"Ambiguous stimulus aliases: {collisions}")
    return result

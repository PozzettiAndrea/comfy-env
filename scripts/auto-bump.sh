#!/bin/bash
pkg="comfy-env"
local_ver=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/' | tr -d '\r')
pypi_ver=$(pip index versions "$pkg" 2>/dev/null | head -1 | grep -oP '\(\K[^)]+' || echo "0.0.0")
higher=$(printf '%s\n%s' "$local_ver" "$pypi_ver" | sort -V | tail -1)
if [ "$local_ver" = "$pypi_ver" ] || [ "$higher" = "$pypi_ver" ]; then
  IFS='.' read -r major minor patch <<< "$pypi_ver"
  new_ver="${major}.${minor}.$((patch + 1))"
  sed -i "s/^version = \".*\"/version = \"${new_ver}\"/" pyproject.toml
  git add pyproject.toml
  echo "Bumped $pkg: $local_ver -> $new_ver (PyPI: $pypi_ver)"
fi

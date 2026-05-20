#!/usr/bin/env bash
# release-local.sh — Cut a llama-cpp plugin release from a local CUDA host.
#
# Why this exists: the standard reusable workflow (build-plugin.yml) runs
# on GHA ubuntu/macos/windows runners that don't ship the CUDA toolkit,
# and the plugin's dynamic-linked layout needs a multi-binary upload
# (cdylib + companion libggml-*.so.0 libraries) that the upstream
# workflow's single-binary contract can't carry.
#
# What it does:
#   1. Validates the tag matches plugin.toml / Cargo.toml version.
#   2. cargo build --release.
#   3. Resolves the companion `lib*.so.0` files from llama-cmake-cache
#      and stages them alongside the renamed cdylib.
#   4. Tarballs the companions into `lib{name}-{platform}-companions.tar.gz`.
#   5. SHA256s the cdylib and the companion tarball.
#   6. Generates `release-manifest.json` matching the resolver schema, plus
#      a `companions` field consumers can use to fetch + extract the
#      side-by-side libraries.
#   7. Pushes the tag (if local-only) and runs `gh release create` /
#      `gh release upload --clobber`.
#
# Usage:
#   scripts/release-local.sh v0.2.0
#   scripts/release-local.sh v0.2.0 --platform x86_64-linux  # default
#   scripts/release-local.sh v0.2.0 --dry-run                # build + stage, no upload
#
# Requirements:
#   - CUDA toolkit + nvcc reachable to llama-cpp-sys-4's CMake step.
#   - rustup / stable cargo.
#   - gh CLI authenticated against RemoteMedia-SDK/llama-cpp (or fork).
#   - Working tree clean on the commit being tagged.

set -euo pipefail

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

TAG=""
PLATFORM="x86_64-linux"
DRY_RUN=0
GH_REPO="${GH_REPO:-RemoteMedia-SDK/llama-cpp}"

while [ $# -gt 0 ]; do
  case "$1" in
    v*|V*)         TAG="$1"; shift ;;
    --platform)    PLATFORM="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --repo)        GH_REPO="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 64
      ;;
  esac
done

if [ -z "$TAG" ]; then
  echo "usage: $0 vX.Y.Z [--platform x86_64-linux] [--dry-run]" >&2
  exit 64
fi

# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DISPLAY_NAME="llama-cpp"
CRATE_LIB="llama_cpp_plugin"     # cargo turns hyphens → underscores
EXT="so"                          # linux/macos override below

case "$PLATFORM" in
  x86_64-linux|aarch64-linux)   EXT="so" ;;
  *-darwin)                      EXT="dylib" ;;
  *-windows)                     EXT="dll" ;;
  *)
    echo "unsupported --platform $PLATFORM" >&2
    exit 64
    ;;
esac

# ---------------------------------------------------------------------------
# Step 1 — version / tag sanity
# ---------------------------------------------------------------------------

PLUGIN_VER="$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' plugin.toml)"
EXPECTED_TAG="v${PLUGIN_VER}"

if [ "$TAG" != "$EXPECTED_TAG" ]; then
  echo "FATAL: tag '$TAG' doesn't match plugin.toml version ($PLUGIN_VER → expected '$EXPECTED_TAG')" >&2
  echo "       bump plugin.toml + Cargo.toml first, or pass the matching tag." >&2
  exit 1
fi

CARGO_VER="$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' Cargo.toml)"
if [ "$CARGO_VER" != "$PLUGIN_VER" ]; then
  echo "FATAL: Cargo.toml version ($CARGO_VER) doesn't match plugin.toml version ($PLUGIN_VER)" >&2
  exit 1
fi

# Clean working tree (no uncommitted changes other than the tag commit itself).
if [ "$DRY_RUN" -eq 0 ] && ! git diff --quiet HEAD --; then
  echo "FATAL: working tree dirty. Commit first." >&2
  git status --short >&2
  exit 1
fi

echo "==> Releasing $DISPLAY_NAME $TAG for $PLATFORM (repo: $GH_REPO, dry-run=$DRY_RUN)"

# ---------------------------------------------------------------------------
# Step 2 — cargo build --release
# ---------------------------------------------------------------------------

echo "==> cargo build --release"
cargo build --release

PLUGIN_SO="target/release/lib${CRATE_LIB}.${EXT}"
if [ ! -f "$PLUGIN_SO" ]; then
  echo "FATAL: build output missing: $PLUGIN_SO" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3 — locate companion libraries (the SONAME .so.0 files)
#
# llama-cpp-sys-4 stages the CMake build at target/llama-cmake-cache/<hash>/.
# target/release/ only contains the unversioned `.so` symlinks; the real
# `.so.0.X.Y` files (and their `.so.0` SONAME symlinks) live in the
# cmake-cache. Walk the cache to find the newest build directory.
# ---------------------------------------------------------------------------

echo "==> Locating companion libraries (.so.0)"
CMAKE_CACHE_ROOT="target/llama-cmake-cache"
if [ ! -d "$CMAKE_CACHE_ROOT" ]; then
  echo "FATAL: $CMAKE_CACHE_ROOT not found. Did llama-cpp-sys-4 build?" >&2
  exit 1
fi

# Pick the most recently modified <hash>/lib directory.
COMPANION_DIR="$(find "$CMAKE_CACHE_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%T@ %p\n' \
  | sort -nr | head -1 | awk '{print $2}')"

if [ -z "$COMPANION_DIR" ] || [ ! -d "$COMPANION_DIR" ]; then
  echo "FATAL: couldn't find a <hash>/lib dir under $CMAKE_CACHE_ROOT" >&2
  exit 1
fi
echo "    companion source: $COMPANION_DIR"

# Expected SONAMEs (per README "Build-time CUDA requirement" table).
EXPECTED_COMPANIONS=(
  libllama.so.0
  libggml.so.0
  libggml-base.so.0
  libggml-cpu.so.0
  libggml-cuda.so.0
  libggml-blas.so.0
  libllama-common.so.0
  libmtmd.so.0
)

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Copy each SONAME (resolving the symlink to the real .so.0.X.Y file but
# renaming the file to the SONAME so the dlopen chain still works after
# extraction).
for soname in "${EXPECTED_COMPANIONS[@]}"; do
  target="$COMPANION_DIR/$soname"
  if [ ! -L "$target" ] && [ ! -f "$target" ]; then
    echo "    warn: $soname missing in $COMPANION_DIR (skipping)"
    continue
  fi
  # Resolve the SONAME link to the real versioned file.
  real="$(readlink -f "$target")"
  cp -L "$real" "$STAGE/$soname"
  echo "    + $soname  (← $(basename "$real"))"
done

# ---------------------------------------------------------------------------
# Step 4 — assemble assets (renamed cdylib + companion tarball)
# ---------------------------------------------------------------------------

ARTIFACTS="$(mktemp -d)"
trap 'rm -rf "$STAGE" "$ARTIFACTS"' EXIT

ASSET_NAME="lib${DISPLAY_NAME}-${PLATFORM}.${EXT}"
COMPANION_NAME="lib${DISPLAY_NAME}-${PLATFORM}-companions.tar.gz"

cp -L "$PLUGIN_SO" "$ARTIFACTS/$ASSET_NAME"
( cd "$STAGE" && tar -czf "$ARTIFACTS/$COMPANION_NAME" -- *.so.0 )

# ---------------------------------------------------------------------------
# Step 5 — SHA256 sidecars
# ---------------------------------------------------------------------------

hash_file () {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    shasum -a 256 "$f" | awk '{print $1}'
  fi
}

ASSET_SHA="$(hash_file "$ARTIFACTS/$ASSET_NAME")"
COMPANION_SHA="$(hash_file "$ARTIFACTS/$COMPANION_NAME")"
printf '%s' "$ASSET_SHA"     > "$ARTIFACTS/$ASSET_NAME.sha256"
printf '%s' "$COMPANION_SHA" > "$ARTIFACTS/$COMPANION_NAME.sha256"

echo "==> Assets:"
echo "    $ASSET_NAME       $ASSET_SHA  ($(du -h "$ARTIFACTS/$ASSET_NAME" | cut -f1))"
echo "    $COMPANION_NAME   $COMPANION_SHA  ($(du -h "$ARTIFACTS/$COMPANION_NAME" | cut -f1))"

# ---------------------------------------------------------------------------
# Step 6 — release-manifest.json
#
# Schema is a superset of the standard reusable workflow's output:
#
#   {
#     "name": "llama-cpp",
#     "version": "v0.2.0",
#     "platforms": {
#       "x86_64-linux": {
#         "file":           "libllama-cpp-x86_64-linux.so",
#         "sha256":         "...",
#         "companions":     "libllama-cpp-x86_64-linux-companions.tar.gz",
#         "companions_sha256": "..."
#       }
#     }
#   }
#
# The `companions*` fields are optional from the resolver's perspective.
# Resolvers that don't know about them still find the canonical cdylib
# and can dlopen it provided the SONAMEs are already on LD_LIBRARY_PATH
# (else they error at load time, prompting the consumer to fetch the
# companions manifest entry by hand).
#
# If a release for the tag already exists, we MERGE this platform's
# entry into the existing release-manifest.json so multiple invocations
# (one per platform / per host) accrete platforms without clobbering
# each other.
# ---------------------------------------------------------------------------

EXISTING_MANIFEST=""
if [ "$DRY_RUN" -eq 0 ] && gh release view "$TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
  if gh release download "$TAG" --repo "$GH_REPO" --pattern release-manifest.json --dir "$ARTIFACTS" 2>/dev/null; then
    EXISTING_MANIFEST="$ARTIFACTS/release-manifest.json"
    echo "==> Merging into existing release-manifest.json from $TAG"
  fi
fi

python3 - "$ARTIFACTS" "$DISPLAY_NAME" "$TAG" "$PLATFORM" "$ASSET_NAME" "$ASSET_SHA" "$COMPANION_NAME" "$COMPANION_SHA" "${EXISTING_MANIFEST:-}" <<'PY'
import json, sys
from pathlib import Path

art, name, tag, platform, asset, asset_sha, comp, comp_sha, existing = sys.argv[1:]
art = Path(art)

if existing and Path(existing).exists():
    manifest = json.loads(Path(existing).read_text())
    manifest.setdefault("platforms", {})
    if manifest.get("version") != tag:
        # If an older tag's manifest accidentally got downloaded, start fresh.
        manifest = {"name": name, "version": tag, "platforms": {}}
else:
    manifest = {"name": name, "version": tag, "platforms": {}}

manifest["name"] = name
manifest["version"] = tag
manifest["platforms"][platform] = {
    "file":              asset,
    "sha256":            asset_sha,
    "companions":        comp,
    "companions_sha256": comp_sha,
}

out = art / "release-manifest.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(out.read_text())
PY

# ---------------------------------------------------------------------------
# Step 7 — push tag + upload
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> --dry-run: staged assets at $ARTIFACTS (NOT uploaded)"
  cp -r "$ARTIFACTS" "release-dry-run-$TAG-$PLATFORM"
  echo "    copied to ./release-dry-run-$TAG-$PLATFORM/ for inspection"
  exit 0
fi

# Ensure the local tag exists and is pushed.
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "FATAL: local tag $TAG doesn't exist. Run: git tag $TAG && git push origin $TAG" >&2
  exit 1
fi
git push origin "$TAG" 2>/dev/null || true   # idempotent; already pushed is fine

# Create the release if it doesn't exist; otherwise reuse it.
if ! gh release view "$TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
  echo "==> Creating release $TAG on $GH_REPO"
  gh release create "$TAG" --repo "$GH_REPO" --title "$TAG" --notes "Local release. See README §Releasing." \
    "$ARTIFACTS/$ASSET_NAME" \
    "$ARTIFACTS/$ASSET_NAME.sha256" \
    "$ARTIFACTS/$COMPANION_NAME" \
    "$ARTIFACTS/$COMPANION_NAME.sha256" \
    "$ARTIFACTS/release-manifest.json"
else
  echo "==> Uploading assets to existing release $TAG (clobber existing)"
  gh release upload "$TAG" --repo "$GH_REPO" --clobber \
    "$ARTIFACTS/$ASSET_NAME" \
    "$ARTIFACTS/$ASSET_NAME.sha256" \
    "$ARTIFACTS/$COMPANION_NAME" \
    "$ARTIFACTS/$COMPANION_NAME.sha256" \
    "$ARTIFACTS/release-manifest.json"
fi

echo "==> Done."
echo "    https://github.com/$GH_REPO/releases/tag/$TAG"

#!/bin/sh
set -eu

bundle_root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
if [ -x "$bundle_root/bin/muser" ]; then
    runtime_root=$bundle_root
    muser_binary="$bundle_root/bin/muser"
else
    echo "Muser's extracted runtime is incomplete; download the release again." >&2
    exit 1
fi
cd "$runtime_root"

# Arguments make the extracted CLI wrapper directly testable (`Start
# Muser.command --help`). Finder launches Muser.app through signed native code.
if [ "$#" -gt 0 ]; then
    exec "$muser_binary" "$@"
fi
exec "$muser_binary" up

# Sourced by pixi on every activation (see [tool.pixi.activation] in
# pyproject.toml): rebuild the rt.rustler extension in place whenever anything
# under rustler/ is newer than the built .so. Cargo is incremental, so an
# unchanged crate costs one mtime scan here; concurrent activations (slurm
# ranks in a shared clone) serialize on the lock and find the rebuild done.
_rt_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_rt_so=$_rt_root/src/rt/rustler.abi3.so
# Sources and manifests only: rustler/target is build output and is always
# newer than the extension it produced.
if [ ! -e "$_rt_so" ] || [ -n "$(find "$_rt_root/rustler/src" \
        "$_rt_root/rustler/Cargo.toml" "$_rt_root/rustler/Cargo.lock" \
        -newer "$_rt_so" -print -quit)" ]; then
    echo "rustler changed: maturin develop" >&2
    (
        cd "$_rt_root" &&
        flock rustler/.develop.lock maturin develop --uv --release >&2
    ) || return 1
fi
unset _rt_root _rt_so

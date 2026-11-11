# Android (bionic) build spike — tts_host free engine

Date: 2026-11-11. Purpose: de-risk running the hand-written TTS engine
(`tts_host --engine free`, the ORT-free path) in Termux userspace on the
locked-bootloader Dimensity 8100 target. No proot. Shipping sources were
not modified; all experiments ran on a scratch copy.

Toolchain: NDK r27.2 (`build/core/version.mk`: NDK_MAJOR 27,
NDK_MINOR 2), compiler `aarch64-linux-android26-clang++` (API level 26),
flags `-std=c++17 -O2`.

## Verdict

The free engine compiles and links for bionic aarch64 with **zero changes
to the engine sources**. The only ORT coupling is in the host glue
(`main.cpp`, `packed.h`), and at the object level it amounts to exactly
one symbol. A free-only bionic `tts_host` linked and passed `file`/symbol
inspection. It could not be *run* here: this machine has no bionic
`/system/bin/linker64`, so QEMU refuses the binary (expected; first run
must happen on the device itself).

## What built successfully

Compiled clean, unmodified, from the original tree
(`host/src/{main,g2p,vits_kernels,free_engine}.cpp`, headers from
`host/src` + `host/ort/include`):

- `g2p.cpp`, `vits_kernels.cpp`, `free_engine.cpp` — no ORT headers
  involved at all. `free_engine.cpp` includes `packed_core.h` (the
  ORT-free core: thread pool, GEMV/GEMM kernels, BlobStore), not
  `packed.h`. `cpu.h` feature detection uses `getauxval(AT_HWCAP /
  AT_HWCAP2)`, which bionic supports, so the runtime SDOT/fp16/i8mm
  dispatch will work on the Dimensity 8100's Cortex-A78 cores.
- `main.cpp` — also compiles clean (the ORT 1.22 C++ headers are
  portable), but its object references exactly **one** ORT symbol:

```
                 U OrtGetApiBase
```

Link without ORT fails with exactly:

```
ld.lld: error: undefined symbol: OrtGetApiBase
>>> referenced by main.cpp
>>>               obj/main.o:(__cxx_global_var_init)
```

With the spike stub (below) added, the link succeeds:

```
tts_host_test: ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV),
dynamically linked, interpreter /system/bin/linker64, not stripped
```

- Size: 2,084,128 B unstripped; 1,434,088 B stripped. (The musl phone
  build is 822,424 B; the delta is mainly the statically linked NDK
  libc++ via `-static-libstdc++`.)
- `readelf -d` NEEDED entries: only `libm.so`, `libdl.so`, `libc.so` —
  all bionic system libraries. No `libonnxruntime` dependency.

## Blockers hit

1. **CMake forces ORT into `tts_host`.** `host/CMakeLists.txt` links the
   imported `onnxruntime` target into `tts_host` unconditionally. The
   already-ORT-free targets (`tf_check_free`, `vits_check_free`) prove
   the split is possible, but there is no free-only `tts_host` target.
2. **`main.cpp` is not free-only compilable as shipped.** It includes
   `<onnxruntime_cxx_api.h>` (line 8), `packed.h` also includes it
   (line 6, for the ORT custom-op glue), and `main()` constructs
   `Ort::Env env(...)` unconditionally before the engine branch
   (the ORT load/run code is interleaved with the free code behind
   runtime `engine_free` checks, ~34 `Ort::` touchpoints). So a
   production free-only binary needs an ifdef/source split of the host
   glue — a build-system task, not an engine task.
3. **No local run validation for bionic binaries.** QEMU on this machine
   has no Android sysroot:

```
qemu-aarch64-static: Could not open '/system/bin/linker64': No such file or directory
```

   A fully static variant (`-static`) links but bionic's loader check
   rejects it under QEMU:

```
error: "./tts_host_test_static": executable's TLS segment is underaligned: alignment is 8 (skew 0), needs to be at least 64 for ARM64 Bionic
```

   The static binary's only TLS data is the two function-local
   `thread_local std::vector<float>` scratch buffers in
   `packed_core.h` (lines 329 and 409; they are the only `thread_local`
   in the whole host tree). This only matters for a fully static build;
   the dynamic Termux build goes through bionic's regular TLS
   allocation and is not affected. Recorded so nobody rediscovers it.

## Changes made (scratch copy only)

Shipping sources: **none**. All compiles read the original tree; the
edits below exist only in the copy.

1. **Added `ort_stub.c`** (new file, spike-only): implements
   `OrtGetApiBase()` returning an API table whose function pointers all
   point at a trap that prints
   `ort_stub: ORT API called in free-only spike build` and aborts.
   This satisfies the single ORT symbol `main.o` needs while making any
   accidental ORT-path execution fail loudly instead of silently.
   It is a spike scaffold, not a shippable component.
2. **Negative result, reverted:** adding `alignas(64)` to the two
   `thread_local` declarations in the copy's `packed_core.h` did not
   change the static binary's TLS segment alignment (stayed 8) — the
   segment alignment is set by the remaining TLS content/CRT, not by
   those declarations. Not a fix; don't retry it as one.

## Remaining port work for tts_host (estimate)

- **Free-only build option (~half a day):** add a CMake option that
  splits the ORT sections out of `main.cpp` (ifdef or a separate
  translation unit for the ORT load/run branches) and drops the
  `onnxruntime` link for that target; drive it with the NDK toolchain
  file (`ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-26`).
  The free branches in `main.cpp` are already cleanly separated by
  `engine_free` checks, so this is mechanical.
- **On-device smoke test (~half a day):** push the binary plus the
  existing asset set (`wo-packs/`, `host/packed_graphs/`, `host/refpacks/`,
  `host/data/`) to the Dimensity 8100, run `--engine free` against the
  standard request lines, and compare wavs with the Mi 4c build using
  the project's same-seed comparison habit. This is also the first
  chance to run the binary at all — budget the time for surprises.
- **No other changes expected:** the line-JSON stdin/stdout protocol is
  unchanged; the hardcoded dev `--root` default is already overridable
  by flag, which the deploy scripts use; no engine/kernel source needs
  a bionic-specific edit based on anything seen here.

Total: roughly one day for a free-only bionic `tts_host` built and
smoke-tested on the device, dominated by the `main.cpp` split and
on-device validation. The engine core itself needed no porting.
# Production split (2026-11-11)

The free-only build option from the estimate above is implemented in the
shipping sources. No stub is involved: with the option on, the ORT code
is preprocessed out and nothing references `OrtGetApiBase`.

## Mechanism

- **CMake option `POTATO_FREE_ONLY`** (default OFF) in `host/CMakeLists.txt`.
  ON: `tts_host` gets `-DPOTATO_FREE_ONLY=1`, links `pthread` on
  non-Android targets only (bionic has pthread inside libc; the Android
  branch of the option adds no `-lpthread`, which does not exist there),
  and does **not** link the imported `onnxruntime` target or set the ORT
  RPATH. OFF: the previous link line and RPATH, unchanged.
- **`src/main.cpp`** guards on the macro of the same name:
  - the `<onnxruntime_cxx_api.h>` include is compiled out;
  - `packed.h` / `vits_packed.h` (the ORT custom-op glue) are bypassed —
    under the macro, main.cpp includes `packed_core.h` instead, which is
    all the free path needs (`packed::BlobStore`, `packed::set_kernel_threads`);
    `packed.h` itself is untouched;
  - the ORT-only host glue (`SessionPack`, `load_and_patch`,
    `make_session`, `f32_tensor`, `i64_tensor`) is one guarded region;
  - in `main()`: the `Ort::Env`/sessions declarations, the ORT packed
    pools + `NamedSession`s + custom-op domain, the whole specs loading
    loop, `vits_sess_p`, and the request-loop `Ort::Allocator` are
    guarded out;
  - the three runtime branches (encoder, first+decode, vits) keep their
    free branch unguarded and have their ORT `else` branches guarded, so
    with the macro defined each collapses to the free branch alone;
    with it undefined the preprocessed token stream is the original code;
  - a free-only binary rejects `--engine ort` at startup with an explicit
    message instead of failing later.
- The engine sources (`free_engine.cpp`, `vits_kernels.cpp`, `g2p.cpp`,
  `packed_core.h`, …) are **not modified at all** — as the spike
  predicted, the split is entirely in the host glue and the build file.

## Build commands

Default (dual-engine, option OFF) and free-only x86-64, from `host/`:

```
cmake -S . -B build-verify      -DCMAKE_BUILD_TYPE=Release
cmake --build build-verify      -j2 --target tts_host tf_check_free vits_check_free

cmake -S . -B build-verify-free -DCMAKE_BUILD_TYPE=Release -DPOTATO_FREE_ONLY=ON
cmake --build build-verify-free -j2 --target tts_host tf_check_free vits_check_free
```

Free-only aarch64 bionic, NDK r27.2 toolchain file:

```
cmake -S . -B build-verify-ndk \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DPOTATO_FREE_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-verify-ndk -j2 --target tts_host
```

where `$NDK` is the local NDK r27.2 tree (`version.mk`: 27.2). One local
environment repair was needed first: this NDK tree was missing its
top-level `source.properties`, which the toolchain file parses for the
revision; a standard one (`Pkg.Revision = 27.2.11579264`) was written
into the tools tree (not the project tree) before configuring.

## Regression results (x86-64, this machine, 4 threads)

Identical in the default build and in the free-only build — same
binaries' check tools from each tree:

- `tf_check_free --mode packed --ref host/logits_packed_x86.bin`:
  `tf_agree 299 / tf_total 361`, `ref_argmax_agree 361`,
  `ref_max_logit_diff 8.86917e-05` — the established numbers, unmoved.
- `vits_check_free`: `free_packed_vs_fp32 corr 0.959880`
  (relrms 0.293293, maxerr 0.339176), `free_baked_vs_fp32 0.959881`,
  layers free_packed vs free_baked worst_corr 1.000000 over 226 — the
  established numbers, unmoved.
- Host-level smoke test, default build `--engine free` vs free-only
  build, same request (`"Hello, this is a test of the free engine."`,
  tag neutral1, seed 42, 4 threads): token sequences identical
  (69 steps), durations identical (2.73294 s), and the output wavs are
  byte-identical (sha256
  `4ddc1493aec97cc56e31c78dc85ac647ed26c065232b9441b4650f48c1c240ee`).
- Free-only x86 `tts_host`: `ldd` shows only libstdc++/libm/libgcc_s/
  libc; no ORT symbols (an `nm | grep -i ort` sweep matches only
  unrelated `std::*sort*` / `__cxa_guard_abort` names).

## Bionic binary inspection

`build-verify-ndk/tts_host` (the bionic binary still has not been *run*
anywhere — no bionic runtime on this machine; link/symbol inspection is
the bar, as in the spike):

- ELF64, AArch64, DYN; compiled `-O3` (plus `-g` from the NDK toolchain
  defaults — see the size note below).
- Undefined symbols: **no `OrtGetApiBase`** (an undefined-symbol sweep
  for "ort" matches only `abort` and `android_set_abort_message`).
- `readelf -d` NEEDED: `libm.so`, `libdl.so`, `libc.so` — bionic system
  libraries only, no `libonnxruntime`.
- Stripped size: **1,099,728 B** (unstripped 11,943,072 B — the NDK
  Release configure keeps `-g` and does not set
  `CMAKE_BUILD_TYPE`-driven strip; the stripped figure is the
  comparable one). This is *smaller* than the spike's hand-rolled
  stripped build (1,434,088 B): the CMake build uses
  `-ffunction-sections`/`-Wl,--gc-sections` from the toolchain, which
  the spike's manual compile line did not.

## Deviations from the spike's estimate

- The split itself took well under the estimated half day; it is
  ifdef-guards in one file plus one CMake option, exactly the
  "mechanical" outcome the spike predicted. The on-device smoke test
  (the other estimated half day) is unchanged in scope and still
  pending a device.
- Two build-system details the spike's manual compile line never hit:
  (1) `-lpthread` must not be linked under bionic (no separate
  library) — handled by the `NOT ANDROID` condition in the option;
  (2) this machine's NDK tree lacked `source.properties` and the
  toolchain file refuses to configure without it (environment repair,
  recorded above; not a project change).
- Verification build trees (`build-verify*`) were removed after the
  runs; the numbers above are from their outputs.

# Wolverine black-screen handoff

## Latest priority and measured restoration (2026-11-03)

User requested restoring splash performance before resuming transition fixes.
Removed the fabricated query-result helper/callsite patches, forced completion-counter stores, and GPU wait-value clearing. Returned Patch 7 to the disabled experimental group. Retained the four title-scoped patches listed by `wolverine_after_timestamp_gate.log`, including the exact branch-only startup workaround at `0x903f65b1d`.

Build succeeded. `wolverine_performance_restored.log` measured sustained **60 FPS / 60 flips per second through frame 1768**, restoring the pre-experiment splash behavior. Original transition slowdown still begins around frame 1796. Controller interaction after the transition has not been verified.

Saved the measured executable as `_Build/windows_perf/kyty_emulator_splash60_baseline.exe`. Previous source is in `diagnostics/performance_restore_before/`. The normal executable has been rebuilt from the rollback; the earlier warning about the malformed-PM4 experiment binary no longer applies.

Next performance improvement: driver cache checkpoint interval changed from five minutes to 30 seconds, only after pipeline creation. Earlier captures initialized an empty cache and were stopped/crashed before the first checkpoint. Rebuild and confirm actual cache save/load and steady splash presentation before further behavior changes.

Interpretation correction: the earlier `0x5e6e492` loop traverses linked nodes, checking each node's +0x18 field. Its exact ownership/completion semantics were NOT established. The failed bypass demonstrates an invalid command stream after bypassing it, not proof of a specific missing producer. Do not restore that bypass or the fabricated query/callback completions.

Older sections below are historical and may describe superseded experiments.

## Goal

Help Marvel's Wolverine (`PPSA03671`, game folder `D:\wol`) progress past the post-splash black screen in the custom KytyPS5 build. Keep compatibility changes title-gated and preserve other games such as Plants vs. Zombies.

## Current source/build state

- Current executable: `_Build\windows_perf\kyty_emulator.exe`.
- Latest build completed successfully with `cmake --build _Build/windows_perf --target kyty_emulator` after removing the adverse experiment below.
- `src/loader/runtimeLinker.cpp` has one new exact-site, Wolverine-only patch for the render completion loop at guest address `0x903F65B1D`. The observed counters were `[R15+0xC0]=0` and `[R15+0xE0]=1`; it NOPs only that `jne` when both the exact guest address and byte signature match.
- The speculative high-bit OR in `CommandProcessor::ReleaseMem` was removed. It was not enough to satisfy the full query and corrupted clock payload semantics.
- A second retry-loop bypass at `0x903F0D48F` was tried and removed. It caused zero flips from early startup and moved the Render Thread to another wait at `0x903F848B6`.
- `src/kernel/pthread.cpp` now captures the 32-qword query-result structure and the 32 qwords at its result pointer while Render Thread is inside the captured query-reader loop. It only labels `[R15+0xC0]/[R15+0xE0]` as completion counters inside the known `0x3F65AA0..0x3F65B20` function range.
- `src/graphics/guest_gpu/graphicsRun.cpp` has capped `ReleaseMem(data_sel=3)` address logging for the two observed query windows and prints all seven data dwords for following PM4 `IT_RELEASE_MEM` packets in a GPU stall dump.
- `src/loader/runtimeLinker.cpp` now applies the RenderThread render-target query branch fix (Patch 7) independently of the broader disabled sync-bypass group. Other speculative bypasses (Patches 4–6 and 8) remain disabled. This is still inside the Wolverine title-ID gate.
- Latest build succeeded. Latest test log is `wolverine_render_query_fix.log`; the emulator reached repeated PRESENTED flips before hitting a new RenderThread stall. The currently observed stop is `eboot.bin!0x5e6e492` (latest dump around frame 860). This is not yet a successful game transition.

## Reproduced behavior

With only the exact `0x903F65B1D` startup patch active, repeated runs eventually present the splash/game screen at about 60 FPS through around frame 1765. Around frames 1820–1860, frames slow sharply and then flips stop at 0. This reproduces the user's transition black screen.

A diagnostic run showed that the startup black period is partly shader/pipeline compilation: it later transitions into 60 FPS. The removed `0x903F0D48F` retry-loop experiment caused a separate permanent early black screen, so it is not in the current source/build.

## Evidence at the post-transition black screen

- Latest full capture: `wolverine_release_mem_dump.log`.
- The Render Thread waits in `eboot.bin!0x3f0d44a/44e`, repeatedly calling the timestamp/query getter that returns busy.
- Its result structure at `0x1200005cf0` points to a 32-qword result array at `0x1200005d50`. At the stall, all 32 qwords are zero. An earlier wait used a similar structure at `0x1200006330` pointing to `0x1200006390`; that array was also all zero.
- The bounded `ReleaseMem` trace sees repeated data_sel=3 writes requested at `0x1200005d50` and `0x1200006390`. Data selector 3 replaces the input with the reference clock. These scalar writes do not populate the 32-slot result array captured by the guest reader.
- No `TriggerEvent(0x39)`/occlusion-counter dump was observed in the captures, although that path is implemented in `CommandProcessor::TriggerEvent` and is supposed to publish the interleaved result layout.
- GPU stall dumps show a long `IT_DISPATCH_INDIRECT` in `ProcessPm4`, while the render thread awaits those results. The following packets include `IT_RELEASE_MEM` writes to addresses around `0x1200006de0..0x1200006e20`, not the stalled result array. For example, one packet's target words decode to `0x1200006e20`.
- This points to a GPU producer/compute or PM4 event publication gap before the query slots are written. Do not add more guest wait bypasses until tracing the blocked dispatch and query producer.

## 2026-11-03 update

- Fresh run with the Wolverine timestamp-query busy-loop bypass disabled in `src/loader/runtimeLinker.cpp` restored startup presentation: the capture reached 60 FPS / 60 flips per second and held it through frame 1744. The bypass had been active in the prior build and was jumping past an `SCE_STATUS_BUSY` query result; that is now the leading explanation for the regression where the game ran internally at high draw/dispatch rates but presented zero frames. Keep this bypass disabled.
- At the original scene transition, presentation still falls off: frames 1835–1836 drop from 4 FPS to 0 flips. The GPU queue is then stalled in `IT_DISPATCH_DIRECT` at PM4 address `0x11e1000d7c`, submit `119611`, and the 32 query slots at `0x1200005bc0` are still all zero. Thus disabling the timestamp bypass restores splash/startup presentation but does not fix the post-splash blocker.
- The last pre-stall dispatch is Wolverine guest shader hash `0x4ee8ad3d8dfc9e05`, guest address `0x11d0376e00`, groups `3673x1x1`, mode `0x8041`. Its host Vulkan pipeline `0x9d3bcb2ff41b9cc6` begins compiling just before the queue stall. A previous capture measured roughly 35 seconds for that pipeline; this run had not completed compilation when stopped. A separate SPIR-V inspection measured about 4.3 MB. This is a major pause, but not yet proven to be the only cause of the missing query writes.
- The run used the existing Vulkan pipeline cache, but it reported `initialized empty` and `loaded 0 programs, 0 modules, 0 pipelines`; ordinary shader-cache files do not prevent this host pipeline compile.
- Latest run log: `wolverine_after_timestamp_gate.log`. The emulator was stopped after capturing the post-transition stall.

## 2026-11-03 continued investigation

- With the stale guest callback completion workaround and exact ring completion patch active, the emulator presented hundreds of buffers. The first run eventually stalled on the RenderThread query loop at `eboot.bin!0x3f017fe` and lost the Vulkan device. This matches the old Patch 7 render-target completion loop.
- Patch 7 was then isolated from the other speculative bypasses and applied by its exact signature under the Wolverine title guard. Build succeeded and startup logged `Patch: unblocked RenderThread GPU render-target query spin loop at [0x0000000903f01717]`.
- Test log `wolverine_render_query_fix.log` confirms frequent presents (e.g. 14–24 flips/s around frames 754–846) and dispatch/draw activity. It later stopped presenting near frame 850; the new RenderThread RIP is repeatedly `eboot.bin!0x5e6e492` (nearby snapshots vary by stack/dispatch state). Registers include `RDX=0x1200002238`, `R9=0x1200002220`, and the call stack passes through `eboot.bin!0x3f72606`, `0x3f769bc`, and `0x3f13653`. The renderer remains at one guest frame per second with zero flips; no visual scene confirmation has been made because the user stopped Computer Use.
- Decoding the captured instruction bytes shows the stall is a queue-node wait, not directly the prior render-target query: at `eboot.bin!0x5e6e48e` it loads a node pointer, then `0x5e6e492` reads `[R10+0x18]`, compares it with `1`, and loops until it becomes `1`. At the captured stop, R10 pointed into guest address `0x11e0...`; that node state was not yet captured. Added focused RenderThread diagnostics for the queue head (`RDI`), node (`R10`), and related object pointers (`RDX`, `R9`) in `src/kernel/pthread.cpp`.
- The queue-node diagnostic run (`wolverine_queue_node_diag.log`) shows multiple linked nodes at `0x11e0...` with qword `+0x18` equal to zero, while their other payload fields are populated. Bypassing this test at the exact call site caused `packet_dw > remaining_dw` in the PM4 parser before rendering resumed, proving the consumer must not pop the node until its producer completes it. This experimental bypass has been removed.
- Current diagnostic extension logs the queue owner and guest caller bytes around the RenderThread stack’s EBOOT return addresses. Next: rebuild and inspect that call path to identify which producer should set node `+0x18`, then repair the producer/completion behavior. Keep broad sync bypasses disabled.
- User then reported the splash screen is no longer steady, starts very slowly, and controller interaction is unavailable after the transition. Work was stopped at the user's request. The latest queue-node bypass trial (`wolverine_queue_node_bypass_trial.log`) failed with a PM4 packet-size fatal error and is unsafe; do not run it. The previous diagnostic build showed repeated presents before the transition stall, but did not prove a visible playable scene.
- The most recent source edit only extends diagnostics for the queue owner/caller and was not rebuilt after the user asked to stop. The current `_Build/windows_perf/kyty_emulator.exe` may therefore still be the prior experiment binary; don't assume it matches the current source.

## Next steps

1. Keep timestamp-query and broad sync bypasses disabled.
2. Continue from the new `0x5e6e492` RenderThread stop and trace its result object/wait condition.
3. Re-test through a visible post-splash scene, then verify a lightweight title remains unaffected.

## Run command

```powershell
.\_Build\windows_perf\kyty_emulator.exe --game "D:\wol" --perf-mode --shader-cache --perf-overlay
```

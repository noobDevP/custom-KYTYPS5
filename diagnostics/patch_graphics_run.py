with open('src/graphics/guest_gpu/graphicsRun.cpp', 'r', encoding='utf-8') as f:
    content = f.read().replace('\r\n', '\n')

# 1. Include compilationTracker.h
inc_target = '#include "graphics/host_gpu/renderer/sync.h"'
new_inc = '''#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/pipeline/compilationTracker.h"'''
assert inc_target in content, 'inc_target not found'
content = content.replace(inc_target, new_inc, 1)

# 2. In NoteSuspendWait: remove wait skip and add ongoing compilation tracking
t_wait = '''		// Title-scoped unblock for Marvel's Wolverine (PPSA03671):
		// If a wait packet has retried repeatedly (>= 16 attempts or >= 20ms) and cannot clear,
		// skip the wait so the GPU command processor seeks to the next packet instead of deadlocking.
		static const bool is_wolverine = [] {
			std::string title_id;
			return Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) &&
			       title_id == "PPSA03671";
		}();
		if (is_wolverine && entry.first_ms != 0 && (now_ms - entry.first_ms >= 1500)) {
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "GPU wait skip: unblocked {} at addr=0x{:016x} value=0x{:016x} (attempts={})\\n",
				    kind, address, value, entry.attempts));
			}
			if (func == 3u && ref == 0 && address != 0) {
				*reinterpret_cast<volatile uint64_t*>(address) = 0;
			}
			entry.reported_blocked = false;
			entry.first_ms         = 0;
			entry.attempts         = 0;
			return false; // Do NOT suspend; advance cursor and seek next packet!
		}

		// Only report blocked if it has been stuck for at least 3 seconds
		if (!entry.reported_blocked && (now_ms - entry.first_ms >= 3000)) {
			entry.reported_blocked = true;
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU suspend: {} blocked addr=0x{:016x} value=0x{:016x} ref=0x{:016x} "\n			    "mask=0x{:016x} func={} attempts={}\\n",
			    kind, address, value, ref, mask, func, entry.attempts));
		} else if (entry.reported_blocked && (now_ms - entry.last_report_ms >= 5000)) {
			entry.last_report_ms = now_ms;
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU suspend: {} STILL blocked after {}s addr=0x{:016x} value=0x{:016x} "\n			    "attempts={}\\n",
			    kind, (now_ms - entry.first_ms) / 1000, address, value, entry.attempts));
		}'''.replace('\r\n', '\n')

r_wait = '''		// Only report blocked if it has been stuck for at least 3 seconds
		if (!entry.reported_blocked && (now_ms - entry.first_ms >= 3000)) {
			entry.reported_blocked = true;
			if (CompilationTracker::HasActive()) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "GPU suspend: {} waiting addr=0x{:016x} value=0x{:016x} (ongoing compilation: {})\n",
				    kind, address, value, CompilationTracker::GetActiveSummary()));
			} else {
				Log::WriteToConsoleAndLog(fmt::format(
				    "GPU suspend: {} blocked addr=0x{:016x} value=0x{:016x} ref=0x{:016x} "
				    "mask=0x{:016x} func={} attempts={} (no active compilations)\n",
				    kind, address, value, ref, mask, func, entry.attempts));
			}
		} else if (entry.reported_blocked && (now_ms - entry.last_report_ms >= 5000)) {
			entry.last_report_ms = now_ms;
			if (CompilationTracker::HasActive()) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "GPU suspend: {} waiting after {}s addr=0x{:016x} (ongoing compilation: {})\n",
				    kind, (now_ms - entry.first_ms) / 1000, address, CompilationTracker::GetActiveSummary()));
			} else {
				Log::WriteToConsoleAndLog(fmt::format(
				    "GPU suspend: {} STILL blocked after {}s addr=0x{:016x} value=0x{:016x} "
				    "attempts={} (no active compilations)\n",
				    kind, (now_ms - entry.first_ms) / 1000, address, value, entry.attempts));
			}
		}'''.replace('\r\n', '\n')
assert t_wait in content, 't_wait not found'
content = content.replace(t_wait, r_wait, 1)

# 3. In GuestGpu::DumpState(): replace raw reinterpret_cast with TryReadBacking
t_pm4 = '''			if (packet_addr != 0 && packet_dw != 0 && cursor_dw + packet_dw <= total_dw) {
				const auto* words = reinterpret_cast<const uint32_t*>(packet_addr);
				uint32_t    offset = packet_dw;
				Log::WriteToConsoleAndLog("  PM4 packets after blocked dispatch (up to 24):\\n");
				for (uint32_t count = 0; count < 24 && cursor_dw + offset < total_dw; count++) {
					const auto header = words[offset];'''.replace('\r\n', '\n')

r_pm4 = '''			if (packet_addr != 0 && packet_dw != 0 && cursor_dw + packet_dw <= total_dw) {
				const uint32_t words_to_read = std::min(total_dw - cursor_dw, packet_dw + 24u * 16u);
				std::vector<uint32_t> words(words_to_read);
				if (Libs::LibKernel::Memory::TryReadBacking(packet_addr, words.data(), words.size() * sizeof(uint32_t))) {
					uint32_t offset = packet_dw;
					Log::WriteToConsoleAndLog("  PM4 packets after blocked dispatch (up to 24):\\n");
					for (uint32_t count = 0; count < 24 && cursor_dw + offset < total_dw && offset < words.size(); count++) {
						const auto header = words[offset];'''.replace('\r\n', '\n')
assert t_pm4 in content, 't_pm4 not found'
content = content.replace(t_pm4, r_pm4, 1)

# Close the if (TryReadBacking) block
t_pm4_end = '''					offset += size;
				}
			}
		}'''.replace('\r\n', '\n')
r_pm4_end = '''					offset += size;
					}
				}
			}
		}'''.replace('\r\n', '\n')
assert t_pm4_end in content, 't_pm4_end not found'
content = content.replace(t_pm4_end, r_pm4_end, 1)

# 4. Multi-draw checked read
t_multi = '''			if (packet_addr != 0 && packet_dw == 10 && cursor_dw + packet_dw <= total_dw) {
				const auto* words = reinterpret_cast<const uint32_t*>(packet_addr);
				const auto  count_indirect = (words[4] >> 30u) & 1u;
				const auto  count_addr = static_cast<uint64_t>(words[6]) |
				                         (static_cast<uint64_t>(words[7]) << 32u);
				Log::WriteToConsoleAndLog(fmt::format(
				    "  Current multi-draw: data_offset=0x{:08x} regs=0x{:04x}/0x{:04x} "
				    "count_indirect={} max_count_or_count={} count_addr=0x{:016x} "
				    "stride={} initiator=0x{:08x}\\n",
				    words[1], words[2] & 0xffffu, words[3] & 0xffffu, count_indirect,
				    words[5], count_addr, words[8], words[9]));
			}'''.replace('\r\n', '\n')

r_multi = '''			if (packet_addr != 0 && packet_dw == 10 && cursor_dw + packet_dw <= total_dw) {
				std::array<uint32_t, 10> words {};
				if (Libs::LibKernel::Memory::TryReadBacking(packet_addr, words.data(), sizeof(words))) {
					const auto  count_indirect = (words[4] >> 30u) & 1u;
					const auto  count_addr = static_cast<uint64_t>(words[6]) |
					                         (static_cast<uint64_t>(words[7]) << 32u);
					Log::WriteToConsoleAndLog(fmt::format(
					    "  Current multi-draw: data_offset=0x{:08x} regs=0x{:04x}/0x{:04x} "
					    "count_indirect={} max_count_or_count={} count_addr=0x{:016x} "
					    "stride={} initiator=0x{:08x}\\n",
					    words[1], words[2] & 0xffffu, words[3] & 0xffffu, count_indirect,
					    words[5], count_addr, words[8], words[9]));
				}
			}'''.replace('\r\n', '\n')
assert t_multi in content, 't_multi not found'
content = content.replace(t_multi, r_multi, 1)

with open('src/graphics/guest_gpu/graphicsRun.cpp', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print('Updated graphicsRun.cpp successfully!')

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <fmt/format.h>

namespace Libs::Graphics {

struct ActiveCompilationInfo {
	uint64_t                              token = 0;
	std::string                           kind;        // "Shader", "ComputePipeline", "GraphicsPipeline"
	uint64_t                              shader_hash = 0;
	uint64_t                              module_id = 0;
	std::string                           stage;       // "CS", "VS", "PS", etc.
	std::chrono::steady_clock::time_point start_time;
};

class CompilationTracker {
public:
	static uint64_t Begin(const char* kind, uint64_t shader_hash, uint64_t module_id, const char* stage) {
		std::lock_guard lock(s_mutex);
		static uint64_t s_next_token = 1;
		const uint64_t token = s_next_token++;
		s_active.push_back(ActiveCompilationInfo {
			.token = token,
			.kind = kind ? kind : "Unknown",
			.shader_hash = shader_hash,
			.module_id = module_id,
			.stage = stage ? stage : "",
			.start_time = std::chrono::steady_clock::now()
		});
		s_active_count.store(static_cast<uint32_t>(s_active.size()), std::memory_order_release);
		return token;
	}

	static void End(uint64_t token) {
		std::lock_guard lock(s_mutex);
		for (auto it = s_active.begin(); it != s_active.end(); ++it) {
			if (it->token == token) {
				s_active.erase(it);
				break;
			}
		}
		s_active_count.store(static_cast<uint32_t>(s_active.size()), std::memory_order_release);
	}

	static uint32_t GetActiveCount() noexcept {
		return s_active_count.load(std::memory_order_acquire);
	}

	static bool HasActive() noexcept {
		return GetActiveCount() > 0;
	}

	static std::string GetActiveSummary() {
		std::lock_guard lock(s_mutex);
		if (s_active.empty()) {
			return "none";
		}
		const auto now = std::chrono::steady_clock::now();
		std::string result;
		for (size_t i = 0; i < s_active.size(); ++i) {
			const auto& item = s_active[i];
			const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - item.start_time).count();
			if (i > 0) result += ", ";
			result += fmt::format("{}[{}: shader=0x{:016x} module=0x{:016x} elapsed={}ms]",
			                      item.kind, item.stage, item.shader_hash, item.module_id, elapsed_ms);
		}
		return result;
	}

	static std::vector<ActiveCompilationInfo> GetActiveSnapshot() {
		std::lock_guard lock(s_mutex);
		return s_active;
	}

private:
	inline static std::mutex                         s_mutex;
	inline static std::vector<ActiveCompilationInfo> s_active;
	inline static std::atomic<uint32_t>              s_active_count {0};
};

struct ScopedCompilationTrack {
	uint64_t token = 0;
	ScopedCompilationTrack(const char* kind, uint64_t shader_hash, uint64_t module_id, const char* stage) {
		token = CompilationTracker::Begin(kind, shader_hash, module_id, stage);
	}
	~ScopedCompilationTrack() {
		if (token != 0) {
			CompilationTracker::End(token);
		}
	}
	ScopedCompilationTrack(const ScopedCompilationTrack&) = delete;
	ScopedCompilationTrack& operator=(const ScopedCompilationTrack&) = delete;
};

} // namespace Libs::Graphics

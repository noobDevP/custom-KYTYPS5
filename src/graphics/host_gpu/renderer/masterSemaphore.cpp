#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/compilationTracker.h"

#include <fmt/format.h>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);

	// Waiting forever hides a submission that never completes, which reaches the user as a frozen
	// window with no output. Wait in slices and report the stall so the cause is visible.
	constexpr uint64_t wait_slice_ns   = 5'000'000'000ull;
	constexpr uint64_t report_every_ns = 20'000'000'000ull;
	uint64_t           waited_ns       = 0;
	uint64_t           next_report_ns  = report_every_ns;

	auto result = vk::Result::eTimeout;
	while (result == vk::Result::eTimeout) {
		result = m_graphics.device.waitSemaphores(&wait_info, wait_slice_ns);
		if (result != vk::Result::eTimeout) {
			break;
		}
		waited_ns += wait_slice_ns;
		if (waited_ns >= next_report_ns) {
			next_report_ns += report_every_ns;
			Refresh();
			if (CompilationTracker::HasActive()) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "GPU wait: waiting for timeline tick {} but GPU is at {} after {} s (ongoing compilation: {})\n",
				    tick, KnownGpuTick(), waited_ns / 1'000'000'000ull, CompilationTracker::GetActiveSummary()));
			} else {
				Log::WriteToConsoleAndLog(fmt::format(
				    "GPU stall: waiting for timeline tick {} but the GPU is still at {} after {} s (no active compilations)\n",
				    tick, KnownGpuTick(), waited_ns / 1'000'000'000ull));
			}
		}
	}
	if (result != vk::Result::eSuccess) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "waitSemaphores FAILED with: {} (tick={}, known={})\n", vk::to_string(result), tick, KnownGpuTick()));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics

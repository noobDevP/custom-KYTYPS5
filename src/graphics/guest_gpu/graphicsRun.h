#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;
	void              DumpState();

	[[nodiscard]] static bool IsGpuThread() noexcept;

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Graphics, Compute, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		Pm4Execution              execution;
		bool                      started         = false;
		bool                      blocked         = false;
		uint64_t                  flip_request_id = 0;
	};

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	std::jthread    m_thread;

	std::atomic<uint64_t>    m_thread_os_id {0};
	std::atomic<const char*> m_current_step {"Idle"};
	std::atomic<uint32_t>    m_current_submission_type {0};
	std::atomic<uint32_t>    m_current_queue_id {0};
	std::atomic<uint64_t>    m_current_submit_id {0};
	std::atomic<uint32_t>    m_current_pm4_opcode {0};
	std::atomic<uint64_t>    m_current_pm4_packet_addr {0};
	std::atomic<uint32_t>    m_current_pm4_packet_header {0};
	std::atomic<uint32_t>    m_current_pm4_packet_dw {0};
	std::atomic<uint32_t>    m_current_pm4_cursor_dw {0};
	std::atomic<uint32_t>    m_current_pm4_total_dw {0};
	std::atomic<uint64_t>    m_pm4_packets_processed {0};
	std::atomic<uint64_t>    m_process_start_time_ms {0};
	std::atomic<bool>        m_current_dispatch_indirect {false};
	std::atomic<uint64_t>    m_current_dispatch_submit_id {0};
	std::atomic<uint64_t>    m_current_dispatch_args_addr {0};
	std::atomic<uint32_t>    m_current_dispatch_mode {0};
	std::atomic<uint64_t>    m_current_dispatch_cs_addr {0};

	friend class CommandProcessor;
};

void GuestGpuDumpState();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */

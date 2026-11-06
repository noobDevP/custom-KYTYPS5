#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDPOOL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDPOOL_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <vector>

namespace Libs::Graphics {

// Thread-local command-pool wrapper for secondary command buffers.
// Each thread lazily creates its own vk::CommandPool; Acquire() returns a secondary
// command buffer that can be recorded independently and submitted via
// vk::CommandBuffer::executeCommands.
class CommandPool {
public:
	CommandPool() = default;
	explicit CommandPool(vk::Device device, uint32_t queue_family_index);
	explicit CommandPool(VkDevice device, uint32_t queue_family_index);
	~CommandPool();

	KYTY_CLASS_NO_COPY(CommandPool);

	void Initialize(vk::Device device, uint32_t queue_family_index);

	// Acquire a secondary command buffer from the calling thread's pool.
	[[nodiscard]] vk::CommandBuffer Acquire();
	[[nodiscard]] VkCommandBuffer   AcquireRaw() { return static_cast<VkCommandBuffer>(Acquire()); }

	// Return a secondary command buffer to the calling thread's free list.
	void Release(vk::CommandBuffer cmd);
	void Release(VkCommandBuffer cmd) { Release(vk::CommandBuffer(cmd)); }

	// Reset the calling thread's command pool and recycle all allocated buffers.
	void ResetThreadPool();

	// Record N passes in parallel across Common::GetThreadPool() into secondary
	// command buffers and execute them on primary_cmd via executeCommands.
	void RecordParallelPasses(
	    vk::CommandBuffer                                       primary_cmd,
	    const vk::CommandBufferInheritanceInfo&                 inheritance_info,
	    std::span<const std::function<void(vk::CommandBuffer)>> pass_recorders,
	    vk::CommandBufferUsageFlags                             usage_flags =
	        vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

private:
	struct ThreadLocalState {
		vk::Device                     device             = nullptr;
		uint32_t                       queue_family_index = 0;
		vk::CommandPool                pool               = nullptr;
		std::vector<vk::CommandBuffer> free_cmds;
		std::vector<vk::CommandBuffer> all_cmds;
	};

	static ThreadLocalState& GetThreadState();
	vk::CommandPool          EnsureThreadPool();

	vk::Device                   m_device             = nullptr;
	uint32_t                     m_queue_family_index = 0;
	std::mutex                   m_mutex;
	std::vector<vk::CommandPool> m_created_pools;
};

} // namespace Libs::Graphics

using CommandPool = Libs::Graphics::CommandPool;

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDPOOL_H_

#include "graphics/host_gpu/renderer/CommandPool.h"

#include "common/assert.h"
#include "common/threadPool.h"

#include <atomic>
#include <condition_variable>

namespace Libs::Graphics {

CommandPool::CommandPool(vk::Device device, uint32_t queue_family_index)
    : m_device(device), m_queue_family_index(queue_family_index) {}

CommandPool::CommandPool(VkDevice device, uint32_t queue_family_index)
    : m_device(device), m_queue_family_index(queue_family_index) {}

CommandPool::~CommandPool() {
	std::scoped_lock lk(m_mutex);
	if (m_device) {
		for (auto pool: m_created_pools) {
			if (pool) {
				m_device.destroyCommandPool(pool, nullptr);
			}
		}
	}
	m_created_pools.clear();
	auto& tls = GetThreadState();
	if (tls.device == m_device) {
		tls.pool = nullptr;
		tls.free_cmds.clear();
		tls.all_cmds.clear();
	}
}

void CommandPool::Initialize(vk::Device device, uint32_t queue_family_index) {
	std::scoped_lock lk(m_mutex);
	m_device             = device;
	m_queue_family_index = queue_family_index;
}

CommandPool::ThreadLocalState& CommandPool::GetThreadState() {
	thread_local ThreadLocalState state;
	return state;
}

vk::CommandPool CommandPool::EnsureThreadPool() {
	auto& tls = GetThreadState();
	if (tls.pool && tls.device == m_device && tls.queue_family_index == m_queue_family_index) {
		return tls.pool;
	}

	vk::CommandPoolCreateInfo info {};
	info.queueFamilyIndex = m_queue_family_index;
	info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
	             vk::CommandPoolCreateFlagBits::eTransient;

	vk::CommandPool pool = nullptr;
	{
		std::scoped_lock lk(m_mutex);
		const vk::Result res = m_device.createCommandPool(&info, nullptr, &pool);
		EXIT_NOT_IMPLEMENTED(res != vk::Result::eSuccess);
		m_created_pools.push_back(pool);
	}

	tls.device             = m_device;
	tls.queue_family_index = m_queue_family_index;
	tls.pool               = pool;
	tls.free_cmds.clear();
	tls.all_cmds.clear();
	return pool;
}

vk::CommandBuffer CommandPool::Acquire() {
	EXIT_IF(!m_device);
	vk::CommandPool pool = EnsureThreadPool();
	auto&           tls  = GetThreadState();

	if (!tls.free_cmds.empty()) {
		vk::CommandBuffer cmd = tls.free_cmds.back();
		tls.free_cmds.pop_back();
		(void)cmd.reset({});
		return cmd;
	}

	vk::CommandBufferAllocateInfo alloc {};
	alloc.commandPool        = pool;
	alloc.level              = vk::CommandBufferLevel::eSecondary;
	alloc.commandBufferCount = 1;

	vk::CommandBuffer cmd = nullptr;
	const vk::Result  res = m_device.allocateCommandBuffers(&alloc, &cmd);
	EXIT_NOT_IMPLEMENTED(res != vk::Result::eSuccess);
	tls.all_cmds.push_back(cmd);
	return cmd;
}

void CommandPool::Release(vk::CommandBuffer cmd) {
	if (!cmd) return;
	auto& tls = GetThreadState();
	tls.free_cmds.push_back(cmd);
}

void CommandPool::ResetThreadPool() {
	auto& tls = GetThreadState();
	if (!tls.pool || !tls.device) return;
	(void)tls.device.resetCommandPool(tls.pool, {});
	tls.free_cmds = tls.all_cmds;
}

void CommandPool::RecordParallelPasses(
    vk::CommandBuffer                                       primary_cmd,
    const vk::CommandBufferInheritanceInfo&                 inheritance_info,
    std::span<const std::function<void(vk::CommandBuffer)>> pass_recorders,
    vk::CommandBufferUsageFlags                             usage_flags) {
	if (pass_recorders.empty()) return;

	const std::size_t              count = pass_recorders.size();
	std::vector<vk::CommandBuffer> secondary_cmds(count);

	if (count == 1) {
		vk::CommandBuffer          sec = Acquire();
		vk::CommandBufferBeginInfo begin_info {};
		begin_info.flags            = usage_flags;
		begin_info.pInheritanceInfo = &inheritance_info;
		EXIT_NOT_IMPLEMENTED(sec.begin(&begin_info) != vk::Result::eSuccess);
		pass_recorders[0](sec);
		EXIT_NOT_IMPLEMENTED(sec.end() != vk::Result::eSuccess);
		primary_cmd.executeCommands(1, &sec);
		return;
	}

	std::atomic<std::size_t> remaining {count};
	std::mutex               done_mtx;
	std::condition_variable  done_cv;

	for (std::size_t i = 0; i < count; ++i) {
		Common::GetThreadPool().Enqueue(
		    [this, i, &pass_recorders, &secondary_cmds, &inheritance_info, usage_flags,
		     &remaining, &done_mtx, &done_cv]() {
			    vk::CommandBuffer          sec = Acquire();
			    vk::CommandBufferBeginInfo begin_info {};
			    begin_info.flags            = usage_flags;
			    begin_info.pInheritanceInfo = &inheritance_info;
			    EXIT_NOT_IMPLEMENTED(sec.begin(&begin_info) != vk::Result::eSuccess);
			    pass_recorders[i](sec);
			    EXIT_NOT_IMPLEMENTED(sec.end() != vk::Result::eSuccess);
			    secondary_cmds[i] = sec;

			    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
				    std::lock_guard lk(done_mtx);
				    done_cv.notify_one();
			    }
		    });
	}

	{
		std::unique_lock lk(done_mtx);
		done_cv.wait(lk, [&] { return remaining.load(std::memory_order_acquire) == 0; });
	}

	primary_cmd.executeCommands(static_cast<uint32_t>(secondary_cmds.size()),
	                            secondary_cmds.data());
}

} // namespace Libs::Graphics

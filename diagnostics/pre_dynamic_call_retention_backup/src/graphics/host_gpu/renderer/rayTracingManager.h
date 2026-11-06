#pragma once

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"

#include <unordered_map>
#include <vector>
#include <array>
#include <mutex>

namespace Libs::Graphics {

struct AccelerationStructureEntry {
	vk::AccelerationStructureKHR handle       = nullptr;
	vk::Buffer                   buffer       = nullptr;
	VmaAllocation                allocation   = nullptr;
	uint64_t                     device_addr  = 0;
	uint32_t                     triangle_count = 0;
};

class RayTracingManager {
public:
	static RayTracingManager& Get();

	void Init(GraphicContext* ctx);

	[[nodiscard]] bool IsEnabled() const {
		return m_ctx != nullptr && m_ctx->acceleration_structure_enabled;
	}

	vk::AccelerationStructureKHR GetOrBuildBlas(uint64_t bvh_base, uint32_t num_records);
	vk::AccelerationStructureKHR GetOrBuildTlas(uint64_t tlas_base, uint32_t num_instances);

private:
	GraphicContext* m_ctx = nullptr;
	std::mutex      m_mutex;
	std::unordered_map<uint64_t, AccelerationStructureEntry> m_blas_cache;
	std::unordered_map<uint64_t, AccelerationStructureEntry> m_tlas_cache;

	bool DecodeBvhTriangles(uint64_t bvh_base, uint32_t num_records,
	                        std::vector<float>& out_vertices,
	                        std::vector<uint32_t>& out_indices);
};

} // namespace Libs::Graphics

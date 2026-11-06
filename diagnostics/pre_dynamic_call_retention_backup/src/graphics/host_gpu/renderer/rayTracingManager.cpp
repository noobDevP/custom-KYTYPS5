#include "graphics/host_gpu/renderer/rayTracingManager.h"
#include "kernel/memory.h"
#include "common/logging/log.h"

#include <queue>
#include <unordered_set>
#include <cstring>

namespace Libs::Graphics {

RayTracingManager& RayTracingManager::Get() {
	static RayTracingManager s_instance;
	return s_instance;
}

void RayTracingManager::Init(GraphicContext* ctx) {
	m_ctx = ctx;
	if (m_ctx != nullptr && m_ctx->acceleration_structure_enabled) {
		LOGF("RayTracingManager: initialized with Vulkan KHR acceleration structure support\n");
	}
}

bool RayTracingManager::DecodeBvhTriangles(uint64_t bvh_base, uint32_t num_records,
                                          std::vector<float>& out_vertices,
                                          std::vector<uint32_t>& out_indices) {
	if (bvh_base == 0) return false;

	std::queue<uint64_t> node_queue;
	std::unordered_set<uint64_t> visited_nodes;

	// Root node at index 0
	node_queue.push(0);

	const uint32_t max_nodes_to_traverse = std::min(num_records > 0 ? num_records : 65536u, 262144u);
	uint32_t nodes_processed = 0;

	while (!node_queue.empty() && nodes_processed < max_nodes_to_traverse) {
		const uint64_t node_desc = node_queue.front();
		node_queue.pop();

		const uint32_t kind = static_cast<uint32_t>(node_desc & 7u);
		const uint64_t node_index = node_desc >> 3u;
		const uint64_t node_addr = bvh_base + (node_index << 6u);

		if (!visited_nodes.insert(node_addr).second) {
			continue;
		}
		++nodes_processed;

		std::array<uint32_t, 16> node_words {};
		if (!Libs::LibKernel::Memory::TryReadBufferBacking(node_addr, node_words.data(), sizeof(node_words))) {
			continue;
		}

		if (kind < 4u) {
			// Triangle node (4 triangles formed by 5 vertices)
			// Vertices 0..4, each 3 floats: x, y, z
			const uint32_t base_index = static_cast<uint32_t>(out_vertices.size() / 3u);

			for (uint32_t v = 0; v < 5; ++v) {
				for (uint32_t axis = 0; axis < 3; ++axis) {
					const uint32_t word_idx = v * 3u + axis;
					if (word_idx < 15u) {
						float val = 0.0f;
						std::memcpy(&val, &node_words[word_idx], sizeof(float));
						out_vertices.push_back(val);
					} else {
						out_vertices.push_back(0.0f);
					}
				}
			}

			// Triangles: (0,1,2), (1,3,2), (2,3,4), (2,4,0)
			constexpr uint32_t tri_indices[4][3] = {
				{0, 1, 2},
				{1, 3, 2},
				{2, 3, 4},
				{2, 4, 0}
			};

			for (const auto& tri : tri_indices) {
				out_indices.push_back(base_index + tri[0]);
				out_indices.push_back(base_index + tri[1]);
				out_indices.push_back(base_index + tri[2]);
			}
		} else {
			// Box node (fp16 half box or fp32 full box)
			// Words 0..3 contain child node references
			for (uint32_t child = 0; child < 4; ++child) {
				const uint32_t child_desc = node_words[child];
				if (child_desc != 0 && child_desc != ~0u) {
					node_queue.push(static_cast<uint64_t>(child_desc));
				}
			}
		}
	}

	return !out_vertices.empty() && !out_indices.empty();
}

vk::AccelerationStructureKHR RayTracingManager::GetOrBuildBlas(uint64_t bvh_base, uint32_t num_records) {
	if (!IsEnabled() || bvh_base == 0) return nullptr;

	std::scoped_lock lock(m_mutex);
	auto it = m_blas_cache.find(bvh_base);
	if (it != m_blas_cache.end()) {
		return it->second.handle;
	}

	std::vector<float> vertices;
	std::vector<uint32_t> indices;
	if (!DecodeBvhTriangles(bvh_base, num_records, vertices, indices)) {
		return nullptr;
	}

	const uint32_t triangle_count = static_cast<uint32_t>(indices.size() / 3u);
	const uint32_t vertex_count = static_cast<uint32_t>(vertices.size() / 3u);

	LOGF("RayTracingManager: decoded PS5 BVH at 0x%016" PRIx64 ": %u vertices, %u triangles\n",
	     bvh_base, vertex_count, triangle_count);

	// Create vertex and index staging/device buffers
	const size_t vertex_bytes = vertices.size() * sizeof(float);
	const size_t index_bytes = indices.size() * sizeof(uint32_t);

	vk::BufferCreateInfo vtx_info {};
	vtx_info.size  = vertex_bytes;
	vtx_info.usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
	                 vk::BufferUsageFlagBits::eShaderDeviceAddress;

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	VkBuffer vtx_buf_raw = VK_NULL_HANDLE;
	VmaAllocation vtx_alloc = nullptr;
	VmaAllocationInfo vtx_res {};
	if (vmaCreateBuffer(m_ctx->allocator, reinterpret_cast<const VkBufferCreateInfo*>(&vtx_info),
	                    &alloc_info, &vtx_buf_raw, &vtx_alloc, &vtx_res) != VK_SUCCESS) {
		return nullptr;
	}
	std::memcpy(vtx_res.pMappedData, vertices.data(), vertex_bytes);

	vk::BufferCreateInfo idx_info {};
	idx_info.size  = index_bytes;
	idx_info.usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
	                 vk::BufferUsageFlagBits::eShaderDeviceAddress;

	VkBuffer idx_buf_raw = VK_NULL_HANDLE;
	VmaAllocation idx_alloc = nullptr;
	VmaAllocationInfo idx_res {};
	if (vmaCreateBuffer(m_ctx->allocator, reinterpret_cast<const VkBufferCreateInfo*>(&idx_info),
	                    &alloc_info, &idx_buf_raw, &idx_alloc, &idx_res) != VK_SUCCESS) {
		vmaDestroyBuffer(m_ctx->allocator, vtx_buf_raw, vtx_alloc);
		return nullptr;
	}
	std::memcpy(idx_res.pMappedData, indices.data(), index_bytes);

	vk::BufferDeviceAddressInfo vtx_bda_info {};
	vtx_bda_info.buffer = vk::Buffer(vtx_buf_raw);
	const auto vtx_addr = m_ctx->device.getBufferAddress(&vtx_bda_info);

	vk::BufferDeviceAddressInfo idx_bda_info {};
	idx_bda_info.buffer = vk::Buffer(idx_buf_raw);
	const auto idx_addr = m_ctx->device.getBufferAddress(&idx_bda_info);

	// Describe geometry
	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
	geometry.flags        = vk::GeometryFlagBitsKHR::eOpaque;
	geometry.geometry.triangles.vertexFormat                 = vk::Format::eR32G32B32Sfloat;
	geometry.geometry.triangles.vertexData.deviceAddress     = vtx_addr;
	geometry.geometry.triangles.vertexStride                 = 3u * sizeof(float);
	geometry.geometry.triangles.maxVertex                    = vertex_count;
	geometry.geometry.triangles.indexType                    = vk::IndexType::eUint32;
	geometry.geometry.triangles.indexData.deviceAddress      = idx_addr;

	vk::AccelerationStructureBuildGeometryInfoKHR build_info {};
	build_info.type          = vk::AccelerationStructureTypeKHR::eBottomLevel;
	build_info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	build_info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	build_info.geometryCount = 1;
	build_info.pGeometries   = &geometry;

	vk::AccelerationStructureBuildSizesInfoKHR size_info {};
	m_ctx->device.getAccelerationStructureBuildSizesKHR(
	    vk::AccelerationStructureBuildTypeKHR::eDevice, &build_info, &triangle_count, &size_info);

	// Allocate buffer for acceleration structure
	vk::BufferCreateInfo as_buf_info {};
	as_buf_info.size  = size_info.accelerationStructureSize;
	as_buf_info.usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
	                    vk::BufferUsageFlagBits::eShaderDeviceAddress;

	VmaAllocationCreateInfo as_alloc_info {};
	as_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VkBuffer as_buf_raw = VK_NULL_HANDLE;
	VmaAllocation as_alloc = nullptr;
	if (vmaCreateBuffer(m_ctx->allocator, reinterpret_cast<const VkBufferCreateInfo*>(&as_buf_info),
	                    &as_alloc_info, &as_buf_raw, &as_alloc, nullptr) != VK_SUCCESS) {
		vmaDestroyBuffer(m_ctx->allocator, vtx_buf_raw, vtx_alloc);
		vmaDestroyBuffer(m_ctx->allocator, idx_buf_raw, idx_alloc);
		return nullptr;
	}

	vk::AccelerationStructureCreateInfoKHR as_create_info {};
	as_create_info.buffer = vk::Buffer(as_buf_raw);
	as_create_info.size   = size_info.accelerationStructureSize;
	as_create_info.type   = vk::AccelerationStructureTypeKHR::eBottomLevel;

	vk::AccelerationStructureKHR as_handle = nullptr;
	if (m_ctx->device.createAccelerationStructureKHR(&as_create_info, nullptr, &as_handle) != vk::Result::eSuccess) {
		vmaDestroyBuffer(m_ctx->allocator, as_buf_raw, as_alloc);
		vmaDestroyBuffer(m_ctx->allocator, vtx_buf_raw, vtx_alloc);
		vmaDestroyBuffer(m_ctx->allocator, idx_buf_raw, idx_alloc);
		return nullptr;
	}

	AccelerationStructureEntry entry {};
	entry.handle         = as_handle;
	entry.buffer         = vk::Buffer(as_buf_raw);
	entry.allocation     = as_alloc;
	entry.triangle_count = triangle_count;

	vk::AccelerationStructureDeviceAddressInfoKHR as_addr_info {};
	as_addr_info.accelerationStructure = as_handle;
	entry.device_addr = m_ctx->device.getAccelerationStructureAddressKHR(&as_addr_info);

	m_blas_cache.emplace(bvh_base, entry);

	LOGF("RayTracingManager: successfully created Vulkan BLAS handle=0x%016" PRIx64 " for guest BVH 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(static_cast<VkAccelerationStructureKHR>(as_handle)), bvh_base);

	return as_handle;
}

vk::AccelerationStructureKHR RayTracingManager::GetOrBuildTlas(uint64_t tlas_base, uint32_t num_instances) {
	if (!IsEnabled() || tlas_base == 0) return nullptr;

	std::scoped_lock lock(m_mutex);
	auto it = m_tlas_cache.find(tlas_base);
	if (it != m_tlas_cache.end()) {
		return it->second.handle;
	}

	return nullptr;
}

} // namespace Libs::Graphics

#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/threadPool.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/translate/Translate.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fmt/format.h>
#include <fstream>
#include <functional>
#include <map>
#include <span>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics::ShaderRecompiler {

std::unordered_map<std::size_t, std::shared_ptr<CompiledShader>> gShaderCache;
std::shared_mutex                                                gCacheMtx;

namespace {

std::string MakeIrDump(std::string_view cfg, const IR::Program& ir) {
	std::string dump = "CFG:\n";
	dump += cfg;
	dump += "\nIR:\n";
	dump += fmt::format("mode={} scratch_dwords={}\n",
	                    ir.dispatcher_fallback ? "dispatcher" : "structured", ir.scratch_dwords);
	dump += IR::ProgramToString(ir);
	return dump;
}

void LogDispatcherFallback(const CompileOptions& options, const CFG::Graph& cfg, const char* phase) {
	const auto* block        = cfg.FindBlock(cfg.failure_block);
	const auto  start        = block != nullptr ? block->start_pc : UINT32_MAX;
	const auto  end          = block != nullptr ? block->end_pc : UINT32_MAX;
	const auto  predecessors = block != nullptr ? block->predecessors.size() : 0u;
	const auto  successors   = block != nullptr ? block->successors.size() : 0u;
	LOGF("ShaderRecompiler %s CFG dispatcher fallback: hash=0x%016" PRIx64
	     " phase=%s failure=%s block=%" PRIu32 " pc=0x%08" PRIx32 "..0x%08" PRIx32 " preds=%" PRIu64
	     " succs=%" PRIu64 " blocks=%" PRIu64 " loops=%" PRIu64 " back_edges=%" PRIu64
	     " reason=%s\n",
	     StageName(options.stage), options.shader_hash, phase,
	     CFG::FailureKindToString(cfg.failure_kind).c_str(), cfg.failure_block, start, end,
	     static_cast<uint64_t>(predecessors), static_cast<uint64_t>(successors),
	     static_cast<uint64_t>(cfg.blocks.size()), static_cast<uint64_t>(cfg.natural_loops.size()),
	     static_cast<uint64_t>(cfg.back_edges.size()), cfg.unsupported_reason.c_str());
}

enum class EmbeddedFetchValueType {
	Unknown,
	Constant,
	AttribTable,
	Attrib,
	BufferTable,
	Buffer
};

struct EmbeddedFetchSgprInfo {
	EmbeddedFetchValueType type      = EmbeddedFetchValueType::Unknown;
	int                    attrib_id = 0;
	uint32_t               value     = 0;
};

using EmbeddedFetchVectorLanes = std::map<uint64_t, EmbeddedFetchSgprInfo>;

uint64_t EmbeddedFetchVectorLaneKey(uint32_t reg, uint32_t lane) {
	return (static_cast<uint64_t>(reg) << 32u) | lane;
}

uint32_t EmbeddedFetchLane(uint32_t lane, uint32_t wave_size) {
	return wave_size == 32 || wave_size == 64 ? lane % wave_size : lane;
}

void ClearEmbeddedFetchVectorLanes(EmbeddedFetchVectorLanes* lanes, uint32_t reg) {
	const auto first = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg, 0));
	const auto last  = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg + 1u, 0));
	lanes->erase(first, last);
}

bool IsDecodedSgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Sgpr || op.kind == Decoder::OperandKind::VccLo ||
	       op.kind == Decoder::OperandKind::VccHi;
}

uint32_t DecodedSgprReg(const Decoder::Operand& op) {
	switch (op.kind) {
		case Decoder::OperandKind::VccLo: return 106u;
		case Decoder::OperandKind::VccHi: return 107u;
		default: return op.reg;
	}
}

bool IsDecodedVgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Vgpr;
}

uint32_t DecodedDstSize(const Decoder::Instruction& inst) {
	return std::max(inst.data_dwords, 1u);
}

uint32_t EmbeddedFetchDstSize(const Decoder::Instruction& inst) {
	return inst.opcode == Decoder::Opcode::V_MAD_U64_U32 ? 2u : DecodedDstSize(inst);
}

void ClearEmbeddedFetchSgprs(std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                             const Decoder::Operand& dst, uint32_t size) {
	if (!IsDecodedSgpr(dst)) {
		return;
	}
	const auto register_id = DecodedSgprReg(dst);
	for (uint32_t i = 0; i < size && register_id + i < sgprs.size(); i++) {
		sgprs[register_id + i] = {};
	}
}

bool TryDecodedOperandConstant(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                               const Decoder::Operand& op, uint32_t& value) {
	switch (op.kind) {
		case Decoder::OperandKind::LiteralConstant:
		case Decoder::OperandKind::IntegerInlineConstant:
		case Decoder::OperandKind::FloatInlineConstant: value = op.value; return true;
		case Decoder::OperandKind::Null: value = 0; return true;
		default: break;
	}
	if (IsDecodedSgpr(op) && DecodedSgprReg(op) < sgprs.size() &&
	    sgprs[DecodedSgprReg(op)].type == EmbeddedFetchValueType::Constant) {
		value = sgprs[DecodedSgprReg(op)].value;
		return true;
	}
	return false;
}

bool TryDecodedSmemOffset(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                          const Decoder::Instruction& inst, uint32_t& raw_offset) {
	uint32_t base = 0;
	if (!TryDecodedOperandConstant(sgprs, inst.src1, base)) {
		return false;
	}
	const auto value = static_cast<uint64_t>(base) + inst.offset;
	if (value > 0xffffffffull) {
		return false;
	}
	raw_offset = static_cast<uint32_t>(value);
	return true;
}

bool IsEmbeddedFetchSLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_LOAD_DWORD:
		case Decoder::Opcode::S_LOAD_DWORDX2:
		case Decoder::Opcode::S_LOAD_DWORDX4:
		case Decoder::Opcode::S_LOAD_DWORDX8:
		case Decoder::Opcode::S_LOAD_DWORDX16: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchBufferLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_X:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XY:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZ:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZW: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchAttribPropagationAlu(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_BFE_U32:
		case Decoder::Opcode::S_AND_B32:
		case Decoder::Opcode::S_ADD_I32:
		case Decoder::Opcode::S_ADD_U32:
		case Decoder::Opcode::S_LSHL_B32: return true;
		default: return false;
	}
}

int BufferTableAttribFromOffset(uint32_t raw_offset, int dword) {
	return static_cast<int>((raw_offset + static_cast<uint32_t>(dword) * 4u) / 16u);
}

Frontend::EmbeddedFetchPlan DetectEmbeddedVertexFetch(
    const Decoder::Program& decoded, const ShaderVertexInputInfo& input_info,
    uint32_t user_data_base, uint32_t user_data_count, uint32_t wave_size) {
	const uint32_t    vertex_index_reg   = input_info.logical_stage == ShaderType::Local ? 2u : 5u;
	const uint32_t    instance_index_reg = input_info.logical_stage == ShaderType::Local ? 5u : 8u;
	Frontend::EmbeddedFetchPlan data;
	data.loads.reserve(input_info.resources_num);
	int32_t vertex_offset_candidate   = -1;
	int32_t instance_offset_candidate = -1;
	bool    vertex_offset_conflict    = false;
	bool    instance_offset_conflict  = false;

	const int shift_regs = 8;
	const int attrib_reg = input_info.fetch_attrib_reg + shift_regs;
	const int buffer_reg = input_info.fetch_buffer_reg + shift_regs;

	std::array<EmbeddedFetchSgprInfo, 108> sgprs {};
	std::array<bool, 256>                 vgpr_is_index {};
	EmbeddedFetchVectorLanes               vector_lanes;
	const bool                             track_vector_lanes =
	    std::none_of(decoded.instructions.begin(), decoded.instructions.end(),
	                 [](const auto& inst) {
		                 return Decoder::IsDirectBranch(inst.opcode) ||
		                        inst.opcode == Decoder::Opcode::S_SETPC_B64;
	                 });

	if (attrib_reg >= 0 && attrib_reg < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg].type = EmbeddedFetchValueType::AttribTable;
	}
	if (attrib_reg + 1 >= 0 && attrib_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg + 1].type = EmbeddedFetchValueType::AttribTable;
	}
	if (buffer_reg >= 0 && buffer_reg < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg].type = EmbeddedFetchValueType::BufferTable;
	}
	if (buffer_reg + 1 >= 0 && buffer_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg + 1].type = EmbeddedFetchValueType::BufferTable;
	}

	for (const auto& inst: decoded.instructions) {
		// Fetch shaders accumulate the draw's vertex offset in v0. The PS5 NGG ABI
		// seeds S_NGG_VERTEX_INDEX in v5 and S_NGG_INSTANCE_INDEX in v8, then applies the
		// corresponding direct-draw offsets before fetching.
		const bool vertex_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == 0 || (user_data_base == 8 && inst.dst.reg == vertex_index_reg));
		const bool instance_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == (user_data_base == 8 ? instance_index_reg : 3u));
		uint32_t   sad_zero = 0;
		const bool index_offset_add =
		    (vertex_index_accumulator || instance_index_accumulator) && IsDecodedSgpr(inst.src0) &&
		    ((inst.opcode == Decoder::Opcode::V_ADD_I32 && IsDecodedVgpr(inst.src1) &&
		      inst.src1.reg == inst.dst.reg) ||
		     (user_data_base == 8 &&
		      (inst.dst.reg == vertex_index_reg || inst.dst.reg == instance_index_reg) &&
		      inst.opcode == Decoder::Opcode::V_SAD_U32 && IsDecodedVgpr(inst.src2) &&
		      inst.src2.reg == inst.dst.reg &&
		      TryDecodedOperandConstant(sgprs, inst.src1, sad_zero) && sad_zero == 0));
		if (data.loads.empty() && index_offset_add) {
			const auto reg = DecodedSgprReg(inst.src0);
			if (reg >= user_data_base && reg - user_data_base < user_data_count) {
				auto& candidate = vertex_index_accumulator ? vertex_offset_candidate
				                                           : instance_offset_candidate;
				auto& conflict  = vertex_index_accumulator ? vertex_offset_conflict
				                                           : instance_offset_conflict;
				if (candidate >= 0 && candidate != static_cast<int32_t>(reg)) {
					conflict = true;
				} else {
					candidate = static_cast<int32_t>(reg);
				}
			}
		}
		switch (inst.opcode) {
			case Decoder::Opcode::V_WRITELANE_B32: {
				uint32_t lane = 0;
				if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
					vgpr_is_index[inst.dst.reg] = false;
				}
				if (track_vector_lanes && IsDecodedVgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size() &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					vector_lanes[EmbeddedFetchVectorLaneKey(inst.dst.reg,
					                                        EmbeddedFetchLane(lane, wave_size))] =
					    sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedVgpr(inst.dst)) {
					ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
				}
				break;
			}
			case Decoder::Opcode::V_READLANE_B32: {
				uint32_t lane = 0;
				if (track_vector_lanes && IsDecodedSgpr(inst.dst) &&
				    DecodedSgprReg(inst.dst) < sgprs.size() && IsDecodedVgpr(inst.src0) &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					const auto found = vector_lanes.find(EmbeddedFetchVectorLaneKey(
					    inst.src0.reg, EmbeddedFetchLane(lane, wave_size)));
					sgprs[DecodedSgprReg(inst.dst)] =
					    found != vector_lanes.end() ? found->second : EmbeddedFetchSgprInfo {};
				} else if (IsDecodedSgpr(inst.dst)) {
					ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
				}
				break;
			}
			case Decoder::Opcode::S_MOV_B32:
				if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size()) {
					sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedSgpr(inst.dst)) {
					uint32_t value = 0;
					if (TryDecodedOperandConstant(sgprs, inst.src0, value)) {
						auto& dst = sgprs[DecodedSgprReg(inst.dst)];
						dst.type  = EmbeddedFetchValueType::Constant;
						dst.value = value;
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
					}
				}
				break;
			case Decoder::Opcode::S_MOVK_I32:
				if (IsDecodedSgpr(inst.dst)) {
					auto& dst = sgprs[DecodedSgprReg(inst.dst)];
					dst.type  = EmbeddedFetchValueType::Constant;
					dst.value = inst.src0.value;
				}
				break;
			default:
				if (IsEmbeddedFetchSLoad(inst)) {
					if (IsDecodedSgpr(inst.src0) && DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type ==
					        EmbeddedFetchValueType::AttribTable) {
						uint32_t raw_offset = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							const auto register_id = DecodedSgprReg(inst.dst);
							const int  index       = static_cast<int>(raw_offset / 4u);
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Attrib;
								dst.attrib_id    = index + static_cast<int>(i);
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else if (IsDecodedSgpr(inst.src0) &&
					           DecodedSgprReg(inst.src0) < sgprs.size() &&
					           sgprs[DecodedSgprReg(inst.src0)].type ==
					               EmbeddedFetchValueType::BufferTable) {
						const auto register_id = DecodedSgprReg(inst.dst);
						uint32_t   raw_offset  = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst = sgprs[register_id + i];
								dst.type  = EmbeddedFetchValueType::Buffer;
								dst.attrib_id =
								    BufferTableAttribFromOffset(raw_offset, static_cast<int>(i));
							}
						} else if (IsDecodedSgpr(inst.src1) &&
						           DecodedSgprReg(inst.src1) < sgprs.size() &&
						           sgprs[DecodedSgprReg(inst.src1)].type ==
						               EmbeddedFetchValueType::Attrib &&
						           (inst.offset & 0x3u) == 0) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Buffer;
								dst.attrib_id    = sgprs[DecodedSgprReg(inst.src1)].attrib_id;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
					}
				} else if (inst.opcode == Decoder::Opcode::V_CNDMASK_B32) {
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
						ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
					}
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size() &&
					    IsDecodedVgpr(inst.src0) && inst.src0.reg == instance_index_reg &&
					    IsDecodedVgpr(inst.src1) && inst.src1.reg == vertex_index_reg) {
						vgpr_is_index[inst.dst.reg] = true;
					}
				} else if (IsEmbeddedFetchAttribPropagationAlu(inst)) {
					if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
					    DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type == EmbeddedFetchValueType::Attrib) {
						sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
					} else if (IsDecodedSgpr(inst.dst)) {
						uint32_t src0 = 0;
						uint32_t src1 = 0;
						if (TryDecodedOperandConstant(sgprs, inst.src0, src0) &&
						    TryDecodedOperandConstant(sgprs, inst.src1, src1)) {
							auto& dst = sgprs[DecodedSgprReg(inst.dst)];
							dst.type  = EmbeddedFetchValueType::Constant;
							switch (inst.opcode) {
								case Decoder::Opcode::S_AND_B32: dst.value = src0 & src1; break;
								case Decoder::Opcode::S_LSHL_B32:
									dst.value = src0 << (src1 & 31u);
									break;
								case Decoder::Opcode::S_BFE_U32:
									dst.value = src0 >> (src1 & 31u);
									break;
								default: dst.value = src0 + src1; break;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
						}
					}
				} else if (IsEmbeddedFetchBufferLoad(inst)) {
					if (IsDecodedVgpr(inst.src0) && inst.src0.reg < vgpr_is_index.size() &&
					    vgpr_is_index[inst.src0.reg] &&
					    IsDecodedSgpr(inst.src1) && DecodedSgprReg(inst.src1) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src1)].type == EmbeddedFetchValueType::Buffer) {
						const auto& buffer = sgprs[DecodedSgprReg(inst.src1)];
						if (data.loads.empty()) {
							if (!vertex_offset_conflict) {
								data.vertex_offset_sgpr = vertex_offset_candidate;
							}
							if (!instance_offset_conflict) {
								data.instance_offset_sgpr = instance_offset_candidate;
							}
						}
						auto& load        = data.loads.emplace_back();
						load.pc           = inst.pc;
						load.attrib_id    = buffer.attrib_id;
						load.components   = DecodedDstSize(inst);
					}
				}
				break;
		}
		if (inst.opcode == Decoder::Opcode::V_MOVRELD_B32) {
			vector_lanes.clear();
		} else if (inst.opcode != Decoder::Opcode::V_WRITELANE_B32 && IsDecodedVgpr(inst.dst)) {
			for (uint32_t i = 0;
			     i < EmbeddedFetchDstSize(inst) && inst.dst.reg + i < vgpr_is_index.size();
			     i++) {
				ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg + i);
			}
		}
	}

	return data;
}

Decoder::Program DecodeFusedProgram(std::span<const uint32_t> front, std::span<const uint32_t> back,
                                    std::vector<uint32_t>& joined_code) {
	EXIT_IF(back.empty());
	auto       result      = Decoder::DecodeFrontProgram(front);
	const auto front_words = static_cast<uint32_t>(result.code.size());
	joined_code.assign(result.code.begin(), result.code.end());
	joined_code.insert(joined_code.end(), back.begin(), back.end());
	// The merged-stage ABI passes the back shader in s[6:7]. Give that handoff an
	// ordinary CFG edge, retaining both bodies in one register and LDS lifetime.
	joined_code[front_words - 1u] = 0xbf820000u; // s_branch to the following instruction
	result.instructions.back()    = {};
	Decoder::DecodeInstruction(joined_code, front_words - 1u, result.instructions.back());
	Decoder::Program back_program;
	Decoder::DecodeProgram(back, back_program);
	result.has_swap_pc |= back_program.has_swap_pc;
	const auto back_pc = front_words * sizeof(uint32_t);
	for (auto& inst: back_program.instructions) {
		// A back-stage PC-relative data reference requires its guest code address.
		EXIT_NOT_IMPLEMENTED(inst.opcode == Decoder::Opcode::S_GETPC_B64);
		inst.pc += back_pc;
		inst.branch_target += back_pc;
		result.instructions.push_back(std::move(inst));
	}
	result.code = joined_code;
	return result;
}

void PrepareCallTargets(ShaderSource& source, const CompileOptions& options) {
	using namespace IR;
	EXIT_NOT_IMPLEMENTED(options.stage != ShaderType::Compute);
	source.wave_size = options.wave_size;
	for (uint32_t i = 0; i < source.decoded.instructions.size(); ++i) {
		if (source.decoded.instructions[i].opcode != Decoder::Opcode::S_SWAPPC_B64) continue;
		source.calls.push_back({.instruction = i});
	}
	const auto graph = CFG::BuildGraph(source.decoded);
	Program query;
	query.stage = options.stage;
	query.user_data_base = options.user_data_base;
	query.user_data_count = static_cast<uint32_t>(options.user_data.size());
	query.wave_size = options.wave_size;
	auto& block = *query.block_storage.emplace_back(std::make_unique<Block>());
	query.blocks.push_back(&block);
	struct Producer { bool ready = false; std::map<uint32_t, Value> values; };
	std::map<uint32_t, Producer> producers;
	std::function<Value(uint32_t, uint32_t)> Register;
	std::function<Value(uint32_t, Value)> Resolve = [&](uint32_t before, Value value) -> Value {
		value = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr) return value;
		uint32_t code = UINT32_MAX;
		switch (inst->GetOpcode()) {
			case ValueOpcode::GetScalarRegister: code = static_cast<uint32_t>(inst->Arg(0).ScalarRegister()); break;
			case ValueOpcode::GetVccLo: code = 106u; break;
			case ValueOpcode::GetVccHi: code = 107u; break;
			default: break;
		}
		if (code != UINT32_MAX) {
			const auto result = Register(before, code);
			inst->ReplaceUsesWith(result);
			return result;
		}
		for (size_t i = 0; i < inst->NumArgs(); ++i) inst->SetArg(i, Resolve(before, inst->Arg(i)));
		return value;
	};
	Register = [&](uint32_t before, uint32_t code) -> Value {
		std::vector<uint32_t> crossed_calls;
		const auto definition = CFG::FindScalarDefinition(source.decoded, graph, before, code,
		                                                  source.wave_size, crossed_calls);
		// Several reaching definitions select the value on the GPU, so no SRT walk evaluates it.
		if (!definition) return Value(&block.AppendNewInst(ValueOpcode::UndefU32));
		const auto index = *definition;
		for (const auto crossed: crossed_calls) {
			std::ranges::find(source.calls, crossed, &ShaderSource::Call::instruction)->preserved_registers.set(code);
		}
		if (index == UINT32_MAX) {
			EXIT_IF(code < query.user_data_base || code - query.user_data_base >= query.user_data_count);
			return IREmitter(&block).GetUserData(static_cast<ScalarReg>(code));
		}
		auto [entry, inserted] = producers.try_emplace(index);
		auto& producer = entry->second;
		if (inserted) {
			Block translated;
			Frontend::Translator translator(query, &translated, 1u,
			    (options.input_info.compute->float_mode & 0x10u) == 0u,
			    !options.input_info.compute->async_compute);
			translator.TranslateInstruction(source.decoded.instructions[index]);
			for (const auto& inst: translated) {
				uint32_t destination = UINT32_MAX;
				Value value;
				switch (inst.GetOpcode()) {
					case ValueOpcode::SetScalarRegister:
						destination = static_cast<uint32_t>(inst.Arg(0).ScalarRegister()); value = inst.Arg(1); break;
					case ValueOpcode::SetVccLo: destination = 106u; value = inst.Arg(0); break;
					case ValueOpcode::SetVccHi: destination = 107u; value = inst.Arg(0); break;
					default: break;
				}
				if (destination != UINT32_MAX) producer.values.insert_or_assign(destination, value);
			}
			for (auto& [destination, value]: producer.values) value = Resolve(index, value);
			for (auto& inst: translated) inst.SetParent(&block);
			block.Instructions().splice(block.end(), translated.Instructions());
			producer.ready = true;
		}
		EXIT_IF(!producer.ready || !producer.values.contains(code));
		return producer.values.at(code).Resolve();
	};
	for (auto& site: source.calls) {
		const auto& call = source.decoded.instructions[site.instruction];
		EXIT_NOT_IMPLEMENTED(call.src0.kind != Decoder::OperandKind::Sgpr ||
		                     call.dst.kind != Decoder::OperandKind::Sgpr ||
		                     call.src0.reg != call.dst.reg || call.dst.reg >= 105u);
		for (uint32_t i = 0; i < 2u; ++i) {
			const auto value = Register(site.instruction, call.src0.reg + i);
			if (!ValidateRuntimeValue(query, value, RuntimeValueType::Integer)) {
				EXIT_IF(value.Instruction()->GetOpcode() != ValueOpcode::ReadConstBuffer);
				source.unsupported_call_pc = call.pc;
				source.calls.clear();
				return;
			}
			auto& root = query.value_storage.emplace_back(ValueOpcode::ReferenceU32);
			root.SetArg(0, value);
			site.target[i] = Value(&root);
		}
	}
	RewriteToSsa(query.blocks);
	ConstantPropagationPass(query.blocks, query.wave_size);
	RemoveIdentities(query);
	EliminateDeadCode(query.blocks);
	for (auto& site: source.calls)
		for (auto& value: site.target) value = value.Instruction()->Arg(0).Resolve();
	query.value_storage.clear();
	for (auto& inst: block) {
		inst.SetParent(nullptr);
		if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 || inst.GetOpcode() == ValueOpcode::ReadConstBuffer)
			inst.SetFlags(MemoryFlags {.index = inst.Flags<MemoryFlags>().index});
	}
	query.value_storage.splice(query.value_storage.end(), block.Instructions());
	source.call_targets = std::move(static_cast<ResourcePlan&>(query));
	std::vector<Value> targets;
	for (const auto& site: source.calls) targets.insert(targets.end(), site.target.begin(), site.target.end());
	CompileSrtWalker(source.call_targets, targets);
}

// A guest Wave64 compute shader can run as native Wave32 on a 32-wide host when no instruction
// observes lanes 32..63 of its wave, and nothing relies on the two halves running in lockstep:
//  - no 64-bit ballots, lane ids, lane shuffles, DPP/permutes, readlane or wave reductions
//    (those are the only consumers of lane indexes above 31 or of a 64-bit mask value);
//  - no LDS/GDS, workgroup barriers or store-completion waits (wave lockstep ordering);
//  - no resource that the same shader both reads and writes (cross-lane read-after-write);
//  - a launch size that is a whole number of guest waves, so Wave32 invocations map 1:1.
// Wave-wide branch votes (S_CBRANCH_EXECZ/VCCZ) then become per-warp OpGroupNonUniformAny/All,
// which is exact because skipped blocks only hold EXEC-masked work.
bool CanPromoteWave64ToWave32(const IR::Program& program, const ShaderStageInputInfo& input_info) {
	// Escape hatch for A/B debugging: KYTY_DISABLE_WAVE32_PROMOTION=1 keeps split-wave emission.
	static const bool disabled = std::getenv("KYTY_DISABLE_WAVE32_PROMOTION") != nullptr;
	if (disabled) {
		return false;
	}
	if (program.stage != ShaderType::Compute || program.wave_size != 64u ||
	    input_info.compute == nullptr || input_info.compute->host_subgroup_size != 32u) {
		return false;
	}
	const auto& cs = *input_info.compute;
	if (cs.threads_num[0] == 0u || cs.threads_num[1] == 0u || cs.threads_num[2] == 0u) {
		return false;
	}
	const uint64_t total_threads = static_cast<uint64_t>(cs.threads_num[0]) * cs.threads_num[1] *
	                               cs.threads_num[2];
	if (total_threads % 64u != 0u || total_threads > 1024u) {
		return false;
	}
	const auto& info = program.info;
	if (info.subgroup_ballot || info.subgroup_shuffle || info.subgroup_arithmetic ||
	    info.subgroup_local_invocation_id || info.subgroup_barrier || info.helper_invocation ||
	    info.compute_derivatives || info.uses_lds || info.function_lds || info.uses_gds ||
	    info.bvh || info.bvh8 || info.bvh_stack) {
		return false;
	}
	if (std::ranges::any_of(info.buffers,
	                        [](const auto& buffer) {
		                        return buffer.live && buffer.read && buffer.written;
	                        }) ||
	    std::ranges::any_of(info.images,
	                        [](const auto& image) { return image.read && image.written; })) {
		return false;
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == IR::ValueOpcode::Barrier) {
				return false;
			}
		}
	}
	return true;
}

} // namespace

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return "CS";
		case ShaderType::Vertex: return "VS";
		case ShaderType::Local: return "LS";
		case ShaderType::TessellationControl: return "HS";
		case ShaderType::TessellationEvaluation: return "TES";
		case ShaderType::Mesh: return "MS";
		case ShaderType::Pixel: return "PS";
		default: return "unknown";
	}
}

ShaderSource PrepareShaderSource(std::span<const uint32_t> code, const CompileOptions& options) {
	EXIT_IF(code.empty());
	ShaderSource source;
	if (!options.back_code.empty()) {
		source.decoded = DecodeFusedProgram(code, options.back_code, source.code);
	} else if (options.stage == ShaderType::Local) {
		source.decoded = Decoder::DecodeFrontProgram(code);
		auto& handoff = source.decoded.instructions.back();
		handoff.opcode = Decoder::Opcode::S_ENDPGM;
		handoff.src_count = 0;
	} else {
		Decoder::DecodeProgram(code, source.decoded);
	}
	if (source.decoded.has_swap_pc) {
		PrepareCallTargets(source, options);
		source.code.assign(code.begin(), code.end());
		source.decoded.code = source.code;
	}
	return source;
}

const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime) {
	EXIT_IF(source.unsupported_call_pc != UINT32_MAX);
	auto& reads = source.reads;
	reads.clear();
	std::vector<uint32_t> flat;
	const bool walked = IR::WalkSrt(source.call_targets, runtime, flat, &reads);
	const auto read_code = [&](uint64_t address, std::span<uint32_t> words) {
		if (runtime.read_specialization_memory == nullptr ||
		    !runtime.read_specialization_memory(runtime.userdata, address, words)) return false;
		reads.emplace_back(address, words.size_bytes());
		return true;
	};
	for (auto& site: source.calls) {
		uint32_t low = 0, high = 0;
		if (!walked || !IR::WalkedValue(source.call_targets, site.target[0], low) ||
		    !IR::WalkedValue(source.call_targets, site.target[1], high))
			EXIT("shader call at pc 0x%08x has an unavailable scalar target",
			     source.decoded.instructions[site.instruction].pc);
		// SWAPPC reads the old aliased pair and ignores the target's low two bits.
		const auto address = ((uint64_t {high} << 32u) | low) & ~uint64_t {3};
		EXIT_IF(address == 0u);
		site.address = address;
	}
	const auto query_reads = reads.size();
	if (source.linked) {
		auto& linked = *source.linked;
		bool unchanged = true;
		for (size_t i = 0; i < source.calls.size(); ++i) {
			const auto& function = linked.functions[i];
			const auto expected = std::span(linked.code).subspan(function.offset, function.size);
			const auto preceding = std::span(source.calls).first(i);
			const auto previous = std::ranges::find(preceding, source.calls[i].address, &ShaderSource::Call::address);
			if (previous != preceding.end()) {
				const auto& observed = linked.functions[previous - preceding.begin()];
				if (std::ranges::equal(expected, std::span(linked.code).subspan(observed.offset, observed.size))) continue;
				unchanged = false;
				break;
			}
			linked.observed_function.resize(function.size);
			if (!read_code(source.calls[i].address, linked.observed_function) ||
			    !std::ranges::equal(expected, linked.observed_function)) {
				unchanged = false;
				break;
			}
		}
		if (unchanged) return linked.decoded;
		reads.resize(query_reads);
	}
	ShaderSource::Linked linked {.code = source.code, .decoded = source.decoded};
	const auto& last = source.decoded.instructions.back();
	EXIT_NOT_IMPLEMENTED(last.opcode != Decoder::Opcode::S_ENDPGM);
	// Keep the caller's data/footer at native offsets. Expand leaf CFGs per call site
	// before SSA, so return edges neither require a dispatcher nor merge resource roots.
	uint32_t function_pc = last.pc + last.word_count * sizeof(uint32_t);
	std::map<uint64_t, std::vector<Decoder::Instruction>> functions;
	for (size_t i = 0; i < source.calls.size(); ++i) {
		const auto& site = source.calls[i];
		const auto& call = source.decoded.instructions[site.instruction];
		const auto address = source.calls[i].address;
		auto [entry, inserted] = functions.try_emplace(address);
		auto& function = entry->second;
		if (inserted) {
			for (uint32_t index = 0;;) {
				std::array<uint32_t, Decoder::MaxInstructionRawWords> words {};
				EXIT_IF(index > UINT32_MAX / sizeof(uint32_t) - words.size() ||
				        address > UINT64_MAX - (uint64_t {index} + words.size()) * sizeof(uint32_t));
				uint32_t count = 1;
				Decoder::Instruction inst;
				for (;;) {
					if (!read_code(address + uint64_t {index} * sizeof(uint32_t), std::span(words).first(count)))
						EXIT("shader call at pc 0x%08x cannot read leaf code", call.pc);
					inst = {};
					Decoder::DecodeInstruction(words, 0u, inst);
					if (inst.word_count <= count) break;
					count = inst.word_count;
				}
				inst.pc = index * sizeof(uint32_t);
				if (Decoder::IsDirectBranch(inst.opcode)) inst.branch_target += inst.pc;
				index += inst.word_count;
				EXIT_NOT_IMPLEMENTED(inst.opcode == Decoder::Opcode::S_SWAPPC_B64 ||
				                     inst.opcode == Decoder::Opcode::S_GETPC_B64 ||
				                     inst.opcode == Decoder::Opcode::S_ENDPGM);
				function.push_back(inst);
				if (inst.opcode == Decoder::Opcode::S_SETPC_B64) break;
			}
		}
		const uint32_t function_size = function.back().pc + function.back().word_count * sizeof(uint32_t);
		EXIT_IF(function_pc > UINT32_MAX - function_size);
		const auto function_offset = static_cast<uint32_t>(linked.code.size());
		for (auto inst: function) {
			EXIT_IF(CFG::MayWriteScalarRegister(inst, call.dst.reg, source.wave_size) ||
			        CFG::MayWriteScalarRegister(inst, call.dst.reg + 1u, source.wave_size));
			for (uint32_t code = 0; code < site.preserved_registers.size(); ++code) {
				if (site.preserved_registers[code] &&
				    CFG::MayWriteScalarRegister(inst, code, source.wave_size))
					EXIT("shader call at pc 0x%08x clobbers scalar target dependency s%u", call.pc, code);
			}
			linked.code.insert(linked.code.end(), inst.raw, inst.raw + inst.word_count);
			if (inst.opcode == Decoder::Opcode::S_SETPC_B64) {
				EXIT_NOT_IMPLEMENTED(inst.src0.kind != Decoder::OperandKind::Sgpr || inst.src0.reg != call.dst.reg);
				inst.opcode = Decoder::Opcode::S_BRANCH;
				inst.branch_target = call.pc + call.word_count * sizeof(uint32_t);
			} else if (Decoder::IsDirectBranch(inst.opcode)) {
				EXIT_IF(inst.branch_target >= function_size);
				inst.branch_target += function_pc;
			}
			inst.pc += function_pc;
			linked.decoded.instructions.push_back(std::move(inst));
		}
		linked.functions.push_back({function_offset, function_size / 4u});
		linked.decoded.instructions[site.instruction].branch_target = function_pc;
		function_pc += function_size;
	}
	reads.resize(query_reads);
	for (size_t i = 0; i < source.calls.size(); ++i) {
		const auto preceding = std::span(source.calls).first(i);
		if (std::ranges::find(preceding, source.calls[i].address, &ShaderSource::Call::address) == preceding.end())
			reads.emplace_back(source.calls[i].address, linked.functions[i].size * sizeof(uint32_t));
	}
	linked.decoded.code = linked.code;
	linked.observed_function.resize(std::ranges::max_element(linked.functions, {},
	    &ShaderSource::Linked::Function::size)->size);

	source.linked = std::move(linked);
	++source.revision;
	return source.linked->decoded;
}

namespace {

// The host driver builds one pipeline out of the concatenation of every callable the shader
// dispatches, so an unbounded table turns into a shader whose pipeline the driver may never
// finish building. The limit is overridable so the real threshold can be found without rebuilding.
constexpr uint64_t DefaultDynamicCallInlineLimit = 1024ull * 1024ull;

uint64_t DynamicCallInlineLimit() {
	static const uint64_t limit = [] {
		if (const char* env = std::getenv("KYTY_SBT_INLINE_MAX_BYTES"); env != nullptr) {
			char*      end  = nullptr;
			const auto size = std::strtoull(env, &end, 10);
			if (end != env && size != 0) {
				return static_cast<uint64_t>(size);
			}
		}
		return DefaultDynamicCallInlineLimit;
	}();
	return limit;
}

} // namespace

bool LinkDynamicCallTargets(ShaderSource& source,
                            std::span<const uint64_t> targets,
                            const IR::SrtRuntime& runtime,
                            uint64_t shader_hash) {
	if (targets.empty() || source.unsupported_call_pc == UINT32_MAX) {
		return false;
	}

	const uint32_t call_pc = source.unsupported_call_pc;
	const auto found_call = std::ranges::find_if(source.decoded.instructions,
	    [call_pc](const auto& inst) { return inst.pc == call_pc; });
	if (found_call == source.decoded.instructions.end()) {
		return false;
	}
	const size_t call_idx = found_call - source.decoded.instructions.begin();
	const auto& call_inst = *found_call;
	const uint32_t return_pc = call_pc + call_inst.word_count * sizeof(uint32_t);
	const uint32_t call_sgpr = call_inst.src0.reg;

	ShaderSource::Linked linked {.code = source.code, .decoded = source.decoded};
	const auto& last = source.decoded.instructions.back();
	uint32_t function_pc = last.pc + last.word_count * sizeof(uint32_t);

	struct CalleeInfo {
		uint64_t target_addr;
		uint32_t func_pc;
	};
	std::vector<CalleeInfo> linked_callees;

	const auto read_code = [&](uint64_t address, std::span<uint32_t> words) {
		if (runtime.read_specialization_memory == nullptr ||
		    !runtime.read_specialization_memory(runtime.userdata, address, words)) return false;
		return true;
	};

	struct DecodedCallee {
		uint64_t target_addr;
		std::vector<Decoder::Instruction> instructions;
		uint32_t size_bytes = 0;
	};
	std::vector<DecodedCallee> decoded_callees;

	for (const auto target_addr: targets) {
		std::vector<Decoder::Instruction> function;
		bool callee_has_unsupported = false;
		uint32_t unsupported_pc = UINT32_MAX;
		Decoder::Instruction unsupported_inst {};
		for (uint32_t index = 0;;) {
			std::array<uint32_t, Decoder::MaxInstructionRawWords> words {};
			size_t count = 1;
			Decoder::Instruction inst {};
			bool read_ok = false;
			const uint32_t current_inst_pc = index * sizeof(uint32_t);
			for (;;) {
				if (!read_code(target_addr + uint64_t{index} * sizeof(uint32_t), std::span(words).first(count))) {
					break;
				}
				inst = {};
				Decoder::DecodeInstruction(words, current_inst_pc, inst);
				if (inst.word_count <= count) {
					read_ok = true;
					break;
				}
				count = inst.word_count;
			}
			if (!read_ok || inst.opcode == Decoder::Opcode::UNKNOWN || inst.word_count == 0) {
				break;
			}
			if (inst.opcode == Decoder::Opcode::UNSUPPORTED) {
				callee_has_unsupported = true;
				unsupported_pc = current_inst_pc;
				unsupported_inst = inst;
				break;
			}
			inst.pc = current_inst_pc;
			index += inst.word_count;
			function.push_back(inst);
			if (inst.opcode == Decoder::Opcode::S_SETPC_B64 || inst.opcode == Decoder::Opcode::S_ENDPGM) {
				break;
			}
			if (index > 4096) break;
		}

		if (callee_has_unsupported || function.empty()) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Shader 0x{:016x}: callee target=0x{:016x} contains unsupported instruction at +0x{:x} (opcode={}, words={:08x} {:08x} {:08x}), falling back to S_NOP stubbing\n",
			    shader_hash, target_addr, unsupported_pc,
			    magic_enum::enum_name(unsupported_inst.opcode),
			    unsupported_inst.raw[0], unsupported_inst.raw[1], unsupported_inst.raw[2]));
			return false;
		}

		const uint32_t function_size = function.back().pc + function.back().word_count * sizeof(uint32_t);
		Log::WriteToConsoleAndLog(fmt::format(
		    "  Callee [{}]: target=0x{:016x} inst_count={} size_bytes={} last_op={}\n",
		    decoded_callees.size(), target_addr, function.size(),
		    function_size, magic_enum::enum_name(function.back().opcode)));
		decoded_callees.push_back({
			.target_addr = target_addr,
			.instructions = std::move(function),
			.size_bytes = function_size
		});
	}

	if (decoded_callees.empty()) {
		return false;
	}

	// Calculate trampoline_pc: immediately follows all linked callees
	uint32_t total_callee_bytes = 0;
	for (const auto& callee: decoded_callees) {
		total_callee_bytes += callee.size_bytes;
	}

	const uint64_t inline_limit = DynamicCallInlineLimit();
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader 0x{:016x}: dynamic call table has {} targets, {} bytes of guest code to inline "
	    "(limit {} bytes)\n",
	    shader_hash, decoded_callees.size(), total_callee_bytes, inline_limit));
	if (static_cast<uint64_t>(total_callee_bytes) > inline_limit) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader 0x{:016x}: not inlining the dynamic call table, {} bytes of guest code exceeds "
		    "the {} byte limit; the callable target stays stubbed so the dispatch can still run. "
		    "Raise KYTY_SBT_INLINE_MAX_BYTES to change the limit.\n",
		    shader_hash, total_callee_bytes, inline_limit));
		return false;
	}

	const uint32_t trampoline_pc = function_pc + total_callee_bytes;

	for (auto& callee: decoded_callees) {
		const uint32_t current_func_pc = function_pc;
		linked_callees.push_back({.target_addr = callee.target_addr, .func_pc = current_func_pc});

		bool terminated_with_branch = false;
		for (auto inst: callee.instructions) {
			if (inst.opcode == Decoder::Opcode::S_SETPC_B64 || inst.opcode == Decoder::Opcode::S_ENDPGM) {
				inst.opcode = Decoder::Opcode::S_BRANCH;
				inst.family = Decoder::Family::SOPP;
				inst.branch_target = trampoline_pc;
				inst.raw[0] = 0xBF820000u;
				inst.word_count = 1;
				inst.dst = {}; inst.src0 = {}; inst.src1 = {};
				inst.src_count = 0;
				terminated_with_branch = true;
			} else if (Decoder::IsDirectBranch(inst.opcode)) {
				inst.branch_target += current_func_pc;
			}
			inst.pc += current_func_pc;
			linked.code.insert(linked.code.end(), inst.raw, inst.raw + inst.word_count);
			linked.decoded.instructions.push_back(std::move(inst));
		}
		if (!terminated_with_branch) {
			Decoder::Instruction ret_branch {};
			ret_branch.pc = function_pc + callee.size_bytes;
			ret_branch.opcode = Decoder::Opcode::S_BRANCH;
			ret_branch.family = Decoder::Family::SOPP;
			ret_branch.branch_target = trampoline_pc;
			ret_branch.raw[0] = 0xBF820000u;
			ret_branch.word_count = 1;
			linked.code.push_back(ret_branch.raw[0]);
			linked.decoded.instructions.push_back(ret_branch);
			function_pc += sizeof(uint32_t);
		}
		function_pc += callee.size_bytes;
	}

	// Trampoline block: V_MOV_B32 v0, 0 then S_BRANCH return_pc.
	// In the active-lane serialization loop (pc=0x00002430..0x00002484), v0 holds the completion flag (0=done, 1=not done).
	// Because material subroutines use v0 as a scratch/argument register, returning without resetting v0 leaves v0 non-zero,
	// causing V_CMPX_NE_U32 exec_lo, 0, v0 to never clear exec_lo, looping infinitely on the GPU.
	// Running V_MOV_B32 v0, 0 under the callee's active execution mask marks completed lanes as finished,
	// ensuring the wave loop terminates after all materials are processed.
	Decoder::Instruction mov_v0 {};
	mov_v0.pc = function_pc;
	mov_v0.word_count = 1;
	mov_v0.raw[0] = 0x7E000280u; // V_MOV_B32 v0, 0
	mov_v0.opcode = Decoder::Opcode::V_MOV_B32;
	mov_v0.family = Decoder::Family::VOP1;
	mov_v0.dst.kind = Decoder::OperandKind::Vgpr;
	mov_v0.dst.reg = 0;
	mov_v0.src0.kind = Decoder::OperandKind::IntegerInlineConstant;
	mov_v0.src0.value = 0;
	mov_v0.src0.signed_val = 0;
	mov_v0.src_count = 1;
	linked.code.push_back(mov_v0.raw[0]);
	linked.decoded.instructions.push_back(mov_v0);
	function_pc += 4;

	Decoder::Instruction trampoline_branch {};
	trampoline_branch.pc = function_pc;
	trampoline_branch.word_count = 1;
	trampoline_branch.raw[0] = 0xBF820000u;
	trampoline_branch.opcode = Decoder::Opcode::S_BRANCH;
	trampoline_branch.family = Decoder::Family::SOPP;
	trampoline_branch.branch_target = return_pc;
	trampoline_branch.src_count = 0;
	linked.code.push_back(trampoline_branch.raw[0]);
	linked.decoded.instructions.push_back(trampoline_branch);
	function_pc += 4;

	// Dispatcher block
	const uint32_t dispatcher_pc = function_pc;

	// At dispatcher entry: S_AND_B32 s[call_sgpr], s[call_sgpr], 0xFFFFFFFCu (~3u)
	// Clear the 2 low mode bits in the function pointer so it cleanly matches target_lo regardless of PS5 AGC mode flags.
	Decoder::Instruction mask_inst {};
	mask_inst.pc = function_pc;
	mask_inst.word_count = 2;
	mask_inst.raw[0] = 0x80000000u | (0x0eu << 23u) | ((call_sgpr & 0x7fu) << 16u) | (0xFFu << 8u) | (call_sgpr & 0xffu);
	mask_inst.raw[1] = 0xFFFFFFFCu;
	mask_inst.opcode = Decoder::Opcode::S_AND_B32;
	mask_inst.family = Decoder::Family::SOP2;
	mask_inst.dst.kind = Decoder::OperandKind::Sgpr;
	mask_inst.dst.reg = call_sgpr;
	mask_inst.src0.kind = Decoder::OperandKind::Sgpr;
	mask_inst.src0.reg = call_sgpr;
	mask_inst.src1.kind = Decoder::OperandKind::LiteralConstant;
	mask_inst.src1.value = 0xFFFFFFFCu;
	mask_inst.src1.signed_val = -4;
	mask_inst.src_count = 2;
	linked.code.push_back(mask_inst.raw[0]);
	linked.code.push_back(mask_inst.raw[1]);
	linked.decoded.instructions.push_back(mask_inst);
	function_pc += 8;

	for (const auto& callee: linked_callees) {
		const uint32_t target_lo = static_cast<uint32_t>(callee.target_addr & 0xFFFFFFFFu) & ~3u;

		// S_CMP_EQ_U32 s[call_sgpr], literal (target_lo)
		Decoder::Instruction cmp_inst {};
		cmp_inst.pc = function_pc;
		cmp_inst.word_count = 2;
		cmp_inst.raw[0] = 0xBF060000u | (0xFFu << 8u) | (call_sgpr & 0xFFu);
		cmp_inst.raw[1] = target_lo;
		cmp_inst.opcode = Decoder::Opcode::S_CMP_EQ_U32;
		cmp_inst.family = Decoder::Family::SOPC;
		cmp_inst.dst.kind = Decoder::OperandKind::Scc;
		cmp_inst.src0.kind = Decoder::OperandKind::Sgpr;
		cmp_inst.src0.reg = call_sgpr;
		cmp_inst.src1.kind = Decoder::OperandKind::LiteralConstant;
		cmp_inst.src1.value = target_lo;
		cmp_inst.src1.signed_val = static_cast<int32_t>(target_lo);
		cmp_inst.src_count = 2;
		linked.code.push_back(cmp_inst.raw[0]);
		linked.code.push_back(cmp_inst.raw[1]);
		linked.decoded.instructions.push_back(cmp_inst);
		function_pc += 8;

		// S_CBRANCH_SCC1 callee.func_pc
		Decoder::Instruction branch_scc1 {};
		branch_scc1.pc = function_pc;
		branch_scc1.word_count = 1;
		branch_scc1.raw[0] = 0xBF850000u;
		branch_scc1.opcode = Decoder::Opcode::S_CBRANCH_SCC1;
		branch_scc1.family = Decoder::Family::SOPP;
		branch_scc1.branch_target = callee.func_pc;
		branch_scc1.src_count = 0;
		linked.code.push_back(branch_scc1.raw[0]);
		linked.decoded.instructions.push_back(branch_scc1);
		function_pc += 4;
	}

	// Default fallback: S_BRANCH trampoline_pc (deactivate lane to prevent infinite loop)
	Decoder::Instruction default_branch {};
	default_branch.pc = function_pc;
	default_branch.word_count = 1;
	default_branch.raw[0] = 0xBF820000u;
	default_branch.opcode = Decoder::Opcode::S_BRANCH;
	default_branch.family = Decoder::Family::SOPP;
	default_branch.branch_target = trampoline_pc;
	default_branch.src_count = 0;
	linked.code.push_back(default_branch.raw[0]);
	linked.decoded.instructions.push_back(default_branch);
	function_pc += 4;

	// Patch call instruction at call_idx to branch to dispatcher_pc
	auto& call = linked.decoded.instructions[call_idx];
	call.opcode = Decoder::Opcode::S_BRANCH;
	call.family = Decoder::Family::SOPP;
	call.branch_target = dispatcher_pc;
	call.raw[0] = 0xBF820000u;
	call.word_count = 1;
	call.dst = {}; call.src0 = {}; call.src1 = {};
	call.src_count = 0;
	linked.code[call_pc / sizeof(uint32_t)] = 0xBF820000u;

	linked.decoded.code = linked.code;
	source.decoded = std::move(linked.decoded);
	source.code = std::move(linked.code);
	source.unsupported_call_pc = UINT32_MAX;
	source.calls.clear();
	source.decoded.has_swap_pc = false;
	++source.revision;
	return true;
}

TranslateResult TranslateProgram(std::span<const uint32_t> code, const CompileOptions& options) {
	auto source = PrepareShaderSource(code, options);
	// Direct translation has no pipeline-cache SBT resolver. Do not let an unresolved
	// GPU-selected call reach TranslateProgram(decoded), where it aborts shader compilation.
	// Match PipelineCache's existing fallback: replace only unresolved S_SWAPPC instructions
	// with S_NOP, then re-decode so the remaining shader can still be compiled.
	uint32_t stubbed_calls = 0;
	while (source.unsupported_call_pc != UINT32_MAX) {
		if (++stubbed_calls > source.decoded.instructions.size()) {
			EXIT("shader 0x%016" PRIx64 " has too many unresolved dynamic calls",
			     options.shader_hash);
		}
		const auto pc = source.unsupported_call_pc;
		const auto instruction = std::ranges::find_if(source.decoded.instructions,
		    [pc](const auto& inst) { return inst.pc == pc; });
		if (instruction == source.decoded.instructions.end() || instruction->word_count == 0 ||
		    pc % sizeof(uint32_t) != 0 || pc / sizeof(uint32_t) >= source.code.size()) {
			EXIT("shader 0x%016" PRIx64 " has invalid unresolved dynamic call at pc 0x%08x",
			     options.shader_hash, pc);
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader 0x{:016x}: stubbing unresolved dynamic call at pc 0x{:08x} in direct translation\n",
		    options.shader_hash, pc));
		auto patched_code = source.code;
		const auto word_index = pc / sizeof(uint32_t);
		for (uint32_t i = 0; i < instruction->word_count && word_index + i < patched_code.size(); ++i) {
			patched_code[word_index + i] = 0xbf800000u; // S_NOP 0
		}
		source = PrepareShaderSource(patched_code, options);
	}
	return TranslateProgram(source.decoded, options);
}

TranslateResult TranslateProgram(const Decoder::Program& decoded, const CompileOptions& options) {
	EXIT_IF(decoded.instructions.empty());
	if (decoded.has_swap_pc) {
		for (const auto& inst: decoded.instructions) {
			if (inst.opcode == Decoder::Opcode::S_SWAPPC_B64 && inst.branch_target == UINT32_MAX)
				EXIT("shader call at pc 0x%08x must be linked before translation", inst.pc);
		}
	}
	if (options.stage != ShaderType::Compute && options.stage != ShaderType::Vertex &&
	    options.stage != ShaderType::Pixel && options.stage != ShaderType::Mesh &&
	    options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		EXIT("shader recompiler received unsupported stage %u\n",
		     static_cast<unsigned>(options.stage));
	}

	const auto* stage        = StageName(options.stage);
	const bool  log_phases   = options.dump_ir || !Config::PerfModeEnabled();
	const auto compile_begin = std::chrono::steady_clock::now();
	const auto phase_ms = [&compile_begin]() {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		    std::chrono::steady_clock::now() - compile_begin).count());
	};

	std::string decoded_dump;
	if (options.dump_ir) {
		decoded_dump = Decoder::ProgramToString(decoded);
		LOGF("ShaderRecompiler %s decoded RDNA2:\n%s", stage, decoded_dump.c_str());
	}

	if (log_phases) {
		LOGF("ShaderRecompiler %s phase begin: hash=0x%016" PRIx64 " CFG BuildGraph\n", stage,
		     options.shader_hash);
	}
	auto native_cfg = CFG::BuildGraph(decoded);
	CFG::Graph structured_cfg;
	auto* selected_cfg = &native_cfg;
	if (log_phases) {
		LOGF("ShaderRecompiler %s phase end: hash=0x%016" PRIx64 " CFG BuildGraph blocks=%" PRIu64
		     " loops=%" PRIu64 " back_edges=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
		     stage, options.shader_hash,
		     static_cast<uint64_t>(native_cfg.blocks.size()),
		     static_cast<uint64_t>(native_cfg.natural_loops.size()),
		     static_cast<uint64_t>(native_cfg.back_edges.size()), phase_ms());
	}
	if (native_cfg.irreducible) {
		LogDispatcherFallback(options, native_cfg, "build");
	} else {
		if (log_phases) {
			LOGF("ShaderRecompiler %s phase begin: hash=0x%016" PRIx64 " CFG Structurize\n", stage,
			     options.shader_hash);
		}
		structured_cfg = CFG::Structurize(native_cfg);
		if (structured_cfg.unsupported) {
			native_cfg.unsupported = true;
			native_cfg.failure_kind = structured_cfg.failure_kind;
			native_cfg.failure_block = structured_cfg.failure_block;
			native_cfg.unsupported_reason = structured_cfg.unsupported_reason;
			LogDispatcherFallback(options, native_cfg, "structurize");
		} else {
			selected_cfg = &structured_cfg;
			if (log_phases) {
				LOGF("ShaderRecompiler %s structured CFG success: blocks=%" PRIu64 "\n", stage,
				     static_cast<uint64_t>(selected_cfg->blocks.size()));
			}
		}
		if (log_phases) {
			LOGF("ShaderRecompiler %s phase end: hash=0x%016" PRIx64 " CFG Structurize blocks=%" PRIu64
			     " loops=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
			     stage, options.shader_hash,
			     static_cast<uint64_t>(selected_cfg->blocks.size()),
			     static_cast<uint64_t>(selected_cfg->natural_loops.size()), phase_ms());
		}
	}

	const auto& cfg = *selected_cfg;
	Frontend::EmbeddedFetchPlan embedded_fetch;
	if ((options.stage == ShaderType::Vertex || options.stage == ShaderType::Local) &&
	    options.input_info.vertex->has_dynamic_vertex_fetch) {
		embedded_fetch = DetectEmbeddedVertexFetch(
		    decoded, *options.input_info.vertex, options.user_data_base,
		    static_cast<uint32_t>(options.user_data.size()), options.wave_size);
		if (!embedded_fetch.loads.empty() && log_phases) {
			LOGF("ShaderRecompiler %s embedded vertex fetch plan: detected=%" PRIu64 "\n", stage,
			     static_cast<uint64_t>(embedded_fetch.loads.size()));
		}
	}
	Frontend::TranslateOptions translate_options {
	    .stage            = options.stage,
	    .wave_size        = options.wave_size,
	    .shader_hash      = options.shader_hash,
	    .user_data_base   = options.user_data_base,
	    .user_data_count  = static_cast<uint32_t>(options.user_data.size()),
	    .input_info       = options.input_info,
	    .embedded_fetch   = embedded_fetch.loads.empty() ? nullptr : &embedded_fetch,
	};
	if (log_phases) {
		LOGF("ShaderRecompiler %s phase begin: hash=0x%016" PRIx64 " IR TranslateProgram\n", stage,
		     options.shader_hash);
	}
	auto ir = Frontend::TranslateProgram(decoded, cfg, translate_options);
	if (log_phases) {
		LOGF("ShaderRecompiler %s phase end: hash=0x%016" PRIx64 " IR TranslateProgram blocks=%" PRIu64
		     " elapsed_ms=%" PRIu64 "\n",
		     stage, options.shader_hash,
		     static_cast<uint64_t>(ir.blocks.size()), phase_ms());
	}
	IR::RewriteToSsa(ir.blocks);
	const auto simplify = [&ir] {
		IR::ConstantPropagationPass(ir.blocks, ir.wave_size);
		IR::RemoveIdentities(ir);
		IR::EliminateDeadCode(ir.blocks);
	};
	simplify();
	const auto read_lane_stats = IR::EliminateReadLane(ir, ir.wave_size);
	if (log_phases &&
	    (read_lane_stats.rewritten_reads != 0 || read_lane_stats.rewritten_reductions != 0)) {
		LOGF("ShaderRecompiler %s read-lane elimination: reads=%" PRIu32
		     " reductions=%" PRIu32 "\n",
		     stage, read_lane_stats.rewritten_reads, read_lane_stats.rewritten_reductions);
	}
	// Cleanup also exposes profitable folds by removing dead predicate consumers.
	simplify();
	if (decoded.has_swap_pc) {
		// Leaf returns are resolved CFG edges; native PC values must not escape into GPU data.
		for (const auto* block: ir.blocks)
			for (const auto& inst: *block)
				EXIT_NOT_IMPLEMENTED(inst.GetOpcode() == IR::ValueOpcode::GetShaderBase);
	}
	LowerTessellationMemory(ir, options);
	std::string cfg_dump;
	if (options.dump_ir) {
		cfg_dump = CFG::GraphToString(cfg);
		LOGF("ShaderRecompiler %s native IR before resource tracking:\n%s", stage,
		     MakeIrDump(cfg_dump, ir).c_str());
	}
	IR::TrackResources(ir, decoded, native_cfg, options.input_info);
	TranslateResult result;
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(decoded_dump);
		result.cfg_dump     = std::move(cfg_dump);
	}
	return result;
}

CompileResult CompileProgram(TranslateResult translated, const CompileOptions& options,
                             const IR::ResourceSpecialization& specialization,
                             uint32_t push_data_start_dword) {
	const auto emit_begin = std::chrono::steady_clock::now();
	auto& ir = translated.program;
	const auto* stage = StageName(ir.stage);
	IR::ApplyResourceSpecialization(ir, specialization);
	// The resource plan owns host descriptor evaluation now. Keep only dependencies consumed
	// by GPU memory operations; bound descriptor dwords must not retain shader instructions.
	for (auto& inst: ir.value_storage) {
		inst.Invalidate();
	}
	for (auto* block: ir.blocks) {
		for (auto& inst: *block) {
			const auto op = inst.GetOpcode();
			uint32_t first = 0;
			if (op == IR::ValueOpcode::GetBufferResource) {
				if (std::ranges::any_of(inst.Uses(), [&](const IR::Use& use) {
					return ir.memory_info[use.user->Flags<IR::MemoryFlags>().index].kind ==
					       IR::ResourceKind::IndirectBuffer;
				})) {
					continue;
				}
				const auto resource = inst.Flags<uint32_t>();
				first               = resource < ir.info.buffers.size() &&
				                              ir.info.buffers[resource].indirect_root == resource
				                          ? 1u
				                          : 0u;
			} else if (op == IR::ValueOpcode::GetImageResource) {
				const auto resource = inst.Flags<uint32_t>();
				first = resource < ir.info.images.size() &&
				                ir.info.images[resource].indirect_root == resource ? 1u : 0u;
			} else if (op == IR::ValueOpcode::GetSamplerResource) {
				const auto resource = inst.Flags<uint32_t>();
				first = resource < ir.info.samplers.size() &&
				                !ir.info.samplers[resource].indirect_resources.empty() ? 1u : 0u;
			} else {
				continue;
			}
			for (size_t index = first; index < inst.NumArgs(); index++) {
				inst.SetArg(index, IR::Value(0u));
			}
		}
	}
	ir.value_storage.clear();
	IR::RemoveIdentities(ir);
	IR::EliminateDeadCode(ir.blocks);

	IR::CollectShaderInfo(ir, options.input_info);
	if (CanPromoteWave64ToWave32(ir, options.input_info)) {
		// The shader has no cross-lane dependency on lanes 32..63, so it runs as native Wave32
		// (one host lane per guest lane) instead of two interleaved halves per host lane.
		ir.wave_size = 32u;
	}
	std::string ir_dump;
	if (options.dump_ir) {
		ir_dump = MakeIrDump(translated.cfg_dump, ir);
		LOGF("ShaderRecompiler %s native IR:\n%s", stage, ir_dump.c_str());
	}

	const bool log_phases = options.dump_ir || !Config::PerfModeEnabled();
	if (log_phases) {
		LOGF("ShaderRecompiler %s phase begin: hash=0x%016" PRIx64 " SPIR-V EmitProgram\n", stage,
		     ir.shader_hash);
	}
	auto spirv = Spirv::EmitProgram(ir, options.input_info, push_data_start_dword, *options.profile,
	                                options.emit_debug_info);
	if (log_phases) {
		LOGF("ShaderRecompiler %s phase end: hash=0x%016" PRIx64 " SPIR-V EmitProgram words=%" PRIu64
		     " wave_size=%u elapsed_ms=%" PRIu64 "\n",
		     stage, ir.shader_hash, static_cast<uint64_t>(spirv.size()), ir.wave_size,
		     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		                               std::chrono::steady_clock::now() - emit_begin)
		                               .count()));
	}
	if (Config::PerfModeEnabled() && Config::ShaderCacheEnabled() && ir.shader_hash != 0 &&
	    !spirv.empty()) {
		const auto hash = static_cast<std::size_t>(ir.shader_hash);
		std::shared_ptr<CompiledShader> entry;
		bool                            needs_disk_write = false;
		{
			std::unique_lock lk(gCacheMtx);
			auto& slot = gShaderCache[hash];
			if (!slot) {
				slot             = std::make_shared<CompiledShader>();
				needs_disk_write = true;
			}
			entry = slot;
		}
		{
			std::scoped_lock lk(entry->mutex);
			if (!entry->ready.load(std::memory_order_relaxed)) {
				needs_disk_write = true;
			}
			entry->spirv = spirv;
			entry->ready.store(true, std::memory_order_release);
		}
		if (needs_disk_write) {
			std::error_code ec;
			const auto      dir = GetShaderDiskCacheDir();
			std::filesystem::create_directories(dir, ec);
			if (!ec) {
				const auto out_path =
				    dir / fmt::format("{:016x}.spv", static_cast<uint64_t>(hash));
				std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
				if (out.is_open()) {
					out.write(reinterpret_cast<const char*>(spirv.data()),
					          static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
				}
			}
		}
	}
	CompileResult result;
	result.spirv   = std::move(spirv);
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(translated.decoded_dump);
		result.ir_dump      = std::move(ir_dump);
	}
	return result;
}

namespace {

const std::vector<uint32_t>& GetPlaceholderSpirv() {
	static const std::vector<uint32_t> kPlaceholder = [] {
		const std::array<std::filesystem::path, 2> candidates = {
		    std::filesystem::path("src/graphics/shader/recompiler/placeholder.spv"),
		    std::filesystem::path("placeholder.spv"),
		};
		for (const auto& path: candidates) {
			std::error_code ec;
			if (!std::filesystem::exists(path, ec)) continue;
			const auto byte_size = std::filesystem::file_size(path, ec);
			if (ec || byte_size < 20 || (byte_size % sizeof(uint32_t)) != 0) continue;
			std::ifstream in(path, std::ios::binary);
			if (!in.is_open()) continue;
			std::vector<uint32_t> words(byte_size / sizeof(uint32_t));
			if (in.read(reinterpret_cast<char*>(words.data()),
			            static_cast<std::streamsize>(byte_size)) &&
			    words[0] == 0x07230203u) {
				return words;
			}
		}
		return std::vector<uint32_t> {
		    0x07230203u, 0x00010500u, 0x000d000bu, 0x00000006u, 0x00000000u, 0x00020011u,
		    0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
		    0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0005000fu, 0x00000004u,
		    0x00000004u, 0x6e69616du, 0x00000000u, 0x00030010u, 0x00000004u, 0x00000007u,
		    0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00050036u,
		    0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
		    0x000100fdu, 0x00010038u,
		};
	}();
	return kPlaceholder;
}

bool TryLoadCachedShaderFromDisk(std::size_t hash, std::vector<uint32_t>& out_spirv) {
	std::error_code ec;
	const auto      path =
	    GetShaderDiskCacheDir() / fmt::format("{:016x}.spv", static_cast<uint64_t>(hash));
	if (!std::filesystem::exists(path, ec)) return false;
	const auto byte_size = std::filesystem::file_size(path, ec);
	if (ec || byte_size < 20 || (byte_size % sizeof(uint32_t)) != 0) return false;
	std::ifstream in(path, std::ios::binary);
	if (!in.is_open()) return false;
	std::vector<uint32_t> words(byte_size / sizeof(uint32_t));
	if (!in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(byte_size))) {
		return false;
	}
	if (words[0] != 0x07230203u) return false;
	out_spirv = std::move(words);
	return true;
}

void SaveCachedShaderToDisk(std::size_t hash, std::span<const uint32_t> spirv) {
	if (spirv.empty()) return;
	std::error_code ec;
	const auto      dir = GetShaderDiskCacheDir();
	std::filesystem::create_directories(dir, ec);
	if (ec) return;
	const auto    path = dir / fmt::format("{:016x}.spv", static_cast<uint64_t>(hash));
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	if (!out.is_open()) return;
	out.write(reinterpret_cast<const char*>(spirv.data()),
	          static_cast<std::streamsize>(spirv.size_bytes()));
}

} // namespace

std::filesystem::path GetShaderDiskCacheDir() {
	if (const char* appdata = std::getenv("APPDATA"); appdata != nullptr && appdata[0] != '\0') {
		return std::filesystem::path(appdata) / "KytyPS5" / "shader_cache";
	}
	return std::filesystem::current_path() / "_ShaderCache";
}

void PreloadShaderCacheFromDisk() {
	std::error_code ec;
	const auto      dir = GetShaderDiskCacheDir();
	if (!std::filesystem::exists(dir, ec)) return;
	std::size_t loaded_count = 0;
	for (const auto& entry: std::filesystem::directory_iterator(dir, ec)) {
		if (!entry.is_regular_file()) continue;
		const auto& path = entry.path();
		if (path.extension() != ".spv") continue;
		const auto stem = path.stem().string();
		if (stem.empty()) continue;
		char*            end_ptr = nullptr;
		const auto       parsed  = std::strtoull(stem.c_str(), &end_ptr, 16);
		if (end_ptr == stem.c_str() || *end_ptr != '\0') continue;
		const auto byte_size = entry.file_size(ec);
		if (ec || byte_size < 20 || (byte_size % sizeof(uint32_t)) != 0) continue;
		std::ifstream in(path, std::ios::binary);
		if (!in.is_open()) continue;
		std::vector<uint32_t> words(byte_size / sizeof(uint32_t));
		if (!in.read(reinterpret_cast<char*>(words.data()),
		             static_cast<std::streamsize>(byte_size))) {
			continue;
		}
		if (words[0] != 0x07230203u) continue;
		auto shader   = std::make_shared<CompiledShader>();
		shader->spirv = std::move(words);
		shader->ready.store(true, std::memory_order_release);
		{
			std::unique_lock lk(gCacheMtx);
			gShaderCache[static_cast<std::size_t>(parsed)] = std::move(shader);
		}
		++loaded_count;
	}
	if (loaded_count > 0) {
		LOGF("ShaderRecompiler preloaded %zu cached SPIR-V shaders from %s\n", loaded_count,
		     dir.string().c_str());
	}
}

void ClearShaderCacheOnDisk() {
	{
		std::unique_lock lk(gCacheMtx);
		gShaderCache.clear();
	}
	std::error_code ec;
	const auto      dir = GetShaderDiskCacheDir();
	if (std::filesystem::exists(dir, ec)) {
		std::filesystem::remove_all(dir, ec);
	}
}

std::shared_ptr<CompiledShader> RequestShader(std::span<const uint8_t> raw_blob) {
	if (raw_blob.empty() || (raw_blob.size() % sizeof(uint32_t)) != 0) {
		auto placeholder   = std::make_shared<CompiledShader>();
		placeholder->spirv = GetPlaceholderSpirv();
		placeholder->ready.store(true, std::memory_order_release);
		return placeholder;
	}
	std::span<const uint32_t> words(reinterpret_cast<const uint32_t*>(raw_blob.data()),
	                                raw_blob.size() / sizeof(uint32_t));
	return RequestShader(words);
}

std::shared_ptr<CompiledShader> RequestShader(std::span<const uint32_t>         code,
                                              const CompileOptions&             options,
                                              const IR::ResourceSpecialization& specialization,
                                              uint32_t push_data_start_dword) {
	const std::size_t hash = options.shader_hash != 0
	                             ? static_cast<std::size_t>(options.shader_hash)
	                             : static_cast<std::size_t>(XXH3_64bits(code.data(), code.size_bytes()));

	// 1. Fast path - already in memory cache
	{
		std::shared_lock lk(gCacheMtx);
		if (auto it = gShaderCache.find(hash); it != gShaderCache.end()) {
			return it->second;
		}
	}

	// 2. Check disk cache before enqueuing background compilation
	std::vector<uint32_t> disk_spirv;
	if (TryLoadCachedShaderFromDisk(hash, disk_spirv)) {
		auto cached   = std::make_shared<CompiledShader>();
		cached->spirv = std::move(disk_spirv);
		cached->ready.store(true, std::memory_order_release);
		std::unique_lock lk(gCacheMtx);
		auto [it, inserted] = gShaderCache.emplace(hash, cached);
		return it->second;
	}

	// 3. Create placeholder entry with valid fallback SPIR-V and enqueue async compilation
	auto placeholder   = std::make_shared<CompiledShader>();
	placeholder->spirv = GetPlaceholderSpirv();
	placeholder->ready.store(false, std::memory_order_release);
	{
		std::unique_lock lk(gCacheMtx);
		auto [it, inserted] = gShaderCache.emplace(hash, placeholder);
		if (!inserted) {
			return it->second;
		}
	}

	std::vector<uint32_t>      code_copy(code.begin(), code.end());
	std::vector<uint32_t>      user_data_copy(options.user_data.begin(), options.user_data.end());
	std::vector<uint32_t>      back_code_copy(options.back_code.begin(), options.back_code.end());
	CompileOptions             opts_copy = options;
	opts_copy.shader_hash                = static_cast<uint64_t>(hash);
	opts_copy.dump_ir                    = false;
	IR::ResourceSpecialization spec_copy = specialization;

	Common::GetThreadPool().Enqueue([placeholder, hash, code_copy = std::move(code_copy),
	                                 user_data_copy = std::move(user_data_copy),
	                                 back_code_copy = std::move(back_code_copy), opts_copy,
	                                 spec_copy      = std::move(spec_copy),
	                                 push_data_start_dword]() mutable {
		opts_copy.user_data = user_data_copy;
		opts_copy.back_code = back_code_copy;
		auto translated     = TranslateProgram(code_copy, opts_copy);
		auto compiled =
		    CompileProgram(std::move(translated), opts_copy, spec_copy, push_data_start_dword);
		{
			std::scoped_lock lk(placeholder->mutex);
			placeholder->spirv = std::move(compiled.spirv);
			placeholder->ready.store(true, std::memory_order_release);
		}
		SaveCachedShaderToDisk(hash, placeholder->spirv);
	});

	return placeholder;
}

} // namespace Libs::Graphics::ShaderRecompiler

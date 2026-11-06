#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVBUILDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVBUILDER_H_

#include "common/common.h"

#include <cstring>
#include <initializer_list>
#include <iterator>
#include <map>
#include <set>
#include <span>
#include <spirv/unified1/spirv.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

class EmitterAttributionTracker;

struct TypeAnnotation {
	spv::Op               opcode = spv::OpNop;
	std::vector<uint32_t> operands; // Already encoded operand words.
};

struct DeferredPhi {
	size_t word_offset    = 0;
};

class Builder {
public:
	explicit Builder(uint32_t version = 0x00010300u);
	~Builder() = default;
	KYTY_CLASS_DEFAULT_COPY(Builder);

	uint32_t AllocateId();
	void     SetAttributionTracker(EmitterAttributionTracker* tracker) { m_attribution_tracker = tracker; }
	[[nodiscard]] EmitterAttributionTracker* GetAttributionTracker() const { return m_attribution_tracker; }
	void     RequireVersion(uint32_t version);
	void     RequireCapability(spv::Capability capability);
	void     RequireExtension(const char* name);
	uint32_t Import(const char* name);
	template <typename... Args>
	uint32_t Type(spv::Op opcode, const Args&... operands) {
		FillTypeKey(opcode, operands...);
		return DeclareType(opcode);
	}

	template <typename... Args>
	uint32_t DecoratedType(spv::Op opcode, std::initializer_list<TypeAnnotation> annotations,
	                       const Args&... operands) {
		FillTypeKey(opcode, operands...);
		return DeclareDecoratedType(opcode, annotations);
	}

	template <typename... Args>
	uint32_t Constant(spv::Op opcode, uint32_t type, const Args&... operands) {
		m_key_scratch.clear();
		AppendOperands(m_key_scratch, opcode, type, operands...);
		return DeclareConstant(opcode);
	}

	uint32_t DefineGlobalVariable(uint32_t pointer_type, spv::StorageClass storage_class);
	void DefineGlobalVariable(uint32_t id, uint32_t pointer_type, spv::StorageClass storage_class);

	void AddMemoryModel(spv::AddressingModel addressing_model, spv::MemoryModel memory_model);
	void AddEntryPoint(spv::ExecutionModel execution_model, uint32_t entry_point, const char* name,
	                   const std::vector<uint32_t>& interfaces);
	template <typename... Args>
	void AddExecutionMode(uint32_t entry_point, spv::ExecutionMode mode, const Args&... operands) {
		AppendInstruction(m_execution_modes, spv::OpExecutionMode, entry_point, mode, operands...);
	}

	uint32_t AddString(const char* text);
	void EnableNames() { m_emit_names = true; }
	void AddName(uint32_t target, const char* name);
	template <typename... Args>
	void AddAnnotation(spv::Op opcode, const Args&... operands) {
		AppendInstruction(m_annotations, opcode, operands...);
	}

	template <typename... Args>
	void AddFunction(spv::Op opcode, const Args&... operands) {
		AppendInstruction(m_functions, opcode, operands...);
	}

	void        AddFunction(std::span<const uint32_t> words);
	DeferredPhi AddDeferredPhi(uint32_t type, uint32_t result, size_t incoming_count);
	void        PatchDeferredPhi(DeferredPhi phi, size_t incoming, uint32_t value, uint32_t parent);

	[[nodiscard]] std::vector<uint32_t> Build() const;

private:
	static void AppendOperand(std::vector<uint32_t>& words, uint32_t value) {
		words.push_back(value);
	}

	static void AppendOperand(std::vector<uint32_t>& words, int32_t value) {
		words.push_back(static_cast<uint32_t>(value));
	}

	template <typename T>
	requires std::is_enum_v<T>
	static void AppendOperand(std::vector<uint32_t>& words, T value) {
		static_assert(sizeof(T) == sizeof(uint32_t));
		words.push_back(static_cast<uint32_t>(value));
	}

	static void AppendOperand(std::vector<uint32_t>& words, std::span<const uint32_t> values) {
		words.insert(words.end(), values.begin(), values.end());
	}

	template <typename... Args>
	static void AppendOperands(std::vector<uint32_t>& words, const Args&... operands) {
		(AppendOperand(words, operands), ...);
	}

	template <typename T>
	static size_t OperandWordCount(const T& operand) {
		if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
			return 1;
		} else {
			return std::size(operand);
		}
	}

	template <typename... Args>
	void FillTypeKey(spv::Op opcode, const Args&... operands) {
		const auto operand_count = static_cast<uint32_t>((0u + ... + OperandWordCount(operands)));
		m_key_scratch.clear();
		AppendOperands(m_key_scratch, opcode, operand_count, operands...);
	}

	template <typename... Args>
	void AppendInstruction(std::vector<uint32_t>& section, spv::Op opcode,
	                       const Args&... operands) {
		const auto offset = section.size();
		AppendOperands(section, opcode, operands...);
		const auto word_count = static_cast<uint32_t>(section.size() - offset);
		section[offset] |= word_count << spv::WordCountShift;
		RecordAttribution(opcode, word_count);
	}

	struct DeclarationKeyHash {
		size_t operator()(const std::vector<uint32_t>& v) const noexcept {
			size_t h = v.size();
			for (uint32_t x: v) {
				h ^= static_cast<size_t>(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
			}
			return h;
		}
	};

	uint32_t    DeclareType(spv::Op opcode);
	uint32_t    DeclareDecoratedType(spv::Op opcode,
	                                 std::initializer_list<TypeAnnotation> annotations);
	uint32_t    DeclareConstant(spv::Op opcode);
	static void AppendString(std::vector<uint32_t>& words, const char* text);

	uint32_t                                  m_next_id = 1;
	uint32_t                                  m_version = 0;
	std::vector<uint32_t>                     m_capabilities;
	std::vector<uint32_t>                     m_extensions;
	std::vector<uint32_t>                     m_ext_inst_imports;
	std::vector<uint32_t>                     m_memory_model;
	std::vector<uint32_t>                     m_entry_points;
	std::vector<uint32_t>                     m_execution_modes;
	std::vector<uint32_t>                     m_debug;
	std::vector<uint32_t>                     m_annotations;
	std::vector<uint32_t>                     m_declarations;
	std::vector<uint32_t>                     m_functions;
	std::vector<uint32_t>                     m_key_scratch;
	std::set<spv::Capability>                 m_required_capabilities;
	std::set<std::string>                     m_required_extensions;
	std::map<std::string, uint32_t>           m_import_ids;
	std::unordered_map<std::vector<uint32_t>, uint32_t, DeclarationKeyHash> m_declaration_ids;
	size_t                                    m_unpatched_phi_incomings = 0;
	bool                                      m_emit_names              = false;
	EmitterAttributionTracker*                m_attribution_tracker     = nullptr;
	void RecordAttribution(spv::Op opcode, uint32_t word_count);
};

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVBUILDER_H_ */

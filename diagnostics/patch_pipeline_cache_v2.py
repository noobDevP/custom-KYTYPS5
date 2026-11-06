with open('src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. CompileModule: add ScopedCompilationTrack and dynamic_call_state check
t1 = '''	const auto profile = SpirvProfile(m_graphics);
	options.profile    = &profile;'''
r1 = '''	ScopedCompilationTrack comp_track("Shader", key.hash, 0, ShaderRecompiler::StageName(key.stage));
	const auto profile = SpirvProfile(m_graphics);
	options.profile    = &profile;'''
assert t1 in content, 't1 not found'
content = content.replace(t1, r1, 1)

t2 = '''	if (program.call_source == nullptr) {
		StoreModule(key, program, module, result.spirv);
	}'''
r2 = '''	if (program.call_source == nullptr && program.dynamic_call_state == nullptr) {
		StoreModule(key, program, module, result.spirv);
	}'''
assert t2 in content, 't2 not found'
content = content.replace(t2, r2, 1)

# 2. Instruction-aware ComputeTargetCodeHashes
t3 = '''static std::vector<uint64_t> ComputeTargetCodeHashes(const std::vector<uint64_t>& target_pcs) {
	std::vector<uint64_t> hashes;
	hashes.reserve(target_pcs.size());
	for (uint64_t target_pc : target_pcs) {
		std::vector<uint32_t> code_chunk(256);
		size_t read_words = 0;
		if (ReadShaderGuestMemory(nullptr, target_pc, code_chunk)) {
			for (size_t i = 0; i < code_chunk.size(); ++i) {
				const uint32_t w = code_chunk[i];
				read_words = i + 1;
				if ((w & 0xFFFF0000u) == 0xBF810000u || (w & 0xFFFF0000u) == 0xBE800000u) {
					break;
				}
			}
			hashes.push_back(XXH3_64bits(code_chunk.data(), read_words * sizeof(uint32_t)));
		} else {
			hashes.push_back(0);
		}
	}
	return hashes;
}'''
r3 = '''static std::vector<uint64_t> ComputeTargetCodeHashes(const std::vector<uint64_t>& target_pcs) {
	std::vector<uint64_t> hashes;
	hashes.reserve(target_pcs.size());
	for (uint64_t target_pc : target_pcs) {
		std::vector<uint32_t> callee_words;
		for (uint32_t index = 0;;) {
			std::array<uint32_t, ShaderRecompiler::Decoder::MaxInstructionRawWords> words {};
			size_t count = 1;
			ShaderRecompiler::Decoder::Instruction inst {};
			bool read_ok = false;
			for (;;) {
				if (!ReadShaderGuestMemory(nullptr, target_pc + uint64_t{index} * sizeof(uint32_t),
				                           std::span(words).first(count))) {
					break;
				}
				inst = {};
				ShaderRecompiler::Decoder::DecodeInstruction(words, 0, inst);
				if (inst.word_count <= count) {
					read_ok = true;
					break;
				}
				count = inst.word_count;
			}
			if (!read_ok || inst.opcode == ShaderRecompiler::Decoder::Opcode::UNKNOWN || inst.word_count == 0) {
				break;
			}
			for (size_t w = 0; w < inst.word_count; ++w) {
				callee_words.push_back(words[w]);
			}
			index += inst.word_count;
			if (inst.opcode == ShaderRecompiler::Decoder::Opcode::S_SETPC_B64 ||
			    inst.opcode == ShaderRecompiler::Decoder::Opcode::S_ENDPGM) {
				break;
			}
			if (index > 4096) break;
		}
		if (!callee_words.empty()) {
			hashes.push_back(XXH3_64bits(callee_words.data(), callee_words.size() * sizeof(uint32_t)));
		} else {
			hashes.push_back(0);
		}
	}
	return hashes;
}'''
assert t3 in content, 't3 not found'
content = content.replace(t3, r3, 1)

# 3. cached.from_disk handling in GetProgram
t4 = '''		if (cached.call_source) {
			source_changed = cached.RefreshCallSource(runtime);
			if (!source_changed) module = cached.FindModule<true>(runtime, push_data_cursor);
		} else if (cached.dynamic_call_state) {'''
r4 = '''		if (cached.from_disk) {
			source = ShaderRecompiler::PrepareShaderSource(params.code, options);
			if (source.unsupported_call_pc != UINT32_MAX || !source.calls.empty()) {
				for (const auto& mod : cached.modules) {
					if (mod.module) m_graphics.device.destroyShaderModule(mod.module, nullptr);
				}
				cached.modules.clear();
				cached.from_disk = false;
			} else {
				cached.from_disk = false;
				module = cached.FindModule<false>(runtime, push_data_cursor);
			}
		} else if (cached.call_source) {
			source_changed = cached.RefreshCallSource(runtime);
			if (!source_changed) module = cached.FindModule<true>(runtime, push_data_cursor);
		} else if (cached.dynamic_call_state) {'''
assert t4 in content, 't4 not found'
content = content.replace(t4, r4, 1)

# 4. entry == m_program_cache.end() || entry->second.dynamic_call_state == nullptr
t5 = '''	if (entry == m_program_cache.end()) {
		DumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
		                                    ShaderRecompiler::StageName(options.stage), params.hash));
		source = ShaderRecompiler::PrepareShaderSource(params.code, options);'''
r5 = '''	if (entry == m_program_cache.end() || entry->second.dynamic_call_state == nullptr) {
		DumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
		                                    ShaderRecompiler::StageName(options.stage), params.hash));
		if (source.decoded.instructions.empty()) {
			source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		}'''
assert t5 in content, 't5 not found'
content = content.replace(t5, r5, 1)

# 5. Relink failure in dynamic_call_state handling
t6 = '''			} else {
				dynamic_link_result = "relink_failed";
				translation_path = "relink_failed_fallback";
			}'''
r6 = '''			} else {
				dyn_state.source.reset();
				dyn_state.target_pcs.clear();
				dyn_state.target_code_hashes.clear();
				entry->second.modules.clear();
				dynamic_link_result = "relink_failed";
				std::vector<uint32_t> patched_code(params.code.begin(), params.code.end());
				while (source.unsupported_call_pc != UINT32_MAX) {
					const auto pc = source.unsupported_call_pc;
					const uint32_t word_idx = pc / sizeof(uint32_t);
					if (word_idx >= patched_code.size()) break;
					const auto found_inst = std::ranges::find_if(source.decoded.instructions, [pc](const auto& inst) {
						return inst.pc == pc;
					});
					const uint32_t count = (found_inst != source.decoded.instructions.end() && found_inst->word_count > 0)
					                           ? found_inst->word_count : 1u;
					for (uint32_t w = 0; w < count && (word_idx + w) < patched_code.size(); ++w) {
						patched_code[word_idx + w] = 0xbf800000u;
					}
					source = ShaderRecompiler::PrepareShaderSource(patched_code, options);
				}
				translation_path = "relink_failed_stubbed_nop";
			}'''
assert t6 in content, 't6 not found'
content = content.replace(t6, r6, 1)

# 6. Current source selection
t7 = '''	const ShaderRecompiler::ShaderSource* current = nullptr;
	if (entry == m_program_cache.end()) {
		current = new_dynamic_state ? new_dynamic_state->source.get() : &source;
	} else if (entry->second.dynamic_call_state != nullptr && entry->second.dynamic_call_state->source != nullptr) {
		current = entry->second.dynamic_call_state->source.get();
	} else if (entry->second.call_source != nullptr) {
		current = entry->second.call_source.get();
		translation_path = "static_call_source";
	}'''
r7 = '''	const ShaderRecompiler::ShaderSource* current = nullptr;
	if (entry == m_program_cache.end()) {
		current = new_dynamic_state ? new_dynamic_state->source.get() : &source;
	} else if (entry->second.dynamic_call_state != nullptr && entry->second.dynamic_call_state->source != nullptr) {
		current = entry->second.dynamic_call_state->source.get();
	} else if (dynamic_link_result == std::string_view("relink_failed")) {
		current = &source;
	} else if (entry->second.call_source != nullptr) {
		current = entry->second.call_source.get();
		translation_path = "static_call_source";
	} else if (new_dynamic_state) {
		current = new_dynamic_state->source.get();
	}'''
assert t7 in content, 't7 not found'
content = content.replace(t7, r7, 1)

# 7. Program emplace / update
t8 = '''	if (entry == m_program_cache.end()) {
		auto retained = (new_dynamic_state == nullptr && !source.calls.empty())
		                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
		                    : nullptr;
		entry = m_program_cache.emplace(m_lookup_key, Program {
		    .call_source        = std::move(retained),
		    .dynamic_call_state = std::move(new_dynamic_state),
		    .resource_plan      = ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
		}).first;
		auto& program = entry->second;
		EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
		    program.resource_plan, runtime, program.resources, program.specialization));
	} else if (source_changed) {
		entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		if (entry->second.call_source) {
			if (const auto* module = entry->second.FindModule<true>(runtime, push_data_cursor)) {
				return bind(*module);
			}
		} else {
			if (const auto* module = entry->second.FindModule<false>(runtime, push_data_cursor)) {
				return bind(*module);
			}
		}
	}'''
r8 = '''	if (entry == m_program_cache.end()) {
		auto retained = (new_dynamic_state == nullptr && !source.calls.empty())
		                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
		                    : nullptr;
		entry = m_program_cache.emplace(m_lookup_key, Program {
		    .call_source        = std::move(retained),
		    .dynamic_call_state = std::move(new_dynamic_state),
		    .resource_plan      = ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
		}).first;
		auto& program = entry->second;
		EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
		    program.resource_plan, runtime, program.resources, program.specialization));
	} else {
		if (new_dynamic_state != nullptr) {
			entry->second.dynamic_call_state = std::move(new_dynamic_state);
		}
		if (source_changed || is_spec_miss) {
			entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			if (entry->second.call_source) {
				if (const auto* module = entry->second.FindModule<true>(runtime, push_data_cursor)) {
					return bind(*module);
				}
			} else {
				if (const auto* module = entry->second.FindModule<false>(runtime, push_data_cursor)) {
					return bind(*module);
				}
			}
		}
	}'''
assert t8 in content, 't8 not found'
content = content.replace(t8, r8, 1)

with open('src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated pipelineCache.cpp successfully!')

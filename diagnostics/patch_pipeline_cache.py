import re

path = "src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update CompileModule to include dynamic call target hashes in module inputs
marker1 = "inputs.Raw(function_code.data(), function_code.size_bytes()); // Last, so it needs no size."
replacement1 = """inputs.Raw(function_code.data(), function_code.size_bytes()); // Last, so it needs no size.
\tif (program.dynamic_call_state) {
\t\tinputs.Raw(program.dynamic_call_state->target_pcs.data(),
\t\t           program.dynamic_call_state->target_pcs.size() * sizeof(uint64_t));
\t\tinputs.Raw(program.dynamic_call_state->target_code_hashes.data(),
\t\t           program.dynamic_call_state->target_code_hashes.size() * sizeof(uint64_t));
\t}"""

assert marker1 in content, "marker1 not found"
content = content.replace(marker1, replacement1, 1)

# 2. Add helper functions before PipelineCache::GetProgram
marker2 = "template <typename InputInfo>\nconst ShaderModule* PipelineCache::GetProgram(const ShaderParams& params, InputInfo& input_info,"
helpers = """static std::vector<uint64_t> ReadSbtTargetPcs(std::span<const uint32_t> user_data) {
\tstd::vector<uint64_t> unique_target_pcs;
\tif (user_data.size() < 14) {
\t\treturn unique_target_pcs;
\t}
\tconst uint64_t s12_ptr = (static_cast<uint64_t>(user_data[13]) << 32) | user_data[12];
\tstd::array<uint32_t, 4> sbt_desc_words {};
\tif (!ReadShaderGuestMemory(nullptr, s12_ptr + 144, sbt_desc_words)) {
\t\treturn unique_target_pcs;
\t}
\tShaderBufferResource sbt_desc;
\tstd::copy(sbt_desc_words.begin(), sbt_desc_words.end(), sbt_desc.fields);
\tconst uint64_t sbt_base = sbt_desc.Base48();
\tconst uint32_t sbt_size = sbt_desc.NumRecords();
\tconst uint32_t total_records = sbt_size / 16u;
\tfor (uint32_t i = 0; i < total_records; ++i) {
\t\tstd::array<uint32_t, 4> record {};
\t\tif (ReadShaderGuestMemory(nullptr, sbt_base + i * 16u, record)) {
\t\t\tconst uint64_t target_pc = (static_cast<uint64_t>(record[1]) << 32) | record[0];
\t\t\tif (target_pc != 0) {
\t\t\t\tconst uint64_t clean_target = target_pc & ~3ull;
\t\t\t\tif (std::ranges::find(unique_target_pcs, clean_target) == unique_target_pcs.end()) {
\t\t\t\t\tunique_target_pcs.push_back(clean_target);
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\treturn unique_target_pcs;
}

static std::vector<uint64_t> ComputeTargetCodeHashes(const std::vector<uint64_t>& target_pcs) {
\tstd::vector<uint64_t> hashes;
\thashes.reserve(target_pcs.size());
\tfor (uint64_t target_pc : target_pcs) {
\t\tstd::vector<uint32_t> code_chunk(256);
\t\tsize_t read_words = 0;
\t\tif (ReadShaderGuestMemory(nullptr, target_pc, code_chunk)) {
\t\t\tfor (size_t i = 0; i < code_chunk.size(); ++i) {
\t\t\t\tconst uint32_t w = code_chunk[i];
\t\t\t\tread_words = i + 1;
\t\t\t\tif ((w & 0xFFFF0000u) == 0xBF810000u || (w & 0xFFFF0000u) == 0xBE800000u) {
\t\t\t\t\tbreak;
\t\t\t\t}
\t\t\t}
\t\t\thashes.push_back(XXH3_64bits(code_chunk.data(), read_words * sizeof(uint32_t)));
\t\t} else {
\t\t\thashes.push_back(0);
\t\t}
\t}
\treturn hashes;
}

static void LogDynamicCallRetentionDiagnostic(uint64_t shader_hash, ShaderType stage,
                                              bool cache_hit, bool spec_miss,
                                              const char* link_result, size_t target_count,
                                              const char* translation_path, uint32_t call_offset) {
\tstatic std::atomic<uint32_t> s_diag_count {0};
\tconst uint32_t count = s_diag_count.fetch_add(1, std::memory_order_relaxed);
\tconst bool is_target_shader = (shader_hash == 0x1a04c58a9fbc2052ull);
\tif (count < 64 || is_target_shader) {
\t\tLog::WriteToConsoleAndLog(fmt::format(
\t\t    "[DynamicCallDiag #{}] shader=0x{:016x} stage={} cache={} spec_miss={} link={} targets={} trans_path={} call_offset=0x{:04x}\\n",
\t\t    count, shader_hash, ShaderRecompiler::StageName(stage),
\t\t    cache_hit ? "HIT" : "MISS", spec_miss ? "YES" : "NO",
\t\t    link_result, target_count, translation_path, call_offset));
\t}
}

"""

assert marker2 in content, "marker2 not found"
content = content.replace(marker2, helpers + marker2, 1)

# 3. Update GetProgram cache lookup and dynamic call retention/re-link
old_lookup_block = """\tbool source_changed = false;
\tif (entry != m_program_cache.end()) {
\t\tauto& cached = entry->second;
\t\tif (cached.skip_dispatch) return nullptr;
\t\tShaderModule* module = nullptr;
\t\tif (cached.call_source) {
\t\t\tsource_changed = cached.RefreshCallSource(runtime);
\t\t\tif (!source_changed) module = cached.FindModule<true>(runtime, push_data_cursor);
\t\t} else {
\t\t\tmodule = cached.FindModule<false>(runtime, push_data_cursor);
\t\t}
\t\tif (module != nullptr) return bind(*module);
\t}"""

new_lookup_block = """\tbool source_changed = false;
\tbool is_spec_miss = false;
\tif (entry != m_program_cache.end()) {
\t\tauto& cached = entry->second;
\t\tif (cached.skip_dispatch) return nullptr;
\t\tShaderModule* module = nullptr;
\t\tif (cached.call_source) {
\t\t\tsource_changed = cached.RefreshCallSource(runtime);
\t\t\tif (!source_changed) module = cached.FindModule<true>(runtime, push_data_cursor);
\t\t} else if (cached.dynamic_call_state) {
\t\t\tconst auto current_targets = ReadSbtTargetPcs(user_data);
\t\t\tconst auto current_hashes  = ComputeTargetCodeHashes(current_targets);
\t\t\tif (current_targets != cached.dynamic_call_state->target_pcs ||
\t\t\t    current_hashes != cached.dynamic_call_state->target_code_hashes) {
\t\t\t\tsource_changed = true;
\t\t\t} else {
\t\t\t\tmodule = cached.FindModule<false>(runtime, push_data_cursor);
\t\t\t}
\t\t} else {
\t\t\tmodule = cached.FindModule<false>(runtime, push_data_cursor);
\t\t}
\t\tif (module != nullptr) return bind(*module);
\t\tis_spec_miss = true;
\t}"""

assert old_lookup_block in content, "old_lookup_block not found"
content = content.replace(old_lookup_block, new_lookup_block, 1)

# 4. Update the translation and Program storage section
old_trans_block = """\toptions.shader_hash     = params.hash;
\toptions.user_data       = user_data;
\toptions.back_code       = params.back_code;
\toptions.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
\toptions.emit_debug_info = Config::GraphicsDebugDumpEnabled();
\tShaderRecompiler::ShaderSource source;
\tif (entry == m_program_cache.end()) {
\t\tDumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
\t\t                                    ShaderRecompiler::StageName(options.stage), params.hash));
\t\tsource = ShaderRecompiler::PrepareShaderSource(params.code, options);
\t\tif (source.unsupported_call_pc != UINT32_MAX) {
\t\t\tif (user_data.size() >= 14) {
\t\t\t\tconst uint64_t s12_ptr = (static_cast<uint64_t>(user_data[13]) << 32) | user_data[12];
\t\t\t\tstd::array<uint32_t, 4> sbt_desc_words {};
\t\t\t\tif (ReadShaderGuestMemory(nullptr, s12_ptr + 144, sbt_desc_words)) {
\t\t\t\t\tShaderBufferResource sbt_desc;
\t\t\t\t\tstd::copy(sbt_desc_words.begin(), sbt_desc_words.end(), sbt_desc.fields);
\t\t\t\t\tconst uint64_t sbt_base = sbt_desc.Base48();
\t\t\t\t\tconst uint32_t sbt_size = sbt_desc.NumRecords();
\t\t\t\t\tstd::vector<uint64_t> unique_target_pcs;
\t\t\t\t\tconst uint32_t total_records = sbt_size / 16u;
\t\t\t\t\tfor (uint32_t i = 0; i < total_records; ++i) {
\t\t\t\t\t\tstd::array<uint32_t, 4> record {};
\t\t\t\t\t\tif (ReadShaderGuestMemory(nullptr, sbt_base + i * 16u, record)) {
\t\t\t\t\t\t\tconst uint64_t target_pc = (static_cast<uint64_t>(record[1]) << 32) | record[0];
\t\t\t\t\t\t\tif (target_pc != 0) {
\t\t\t\t\t\t\t\tconst uint64_t clean_target = target_pc & ~3ull;
\t\t\t\t\t\t\t\tif (std::ranges::find(unique_target_pcs, clean_target) == unique_target_pcs.end()) {
\t\t\t\t\t\t\t\t\tunique_target_pcs.push_back(clean_target);
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t\tif (!unique_target_pcs.empty()) {
\t\t\t\t\t\tPipelineCacheLog("Shader 0x{:016x}: linking dynamic call table with {} targets from SBT",
\t\t\t\t\t\t                 params.hash, unique_target_pcs.size());
\t\t\t\t\t\tif (ShaderRecompiler::LinkDynamicCallTargets(source, unique_target_pcs, runtime,
\t\t\t\t\t\t                                             params.hash)) {
\t\t\t\t\t\t\tPipelineCacheLog("Shader 0x{:016x}: successfully linked dynamic call table! Dispatch will execute real material shaders on GPU.",
\t\t\t\t\t\t                 params.hash);
\t\t\t\t\t\t\tstd::string disasm = ShaderRecompiler::Decoder::ProgramToString(source.decoded);
\t\t\t\t\t\t\tCommon::File disasm_file(fmt::format("CS_{:016x}_linked_disasm.txt", params.hash));
\t\t\t\t\t\t\tif (!disasm_file.IsInvalid()) {
\t\t\t\t\t\t\t\tdisasm_file.Write(disasm.data(), static_cast<uint32_t>(disasm.size()));
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\tstd::vector<uint32_t> patched_code(params.code.begin(), params.code.end());
\t\t\twhile (source.unsupported_call_pc != UINT32_MAX) {
\t\t\t\tconst auto pc = source.unsupported_call_pc;
\t\t\t\tPipelineCacheLog("Shader 0x{:016x} pc=0x{:08x}: stubbing GPU-selected callable target with S_NOP "
\t\t\t\t                 "to allow compute dispatch execution",
\t\t\t\t                 params.hash, pc);
\t\t\t\tfor (const auto& inst: source.decoded.instructions) {
\t\t\t\t\tif (inst.pc >= (pc > 32 ? pc - 32 : 0) && inst.pc <= pc + 32) {
\t\t\t\t\t\tPipelineCacheLog("    pc=0x{:08x}: {}", inst.pc, ShaderRecompiler::Decoder::InstructionToString(inst));
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tstd::string disasm = ShaderRecompiler::Decoder::ProgramToString(source.decoded);
\t\t\t\tCommon::File disasm_file(fmt::format("CS_{:016x}_disasm.txt", params.hash));
\t\t\t\tif (!disasm_file.IsInvalid()) {
\t\t\t\t\tdisasm_file.Write(disasm.data(), static_cast<uint32_t>(disasm.size()));
\t\t\t\t}
\t\t\t\tconst uint32_t word_idx = pc / sizeof(uint32_t);
\t\t\t\tif (word_idx >= patched_code.size()) {
\t\t\t\t\tbreak;
\t\t\t\t}
\t\t\t\tconst auto found_inst = std::ranges::find_if(source.decoded.instructions, [pc](const auto& inst) {
\t\t\t\t\treturn inst.pc == pc;
\t\t\t\t});
\t\t\t\tconst uint32_t count = (found_inst != source.decoded.instructions.end() && found_inst->word_count > 0)
\t\t\t\t                           ? found_inst->word_count : 1u;
\t\t\t\tfor (uint32_t w = 0; w < count && (word_idx + w) < patched_code.size(); ++w) {
\t\t\t\t\t// 0xbf800000u is GCN/RDNA2 's_nop 0' (SOPP opcode 0)
\t\t\t\t\tpatched_code[word_idx + w] = 0xbf800000u;
\t\t\t\t}
\t\t\t\tsource = ShaderRecompiler::PrepareShaderSource(patched_code, options);
\t\t\t}
\t\t}
\t}
\tauto* current = entry == m_program_cache.end() ? &source : entry->second.call_source.get();
\tShaderRecompiler::TranslateResult translated;
\tif (current != nullptr) {
\t\tif (!current->calls.empty()) {
\t\t\tif (entry == m_program_cache.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
\t\t\ttranslated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
\t\t\ttranslated.program.source_reads = current->reads;
\t\t} else {
\t\t\ttranslated = ShaderRecompiler::TranslateProgram(current->decoded, options);
\t\t}
\t} else {
\t\ttranslated = ShaderRecompiler::TranslateProgram(params.code, options);
\t}
\tif (entry == m_program_cache.end()) {
\t\tauto retained = !source.calls.empty()
\t\t                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
\t\t                    : nullptr;
\t\tentry = m_program_cache.emplace(m_lookup_key, Program {
\t\t    .call_source   = std::move(retained),
\t\t    .resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
\t\t}).first;
\t\tauto& program = entry->second;
\t\tEXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
\t\t    program.resource_plan, runtime, program.resources, program.specialization));
\t} else if (source_changed) {
\t\tentry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
\t\tif (const auto* module = entry->second.FindModule<true>(runtime, push_data_cursor)) {
\t\t\treturn bind(*module);
\t\t}
\t}"""

new_trans_block = """\toptions.shader_hash     = params.hash;
\toptions.user_data       = user_data;
\toptions.back_code       = params.back_code;
\toptions.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
\toptions.emit_debug_info = Config::GraphicsDebugDumpEnabled();
\tShaderRecompiler::ShaderSource source;
\tstd::unique_ptr<Program::DynamicCallState> new_dynamic_state;
\tconst char* dynamic_link_result = "none";
\tconst char* translation_path = "direct_params_code";
\tsize_t dynamic_target_count = 0;
\tuint32_t call_offset = UINT32_MAX;

\tif (entry == m_program_cache.end()) {
\t\tDumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
\t\t                                    ShaderRecompiler::StageName(options.stage), params.hash));
\t\tsource = ShaderRecompiler::PrepareShaderSource(params.code, options);
\t\tif (source.unsupported_call_pc != UINT32_MAX) {
\t\t\tcall_offset = source.unsupported_call_pc;
\t\t\tconst auto target_pcs = ReadSbtTargetPcs(user_data);
\t\t\tdynamic_target_count = target_pcs.size();
\t\t\tif (!target_pcs.empty()) {
\t\t\t\tPipelineCacheLog("Shader 0x{:016x}: linking dynamic call table with {} targets from SBT",
\t\t\t\t                 params.hash, target_pcs.size());
\t\t\t\tif (ShaderRecompiler::LinkDynamicCallTargets(source, target_pcs, runtime, params.hash)) {
\t\t\t\t\tPipelineCacheLog("Shader 0x{:016x}: successfully linked dynamic call table! Dispatch will execute real material shaders on GPU.",
\t\t\t\t\t                 params.hash);
\t\t\t\t\tdynamic_link_result = "success";
\t\t\t\t\ttranslation_path = "initial_dynamic_link";
\t\t\t\t\tconst auto target_hashes = ComputeTargetCodeHashes(target_pcs);
\t\t\t\t\tnew_dynamic_state = std::make_unique<Program::DynamicCallState>(Program::DynamicCallState {
\t\t\t\t\t    .source = std::make_unique<ShaderRecompiler::ShaderSource>(source),
\t\t\t\t\t    .target_pcs = target_pcs,
\t\t\t\t\t    .target_code_hashes = target_hashes,
\t\t\t\t\t    .call_pc = call_offset,
\t\t\t\t\t});
\t\t\t\t\tstd::string disasm = ShaderRecompiler::Decoder::ProgramToString(source.decoded);
\t\t\t\t\tCommon::File disasm_file(fmt::format("CS_{:016x}_linked_disasm.txt", params.hash));
\t\t\t\t\tif (!disasm_file.IsInvalid()) {
\t\t\t\t\t\tdisasm_file.Write(disasm.data(), static_cast<uint32_t>(disasm.size()));
\t\t\t\t\t}
\t\t\t\t} else {
\t\t\t\t\tdynamic_link_result = "link_failed";
\t\t\t\t}
\t\t\t} else {
\t\t\t\tdynamic_link_result = "no_targets";
\t\t\t}
\t\t\tstd::vector<uint32_t> patched_code(params.code.begin(), params.code.end());
\t\t\twhile (source.unsupported_call_pc != UINT32_MAX) {
\t\t\t\tconst auto pc = source.unsupported_call_pc;
\t\t\t\tPipelineCacheLog("Shader 0x{:016x} pc=0x{:08x}: stubbing GPU-selected callable target with S_NOP "
\t\t\t\t                 "to allow compute dispatch execution",
\t\t\t\t                 params.hash, pc);
\t\t\t\tfor (const auto& inst: source.decoded.instructions) {
\t\t\t\t\tif (inst.pc >= (pc > 32 ? pc - 32 : 0) && inst.pc <= pc + 32) {
\t\t\t\t\t\tPipelineCacheLog("    pc=0x{:08x}: {}", inst.pc, ShaderRecompiler::Decoder::InstructionToString(inst));
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tstd::string disasm = ShaderRecompiler::Decoder::ProgramToString(source.decoded);
\t\t\t\tCommon::File disasm_file(fmt::format("CS_{:016x}_disasm.txt", params.hash));
\t\t\t\tif (!disasm_file.IsInvalid()) {
\t\t\t\t\tdisasm_file.Write(disasm.data(), static_cast<uint32_t>(disasm.size()));
\t\t\t\t}
\t\t\t\tconst uint32_t word_idx = pc / sizeof(uint32_t);
\t\t\t\tif (word_idx >= patched_code.size()) {
\t\t\t\t\tbreak;
\t\t\t\t}
\t\t\t\tconst auto found_inst = std::ranges::find_if(source.decoded.instructions, [pc](const auto& inst) {
\t\t\t\t\treturn inst.pc == pc;
\t\t\t\t});
\t\t\t\tconst uint32_t count = (found_inst != source.decoded.instructions.end() && found_inst->word_count > 0)
\t\t\t\t                           ? found_inst->word_count : 1u;
\t\t\t\tfor (uint32_t w = 0; w < count && (word_idx + w) < patched_code.size(); ++w) {
\t\t\t\t\t// 0xbf800000u is GCN/RDNA2 's_nop 0' (SOPP opcode 0)
\t\t\t\t\tpatched_code[word_idx + w] = 0xbf800000u;
\t\t\t\t}
\t\t\t\tsource = ShaderRecompiler::PrepareShaderSource(patched_code, options);
\t\t\t}
\t\t\tif (new_dynamic_state == nullptr) {
\t\t\t\ttranslation_path = "direct_stubbed_nop";
\t\t\t}
\t\t}
\t} else if (entry->second.dynamic_call_state != nullptr) {
\t\tauto& dyn_state = *entry->second.dynamic_call_state;
\t\tcall_offset = dyn_state.call_pc;
\t\tdynamic_target_count = dyn_state.target_pcs.size();
\t\tif (source_changed) {
\t\t\tsource = ShaderRecompiler::PrepareShaderSource(params.code, options);
\t\t\tconst auto target_pcs = ReadSbtTargetPcs(user_data);
\t\t\tdynamic_target_count = target_pcs.size();
\t\t\tif (!target_pcs.empty() && ShaderRecompiler::LinkDynamicCallTargets(source, target_pcs, runtime, params.hash)) {
\t\t\t\tconst auto target_hashes = ComputeTargetCodeHashes(target_pcs);
\t\t\t\tdyn_state.source = std::make_unique<ShaderRecompiler::ShaderSource>(source);
\t\t\t\tdyn_state.target_pcs = target_pcs;
\t\t\t\tdyn_state.target_code_hashes = target_hashes;
\t\t\t\tentry->second.modules.clear();
\t\t\t\tdynamic_link_result = "relinked";
\t\t\t\ttranslation_path = "dynamic_source_relinked";
\t\t\t} else {
\t\t\t\tdynamic_link_result = "relink_failed";
\t\t\t\ttranslation_path = "relink_failed_fallback";
\t\t\t}
\t\t} else {
\t\t\tdynamic_link_result = "reused";
\t\t\ttranslation_path = "dynamic_source_reused";
\t\t}
\t}

\tconst ShaderRecompiler::ShaderSource* current = nullptr;
\tif (entry == m_program_cache.end()) {
\t\tcurrent = &source;
\t} else if (entry->second.dynamic_call_state != nullptr && entry->second.dynamic_call_state->source != nullptr) {
\t\tcurrent = entry->second.dynamic_call_state->source.get();
\t} else if (entry->second.call_source != nullptr) {
\t\tcurrent = entry->second.call_source.get();
\t\ttranslation_path = "static_call_source";
\t}

\tShaderRecompiler::TranslateResult translated;
\tif (current != nullptr) {
\t\tif (!current->calls.empty()) {
\t\t\tif (entry == m_program_cache.end()) ShaderRecompiler::RefreshShaderSource(*const_cast<ShaderRecompiler::ShaderSource*>(current), runtime);
\t\t\ttranslated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
\t\t\ttranslated.program.source_reads = current->reads;
\t\t} else {
\t\t\ttranslated = ShaderRecompiler::TranslateProgram(current->decoded, options);
\t\t}
\t} else {
\t\ttranslated = ShaderRecompiler::TranslateProgram(params.code, options);
\t}

\tLogDynamicCallRetentionDiagnostic(params.hash, options.stage,
\t                                  /*cache_hit=*/entry != m_program_cache.end(),
\t                                  /*spec_miss=*/is_spec_miss,
\t                                  dynamic_link_result, dynamic_target_count,
\t                                  translation_path, call_offset);

\tif (entry == m_program_cache.end()) {
\t\tauto retained = !source.calls.empty()
\t\t                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
\t\t                    : nullptr;
\t\tentry = m_program_cache.emplace(m_lookup_key, Program {
\t\t    .call_source        = std::move(retained),
\t\t    .dynamic_call_state = std::move(new_dynamic_state),
\t\t    .resource_plan      = ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
\t\t}).first;
\t\tauto& program = entry->second;
\t\tEXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
\t\t    program.resource_plan, runtime, program.resources, program.specialization));
\t} else if (source_changed) {
\t\tentry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
\t\tif (entry->second.call_source) {
\t\t\tif (const auto* module = entry->second.FindModule<true>(runtime, push_data_cursor)) {
\t\t\t\treturn bind(*module);
\t\t\t}
\t\t} else {
\t\t\tif (const auto* module = entry->second.FindModule<false>(runtime, push_data_cursor)) {
\t\t\t\treturn bind(*module);
\t\t\t}
\t\t}
\t}"""

assert old_trans_block in content, "old_trans_block not found"
content = content.replace(old_trans_block, new_trans_block, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Pipeline cache patched successfully!")

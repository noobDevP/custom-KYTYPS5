path = "src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace lines 530-660
idx_start = content.find("if (entry == m_program_cache.end()) {\n\t\tDumpShader(params.code, fmt::format(\"original/{}_{:016x}.bin\"")
assert idx_start != -1, "idx_start not found"

idx_end = content.find("EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(", idx_start)
assert idx_end != -1, "idx_end not found"

# Find the end of that block up to the start of "auto& program = entry->second;"
idx_emplace = content.find("auto& program = entry->second;", idx_start)
assert idx_emplace != -1, "idx_emplace not found"

old_chunk = content[idx_start:idx_emplace]

new_chunk = """if (entry == m_program_cache.end()) {
\t\tDumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
\t\t                                    ShaderRecompiler::StageName(options.stage), params.hash));
\t\tsource = ShaderRecompiler::PrepareShaderSource(params.code, options);
\t\tif (source.unsupported_call_pc != UINT32_MAX) {
\t\t\tcall_offset = source.unsupported_call_pc;
\t\t\tconst auto target_pcs = ReadSbtTargetPcs(user_data);
\t\t\tdynamic_target_count = target_pcs.size();
\t\t\tbool linked = false;
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
\t\t\t\t\t    .source = std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)),
\t\t\t\t\t    .target_pcs = target_pcs,
\t\t\t\t\t    .target_code_hashes = target_hashes,
\t\t\t\t\t    .call_pc = call_offset,
\t\t\t\t\t});
\t\t\t\t\tstd::string disasm = ShaderRecompiler::Decoder::ProgramToString(new_dynamic_state->source->decoded);
\t\t\t\t\tCommon::File disasm_file(fmt::format("CS_{:016x}_linked_disasm.txt", params.hash));
\t\t\t\t\tif (!disasm_file.IsInvalid()) {
\t\t\t\t\t\tdisasm_file.Write(disasm.data(), static_cast<uint32_t>(disasm.size()));
\t\t\t\t\t}
\t\t\t\t\tlinked = true;
\t\t\t\t} else {
\t\t\t\t\tdynamic_link_result = "link_failed";
\t\t\t\t}
\t\t\t} else {
\t\t\t\tdynamic_link_result = "no_targets";
\t\t\t}
\t\t\tif (!linked) {
\t\t\t\tstd::vector<uint32_t> patched_code(params.code.begin(), params.code.end());
\t\t\t\twhile (source.unsupported_call_pc != UINT32_MAX) {
\t\t\t\t\tconst auto pc = source.unsupported_call_pc;
\t\t\t\t\tPipelineCacheLog("Shader 0x{:016x} pc=0x{:08x}: stubbing GPU-selected callable target with S_NOP "
\t\t\t\t\t                 "to allow compute dispatch execution",
\t\t\t\t\t                 params.hash, pc);
\t\t\t\t\tfor (const auto& inst: source.decoded.instructions) {
\t\t\t\t\t\tif (inst.pc >= (pc > 32 ? pc - 32 : 0) && inst.pc <= pc + 32) {
\t\t\t\t\t\t\tPipelineCacheLog("    pc=0x{:08x}: {}", inst.pc, ShaderRecompiler::Decoder::InstructionToString(inst));
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t\tstd::string disasm = ShaderRecompiler::Decoder::ProgramToString(source.decoded);
\t\t\t\t\tCommon::File disasm_file(fmt::format("CS_{:016x}_disasm.txt", params.hash));
\t\t\t\t\tif (!disasm_file.IsInvalid()) {
\t\t\t\t\t\tdisasm_file.Write(disasm.data(), static_cast<uint32_t>(disasm.size()));
\t\t\t\t\t}
\t\t\t\t\tconst uint32_t word_idx = pc / sizeof(uint32_t);
\t\t\t\t\tif (word_idx >= patched_code.size()) {
\t\t\t\t\t\tbreak;
\t\t\t\t\t}
\t\t\t\t\tconst auto found_inst = std::ranges::find_if(source.decoded.instructions, [pc](const auto& inst) {
\t\t\t\t\t\treturn inst.pc == pc;
\t\t\t\t\t});
\t\t\t\t\tconst uint32_t count = (found_inst != source.decoded.instructions.end() && found_inst->word_count > 0)
\t\t\t\t\t                           ? found_inst->word_count : 1u;
\t\t\t\t\tfor (uint32_t w = 0; w < count && (word_idx + w) < patched_code.size(); ++w) {
\t\t\t\t\t\t// 0xbf800000u is GCN/RDNA2 's_nop 0' (SOPP opcode 0)
\t\t\t\t\t\tpatched_code[word_idx + w] = 0xbf800000u;
\t\t\t\t\t}
\t\t\t\t\tsource = ShaderRecompiler::PrepareShaderSource(patched_code, options);
\t\t\t\t}
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
\t\t\t\tdyn_state.source = std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source));
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
\t\tcurrent = new_dynamic_state ? new_dynamic_state->source.get() : &source;
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
\t\tauto retained = (new_dynamic_state == nullptr && !source.calls.empty())
\t\t                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
\t\t                    : nullptr;
\t\tentry = m_program_cache.emplace(m_lookup_key, Program {
\t\t    .call_source        = std::move(retained),
\t\t    .dynamic_call_state = std::move(new_dynamic_state),
\t\t    .resource_plan      = ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
\t\t}).first;
\t\t"""

content = content[:idx_start] + new_chunk + content[idx_emplace:]

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated pipelineCache.cpp with move semantics successfully!")

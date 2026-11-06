path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Remove the std::cout lines
content = content.replace('    std::cout << "[DEBUG] Entered CheckDynamicCallBranchRelocation" << std::endl;\n', '')
content = content.replace('std::cout << "[DEBUG] After ShaderMapUserData" << std::endl;\n    ', '')

# Now let's update the direct linking section in CheckDynamicCallBranchRelocation
old_linking_section = '''    // 1. Direct linking and CFG verification:
    ShaderRecompiler::CompileOptions options{.wave_size = 64};
    options.stage = ShaderType::Compute;
    auto source = ShaderRecompiler::PrepareShaderSource(code, options);
    auto read_mem = [](void*, uint64_t address, std::span<uint32_t> values) {
      return !values.empty() && LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
    };
    ShaderRecompiler::IR::SrtRuntime runtime{
        .read_specialization_memory = read_mem,
    };
    const uint64_t callee_target = callee_addr;
    const bool link_ok = ShaderRecompiler::LinkDynamicCallTargets(
        source, std::span<const uint64_t>(&callee_target, 1), runtime, 0xd1ca11ull);
    Require(name, "dynamic linking succeeded", link_ok, "LinkDynamicCallTargets failed");'''

new_linking_section = '''    // 1. Direct linking and CFG verification:
    ShaderRecompiler::CompileOptions options{.wave_size = 64};
    options.stage = ShaderType::Compute;
    auto source = ShaderRecompiler::PrepareShaderSource(code, options);
    source.unsupported_call_pc = 0x10u; // S_SWAPPC_B64 is at offset 0x10
    auto read_mem = [](void*, uint64_t address, std::span<uint32_t> values) {
      return !values.empty() && LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
    };
    ShaderRecompiler::IR::SrtRuntime runtime{
        .read_specialization_memory = read_mem,
    };
    const uint64_t callee_target = callee_addr;
    const bool link_ok = ShaderRecompiler::LinkDynamicCallTargets(
        source, std::span<const uint64_t>(&callee_target, 1), runtime, 0xd1ca11ull);
    Require(name, "dynamic linking succeeded", link_ok, "LinkDynamicCallTargets failed");'''

assert old_linking_section in content, "old_linking_section not found"
content = content.replace(old_linking_section, new_linking_section, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated CheckDynamicCallBranchRelocation with source.unsupported_call_pc = 0x10u!")

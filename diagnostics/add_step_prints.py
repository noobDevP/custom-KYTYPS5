path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add step prints in CheckDynamicCallBranchRelocation
old_start = 'void CheckDynamicCallBranchRelocation() {'
new_start = '''void CheckDynamicCallBranchRelocation() {
    std::printf("[gpu] step 1: enter\\n");'''

content = content.replace(old_start, new_start, 1)

old_alloc = 'Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory('
new_alloc = 'std::printf("[gpu] step 2: before alloc\\n"); Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory('
content = content.replace(old_alloc, new_alloc, 1)

old_link = 'const bool link_ok = ShaderRecompiler::LinkDynamicCallTargets('
new_link = 'std::printf("[gpu] step 3: before link\\n"); const bool link_ok = ShaderRecompiler::LinkDynamicCallTargets('
content = content.replace(old_link, new_link, 1)

old_cfg = 'const auto graph = ShaderRecompiler::CFG::BuildGraph(source.decoded);'
new_cfg = 'std::printf("[gpu] step 4: before cfg\\n"); const auto graph = ShaderRecompiler::CFG::BuildGraph(source.decoded);'
content = content.replace(old_cfg, new_cfg, 1)

old_ctx = 'RenderContext context(m_runtime_context);'
# only in CheckDynamicCallBranchRelocation:
idx = content.find("CheckDynamicCallBranchRelocation")
idx_ctx = content.find("RenderContext context(m_runtime_context);", idx)
content = content[:idx_ctx] + 'std::printf("[gpu] step 5: before context\\n"); ' + content[idx_ctx:]

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Added step prints!")

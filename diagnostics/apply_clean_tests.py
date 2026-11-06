path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Include ShaderCFG.h
inc_target = '#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"'
new_inc = '''#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"'''
assert inc_target in content
content = content.replace(inc_target, new_inc, 1)

# 2. Add InvalidateProgramModules and GetDynamicDecodedProgram to PipelineCacheTestAccess
access_target = '''  static std::array<size_t, 2> Pipelines(const PipelineCache &cache) {
    return {cache.m_graphics_pipelines.size(), cache.m_compute_pipelines.size()};
  }'''
new_access = '''  static std::array<size_t, 2> Pipelines(const PipelineCache &cache) {
    return {cache.m_graphics_pipelines.size(), cache.m_compute_pipelines.size()};
  }

  static void InvalidateProgramModules(PipelineCache &cache) {
    for (auto &[key, prog] : cache.m_program_cache) {
      prog.modules.clear();
    }
  }

  static const ShaderRecompiler::Decoder::Program* GetDynamicDecodedProgram(const PipelineCache &cache) {
    for (const auto &[key, prog] : cache.m_program_cache) {
      if (prog.dynamic_call_state && prog.dynamic_call_state->source) {
        return &prog.dynamic_call_state->source->decoded;
      }
    }
    return nullptr;
  }'''
assert access_target in content
content = content.replace(access_target, new_access, 1)

# 3. Add both test methods right before CheckNullStorageBufferDescriptor
target_method_anchor = '''  // An empty binding is a null descriptor, so loads and stores without shader checks touch no memory.
  void CheckNullStorageBufferDescriptor() {'''

tests_code = '''  void CheckDynamicCallSpecializationRetention() {
    constexpr const char* name = "DynamicCallSpecializationRetention";
    constexpr uintptr_t base = 0x0000000205800000ull;
    constexpr uint64_t size = 0x10000;
    constexpr auto table = base + 0x100;
    constexpr auto sbt_base = base + 0x800;
    constexpr auto first = base + 0x1000;
    constexpr auto second = base + 0x2000;
    constexpr auto s12_base = base + 0x3000;
    constexpr auto output = base + 0x4000;

    // Caller compute shader with dynamic call at S_SWAPPC_B64 s14, s14
    std::vector<u32> code{
        EncodeVop1(0x02, 4, 256),                     // v_readfirstlane_b32 s4, v0
        EncodeSmem0(0x09, 14, 0), EncodeSmem1(0, 4),  // s_buffer_load_dwordx2 s14, s[0:3], s4
        0xbf8cc07fu,                                  // s_waitcnt 0
        EncodeSop1(0x21, 14, 14),                     // s_swappc_b64 s14, s14
        EncodeSMovB32(20, InlineU32(1)),
        0xbf810000u                                   // s_endpgm
    };
    ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
        {.type = Prospero::ShaderBinaryType::kCs,
         .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});

    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
        "call fixture allocation failed");
    void* mapped = reinterpret_cast<void*>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, size, 0x3, 0x10, physical, size) == 0 && mapped == reinterpret_cast<void*>(base),
        "call fixture mapping failed");
    std::memset(mapped, 0, size);

    const ShaderBufferResource table_desc{{static_cast<u32>(table),
        static_cast<u32>(table >> 32u) | (16u << 16u), 8, 0x5204u}};
    LibKernel::Memory::WriteBacking(base, table_desc.fields, sizeof(table_desc.fields));

    // SBT descriptor at s12_base + 144 pointing to sbt_base
    ShaderBufferResource sbt_desc{};
    sbt_desc.UpdateAddress48(sbt_base);
    sbt_desc.fields[2] = 32; // 2 records of 16 bytes
    sbt_desc.fields[3] = 0x5204u;
    LibKernel::Memory::WriteBacking(s12_base + 144, sbt_desc.fields, sizeof(sbt_desc.fields));

    // SBT records at sbt_base
    std::array<u32, 4> record1{static_cast<u32>(first), static_cast<u32>(first >> 32u), 0, 0};
    std::array<u32, 4> record2{static_cast<u32>(second), static_cast<u32>(second >> 32u), 0, 0};
    LibKernel::Memory::WriteBacking(sbt_base, record1.data(), sizeof(record1));
    LibKernel::Memory::WriteBacking(sbt_base + 16, record2.data(), sizeof(record2));

    // Callees at first and second
    const std::array leaf1{EncodeSMovB32(20, 255), 41u, EncodeSop1(0x21, 125, 14)};
    const std::array leaf2{EncodeSMovB32(20, 255), 73u, EncodeSop1(0x21, 125, 14)};
    LibKernel::Memory::WriteBacking(first, leaf1.data(), sizeof(leaf1));
    LibKernel::Memory::WriteBacking(second, leaf2.data(), sizeof(leaf2));

    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      context.MapMemory(base, size);
      auto& shaders = processor.GetShCtx();
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                           .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
                           .wave_size = 64, .user_sgpr = 14});
      ShaderBufferResource out{};
      out.UpdateAddress48(output);
      out.fields[2] = 4;
      out.fields[3] = DstSel(4, 5, 6, 7) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      for (u32 i = 0; i < 4; ++i)
        shaders.SetCsUserSgpr(i, out.fields[i]);
      shaders.SetCsUserSgpr(12, static_cast<u32>(s12_base));
      shaders.SetCsUserSgpr(13, static_cast<u32>(s12_base >> 32u));

      // 1. Initial compile: wave_size = 64
      ShaderComputeInputInfo input1{};
      const auto program1 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input1);
      Require(name, "initial dynamic call compilation", program1 != nullptr,
              "initial dynamic call compile failed");
      const uint64_t initial_id = program1->id;

      // 2. Specialization miss on existing Program entry: invalidate compiled modules
      // Tests that when an existing Program entry has a module miss, the retained
      // dynamic call source is reused rather than falling back to unlinked NOPs.
      PipelineCacheTestAccess::InvalidateProgramModules(context.GetPipelineCache());
      ShaderComputeInputInfo input2{};
      const auto program2 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input2);
      Require(name, "second specialization compilation", program2 != nullptr,
              "specialization miss compile failed");
      Require(name, "retained dynamic call implementation",
              program2->id == initial_id,
              "dynamic call implementation was lost on specialization miss");

      // 3. Stale cache invalidation check: modify callee code
      const std::array modified_leaf{EncodeSMovB32(20, 255), 99u, EncodeSop1(0x21, 125, 14)};
      LibKernel::Memory::WriteBacking(first, modified_leaf.data(), sizeof(modified_leaf));
      Require(name, "invalidate memory", context.InvalidateMemory(first, sizeof(modified_leaf)),
              "callee invalidation failed");

      // Trigger re-compilation with callee code change
      ShaderComputeInputInfo input3{};
      const auto program3 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input3);
      Require(name, "re-linked module on code change", program3 != nullptr && program3->id != initial_id,
              "stale linked dynamic shader was reused when callable code changed");

      context.UnmapMemory(base, size);
      context.GetCommandScheduler().Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "dynamic call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }

  void CheckDynamicCallBranchRelocation() {
    constexpr const char* name = "DynamicCallBranchRelocation";
    constexpr uintptr_t base = 0x0000000205900000ull;
    constexpr uint64_t size = 0x10000;
    constexpr auto table = base + 0x100;
    constexpr auto sbt_base = base + 0x800;
    constexpr auto first = base + 0x1000;
    constexpr auto second = base + 0x2000;
    constexpr auto s12_base = base + 0x3000;
    constexpr auto output = base + 0x4000;

    // Caller compute shader with dynamic call at S_SWAPPC_B64 s14, s14
    std::vector<u32> code{
        EncodeVop1(0x02, 4, 256),                     // v_readfirstlane_b32 s4, v0
        EncodeSmem0(0x09, 14, 0), EncodeSmem1(0, 4),  // s_buffer_load_dwordx2 s14, s[0:3], s4
        0xbf8cc07fu,                                  // s_waitcnt 0
        EncodeSop1(0x21, 14, 14),                     // s_swappc_b64 s14, s14
        EncodeSMovB32(20, InlineU32(1)),
        0xbf810000u                                   // s_endpgm
    };
    ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
        {.type = Prospero::ShaderBinaryType::kCs,
         .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});

    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
        0, LibKernel::Memory::KernelGetDirectMemorySize(), size, size, 0, &physical) == 0,
        "call fixture allocation failed");
    void* mapped = reinterpret_cast<void*>(base);
    Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
        &mapped, size, 0x3, 0x10, physical, size) == 0 && mapped == reinterpret_cast<void*>(base),
        "call fixture mapping failed");
    std::memset(mapped, 0, size);

    const ShaderBufferResource table_desc{{static_cast<u32>(table),
        static_cast<u32>(table >> 32u) | (16u << 16u), 8, 0x5204u}};
    LibKernel::Memory::WriteBacking(base, table_desc.fields, sizeof(table_desc.fields));

    // SBT descriptor at s12_base + 144 pointing to sbt_base
    ShaderBufferResource sbt_desc{};
    sbt_desc.UpdateAddress48(sbt_base);
    sbt_desc.fields[2] = 32; // 2 records of 16 bytes
    sbt_desc.fields[3] = 0x5204u;
    LibKernel::Memory::WriteBacking(s12_base + 144, sbt_desc.fields, sizeof(sbt_desc.fields));

    // SBT records at sbt_base
    std::array<u32, 4> record1{static_cast<u32>(first), static_cast<u32>(first >> 32u), 0, 0};
    std::array<u32, 4> record2{static_cast<u32>(second), static_cast<u32>(second >> 32u), 0, 0};
    LibKernel::Memory::WriteBacking(sbt_base, record1.data(), sizeof(record1));
    LibKernel::Memory::WriteBacking(sbt_base + 16, record2.data(), sizeof(record2));

    // Dynamically linked callee 1 with conditional forward and backward branches at non-zero offsets:
    // Offset 0x00: s_mov_b32 s20, 0
    // Offset 0x04: s_add_u32 s20, s20, 1 (backward loop header)
    // Offset 0x08: s_cmp_lt_i32 s20, 10
    // Offset 0x0c: s_cbranch_scc1 +1 (conditional forward branch at nonzero offset +0x0c, targets 0x14)
    // Offset 0x10: s_mov_b32 s21, 1 (skipped if forward branch taken)
    // Offset 0x14: s_cmp_lt_u32 s20, 5 (forward branch target)
    // Offset 0x18: s_cbranch_scc1 -6 (conditional backward branch at nonzero offset +0x18, targets 0x04)
    // Offset 0x1c: s_setpc_b64 s[14:15] (callee return)
    const std::array branched_leaf{
        EncodeSMovB32(20, InlineU32(0)),
        EncodeSop2(0x00, 20, 20, InlineU32(1)),
        EncodeSopc(0x04, 20, InlineU32(10)),
        EncodeSopp(0x05, 1),
        EncodeSMovB32(21, InlineU32(1)),
        EncodeSopc(0x0a, 20, InlineU32(5)),
        EncodeSopp(0x05, static_cast<uint16_t>(-6)),
        EncodeSop1(0x21, 125, 14)
    };
    const std::array leaf2{EncodeSMovB32(20, 255), 73u, EncodeSop1(0x21, 125, 14)};
    LibKernel::Memory::WriteBacking(first, branched_leaf.data(), sizeof(branched_leaf));
    LibKernel::Memory::WriteBacking(second, leaf2.data(), sizeof(leaf2));

    RenderContext context(m_runtime_context);
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      context.MapMemory(base, size);
      auto& shaders = processor.GetShCtx();
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                           .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
                           .wave_size = 64, .user_sgpr = 14});
      ShaderBufferResource out{};
      out.UpdateAddress48(output);
      out.fields[2] = 4;
      out.fields[3] = DstSel(4, 5, 6, 7) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      for (u32 i = 0; i < 4; ++i)
        shaders.SetCsUserSgpr(i, out.fields[i]);
      shaders.SetCsUserSgpr(12, static_cast<u32>(s12_base));
      shaders.SetCsUserSgpr(13, static_cast<u32>(s12_base >> 32u));

      // Compile compute program through pipeline cache with dynamic SBT linking
      ShaderComputeInputInfo input{};
      const auto program = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input);
      Require(name, "dynamic call compilation", program != nullptr,
              "dynamic call compile with branched callee failed");

      // Verify relocated targets and build CFG explicitly
      const auto* decoded = PipelineCacheTestAccess::GetDynamicDecodedProgram(context.GetPipelineCache());
      Require(name, "dynamic decoded program retained", decoded != nullptr,
              "could not retrieve dynamically decoded program from pipeline cache");

      // The caller has 6 instructions: size = 24 bytes (0x18). Callee starts at func_pc = 0x18.
      const uint32_t func_pc = static_cast<uint32_t>(code.size() * sizeof(uint32_t));
      const uint32_t expected_fwd_pc = func_pc + 0x0cu;
      const uint32_t expected_fwd_target = func_pc + 0x14u;
      const uint32_t expected_bwd_pc = func_pc + 0x18u;
      const uint32_t expected_bwd_target = func_pc + 0x04u;

      bool found_fwd = false;
      bool found_bwd = false;
      for (const auto& inst : decoded->instructions) {
        if (inst.pc == expected_fwd_pc) {
          Require(name, "forward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1,
                  "expected S_CBRANCH_SCC1");
          Require(name, "relocated forward target", inst.branch_target == expected_fwd_target,
                  fmt::format("fwd target got 0x{:x}, expected 0x{:x}", inst.branch_target, expected_fwd_target).c_str());
          found_fwd = true;
        }
        if (inst.pc == expected_bwd_pc) {
          Require(name, "backward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1,
                  "expected S_CBRANCH_SCC1");
          Require(name, "relocated backward target", inst.branch_target == expected_bwd_target,
                  fmt::format("bwd target got 0x{:x}, expected 0x{:x}", inst.branch_target, expected_bwd_target).c_str());
          found_bwd = true;
        }
      }
      Require(name, "found forward branch instruction", found_fwd, "fwd branch instruction not found in decoded program");
      Require(name, "found backward branch instruction", found_bwd, "bwd branch instruction not found in decoded program");

      // Explicitly build the control-flow graph (CFG)
      const auto graph = ShaderRecompiler::CFG::BuildGraph(*decoded);
      Require(name, "CFG build succeeded without failure", graph.failure_kind == ShaderRecompiler::CFG::FailureKind::None,
              fmt::format("CFG build failed: {}", graph.unsupported_reason).c_str());
      Require(name, "CFG has basic blocks", !graph.blocks.empty(), "CFG has no basic blocks");
      Require(name, "CFG detected back edge/loop", !graph.back_edges.empty() || !graph.natural_loops.empty(),
              "expected back edges or natural loop in CFG for backward branch");

      context.UnmapMemory(base, size);
      context.GetCommandScheduler().Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }
'''

assert target_method_anchor in content
content = content.replace(target_method_anchor, tests_code + "\n" + target_method_anchor, 1)

# 4. Add CLI arguments
target_cli = '''  if (argc == 2 && std::strcmp(argv[1], "--scalar-leaf-call-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckScalarLeafCall();
    return 0;
  }'''
new_cli = '''  if (argc == 2 && std::strcmp(argv[1], "--dynamic-call-retention-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckDynamicCallSpecializationRetention();
    vulkan.CheckDynamicCallBranchRelocation();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--dynamic-call-branch-relocation-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckDynamicCallBranchRelocation();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--scalar-leaf-call-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckScalarLeafCall();
    return 0;
  }'''
assert target_cli in content
content = content.replace(target_cli, new_cli, 1)

# 5. Add to main test list
target_main = '''  vulkan.CheckGpuSuspendPoint();
  vulkan.CheckGpuCommandLane();
  vulkan.CheckScalarLeafCall();'''
new_main = '''  vulkan.CheckGpuSuspendPoint();
  vulkan.CheckGpuCommandLane();
  vulkan.CheckDynamicCallSpecializationRetention();
  vulkan.CheckDynamicCallBranchRelocation();
  vulkan.CheckScalarLeafCall();'''
assert target_main in content
content = content.replace(target_main, new_main, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Applied clean tests successfully!")

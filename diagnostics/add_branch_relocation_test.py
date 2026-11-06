path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add #include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
include_target = '#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"'
new_include = '''#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"'''

assert include_target in content, "include_target not found"
content = content.replace(include_target, new_include, 1)

# 2. Add CheckDynamicCallBranchRelocation method
target_method_anchor = '''    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "dynamic call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }'''

new_method = '''    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "dynamic call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }

  void CheckDynamicCallBranchRelocation() {
    constexpr const char* name = "DynamicCallBranchRelocation";
    constexpr uintptr_t base = 0x0000000205900000ull;
    constexpr uint64_t size = 0x10000;
    constexpr auto table = base + 0x100;
    constexpr auto sbt_base = base + 0x800;
    constexpr auto callee_addr = base + 0x1000;
    constexpr auto s12_base = base + 0x3000;
    constexpr auto output = base + 0x4000;

    // Caller compute shader with dynamic call at S_SWAPPC_B64 s14, s14
    std::vector<u32> code{
        EncodeVop1(0x02, 4, 256),                     // v_readfirstlane_b32 s4, v0 (0x00)
        EncodeSmem0(0x09, 14, 0), EncodeSmem1(0, 4),  // s_buffer_load_dwordx2 s14, s[0:3], s4 (0x04)
        0xbf8cc07fu,                                  // s_waitcnt 0 (0x0c)
        EncodeSop1(0x21, 14, 14),                     // s_swappc_b64 s14, s14 (0x10)
        EncodeSMovB32(20, InlineU32(1)),              // s_mov_b32 s20, 1 (0x14)
        0xbf810000u                                   // s_endpgm (0x18)
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
    sbt_desc.fields[2] = 16; // 1 record of 16 bytes
    sbt_desc.fields[3] = 0x5204u;
    LibKernel::Memory::WriteBacking(s12_base + 144, sbt_desc.fields, sizeof(sbt_desc.fields));

    // SBT record pointing to callee_addr
    std::array<u32, 4> record{static_cast<u32>(callee_addr), static_cast<u32>(callee_addr >> 32u), 0, 0};
    LibKernel::Memory::WriteBacking(sbt_base, record.data(), sizeof(record));

    // Callee with conditional forward branch and backward loop branch at non-zero offsets
    // Offset 0x00: s_mov_b32 s20, 0
    // Offset 0x04: s_add_u32 s20, s20, 1 (backward branch target / loop header)
    // Offset 0x08: s_cmp_lt_i32 s20, 10
    // Offset 0x0c: s_cbranch_scc1 +1 (conditional forward branch at nonzero offset +0x0c, targets 0x14)
    // Offset 0x10: s_mov_b32 s21, 1 (skipped if branch taken)
    // Offset 0x14: s_cmp_lt_u32 s20, 5 (forward branch target)
    // Offset 0x18: s_cbranch_scc1 -6 (conditional backward branch at nonzero offset +0x18, targets 0x04)
    // Offset 0x1c: s_setpc_b64 s[14:15] (return to caller)
    const std::array callee{
        EncodeSMovB32(20, InlineU32(0)),
        EncodeSop2(0x00, 20, 20, InlineU32(1)),
        EncodeSopc(0x04, 20, InlineU32(10)),
        EncodeSopp(0x05, 1),
        EncodeSMovB32(21, InlineU32(1)),
        EncodeSopc(0x0a, 20, InlineU32(5)),
        EncodeSopp(0x05, static_cast<u16>(-6)),
        EncodeSop1(0x21, 125, 14)
    };
    LibKernel::Memory::WriteBacking(callee_addr, callee.data(), sizeof(callee));

    // 1. Direct linking and CFG verification:
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
    Require(name, "dynamic linking succeeded", link_ok, "LinkDynamicCallTargets failed");

    // Relocated instructions verification:
    // Callee function begins immediately after caller code (0x1c)
    const uint32_t func_pc = static_cast<uint32_t>(code.size() * sizeof(uint32_t));
    const uint32_t expected_fwd_branch_pc = func_pc + 0x0cu;
    const uint32_t expected_fwd_target = func_pc + 0x14u;
    const uint32_t expected_bwd_branch_pc = func_pc + 0x18u;
    const uint32_t expected_bwd_target = func_pc + 0x04u;

    bool found_fwd = false;
    bool found_bwd = false;
    for (const auto& inst : source.decoded.instructions) {
      if (inst.pc == expected_fwd_branch_pc) {
        Require(name, "forward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1);
        Require(name, "relocated forward branch target", inst.branch_target == expected_fwd_target,
                fmt::format("fwd target got 0x{:x}, expected 0x{:x}", inst.branch_target, expected_fwd_target).c_str());
        found_fwd = true;
      }
      if (inst.pc == expected_bwd_branch_pc) {
        Require(name, "backward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1);
        Require(name, "relocated backward branch target", inst.branch_target == expected_bwd_target,
                fmt::format("bwd target got 0x{:x}, expected 0x{:x}", inst.branch_target, expected_bwd_target).c_str());
        found_bwd = true;
      }
    }
    Require(name, "found forward branch instruction", found_fwd);
    Require(name, "found backward branch instruction", found_bwd);

    // Build CFG and verify
    const auto graph = ShaderRecompiler::CFG::BuildGraph(source.decoded);
    Require(name, "CFG build succeeded without failure", graph.failure_kind == ShaderRecompiler::CFG::FailureKind::None,
            fmt::format("CFG build failed: {}", graph.unsupported_reason).c_str());
    Require(name, "CFG has basic blocks", !graph.blocks.empty());
    Require(name, "CFG detected back edge/loop", !graph.back_edges.empty() || !graph.natural_loops.empty(),
            "expected back edges in CFG for backward branch");

    // 2. Pipeline cache end-to-end compilation:
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

      ShaderComputeInputInfo input{};
      const auto program = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input);
      Require(name, "dynamic call branch relocation compile", program != nullptr,
              "pipeline cache compilation failed for shader with branched callee");

      context.UnmapMemory(base, size);
      context.GetCommandScheduler().Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }'''

assert target_method_anchor in content, "target_method_anchor not found"
content = content.replace(target_method_anchor, new_method, 1)

# 3. Add to CLI
target_cli = '''  if (argc == 2 && std::strcmp(argv[1], "--dynamic-call-retention-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckDynamicCallSpecializationRetention();
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
  }'''

assert target_cli in content, "target_cli not found"
content = content.replace(target_cli, new_cli, 1)

# 4. Add to main test list
target_main = '''  vulkan.CheckDynamicCallSpecializationRetention();
  vulkan.CheckScalarLeafCall();'''

new_main = '''  vulkan.CheckDynamicCallSpecializationRetention();
  vulkan.CheckDynamicCallBranchRelocation();
  vulkan.CheckScalarLeafCall();'''

assert target_main in content, "target_main not found"
content = content.replace(target_main, new_main, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Successfully added CheckDynamicCallBranchRelocation!")

path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read().replace('\r\n', '\n')

# 1. Update PipelineCacheTestAccess with helpers
access_target = '''  static const ShaderRecompiler::Decoder::Program* GetDynamicDecodedProgram(const PipelineCache &cache) {
    for (const auto &[key, prog] : cache.m_program_cache) {
      if (prog.dynamic_call_state && prog.dynamic_call_state->source) {
        return &prog.dynamic_call_state->source->decoded;
      }
    }
    return nullptr;
  }'''

access_replacement = '''  static const ShaderRecompiler::Decoder::Program* GetDynamicDecodedProgram(const PipelineCache &cache) {
    for (const auto &[key, prog] : cache.m_program_cache) {
      if (prog.dynamic_call_state && prog.dynamic_call_state->source) {
        return &prog.dynamic_call_state->source->decoded;
      }
    }
    return nullptr;
  }

  static void SimulateDiskReloadWithIncompatibleRecord(PipelineCache &cache) {
    for (auto &[key, prog] : cache.m_program_cache) {
      prog.dynamic_call_state.reset();
      prog.from_disk = true;
      if (prog.modules.empty()) {
        prog.modules.push_back(ShaderModule{.id = 0xdeadbeefull, .module = nullptr});
      }
    }
  }

  static bool HasDynamicCallSource(const PipelineCache &cache) {
    for (const auto &[key, prog] : cache.m_program_cache) {
      if (prog.dynamic_call_state && prog.dynamic_call_state->source != nullptr) {
        return true;
      }
    }
    return false;
  }'''

assert access_target in content, 'access_target not found'
content = content.replace(access_target, access_replacement, 1)

# 2. Add the 3 new test methods right before CheckNullStorageBufferDescriptor
test_anchor = '''  // An empty binding is a null descriptor, so loads and stores without shader checks touch no memory.
  void CheckNullStorageBufferDescriptor() {'''

new_tests_code = '''  void CheckDynamicCallRestartReload() {
    constexpr const char* name = "DynamicCallRestartReload";
    constexpr uintptr_t base = 0x0000000205a00000ull;
    constexpr uint64_t size = 0x10000;
    constexpr auto table = base + 0x100;
    constexpr auto sbt_base = base + 0x800;
    constexpr auto first = base + 0x1000;
    constexpr auto second = base + 0x2000;
    constexpr auto s12_base = base + 0x3000;
    constexpr auto output = base + 0x4000;

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

    ShaderBufferResource sbt_desc{};
    sbt_desc.UpdateAddress48(sbt_base);
    sbt_desc.fields[2] = 32;
    sbt_desc.fields[3] = 0x5204u;
    LibKernel::Memory::WriteBacking(s12_base + 144, sbt_desc.fields, sizeof(sbt_desc.fields));

    std::array<u32, 4> record1{static_cast<u32>(first), static_cast<u32>(first >> 32u), 0, 0};
    std::array<u32, 4> record2{static_cast<u32>(second), static_cast<u32>(second >> 32u), 0, 0};
    LibKernel::Memory::WriteBacking(sbt_base, record1.data(), sizeof(record1));
    LibKernel::Memory::WriteBacking(sbt_base + 16, record2.data(), sizeof(record2));

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

      // 1. Initial compile
      ShaderComputeInputInfo input1{};
      const auto program1 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input1);
      Require(name, "initial dynamic call compilation", program1 != nullptr,
              "initial dynamic call compile failed");
      Require(name, "dynamic call state created",
              PipelineCacheTestAccess::HasDynamicCallSource(context.GetPipelineCache()),
              "dynamic call state was not created");

      // 2. Simulate disk reload with incompatible record
      PipelineCacheTestAccess::SimulateDiskReloadWithIncompatibleRecord(context.GetPipelineCache());
      Require(name, "simulated disk reload reset dynamic state",
              !PipelineCacheTestAccess::HasDynamicCallSource(context.GetPipelineCache()),
              "simulated disk reload failed to reset dynamic call state");

      // 3. Compile again after simulated reload: should detect dynamic calls, clear incompatible record, and relink cleanly
      ShaderComputeInputInfo input2{};
      const auto program2 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input2);
      Require(name, "re-compile after disk reload", program2 != nullptr,
              "compile after disk reload failed");
      Require(name, "relinked dynamic call state after disk reload",
              PipelineCacheTestAccess::HasDynamicCallSource(context.GetPipelineCache()),
              "dynamic call state was not restored after disk reload");

      context.UnmapMemory(base, size);
      context.GetCommandScheduler().Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }

  void CheckDynamicCallCalleeModificationBeyond1024() {
    constexpr const char* name = "DynamicCallCalleeModBeyond1024";
    constexpr uintptr_t base = 0x0000000205b00000ull;
    constexpr uint64_t size = 0x20000;
    constexpr auto table = base + 0x100;
    constexpr auto sbt_base = base + 0x800;
    constexpr auto first = base + 0x1000;
    constexpr auto second = base + 0x3000;
    constexpr auto s12_base = base + 0x5000;
    constexpr auto output = base + 0x6000;

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

    ShaderBufferResource sbt_desc{};
    sbt_desc.UpdateAddress48(sbt_base);
    sbt_desc.fields[2] = 32;
    sbt_desc.fields[3] = 0x5204u;
    LibKernel::Memory::WriteBacking(s12_base + 144, sbt_desc.fields, sizeof(sbt_desc.fields));

    std::array<u32, 4> record1{static_cast<u32>(first), static_cast<u32>(first >> 32u), 0, 0};
    std::array<u32, 4> record2{static_cast<u32>(second), static_cast<u32>(second >> 32u), 0, 0};
    LibKernel::Memory::WriteBacking(sbt_base, record1.data(), sizeof(record1));
    LibKernel::Memory::WriteBacking(sbt_base + 16, record2.data(), sizeof(record2));

    // Build large callee with 280 instructions (1120 bytes > 1024 bytes)
    std::vector<u32> large_callee;
    large_callee.push_back(EncodeSMovB32(20, InlineU32(0)));
    for (int i = 1; i < 279; ++i) {
      large_callee.push_back(EncodeSMovB32(20, InlineU32(1)));
    }
    large_callee.push_back(EncodeSop1(0x21, 125, 14)); // s_setpc_b64 s[14:15]
    const std::array leaf2{EncodeSMovB32(20, 255), 73u, EncodeSop1(0x21, 125, 14)};

    LibKernel::Memory::WriteBacking(first, large_callee.data(), large_callee.size() * sizeof(u32));
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

      // 1. Initial compile
      ShaderComputeInputInfo input1{};
      const auto program1 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input1);
      Require(name, "initial compile large callee", program1 != nullptr, "initial compile failed");
      const uint64_t initial_id = program1->id;

      // 2. Modify instruction 265 at byte 1060 (beyond byte 1024)
      const u32 modified_word = EncodeSMovB32(20, InlineU32(99));
      LibKernel::Memory::WriteBacking(first + 265 * sizeof(u32), &modified_word, sizeof(modified_word));
      Require(name, "invalidate memory",
              context.InvalidateMemory(first, large_callee.size() * sizeof(u32)),
              "memory invalidation failed");

      // 3. Second compile: instruction-aware hashing beyond byte 1024 must detect change and recompile
      ShaderComputeInputInfo input2{};
      const auto program2 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input2);
      Require(name, "compile after mod beyond 1024", program2 != nullptr, "second compile failed");
      Require(name, "detected modification beyond byte 1024", program2->id != initial_id,
              "hash failed to detect modification beyond byte 1024");

      context.UnmapMemory(base, size);
      context.GetCommandScheduler().Finish();
    });
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }

  void CheckDynamicCallRelinkFailureAfterTargetTableChange() {
    constexpr const char* name = "DynamicCallRelinkFailure";
    constexpr uintptr_t base = 0x0000000205c00000ull;
    constexpr uint64_t size = 0x10000;
    constexpr auto table = base + 0x100;
    constexpr auto sbt_base = base + 0x800;
    constexpr auto first = base + 0x1000;
    constexpr auto second = base + 0x2000;
    constexpr auto s12_base = base + 0x3000;
    constexpr auto output = base + 0x4000;

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

    ShaderBufferResource sbt_desc{};
    sbt_desc.UpdateAddress48(sbt_base);
    sbt_desc.fields[2] = 32;
    sbt_desc.fields[3] = 0x5204u;
    LibKernel::Memory::WriteBacking(s12_base + 144, sbt_desc.fields, sizeof(sbt_desc.fields));

    std::array<u32, 4> record1{static_cast<u32>(first), static_cast<u32>(first >> 32u), 0, 0};
    std::array<u32, 4> record2{static_cast<u32>(second), static_cast<u32>(second >> 32u), 0, 0};
    LibKernel::Memory::WriteBacking(sbt_base, record1.data(), sizeof(record1));
    LibKernel::Memory::WriteBacking(sbt_base + 16, record2.data(), sizeof(record2));

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

      // 1. Initial valid compile
      ShaderComputeInputInfo input1{};
      const auto program1 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input1);
      Require(name, "initial dynamic call compile", program1 != nullptr, "initial compile failed");
      const uint64_t initial_id = program1->id;
      Require(name, "dynamic call state created",
              PipelineCacheTestAccess::HasDynamicCallSource(context.GetPipelineCache()),
              "dynamic call state was not created");

      // 2. Corrupt callee target with unsupported instruction so relinking fails
      const u32 unsupported_code[2] = {0x00000060u, 0x00000000u};
      LibKernel::Memory::WriteBacking(first, unsupported_code, sizeof(unsupported_code));
      Require(name, "invalidate memory",
              context.InvalidateMemory(first, sizeof(unsupported_code)),
              "memory invalidation failed");

      // 3. Compile again: relinking fails, should prevent stale source/module reuse
      ShaderComputeInputInfo input2{};
      const auto program2 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input2);
      Require(name, "compile fallback on relink failure", program2 != nullptr,
              "fallback compilation failed");
      Require(name, "stale module not reused", program2->id != initial_id,
              "stale module was reused after relink failure");
      Require(name, "stale dynamic call source cleared",
              !PipelineCacheTestAccess::HasDynamicCallSource(context.GetPipelineCache()),
              "stale dynamic call source was retained after relink failure");

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

assert test_anchor in content, 'test_anchor not found'
content = content.replace(test_anchor, new_tests_code + '\n' + test_anchor, 1)

# 3. Update CLI in main
old_cli = '''  if (argc == 2 && std::strcmp(argv[1], "--dynamic-call-retention-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckDynamicCallSpecializationRetention();
    vulkan.CheckDynamicCallBranchRelocation();
    return 0;
  }'''

new_cli = '''  if (argc == 2 && std::strcmp(argv[1], "--dynamic-call-retention-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckDynamicCallSpecializationRetention();
    vulkan.CheckDynamicCallBranchRelocation();
    vulkan.CheckDynamicCallRestartReload();
    vulkan.CheckDynamicCallCalleeModificationBeyond1024();
    vulkan.CheckDynamicCallRelinkFailureAfterTargetTableChange();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--dynamic-call-all-tests-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckDynamicCallSpecializationRetention();
    vulkan.CheckDynamicCallBranchRelocation();
    vulkan.CheckDynamicCallRestartReload();
    vulkan.CheckDynamicCallCalleeModificationBeyond1024();
    vulkan.CheckDynamicCallRelinkFailureAfterTargetTableChange();
    return 0;
  }'''

assert old_cli in content, 'old_cli not found'
content = content.replace(old_cli, new_cli, 1)

with open(path, "w", encoding="utf-8", newline='\n') as f:
    f.write(content)

print("Applied 3 new tests successfully!")

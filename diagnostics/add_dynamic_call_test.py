path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target1 = """    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }

  // An empty binding is a null descriptor, so loads and stores without shader checks touch no memory."""

new_test = """    Require(name, "release", LibKernel::Memory::KernelCheckedReleaseDirectMemory(physical, size) == 0,
            "call fixture release failed");
    std::printf("[gpu]     %-32s ok\\n", name);
  }

  void CheckDynamicCallSpecializationRetention() {
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
        EncodeSMovB32(20, InlineU32(0)),
        0xbeea03ffu, static_cast<u32>(base), 0xbeeb03ffu, static_cast<u32>(base >> 32u),
        0xf4080135u, 0xfa000000u, 0xbf8cc07fu, 0xf4240382u, 0xfa000020u,
        0xbf8cc07fu, 0xbf130e80u, 0xbf840004u, 0xf4240402u, 0xfa000028u,
        0xbf8cc07fu, 0xbe8e220eu, EncodeVop1(0x01, 1, 20),
        EncodeMubuf0(0x1c, 0, false), EncodeMubuf1(1, 0, 0)};
    AppendEnd(&code);
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

      // 2. Specialization miss: change wave_size to 32
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                           .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
                           .wave_size = 32, .user_sgpr = 14});
      ShaderComputeInputInfo input2{};
      const auto program2 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input2);
      Require(name, "second specialization compilation", program2 != nullptr,
              "specialization miss compile failed");
      Require(name, "retained dynamic call implementation",
              program2->info.bindings.wave_size == 32 && program2->id != program1->id,
              "dynamic call implementation was lost on specialization miss");

      // 3. Stale cache invalidation check: modify callee code
      const std::array modified_leaf{EncodeSMovB32(20, 255), 99u, EncodeSop1(0x21, 125, 14)};
      LibKernel::Memory::WriteBacking(first, modified_leaf.data(), sizeof(modified_leaf));
      Require(name, "invalidate memory", context.InvalidateMemory(first, sizeof(modified_leaf)),
              "callee invalidation failed");

      // Trigger re-compilation
      shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                           .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
                           .wave_size = 64, .user_sgpr = 14});
      ShaderComputeInputInfo input3{};
      const auto program3 = context.GetPipelineCache().GetComputeProgram(
          shaders.GetCs(), input3);
      Require(name, "re-linked module on code change", program3 != nullptr && program3->id != program1->id,
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

  // An empty binding is a null descriptor, so loads and stores without shader checks touch no memory."""

assert target1 in content, "target1 not found"
content = content.replace(target1, new_test, 1)

# Add CLI option and main call
target2 = """  if (argc == 2 && std::strcmp(argv[1], "--scalar-leaf-call-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckScalarLeafCall();
    return 0;
  }"""

new_cli = """  if (argc == 2 && std::strcmp(argv[1], "--dynamic-call-retention-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckDynamicCallSpecializationRetention();
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--scalar-leaf-call-only") == 0) {
    VulkanHarness vulkan;
    vulkan.CheckScalarLeafCall();
    return 0;
  }"""

assert target2 in content, "target2 not found"
content = content.replace(target2, new_cli, 1)

target3 = """  vulkan.CheckGpuSuspendPoint();
  vulkan.CheckGpuCommandLane();
  vulkan.CheckScalarLeafCall();"""

new_main_call = """  vulkan.CheckGpuSuspendPoint();
  vulkan.CheckGpuCommandLane();
  vulkan.CheckDynamicCallSpecializationRetention();
  vulkan.CheckScalarLeafCall();"""

assert target3 in content, "target3 not found"
content = content.replace(target3, new_main_call, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Added CheckDynamicCallSpecializationRetention test successfully!")

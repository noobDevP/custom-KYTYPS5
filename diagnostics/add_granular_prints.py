path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = '    std::printf("[gpu] step 1: enter\\n");'
replacement = '''    std::printf("[gpu] step 1: enter\\n"); fflush(stdout);
    constexpr const char* name = "DynamicCallBranchRelocation";
    constexpr uintptr_t base = 0x0000000205900000ull;
    constexpr uint64_t size = 0x10000;
    constexpr auto table = base + 0x100;
    constexpr auto sbt_base = base + 0x800;
    constexpr auto callee_addr = base + 0x1000;
    constexpr auto s12_base = base + 0x3000;
    constexpr auto output = base + 0x4000;

    std::printf("[gpu] step 1.1: before code\\n"); fflush(stdout);
    // Caller compute shader with dynamic call at S_SWAPPC_B64 s14, s14
    std::vector<u32> code{
        EncodeVop1(0x02, 4, 256),                     // v_readfirstlane_b32 s4, v0 (0x00)
        EncodeSmem0(0x09, 14, 0), EncodeSmem1(0, 4),  // s_buffer_load_dwordx2 s14, s[0:3], s4 (0x04)
        0xbf8cc07fu,                                  // s_waitcnt 0 (0x0c)
        EncodeSop1(0x21, 14, 14),                     // s_swappc_b64 s14, s14 (0x10)
        EncodeSMovB32(20, InlineU32(1)),              // s_mov_b32 s20, 1 (0x14)
        0xbf810000u                                   // s_endpgm (0x18)
    };
    std::printf("[gpu] step 1.2: before ShaderMapUserData\\n"); fflush(stdout);
    ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
        {.type = Prospero::ShaderBinaryType::kCs,
         .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
    std::printf("[gpu] step 1.3: before EnsureRuntimeContext\\n"); fflush(stdout);
    EnsureRuntimeContext();
    std::printf("[gpu] step 1.4: after EnsureRuntimeContext\\n"); fflush(stdout);'''

# Find the block in CheckDynamicCallBranchRelocation
idx = content.find('CheckDynamicCallBranchRelocation')
idx_end = content.find('std::printf("[gpu] step 2: before alloc\\n");', idx)

content = content[:idx] + content[idx:idx_end].replace(content[idx:idx_end], replacement + '\n    int64_t physical = -1;\n    ') + content[idx_end:]

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Added granular prints!")

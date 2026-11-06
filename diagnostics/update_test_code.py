path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

old_block = """    // Caller compute shader with dynamic call at S_SWAPPC_B64 s14, s14
    std::vector<u32> code{
        EncodeSMovB32(20, InlineU32(0)),
        0xbeea03ffu, static_cast<u32>(base), 0xbeeb03ffu, static_cast<u32>(base >> 32u),
        0xf4080135u, 0xfa000000u, 0xbf8cc07fu, 0xf4240382u, 0xfa000020u,
        0xbf8cc07fu, 0xbf130e80u, 0xbf840004u, 0xf4240402u, 0xfa000028u,
        0xbf8cc07fu, 0xbe8e210eu, EncodeVop1(0x01, 1, 20),
        EncodeMubuf0(0x1c, 0, false), EncodeMubuf1(1, 0, 0)};
    AppendEnd(&code);"""

new_block = """    // Caller compute shader with dynamic call at S_SWAPPC_B64 s14, s14
    std::vector<u32> code{
        0xf4240380u, 0xfa000000u, // s_buffer_load_dwordx2 s14, s[0:3], 0
        0xbf8cc07fu,              // s_waitcnt 0
        0xbe8e210eu,              // s_swappc_b64 s14, s14
        EncodeSMovB32(20, 1),
        0xbf810000u               // s_endpgm
    };"""

assert old_block in c, "old_block not found"
c = c.replace(old_block, new_block, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(c)
print("Updated caller code in test successfully!")

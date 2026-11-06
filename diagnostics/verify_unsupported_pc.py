# The 4 instructions:
# 1. s_buffer_load_dwordx2 s14, s[0:3], 0 -> 0xf4240380u, 0xfa000000u
# 2. s_waitcnt 0                          -> 0xbf8cc07fu
# 3. s_swappc_b64 s14, s14                -> 0xbe8e210eu
# 4. s_endpgm                             -> 0xbf810000u
code = [0xf4240380, 0xfa000000, 0xbf8cc07f, 0xbe8e210e, 0xbf810000]
print("Code size:", len(code), "words")

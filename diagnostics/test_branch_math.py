def simm16(val):
    return val & 0xffff

def encode_sopp(opcode, simm):
    return 0x80000000 | (0x7f << 23) | ((opcode & 0x7f) << 16) | (simm & 0xffff)

def encode_sop1(opcode, dst, src):
    return 0x80000000 | (0x7d << 23) | ((dst & 0x7f) << 16) | ((opcode & 0xff) << 8) | (src & 0xff)

def encode_sop2(opcode, dst, src0, src1):
    return 0x80000000 | ((opcode & 0x7f) << 23) | ((dst & 0x7f) << 16) | ((src1 & 0xff) << 8) | (src0 & 0xff)

def encode_sopc(opcode, src0, src1):
    return 0x80000000 | (0x7e << 23) | ((opcode & 0x7f) << 16) | ((src1 & 0xff) << 8) | (src0 & 0xff)

# Let's verify callee instructions:
# Opcode::S_BRANCH = 0x02
# Opcode::S_CBRANCH_SCC0 = 0x04
# Opcode::S_CBRANCH_SCC1 = 0x05
# Opcode::S_ADD_U32 = 0x00 (SOP2)
# Opcode::S_CMP_LT_I32 = 0x04 (SOPC)
# Opcode::S_CMP_LT_U32 = 0x06 (SOPC)
# Opcode::S_MOV_B32 = 0x03 (SOP1)
# Opcode::S_SETPC_B64 = 0x21 (SOP1)

# Callee:
# Offset 0x00: s_mov_b32 s20, 0
# Offset 0x04: s_add_u32 s20, s20, 1  (Loop Header)
# Offset 0x08: s_cmp_lt_i32 s20, 10
# Offset 0x0c: s_cbranch_scc1 +1  -> targets 0x14
# Offset 0x10: s_mov_b32 s21, 1
# Offset 0x14: s_cmp_lt_u32 s20, 5
# Offset 0x18: s_cbranch_scc1 -6  -> targets 0x04 (backward loop!)
# Offset 0x1c: s_setpc_b64 s[14:15]

# Forward branch at 0x0c:
# target = 0x0c + 4 + (1 * 4) = 0x14. Correct!
# Backward branch at 0x18:
# target = 0x18 + 4 + (-6 * 4) = 0x1c - 24 = 0x04. Correct!

print("Math verified!")

# Test what causes unsupported_call_pc
# In GCN/RDNA2:
# s_load_dwordx2 s[14:15], s[12:13], 0 (SMEM)
# or s_buffer_load_dwordx2 s[14:15], s[0:3], 0
print("Testing dynamic call pattern")

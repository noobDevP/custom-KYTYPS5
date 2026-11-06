path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix static_cast<u16>(-6) -> static_cast<uint16_t>(-6)
content = content.replace("static_cast<u16>(-6)", "static_cast<uint16_t>(-6)")

# Fix Require calls with 3 args:
# Line 12154: Require(name, "forward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1);
old_r1 = 'Require(name, "forward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1);'
new_r1 = 'Require(name, "forward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1, "unexpected opcode");'
assert old_r1 in content, "old_r1 not found"
content = content.replace(old_r1, new_r1, 1)

# Line 12160: Require(name, "backward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1);
old_r2 = 'Require(name, "backward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1);'
new_r2 = 'Require(name, "backward branch opcode", inst.opcode == ShaderRecompiler::Decoder::Opcode::S_CBRANCH_SCC1, "unexpected opcode");'
assert old_r2 in content, "old_r2 not found"
content = content.replace(old_r2, new_r2, 1)

# Line 12166: Require(name, "found forward branch instruction", found_fwd);
old_r3 = 'Require(name, "found forward branch instruction", found_fwd);'
new_r3 = 'Require(name, "found forward branch instruction", found_fwd, "instruction not found");'
assert old_r3 in content, "old_r3 not found"
content = content.replace(old_r3, new_r3, 1)

# Line 12167: Require(name, "found backward branch instruction", found_bwd);
old_r4 = 'Require(name, "found backward branch instruction", found_bwd);'
new_r4 = 'Require(name, "found backward branch instruction", found_bwd, "instruction not found");'
assert old_r4 in content, "old_r4 not found"
content = content.replace(old_r4, new_r4, 1)

# Line 12173: Require(name, "CFG has basic blocks", !graph.blocks.empty());
old_r5 = 'Require(name, "CFG has basic blocks", !graph.blocks.empty());'
new_r5 = 'Require(name, "CFG has basic blocks", !graph.blocks.empty(), "CFG has no basic blocks");'
assert old_r5 in content, "old_r5 not found"
content = content.replace(old_r5, new_r5, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated Require calls and uint16_t successfully!")

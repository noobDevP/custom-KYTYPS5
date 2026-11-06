path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()
assert "program2->info.bindings.wave_size == 32" in c
c = c.replace("program2->info.bindings.wave_size == 32", "program2->info.wave_size == 32", 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(c)
print("Replaced wave_size in test!")

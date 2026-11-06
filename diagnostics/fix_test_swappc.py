path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

assert "0xbf8cc07fu, 0xbe8e220eu, EncodeVop1(0x01, 1, 20)," in c
c = c.replace("0xbf8cc07fu, 0xbe8e220eu, EncodeVop1(0x01, 1, 20),", "0xbf8cc07fu, 0xbe8e210eu, EncodeVop1(0x01, 1, 20),", 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(c)
print("Fixed 0xbe8e220eu -> 0xbe8e210eu")

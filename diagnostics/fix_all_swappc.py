path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

count = c.count("0xbe8e220eu")
print(f"Occurrences of 0xbe8e220eu: {count}")
c = c.replace("0xbe8e220eu", "0xbe8e210eu")

with open(path, "w", encoding="utf-8") as f:
    f.write(c)

print("Replaced all 0xbe8e220eu with 0xbe8e210eu!")

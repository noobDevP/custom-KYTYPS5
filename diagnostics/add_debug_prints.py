path = "tests/ShaderRecompilerComputeTests.cpp"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = 'void CheckDynamicCallBranchRelocation() {'
replacement = '''void CheckDynamicCallBranchRelocation() {
    std::cout << "[DEBUG] Entered CheckDynamicCallBranchRelocation" << std::endl;'''

assert target in content
content = content.replace(target, replacement, 1)

target2 = 'int64_t physical = -1;'
# replace only inside CheckDynamicCallBranchRelocation
idx = content.find("Entered CheckDynamicCallBranchRelocation")
idx2 = content.find("int64_t physical = -1;", idx)
content = content[:idx2] + 'std::cout << "[DEBUG] After ShaderMapUserData" << std::endl;\n    ' + content[idx2:]

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Added debug prints!")

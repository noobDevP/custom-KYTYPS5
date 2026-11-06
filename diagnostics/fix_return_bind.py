path = "src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "CompileModule(entry->second, entry->first" in line and not "return bind(" in line:
        lines[i] = "\treturn bind(CompileModule(entry->second, entry->first, options, std::move(translated),\n"
        print("Updated line", i + 1)
        break

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)

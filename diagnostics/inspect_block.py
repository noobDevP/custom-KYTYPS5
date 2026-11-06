with open("src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp", "r", encoding="utf-8") as f:
    text = f.read()

idx1 = text.find("options.shader_hash")
idx2 = text.find("CompileModule(entry->second, entry->first")
print("idx1:", idx1, "idx2:", idx2)
if idx1 != -1 and idx2 != -1:
    with open("diagnostics/actual_trans_block.txt", "w", encoding="utf-8") as out:
        out.write(text[idx1:idx2])
    print("Wrote actual block, length:", idx2 - idx1)

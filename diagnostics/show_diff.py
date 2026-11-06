import difflib

def diff_files(f1, f2):
    with open(f1, "r", encoding="utf-8") as a, open(f2, "r", encoding="utf-8") as b:
        lines_a = a.readlines()
        lines_b = b.readlines()
    diff = difflib.unified_diff(lines_a, lines_b, fromfile=f1, tofile=f2)
    return "".join(diff)

print(diff_files("diagnostics/pre_dynamic_call_retention_backup/src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp",
                 "src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp"))

import sys

with open("CS_1a04c58a9fbc2052_linked_disasm.txt", "r") as f:
    lines = f.readlines()

insts = []
for line in lines:
    line = line.strip()
    if not line or not line.startswith("0x"):
        continue
    parts = line.split(":", 1)
    addr = int(parts[0], 16)
    rest = parts[1].strip()
    insts.append((addr, rest))

inst_map = {addr: rest for addr, rest in insts}
print(f"Total instructions: {len(insts)}")

# Find dispatcher start
dispatcher_addr = None
for addr, rest in insts:
    if "s14, 0xfffffffc" in rest:
        # dispatcher starts around here
        dispatcher_addr = addr
        break
print(f"Dispatcher starts around: 0x{dispatcher_addr:08x}" if dispatcher_addr else "No dispatcher found")

branch_mnemonics = [
    "S_BRANCH", "S_CBRANCH_SCC0", "S_CBRANCH_SCC1", "S_CBRANCH_VCCZ", 
    "S_CBRANCH_VCCNZ", "S_CBRANCH_EXECZ", "S_CBRANCH_EXECNZ",
    "S_CBRANCH_CDBGSYS", "S_CBRANCH_CDBGUSER", "S_CBRANCH_CDBGSYS_OR_USER",
    "S_CBRANCH_CDBGSYS_AND_USER"
]

callee_branches = []
for addr, rest in insts:
    if 0xab48 <= addr < (dispatcher_addr or 0x0002ad90):
        for b in branch_mnemonics:
            if rest.startswith(b + " "):
                target_str = rest.split()[1]
                target = int(target_str, 16)
                callee_branches.append((addr, b, target, rest))
                break

print(f"Callee branches found in linked code: {len(callee_branches)}")
for addr, b, target, rest in callee_branches:
    target_in_old = target in inst_map
    print(f"Branch at 0x{addr:08x}: {rest:40s} -> target 0x{target:08x} (in old map: {target_in_old})")

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

# Callee 0 starts at 0xab48.
# Find all callee boundaries: each callee ends with "S_BRANCH 0x0002ad90" (or similar trampoline jump)
# followed by the next callee, up to dispatcher at 0x0002ad90.

callees = []
current_start = 0xab48
for i, (addr, rest) in enumerate(insts):
    if addr < 0xab48:
        continue
    if addr >= 0x0002ad90:
        break
    if rest == "S_BRANCH 0x0002ad90":
        # Callee ends at this instruction
        next_addr = insts[i + 1][0] if i + 1 < len(insts) else None
        callees.append((current_start, addr, next_addr))
        current_start = next_addr

print(f"Total callees detected: {len(callees)}")
for idx, (c_start, c_end, next_start) in enumerate(callees):
    print(f"Callee {idx:2d}: [0x{c_start:08x} - 0x{c_end:08x}] (size: {c_end - c_start + 4} bytes)")

branch_mnemonics = [
    "S_BRANCH", "S_CBRANCH_SCC0", "S_CBRANCH_SCC1", "S_CBRANCH_VCCZ", 
    "S_CBRANCH_VCCNZ", "S_CBRANCH_EXECZ", "S_CBRANCH_EXECNZ",
    "S_CBRANCH_CDBGSYS", "S_CBRANCH_CDBGUSER", "S_CBRANCH_CDBGSYS_OR_USER",
    "S_CBRANCH_CDBGSYS_AND_USER"
]

# Now for each callee, check all direct branches inside it (excluding the synthetic S_BRANCH to trampoline)
all_branches_checked = 0
branches_valid_with_fix = 0
invalid_branches = []

for idx, (c_start, c_end, _) in enumerate(callees):
    for addr, rest in insts:
        if c_start <= addr <= c_end:
            if rest == "S_BRANCH 0x0002ad90":
                continue # synthetic return branch, targets trampoline
            for b in branch_mnemonics:
                if rest.startswith(b + " "):
                    target_str = rest.split()[1]
                    old_target = int(target_str, 16)
                    inst_offset_in_callee = addr - c_start
                    new_target = old_target + inst_offset_in_callee
                    
                    is_valid = new_target in inst_map
                    all_branches_checked += 1
                    if is_valid:
                        branches_valid_with_fix += 1
                    else:
                        invalid_branches.append((idx, addr, rest, old_target, new_target))
                    break

print("\n--- RESULTS ---")
print(f"Total internal callee branches checked: {all_branches_checked}")
print(f"Valid destinations with inst.pc fix:    {branches_valid_with_fix}")
print(f"Invalid destinations:                   {len(invalid_branches)}")

if invalid_branches:
    print("\nInvalid branches details:")
    for c_idx, addr, rest, old_t, new_t in invalid_branches[:20]:
        print(f"Callee {c_idx} @ 0x{addr:08x}: {rest} -> old 0x{old_t:08x}, new 0x{new_t:08x}")
else:
    print("\n100% OF DIRECT BRANCH DESTINATIONS ARE VALID INSTRUCTION BOUNDARIES!")

# Check specifically 0xab88
print("\n--- Specific check for 0x0000ab88 ---")
for addr, rest in insts:
    if addr == 0x0000ab88:
        old_target = int(rest.split()[1], 16)
        new_target = old_target + (0xab88 - 0xab48)
        print(f"Instruction at 0x0000ab88: {rest}")
        print(f"Old target: 0x{old_target:08x} (valid: {old_target in inst_map})")
        print(f"New target: 0x{new_target:08x} (valid: {new_target in inst_map})")
        print(f"Target instruction at 0x{new_target:08x}: {inst_map.get(new_target, 'NOT FOUND')}")

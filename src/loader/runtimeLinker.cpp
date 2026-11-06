#include "loader/runtimeLinker.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/platform/sysDbg.h"
#include "common/profiler.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "loader/elf.h"
#include "loader/gamePatch.h"
#include "loader/guestInstructionPatcher.h"
#include "loader/jit.h"
#include "loader/symbolDatabase.h"
#include "loader/systemContent.h"
#include "loader/x64InstructionEmulator.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <memory>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <sys/uio.h>
#include <unistd.h>
#endif
#endif

namespace Libs::LibKernel {
void SetProgName(const std::string& name);
} // namespace Libs::LibKernel

namespace Loader {

Program::Program() = default;

Program::~Program() = default;

static void FreeTlsBlock(ThreadLocalStorage::Block* block) {
	if (block == nullptr || block->ptr == nullptr) {
		return;
	}

	if (block->free_func != nullptr) {
		block->free_func(block->ptr);
	} else if (block->vm_alloc) {
		EXIT_IF(!Libs::LibKernel::Memory::FreeGuestMemory(reinterpret_cast<uint64_t>(block->ptr),
		                                                  block->alloc_size));
	} else {
		delete[] block->ptr;
	}

	block->ptr        = nullptr;
	block->free_func  = nullptr;
	block->vm_alloc   = false;
	block->alloc_size = 0;
}

static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
	return alignment != 0 ? (value + alignment - 1) & ~(alignment - 1) : value;
}

ThreadLocalStorage::~ThreadLocalStorage() {
	for (auto& [_, block]: tlss) {
		FreeTlsBlock(&block);
	}
}

using atexit_func_t = KYTY_SYSV_ABI void (*)();
using entry_func_t  = KYTY_SYSV_ABI void (*)(EntryParams* params, atexit_func_t atexit_func);
using module_ini_fini_func_t = KYTY_SYSV_ABI int (*)(size_t args, const void* argp,
                                                     module_func_t func);

enum class BindType { Unknown, Local, Global, Weak };

struct RelocationInfo {
	bool        resolved   = false;
	BindType    bind       = BindType::Unknown;
	SymbolType  type       = SymbolType::Unknown;
	uint64_t    value      = 0;
	uint64_t    vaddr      = 0;
	uint64_t    base_vaddr = 0;
	std::string name;
	std::string dbg_name;
	bool        bind_self = false;
};

struct StubbedImportRecord {
	uint32_t    index       = 0;
	uint64_t    patch_vaddr = 0;
	uint64_t    thunk_vaddr = 0;
	std::string name;
	SymbolType  type = SymbolType::Unknown;
	BindType    bind = BindType::Unknown;
	std::string program;
};

static std::vector<StubbedImportRecord> g_stubbed_imports;
static std::atomic_uint32_t             g_unresolved_stub_call_log_count {0};
static std::vector<uint64_t>            g_unresolved_stub_thunk_pages;
static uint64_t                         g_unresolved_stub_thunk_offset = 0;
static constexpr uint64_t               UNRESOLVED_STUB_PAGE_SIZE      = 4096;

static KYTY_SYSV_ABI uint64_t UnresolvedImportStub(uint64_t record_id);

static bool PatchGuestMemory64(uint64_t vaddr, uint64_t value) {
	auto* ptr     = reinterpret_cast<uint64_t*>(vaddr);
	bool  changed = (*ptr != value);
	std::memcpy(ptr, &value, sizeof(value));
	return changed;
}

static uint64_t AllocateUnresolvedImportThunk(uint64_t record_id) {
	constexpr uint64_t thunk_size = 34;

	if (g_unresolved_stub_thunk_pages.empty() ||
	    g_unresolved_stub_thunk_offset + thunk_size > UNRESOLVED_STUB_PAGE_SIZE) {
		auto page = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    0, UNRESOLVED_STUB_PAGE_SIZE, Common::VirtualMemory::Mode::ExecuteReadWrite,
		    "unresolved_import_thunk");
		EXIT_NOT_IMPLEMENTED(page == 0);
		g_unresolved_stub_thunk_pages.push_back(page);
		g_unresolved_stub_thunk_offset = 0;
	}

	auto* code = reinterpret_cast<uint8_t*>(g_unresolved_stub_thunk_pages.back() +
	                                        g_unresolved_stub_thunk_offset);
	g_unresolved_stub_thunk_offset += thunk_size;

	uint8_t bytes[thunk_size] = {
	    0x48, 0x83, 0xec, 0x08,               // sub rsp, 8
	    0x48, 0xbf, 0, 0, 0, 0, 0, 0, 0, 0, // mov rdi, record_id
	    0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, // mov rax, UnresolvedImportStub
	    0xff, 0xd0,                           // call rax
	    0x48, 0x83, 0xc4, 0x08,               // add rsp, 8
	    0x0f, 0x57, 0xc0,                     // xorps xmm0, xmm0
	    0xc3,                                 // ret
	};
	const auto target = reinterpret_cast<uint64_t>(UnresolvedImportStub);
	std::memcpy(bytes + 6, &record_id, sizeof(record_id));
	std::memcpy(bytes + 16, &target, sizeof(target));
	std::memcpy(code, bytes, sizeof(bytes));
	Common::VirtualMemory::FlushInstructionCache(reinterpret_cast<uint64_t>(code), thunk_size);
	return reinterpret_cast<uint64_t>(code);
}

static uint64_t RegisterStubbedImport(uint32_t index, const Program* program,
                                      const RelocationInfo& ri) {
	for (const auto& record: g_stubbed_imports) {
		if (record.patch_vaddr == ri.vaddr) {
			return record.thunk_vaddr;
		}
	}

	StubbedImportRecord record {};
	record.index       = index;
	record.patch_vaddr = ri.vaddr;
	record.name        = ri.name;
	record.type        = ri.type;
	record.bind        = ri.bind;
	record.program     = Common::PathToString(program->file_name);
	g_stubbed_imports.push_back(record);
	const auto record_id                     = g_stubbed_imports.size() - 1;
	const auto thunk                         = AllocateUnresolvedImportThunk(record_id);
	g_stubbed_imports[record_id].thunk_vaddr = thunk;
	return thunk;
}

static KYTY_SYSV_ABI uint64_t UnresolvedImportStub(uint64_t record_id) {
	Common::DebugCounters::Add(Common::DebugCounters::Counter::UnresolvedImportCalls);
	const auto log_index = g_unresolved_stub_call_log_count.fetch_add(1);
	if (log_index < 1024) {
		if (record_id < g_stubbed_imports.size()) {
			const auto& record = g_stubbed_imports[record_id];
			printf("Unresolved import stub called: %s\n", record.name.c_str());
			LOGF("Unresolved import stub called [%u]: patch_vaddr=0x%016" PRIx64
			     " jmprela_index=%" PRIu32 " symbol=%s type=%s bind=%s program=%s\n",
			     log_index, record.patch_vaddr, record.index, record.name.c_str(),
			     magic_enum::enum_name(record.type), magic_enum::enum_name(record.bind),
			     record.program.c_str());
		} else {
			printf("Unresolved import stub called: <bad-record>\n");
			LOGF("Unresolved import stub called [%u]: record_id=%" PRIu64 " symbol=<bad-record>\n",
			     log_index, record_id);
		}
	}
	return 0;
}

constexpr uint64_t SYSTEM_RESERVED  = 0x800000000u;
constexpr uint64_t CODE_BASE_INCR   = 0x010000000u;
constexpr uint64_t INVALID_OFFSET   = 0x040000000u;
constexpr uint64_t CODE_BASE_OFFSET = 0x100000000u;
constexpr uint64_t INVALID_MEMORY   = SYSTEM_RESERVED + INVALID_OFFSET;

static uint64_t g_desired_base_addr = SYSTEM_RESERVED + CODE_BASE_OFFSET;
static uint64_t g_invalid_memory    = 0;

static Program*              g_tls_main_program        = nullptr;
static thread_local Program* g_tls_cached_main_program = nullptr;
static thread_local uint8_t* g_tls_cached_main_tcb     = nullptr;

static KYTY_SYSV_ABI void RunEntry(uint64_t addr, EntryParams* params, atexit_func_t atexit_func,
                                   void* stack_top) {
#if defined(__x86_64__) || defined(_M_X64)
	auto* func = reinterpret_cast<entry_func_t>(addr);

	if (stack_top != nullptr) {
		const auto aligned_stack_top =
		    reinterpret_cast<uintptr_t>(stack_top) & ~static_cast<uintptr_t>(0x0f);
		const auto guest_rsp = aligned_stack_top - 2u * sizeof(uintptr_t);
		const auto guest_rbp = guest_rsp;

		auto* guest_root_frame = reinterpret_cast<uintptr_t*>(guest_rbp);
		guest_root_frame[0]    = 0;
		guest_root_frame[1]    = 0;

#if defined(__APPLE__)
		// Clang on macOS can allocate plain "r" inputs to r12/r13, which the template
		// clobbers before consuming them. Pin the inputs to registers the SysV guest
		// preserves without changing register allocation on Windows or Linux.
		register entry_func_t func_reg asm("rbx")      = func;
		register uintptr_t    guest_rsp_reg asm("r14") = guest_rsp;
		register uintptr_t    guest_rbp_reg asm("r15") = guest_rbp;
#endif

#if defined(__APPLE__)
		asm volatile(
		    "pushq %%r12\n\t"
		    "pushq %%r13\n\t"
		    "movq %%rsp, %%r12\n\t"
		    "movq %%rbp, %%r13\n\t"
		    "movq %[guest_rsp], %%rsp\n\t"
		    "movq %[guest_rbp], %%rbp\n\t"
		    "callq *%[func]\n\t"
		    "movq %%r13, %%rbp\n\t"
		    "movq %%r12, %%rsp\n\t"
		    "popq %%r13\n\t"
		    "popq %%r12\n\t"
		    :
		    : [func] "r"(func_reg), "D"(params),
		      "S"(atexit_func), [guest_rsp] "r"(guest_rsp_reg), [guest_rbp] "r"(guest_rbp_reg)
		    : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2",
		      "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12",
		      "xmm13", "xmm14", "xmm15");
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Windows stack probes use the TEB stack limits during the guest stack switch.
		// bounds, which describe the host stack and are invalid while RSP is in guest memory.
		register entry_func_t func_reg asm("rbx")     = func;
		register uintptr_t    guest_rsp_reg asm("r8") = guest_rsp;
		register uintptr_t    guest_rbp_reg asm("r9") = guest_rbp;
		asm volatile("pushq %%r12\n\t"
		             "pushq %%r13\n\t"
		             "pushq %%r14\n\t"
		             "pushq %%r15\n\t"
		             "movq %%gs:0x08, %%r14\n\t"
		             "movq %%gs:0x10, %%r15\n\t"
		             "xorq %%rcx, %%rcx\n\t"
		             "movq %%rcx, %%gs:0x08\n\t"
		             "movq %%rcx, %%gs:0x10\n\t"
		             "movq %%rsp, %%r12\n\t"
		             "movq %%rbp, %%r13\n\t"
		             "movq %[guest_rsp], %%rsp\n\t"
		             "movq %[guest_rbp], %%rbp\n\t"
		             "callq *%[func]\n\t"
		             "movq %%r13, %%rbp\n\t"
		             "movq %%r12, %%rsp\n\t"
		             "movq %%r14, %%gs:0x08\n\t"
		             "movq %%r15, %%gs:0x10\n\t"
		             "popq %%r15\n\t"
		             "popq %%r14\n\t"
		             "popq %%r13\n\t"
		             "popq %%r12\n\t"
		             : [guest_rsp] "+r"(guest_rsp_reg), [guest_rbp] "+r"(guest_rbp_reg)
		             : [func] "r"(func_reg), "D"(params), "S"(atexit_func)
		             : "cc", "memory", "rax", "rcx", "rdx", "r10", "r11", "xmm0", "xmm1", "xmm2",
		               "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11",
		               "xmm12", "xmm13", "xmm14", "xmm15");
#else
		// Clobbers prevent inputs from being allocated to r12/r13.
		asm volatile("movq %%rsp, %%r12\n\t"
		             "movq %%rbp, %%r13\n\t"
		             "movq %[guest_rsp], %%rsp\n\t"
		             "movq %[guest_rbp], %%rbp\n\t"
		             "callq *%[func]\n\t"
		             "movq %%r13, %%rbp\n\t"
		             "movq %%r12, %%rsp\n\t"
		             :
		             : [func] "r"(func), "D"(params),
		               "S"(atexit_func), [guest_rsp] "r"(guest_rsp), [guest_rbp] "r"(guest_rbp)
		             : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "r12", "r13",
		               "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8",
		               "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
#endif
		return;
	}

	uintptr_t guest_root_frame[2] = {};

#if defined(__APPLE__)
	register entry_func_t func_reg asm("rbx")      = func;
	register uintptr_t    guest_rbp_reg asm("r14") = reinterpret_cast<uintptr_t>(guest_root_frame);
#endif

#if defined(__APPLE__) || KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	asm volatile("pushq %%r12\n\t"
	             "pushq %%r13\n\t"
	             "movq %%rbp, %%r12\n\t"
	             "movq %[guest_rbp], %%rbp\n\t"
	             "callq *%[func]\n\t"
	             "movq %%r12, %%rbp\n\t"
	             "popq %%r13\n\t"
	             "popq %%r12\n\t"
	             :
#if defined(__APPLE__)
	             : [func] "r"(func_reg), "D"(params),
	               "S"(atexit_func), [guest_rbp] "r"(guest_rbp_reg)
#else
	             : [func] "r"(func), "D"(params),
	               "S"(atexit_func), [guest_rbp] "r"(guest_root_frame)
#endif
	             : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1",
	               "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11",
	               "xmm12", "xmm13", "xmm14", "xmm15");
#else
	// Keep inputs out of r12.
	asm volatile("movq %%rbp, %%r12\n\t"
	             "movq %[guest_rbp], %%rbp\n\t"
	             "callq *%[func]\n\t"
	             "movq %%r12, %%rbp\n\t"
	             :
	             : [func] "r"(func), "D"(params),
	               "S"(atexit_func), [guest_rbp] "r"(guest_root_frame)
	             : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "r12", "xmm0",
	               "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
	               "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
#endif
#else
	(void)stack_top;
	reinterpret_cast<entry_func_t>(addr)(params, atexit_func);
#endif
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
struct MainEntryStackTestState {
	bool      called = false;
	uintptr_t rsp    = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uintptr_t teb_stack_base  = UINTPTR_MAX;
	uintptr_t teb_stack_limit = UINTPTR_MAX;
#endif
};

static KYTY_SYSV_ABI void TestMainEntryStackCallback(EntryParams* params,
                                                     atexit_func_t /*atexit_func*/) {
	auto* state = reinterpret_cast<MainEntryStackTestState*>(const_cast<char*>(params->argv[0]));
	asm volatile("pushq %%r15\n\t"
	             "pushq %%r14\n\t"
	             "popq %%r14\n\t"
	             "popq %%r15\n\t"
	             :
	             :
	             : "memory");
	asm volatile("movq %%rsp, %0" : "=r"(state->rsp) : : "memory");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	asm volatile("movq %%gs:0x08, %0\n\t"
	             "movq %%gs:0x10, %1\n\t"
	             : "=r"(state->teb_stack_base), "=r"(state->teb_stack_limit)
	             :
	             : "memory");
#endif
	state->called = true;
}

bool TestMainEntryUsesGuestStack() {
	constexpr uint64_t stack_size = 0x10000;
	const auto         stack_base = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    0, stack_size, Common::VirtualMemory::Mode::ReadWrite, "main_entry_stack_test");
	if (stack_base == 0) {
		return false;
	}

	MainEntryStackTestState state {};
	EntryParams             params {};
	params.argv[0] = reinterpret_cast<const char*>(&state);
	std::memset(reinterpret_cast<void*>(stack_base), 0xcd, stack_size);
	auto* root_frame = reinterpret_cast<const uintptr_t*>(stack_base + stack_size) - 2;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uintptr_t original_teb_stack_base  = 0;
	uintptr_t original_teb_stack_limit = 0;
	asm volatile("movq %%gs:0x08, %0\n\t"
	             "movq %%gs:0x10, %1\n\t"
	             : "=r"(original_teb_stack_base), "=r"(original_teb_stack_limit)
	             :
	             : "memory");
#endif

	RunEntry(reinterpret_cast<uint64_t>(TestMainEntryStackCallback), &params, nullptr,
	         reinterpret_cast<void*>(stack_base + stack_size));

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uintptr_t restored_teb_stack_base  = 0;
	uintptr_t restored_teb_stack_limit = 0;
	asm volatile("movq %%gs:0x08, %0\n\t"
	             "movq %%gs:0x10, %1\n\t"
	             : "=r"(restored_teb_stack_base), "=r"(restored_teb_stack_limit)
	             :
	             : "memory");
	const bool teb_ok = state.teb_stack_base == 0 && state.teb_stack_limit == 0 &&
	                    restored_teb_stack_base == original_teb_stack_base &&
	                    restored_teb_stack_limit == original_teb_stack_limit;
#else
	constexpr bool teb_ok = true;
#endif

	const bool rsp_ok  = state.rsp >= stack_base && state.rsp < stack_base + stack_size;
	const bool root_ok = root_frame[0] == 0 && root_frame[1] == 0;
	const bool freed   = Libs::LibKernel::Memory::FreeGuestMemory(stack_base, stack_size);
	return state.called && rsp_ok && root_ok && teb_ok && freed;
}

bool TestModuleRelocationUsesWritableHostMapping() {
	constexpr uint64_t page_size = 0x4000;
	constexpr uint64_t value     = 0x4b59545950415443;
	const auto         base      = Libs::LibKernel::Memory::AllocateProgramMemory(
	    0, page_size, Common::VirtualMemory::Mode::ReadWrite, "host_only_patch_test");
	if (base == 0) {
		return false;
	}
	Libs::LibKernel::Memory::SetProgramMemoryProtection(base, page_size,
	                                                    Common::VirtualMemory::Mode::Read);

	Libs::LibKernel::Memory::VirtualQueryInfo before {};
	Libs::LibKernel::Memory::VirtualQueryInfo after {};
	const bool                                before_ok =
	    Libs::LibKernel::Memory::KernelVirtualQuery(reinterpret_cast<const void*>(base), 0, &before,
	                                                sizeof(before)) == 0;
	const bool changed  = PatchGuestMemory64(base, value);
	const bool after_ok = Libs::LibKernel::Memory::KernelVirtualQuery(
	                          reinterpret_cast<const void*>(base), 0, &after, sizeof(after)) == 0;
	const bool value_ok = *reinterpret_cast<const uint64_t*>(base) == value;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MEMORY_BASIC_INFORMATION mbi {};
	const bool               host_mode_ok =
	    VirtualQuery(reinterpret_cast<const void*>(base), &mbi, sizeof(mbi)) != 0 &&
	    mbi.Protect == PAGE_READWRITE;
#else
	constexpr bool host_mode_ok = true;
#endif
	const bool freed = Libs::LibKernel::Memory::FreeGuestMemory(base, page_size);

	return before_ok && after_ok && changed && value_ok && host_mode_ok && freed &&
	       before.protection == after.protection;
}
#endif

static uint64_t GetAlignedSize(const Elf64_Phdr* p) {
	return (p->p_align != 0 ? (p->p_memsz + (p->p_align - 1)) & ~(p->p_align - 1) : p->p_memsz);
}

static void DbgDumpSymbols(const std::string& folder, Elf64_Sym* symbols, uint64_t size,
                           const char* names) {
	auto folder_str = Common::FixDirectorySlash(folder);

	Common::File::CreateDirectories(folder_str);

	Common::File f;
	f.Create(folder_str + "symbols.txt");

	for (auto* sym = symbols;
	     reinterpret_cast<uint8_t*>(sym) < reinterpret_cast<uint8_t*>(symbols) + size; sym++) {
		f.Printf("----\n");
		f.Printf("st_name = %" PRIu32 ", %s\n", sym->st_name, names + sym->st_name);
		f.Printf("st_info = 0x%02" PRIx8 "\n", sym->st_info);
		f.Printf("st_other = 0x%02" PRIx8 "\n", sym->st_other);
		f.Printf("st_shndx = 0x%04" PRIx16 "\n", sym->st_shndx);
		f.Printf("st_value = 0x%016" PRIx64 "\n", sym->st_value);
		f.Printf("st_size = %" PRIu64 "\n", sym->st_size);
	}

	f.Close();
}

static void DbgDumpRela(const std::string& folder, Elf64_Rela* records, uint64_t size,
                        const char* /*names*/, const char* file_name) {
	auto folder_str = Common::FixDirectorySlash(folder);

	Common::File::CreateDirectories(folder_str);

	Common::File f;
	f.Create(folder_str + file_name);

	for (auto* r = records;
	     reinterpret_cast<uint8_t*>(r) < reinterpret_cast<uint8_t*>(records) + size; r++) {
		f.Printf("----\n"
		         "r_offset = 0x%016" PRIx64 "\n"
		         "r_info = 0x%016" PRIx64 "\n"
		         "r_addend = %" PRId64 "\n",
		         r->r_offset, r->r_info, r->r_addend);
	}

	f.Close();
}

static Common::VirtualMemory::Mode GetMode(Elf64_Word flags) {
	switch (flags) {
		case PF_R: return Common::VirtualMemory::Mode::Read;
		case PF_W: return Common::VirtualMemory::Mode::Write;
		case PF_R | PF_W: return Common::VirtualMemory::Mode::ReadWrite;
		case PF_X: return Common::VirtualMemory::Mode::Execute;
		case PF_X | PF_R: return Common::VirtualMemory::Mode::ExecuteRead;
		case PF_X | PF_W: return Common::VirtualMemory::Mode::ExecuteWrite;
		case PF_X | PF_W | PF_R: return Common::VirtualMemory::Mode::ExecuteReadWrite;

		default: return Common::VirtualMemory::Mode::NoAccess;
	}
}

struct GuestStackFrame {
	GuestStackFrame* next;
	uintptr_t        return_address;
};

static bool IsReadableRange(uint64_t addr, uint64_t size);

static int WalkGuestStack(uint64_t rbp, uint64_t rsp, void** stack, int capacity) {
	constexpr uintptr_t STACK_SIZE = 1024u * 1024u;
	const uintptr_t     code_start = SYSTEM_RESERVED + CODE_BASE_OFFSET;
	const uintptr_t     stack_end  = rsp + STACK_SIZE;
	if (stack == nullptr || capacity <= 0 || rsp == 0 || rbp < rsp || stack_end < rsp ||
	    g_desired_base_addr <= code_start) {
		return 0;
	}

	auto* frame = reinterpret_cast<GuestStackFrame*>(rbp);

	int depth = 0;
	while (depth < capacity) {
		const auto frame_addr = reinterpret_cast<uintptr_t>(frame);
		if (frame_addr < rsp || frame_addr > stack_end - sizeof(GuestStackFrame) ||
		    !IsReadableRange(frame_addr, sizeof(GuestStackFrame)) ||
		    frame->return_address < code_start || frame->return_address >= g_desired_base_addr) {
			break;
		}
		stack[depth++] = reinterpret_cast<void*>(frame->return_address);
		if (reinterpret_cast<uintptr_t>(frame->next) <= frame_addr) {
			break;
		}
		frame = frame->next;
	}
	return depth;
}

// Probe diagnostic ranges without raising another fault.
static bool IsReadableRange(uint64_t addr, uint64_t size) {
	if (addr == 0 || size == 0) {
		return false;
	}

	const uint64_t end = addr + size;
	if (end < addr) {
		return false;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uint64_t current = addr;
	while (current < end) {
		MEMORY_BASIC_INFORMATION mbi {};
		if (VirtualQuery(reinterpret_cast<const void*>(current), &mbi, sizeof(mbi)) == 0 ||
		    mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
			return false;
		}
		const auto region_end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
		if (region_end <= current) {
			return false;
		}
		current = std::min(region_end, end);
	}
#elif defined(__APPLE__)
	// Walk the Mach regions covering the range and require read permission. The fatal
	// report dumps memory behind raw register values, and a fault inside the reporter
	// re-enters the signal handler and wedges the reporting thread.
	uint64_t current = addr;
	while (current < end) {
		mach_vm_address_t              region_addr = current;
		mach_vm_size_t                 region_size = 0;
		vm_region_basic_info_data_64_t info {};
		mach_msg_type_number_t         count       = VM_REGION_BASIC_INFO_COUNT_64;
		mach_port_t                    object_name = MACH_PORT_NULL;
		if (mach_vm_region(mach_task_self(), &region_addr, &region_size, VM_REGION_BASIC_INFO_64,
		                   reinterpret_cast<vm_region_info_t>(&info), &count,
		                   &object_name) != KERN_SUCCESS ||
		    region_addr > current || (info.protection & VM_PROT_READ) == 0) {
			return false;
		}
		current = region_addr + region_size;
	}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	const auto page_size = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
	if (page_size == 0) {
		return false;
	}

	for (uint64_t current = addr; current < end;) {
		uint8_t probe = 0;

		iovec local {&probe, sizeof(probe)};
		iovec remote {reinterpret_cast<void*>(current), sizeof(probe)};

		if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) !=
		    static_cast<ssize_t>(sizeof(probe))) {
			return false;
		}

		const uint64_t next = (current & ~(page_size - 1)) + page_size;
		if (next <= current) { // wrapped at the top of the address space
			break;
		}
		current = next;
	}
#else
	(void)end;
#endif
	return true;
}

static bool KytyExceptionHandler(const Common::HostException::ExceptionInfo& exception_info) {
	const auto* info = &exception_info;

	if (info->type == Common::HostException::ExceptionType::IllegalInstruction &&
	    Loader::X64InstructionEmulator::TryEmulate(info->native_context)) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::EmulatedInstructions);
		return true;
	}

	if (info->type == Common::HostException::ExceptionType::AccessViolation) {
		using CoreAccess = Common::HostException::AccessViolationType;
		using GpuAccess  = Libs::Graphics::PageFaultAccess;
		GpuAccess access;
		switch (info->access_violation_type) {
			case CoreAccess::Read: access = GpuAccess::Read; break;
			case CoreAccess::Write: access = GpuAccess::Write; break;
			case CoreAccess::Execute: access = GpuAccess::Execute; break;
			case CoreAccess::Unknown: return false;
		}
		if (Libs::LibKernel::Memory::HandleGpuFault(access, info->access_violation_vaddr)) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuFaults);
			return true;
		}
		if (const auto length =
		        Libs::Graphics::ShaderRecompiler::IR::SrtWalkerLoadLength(info->exception_address)) {
			Loader::X64InstructionEmulator::SkipLoad(info->native_context, length);
			return true;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (info->access_violation_vaddr < 0x10000 && info->native_context != nullptr) {
			auto* ctx = static_cast<PCONTEXT>(info->native_context);
			// Reroute RenderThread invalid handle fault at 0x03af54eb to safe null path
			if ((info->exception_address & 0x00000000ffffffffULL) == 0x03af54ebULL) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "VEH: redirected RenderThread invalid handle fault at 0x{:016x} to safe null path\n",
				    info->exception_address));
				ctx->Rax = 0;
				ctx->Rip = (info->exception_address & ~0x00000000ffffffffULL) | 0x03af5520ULL;
				return true;
			}
		}
#endif
	}
	// Report whatever guest context can be read safely before terminating: which guest thread
	// faulted, the register file, the faulting code bytes and the top of its stack.
	{
		char thread_name[64] = "(host thread)";
		if (auto self = Libs::LibKernel::PthreadSelfOrNull(); self != nullptr) {
			if (Libs::LibKernel::PthreadGetname(self, thread_name) != 0) {
				std::snprintf(thread_name, sizeof(thread_name), "(unnamed guest thread)");
			}
		}
		std::printf("--- Guest fault context ---\n");
		std::printf("thread: %s\n", thread_name);
		std::printf("rax=%016" PRIx64 " rbx=%016" PRIx64 " rcx=%016" PRIx64 " rdx=%016" PRIx64 "\n"
		            "rsi=%016" PRIx64 " rdi=%016" PRIx64 " rbp=%016" PRIx64 " rsp=%016" PRIx64 "\n"
		            "r8 =%016" PRIx64 " r9 =%016" PRIx64 " r10=%016" PRIx64 " r11=%016" PRIx64 "\n"
		            "r12=%016" PRIx64 " r13=%016" PRIx64 " r14=%016" PRIx64 " r15=%016" PRIx64 "\n",
		            info->rax, info->rbx, info->rcx, info->rdx, info->rsi, info->rdi, info->rbp,
		            info->rsp, info->r8, info->r9, info->r10, info->r11, info->r12, info->r13,
		            info->r14, info->r15);
		if (IsReadableRange(info->exception_address - 48, 96)) {
			const auto* code = reinterpret_cast<const uint8_t*>(info->exception_address - 48);
			std::printf("code (pc-48 .. pc+48, fault at byte 48):");
			for (int i = 0; i < 96; i++) {
				std::printf("%s%02x", (i % 16 == 0) ? "\n " : " ", code[i]);
			}
			std::printf("\n");
		}
		if (IsReadableRange(info->rsp, 32 * sizeof(uint64_t))) {
			const auto* stack = reinterpret_cast<const uint64_t*>(info->rsp);
			std::printf("stack:");
			for (int i = 0; i < 32; i++) {
				std::printf("%s %016" PRIx64, (i % 4 == 0) ? "\n " : "", stack[i]);
			}
			std::printf("\n");
		}
		std::fflush(stdout);
	}
	EXIT("Unhandled host exception: type=%u code=%u pc=0x%016" PRIx64
	     " access=%u address=0x%016" PRIx64 "\n",
	     static_cast<unsigned>(info->type), info->native_code, info->exception_address,
	     static_cast<unsigned>(info->access_violation_type), info->access_violation_vaddr);
}

static void EncodeId64(uint16_t in_id, std::string* out_id) {
	static const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
	if (in_id < 0x40u) {
		*out_id += str[in_id];
	} else {
		if (in_id < 0x1000u) {
			*out_id += str[static_cast<uint16_t>(in_id >> 6u) & 0x3fu];
			*out_id += str[in_id & 0x3fu];
		} else {
			*out_id += str[static_cast<uint16_t>(in_id >> 12u) & 0x3fu];
			*out_id += str[static_cast<uint16_t>(in_id >> 6u) & 0x3fu];
			*out_id += str[in_id & 0x3fu];
		}
	}
}

template <class T>
static void GetDynDataOs(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = elf->GetDynamicData<T>(dyn->d_un.d_ptr);
	}
}

template <class T>
static void GetDynData(Elf64* elf, uint64_t base_vaddr, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = reinterpret_cast<T>(base_vaddr + dyn->d_un.d_ptr);
	}
}

template <class T>
static void GetDynValue(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = dyn->d_un.d_val;
	}
}

template <class T>
static void GetDynValues(Elf64* elf, T* out, Elf64_Sxword tag) {
	for (const auto* dyn: elf->GetDynList(tag)) {
		out->push_back(dyn->d_un.d_val);
	}
}

template <class T>
static void GetDynPtr(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = dyn->d_un.d_ptr;
	}
}

static void KYTY_SYSV_ABI ProgramExitHandler() {
	Common::Singleton<RuntimeLinker>::Instance()->StopAllModules();

	LOGF("exit!!!\n");
}

template <class T>
static void GetDynModules(Elf64* elf, T* out, const char* names, Elf64_Sxword tag) {
	std::vector<uint64_t> needed_modules;
	GetDynValues(elf, &needed_modules, tag);
	for (auto need: needed_modules) {
		ModuleId id {};
		// id.id            = static_cast<int>((need >> 48u) & 0xffffu);
		EncodeId64(static_cast<uint16_t>((need >> 48u) & 0xffffu), &id.id);
		id.version_major = static_cast<int>((need >> 40u) & 0xffu);
		id.version_minor = static_cast<int>((need >> 32u) & 0xffu);
		id.name          = names + (need & 0xffffffff);
		out->push_back(id);
	}
}

template <class T>
static void GetDynLibs(Elf64* elf, T* out, const char* names, Elf64_Sxword tag) {
	std::vector<uint64_t> needed_modules;
	GetDynValues(elf, &needed_modules, tag);
	for (auto need: needed_modules) {
		LibraryId id {};
		// id.id      = static_cast<int>((need >> 48u) & 0xffffu);
		EncodeId64(static_cast<uint16_t>((need >> 48u) & 0xffffu), &id.id);
		id.version = static_cast<int>((need >> 32u) & 0xffffu);
		id.name    = names + (need & 0xffffffff);
		out->push_back(id);
	}
}

static RelocationInfo GetRelocationInfo(Elf64_Rela* r, Program* program) {
	KYTY_PROFILER_FUNCTION();

	RelocationInfo ret;
	// SymbolRecord   sr {};

	auto         type    = r->GetType();
	auto         symbol  = r->GetSymbol();
	Elf64_Sxword addend  = r->r_addend;
	auto*        symbols = program->dynamic_info->symbol_table;
	auto*        names   = program->dynamic_info->str_table;
	ret.base_vaddr       = program->base_vaddr;
	ret.vaddr            = ret.base_vaddr + r->r_offset;
	ret.bind_self        = false;

	switch (type) {
		case R_X86_64_GLOB_DAT:
		case R_X86_64_JUMP_SLOT: addend = 0; [[fallthrough]];
		case R_X86_64_64: {
			auto         sym          = symbols[symbol];
			auto         bind         = sym.GetBind();
			auto         sym_type     = sym.GetType();
			uint64_t     symbol_vaddr = 0;
			SymbolRecord sr {};
			switch (sym_type) {
				case STT_NOTYPE: ret.type = SymbolType::NoType; break;
				case STT_FUNC: ret.type = SymbolType::Func; break;
				case STT_OBJECT: ret.type = SymbolType::Object; break;
				default: EXIT("unknown symbol type: %d\n", (int)sym_type);
			}
			switch (bind) {
				case STB_LOCAL:
					symbol_vaddr = ret.base_vaddr + sym.st_value;
					ret.bind     = BindType::Local;
					break;
				case STB_GLOBAL: ret.bind = BindType::Global; [[fallthrough]];
				case STB_WEAK: {
					ret.bind = (ret.bind == BindType::Unknown ? BindType::Weak : ret.bind);
					ret.name = names + sym.st_name;
					program->rt->Resolve(ret.name, ret.type, program, &sr, &ret.bind_self);
					symbol_vaddr = sr.vaddr;
				} break;
				default: EXIT("unknown bind: %d\n", (int)bind);
			}
			ret.resolved = (symbol_vaddr != 0);
			ret.value    = (ret.resolved ? symbol_vaddr + addend : 0);
			ret.name     = sr.name;
			ret.dbg_name = sr.dbg_name;
		} break;
		case R_X86_64_RELATIVE:
			ret.value    = ret.base_vaddr + addend;
			ret.resolved = true;
			break;
		case R_X86_64_DTPMOD64:
			ret.value    = reinterpret_cast<uint64_t>(program);
			ret.resolved = true;
			ret.type     = SymbolType::TlsModule;
			ret.bind     = BindType::Local;
			ret.dbg_name = Common::PathToString(program->file_name);
			break;
		default: EXIT("unknown type: %d\n", (int)type);
	}

	return ret;
}

static bool RelocateRecord(uint32_t index, Elf64_Rela* r, Program* program, bool jmprela_table) {
	KYTY_PROFILER_FUNCTION();

	const auto ri      = GetRelocationInfo(r, program);
	auto       value   = ri.value;
	bool       stubbed = false;
	if (!ri.resolved) {
		const bool weak = ri.bind == BindType::Weak || !program->fail_if_global_not_resolved;
		if (!weak) {
			LOGF("Stubbed: %s\n",
			     fmt::format("[{:016x}] <- {:016x}, {}, {}, {}, {}", ri.vaddr, ri.value, ri.name,
			                 magic_enum::enum_name(ri.type), magic_enum::enum_name(ri.bind),
			                 ri.dbg_name).c_str());
		}
		if (ri.type == SymbolType::Object) {
			value = g_invalid_memory;
		} else {
			value   = RegisterStubbedImport(index, program, ri);
			stubbed = true;
		}
	}

	const bool patched = PatchGuestMemory64(ri.vaddr, value);
	if (patched && stubbed) {
		LOGF("Relocate: unresolved %s import patched to stub [%u] [%016" PRIx64 "] <- %016" PRIx64
		     ", %s, %s, %s, %s\n",
		     jmprela_table ? "PLT" : "non-PLT", index, ri.vaddr, value, ri.name.c_str(),
		     magic_enum::enum_name(ri.type), magic_enum::enum_name(ri.bind),
		     Common::PathToString(program->file_name).c_str());
	}

	if (program->dbg_print_reloc && patched && !ri.bind_self &&
	    (ri.bind == BindType::Global || ri.bind == BindType::Weak ||
	     ri.type == SymbolType::TlsModule)) {
		LOGF("Relocate: %s\n",
		     fmt::format("[{:016x}] <- {:016x}, {}, {}, {}, {}", ri.vaddr, ri.value, ri.name,
		                 magic_enum::enum_name(ri.type), magic_enum::enum_name(ri.bind),
		                 ri.dbg_name).c_str());
	}
	return ri.resolved;
}

static void ForEachRelocation(Program* program, auto&& func) {
	const auto& info = *program->dynamic_info;
	for (uint32_t i = 0; i < info.rela_table_total_size / sizeof(Elf64_Rela); ++i) {
		func(info.rela_table + i, i, false);
	}
	for (uint32_t i = 0; i < info.jmprela_table_size / sizeof(Elf64_Rela); ++i) {
		func(info.jmprela_table + i, i, true);
	}
}

static KYTY_MS_ABI uint8_t* TlsMainGetAddr() {
	EXIT_IF(g_tls_main_program == nullptr);

	if (g_tls_cached_main_program == g_tls_main_program && g_tls_cached_main_tcb != nullptr) {
		return g_tls_cached_main_tcb;
	}

	g_tls_cached_main_program = g_tls_main_program;
	g_tls_cached_main_tcb =
	    RuntimeLinker::TlsGetAddr(g_tls_main_program) + g_tls_main_program->tls.tcb_offset;
	return g_tls_cached_main_tcb;
}

static void PatchProgram(Program* program, uint64_t address, uint64_t size) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->elf == nullptr);

	if (size >= 12) {
		// Replace guest stack-canary/errno stores through fs:[0x28] with nops.
		// Windows x64 cannot host guest FS directly, and an unpatched shared-library access faults
		// at address 0x28.
		const uint8_t fs_store_pattern[8] = {0x64, 0xc7, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00};
		auto*         start_ptr           = reinterpret_cast<uint8_t*>(address);
		auto*         end_ptr             = start_ptr + size - 12;

		for (auto* ptr = start_ptr; ptr <= end_ptr;) {
			ptr = static_cast<uint8_t*>(
			    std::memchr(ptr, fs_store_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
			if (ptr == nullptr) {
				break;
			}
			if (memcmp(ptr, fs_store_pattern, sizeof(fs_store_pattern)) == 0) {
				if (!Config::PerfModeEnabled()) {
					LOGF("Patch fs:[0x28] store at addr: [%016" PRIx64 "]\n",
					     reinterpret_cast<uint64_t>(ptr));
				}
				if (ptr + 16 < start_ptr + size && ptr[12] == 0xcd && ptr[13] == 0x45 &&
				    ptr[14] == 0x90 && ptr[15] == 0x0f && ptr[16] == 0x0b) {
					ptr[0] = 0x5d; // pop rbp
					ptr[1] = 0xc3; // ret
					std::memset(ptr + 2, 0x90, 15);
					ptr += 17;
					continue;
				}
				std::memset(ptr, 0x90, 12);
				ptr += 12;
				continue;
			}
			++ptr;
		}
	}

	if (!program->elf->IsShared() && program->tls.handler_vaddr != 0 &&
	    size >= Jit::Call9::GetSize()) {
		// Replace:
		//   66 66 66
		//   mov <reg>, qword ptr fs:[0x00]
		// with:
		//   call <handler>
		//   mov <reg>,rax
		//   nop ...
		const uint8_t tls_pattern[5]       = {0x64, 0x48, 0x8B, 0x00, 0x25};
		const uint8_t zero_displacement[4] = {};

		EXIT_IF(Jit::Call9::GetSize() != 9);

		auto* start_ptr = reinterpret_cast<uint8_t*>(address);
		auto* end_ptr   = start_ptr + size - Jit::Call9::GetSize();

		for (auto* inst_ptr = start_ptr; inst_ptr <= end_ptr;) {
			inst_ptr = static_cast<uint8_t*>(
			    std::memchr(inst_ptr, tls_pattern[0], static_cast<size_t>(end_ptr - inst_ptr + 1)));
			if (inst_ptr == nullptr) {
				break;
			}

			const uint8_t modrm = inst_ptr[3];
			if (memcmp(inst_ptr, tls_pattern, 3) == 0 && (modrm & 0xc7u) == 0x04u &&
			    inst_ptr[4] == tls_pattern[4] &&
			    memcmp(inst_ptr + 5, zero_displacement, sizeof(zero_displacement)) == 0) {
				if (!Config::PerfModeEnabled()) {
					LOGF("Patch tls at addr: [%016" PRIx64 "]\n",
					     reinterpret_cast<uint64_t>(inst_ptr));
				}

				const auto reg = (modrm >> 3u) & 7u;
				EXIT_NOT_IMPLEMENTED(reg == 4u);

				// A raw scan can encounter a 0x66 in the preceding instruction, so do not
				// overwrite it. Call9 starts with REX.W to neutralize genuine 0x66
				// prefixes on AMD processors (before it could turn E8 into callw 16bit).
				auto* code = new (inst_ptr) Jit::Call9;
				code->SetFunc(reg == 0
				                  ? program->tls.handler_vaddr
				                  : program->tls.handler_vaddr + Jit::TlsRegStub::GetOffset(reg));
				inst_ptr += Jit::Call9::GetSize();
				continue;
			}
			++inst_ptr;
		}
	}

	// Patch GPU occlusion buffer readback spin deadlock for Marvel's Wolverine (PPSA03671):
	// In Insomniac engine games, MainThread waits for the GPU to downsample and write back
	// the occlusion depth buffer to CPU guest memory. If the GPU compute results are not
	// coherent with CPU guest memory, MainThread spins at 0x3e5e9a0..0x3e5e9c4 indefinitely.
	// Bypassing the wait directly to 0x3e5ea10 allows MainThread to immediately proceed,
	// kick off RenderThread, and maintain continuous frame presentation.
	// This patch is strictly scoped to Wolverine (PPSA03671) to ensure zero side effects on other games.
	std::string title_id;
	const bool is_wolverine =
	    SystemContentParamSfoGetString("TITLE_ID", &title_id) && title_id == "PPSA03671";
	if (is_wolverine) {
	const uint8_t occlusion_spin_pattern[13] = {
	    0x80, 0xbb, 0x8a, 0x01, 0x00, 0x00, 0x00, // cmp byte ptr [rbx + 0x18a], 0
	    0x0f, 0x84, 0xb8, 0x00, 0x00, 0x00         // je +0xb8 (0x3e5e9f8)
	};
	if (is_wolverine && size >= sizeof(occlusion_spin_pattern)) {
		auto* start_ptr = reinterpret_cast<uint8_t*>(address);
		auto* end_ptr   = start_ptr + size - sizeof(occlusion_spin_pattern);
		for (auto* ptr = start_ptr; ptr <= end_ptr;) {
			ptr = static_cast<uint8_t*>(
			    std::memchr(ptr, occlusion_spin_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
			if (ptr == nullptr) {
				break;
			}
			if (std::memcmp(ptr, occlusion_spin_pattern, sizeof(occlusion_spin_pattern)) == 0) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Patch: bypassed occlusion readback spin lock at [0x{:016x}]\n",
				    reinterpret_cast<uint64_t>(ptr)));
				// Replace with:
				//   jmp +0xd8 (0x3e5ea10) -> e9 d8 00 00 00
				//   nop * 8               -> 90 90 90 90 90 90 90 90
				const uint8_t patch[13] = {
				    0xe9, 0xd8, 0x00, 0x00, 0x00,
				    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90
				};
				std::memcpy(ptr, patch, sizeof(patch));
				ptr += sizeof(patch);
				continue;
			}
			++ptr;
		}

		// Patch 2: Invalid handle / null dereference in RenderPass loop at 0x03af54eb
		// During RenderThread pass setup (0x03af5300 called from 0x03f16574), each element loads an object
		// pointer from [r12 + rax*8 + 0x90]. If the resource is an uninitialized handle / index (e.g. 0x33),
		// `test rax, rax; je 0x903af5520` falls through and dereferences [rax + 0x40], crashing at 0x73.
		// We patch 0x03af54e6 (offset 17) to jump to the adjacent 14-byte NOP padding at 0x03af5512 (offset 61),
		// which checks `cmp rax, 0x10000`. If rax < 0x10000, it clears rax to 0 and jumps to 0x03af5520 (safe null path).
		// If rax >= 0x10000, it jumps back to 0x03af54eb to proceed normally.
		const uint8_t render_handle_pattern[75] = {
		    // 0x00: movzx eax, word ptr [r12 + 0x92]
		    0x41, 0x0f, 0xb7, 0x84, 0x24, 0x92, 0x00, 0x00, 0x00,
		    // 0x09: mov rax, qword ptr [r12 + rax*8 + 0x90]
		    0x49, 0x8b, 0x84, 0xc4, 0x90, 0x00, 0x00, 0x00,
		    // 0x11: test rax, rax
		    0x48, 0x85, 0xc0,
		    // 0x14: je +0x35 (0x03af5520)
		    0x74, 0x35,
		    // 0x16: mov rdx, qword ptr [rax + 0x40]
		    0x48, 0x8b, 0x50, 0x40,
		    // 0x1a: movzx ecx, byte ptr [rip + 0x83bb31a]
		    0x0f, 0xb6, 0x0d, 0x1a, 0xb3, 0x3b, 0x08,
		    // 0x21: test rdx, rdx
		    0x48, 0x85, 0xd2,
		    // 0x24: je +0x2c (0x03af5527)
		    0x74, 0x2c,
		    // 0x26: xor esi, esi; test cl, cl; sete sil
		    0x31, 0xf6, 0x84, 0xc9, 0x40, 0x0f, 0x94, 0xc6,
		    // 0x2e: lea esi, [rsi*4 - 9]; and dword ptr [rdx + 0xa0], esi; jmp +0x15
		    0x8d, 0x34, 0xb5, 0xf7, 0xff, 0xff, 0xff,
		    0x21, 0xb2, 0xa0, 0x00, 0x00, 0x00,
		    0xeb, 0x15,
		    // 0x3d: nop word ptr cs:[rax + rax] (14 bytes)
		    0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
		};
		if (size >= sizeof(render_handle_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(render_handle_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(
				    std::memchr(ptr, render_handle_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) {
					break;
				}
				if (std::memcmp(ptr, render_handle_pattern, sizeof(render_handle_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: fixed RenderThread invalid handle dereference at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr + 17)));
					// Patch 0x03af54e6: jmp +0x27 (to trampoline at offset 61)
					const uint8_t jmp_to_trampoline[5] = {0xe9, 0x27, 0x00, 0x00, 0x00};
					std::memcpy(ptr + 17, jmp_to_trampoline, sizeof(jmp_to_trampoline));

					// Trampoline at 0x03af5512 (14 bytes replacing NOP padding):
					//   cmp rax, 0x10000 -> 48 3d 00 00 01 00
					//   jae +4           -> 73 04 (jmp to 0x03af54eb)
					//   xor eax, eax     -> 31 c0
					//   jmp +2           -> eb 02 (jmp to 0x03af5520)
					//   jmp -0x35        -> eb cb (jmp to 0x03af54eb)
					const uint8_t trampoline[14] = {
					    0x48, 0x3d, 0x00, 0x00, 0x01, 0x00,
					    0x73, 0x04,
					    0x31, 0xc0,
					    0xeb, 0x02,
					    0xeb, 0xcb
					};
					std::memcpy(ptr + 61, trampoline, sizeof(trampoline));
					ptr += sizeof(render_handle_pattern);
					continue;
				}
				++ptr;
			}
		}

		// The live black-screen capture placed Wolverine's Render Thread in the matching
		// timeline-fence pause loop, so test this one Wolverine-only bypass in isolation.
		constexpr bool enable_timeline_fence_bypass = true;
		if (enable_timeline_fence_bypass) {
		// Patch 3: GPU timeline fence spin-wait deadlock in RenderThread at 0x03f31884
		// At the end of each frame, RenderThread waits for the GPU timeline fence to advance
		// to `r13` (0x3f3187b: mov rax, [rbx]; mov rdi, rbx; call [rax + 0x38]; cmp rax, r13; jae 0x3f31908).
		// Because the GPU timeline writeback does not signal `[rax + 0x38] >= r13` synchronously in Kyty,
		// RenderThread spins indefinitely in a 50x PAUSE loop, dropping flips to 0 and leaving
		// the presentation window stuck on the initial black frame.
		// Patching `jae 0x3f31908` (0x73 0x7f) to `jmp 0x3f31908` (0xeb 0x7f) treats the fence wait
		// as immediately satisfied after the query call, allowing RenderThread to proceed,
		// emit continuous flips, unblock MainThread, and present the rendered 3D frames.
		const uint8_t fence_wait_pattern[14] = {
		    0x48, 0x8b, 0x03, // mov rax, qword ptr [rbx]
		    0x48, 0x89, 0xdf, // mov rdi, rbx
		    0xff, 0x50, 0x38, // call qword ptr [rax + 0x38]
		    0x4c, 0x39, 0xe8, // cmp rax, r13
		    0x73, 0x7f        // jae +0x7f (0x3f31908)
		};
		if (size >= sizeof(fence_wait_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(fence_wait_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(
				    std::memchr(ptr, fence_wait_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) {
					break;
				}
				if (std::memcmp(ptr, fence_wait_pattern, sizeof(fence_wait_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked GPU timeline fence spin-wait at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr + 12)));
					ptr[12] = 0xeb; // jae -> jmp
					ptr += sizeof(fence_wait_pattern);
					continue;
				}
				++ptr;
			}
		}
		}

		// Keep the timestamp-query wait intact while isolating the no-present regression.
		// This bypass lets the guest continue after SCE_STATUS_BUSY without a ready result,
		// which can race query-buffer reuse and render submission. Re-enable only after the
		// query producer path is understood.
		constexpr bool enable_timestamp_busy_bypass = false;
		if (enable_timestamp_busy_bypass) {
		// Patch the specific busy-query loop captured in Wolverine's Render Thread at
		// eboot.bin!0x3f0d312. The guest query returns SCE_STATUS_BUSY (eax == 1), so its
		// jne branch falls into an unbounded pause loop and frame presentation stops.
		const uint8_t timestamp_busy_pattern[15] = {
		    0xff, 0x90, 0x50, 0x03, 0x00, 0x00, // call qword ptr [rax + 0x350]
		    0x83, 0xf8, 0x01,                   // cmp eax, 1
		    0x0f, 0x85, 0xa6, 0x00, 0x00, 0x00  // jne past the busy-wait loop
		};
		if (size >= sizeof(timestamp_busy_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(timestamp_busy_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(
				    std::memchr(ptr, timestamp_busy_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) {
					break;
				}
				if (std::memcmp(ptr, timestamp_busy_pattern, sizeof(timestamp_busy_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: bypassed Wolverine busy timestamp-query loop at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr + 9)));
					const uint8_t jump_past_wait[6] = {0xe9, 0xa7, 0x00, 0x00, 0x00, 0x90};
					std::memcpy(ptr + 9, jump_past_wait, sizeof(jump_past_wait));
					ptr += sizeof(timestamp_busy_pattern);
					continue;
				}
				++ptr;
			}
		}
		}

		// Preserve the exact startup workaround used by the measured 60-FPS baseline.
		// This skips only the branch; it must not fabricate a completion-counter write.
		constexpr bool enable_render_completion_counter_bypass = true;
		if (enable_render_completion_counter_bypass) {
		// This render completion wait stays at [R15+0xc0] == 0 while its target
		// [R15+0xe0] is 1 in the captured Wolverine startup stall. The loop calls the
		// guest wait helper and then branches back while those sequence counters differ.
		// Bypass only this exact loop shape for Wolverine so we can test whether the
		// missing completion notification is what prevents the first frame from flipping.
		const uint8_t render_completion_wait_pattern[16] = {
		    0x49, 0x8b, 0x87, 0xe0, 0x00, 0x00, 0x00, // mov rax, [r15 + 0xe0]
		    0x49, 0x3b, 0x87, 0xc0, 0x00, 0x00, 0x00, // cmp rax, [r15 + 0xc0]
		    0x75, 0x81                                // jne back to wait loop
		};
		if (size >= sizeof(render_completion_wait_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(render_completion_wait_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(
				    ptr, render_completion_wait_pattern[0],
				    static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) {
					break;
				}
				if (reinterpret_cast<uint64_t>(ptr + 14) == 0x0000000903f65b1dull &&
				    std::memcmp(ptr, render_completion_wait_pattern,
				                sizeof(render_completion_wait_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: bypassed Wolverine render completion counter wait at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr + 14)));
					ptr[14] = 0x90;
					ptr[15] = 0x90;
					ptr += sizeof(render_completion_wait_pattern);
					continue;
				}
				++ptr;
			}
		}
		}

		constexpr bool enable_render_query_bypass = true;
		if (enable_render_query_bypass) {
	// Patch 7: RenderThread GPU render-target query completion spin loop at 0x03f01717 / 0x03f31aa7
		// In the frame render pass completion stage (0x03f31960 called by 0x03f6eed3 in RenderThread),
		// the engine checks the GPU render target completion status via [rax + 0x350].
		// If the query returns 1 (SCE_STATUS_BUSY) or rdi is 0, RenderThread enters a 50x PAUSE loop
		// and spins indefinitely (stalling at 0x3f017fe / 0x3f31b8e), halting frame presentation (flips/s = 0).
		// By patching the branches:
		//   1. 0x03f31ab9: je +0x138 (0x0f 0x84 0x38 0x01 0x00 0x00) -> jmp +0x139 (0xe9 0x39 0x01 0x00 0x00 0x90)
		//   2. 0x03f31bc7: je -0x10d (0x0f 0x84 0xf3 0xfe 0xff 0xff) -> je +0x2a (0x0f 0x84 0x2a 0x00 0x00 0x00)
		//   3. 0x03f31bea: jne -0x130 (0x0f 0x85 0xd0 0xfe 0xff 0xff) -> nop * 6 (0x90 * 6)
		// RenderThread immediately treats the query as satisfied, returns 1 (success) to the caller,
		// and continues the render loop to emit continuous flips.
		const uint8_t render_query_pattern[24] = {
		    0x41, 0xb8, 0x01, 0x00, 0x00, 0x00,       // mov r8d, 1
		    0x48, 0x8b, 0x4d, 0xa8,                   // mov rcx, qword ptr [rbp - 0x58]
		    0xff, 0x90, 0x50, 0x03, 0x00, 0x00,       // call qword ptr [rax + 0x350]
		    0x85, 0xc0,                               // test eax, eax
		    0x0f, 0x84, 0x38, 0x01, 0x00, 0x00        // je +0x138 (0x3f31bf7)
		};
		if (size >= sizeof(render_query_pattern) + 0x149) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(render_query_pattern) - 0x149;
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(
				    std::memchr(ptr, render_query_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) {
					break;
				}
				if (std::memcmp(ptr, render_query_pattern, sizeof(render_query_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked RenderThread GPU render-target query spin loop at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					// 1. 0x03f31ab9: jmp +0x139 (to 0x3f31bf7)
					const uint8_t jmp_success[6] = {0xe9, 0x39, 0x01, 0x00, 0x00, 0x90};
					std::memcpy(ptr + 18, jmp_success, sizeof(jmp_success));

					// 2. 0x03f31bc7: redirect null check jump to 0x3f31bf7
					const uint8_t je_success[6] = {0x0f, 0x84, 0x2a, 0x00, 0x00, 0x00};
					std::memcpy(ptr + 0x120, je_success, sizeof(je_success));

					// 3. 0x03f31bea: NOP out retry loop jump
					const uint8_t nops[6] = {0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
					std::memcpy(ptr + 0x143, nops, sizeof(nops));

					ptr += sizeof(render_query_pattern) + 0x149;
					continue;
				}
				++ptr;
			}
		}
		}

		// Keep the other speculative sync bypasses disabled; they can let the guest reuse
		// command buffers or present frames before GPU work has completed.
		constexpr bool enable_frame_timing_query_bypass = false;
		if (enable_frame_timing_query_bypass) {
		// Patch 4: GPU timestamp query readback spin-wait deadlock in RenderThread at 0x03f0d070
		// In the frame completion pipeline (0x03f38540 right after 0x03f013f0), RenderThread invokes 0x03f0d070
		// to query 4 GPU timestamp stages. If any query returns 1 (SCE_STATUS_BUSY), it spins in a 50x PAUSE loop
		// indefinitely, preventing RenderThread from signaling MainThread and emitting flips.
		// Replacing the entry of 0x03f0d070 with an immediate buffer-index toggle and return:
		//   movzx eax, byte ptr [rdi + 0x43]  (0f b6 47 43)
		//   xor al, 1                         (34 01)
		//   mov byte ptr [rdi + 0x43], al     (88 47 43)
		//   shl al, 2                         (c0 e0 02)
		//   mov byte ptr [rdi + 0x44], al     (88 47 44)
		//   ret                               (c3)
		// completes the timestamp profiling stage instantly and allows RenderThread to signal MainThread
		// and present continuous flips.
		const uint8_t timestamp_query_pattern[24] = {
		    0x55, 0x48, 0x89, 0xe5,             // push rbp; mov rbp, rsp
		    0x41, 0x57, 0x41, 0x56, 0x41, 0x55, // push r15; push r14; push r13
		    0x41, 0x54, 0x53,                   // push r12; push rbx
		    0x48, 0x83, 0xec, 0x38,             // sub rsp, 0x38
		    0x48, 0x8b, 0x05, 0x20, 0xd8, 0x8f, 0x05 // mov rax, qword ptr [rip + 0x58fd820]
		};
		if (size >= sizeof(timestamp_query_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(timestamp_query_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(
				    std::memchr(ptr, timestamp_query_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) {
					break;
				}
				if (std::memcmp(ptr, timestamp_query_pattern, sizeof(timestamp_query_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: bypassed GPU timestamp query readback spin loop at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					const uint8_t fast_ret[16] = {
					    0x0f, 0xb6, 0x47, 0x43, // movzx eax, byte ptr [rdi + 0x43]
					    0x34, 0x01,             // xor al, 1
					    0x88, 0x47, 0x43,       // mov byte ptr [rdi + 0x43], al
					    0xc0, 0xe0, 0x02,       // shl al, 2
					    0x88, 0x47, 0x44,       // mov byte ptr [rdi + 0x44], al
					    0xc3                    // ret
					};
					std::memcpy(ptr, fast_ret, sizeof(fast_ret));
					ptr += sizeof(timestamp_query_pattern);
					continue;
				}
				++ptr;
			}
		}

		// Patch 5: RenderThread GPU query readback spin loop in frame pacing at 0x03fb4be5 / 0x03fb4cdb
		// In the frame pacing / GPU timing subsystem (0x03fb4920 called by 0x03f5256f in RenderThread),
		// the engine executes two query poll calls to [rax + 0x350] to measure frame time deltas.
		// If either query returns 1 (SCE_STATUS_BUSY), the engine enters a 50x PAUSE loop and spins indefinitely,
		// preventing RenderThread from returning to the presentation loop and stopping flips (flips/s = 0).
		// By patching the status branches:
		//   1. 0x03fb4bee: jne +0xb5 (0x0f 0x85 0xb5 0x00 0x00 0x00) -> jmp +0xb6 (0xe9 0xb6 0x00 0x00 0x00 0x90)
		//   2. 0x03fb4ca3: je -0x99 (0x0f 0x84 0x67 0xff 0xff 0xff)  -> nop * 6 (0x90 * 6)
		//   3. 0x03fb4ce4: jne +0xaf (0x0f 0x85 0xaf 0x00 0x00 0x00) -> jmp +0xb0 (0xe9 0xb0 0x00 0x00 0x00 0x90)
		//   4. 0x03fb4d93: je -0x99 (0x0f 0x84 0x67 0xff 0xff 0xff)  -> nop * 6 (0x90 * 6)
		// RenderThread immediately bypasses both spin loops, safely finishes the frame pacing stage,
		// and returns to emit continuous flips.
		const uint8_t query_spin_pattern[29] = {
		    0xff, 0x90, 0x50, 0x03, 0x00, 0x00,             // call qword ptr [rax + 0x350]
		    0x83, 0xf8, 0x01,                               // cmp eax, 1
		    0x0f, 0x85, 0xb5, 0x00, 0x00, 0x00,             // jne +0xb5 (0x3fb4ca9)
		    0x4c, 0x8d, 0x3d, 0x1e, 0x1e, 0x69, 0x03,       // lea r15, [rip + 0x3691e1e]
		    0x4c, 0x8d, 0xa5, 0x78, 0xff, 0xff, 0xff        // lea r12, [rbp - 0x88]
		};
		if (size >= sizeof(query_spin_pattern) + 0x1b4) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(query_spin_pattern) - 0x1b4;
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(
				    std::memchr(ptr, query_spin_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) {
					break;
				}
				if (std::memcmp(ptr, query_spin_pattern, sizeof(query_spin_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked frame-pacing query spin loops at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					// 1. 0x03fb4bee: jmp +0xb6 (0x3fb4ca9)
					const uint8_t jmp_query1[6] = {0xe9, 0xb6, 0x00, 0x00, 0x00, 0x90};
					std::memcpy(ptr + 9, jmp_query1, sizeof(jmp_query1));

					// 2. 0x03fb4ca3: NOP out retry loop 1 jump
					const uint8_t nops[6] = {0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
					std::memcpy(ptr + 0xbe, nops, sizeof(nops));

					// 3. 0x03fb4ce4: jmp +0xb0 (0x3fb4d99)
					const uint8_t jmp_query2[6] = {0xe9, 0xb0, 0x00, 0x00, 0x00, 0x90};
					std::memcpy(ptr + 0xff, jmp_query2, sizeof(jmp_query2));

					// 4. 0x03fb4d93: NOP out retry loop 2 jump
					std::memcpy(ptr + 0x1ae, nops, sizeof(nops));

					ptr += sizeof(query_spin_pattern) + 0x1b4;
					continue;
				}
				++ptr;
			}
		}
		}

		constexpr bool enable_ring_buffer_bypass = false;
		if (enable_ring_buffer_bypass) {
		// Patch 6: Command Buffer Ring Allocator spin-wait loops in Insomniac Engine
		// In the command buffer ring allocator (0x3f95e9f, 0x3f9604f, 0x3fa0baf, 0x3fa0d5f),
		// RenderThread waits for the GPU to update its completion offset [r15 + 0xe0]
		// to match the ring allocation offset [r15 + 0xc0].
		// If unequal, RenderThread enters a 50x PAUSE loop and spins indefinitely, stalling flips.
		// Original pattern (16 bytes):
		//   49 8b 87 e0 00 00 00  ; mov rax, qword ptr [r15 + 0xe0]
		//   49 3b 87 c0 00 00 00  ; cmp rax, qword ptr [r15 + 0xc0]
		//   75 81                 ; jne -0x7f (back to pause loop)
		// Replacement pattern (16 bytes):
		//   49 8b 87 c0 00 00 00  ; mov rax, qword ptr [r15 + 0xc0]
		//   49 89 87 e0 00 00 00  ; mov qword ptr [r15 + 0xe0], rax
		//   90 90                 ; nop; nop
		// This advances the GPU completion offset to match the allocation offset and eliminates
		// the backward spin branch, allowing RenderThread to proceed without stalling.
		const uint8_t ring_alloc_pattern[16] = {
		    0x49, 0x8b, 0x87, 0xe0, 0x00, 0x00, 0x00, // mov rax, qword ptr [r15 + 0xe0]
		    0x49, 0x3b, 0x87, 0xc0, 0x00, 0x00, 0x00, // cmp rax, qword ptr [r15 + 0xc0]
		    0x75, 0x81                                 // jne -0x7f
		};
		if (size >= sizeof(ring_alloc_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(ring_alloc_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(
				    std::memchr(ptr, ring_alloc_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) {
					break;
				}
				if (std::memcmp(ptr, ring_alloc_pattern, sizeof(ring_alloc_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked command buffer ring allocator spin loop at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					const uint8_t sync_and_fallthrough[16] = {
					    0x49, 0x8b, 0x87, 0xc0, 0x00, 0x00, 0x00, // mov rax, qword ptr [r15 + 0xc0]
					    0x49, 0x89, 0x87, 0xe0, 0x00, 0x00, 0x00, // mov qword ptr [r15 + 0xe0], rax
					    0x90, 0x90                                 // nop; nop
					};
					std::memcpy(ptr, sync_and_fallthrough, sizeof(sync_and_fallthrough));
					ptr += sizeof(ring_alloc_pattern);
					continue;
				}
				++ptr;
			}
		}
		}

		constexpr bool enable_timeline_fence_bypasses = true;
		if (enable_timeline_fence_bypasses) {
		// Patch 8: Unblock remaining timeline fence and query spin loops (Sites A - G)
		// Exhaustive resolution of all 23 callers of the 50x PAUSE yield function (0x3f39130).

		// Site A entry (0x3fa6be5 / vaddr 0x3f7685c): timeline fence wait in swapchain present
		const uint8_t site_a_entry_pattern[22] = {
		    0x4d, 0x8b, 0x2f,                         // mov r13, qword ptr [r15]
		    0x4c, 0x89, 0xff,                         // mov rdi, r15
		    0x49, 0x89, 0xc4,                         // mov r12, rax
		    0x41, 0xff, 0x55, 0x38,                   // call qword ptr [r13 + 0x38]
		    0x48, 0x39, 0xd8,                         // cmp rax, rbx
		    0x0f, 0x84, 0x8b, 0x00, 0x00, 0x00        // je +0x8b (0x3f768fd)
		};
		if (size >= sizeof(site_a_entry_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(site_a_entry_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(ptr, site_a_entry_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) break;
				if (std::memcmp(ptr, site_a_entry_pattern, sizeof(site_a_entry_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked GPU timeline fence entry wait (Site A entry) at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					const uint8_t jmp_ret[6] = {0xe9, 0x8c, 0x00, 0x00, 0x00, 0x90};
					std::memcpy(ptr + 16, jmp_ret, sizeof(jmp_ret));
					ptr += sizeof(site_a_entry_pattern);
					continue;
				}
				++ptr;
			}
		}

		// Site A loop (0x3fa6c85 / vaddr 0x3f768f5): timeline fence wait (cmp rax, rbx; jne -0x7d)
		const uint8_t site_a_pattern[8] = {
		    0xff, 0x50, 0x38, // call qword ptr [rax + 0x38]
		    0x48, 0x39, 0xd8, // cmp rax, rbx
		    0x75, 0x83        // jne -0x7d
		};
		if (size >= sizeof(site_a_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(site_a_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(ptr, site_a_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) break;
				if (std::memcmp(ptr, site_a_pattern, sizeof(site_a_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked GPU timeline fence spin-wait (Site A) at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					ptr[6] = 0x90; // nop
					ptr[7] = 0x90; // nop
					ptr += sizeof(site_a_pattern);
					continue;
				}
				++ptr;
			}
		}

		constexpr bool enable_sites_b_through_g = false;
		if (enable_sites_b_through_g) {
		// Site B (0x3fa69e5 / vaddr 0x3f76655): timeline fence wait (test rax, rax; jne -0x7d)
		const uint8_t site_b_pattern[8] = {
		    0xff, 0x50, 0x38, // call qword ptr [rax + 0x38]
		    0x48, 0x85, 0xc0, // test rax, rax
		    0x75, 0x83        // jne -0x7d
		};
		if (size >= sizeof(site_b_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(site_b_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(ptr, site_b_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) break;
				if (std::memcmp(ptr, site_b_pattern, sizeof(site_b_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked GPU timeline fence spin-wait (Site B) at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					ptr[6] = 0x90; // nop
					ptr[7] = 0x90; // nop
					ptr += sizeof(site_b_pattern);
					continue;
				}
				++ptr;
			}
		}

		// Site C (0x3f92ac5 / vaddr 0x3f62735): timeline fence wait (cmp rax, rbx; jbe -0x7d)
		const uint8_t site_c_pattern[8] = {
		    0xff, 0x50, 0x38, // call qword ptr [rax + 0x38]
		    0x48, 0x39, 0xd8, // cmp rax, rbx
		    0x76, 0x83        // jbe -0x7d
		};
		if (size >= sizeof(site_c_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(site_c_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(ptr, site_c_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) break;
				if (std::memcmp(ptr, site_c_pattern, sizeof(site_c_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked GPU timeline fence spin-wait (Site C) at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					ptr[6] = 0x90; // nop
					ptr[7] = 0x90; // nop
					ptr += sizeof(site_c_pattern);
					continue;
				}
				++ptr;
			}
		}

		// Site D (0x3f3ab33 / vaddr 0x3f0a7a3): timeline fence wait loop
		const uint8_t site_d_pattern[29] = {
		    0xff, 0x50, 0x38,                         // call qword ptr [rax + 0x38]
		    0x89, 0x43, 0x04,                         // mov dword ptr [rbx + 4], eax
		    0x8b, 0x13,                               // mov edx, dword ptr [rbx]
		    0x48, 0x8b, 0x4b, 0x10,                   // mov rcx, qword ptr [rbx + 0x10]
		    0x41, 0x39, 0xd5,                         // cmp r13d, edx
		    0x0f, 0x84, 0x68, 0xff, 0xff, 0xff,       // je -0x98 (0x3f0a720)
		    0x39, 0xc2,                               // cmp edx, eax
		    0x0f, 0x85, 0x60, 0xff, 0xff, 0xff        // jne -0xa0 (0x3f0a720)
		};
		if (size >= sizeof(site_d_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(site_d_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(ptr, site_d_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) break;
				if (std::memcmp(ptr, site_d_pattern, sizeof(site_d_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked GPU timeline fence spin-wait (Site D) at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					// Set [rbx + 4] = edx (target fence value)
					ptr[4] = 0x53; // mov [rbx + 4], edx (89 53 04)
					// NOP out backward jump 1 (offset 15..20)
					std::memset(ptr + 15, 0x90, 6);
					// NOP out backward jump 2 (offset 23..28)
					std::memset(ptr + 23, 0x90, 6);
					ptr += sizeof(site_d_pattern);
					continue;
				}
				++ptr;
			}
		}

		// Site E (0x03f5d0ec / vaddr 0x03f2cd5c): timeline fence wait (cmp rax, [rbx + 0x10]; jae +0x0e)
		const uint8_t site_e_pattern[9] = {
		    0xff, 0x50, 0x38,       // call qword ptr [rax + 0x38]
		    0x48, 0x3b, 0x43, 0x10, // cmp rax, qword ptr [rbx + 0x10]
		    0x73, 0x0e              // jae +0x0e
		};
		if (size >= sizeof(site_e_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(site_e_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(ptr, site_e_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) break;
				if (std::memcmp(ptr, site_e_pattern, sizeof(site_e_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked GPU timeline fence spin-wait (Site E) at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					ptr[7] = 0xeb; // jae -> jmp
					ptr += sizeof(site_e_pattern);
					continue;
				}
				++ptr;
			}
		}

		// Site F (0x03f9181f / vaddr 0x03f6148f): pacing wait loop (test eax, eax; jg -0x7e)
		const uint8_t site_f_pattern[15] = {
		    0x8b, 0xbb, 0xcc, 0x00, 0x00, 0x00, // mov edi, dword ptr [rbx + 0xcc]
		    0xe8, 0x16, 0xff, 0xfb, 0x01,       // call 0x5f213b0
		    0x85, 0xc0,                         // test eax, eax
		    0x7f, 0x82                          // jg -0x7e
		};
		if (size >= sizeof(site_f_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(site_f_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(ptr, site_f_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) break;
				if (std::memcmp(ptr, site_f_pattern, sizeof(site_f_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked pacing spin-wait (Site F) at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					ptr[13] = 0x90; // nop
					ptr[14] = 0x90; // nop
					ptr += sizeof(site_f_pattern);
					continue;
				}
				++ptr;
			}
		}

		// Site G (0x03f31dcd / vaddr 0x03f01a3d): query completion check (cmp eax, 1; je -0x13c)
		const uint8_t site_g_pattern[15] = {
		    0xff, 0x90, 0x50, 0x03, 0x00, 0x00, // call qword ptr [rax + 0x350]
		    0x83, 0xf8, 0x01,                   // cmp eax, 1
		    0x0f, 0x84, 0xc4, 0xfe, 0xff, 0xff  // je -0x13c
		};
		if (size >= sizeof(site_g_pattern)) {
			auto* start_ptr = reinterpret_cast<uint8_t*>(address);
			auto* end_ptr   = start_ptr + size - sizeof(site_g_pattern);
			for (auto* ptr = start_ptr; ptr <= end_ptr;) {
				ptr = static_cast<uint8_t*>(std::memchr(ptr, site_g_pattern[0], static_cast<size_t>(end_ptr - ptr + 1)));
				if (ptr == nullptr) break;
				if (std::memcmp(ptr, site_g_pattern, sizeof(site_g_pattern)) == 0) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Patch: unblocked query completion spin-wait (Site G) at [0x{:016x}]\n",
					    reinterpret_cast<uint64_t>(ptr)));
					std::memset(ptr + 9, 0x90, 6); // 6x nop
					ptr += sizeof(site_g_pattern);
					continue;
				}
				++ptr;
			}
		}
		}
		}
	}
}

}

uint64_t RuntimeLinker::GetEntry() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		if (p->elf != nullptr && !p->elf->IsShared()) {
			return p->elf->GetEntry() + p->base_vaddr;
		}
	}
	return 0;
}

uint64_t RuntimeLinker::GetProcParam() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		if (p->elf != nullptr && !p->elf->IsShared()) {
			return p->proc_param_vaddr;
		}
	}
	return 0;
}

void RuntimeLinker::DbgDump(const std::string& folder) {
	KYTY_PROFILER_FUNCTION();

	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		auto folder_str = Common::FixDirectorySlash(folder);
		folder_str += Common::FilenameWithoutDirectory(Common::PathToGenericString(p->file_name));

		EXIT_IF(p->elf == nullptr);

		p->elf->DbgDump(folder_str);

		if (p->dynamic_info != nullptr) {
			EXIT_NOT_IMPLEMENTED(p->dynamic_info->symbol_table_entry_size != 0 &&
			                     p->dynamic_info->symbol_table_entry_size != sizeof(Elf64_Sym));
			EXIT_NOT_IMPLEMENTED(p->dynamic_info->rela_table_entry_size != 0 &&
			                     p->dynamic_info->rela_table_entry_size != sizeof(Elf64_Rela));
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->jmprela_table == nullptr);
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->rela_table == nullptr);
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->symbol_table == nullptr);

			if (p->dynamic_info->symbol_table != nullptr) {
				DbgDumpSymbols(folder_str, p->dynamic_info->symbol_table,
				               p->dynamic_info->symbol_table_total_size,
				               p->dynamic_info->str_table);
			}
			if (p->dynamic_info->jmprela_table != nullptr) {
				DbgDumpRela(folder_str, p->dynamic_info->jmprela_table,
				            p->dynamic_info->jmprela_table_size, p->dynamic_info->str_table,
				            "jmprela_table.txt");
			}
			if (p->dynamic_info->rela_table != nullptr) {
				DbgDumpRela(folder_str, p->dynamic_info->rela_table,
				            p->dynamic_info->rela_table_total_size, p->dynamic_info->str_table,
				            "rela_table.txt");
			}
		}

		if (p->export_symbols != nullptr) {
			p->export_symbols->DbgDump(folder_str, "export_symbols.txt");
		}
		if (p->import_symbols != nullptr) {
			p->import_symbols->DbgDump(folder_str, "import_symbols.txt");
		}
	}
}

void RuntimeLinker::RelocateAll() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		Relocate(p);
	}
}

void RuntimeLinker::RelocateProgram(Program* program) {
	Common::LockGuard lock(m_mutex);

	EXIT_IF(program == nullptr);
	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	RelocateAll();
	if (!GamePatch::ApplyPending(program)) {
		EXIT("Failed to apply pending game cheat\n");
	}
}

void RuntimeLinker::UnloadProgram(Program* program) {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	auto it = std::find(m_programs.begin(), m_programs.end(), program);
	EXIT_IF(it == m_programs.end());
	m_programs.erase(it);

	for (auto* importer: m_programs) {
		const auto& info = *importer->dynamic_info;
		const auto rela_count =
		    static_cast<uint32_t>(info.rela_table_total_size / sizeof(Elf64_Rela));
		ForEachRelocation(importer, [&](Elf64_Rela* r, uint32_t index, bool jmprela) {
			const auto bit_index = (jmprela ? rela_count : 0) + index;
			auto&      bits      = importer->rela_bits[bit_index >> 3u];
			const auto mask      = 1u << (bit_index & 7u);
			const auto type      = r->GetType();
			if ((bits & mask) == 0 || (type != R_X86_64_64 && type != R_X86_64_GLOB_DAT &&
			                           type != R_X86_64_JUMP_SLOT)) {
				return;
			}
			const auto& symbol = info.symbol_table[r->GetSymbol()];
			if (symbol.GetBind() != STB_GLOBAL && symbol.GetBind() != STB_WEAK) {
				return;
			}
			const auto value =
			    *reinterpret_cast<const uint64_t*>(importer->base_vaddr + r->r_offset);
			const auto address = value - (type == R_X86_64_64 ? r->r_addend : 0);
			if (address < program->base_vaddr ||
			    address >= program->base_vaddr + program->mapped_size) {
				return;
			}
			SymbolType symbol_type;
			switch (symbol.GetType()) {
				case STT_NOTYPE: symbol_type = SymbolType::NoType; break;
				case STT_FUNC: symbol_type = SymbolType::Func; break;
				case STT_OBJECT: symbol_type = SymbolType::Object; break;
				default: return;
			}
			const std::string name = info.str_table + symbol.st_name;
			if (program->export_symbols->FindByNid(name.substr(0, name.find('#')), symbol_type,
			                                       address) != nullptr) {
				bits &= ~mask;
				if (RelocateRecord(index, r, importer, jmprela)) {
					bits |= mask;
				}
			}
		});
	}
	DeleteProgram(program);
}

RuntimeLinker::RuntimeLinker(): m_symbols(std::make_unique<SymbolDatabase>()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RuntimeLinker::~RuntimeLinker() {
	Clear();
}

Program* RuntimeLinker::LoadProgram(const std::filesystem::path& elf_name) {
	KYTY_PROFILER_FUNCTION();

	Common::LockGuard lock(m_mutex);

	static int32_t id_seq = 0;

	LOGF("Loading: %s\n", Common::PathToString(elf_name).c_str());

	auto  program_owner = std::make_unique<Program>();
	auto* program       = program_owner.get();

	program->rt        = this;
	program->file_name = elf_name;
	program->unique_id = ++id_seq;

	program->elf = std::make_unique<Elf64>();
	program->elf->Open(elf_name);

	if (program->elf->IsValid()) {
		LoadProgramToMemory(program);
		ParseProgramDynamicInfo(program);
		CreateSymbolDatabase(program);
	} else {
		EXIT("elf is not valid: %s\n", Common::PathToString(elf_name).c_str());
	}

	m_programs.push_back(program_owner.release());

	if (!program->elf->IsShared()) {
		program->fail_if_global_not_resolved = false;
		Libs::LibKernel::SetProgName(elf_name.filename().string());
	}

	if (Common::ToLower(Common::DirectoryWithoutFilename(Common::PathToGenericString(elf_name)))
	        .ends_with("_module/")) {
		program->fail_if_global_not_resolved = false;
	}

	return program;
}

void RuntimeLinker::SaveMainProgram(const std::filesystem::path& elf_name) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		EXIT_IF(p->elf == nullptr);

		if (!p->elf->IsShared()) {
			p->elf->Save(elf_name);
			break;
		}
	}
}

void RuntimeLinker::SaveProgram(Program* program, const std::filesystem::path& elf_name) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	if (auto it = std::find(m_programs.begin(), m_programs.end(), program);
	    it != m_programs.end()) {
		EXIT_IF((*it)->elf == nullptr);

		(*it)->elf->Save(elf_name);
	} else {
		EXIT("program not found");
	}
}

void RuntimeLinker::Execute(const std::filesystem::path& game_patch) {
	KYTY_PROFILER_THREAD("Thread_Main");

	Libs::LibKernel::PthreadInitSelfForMainThread();
	auto* main_stack_top = Libs::LibKernel::PthreadCreateMainGuestStack();

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Guest code has no Windows stack probes and may jump over the guard page. Module
	// initializers execute on the host stack too, so grow it before calling any guest code.
	size_t expanded_size = 0;
	while (expanded_size < static_cast<size_t>(768) * 1024) {
		sys_dbg_stack_info_t stack {};
		SysStackUsage(stack);
		*reinterpret_cast<uint32_t*>(stack.guard_addr) = 0;
		expanded_size += stack.guard_size;
	}
#endif

	// The runtime automatically loads libc; other PRXs are requested by the application.
	const auto libc_path = Libs::LibKernel::FileSystem::GetRealFilename("/app0/sce_module/libc.prx");
	auto* libc = Common::File::IsFileExisting(libc_path) ? LoadProgram(libc_path) : nullptr;
	RelocateAll();

	if (!game_patch.empty()) {
		if (!GamePatch::Apply(game_patch, m_programs.empty() ? nullptr : m_programs.front(),
		                      m_programs)) {
			EXIT("Failed to apply game cheat\n");
		}
	}
	if (libc != nullptr && libc->dynamic_info->init_vaddr != 0) {
		StartModule(libc, 0, nullptr, nullptr);
	}

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Execute: %s\n---\n", "Main");

	if (auto entry = GetEntry(); entry != 0) {
		const auto& arguments = Config::GetGuestArguments();
		const auto params_size = offsetof(EntryParams, argv) +
		                         (arguments.size() + 2) * sizeof(const char*);
		auto* params = reinterpret_cast<EntryParams*>(
		    (reinterpret_cast<uintptr_t>(main_stack_top) - params_size - 0x100u) &
		    ~static_cast<uintptr_t>(0x0f));
		std::memset(params, 0, params_size);
		for (const auto& argument: arguments) {
			params->argv[params->argc++] = argument.c_str();
		}

		LOGF("stack_addr = %" PRIx64 "\n", reinterpret_cast<uint64_t>(params));

		RunEntry(entry, params, ProgramExitHandler,
		         reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(params) - 0x1000u));
	}
}

void RuntimeLinker::Clear() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);
	GamePatch::Clear();

	for (auto* p: m_programs) {
		DeleteProgram(p);
	}
	m_programs.clear();
	for (const auto page: g_unresolved_stub_thunk_pages) {
		EXIT_IF(!Libs::LibKernel::Memory::FreeGuestMemory(page, UNRESOLVED_STUB_PAGE_SIZE));
	}
	g_unresolved_stub_thunk_pages.clear();
	g_unresolved_stub_thunk_offset = 0;
	g_stubbed_imports.clear();
	g_unresolved_stub_call_log_count.store(0);
	if (g_invalid_memory != 0) {
		EXIT_IF(!Libs::LibKernel::Memory::FreeGuestMemory(g_invalid_memory, 4096));
		g_invalid_memory = 0;
	}
	g_tls_main_program        = nullptr;
	g_tls_cached_main_program = nullptr;
	g_tls_cached_main_tcb     = nullptr;
	g_desired_base_addr       = SYSTEM_RESERVED + CODE_BASE_OFFSET;
	m_symbols.reset();
}

void RuntimeLinker::Resolve(const std::string& name, SymbolType type, Program* program,
                            SymbolRecord* out_info, bool* bind_self) {
	KYTY_PROFILER_FUNCTION();

	Common::LockGuard lock(m_mutex);

	EXIT_IF(out_info == nullptr);

	auto ids = Common::Split(name, '#');

	if (bind_self != nullptr) {
		*bind_self = false;
	}

	if (ids.size() == 3) {
		const LibraryId* l = FindLibrary(*program, ids.at(1));
		const ModuleId*  m = FindModule(*program, ids.at(2));

		auto resolve_by_nid = [this, type](const std::string& nid, SymbolRecord* out) -> bool {
			EXIT_IF(out == nullptr);

			if (m_symbols != nullptr) {
				if (const auto* rec = m_symbols->FindByNid(nid, type); rec != nullptr) {
					*out = *rec;
					return true;
				}
			}

			for (auto* p: m_programs) {
				if (p != nullptr && p->export_symbols != nullptr) {
					if (const auto* rec = p->export_symbols->FindByNid(nid, type); rec != nullptr) {
						*out = *rec;
						return true;
					}
				}
			}

			return false;
		};

		if (l != nullptr && m != nullptr) {
			SymbolResolve sr {};
			sr.name                 = ids.at(0);
			sr.library              = l->name;
			sr.library_version      = l->version;
			sr.module               = m->name;
			sr.module_version_major = m->version_major;
			sr.module_version_minor = m->version_minor;
			sr.type                 = type;

			const SymbolRecord* rec = nullptr;

			if (m_symbols != nullptr) {
				rec = m_symbols->Find(sr);
			}

			if (rec == nullptr) {
				for (auto* p: m_programs) {
					const auto& libs = p->dynamic_info->export_libs;
					const auto& modules = p->dynamic_info->export_modules;
					if (p->export_symbols == nullptr ||
					    std::find(libs.begin(), libs.end(), *l) == libs.end() ||
					    std::find(modules.begin(), modules.end(), *m) == modules.end()) {
						continue;
					}
					rec = p->export_symbols->Find(sr);
					if (rec != nullptr) {
						if (bind_self != nullptr) {
							*bind_self = (p == program);
						}
						break;
					}
				}
			}

			if (rec == nullptr) {
				if (resolve_by_nid(sr.name, out_info)) {
					LOGF("PS5 NID fallback: %s -> %s\n", sr.name.c_str(), out_info->name.c_str());
					return;
				}
			}

			if (rec != nullptr) {
				*out_info = *rec;
			} else {
				out_info->vaddr    = 0;
				out_info->name     = SymbolDatabase::GenerateName(sr);
				out_info->dbg_name = "";
			}
		} else {
			if (resolve_by_nid(ids.at(0), out_info)) {
				LOGF("PS5 NID fallback: %s -> %s (missing lib/module metadata)\n",
				     ids.at(0).c_str(), out_info->name.c_str());
				return;
			}

			EXIT("l == nullptr || m == nullptr");
		}
	} else {
		out_info->vaddr    = 0;
		out_info->name     = name;
		out_info->dbg_name = "";
	}
}

uint64_t RuntimeLinker::ReadFromElf(Program* program, uint64_t vaddr) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->base_vaddr == 0 || program->base_size == 0);
	EXIT_IF(program->elf == nullptr);

	uint64_t ret = 0;

	const auto* ehdr = program->elf->GetEhdr();
	const auto* phdr = program->elf->GetPhdr();

	EXIT_IF(phdr == nullptr || ehdr == nullptr);

	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t segment_addr      = phdr[i].p_vaddr + program->base_vaddr;
			uint64_t segment_file_size = phdr[i].p_filesz;

			if (vaddr >= segment_addr && vaddr < segment_addr + segment_file_size) {
				program->elf->LoadSegment(reinterpret_cast<uint64_t>(&ret),
				                          phdr[i].p_offset + vaddr - segment_addr, sizeof(ret));
				break;
			}
		}
	}

	return ret;
}

Program* RuntimeLinker::FindProgramById(int32_t id) {
	Common::LockGuard lock(m_mutex);

	// Id 0 is reserved for main program
	if (id == 0 && !m_programs.empty()) {
		return m_programs.front();
	}

	for (auto* p: m_programs) {
		if (p->unique_id == id) {
			return p;
		}
	}

	return nullptr;
}

Program* RuntimeLinker::FindProgramByFileName(const std::filesystem::path& elf_name) {
	Common::LockGuard lock(m_mutex);

	auto fixed_name = Common::FixFilenameSlash(Common::PathToGenericString(elf_name));
	for (auto* p: m_programs) {
		if (Common::EqualNoCase(Common::FixFilenameSlash(Common::PathToGenericString(p->file_name)),
		                        fixed_name)) {
			return p;
		}
	}

	return nullptr;
}

Program* RuntimeLinker::FindProgramByAddr(uint64_t vaddr) {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		const auto* ehdr = p->elf->GetEhdr();
		const auto* phdr = p->elf->GetPhdr();

		EXIT_IF(phdr == nullptr || ehdr == nullptr);

		for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
			if (phdr[i].p_memsz != 0 &&
			    (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
				uint64_t segment_addr = phdr[i].p_vaddr + p->base_vaddr;
				uint64_t segment_size = GetAlignedSize(phdr + i);

				if (vaddr >= segment_addr && vaddr < segment_addr + segment_size) {
					return p;
				}
			}
		}
	}

	return nullptr;
}

void RuntimeLinker::StackTrace(uint64_t frame_ptr, uint64_t stack_ptr) {
	void* stack[20];
	int   depth = WalkGuestStack(frame_ptr, stack_ptr, stack, static_cast<int>(std::size(stack)));

	LOGF("Stack trace [thread = %d]:\n", Common::Thread::GetThreadIdUnique());

	for (int i = 0; i < depth; i++) {
		auto  vaddr = reinterpret_cast<uint64_t>(stack[i]);
		auto* p     = FindProgramByAddr(vaddr);
		LOGF("[%d] %016" PRIx64 ", off=%016" PRIx64 ", %s\n", i, vaddr,
		     (p == nullptr ? 0 : vaddr - p->base_vaddr),
		     (p == nullptr
		          ? "???"
		          : Common::FilenameWithoutDirectory(Common::PathToGenericString(p->file_name))
		                .c_str()));
	}
}

void RuntimeLinker::StopAllModules() {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		if (p->elf->IsShared() && p->dynamic_info->fini_vaddr != 0) {
			StopModule(p, 0, nullptr, nullptr);
		}
	}
}

int RuntimeLinker::StartModule(Program* program, size_t args, const void* argp,
                               module_func_t func) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->dynamic_info == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(!program->elf->IsShared());

	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Start module: %s\n---\n",
	           Common::PathToString(program->file_name).c_str());

	return reinterpret_cast<module_ini_fini_func_t>(program->dynamic_info->init_vaddr +
	                                                program->base_vaddr)(args, argp, func);
}

int RuntimeLinker::StopModule(Program* program, size_t args, const void* argp, module_func_t func) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->dynamic_info == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(!program->elf->IsShared());

	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Stop module: %s\n---\n",
	           Common::PathToString(program->file_name).c_str());

	int result = reinterpret_cast<module_ini_fini_func_t>(program->dynamic_info->fini_vaddr +
	                                                      program->base_vaddr)(args, argp, func);

	Libs::LibKernel::PthreadDeleteStaticObjects(program);

	return result;
}

uint8_t* RuntimeLinker::TlsGetAddr(Program* program) {
	EXIT_IF(program == nullptr);

	Common::LockGuard lock(program->tls.mutex);

	auto& tls = program->tls.tlss[Common::Thread::GetThreadIdUnique()];

	if (tls.ptr == nullptr) {
		constexpr uint64_t TCB_SIZE  = 0x40;
		constexpr uint64_t TCB_ALIGN = 0x20;

		const auto tcb_offset =
		    program->tls.tcb_offset != 0 ? program->tls.tcb_offset : program->tls.image_size;
		const auto alloc_size = AlignUp(tcb_offset, TCB_ALIGN) + TCB_SIZE;
		tls.ptr        = reinterpret_cast<uint8_t*>(Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    0, alloc_size, Common::VirtualMemory::Mode::ReadWrite, "thread_local_storage"));
		tls.free_func  = nullptr;
		tls.vm_alloc   = true;
		tls.alloc_size = alloc_size;

		EXIT_IF(tls.ptr == nullptr);

		std::memset(tls.ptr, 0, alloc_size);

		if (!program->tls.init_image.empty()) {
			std::memcpy(tls.ptr, program->tls.init_image.data(), program->tls.init_image.size());
		} else {
			std::memcpy(tls.ptr, reinterpret_cast<void*>(program->tls.image_vaddr),
			            program->tls.init_size);
		}

		auto* tcb = reinterpret_cast<uint64_t*>(tls.ptr + tcb_offset);
		tcb[0]    = reinterpret_cast<uint64_t>(tcb);
	}

	return tls.ptr;
}

void RuntimeLinker::DeleteTls(Program* program, int thread_id) {
	EXIT_IF(program == nullptr);

	if (thread_id == Common::Thread::GetThreadIdUnique() && g_tls_cached_main_program == program) {
		g_tls_cached_main_program = nullptr;
		g_tls_cached_main_tcb     = nullptr;
	}

	Common::LockGuard lock(program->tls.mutex);

	if (auto it = program->tls.tlss.find(thread_id); it != program->tls.tlss.end()) {
		FreeTlsBlock(&it->second);
		program->tls.tlss.erase(it);
	}
}

constexpr uint64_t GUEST_PAGE_SIZE = 0x4000;

static uint64_t CalcBaseSize(const Elf64_Ehdr* ehdr, const Elf64_Phdr* phdr) {
	uint64_t base_size = 0;
	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t last_addr = phdr[i].p_vaddr + GetAlignedSize(phdr + i);
			if (last_addr > base_size) {
				base_size = last_addr;
			}
		}
	}
	return base_size;
}

// A module's E, R and RW LOAD segments use flexible memory in 16 KiB pages; GNU_RELRO adds none
// since it lies inside an RW LOAD.
uint64_t CalcFlexibleMemorySize(const Elf64_Phdr* phdr, Elf64_Half phnum) {
	uint64_t size = 0;
	for (Elf64_Half i = 0; i < phnum; i++) {
		const auto flags = phdr[i].p_flags;
		if (phdr[i].p_type == PT_LOAD &&
		    (flags == PF_X || flags == PF_R || flags == (PF_R | PF_W))) {
			size += AlignUp(phdr[i].p_memsz, GUEST_PAGE_SIZE);
		}
	}
	return size;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RuntimeLinker::LoadProgramToMemory(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr || program->base_vaddr != 0 || program->base_size != 0 ||
	        program->elf == nullptr);

	// static uint64_t desired_base_addr = DESIRED_BASE_ADDR;

	bool is_shared   = program->elf->IsShared();
	bool is_next_gen = program->elf->IsNextGen();

	EXIT_NOT_IMPLEMENTED(!is_shared && !is_next_gen);

	const auto* ehdr = program->elf->GetEhdr();
	const auto* phdr = program->elf->GetPhdr();

	EXIT_IF(phdr == nullptr || ehdr == nullptr);

	program->base_size = CalcBaseSize(ehdr, phdr);
	EXIT_IF(program->base_size > UINT64_MAX - (GUEST_PAGE_SIZE - 1));
	program->base_size_aligned = AlignUp(program->base_size, GUEST_PAGE_SIZE);

	uint64_t tls_handler_size = is_shared ? 0 : Jit::SafeCall::GetSize();
	EXIT_IF(tls_handler_size > UINT64_MAX - program->base_size_aligned);
	program->mapped_size = program->base_size_aligned + tls_handler_size;
	const bool emulate_amd = Config::AmdCpuEnabled();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const bool protect_memory_faults = Config::RedZoneProtectionEnabled();
#else
	const bool protect_memory_faults = false;
#endif
	const uint64_t tsc_multiplier =
	    Config::GuestTscEnabled() ? Libs::LibKernel::KernelGuestTscMultiplier() : 0;
	const bool patch_guest_instructions =
	    protect_memory_faults || emulate_amd || tsc_multiplier != 0;

	constexpr uint64_t INSTRUCTION_TRAMPOLINE_SIZE = 8u * 1024u * 1024u;
	if (patch_guest_instructions) {
		EXIT_IF(INSTRUCTION_TRAMPOLINE_SIZE > UINT64_MAX - program->mapped_size);
		program->mapped_size += INSTRUCTION_TRAMPOLINE_SIZE;
	}

	program->base_vaddr = Libs::LibKernel::Memory::AllocateProgramMemory(
	    g_desired_base_addr, program->mapped_size, Common::VirtualMemory::Mode::ExecuteReadWrite,
	    Common::PathToString(program->file_name.filename()).c_str());
	EXIT_IF(program->base_vaddr == 0);
	program->flexible_size = CalcFlexibleMemorySize(phdr, ehdr->e_phnum);
	Libs::LibKernel::Memory::ChargeFlexibleMemory(program->flexible_size);

	if (patch_guest_instructions) {
		const auto trampoline_addr           = program->base_vaddr + program->base_size_aligned;
		program->instruction_trampoline_size = INSTRUCTION_TRAMPOLINE_SIZE;
		RegisterGuestInstructionPatchModule(
		    reinterpret_cast<void*>(program->base_vaddr), program->base_size_aligned,
		    reinterpret_cast<void*>(trampoline_addr), program->instruction_trampoline_size);
	}
	if (!is_shared) {
		program->tls.handler_vaddr =
		    program->base_vaddr + program->base_size_aligned + program->instruction_trampoline_size;
	}

	g_desired_base_addr += CODE_BASE_INCR * (1 + program->mapped_size / CODE_BASE_INCR);

	EXIT_IF(program->base_size_aligned < program->base_size);
	LOGF("base_vaddr             = 0x%016" PRIx64 "\n"
	     "base_size              = 0x%016" PRIx64 "\n"
	     "base_size_aligned      = 0x%016" PRIx64 "\n"
	     "mapped_size            = 0x%016" PRIx64 "\n",
	     program->base_vaddr, program->base_size, program->base_size_aligned, program->mapped_size);
	if (!is_shared) {
		LOGF("tls_handler_size       = 0x%016" PRIx64 "\n", tls_handler_size);
	}

	if (!Common::HostException::InstallHandler(KytyExceptionHandler)) {
		EXIT("Failed to install the required vectored exception handler\n");
	}

	std::vector<std::pair<uint64_t, uint64_t>> executable_segments;
	uint64_t                                   eh_frame_header_addr = 0;
	uint64_t                                   eh_frame_header_size = 0;

	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t segment_addr        = phdr[i].p_vaddr + program->base_vaddr;
			uint64_t segment_file_size   = phdr[i].p_filesz;
			uint64_t segment_memory_size = GetAlignedSize(phdr + i);
			auto     mode                = GetMode(phdr[i].p_flags);

			LOGF("[%d] addr        = 0x%016" PRIx64 "\n"
			     "[%d] file_size   = %" PRIu64 "\n"
			     "[%d] memory_size = %" PRIu64 "\n"
			     "[%d] mode        = %s\n",
			     i, segment_addr, i, segment_file_size, i, segment_memory_size, i,
			     magic_enum::enum_name(mode));

			program->elf->LoadSegment(segment_addr, phdr[i].p_offset, segment_file_size);

			bool skip_protect = (phdr[i].p_type == PT_LOAD && is_next_gen &&
			                     mode == Common::VirtualMemory::Mode::NoAccess);

			if (Common::VirtualMemory::IsExecute(mode)) {
				PatchProgram(program, segment_addr, segment_memory_size);
				executable_segments.emplace_back(segment_addr, segment_file_size);
			}

			if (!skip_protect) {
				Libs::LibKernel::Memory::SetProgramMemoryProtection(segment_addr,
				                                                    segment_memory_size, mode);

				if (Common::VirtualMemory::IsExecute(mode)) {
					Common::VirtualMemory::FlushInstructionCache(segment_addr, segment_memory_size);
				}
			}
		}

		if (phdr[i].p_type == PT_TLS) {
			EXIT_IF(phdr[i].p_vaddr >= program->base_size);

			program->tls.image_vaddr = phdr[i].p_vaddr + program->base_vaddr;
			program->tls.init_size   = std::min(phdr[i].p_filesz, GetAlignedSize(phdr + i));
			program->tls.image_size  = GetAlignedSize(phdr + i);
			program->tls.tcb_offset  = program->tls.image_size;

			LOGF("tls addr = 0x%016" PRIx64 "\n"
			     "tls init   = %" PRIu64 "\n"
			     "tls size   = %" PRIu64 "\n"
			     "tls offset = %" PRIu64 "\n",
			     program->tls.image_vaddr, program->tls.init_size, program->tls.image_size,
			     program->tls.tcb_offset);
		}

		if (phdr[i].p_type == PT_OS_PROCPARAM) {
			EXIT_IF(program->proc_param_vaddr != 0);
			EXIT_IF(phdr[i].p_vaddr >= program->base_size);

			program->proc_param_vaddr = phdr[i].p_vaddr + program->base_vaddr;
		}

		if (patch_guest_instructions && phdr[i].p_type == PT_GNU_EH_FRAME) {
			eh_frame_header_addr = phdr[i].p_vaddr + program->base_vaddr;
			eh_frame_header_size = phdr[i].p_memsz;
		}
	}

	if (patch_guest_instructions) {
		std::vector<uintptr_t> function_starts;
		const bool             have_function_starts =
		    DecodeEhFrameFunctionStarts(eh_frame_header_addr, eh_frame_header_size,
		                                &function_starts) &&
		    !function_starts.empty();
		const auto module_name = Common::PathToString(program->file_name.filename());
		if (!have_function_starts) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Guest instruction patches: {} not applied (function boundaries unavailable)\n",
			    module_name));
		}
		GuestInstructionPatchResult totals {};
		for (const auto& [segment_addr, segment_size]: executable_segments) {
			const auto result = PatchGuestInstructions(
			    segment_addr, segment_size, function_starts, protect_memory_faults, emulate_amd,
			    GetGuestInstructionHostFeatures(), tsc_multiplier);
			totals.reciprocal_sqrt += result.reciprocal_sqrt;
			totals.extrq += result.extrq;
			totals.insertq += result.insertq;
			totals.rdpid += result.rdpid;
			totals.clwb += result.clwb;
			totals.rdtsc += result.rdtsc;
			if (protect_memory_faults) {
				LOGF("Windows guest red-zone patching: %s, functions=%" PRIu64 ", red_zone=%" PRIu64
				     ", memory=%" PRIu64 ", patched=%" PRIu64 ", short=%" PRIu64 ", stack=%" PRIu64
				     ", control=%" PRIu64 ", unrelocatable=%" PRIu64 "\n",
				     module_name.c_str(),
				     result.function_count, result.red_zone_function_count,
				     result.memory_instruction_count, result.patched_memory_instruction_count,
				     result.short_memory_instruction_count,
				     result.stack_dependent_memory_instruction_count,
				     result.control_flow_memory_instruction_count,
				     result.unrelocatable_memory_instruction_count);
			}
		}
		if (totals.rdtsc.found != 0) {
			Log::WriteToConsoleAndLog(
			    fmt::format("Guest TSC: {} (RDTSC/RDTSCP: native={}, skipped={})\n", module_name,
			                totals.rdtsc.native, totals.rdtsc.Skipped()));
		}
		if (emulate_amd && have_function_starts) {
			InstructionPatchCounts combined {};
			std::string            details;
			for (const auto& [name, counts]: {std::pair {"VRSQRTPS", totals.reciprocal_sqrt},
			                                  {"EXTRQ", totals.extrq},
			                                  {"INSERTQ", totals.insertq},
			                                  {"RDPID", totals.rdpid},
			                                  {"CLWB", totals.clwb}}) {
				combined += counts;
				if (!details.empty()) details += "; ";
				details += fmt::format("{}: native={}, trapped={}, skipped={}", name, counts.native,
				                       counts.trapped, counts.Skipped());
			}
			const auto  found   = combined.found;
			const auto  skipped = combined.Skipped();
			const char* status  = found == 0         ? "no matching instructions"
			                      : skipped == found ? "not patched"
			                      : skipped != 0     ? "partially patched"
			                                         : "patched";
			Log::WriteToConsoleAndLog(
			    fmt::format("AMD CPU compatibility: {} {} ({})\n", module_name, status, details));
		}
	}

	if (!is_shared) {
		SetupTlsHandler(program);
	}

	LOGF("entry = 0x%016" PRIx64 "\n", program->elf->GetEntry() + program->base_vaddr);
}

void RuntimeLinker::DeleteProgram(Program* p) {
	auto program = std::unique_ptr<Program>(p);
	if (g_tls_main_program == program.get()) {
		g_tls_main_program = nullptr;
	}
	if (g_tls_cached_main_program == program.get()) {
		g_tls_cached_main_program = nullptr;
		g_tls_cached_main_tcb     = nullptr;
	}
	for (auto& record: g_stubbed_imports) {
		if (record.patch_vaddr >= program->base_vaddr &&
		    record.patch_vaddr < program->base_vaddr + program->mapped_size) {
			record.patch_vaddr = 0;
		}
	}

	if (program->base_vaddr != 0 || program->mapped_size != 0) {
		EXIT_IF(program->base_vaddr == 0 || program->mapped_size == 0);
		if (program->instruction_trampoline_size != 0) {
			UnregisterGuestInstructionPatchModule(reinterpret_cast<void*>(program->base_vaddr));
		}
		EXIT_IF(
		    !Libs::LibKernel::Memory::FreeGuestMemory(program->base_vaddr, program->mapped_size));
		Libs::LibKernel::Memory::ReleaseFlexibleMemory(program->flexible_size);
	}
}

void RuntimeLinker::ParseProgramDynamicInfo(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(program->dynamic_info != nullptr);

	program->dynamic_info = std::make_unique<DynamicInfo>();

	auto* elf = program->elf.get();

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_HASH) && elf->HasDynValue(DT_HASH));
	GetDynDataOs(elf, &program->dynamic_info->hash_table, DT_OS_HASH);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->hash_table, DT_HASH);
	GetDynValue(elf, &program->dynamic_info->hash_table_size, DT_OS_HASHSZ);

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_STRTAB) && elf->HasDynValue(DT_STRTAB));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_STRSZ) && elf->HasDynValue(DT_STRSZ));
	GetDynDataOs(elf, &program->dynamic_info->str_table, DT_OS_STRTAB);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->str_table, DT_STRTAB);
	GetDynValue(elf, &program->dynamic_info->str_table_size, DT_OS_STRSZ);
	GetDynValue(elf, &program->dynamic_info->str_table_size, DT_STRSZ);

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_SYMTAB) && elf->HasDynValue(DT_SYMTAB));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_SYMENT) && elf->HasDynValue(DT_SYMENT));
	GetDynDataOs(elf, &program->dynamic_info->symbol_table, DT_OS_SYMTAB);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->symbol_table, DT_SYMTAB);
	GetDynValue(elf, &program->dynamic_info->symbol_table_total_size, DT_OS_SYMTABSZ);
	GetDynValue(elf, &program->dynamic_info->symbol_table_entry_size, DT_OS_SYMENT);
	GetDynValue(elf, &program->dynamic_info->symbol_table_entry_size, DT_SYMENT);

	GetDynPtr(elf, &program->dynamic_info->init_vaddr, DT_INIT);
	GetDynPtr(elf, &program->dynamic_info->fini_vaddr, DT_FINI);
	GetDynPtr(elf, &program->dynamic_info->init_array_vaddr, DT_INIT_ARRAY);
	GetDynPtr(elf, &program->dynamic_info->fini_array_vaddr, DT_FINI_ARRAY);
	GetDynPtr(elf, &program->dynamic_info->preinit_array_vaddr, DT_PREINIT_ARRAY);
	GetDynValue(elf, &program->dynamic_info->init_array_size, DT_INIT_ARRAYSZ);
	GetDynValue(elf, &program->dynamic_info->fini_array_size, DT_FINI_ARRAYSZ);
	GetDynValue(elf, &program->dynamic_info->preinit_array_size, DT_PREINIT_ARRAYSZ);

	Elf64_Sxword jmprel_type = 0;
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_PLTREL) && elf->HasDynValue(DT_PLTREL));
	GetDynValue(elf, &jmprel_type, DT_OS_PLTREL);
	GetDynValue(elf, &jmprel_type, DT_PLTREL);

	EXIT_NOT_IMPLEMENTED(jmprel_type != DT_RELA);
	if (jmprel_type == DT_RELA) {
		EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_JMPREL) && elf->HasDynValue(DT_JMPREL));
		EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_PLTRELSZ) && elf->HasDynValue(DT_PLTRELSZ));
		GetDynDataOs(elf, &program->dynamic_info->jmprela_table, DT_OS_JMPREL);
		GetDynData(elf, program->base_vaddr, &program->dynamic_info->jmprela_table, DT_JMPREL);
		GetDynValue(elf, &program->dynamic_info->jmprela_table_size, DT_OS_PLTRELSZ);
		GetDynValue(elf, &program->dynamic_info->jmprela_table_size, DT_PLTRELSZ);
	}

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_RELA) && elf->HasDynValue(DT_RELA));
	GetDynDataOs(elf, &program->dynamic_info->rela_table, DT_OS_RELA);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->rela_table, DT_RELA);
	GetDynValue(elf, &program->dynamic_info->rela_table_total_size, DT_OS_RELASZ);
	GetDynValue(elf, &program->dynamic_info->rela_table_total_size, DT_RELASZ);
	GetDynValue(elf, &program->dynamic_info->rela_table_entry_size, DT_OS_RELAENT);
	GetDynValue(elf, &program->dynamic_info->rela_table_entry_size, DT_RELAENT);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->rela_table_total_size % sizeof(Elf64_Rela) != 0 ||
	                     program->dynamic_info->jmprela_table_size % sizeof(Elf64_Rela) != 0);
	const auto rela_count = program->dynamic_info->rela_table_total_size / sizeof(Elf64_Rela);
	const auto jmprela_count = program->dynamic_info->jmprela_table_size / sizeof(Elf64_Rela);
	program->rela_bits.resize((rela_count + jmprela_count + 7u) / 8u);

	GetDynValue(elf, &program->dynamic_info->relative_count, DT_RELACOUNT);

	GetDynValue(elf, &program->dynamic_info->debug, DT_DEBUG);
	GetDynValue(elf, &program->dynamic_info->flags, DT_FLAGS);
	GetDynValue(elf, &program->dynamic_info->textrel, DT_TEXTREL);

	EXIT_NOT_IMPLEMENTED(program->dynamic_info->debug != 0);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->textrel != 0);

	uint64_t so_name = 0;
	GetDynValue(elf, &so_name, DT_SONAME);
	program->dynamic_info->so_name = program->dynamic_info->str_table + so_name;

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_NEEDED_MODULE) &&
	                     elf->HasDynValue(DT_OS_NEEDED_MODULE_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_MODULE_INFO) &&
	                     elf->HasDynValue(DT_OS_MODULE_INFO_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_IMPORT_LIB) &&
	                     elf->HasDynValue(DT_OS_IMPORT_LIB_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_EXPORT_LIB) &&
	                     elf->HasDynValue(DT_OS_EXPORT_LIB_1));
	GetDynModules(elf, &program->dynamic_info->import_modules, program->dynamic_info->str_table,
	              DT_OS_NEEDED_MODULE);
	GetDynModules(elf, &program->dynamic_info->import_modules, program->dynamic_info->str_table,
	              DT_OS_NEEDED_MODULE_1);
	GetDynModules(elf, &program->dynamic_info->export_modules, program->dynamic_info->str_table,
	              DT_OS_MODULE_INFO);
	GetDynModules(elf, &program->dynamic_info->export_modules, program->dynamic_info->str_table,
	              DT_OS_MODULE_INFO_1);
	GetDynLibs(elf, &program->dynamic_info->import_libs, program->dynamic_info->str_table,
	           DT_OS_IMPORT_LIB);
	GetDynLibs(elf, &program->dynamic_info->import_libs, program->dynamic_info->str_table,
	           DT_OS_IMPORT_LIB_1);
	GetDynLibs(elf, &program->dynamic_info->export_libs, program->dynamic_info->str_table,
	           DT_OS_EXPORT_LIB);
	GetDynLibs(elf, &program->dynamic_info->export_libs, program->dynamic_info->str_table,
	           DT_OS_EXPORT_LIB_1);
}

void RuntimeLinker::Relocate(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);

	if (g_invalid_memory == 0) {
		g_invalid_memory = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    INVALID_MEMORY, 4096, Common::VirtualMemory::Mode::NoAccess, "invalid_memory", true);
		EXIT_NOT_IMPLEMENTED(g_invalid_memory == 0);
	}

	LOGF_COLOR(Log::Color::White, "--- Relocate program: %s ---\n",
	           Common::PathToString(program->file_name).c_str());

	EXIT_NOT_IMPLEMENTED(program->dynamic_info->symbol_table_entry_size != sizeof(Elf64_Sym));
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->rela_table_entry_size != sizeof(Elf64_Rela));
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->jmprela_table == nullptr);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->rela_table == nullptr);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->symbol_table == nullptr);

	const auto rela_count = static_cast<uint32_t>(
	    program->dynamic_info->rela_table_total_size / sizeof(Elf64_Rela));

	ForEachRelocation(program, [&](Elf64_Rela* r, uint32_t index, bool jmprela) {
		const auto bit_index = (jmprela ? rela_count : 0) + index;
		if ((program->rela_bits[bit_index >> 3u] & (1u << (bit_index & 7u))) == 0 &&
		    RelocateRecord(index, r, program, jmprela)) {
			program->rela_bits[bit_index >> 3u] |= 1u << (bit_index & 7u);
		}
	});

	if (program->tls.image_vaddr != 0 && program->tls.init_size != 0 &&
	    program->tls.init_image.empty()) {
		const auto* src = reinterpret_cast<const uint8_t*>(program->tls.image_vaddr);
		program->tls.init_image.assign(src, src + program->tls.init_size);
	}
}

const ModuleId* RuntimeLinker::FindModule(const Program& program, const std::string& id) {
	const auto& import_modules = program.dynamic_info->import_modules;

	if (auto it = std::find_if(import_modules.begin(), import_modules.end(),
	                           [&id](const auto& module) { return module.id == id; });
	    it != import_modules.end()) {
		return &(*it);
	}

	const auto& export_modules = program.dynamic_info->export_modules;

	if (auto it = std::find_if(export_modules.begin(), export_modules.end(),
	                           [&id](const auto& module) { return module.id == id; });
	    it != export_modules.end()) {
		return &(*it);
	}

	return nullptr;
}

const LibraryId* RuntimeLinker::FindLibrary(const Program& program, const std::string& id) {
	const auto& import_libs = program.dynamic_info->import_libs;

	if (auto it = std::find_if(import_libs.begin(), import_libs.end(),
	                           [&id](const auto& lib) { return lib.id == id; });
	    it != import_libs.end()) {
		return &(*it);
	}

	const auto& export_libs = program.dynamic_info->export_libs;

	if (auto it = std::find_if(export_libs.begin(), export_libs.end(),
	                           [&id](const auto& lib) { return lib.id == id; });
	    it != export_libs.end()) {
		return &(*it);
	}

	return nullptr;
}

void RuntimeLinker::CreateSymbolDatabase(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);
	EXIT_IF(program->export_symbols != nullptr);
	EXIT_IF(program->import_symbols != nullptr);

	program->export_symbols = std::make_unique<SymbolDatabase>();
	program->import_symbols = std::make_unique<SymbolDatabase>();

	auto syms = [](Program* program, SymbolDatabase* symbols, bool is_export) {
		if (program->dynamic_info->symbol_table == nullptr ||
		    program->dynamic_info->str_table == nullptr) {
			return;
		}

		for (auto* sym = program->dynamic_info->symbol_table;
		     reinterpret_cast<uint8_t*>(sym) <
		     reinterpret_cast<uint8_t*>(program->dynamic_info->symbol_table) +
		         program->dynamic_info->symbol_table_total_size;
		     sym++) {
			std::string id   = std::string(program->dynamic_info->str_table + sym->st_name);
			auto        bind = sym->GetBind();
			auto        type = sym->GetType();
			auto        ids  = Common::Split(id, '#');

			if (ids.size() == 3) {
				const auto* l = FindLibrary(*program, ids.at(1));
				const auto* m = FindModule(*program, ids.at(2));

				if (l != nullptr && m != nullptr && (bind == STB_GLOBAL || bind == STB_WEAK) &&
				    (type == STT_FUNC || type == STT_OBJECT || type == STT_NOTYPE) &&
				    is_export == (sym->st_value != 0)) {
					SymbolResolve sr {};
					sr.name                 = ids.at(0);
					sr.library              = l->name;
					sr.library_version      = l->version;
					sr.module               = m->name;
					sr.module_version_major = m->version_major;
					sr.module_version_minor = m->version_minor;
					switch (type) {
						case STT_NOTYPE: sr.type = SymbolType::NoType; break;
						case STT_FUNC: sr.type = SymbolType::Func; break;
						case STT_OBJECT: sr.type = SymbolType::Object; break;
						default: sr.type = SymbolType::Unknown; break;
					}
					symbols->Add(sr, (is_export ? sym->st_value + program->base_vaddr : 0));
				}
			}
		}
	};

	syms(program, program->export_symbols.get(), true);
	syms(program, program->import_symbols.get(), false);
}

void RuntimeLinker::SetupTlsHandler(Program* program) {
	EXIT_IF(program == nullptr);
	EXIT_IF(g_tls_main_program != nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(program->elf->IsShared());
	EXIT_IF(program->tls.handler_vaddr == 0);

	g_tls_main_program = program;

	auto* code = new (reinterpret_cast<void*>(program->tls.handler_vaddr)) Jit::SafeCall;

	code->SetFunc(TlsMainGetAddr);

	for (uint8_t reg = 1; reg < 8; reg++) {
		if (reg == 4) {
			continue;
		}

		auto* stub = new (reinterpret_cast<void*>(program->tls.handler_vaddr +
		                                          Jit::TlsRegStub::GetOffset(reg))) Jit::TlsRegStub;
		stub->SetFunc(program->tls.handler_vaddr);
		stub->SetOutputReg(reg);
	}

	EXIT_IF(!Libs::LibKernel::Memory::ProtectGuestMemory(program->tls.handler_vaddr,
	                                                     Jit::SafeCall::GetSize(),
	                                                     Common::VirtualMemory::Mode::Execute));
	Common::VirtualMemory::FlushInstructionCache(program->tls.handler_vaddr,
	                                             Jit::SafeCall::GetSize());
}

void RuntimeLinker::DeleteTlss(int thread_id) {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		DeleteTls(p, thread_id);
	}
}

void RuntimeLinker::SetApplicationHeapApi(void* const api[10]) {
	Common::LockGuard lock(m_mutex);

	if (api == nullptr || api[0] == nullptr || api[1] == nullptr) {
		m_application_heap_malloc         = nullptr;
		m_application_heap_free           = nullptr;
		m_application_heap_posix_memalign = nullptr;
		return;
	}

	m_application_heap_malloc = reinterpret_cast<application_heap_malloc_func_t>(api[0]);
	m_application_heap_free   = reinterpret_cast<application_heap_free_func_t>(api[1]);
	m_application_heap_posix_memalign =
	    reinterpret_cast<application_heap_posix_memalign_func_t>(api[6]);
}

void* RuntimeLinker::ApplicationHeapMemalign(uint64_t alignment, uint64_t size) {
	Common::LockGuard lock(m_mutex);

	if (m_application_heap_posix_memalign != nullptr) {
		void* ptr = nullptr;
		return m_application_heap_posix_memalign(&ptr, alignment, size) == 0 ? ptr : nullptr;
	}

	return nullptr;
}

void* RuntimeLinker::ApplicationHeapMalloc(uint64_t size) {
	Common::LockGuard lock(m_mutex);

	return m_application_heap_malloc != nullptr ? m_application_heap_malloc(size) : nullptr;
}

} // namespace Loader

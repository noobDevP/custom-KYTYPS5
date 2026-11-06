#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_EMITTER_ATTRIBUTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_EMITTER_ATTRIBUTION_H_

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <map>
#include <span>
#include <spirv/unified1/spirv.hpp>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

inline const char* GetValueOpcodeName(IR::ValueOpcode op) {
	switch (op) {
#define VALUE_OPCODE(name, ...) case IR::ValueOpcode::name: return #name;
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.inc"
#undef VALUE_OPCODE
		default: return "None";
	}
}

enum class ConditionKind {
	ConstantTrue,
	ConstantFalse,
	Dynamic
};

enum class ScopeCategory : uint8_t {
	Declarations,
	HelperFunction,
	MainFunction
};

inline const char* ScopeCategoryName(ScopeCategory s) {
	switch (s) {
		case ScopeCategory::Declarations: return "Declarations/Preamble";
		case ScopeCategory::HelperFunction: return "Helper Functions";
		case ScopeCategory::MainFunction: return "Main Function";
		default: return "Unknown";
	}
}

enum class OriginCategory : uint8_t {
	NonControlFlow,
	OriginalCfg,
	EmitterControlFlow
};

inline const char* OriginCategoryName(OriginCategory o) {
	switch (o) {
		case OriginCategory::NonControlFlow: return "Non-Control-Flow";
		case OriginCategory::OriginalCfg: return "Original CFG";
		case OriginCategory::EmitterControlFlow: return "Emitter Control Flow (Guards/Helpers)";
		default: return "Unknown";
	}
}

struct InstructionRecord {
	spv::Op         opcode;
	uint32_t        word_count;
	ScopeCategory   scope;
	OriginCategory  origin;
	IR::ValueOpcode ir_opcode;
	const char*     call_site;
};

class EmitterAttributionTracker {
public:
	uint64_t shader_hash = 0;

	// Condition counts for helpers
	uint64_t if_condition_const_true    = 0;
	uint64_t if_condition_const_false   = 0;
	uint64_t if_condition_dynamic       = 0;

	uint64_t value_or_default_const_true  = 0;
	uint64_t value_or_default_const_false = 0;
	uint64_t value_or_default_dynamic     = 0;

	// Active context tags
	ScopeCategory   current_scope     = ScopeCategory::Declarations;
	OriginCategory  current_origin    = OriginCategory::NonControlFlow;
	IR::ValueOpcode current_ir_opcode = IR::ValueOpcode::Count;
	const char*     current_call_site = "Unclassified";

	// Flat list: every emitted instruction recorded exactly once
	std::vector<InstructionRecord> records;

	void RecordInstruction(spv::Op opcode, uint32_t word_count) {
		const bool is_cf_or_phi = (opcode == spv::OpLabel || opcode == spv::OpBranch ||
		                           opcode == spv::OpBranchConditional || opcode == spv::OpSelectionMerge ||
		                           opcode == spv::OpLoopMerge || opcode == spv::OpSwitch ||
		                           opcode == spv::OpKill || opcode == spv::OpReturn ||
		                           opcode == spv::OpReturnValue || opcode == spv::OpPhi);
		OriginCategory origin = OriginCategory::NonControlFlow;
		if (is_cf_or_phi) {
			if (current_origin != OriginCategory::NonControlFlow) {
				origin = current_origin;
			} else if (current_call_site && std::strncmp(current_call_site, "OriginalCFG", 11) == 0) {
				origin = OriginCategory::OriginalCfg;
			} else if (current_call_site && (std::strncmp(current_call_site, "EmitIf", 6) == 0 ||
			                                 std::strncmp(current_call_site, "EmitValue", 9) == 0 ||
			                                 std::strncmp(current_call_site, "Memory", 6) == 0 ||
			                                 std::strncmp(current_call_site, "Image", 5) == 0 ||
			                                 std::strncmp(current_call_site, "BVH", 3) == 0 ||
			                                 std::strncmp(current_call_site, "Flow", 4) == 0 ||
			                                 std::strncmp(current_call_site, "Trinity", 7) == 0 ||
			                                 std::strncmp(current_call_site, "AtomicUpdate", 12) == 0 ||
			                                 std::strncmp(current_call_site, "EmitIndexSwitch", 15) == 0)) {
				origin = OriginCategory::EmitterControlFlow;
			} else {
				origin = OriginCategory::EmitterControlFlow;
			}
		}
		records.push_back(InstructionRecord{
		    .opcode     = opcode,
		    .word_count = word_count,
		    .scope      = current_scope,
		    .origin     = origin,
		    .ir_opcode  = current_ir_opcode,
		    .call_site  = current_call_site ? current_call_site : "Unclassified"
		});
	}

	void RecordEmitIfCondition(ConditionKind kind) {
		switch (kind) {
			case ConditionKind::ConstantTrue:  if_condition_const_true++; break;
			case ConditionKind::ConstantFalse: if_condition_const_false++; break;
			case ConditionKind::Dynamic:       if_condition_dynamic++; break;
		}
	}

	void RecordEmitValueOrDefault(ConditionKind kind) {
		switch (kind) {
			case ConditionKind::ConstantTrue:  value_or_default_const_true++; break;
			case ConditionKind::ConstantFalse: value_or_default_const_false++; break;
			case ConditionKind::Dynamic:       value_or_default_dynamic++; break;
		}
	}

	void PrintSummaryTables(std::span<const uint32_t> final_spirv) const {
		Log::WriteToConsoleAndLog("\n========================================================================================\n");
		Log::WriteToConsoleAndLog(fmt::format("SPIR-V EMITTER EXPANSION ATTRIBUTION REPORT (Shader 0x{:016x})\n", shader_hash));
		Log::WriteToConsoleAndLog("========================================================================================\n");

		// 1. Independent Binary Parse of final SPIR-V words
		size_t bin_total_words = final_spirv.size();
		size_t bin_instructions = 0;
		size_t bin_labels = 0;
		size_t bin_branch_cond = 0;
		size_t bin_branch_uncond = 0;
		size_t bin_selection_merges = 0;
		size_t bin_loop_merges = 0;
		size_t bin_phis = 0;
		size_t bin_functions = 0;

		if (final_spirv.size() >= 5) {
			for (size_t offset = 5; offset < final_spirv.size();) {
				const uint32_t first = final_spirv[offset];
				const auto opcode = static_cast<spv::Op>(first & spv::OpCodeMask);
				const uint32_t word_count = first >> spv::WordCountShift;
				if (word_count == 0) break;
				bin_instructions++;
				if (opcode == spv::OpLabel) bin_labels++;
				else if (opcode == spv::OpBranchConditional) bin_branch_cond++;
				else if (opcode == spv::OpBranch) bin_branch_uncond++;
				else if (opcode == spv::OpSelectionMerge) bin_selection_merges++;
				else if (opcode == spv::OpLoopMerge) bin_loop_merges++;
				else if (opcode == spv::OpPhi) bin_phis++;
				else if (opcode == spv::OpFunction) bin_functions++;
				offset += word_count;
			}
		}

		// 2. Compute tracker totals directly from flat records (no double counting)
		size_t rec_total_words = 5; // SPIR-V header is 5 words
		size_t rec_instructions = records.size();
		size_t rec_labels = 0;
		size_t rec_branch_cond = 0;
		size_t rec_branch_uncond = 0;
		size_t rec_selection_merges = 0;
		size_t rec_loop_merges = 0;
		size_t rec_phis = 0;
		size_t rec_functions = 0;

		for (const auto& r : records) {
			rec_total_words += r.word_count;
			if (r.opcode == spv::OpLabel) rec_labels++;
			else if (r.opcode == spv::OpBranchConditional) rec_branch_cond++;
			else if (r.opcode == spv::OpBranch) rec_branch_uncond++;
			else if (r.opcode == spv::OpSelectionMerge) rec_selection_merges++;
			else if (r.opcode == spv::OpLoopMerge) rec_loop_merges++;
			else if (r.opcode == spv::OpPhi) rec_phis++;
			else if (r.opcode == spv::OpFunction) rec_functions++;
		}

		// Print Reconciliation Table
		Log::WriteToConsoleAndLog("\n--- 1. Totals Reconciliation (Tracker Records vs Independent Binary Parse) ---\n");
		Log::WriteToConsoleAndLog(fmt::format("{:<28} | {:>14} | {:>14} | {:>8} | {:<7}\n",
		                                     "Metric", "Tracker Record", "Binary Parse", "Delta", "Status"));
		Log::WriteToConsoleAndLog("-----------------------------------------------------------------------------\n");
		auto print_recon_row = [](const char* name, size_t tracker_val, size_t bin_val) {
			const int64_t delta = static_cast<int64_t>(tracker_val) - static_cast<int64_t>(bin_val);
			Log::WriteToConsoleAndLog(fmt::format("{:<28} | {:>14} | {:>14} | {:>+8} | {:<7}\n",
			                                     name, tracker_val, bin_val, delta, delta == 0 ? "EXACT" : "DIFF"));
		};
		print_recon_row("Total Instructions", rec_instructions, bin_instructions);
		print_recon_row("Total Words", rec_total_words, bin_total_words);
		print_recon_row("Basic Blocks (OpLabel)", rec_labels, bin_labels);
		print_recon_row("Phi Nodes (OpPhi)", rec_phis, bin_phis);
		print_recon_row("Cond Branches (BranchCond)", rec_branch_cond, bin_branch_cond);
		print_recon_row("Uncond Branches (OpBranch)", rec_branch_uncond, bin_branch_uncond);
		print_recon_row("Selection Merges (Merge)", rec_selection_merges, bin_selection_merges);
		print_recon_row("Loop Merges (LoopMerge)", rec_loop_merges, bin_loop_merges);
		print_recon_row("Functions (OpFunction)", rec_functions, bin_functions);

		// 3. Main Function vs Generated Helper Functions vs Declarations
		Log::WriteToConsoleAndLog("\n--- 2. Breakdown by Function Scope ---\n");
		Log::WriteToConsoleAndLog(fmt::format("{:<24} | {:>12} | {:>12} | {:>8} | {:>8} | {:>8}\n",
		                                     "Scope", "Instructions", "Words", "Labels", "Phis", "SelMerges"));
		Log::WriteToConsoleAndLog("----------------------------------------------------------------------------\n");
		for (auto scope : {ScopeCategory::MainFunction, ScopeCategory::HelperFunction, ScopeCategory::Declarations}) {
			size_t sc_inst = 0, sc_words = 0, sc_labels = 0, sc_phis = 0, sc_merges = 0;
			for (const auto& r : records) {
				if (r.scope == scope) {
					sc_inst++;
					sc_words += r.word_count;
					if (r.opcode == spv::OpLabel) sc_labels++;
					else if (r.opcode == spv::OpPhi) sc_phis++;
					else if (r.opcode == spv::OpSelectionMerge) sc_merges++;
				}
			}
			Log::WriteToConsoleAndLog(fmt::format("{:<24} | {:>12} | {:>12} | {:>8} | {:>8} | {:>8}\n",
			                                     ScopeCategoryName(scope), sc_inst, sc_words, sc_labels, sc_phis, sc_merges));
		}

		// 4. Original CFG vs Additional Emitter Control Flow
		Log::WriteToConsoleAndLog("\n--- 3. Breakdown by Control Flow Origin ---\n");
		Log::WriteToConsoleAndLog(fmt::format("{:<32} | {:>8} | {:>8} | {:>10} | {:>8} | {:>10}\n",
		                                     "Origin", "Labels", "Phis", "CondBranch", "SelMerge", "Words"));
		Log::WriteToConsoleAndLog("--------------------------------------------------------------------------------\n");
		for (auto origin : {OriginCategory::OriginalCfg, OriginCategory::EmitterControlFlow, OriginCategory::NonControlFlow}) {
			size_t o_words = 0, o_labels = 0, o_phis = 0, o_branches = 0, o_merges = 0;
			for (const auto& r : records) {
				if (r.origin == origin) {
					o_words += r.word_count;
					if (r.opcode == spv::OpLabel) o_labels++;
					else if (r.opcode == spv::OpPhi) o_phis++;
					else if (r.opcode == spv::OpBranchConditional) o_branches++;
					else if (r.opcode == spv::OpSelectionMerge) o_merges++;
				}
			}
			Log::WriteToConsoleAndLog(fmt::format("{:<32} | {:>8} | {:>8} | {:>10} | {:>8} | {:>10}\n",
			                                     OriginCategoryName(origin), o_labels, o_phis, o_branches, o_merges, o_words));
		}

		// 5. Conditional Helpers Evaluation
		Log::WriteToConsoleAndLog("\n--- 4. Conditional Helper Invocations Breakdown ---\n");
		Log::WriteToConsoleAndLog(fmt::format("{:<32} | {:>14} | {:>15} | {:>12} | {:>12}\n",
		                                     "Helper", "Constant-True", "Constant-False", "Dynamic", "Total Calls"));
		Log::WriteToConsoleAndLog("--------------------------------------------------------------------------------------------\n");
		const uint64_t if_total = if_condition_const_true + if_condition_const_false + if_condition_dynamic;
		Log::WriteToConsoleAndLog(fmt::format("{:<32} | {:>14} | {:>15} | {:>12} | {:>12}\n",
		                                     "EmitIfCondition", if_condition_const_true, if_condition_const_false, if_condition_dynamic, if_total));
		const uint64_t val_total = value_or_default_const_true + value_or_default_const_false + value_or_default_dynamic;
		Log::WriteToConsoleAndLog(fmt::format("{:<32} | {:>14} | {:>15} | {:>12} | {:>12}\n",
		                                     "EmitValueOrDefaultIfCondition", value_or_default_const_true, value_or_default_const_false, value_or_default_dynamic, val_total));

		// 6. Top 15 Contributors by Added Labels
		struct ContributorKey {
			std::string call_site;
			IR::ValueOpcode ir_opcode;
			bool operator<(const ContributorKey& o) const {
				if (call_site != o.call_site) return call_site < o.call_site;
				return ir_opcode < o.ir_opcode;
			}
		};
		std::map<ContributorKey, size_t> label_contributors;
		for (const auto& r : records) {
			if (r.opcode == spv::OpLabel) {
				label_contributors[{r.call_site, r.ir_opcode}]++;
			}
		}
		std::vector<std::pair<ContributorKey, size_t>> sorted_labels(label_contributors.begin(), label_contributors.end());
		std::sort(sorted_labels.begin(), sorted_labels.end(), [](const auto& a, const auto& b) {
			return a.second > b.second;
		});

		Log::WriteToConsoleAndLog("\n--- 5. Top 15 Contributors by Added Labels (OpLabel) ---\n");
		Log::WriteToConsoleAndLog(fmt::format("{:>4} | {:<34} | {:<24} | {:>8} | {:>7}\n",
		                                     "Rank", "Guard / Call Site", "IR Opcode", "Labels", "% Total"));
		Log::WriteToConsoleAndLog("--------------------------------------------------------------------------------\n");
		for (size_t i = 0; i < std::min<size_t>(15, sorted_labels.size()); ++i) {
			const auto& [key, count] = sorted_labels[i];
			const double pct = rec_labels > 0 ? (100.0 * count / static_cast<double>(rec_labels)) : 0.0;
			Log::WriteToConsoleAndLog(fmt::format("{:>4} | {:<34} | {:<24} | {:>8} | {:>6.2f}%\n",
			                                     i + 1, key.call_site, GetValueOpcodeName(key.ir_opcode), count, pct));
		}
		if (sorted_labels.size() > 15) {
			size_t rem = 0;
			for (size_t i = 15; i < sorted_labels.size(); ++i) rem += sorted_labels[i].second;
			const double pct = rec_labels > 0 ? (100.0 * rem / static_cast<double>(rec_labels)) : 0.0;
			Log::WriteToConsoleAndLog(fmt::format("{:>4} | {:<34} | {:<24} | {:>8} | {:>6.2f}%\n",
			                                     "-", "Remaining Contributors", "-", rem, pct));
		}
		Log::WriteToConsoleAndLog("--------------------------------------------------------------------------------\n");
		Log::WriteToConsoleAndLog(fmt::format("{:<63} | {:>8} | {:>7}\n", "Total Labels", rec_labels, "100.00%"));

		// 7. Top 15 Contributors by Added Phi Nodes
		std::map<ContributorKey, size_t> phi_contributors;
		for (const auto& r : records) {
			if (r.opcode == spv::OpPhi) {
				phi_contributors[{r.call_site, r.ir_opcode}]++;
			}
		}
		std::vector<std::pair<ContributorKey, size_t>> sorted_phis(phi_contributors.begin(), phi_contributors.end());
		std::sort(sorted_phis.begin(), sorted_phis.end(), [](const auto& a, const auto& b) {
			return a.second > b.second;
		});

		Log::WriteToConsoleAndLog("\n--- 6. Top 15 Contributors by Added Phi Nodes (OpPhi) ---\n");
		Log::WriteToConsoleAndLog(fmt::format("{:>4} | {:<34} | {:<24} | {:>8} | {:>7}\n",
		                                     "Rank", "Guard / Call Site", "IR Opcode", "Phis", "% Total"));
		Log::WriteToConsoleAndLog("--------------------------------------------------------------------------------\n");
		for (size_t i = 0; i < std::min<size_t>(15, sorted_phis.size()); ++i) {
			const auto& [key, count] = sorted_phis[i];
			const double pct = rec_phis > 0 ? (100.0 * count / static_cast<double>(rec_phis)) : 0.0;
			Log::WriteToConsoleAndLog(fmt::format("{:>4} | {:<34} | {:<24} | {:>8} | {:>6.2f}%\n",
			                                     i + 1, key.call_site, GetValueOpcodeName(key.ir_opcode), count, pct));
		}
		if (sorted_phis.size() > 15) {
			size_t rem = 0;
			for (size_t i = 15; i < sorted_phis.size(); ++i) rem += sorted_phis[i].second;
			const double pct = rec_phis > 0 ? (100.0 * rem / static_cast<double>(rec_phis)) : 0.0;
			Log::WriteToConsoleAndLog(fmt::format("{:>4} | {:<34} | {:<24} | {:>8} | {:>6.2f}%\n",
			                                     "-", "Remaining Contributors", "-", rem, pct));
		}
		Log::WriteToConsoleAndLog("--------------------------------------------------------------------------------\n");
		Log::WriteToConsoleAndLog(fmt::format("{:<63} | {:>8} | {:>7}\n", "Total Phi Nodes", rec_phis, "100.00%"));
		Log::WriteToConsoleAndLog("========================================================================================\n\n");
	}
};

struct ScopedAttributionSite {
	EmitterAttributionTracker* tracker;
	const char*                prev_site   = nullptr;
	OriginCategory             prev_origin = OriginCategory::NonControlFlow;

	ScopedAttributionSite(EmitterAttributionTracker* t, const char* site,
	                      OriginCategory origin = OriginCategory::NonControlFlow)
	    : tracker(t) {
		if (tracker) {
			prev_site = tracker->current_call_site;
			prev_origin = tracker->current_origin;
			tracker->current_call_site = site;
			if (origin != OriginCategory::NonControlFlow) {
				tracker->current_origin = origin;
			}
		}
	}
	~ScopedAttributionSite() {
		if (tracker) {
			tracker->current_call_site = prev_site;
			tracker->current_origin = prev_origin;
		}
	}
};

struct ScopedFunctionScope {
	EmitterAttributionTracker* tracker;
	ScopeCategory              prev_scope = ScopeCategory::Declarations;
	const char*                prev_site  = nullptr;

	ScopedFunctionScope(EmitterAttributionTracker* t, ScopeCategory scope, const char* site = nullptr)
	    : tracker(t) {
		if (tracker) {
			prev_scope = tracker->current_scope;
			prev_site = tracker->current_call_site;
			tracker->current_scope = scope;
			if (site) tracker->current_call_site = site;
		}
	}
	~ScopedFunctionScope() {
		if (tracker) {
			tracker->current_scope = prev_scope;
			tracker->current_call_site = prev_site;
		}
	}
};

struct ScopedIrOpcode {
	EmitterAttributionTracker* tracker;
	IR::ValueOpcode            prev_opcode = IR::ValueOpcode::Count;

	ScopedIrOpcode(EmitterAttributionTracker* t, IR::ValueOpcode op) : tracker(t) {
		if (tracker) {
			prev_opcode = tracker->current_ir_opcode;
			tracker->current_ir_opcode = op;
		}
	}
	~ScopedIrOpcode() {
		if (tracker) {
			tracker->current_ir_opcode = prev_opcode;
		}
	}
};

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_EMITTER_ATTRIBUTION_H_ */

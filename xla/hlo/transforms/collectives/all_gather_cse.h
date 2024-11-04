/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ALL_GATHER_CSE_H
#define ALL_GATHER_CSE_H

#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
/*
    This pass attempts to perform common subexpression elimination (CSE) on
    all-gathers of parameters. This is especially effective if you do not want
   to run all-gathers in the backward pass. Example:

    Before the pass:
    while_loop {
        all-gather.1 = all-gather(param_0)
        some_computation.1 = compute(all-gather.1)
        all-gather.2 = all-gather(param_0)
        some_computation.2 = compute(all-gather.2)
    }

    After the pass:
    while_loop {
        all-gather.0 = all-gather(param_0)
        some_computation.1 = compute(all-gather.0)
        some_computation.2 = compute(all-gather.0)
    }
*/
class AllGatherCSE : public HloModulePass {
 public:
  absl::string_view name() const override { return "all-gather-cse"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::tuple<HloInstruction*, int64_t, PrimitiveType> FindRawParameter(
      HloInstruction* instruction);
};

}  // namespace xla

#endif  // ALL_GATHER_CSE_H
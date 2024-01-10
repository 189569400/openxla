/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
#define XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_

#include "xla/service/buffer_assignment.h"

namespace xla {
namespace gpu {

inline constexpr int64_t kCollectiveMemorySpaceColor = 1;

// Set memory space to kCollectiveMemorySpaceColor for all allocations used by
// all-reduce, all-gather, and reduce-scatter. This memory space maps to
// collective memory using ncclMemAlloc in the runtime.
BufferAssigner::Colorer CollectiveColorer() {
  return [](HloAliasAnalysis* alias_analysis, const HloOrdering&) {
    for (HloValue* value : alias_analysis->dataflow_analysis().values()) {
      auto& buffer = alias_analysis->GetBufferContainingValue(*value);
      for (const auto& alias : buffer.values()) {
        if ((alias->instruction()->opcode() == HloOpcode::kAllReduce ||
             alias->instruction()->opcode() == HloOpcode::kAllReduceStart ||
             alias->instruction()->opcode() == HloOpcode::kAllReduceDone ||
             alias->instruction()->opcode() == HloOpcode::kAllGather ||
             alias->instruction()->opcode() == HloOpcode::kAllGatherStart ||
             alias->instruction()->opcode() == HloOpcode::kAllGatherDone ||
             alias->instruction()->opcode() == HloOpcode::kReduceScatter) ||
            ((alias->instruction()->opcode() == HloOpcode::kAsyncStart ||
              alias->instruction()->opcode() == HloOpcode::kAsyncDone) &&
             alias->instruction()->async_wrapped_opcode() ==
                 HloOpcode::kReduceScatter)) {
          value->set_color(kCollectiveMemorySpaceColor);
        }
      }
      if (!value->has_color()) {
        value->set_color(0);
      }
    }
    return OkStatus();
  };
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_

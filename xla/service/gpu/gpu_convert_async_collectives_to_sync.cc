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

#include "xla/service/gpu/gpu_convert_async_collectives_to_sync.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla {
namespace gpu {

Status GpuConvertAsyncCollectivesToSync::ConvertAsyncInstructionsToSync(
    HloComputation* computation,
    absl::Span<const std::pair<HloInstruction*, HloInstruction*>> async_pairs)
    const {
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replaced_ops;
  for (auto& [async_start, async_done] : async_pairs) {
    TF_RETURN_IF_ERROR(
        async_start->update_backend_config<CollectiveBackendConfig>(
            [](auto& config) { config.set_is_sync(true); }));
    replaced_ops[async_start] = nullptr;
    replaced_ops[async_done] = async_start;
  }

  // Update schedule.
  HloModule* module = computation->parent();
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  std::vector<HloInstruction*> new_sequence;
  new_sequence.reserve(sequence.size());
  for (HloInstruction* instr : sequence.instructions()) {
    auto it = replaced_ops.find(instr);
    // If its not a start or done, add it to new schedule.
    if (it == replaced_ops.end()) {
      new_sequence.push_back(instr);
      continue;
    }

    // If its a start op, do not add it to the schedule yet.
    if (it->second == nullptr) {
      continue;
    }

    // Its a done op. First add the start and then the done.
    new_sequence.push_back(it->second);
    new_sequence.push_back(instr);
  }
  module->schedule().set_sequence(computation, new_sequence);
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla

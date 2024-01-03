/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_collective_broadcast_thunk.h"
#include <stdexcept>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/xla_data.pb.h"
#include "xla/service/gpu/nccl_api.h"

namespace xla::gpu {

using mlir::lmhlo_gpu::CollectiveBroadcastStartOp;

NcclCollectiveBroadcastStartThunk::NcclCollectiveBroadcastStartThunk(
    ThunkInfo thunk_info, NcclApi* nccl_api,
    const HloCollectiveBroadcastInstruction* instr, std::vector<Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclCollectiveBroadcastStart, thunk_info,
                          nccl_api, IsSyncCollective(instr)),
      config_(GetNcclCollectiveConfig(instr, std::nullopt)),
      buffers_(std::move(buffers)) {}

/*static*/ Status NcclCollectiveBroadcastStartThunk::CheckImplementable(
    const HloInstruction* instr, int64_t replica_count, int64_t partition_count) {
  return OkStatus();
}

/*static*/ bool NcclCollectiveBroadcastStartThunk::IsDegenerate(
    CollectiveBroadcastStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return false;
}

/*static*/ CollectiveOpGroupMode
NcclCollectiveBroadcastStartThunk::GetGroupMode(
    const HloCollectiveBroadcastInstruction* inst) {
  return GetNcclCollectiveConfig(inst, std::nullopt).group_mode;
}

Status NcclCollectiveBroadcastStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    NcclApi::NcclCommHandle comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_, config_.operand_element_type));
  return ::xla::gpu::RunCollectiveBroadcast(device_buffers, stream, comm,
                                            nccl_api());
}

Status RunCollectiveBroadcast(std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, NcclApi::NcclCommHandle comm,
                              NcclApi* nccl_api) {
  TF_RETURN_IF_ERROR(nccl_api->GroupStart());
  for (auto buffer : buffers) {
    se::DeviceMemoryBase src_addr = buffer.source_buffer;
    se::DeviceMemoryBase dest_addr = buffer.destination_buffer;
    TF_RETURN_IF_ERROR(nccl_api->Broadcast(
        // Always use rank 0 since we always broadcast from the first id in
        // replica_groups
        src_addr, dest_addr, buffer.element_type, buffer.element_count, 0, comm,
        &stream));
  }
  return nccl_api->GroupEnd();
}

}  // namespace xla::gpu

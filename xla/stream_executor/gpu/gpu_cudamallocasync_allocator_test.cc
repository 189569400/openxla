/* Copyright 2019 The OpenXLA Authors.

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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/framework/device_id.h"

#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

namespace se = stream_executor;

namespace {
static se::StreamExecutor* GpuExecutor() {
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}
}  // namespace

namespace stream_executor {

TEST(GpuCudaMallocAsyncAllocator, AddressAlignedDefaultPool) {
#if CUDA_VERSION < 11030
  GTEST_SKIP() << "Cuda async memory allocator is not supported for CUDA "
                  "version less than 11030";
#endif

  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  auto allocator = GpuCudaMallocAsyncAllocator(
      /*platform_device_id*/ tsl::PlatformDeviceId(executor->device_ordinal()),
      /*pool_size*/ 2048,
      /*new_pool_size*/ true,
      /*release_threshold*/ true);
  allocator.SetStreamAndPreallocateMemory(
      se::gpu::AsGpuStreamValue(stream.get()));
  void* addr1 = allocator.AllocateRaw(128, 127);
  void* addr2 = allocator.AllocateRaw(128, 129);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr1) & 127), 0);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr2) & 127), 0);
}

TEST(GpuCudaMallocAsyncAllocator, AddressAlignedNewPool) {
#if CUDA_VERSION < 11030
  GTEST_SKIP() << "Cuda async memory allocator is not supported for CUDA "
                  "version less than 11030";
#endif
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  auto allocator = GpuCudaMallocAsyncAllocator(
      /*platform_device_id*/ tsl::PlatformDeviceId(executor->device_ordinal()),
      /*create_new_pool*/ true,
      /*new_pool_size*/ 2048,
      /*release_threshold*/ 0,
      /*reserve_memory*/ true,
      /*compute_stats*/ true);
  allocator.SetStreamAndPreallocateMemory(
      se::gpu::AsGpuStreamValue(stream.get()));

  void* addr1 = allocator.AllocateRaw(128, 127);
  void* addr2 = allocator.AllocateRaw(128, 129);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr1) & 127), 0);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr2) & 127), 0);
}

}  // namespace stream_executor

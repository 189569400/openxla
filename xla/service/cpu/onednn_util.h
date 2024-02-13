/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_ONEDNN_UTIL_H_
#define XLA_SERVICE_CPU_ONEDNN_UTIL_H_
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "tsl/platform/cpu_info.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

inline bool IsSupportedType(xla::PrimitiveType dtype) {
  using tsl::port::CPUFeature;
  // TODO(intel-tf): Enable more types.
  switch (dtype) {
    case F32:
      return true;
    case BF16:
      return TestCPUFeature(CPUFeature::AVX512F) ||
             TestCPUFeature(CPUFeature::AVX_NE_CONVERT) ||
             TestCPUFeature(CPUFeature::AMX_BF16);
    case F16:
      return TestCPUFeature(CPUFeature::AVX512BW) &&
             (TestCPUFeature(CPUFeature::AVX512_FP16) ||
              TestCPUFeature(CPUFeature::AMX_FP16) ||
              TestCPUFeature(CPUFeature::AVX_NE_CONVERT));
    default:
      return false;
  }
  return false;
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
#endif  // XLA_SERVICE_CPU_ONEDNN_UTIL_H_

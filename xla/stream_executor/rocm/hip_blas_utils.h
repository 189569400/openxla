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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_UTILS_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_UTILS_H_

#include <string>
#include "xla/stream_executor/rocm/hipblaslt_wrapper.h"
#include "xla/stream_executor/blas.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"


namespace stream_executor {
namespace rocm {

#define SE_HIPBLAS_RETURN_IF_ERROR(expr) \
  TF_RETURN_IF_ERROR(::stream_executor::rocm::ToStatus(expr, #expr))



xla::Status ToStatus(hipblasStatus_t status, const char* prefix);
hipblasDatatype_t AsHipblasDataType(blas::DataType type);
hipblasLtComputeType_t AsHipblasComputeType(blas::ComputationType type);
hipblasOperation_t AsHipblasOperation(blas::Transpose trans);

}  // namespace rocm
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_UTILS_H_

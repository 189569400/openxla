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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_

#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host_or_device_scalar.h"
#include "tsl/platform/status.h"

#if TF_HIPBLASLT

#include "xla/status.h"

#include "rocm/rocm_config.h"
#include "xla/stream_executor/rocm/hip_blas_utils.h"
#include "xla/stream_executor/rocm/hipblaslt_wrapper.h"

namespace stream_executor {

namespace gpu {
class GpuExecutor;
}  // namespace gpu

namespace rocm {

class BlasLt {
  template <typename T>
  using Owned =
      std::unique_ptr<std::remove_pointer_t<T>, hipblasStatus_t (*)(T)>;

 public:
  class MatrixLayout {
   public:
    enum class Order { kRowMajor, kColumnMajor };

    // If `leading_dim_stride` is not specified, it defaults to:
    //  - `num_cols` if `order == kRowMajor`,
    //  - `num_rows` if `order == kColumnMajor`.
    // If `batch_stride` is not specified, it defaults to `num_rows * num_cols`
    // if `batch_size > 1`, otherwise `0`.
    static tsl::StatusOr<MatrixLayout> Create(
        blas::DataType type, size_t num_rows, size_t num_cols, Order order,
        size_t batch_size = 1,
        std::optional<int64_t> leading_dim_stride = std::nullopt,
        std::optional<int64_t> batch_stride = std::nullopt);

    hipblasDatatype_t type() const;

    hipblasLtMatrixLayout_t get() const { return handle_.get(); }

   private:
    explicit MatrixLayout(hipblasLtMatrixLayout_t handle)
        : handle_(handle, wrap::hipblasLtMatrixLayoutDestroy) {}

    Owned<hipblasLtMatrixLayout_t> handle_;
  };

  enum class Epilogue {
    kDefault = 1,                   // No special postprocessing
    kReLU = 2,                      // Apply point-wise ReLU function
    kBias = 4,                      // Add broadcasted bias vector
    kBiasThenReLU = kBias | kReLU,  // Apply bias and then ReLU transform
    kGELU = 32,                // Apply GELU point-wise transform to the results
    kGELUWithAux = 32 | 1024,  // Apply GELU with auxiliary output.
    kBiasThenGELU = kBias | kGELU,  // Apply bias and then approximate GELU.
    kBiasThenGELUWithAux = kBiasThenGELU | 1024,
  };

  // Describes the location of pointers for the scaling factors alpha and beta.
  enum class PointerMode {
    kHost,
    kDevice,
  };

  class MatmulDesc {
   public:
    static tsl::StatusOr<MatmulDesc> Create(
        blas::ComputationType compute_type, blas::DataType scale_type,
        blas::Transpose trans_a = blas::Transpose::kNoTranspose,
        blas::Transpose trans_b = blas::Transpose::kNoTranspose,
        Epilogue epilogue = Epilogue::kDefault,
        PointerMode pointer_mode = PointerMode::kHost);

    hipblasLtComputeType_t compute_type() const;
    hipblasDatatype_t scale_type() const;
    hipblasPointerMode_t pointer_mode() const;

    hipblasLtMatmulDesc_t get() const { return handle_.get(); }

   private:
    explicit MatmulDesc(hipblasLtMatmulDesc_t handle)
        : handle_(handle, wrap::hipblasLtMatmulDescDestroy) {}

    Owned<hipblasLtMatmulDesc_t> handle_;
  };

  // TODO(cjfj): Add consistency checks for types, shapes, etc.?
  struct MatmulPlan {
    MatmulDesc op_desc;
    MatrixLayout a_desc;
    MatrixLayout b_desc;
    MatrixLayout c_desc;
    MatrixLayout d_desc;
  };

  class MatmulPreference {
   public:
    static tsl::StatusOr<MatmulPreference> Create(size_t max_workspace_size);

    hipblasLtMatmulPreference_t get() const { return handle_.get(); }

   private:
    explicit MatmulPreference(hipblasLtMatmulPreference_t handle)
        : handle_(handle, wrap::hipblasLtMatmulPreferenceDestroy) {}

    Owned<hipblasLtMatmulPreference_t> handle_;
  };

  struct MatmulAlgorithm {
    hipblasLtMatmulAlgo_t algo;
    size_t workspace_size;
  };

  explicit BlasLt(gpu::GpuExecutor* parent)
      : parent_(parent), blas_lt_(nullptr, wrap::hipblasLtDestroy) {}

  tsl::Status Init();

  // Returns a list of supported algorithms for DoMatmul. The algorithms are
  // returned in the order of increasing estimated compute time according to an
  // internal heuristic.
  tsl::StatusOr<std::vector<MatmulAlgorithm>> GetMatmulAlgorithms(
      const MatmulPlan& plan, const MatmulPreference& preference,
      size_t max_algorithm_count = 128);

  template <typename A, typename B, typename C, typename D, typename Scale>
  tsl::Status DoMatmul(Stream* stream, const MatmulPlan& plan,
                       const HostOrDeviceScalar<Scale>& alpha,
                       const DeviceMemory<A>& a, const DeviceMemory<B>& b,
                       const HostOrDeviceScalar<Scale>& beta,
                       const DeviceMemory<C>& c, DeviceMemory<D>& d,
                       const MatmulAlgorithm& algorithm,
                       ScratchAllocator& scratch_allocator,
                       const DeviceMemory<C>& bias = {},
                       const DeviceMemoryBase& aux = DeviceMemory<uint8_t>{},
                       const DeviceMemory<Scale>& a_scale = {},
                       const DeviceMemory<Scale>& b_scale = {},
                       const DeviceMemory<Scale>& c_scale = {},
                       const DeviceMemory<Scale>& d_scale = {},
                       const DeviceMemory<Scale>& d_amax = {},
                       blas::ProfileResult* profile_result = nullptr) {
    if (AsHipblasDataType(blas::ToDataType<Scale>::value) !=
        plan.op_desc.scale_type()) {
      return tsl::errors::InvalidArgument("mismatched scale types");
    }

    bool expect_scale_factor_on_device =
        (plan.op_desc.pointer_mode() == HIPBLAS_POINTER_MODE_DEVICE);

    if (alpha.on_device() != expect_scale_factor_on_device) {
      return tsl::errors::InvalidArgument("wrong location for alpha");
    }

    if (beta.on_device() != expect_scale_factor_on_device) {
      return tsl::errors::InvalidArgument("wrong location for beta");
    }

    if (AsHipblasDataType(blas::ToDataType<A>::value) != plan.a_desc.type()) {
      return tsl::errors::InvalidArgument("mismatched A matrix types");
    }

    if (AsHipblasDataType(blas::ToDataType<B>::value) != plan.b_desc.type()) {
      return tsl::errors::InvalidArgument("mismatched B matrix types");
    }

    if (AsHipblasDataType(blas::ToDataType<C>::value) != plan.c_desc.type()) {
      return tsl::errors::InvalidArgument("mismatched C matrix types");
    }

    if (AsHipblasDataType(blas::ToDataType<D>::value) != plan.d_desc.type()) {
      return tsl::errors::InvalidArgument("mismatched D matrix types");
    }

    return DoMatmul(stream, plan, alpha.opaque(), a, b, beta.opaque(), c, d,
                    algorithm, scratch_allocator, bias, aux, a_scale, b_scale,
                    c_scale, d_scale, d_amax, profile_result);
  }

  template <typename A, typename B, typename C, typename D, typename Scale>
  tsl::Status DoMatmul(Stream* stream, const MatmulPlan& plan,
                       const HostOrDeviceScalar<Scale>& alpha,
                       const DeviceMemory<A>& a, const DeviceMemory<B>& b,
                       const HostOrDeviceScalar<Scale>& beta,
                       const DeviceMemory<C>& c, DeviceMemory<D>& d,
                       const MatmulAlgorithm& algorithm,
                       ScratchAllocator& scratch_allocator,
                       const DeviceMemory<C>& bias = {},
                       const DeviceMemoryBase& aux = DeviceMemory<uint8_t>{},
                       blas::ProfileResult* profile_result = nullptr) {
    return DoMatmul(stream, plan, alpha, a, b, beta, c, d, algorithm,
                    scratch_allocator, bias, aux, {}, {}, {}, {}, {},
                    profile_result);
  }

 private:
  tsl::Status DoMatmul(Stream* stream, const MatmulPlan& plan,
                       const void* alpha, DeviceMemoryBase a,
                       DeviceMemoryBase b, const void* beta, DeviceMemoryBase c,
                       DeviceMemoryBase d, const MatmulAlgorithm& algorithm,
                       ScratchAllocator& scratch_allocator,
                       DeviceMemoryBase bias, DeviceMemoryBase aux,
                       DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
                       DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
                       DeviceMemoryBase d_amax,
                       blas::ProfileResult* profile_result);

  gpu::GpuExecutor* parent_;

  absl::Mutex mu_;
  Owned<hipblasLtHandle_t> blas_lt_ ABSL_GUARDED_BY(mu_);
};

// Returns `BlasLt` implementation for a stream if available, or `nullptr`.
BlasLt* GetBlasLt(Stream* stream);

}   // namespace rocm

namespace gpu{
  using BlasLt = ::stream_executor::rocm::BlasLt;
  inline BlasLt* GetBlasLt(Stream* stream) { return rocm::GetBlasLt(stream); }
} // namespace gpu

}  // namespace stream_executor

#endif

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_

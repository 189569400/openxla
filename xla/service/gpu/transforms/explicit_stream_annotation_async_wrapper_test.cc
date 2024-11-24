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

#include "xla/service/gpu/transforms/explicit_stream_annotation_async_wrapper.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/test.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ExplicitStreamAnnotationAsyncWrapperTest = HloTestBase;

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest, AnnotatedOpIsWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %sub (lhs: f32[]) -> f32[] {
    %lhs = f32[] parameter(0)
    %rhs = f32[] constant(1)
    ROOT %sub = f32[] subtract(f32[] %lhs, f32[] %rhs)
  }

  ENTRY %main () -> f32[] {
    %lhs = f32[] constant(42)
    %call1 = f32[] call(f32[] %lhs), to_apply=%sub, frontend_attributes={_xla_stream_annotation="1"}
  })";

  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %lhs.1 = f32[] constant(42)
  // CHECK: %call-start = ((f32[]), f32[]) call-start(f32[] %lhs.1), async_execution_thread="explicit", to_apply=%sub, frontend_attributes={_xla_stream_annotation="1"}
  // CHECK: ROOT %call-done = f32[] call-done(((f32[]), f32[]) %call-start), frontend_attributes={_xla_stream_annotation="1"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false}
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest, OverlappingGemms) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %stream1.3 (Arg_0.4: f32[2048,2048], Arg_1.5: f32[2048,2048]) -> f32[2048,2048] {
    %Arg_1.5 = f32[2048,2048]{1,0} parameter(1)
    %Arg_0.4 = f32[2048,2048]{1,0} parameter(0)
    %custom-call.1 = (f32[2048,2048]{1,0}, s8[33554432]{0}) custom-call(f32[2048,2048]{1,0} %Arg_0.4, f32[2048,2048]{1,0} %Arg_1.5), custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"lhs_stride":"4194304","rhs_stride":"4194304","grad_x":false,"grad_y":false},"force_earliest_schedule":false}
    ROOT %get-tuple-element = f32[2048,2048]{1,0} get-tuple-element((f32[2048,2048]{1,0}, s8[33554432]{0}) %custom-call.1), index=0, metadata={op_name="jit(test)/jit(main)/jit(stream1)/dot_general" source_file="/mounted/hacking/stream_setting_demo.py" source_line=7 scheduling_name="get-tuple-element"}
  }
  %stream1.4 (Arg_0.4: f32[2048,2048], Arg_1.5: f32[2048,2048]) -> f32[2048,2048] {
    %Arg_1.6 = f32[2048,2048]{1,0} parameter(1)
    %Arg_0.7 = f32[2048,2048]{1,0} parameter(0)
    %custom-call.2 = (f32[2048,2048]{1,0}, s8[33554432]{0}) custom-call(f32[2048,2048]{1,0} %Arg_0.7, f32[2048,2048]{1,0} %Arg_1.6), custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"lhs_stride":"4194304","rhs_stride":"4194304","grad_x":false,"grad_y":false},"force_earliest_schedule":false}
    ROOT %get-tuple-element = f32[2048,2048]{1,0} get-tuple-element((f32[2048,2048]{1,0}, s8[33554432]{0}) %custom-call.2), index=0, metadata={op_name="jit(test)/jit(main)/jit(stream1)/dot_general" source_file="/mounted/hacking/stream_setting_demo.py" source_line=7 scheduling_name="get-tuple-element"}
  }

  ENTRY %main () -> f32[2048,2048]{1,0} {
    %Arg_1.2.0 = f32[2048,2048]{1,0} parameter(1), metadata={op_name="b" scheduling_name="Arg_1.2.0"}
    %Arg_0.1.0 = f32[2048,2048]{1,0} parameter(0), metadata={op_name="a" scheduling_name="Arg_0.1.0"}
    %call1 =  f32[2048,2048]{1,0} call(f32[2048,2048]{1,0} %Arg_1.2.0, f32[2048,2048]{1,0} %Arg_0.1.0 ), to_apply=%stream1.3, frontend_attributes={_xla_stream_annotation="1"}
    ROOT %call2 =  f32[2048,2048]{1,0} call(f32[2048,2048]{1,0} %Arg_1.2.0, f32[2048,2048]{1,0} %Arg_0.1.0), to_apply=%stream1.4, frontend_attributes={_xla_stream_annotation="2"}
  })";

  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %call-start = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(f32[2048,2048]{1,0} %Arg_1.2.0, f32[2048,2048]{1,0} %Arg_0.1.0), async_execution_thread="explicit", to_apply=%stream1.3, frontend_attributes={_xla_stream_annotation="1"}
  // CHECK: %call-done = f32[2048,2048]{1,0} call-done(((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) %call-start), frontend_attributes={_xla_stream_annotation="1"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false}
  // CHECK: %call-start.1 = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(f32[2048,2048]{1,0} %Arg_1.2.0, f32[2048,2048]{1,0} %Arg_0.1.0), async_execution_thread="explicit", to_apply=%stream1.4, frontend_attributes={_xla_stream_annotation="2"}
  // CHECK: ROOT %call-done.1 = f32[2048,2048]{1,0} call-done(((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) %call-start.1), frontend_attributes={_xla_stream_annotation="2"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false}
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}
}  // namespace
}  // namespace xla::gpu

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/pjrt/c/pjrt_c_api_cpu.h"

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_test.h"
#include "xla/pjrt/c/pjrt_c_api_test_base.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {
namespace pjrt {
namespace {

const bool kUnused =
    (RegisterPjRtCApiTestFactory([]() { return GetPjrtApi(); }), true);

class PjrtCApiCpuTest : public PjrtCApiTestBase {
 public:
  PjrtCApiCpuTest() : PjrtCApiTestBase() {
    api_ = GetPjrtApi();
    initialize();
  }
};

TEST_F(PjrtCApiCpuTest, PlatformName) {
  PJRT_Client_PlatformName_Args args;
  args.client = client_;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.priv = nullptr;
  PJRT_Error* error = api_->PJRT_Client_PlatformName(&args);
  ASSERT_EQ(error, nullptr);
  absl::string_view platform_name(args.platform_name, args.platform_name_size);
  ASSERT_EQ("cpu", platform_name);
}

}  // namespace
}  // namespace pjrt
}  // namespace xla

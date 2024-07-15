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

#include "mlir/InitAllExtensions.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "xla/service/gpu/prevent_mmav3_loop_unrolling.h"
#include "xla/service/gpu/triton_sparse_extensions.h"
#include "third_party/triton/bin/RegisterTritonDialects.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  registerTritonDialects(registry);  // This registers all passes as well.
  xla::gpu::RegisterSparsePasses();
  xla::gpu::RegisterPreventMmaV3LoopUnrollingPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "xla-opt modular optimizer driver\n", registry));
}

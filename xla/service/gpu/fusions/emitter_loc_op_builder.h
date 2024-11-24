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

#ifndef XLA_SERVICE_GPU_FUSIONS_EMITTER_LOC_OP_BUILDER_H_
#define XLA_SERVICE_GPU_FUSIONS_EMITTER_LOC_OP_BUILDER_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

namespace xla::gpu {

// A class that sets the location of the created instructions to the location of
// the caller. It is useful for tracking up the the emitter file and line from
// the generated MLIR. If a function receives the builder by value then the
// location of the instructions created by this builder will be chained with the
// location of the original builder.
class EmitterLocOpBuilder : public mlir::ImplicitLocOpBuilder {
 public:
// TODO(loislo): Remove this once we migrate XLA to C++20.
#if defined(PLATFORM_GOOGLE)
  typedef absl::SourceLocation SourceLocation;
#else
  class SourceLocation {
   public:
    static SourceLocation current() { return SourceLocation(); }
    std::string_view file_name() const { return "unknown"; }
    int line() const { return 0; }
  };
#endif

  // Constructor that takes the op builder and a flag indicating whether to
  // annotate the location of the operations.
  EmitterLocOpBuilder(mlir::ImplicitLocOpBuilder& op_builder, bool annotate_loc)
      : mlir::ImplicitLocOpBuilder(op_builder),
        current_loc_(op_builder.getLoc()),
        annotate_loc_(annotate_loc) {}

  EmitterLocOpBuilder(mlir::Location loc, mlir::OpBuilder& op_builder,
                      bool annotate_loc = false)
      : mlir::ImplicitLocOpBuilder(loc, op_builder),
        current_loc_(loc),
        annotate_loc_(annotate_loc) {}

  EmitterLocOpBuilder(mlir::Location loc, mlir::MLIRContext* mlir_context,
                      bool annotate_loc = false)
      : mlir::ImplicitLocOpBuilder(loc, mlir_context),
        current_loc_(loc),
        annotate_loc_(annotate_loc) {}

  explicit EmitterLocOpBuilder(
      mlir::MLIRContext* mlir_context, bool annotate_loc = false,
      SourceLocation location = SourceLocation::current())
      : mlir::ImplicitLocOpBuilder(loc(location), mlir_context),
        current_loc_(loc(location)),
        annotate_loc_(annotate_loc) {}

  EmitterLocOpBuilder& operator=(const EmitterLocOpBuilder&) = delete;

  // Copy constructor that also remembers the source location where the copy
  // was created. If the helper functions that gets the builder as the argument
  // receives the argument by value then the current location points to the
  // place where the copy was created.
  EmitterLocOpBuilder(const EmitterLocOpBuilder& builder,
                      SourceLocation location = SourceLocation::current())
      : mlir::ImplicitLocOpBuilder(builder),
        current_loc_(builder.loc(location)),
        annotate_loc_(builder.annotate_loc_) {}

  // Helper function to create a location from a source location.
  mlir::Location loc(SourceLocation location) const {
    if (!annotate_loc_ || location.line() == 0) {
      return current_loc_;
    }
    std::vector<std::string> file_name =
        absl::StrSplit(location.file_name(), '/');
    std::string previous_loc;
    if (mlir::isa<mlir::NameLoc>(current_loc_)) {
      auto name_loc = mlir::cast<mlir::NameLoc>(current_loc_);
      previous_loc = name_loc.getName().str();
    }

    const std::string text = absl::StrCat(
        previous_loc, " -> ", file_name.back(), ":", location.line());
    return mlir::NameLoc::get(mlir::StringAttr::get(getContext(), text));
  }

  // Below is the set of create() methods that are used to create operations.
  // These are all templated to allow for the creation of operations with
  // different numbers of arguments.
  //
  // For some reason the version of create that accepts the variadic arguments
  // and a source location with the default value does not work.

  template <typename OpTy>
  OpTy create(SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(loc(location));
  }

  // Create an operation with the given type and one argument.
  template <typename OpTy, typename Arg0>
  OpTy create(Arg0&& arg, SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(loc(location), std::forward<Arg0>(arg));
  }
  template <typename OpTy, typename Arg0, typename Arg1>
  OpTy create(Arg0&& arg0, Arg1&& arg1,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(loc(location), std::forward<Arg0>(arg0),
                                   std::forward<Arg1>(arg1));
  }
  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(loc(location), std::forward<Arg0>(arg0),
                                   std::forward<Arg1>(arg1),
                                   std::forward<Arg2>(arg2));
  }

  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(
        loc(location), std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
        std::forward<Arg2>(arg2), std::forward<Arg3>(arg3));
  }

  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3, typename Arg4>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(
        loc(location), std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
        std::forward<Arg2>(arg2), std::forward<Arg3>(arg3),
        std::forward<Arg4>(arg4));
  }

  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3, typename Arg4, typename Arg5>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4,
              Arg5&& arg5,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(
        loc(location), std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
        std::forward<Arg2>(arg2), std::forward<Arg3>(arg3),
        std::forward<Arg4>(arg4), std::forward<Arg5>(arg5));
  }
  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4,
              Arg5&& arg5, Arg6&& arg6,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(
        loc(location), std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
        std::forward<Arg2>(arg2), std::forward<Arg3>(arg3),
        std::forward<Arg4>(arg4), std::forward<Arg5>(arg5),
        std::forward<Arg6>(arg6));
  }

  mlir::Location current_loc() const { return current_loc_; }

  bool annotate_loc() const { return annotate_loc_; }

 private:
  const mlir::Location current_loc_;
  const bool annotate_loc_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_FUSIONS_EMITTER_LOC_OP_BUILDER_H_

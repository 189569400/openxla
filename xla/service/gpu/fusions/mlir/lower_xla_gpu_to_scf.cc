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
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_LOWERXLAGPUTOSCFPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

using mlir::success;

struct RewritePredicatedInsert : mlir::OpRewritePattern<PredicatedInsertOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      PredicatedInsertOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
        op, op.getCondition(),
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(
              loc, b.create<mlir::tensor::InsertOp>(
                        loc, op.getValue(), op.getDest(), op.getIndices())
                       .getResult());
        },
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, op.getDest());
        });
    return success();
  }
};

struct RewritePredicatedExtract : mlir::OpRewritePattern<PredicatedExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      PredicatedExtractOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
        op, op.getCondition(),
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(
              loc, b.create<mlir::tensor::ExtractOp>(loc, op.getSrc(),
                                                     op.getIndices())
                       .getResult());
        },
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, op.getFallback());
        });
    return success();
  }
};

struct RewriteShuffleReduce : mlir::OpRewritePattern<ShuffleReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ShuffleReduceOp op, mlir::PatternRewriter& rewriter) const override {
    int max_distance =
        op->getAttr("max_distance").cast<mlir::IntegerAttr>().getInt();
    // TODO(jreiffers): Do this in a verifier.
    if (max_distance & (max_distance - 1) || max_distance >= WarpSize()) {
      return op->emitOpError("max_distance must be a power of 2 < WarpSize()");
    }

    auto loc = op.getLoc();
    mlir::ValueRange values = op.getOperands();
    for (int distance = max_distance; distance > 0; distance /= 2) {
      llvm::SmallVector<mlir::Value> args = values;
      for (auto value : values) {
        // Shuffle within the warps.
        // TODO(jreiffers): Fix the lowering for types that are not supported by
        // mlir::gpu::ShuffleOp.
        args.push_back(
            rewriter
                .create<mlir::gpu::ShuffleOp>(loc, value, distance, WarpSize(),
                                              mlir::gpu::ShuffleMode::DOWN)
                .getShuffleResult());
      }
      values =
          rewriter
              .create<mlir::func::CallOp>(loc, op.getReducerAttr().getAttr(),
                                          op.getResultTypes(), args)
              .getResults();
    }
    rewriter.replaceOp(op, values);
    return success();
  }
};

class LowerXlaGpuToScfPass
    : public impl::LowerXlaGpuToScfPassBase<LowerXlaGpuToScfPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewritePredicatedInsert, RewritePredicatedExtract,
                 RewriteShuffleReduce>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerXlaGpuToScfPass() {
  return std::make_unique<LowerXlaGpuToScfPass>();
}

}  // namespace gpu
}  // namespace xla

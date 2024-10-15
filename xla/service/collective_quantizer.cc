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

#include "xla/service/collective_quantizer.h"

#include "xla/service/hlo_replication_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

namespace m = match;

// Holds the ops of a subgraph describing quantization (conversion to a narrower
// type after scaling by a broadcasted scalar and clamping), dequantization
// (scaling by a broadcasted scalar after type conversion to a wider type) or
// plain type conversion.
struct ConversionSubgraph {
  HloInstruction* convert = nullptr;
  HloInstruction* binary = nullptr;
  HloInstruction* clamp = nullptr;
  HloInstruction* scale_bcast = nullptr;
  // Unary instructions following a dequantization or preceding a quantization
  // in top-down order (operand-to-user).
  std::vector<HloInstruction*> unaries;

  std::string ToString() const {
    std::string result;
    absl::StrAppend(&result, "\n  Convert Op: ",
                    convert == nullptr ? "null" : convert->name());
    absl::StrAppend(&result, "\n  Binary Op: ",
                    binary == nullptr ? "null" : binary->name());
    absl::StrAppend(
        &result, "\n  Clamp Op: ", clamp == nullptr ? "null" : clamp->name());
    absl::StrAppend(&result, "\n  Scale Broadcast Op: ",
                    scale_bcast == nullptr ? "null" : scale_bcast->name());

    for (const auto* unary : unaries) {
      CHECK(unary != nullptr);
      absl::StrAppend(&result, "\n  Unary Op: ", unary->name());
    }

    return result;
  }
};

// Matches a broadcast of a scalar operand.
template <typename... Args>
auto ScalarBroadcast(Args... args) {
  return m::Broadcast(args...).WithPredicate([](const HloInstruction* instr) {
    return ShapeUtil::IsEffectiveScalar(instr->operand(0)->shape());
  });
}

// Matches a bitcast that preserves the element type of the operand.
auto BitcastPreservesElementType() {
  return m::Bitcast().WithPredicate([](const HloInstruction* instr) {
    return ShapeUtil::SameElementType(instr->shape(),
                                      instr->operand(0)->shape());
  });
}

// Matches a type conversion to a type with a smaller byte size than that of the
// operand.
auto ConvertToNarrowerType() {
  auto converts_to_narrower_type = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::ByteSizeOfPrimitiveType(instr->shape().element_type()) <
           ShapeUtil::ByteSizeOfPrimitiveType(
               instr->operand(0)->shape().element_type());
  };
  return m::Convert().WithPredicate(converts_to_narrower_type);
}

// Matches a type conversion to a type with a larger byte size than that of the
// operand.
auto ConvertToWiderType() {
  auto converts_to_wider_type = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::ByteSizeOfPrimitiveType(instr->shape().element_type()) >
           ShapeUtil::ByteSizeOfPrimitiveType(
               instr->operand(0)->shape().element_type());
  };
  return m::Convert().WithPredicate(converts_to_wider_type);
}

bool IsSupportedCollective(HloInstruction* instr) {
  return instr->operand_count() == 1 &&
         (instr->opcode() == HloOpcode::kAllGather ||
          instr->opcode() == HloOpcode::kAllToAll ||
          instr->opcode() == HloOpcode::kCollectiveBroadcast ||
          instr->opcode() == HloOpcode::kCollectivePermute);
}

// Sequentially applies the ops in unaries to the output of instr.
HloInstruction* ApplyUnaries(HloInstruction* instr,
                             const std::vector<HloInstruction*>& unaries) {
  for (HloInstruction* unary : unaries) {
    instr = instr->AddInstruction(unary->CloneWithNewOperands(
        ShapeUtil::MakeShapeWithDenseLayout(
            instr->shape().element_type(), unary->shape().dimensions(),
            unary->shape().layout().minor_to_major()),
        {instr}));
  }
  return instr;
}

// Recursively collects and returns unary, divide, or multiply operands of instr
// until a conversion to a wider type is reached. Returns the collected ops in
// bottom-up order (user-to-operand). Returns an empty vector when no conversion
// is reached.
std::vector<HloInstruction*> FindDequantizationSubgraphRecursive(
    HloInstruction* instr, absl::flat_hash_set<int>& visited_instrs,
    std::vector<HloInstruction*> subgraph) {
  // Avoid visiting the same instruction more than once.
  if (!visited_instrs.emplace(instr->unique_id()).second) {
    return {};
  }

  subgraph.emplace_back(instr);
  if (Match(instr, ConvertToWiderType())) {
    return subgraph;
  }
  if (instr->operand_count() == 1 || instr->opcode() == HloOpcode::kDivide) {
    return FindDequantizationSubgraphRecursive(instr->mutable_operand(0),
                                               visited_instrs, subgraph);
  } else if (instr->opcode() == HloOpcode::kMultiply) {
    for (HloInstruction* operand : instr->unique_operands()) {
      auto binary_subgraph = FindDequantizationSubgraphRecursive(
          operand, visited_instrs, subgraph);
      if (!binary_subgraph.empty()) {
        return binary_subgraph;
      }
    }
  }
  return {};
}

// Returns non-nullopt if instr describes a dequantization, i.e. a
// multiplication or division by a broadcasted scalar operating on a type
// conversion, or a plain type conversion to a wider type. Also returns
// non-nullopt if instr is the last instruction of a sequence of unary bitcast,
// copy, reshape or slice ops preceded by a dequantization.
std::optional<ConversionSubgraph> IsSupportedDequantization(
    HloInstruction* instr) {
  ConversionSubgraph subgraph;
  absl::flat_hash_set<int> visited_instrs;
  std::vector<HloInstruction*> candidate_subgraph =
      FindDequantizationSubgraphRecursive(instr, visited_instrs,
                                          std::vector<HloInstruction*>{});
  std::reverse(candidate_subgraph.begin(), candidate_subgraph.end());

  // In the dequantization case, the type conversion is followed by a
  // multiplication or division by a broadcasted scalar.
  if (candidate_subgraph.size() > 1 &&
      (Match(
           candidate_subgraph[1],
           m::MultiplyAnyOrder(&subgraph.binary, m::Convert(&subgraph.convert),
                               ScalarBroadcast(&subgraph.scale_bcast))) ||
       Match(candidate_subgraph[1],
             m::Divide(&subgraph.binary, m::Convert(&subgraph.convert),
                       ScalarBroadcast(&subgraph.scale_bcast))))) {
    subgraph.unaries = {candidate_subgraph.begin() + 2,
                        candidate_subgraph.end()};
  } else if (!candidate_subgraph.empty() &&
             Match(candidate_subgraph[0], m::Convert(&subgraph.convert))) {
    subgraph.unaries = {candidate_subgraph.begin() + 1,
                        candidate_subgraph.end()};
  } else {
    VLOG(5) << "Did not find type conversion or dequantization pattern.";
    return std::nullopt;
  }

  VLOG(5) << "Conversion subgraph found for supporting dequantization of the "
             "instruction: "
          << instr->name() << subgraph.ToString();

  // The collected unary ops between dequantization/type conversion and
  // collective may only include bitcast, copy, reshape and slice instructions.
  for (HloInstruction* unary : subgraph.unaries) {
    if (!Match(unary, m::AnyOf<HloInstruction>(m::Bitcast(), m::Copy(),
                                               m::Reshape(), m::Slice()))) {
      VLOG(5) << "Unexpected instruction in unary ops.";
      return std::nullopt;
    }
  }
  return std::make_optional<ConversionSubgraph>(std::move(subgraph));
}

// Returns non-nullopt if instr describes a quantization, i.e. a multiplication
// or division by a broadcasted scalar followed by a clamp and a type
// conversion, or a plain type conversion to a narrower type. Also returns
// non-nullopt if instr is the first instruction of a sequence of unary bitcast,
// copy, reshape or slice ops followed by a quantization.
std::optional<ConversionSubgraph> IsSupportedQuantization(
    HloInstruction* instr) {
  ConversionSubgraph subgraph;
  std::vector<HloInstruction*> ops;
  while (instr->user_count() <= 1) {
    if (Match(instr, m::AnyOf<HloInstruction>(
                         BitcastPreservesElementType(), m::Copy(), m::Reshape(),
                         m::Slice(), m::Multiply(), m::Divide(), m::Clamp()))) {
      if (instr->user_count() > 0) {
        ops.emplace_back(instr);
        instr = instr->users()[0];
        continue;
      }
      break;
    }

    if (Match(instr, ConvertToNarrowerType())) {
      ops.emplace_back(instr);
      break;
    }
    VLOG(5) << "Unsupported instruction.";
    return std::nullopt;
  }

  // In the quantization case, the type conversion is preceded by a
  // multiplication or division by a broadcasted scalar and a clamp instruction.
  if (ops.size() > 2 &&
      (Match(
           ops.back(),
           m::Convert(&subgraph.convert,
                      m::Clamp(&subgraph.clamp, ScalarBroadcast(m::Constant()),
                               m::MultiplyAnyOrder(
                                   &subgraph.binary, m::Op(),
                                   ScalarBroadcast(&subgraph.scale_bcast)),
                               ScalarBroadcast(m::Constant())))) ||
       Match(ops.back(),
             m::Convert(
                 &subgraph.convert,
                 m::Clamp(&subgraph.clamp, ScalarBroadcast(m::Constant()),
                          m::Divide(&subgraph.binary, m::Op(),
                                    ScalarBroadcast(&subgraph.scale_bcast)),
                          ScalarBroadcast(m::Constant())))))) {
    subgraph.unaries = {ops.begin(), ops.end() - 3};
  } else if (!ops.empty() && Match(ops.back(), m::Convert(&subgraph.convert))) {
    subgraph.unaries = {ops.begin(), ops.end() - 1};
  } else {
    VLOG(5) << "Did not find type conversion or quantization pattern.";
    return std::nullopt;
  }

  VLOG(5) << "Conversion subgraph found for supporting quantization of the "
             "instruction: "
          << instr->name() << subgraph.ToString();

  // The collected unary ops between collective and quantization/type conversion
  // may only include bitcast, copy, reshape and slice instructions.
  for (HloInstruction* unary : subgraph.unaries) {
    if (!Match(unary, m::AnyOf<HloInstruction>(m::Bitcast(), m::Copy(),
                                               m::Reshape(), m::Slice()))) {
      VLOG(5) << "Unexpected instruction in unary ops.";
      return std::nullopt;
    }
  }
  return std::make_optional<ConversionSubgraph>(std::move(subgraph));
}

absl::StatusOr<bool> MatchDequantization(HloInstruction* instr) {
  VLOG(5) << "Attempt to dequantize collective " << instr->ToShortString();
  std::optional<ConversionSubgraph> subgraph =
      IsSupportedDequantization(instr->mutable_operand(0));

  if (!subgraph.has_value()) {
    return false;
  }

  HloInstruction* new_coll_operand = subgraph->convert->mutable_operand(0);

  // Insert the collected unary ops ahead of the new collective.
  new_coll_operand = ApplyUnaries(new_coll_operand, subgraph->unaries);

  // Move the collective before the conversion to the wider type.
  Shape new_coll_shape = ShapeUtil::ChangeElementType(
      instr->shape(), new_coll_operand->shape().element_type());
  HloInstruction* new_collective = instr->AddInstruction(
      instr->CloneWithNewOperands(new_coll_shape, {new_coll_operand}));
  Shape new_convert_shape = ShapeUtil::ChangeElementType(
      new_collective->shape(), subgraph->convert->shape().element_type());
  HloInstruction* new_convert =
      instr->AddInstruction(subgraph->convert->CloneWithNewOperands(
          new_convert_shape, {new_collective}));

  HloInstruction* new_binary;
  // When there is a dequantization, insert the scale ops.
  if (subgraph->binary) {
    auto* new_scale_bcast_operand = subgraph->scale_bcast->mutable_operand(0);
    if (!ShapeUtil::IsScalar(new_scale_bcast_operand->shape())) {
      new_scale_bcast_operand =
          instr->parent()->AddInstruction(HloInstruction::CreateReshape(
              ShapeUtil::MakeScalarShape(
                  subgraph->scale_bcast->shape().element_type()),
              new_scale_bcast_operand));
    }
    HloInstruction* new_scale_bcast =
        instr->AddInstruction(HloInstruction::CreateBroadcast(
            new_convert->shape(), new_scale_bcast_operand, {}));
    new_binary = instr->AddInstruction(subgraph->binary->CloneWithNewOperands(
        new_convert->shape(), {new_convert, new_scale_bcast}));
  }

  TF_RETURN_IF_ERROR(
      instr->ReplaceAllUsesWith(subgraph->binary ? new_binary : new_convert));

  VLOG(5) << "Dequantized collective " << new_collective->ToShortString();
  return true;
}

absl::StatusOr<bool> MatchQuantization(HloInstruction* instr) {
  if (instr->user_count() != 1) {
    return false;
  }

  VLOG(5) << "Attempt to quantize collective " << instr->ToShortString();

  std::optional<ConversionSubgraph> subgraph =
      IsSupportedQuantization(instr->users()[0]);
  if (!subgraph.has_value()) {
    return false;
  }

  HloInstruction* coll_operand = instr->mutable_operand(0);

  HloInstruction *new_binary, *new_clamp;
  // When there is a quantization, insert the scale and clamp ops.
  if (subgraph->binary) {
    auto* new_scale_bcast_operand = subgraph->scale_bcast->mutable_operand(0);
    if (!ShapeUtil::IsScalar(new_scale_bcast_operand->shape())) {
      new_scale_bcast_operand =
          instr->parent()->AddInstruction(HloInstruction::CreateReshape(
              ShapeUtil::MakeScalarShape(
                  subgraph->scale_bcast->shape().element_type()),
              new_scale_bcast_operand));
    }
    HloInstruction* new_scale_bcast =
        instr->AddInstruction(HloInstruction::CreateBroadcast(
            coll_operand->shape(), new_scale_bcast_operand, {}));
    new_binary = instr->AddInstruction(subgraph->binary->CloneWithNewOperands(
        coll_operand->shape(), {coll_operand, new_scale_bcast}));
    HloInstruction* new_clamp_lower = instr->AddInstruction(
        subgraph->clamp->operand(0)->CloneWithNewShape(coll_operand->shape()));
    HloInstruction* new_clamp_upper = instr->AddInstruction(
        subgraph->clamp->operand(2)->CloneWithNewShape(coll_operand->shape()));
    new_clamp = instr->AddInstruction(subgraph->clamp->CloneWithNewOperands(
        coll_operand->shape(), {new_clamp_lower, new_binary, new_clamp_upper}));
  }

  // Move the collective past the conversion to the narrow type.
  Shape new_convert_shape = ShapeUtil::ChangeElementType(
      coll_operand->shape(), subgraph->convert->shape().element_type());
  HloInstruction* new_convert =
      instr->AddInstruction(subgraph->convert->CloneWithNewOperands(
          new_convert_shape, {subgraph->binary ? new_clamp : coll_operand}));
  Shape new_collective_shape = ShapeUtil::ChangeElementType(
      instr->shape(), subgraph->convert->shape().element_type());
  HloInstruction* new_collective = instr->AddInstruction(
      instr->CloneWithNewOperands(new_collective_shape, {new_convert}));

  // Insert the collected unary ops after the new collective.
  new_collective = ApplyUnaries(new_collective, subgraph->unaries);
  TF_RETURN_IF_ERROR(subgraph->convert->ReplaceAllUsesWith(new_collective));

  VLOG(5) << "Quantized collective " << new_collective->ToShortString();
  return true;
}

}  // namespace

absl::StatusOr<bool> CollectiveQuantizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (!IsSupportedCollective(instr)) {
        continue;
      }

      // Either tries matching dequantization or quantization.
      TF_ASSIGN_OR_RETURN(bool is_dequantized, MatchDequantization(instr));
      changed |= is_dequantized;

      if (is_dequantized) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(bool is_quantized, MatchQuantization(instr));
      changed |= is_quantized;
    }
  }

  return changed;
}

}  // namespace xla

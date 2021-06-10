/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdexcept>

#include <backend/basic/KernelGeneratorBase.h>

#include "KernelGenerator.h"

#include "ClTensorRegistry.h"
#include "ClFunction.h"
#include "TensorManager.h"

#include "open_cl/kernels/Elementwise.h"
#include "open_cl/selectors/SimpleSelectors.h"
#include "open_cl/selectors/ConvolutionSelector.h"
#include "open_cl/selectors/Subgraph.h"

#include "ir/Operations.h"
#include "ir/Operations.Include.h"
#include "ir/Index.h"
#include "ir/DataType.h"
#include "ir/InternalType.h"
#include "exec/NopFunction.h"
#include "exec/FunctionSequence.h"
#include "util/logging.h"
#include "util/Utils.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

KernelGenerator::KernelGenerator(const ir::Graph &graph,
                                 const std::shared_ptr<TensorBuilder> &tensor_builder,
                                 const std::shared_ptr<ClTensorRegistry<TensorManager>> &tensor_reg,
                                 const std::shared_ptr<CreationContext> &creation_context)
  : basic::KernelGeneratorBase{graph}, _ctx(graph.operands()),
    _operations_ctx(graph.operations()), _current_layout{graph.layout()},
    _tensor_builder(tensor_builder), _tensor_reg(tensor_reg), _creation_context(creation_context)
{
}

std::unique_ptr<exec::FunctionSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto ret = std::make_unique<exec::FunctionSequence>();
  ret->enableDynamicShapeInferer(false);

  const auto &op = _graph.operations().at(ind);
  op.accept(*this);
  ret->append(releaseFunction());
  return ret;
}

void KernelGenerator::visit(const ir::operation::BinaryArithmetic &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS)};

  // const auto activation = node.param().activation;

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(lhs_index)->descriptor);
  auto lhs_shape = _tensor_reg->getClTensorReserver(lhs_index)->shape;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(rhs_index)->descriptor);
  auto rhs_shape = _tensor_reg->getClTensorReserver(rhs_index)->shape;

  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(ofm_index)->descriptor);
  auto out_shape = _tensor_reg->getClTensorReserver(ofm_index)->shape;

  auto fn = std::make_unique<ClFunction>();

  std::unique_ptr<GPUOperation> gpu_op;
  switch (node.param().arithmetic_type)
  {
    case ir::operation::BinaryArithmetic::ArithmeticType::ADD:
    {
      std::vector<int> channels(2);
      channels[0] = lhs_shape.c;
      channels[1] = rhs_shape.c;
      SelectAdd(op_def, channels, out_shape.c, &gpu_op);

      auto ofm_tensor = _tensor_reg->getClTensor(ofm_index);
      auto lhs_tensor = _tensor_reg->getClTensor(lhs_index);
      auto rhs_tensor = _tensor_reg->getClTensor(rhs_index);
      gpu_op->SetSrc(lhs_tensor->handle(), ir::operation::BinaryArithmetic::Input::LHS);
      gpu_op->SetSrc(rhs_tensor->handle(), ir::operation::BinaryArithmetic::Input::RHS);
      gpu_op->SetDst(ofm_tensor->handle(), 0);

      fn->configure(std::move(gpu_op), _creation_context);
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::SUB:
    {
      // NYI
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::MUL:
    {
      // NYI
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::DIV:
    {
      // NYI
      break;
    }
    default:
      assert(false && "The BinaryArithmetic operation supports only binary arithmetic operations");
      break;
  }

  _return_fn = std::move(fn);
}

gpu_cl::Convolution2DAttributes convertConv2DParamForGPU(ir::operation::Conv2D::Param param)
{
  gpu_cl::Convolution2DAttributes attr;

  /*
  struct Param
  {
    Stride stride;
    Padding padding;
    Activation activation;
    Dilation dilation;
  };
  ------------------------------------------
  struct Convolution2DAttributes
  {
    HW strides = HW(1, 1);   // Along each axis.
    HW dilations = HW(1, 1); // Along each axis.
    Padding2D padding;
    InternalTensor<OHWI, DataType::FLOAT32> weights;
    InternalTensor<Linear, DataType::FLOAT32> bias; // optional
  };
  */
  attr.strides = HW(param.stride.horizontal, param.stride.vertical);
  attr.dilations = HW(param.dilation.height_factor, param.dilation.width_factor);

  attr.padding.prepended = HW(param.padding.param.left, param.padding.param.top);
  attr.padding.appended = HW(param.padding.param.right, param.padding.param.bottom);

  return attr;
}

void KernelGenerator::visit(const ir::operation::Conv2D &node)
{
  // Make Input/Output
  auto input{node.getInputs().at(ir::operation::Conv2D::INPUT)};
  auto kernel{node.getInputs().at(ir::operation::Conv2D::KERNEL)};
  auto bias{node.getInputs().at(ir::operation::Conv2D::BIAS)};

  auto output{node.getOutputs().at(0)};

  const auto param = node.param();

  auto fn = std::make_unique<ClFunction>();

  OperationDef op_def;
  op_def.precision = CalculationsPrecision::F32;

  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(input)->descriptor);
  // auto input_shape = _tensor_reg->getClTensorReserver(input)->shape;
  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(kernel)->descriptor);
  op_def.src_tensors.push_back(_tensor_reg->getClTensorReserver(bias)->descriptor);
 
  op_def.dst_tensors.push_back(_tensor_reg->getClTensorReserver(output)->descriptor);
  auto output_shape = _tensor_reg->getClTensorReserver(output)->shape;
 
  gpu_cl::Convolution2DAttributes attr = convertConv2DParamForGPU(param/*, weight, bias*/);

  gpu_cl::Node op_node{1, Operation{"convolution_2d", attr}};

  ModelHints hints;

  std::unique_ptr<GPUOperation> gpu_op; // = InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);
  gpu_op = SelectConvolution(attr, output_shape, _creation_context->GetDeviceInfo(), op_def, hints);

  fn->configure(std::move(gpu_op), _creation_context);
  _return_fn = std::move(fn);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

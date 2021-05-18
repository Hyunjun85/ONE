/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "InferenceContext.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "Buffer.h"
#include "ClDevice.h"

#include "kernels/GpuOperation.h"
#include "ModelHints.h"
#include "Precision.h"
#include "selectors/OperationSelector.h"
#include "StorageTypeUtil.h"
#include "TensorType.h"
#include "DataType.h"
#include "MemoryManagement.h"
#include "Model.h"
#include "Operations.h"
#include "Shape.h"
#include "Types.h"
#include "Util.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

namespace
{
bool IsReady(const std::set<ValueId> &ready_tensors, const CLNode &node)
{
  for (const ValueId in_id : node.inputs)
  {
    if (ready_tensors.find(in_id) == ready_tensors.end())
    {
      return false;
    }
  }
  return true;
}

bool MergeCLNodes(CLNode *src, CLNode *dst)
{
  for (uint32_t j = 1; j < src->inputs.size(); ++j)
  {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " linked : " + src->name;
  return dst->operation->AddOperation(src->operation.get());
}

} // namespace

CLNode::CLNode(CLNode &&node)
  : operation(std::move(node.operation)), inputs(std::move(node.inputs)),
    outputs(std::move(node.outputs)), name(std::move(node.name))
{
}

CLNode &CLNode::operator=(CLNode &&node)
{
  if (this != &node)
  {
    operation = std::move(node.operation);
    inputs = std::move(node.inputs);
    outputs = std::move(node.outputs);
    name = std::move(node.name);
  }
  return *this;
}

bool InferenceContext::Merge()
{
  std::set<ValueId> ready_tensors;
  for (const auto &input_id : input_ids_)
  {
    ready_tensors.insert(input_id);
  }
  for (uint32_t i = 0; i < nodes_.size(); ++i)
  {
    auto &node = nodes_[i];
    for (const auto &out_id : node.outputs)
    {
      ready_tensors.insert(out_id);
    }
    if (node.outputs.size() != 1)
    {
      continue;
    }
    std::vector<int> next_nodes;
    int link_index = 0;
    for (uint32_t j = i + 1; j < nodes_.size(); ++j)
    {
      for (uint32_t k = 0; k < nodes_[j].inputs.size(); ++k)
      {
        if (nodes_[j].inputs[k] == node.outputs[0])
        {
          next_nodes.push_back(j);
          link_index = k;
        }
      }
    }
    if (next_nodes.size() != 1 || link_index != 0)
    {
      continue;
    }
    auto &linkable_node = nodes_[next_nodes[0]];
    if (!linkable_node.operation->IsLinkable() || linkable_node.outputs.size() != 1 ||
        !IsReady(ready_tensors, linkable_node))
    {
      continue;
    }
    const auto &original_dst_def = node.operation->GetDefinition().dst_tensors[0];
    const auto &link_dst_def = linkable_node.operation->GetDefinition().dst_tensors[0];
    if (original_dst_def != link_dst_def)
    {
      continue;
    }
    if (MergeCLNodes(&linkable_node, &node) == false)
      return false;
    nodes_.erase(nodes_.begin() + next_nodes[0]);
    i -= 1;
  }
  return true;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

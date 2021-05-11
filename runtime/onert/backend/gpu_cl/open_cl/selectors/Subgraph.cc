/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "Subgraph.h"

#include <memory>

#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/TensorType.h"
#include "open_cl/Model.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

std::unique_ptr<GPUOperation> *InitSingleOpSubgraph(const std::vector<Value *> &inputs,
                                                    const std::vector<Value *> &outputs,
                                                    GPUOperationsSubgraph *gpu_subgraph)
{
  gpu_subgraph->operations.clear();
  gpu_subgraph->new_tensors.clear();
  gpu_subgraph->operations.push_back({});
  for (uint32_t i = 0; i < inputs.size(); ++i)
  {
    gpu_subgraph->operations[0].input_ids.push_back(inputs[i]->id);
  }
  for (uint32_t i = 0; i < outputs.size(); ++i)
  {
    gpu_subgraph->operations[0].output_ids.push_back(outputs[i]->id);
  }

  return &gpu_subgraph->operations[0].operation;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

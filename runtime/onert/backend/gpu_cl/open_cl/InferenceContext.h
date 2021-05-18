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

#ifndef __ONERT_BACKEND_GPU_CL_INFERENCE_CONTEXT_H__
#define __ONERT_BACKEND_GPU_CL_INFERENCE_CONTEXT_H__

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <unordered_map>

#include "Buffer.h"
#include "ClCommandQueue.h"
#include "Environment.h"
#include "GpuObject.h"
#include "kernels/GpuOperation.h"
#include "ModelHints.h"
#include "OpenclWrapper.h"
#include "Precision.h"
#include "TensorType.h"
#include "Model.h"
#include "InternalTensor.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

struct CLNode
{
  std::unique_ptr<GPUOperation> operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Mostly for debug purposes.
  std::string name;

  CLNode() = default;

  CLNode(CLNode &&node);
  CLNode &operator=(CLNode &&node);
  CLNode(const CLNode &) = delete;
  CLNode &operator=(const CLNode &) = delete;
};

class InferenceContext
{
public:
  struct CreateInferenceInfo
  {
    CalculationsPrecision precision;
    TensorStorageType storage_type;
    ModelHints hints;
  };

  struct DummyTensor
  {
    BHWC shape;
    TensorDescriptor descriptor;

    bool operator==(const DummyTensor &b) const
    {
      return shape == b.shape && descriptor == b.descriptor;
    }
  };

  class TensorReserver
  {
  public:
    ValueId Add(const std::shared_ptr<DummyTensor> dummy)
    {
      reservations_[next_] = std::move(dummy);
      return next_++;
    }
    void Add(ValueId id, const std::shared_ptr<DummyTensor> dummy)
    {
      reservations_[id] = std::move(dummy);
    }
    void SetNext(ValueId id) { next_ = id; }
    bool HaveTensor(ValueId id) { return reservations_.find(id) != reservations_.end(); }
    std::shared_ptr<DummyTensor> Get(ValueId id) { return reservations_[id]; }

    std::vector<std::pair<ValueId, TensorDescriptor>> GetTensorDescs() const
    {
      std::vector<std::pair<ValueId, TensorDescriptor>> result;
      for (auto &v : reservations_)
      {
        TensorDescriptor desc = v.second->descriptor;
        desc.shape.b = v.second->shape.b;
        desc.shape.h = v.second->shape.h;
        desc.shape.w = v.second->shape.w;
        desc.shape.d = 1;
        desc.shape.c = v.second->shape.c;
        result.push_back({v.first, desc});
      }
      return result;
    }

    void Add(const std::vector<std::pair<ValueId, TensorDescriptor>> &tensors)
    {
      for (auto &v : tensors)
      {
        auto dummy = std::make_shared<DummyTensor>();
        dummy->descriptor = v.second;
        dummy->shape.b = v.second.shape.b;
        dummy->shape.h = v.second.shape.h;
        dummy->shape.w = v.second.shape.w;
        dummy->shape.c = v.second.shape.c;
        Add(v.first, dummy);
      }
    }

  private:
    std::unordered_map<ValueId, std::shared_ptr<DummyTensor>> reservations_;
    ValueId next_;
  };

private:
  enum TensorMemoryType
  {
    STRONG_SHAPE = 0,
    BUFFER = 1,
    VARIABLE = 2
  };

  void CopyInAndOutIds(const GraphFloat32 &graph);
  // bool ConvertOperations(const DeviceInfo& device_info,
  //                                const GraphFloat32& graph, ModelHints hints);
  void CreateLinks();
  void ReserveGraphTensors(const CreateInferenceInfo &create_info, const DeviceInfo &device_info,
                           const GraphFloat32 &graph);
  bool Merge();

  // performance hacks
  bool need_flush_ = false;

  bool flush_periodically_ = false;
  int flush_period_ = 1;

  // In order to reduce memory leak on Mali a pipeline needs to be synchronized
  // with CPU to prevent growing internal global OpenCL kernel pool. One trick
  // is to enqueue an event from a previous run. Most of the time is should
  // already be executed on GPU and should not stall the pipeline.
  bool need_manual_release_ = false;
  CLEvent prev_enqueue_start_point_;

  CalculationsPrecision precision_;
  TensorStorageType storage_type_;

  // Directly mapped nodes from graph, but some of them "inactive" due
  //  to fusion (inactive = fused).
  // Memory is allocated only once, in ConvertOperations, and is not modified
  //  anywhere.
  std::vector<CLNode> nodes_;

  TensorReserver tensor_reserver_;

  std::map<ValueId, Tensor> variable_tensors_;
  std::vector<Buffer> shared_buffers_;
  std::vector<Tensor> shared_buffer_tensors_; // use references to memory from shared_buffers_
  std::map<ValueId, int> graph_ids_to_shared_buffer_tensors_;

  std::map<ValueId, Tensor> strong_shape_tensors_;
  std::map<ValueId, ValueId> graph_ids_to_strong_shape_tensors_;

  std::vector<ValueId> input_ids_;
  std::map<ValueId, ValueId> variable_ids_and_refs_;
  std::vector<ValueId> output_ids_;
};

// Runs OpenCL specific transforms for the graph.
bool RunGraphTransforms(GraphFloat32 *graph);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_INFERENCE_CONTEXT_H__

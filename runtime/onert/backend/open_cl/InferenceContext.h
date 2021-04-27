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
    ValueId Add(const DummyTensor &dummy)
    {
      reservations_[next_] = dummy;
      return next_++;
    }
    void Add(ValueId id, const DummyTensor &dummy) { reservations_[id] = dummy; }
    void SetNext(ValueId id) { next_ = id; }
    DummyTensor Get(ValueId id) { return reservations_[id]; }

    std::vector<std::pair<ValueId, TensorDescriptor>> GetTensorDescs() const
    {
      std::vector<std::pair<ValueId, TensorDescriptor>> result;
      for (auto &v : reservations_)
      {
        TensorDescriptor desc = v.second.descriptor;
        desc.shape.b = v.second.shape.b;
        desc.shape.h = v.second.shape.h;
        desc.shape.w = v.second.shape.w;
        desc.shape.d = 1;
        desc.shape.c = v.second.shape.c;
        result.push_back({v.first, desc});
      }
      return result;
    }

    void Add(const std::vector<std::pair<ValueId, TensorDescriptor>> &tensors)
    {
      for (auto &v : tensors)
      {
        DummyTensor dummy;
        dummy.descriptor = v.second;
        dummy.shape.b = v.second.shape.b;
        dummy.shape.h = v.second.shape.h;
        dummy.shape.w = v.second.shape.w;
        dummy.shape.c = v.second.shape.c;
        Add(v.first, dummy);
      }
    }

  private:
    std::unordered_map<ValueId, DummyTensor> reservations_;
    ValueId next_;
  };
  bool InitFromGraph(const CreateInferenceInfo &create_info, const GraphFloat32 &graph,
                     Environment *env, std::vector<uint8_t> *serialized_model = nullptr);

  // Applies OpenCL-specific transformations to the graph before the
  // initialization. These transformations are either impossible or useless in
  // other backends.
  bool InitFromGraphWithTransforms(const CreateInferenceInfo &create_info, GraphFloat32 *graph,
                                   Environment *env,
                                   std::vector<uint8_t> *serialized_model = nullptr);

  bool AddToQueue(CLCommandQueue *queue);
  // bool Profile(ProfilingCommandQueue* queue, ProfilingInfo* result);
  // for profiling and memory statistics
  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;

  // bool SetInputTensor(ValueId id, const TensorFloat32& tensor,
  //                             CLCommandQueue* queue);

  // It will work only with input/output tensor ids. For all other ids we don't
  // have any guarantees.
  Tensor *GetTensor(ValueId id);

  // bool GetOutputTensor(ValueId id, CLCommandQueue* queue,
  //                              TensorFloat32* result);

  const std::vector<ValueId> &GetInputIds() const { return input_ids_; }
  const std::vector<ValueId> &GetOutputIds() const { return output_ids_; }

  bool RestoreDeserialized(const std::vector<uint8_t> &serialized_model, Environment *env);

private:
  enum TensorMemoryType
  {
    STRONG_SHAPE = 0,
    BUFFER = 1,
    VARIABLE = 2
  };

  // friend flatbuffers::Offset<data::InferenceContext> Encode(
  //     const InferenceContext& inference,
  //     flatbuffers::FlatBufferBuilder* builder);
  // friend bool Decode(CLContext* context,
  //                            const data::InferenceContext* fb_inference,
  //                            InferenceContext* inference);

  void CopyInAndOutIds(const GraphFloat32 &graph);
  // bool ConvertOperations(const DeviceInfo& device_info,
  //                                const GraphFloat32& graph, ModelHints hints);
  void CreateLinks();
  void ReserveGraphTensors(const CreateInferenceInfo &create_info, const DeviceInfo &device_info,
                           const GraphFloat32 &graph);
  bool Merge();
  bool AllocateMemory(CLContext *context);

  bool AllocateMemoryForVariableTensors(CLContext *context);

  bool AllocateMemoryForBuffers(CLContext *context);

  bool AllocateMemoryForStrongShapes(CLContext *context);

  // utility function
  void GetUsages(const std::function<bool(ValueId)> &functor, std::map<ValueId, int2> *usages);

  TensorMemoryType GetTensorMemoryType(ValueId id);

  void BindMemoryToOperations();
  bool Compile(const CreationContext &creation_context);
  bool Tune(const TuningParameters &tuning_parameters);
  bool UpdateParams();

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

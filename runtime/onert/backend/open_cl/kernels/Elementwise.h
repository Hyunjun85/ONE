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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ELEMENTWISE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ELEMENTWISE_H_

#include <string>

#include "GpuOperation.h"
#include "../Operations.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

// Creates simple one input operation without any parameters, for example
// log, sin, cos, etc.
GPUOperation CreateElementwiseOneInput(const OperationDef &definition,
                                       const OperationType &op_type);

// Creates simple two input(first input is runtime tensor and second input is
// constant or linear/hwc tensor) operation, for example sub, div and etc.
GPUOperation CreateElementwise(const DeviceInfo &device_info, const OperationDef &definition,
                               const OperationType &op_type, const ElementwiseAttributes &attr);

// Creates simple two input(2 runtime tensors) operation, for example
// sub, div and etc.
GPUOperation CreateElementwiseTwoInput(const OperationDef &definition, const OperationType &op_type,
                                       const BHWC &shape);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ELEMENTWISE_H_

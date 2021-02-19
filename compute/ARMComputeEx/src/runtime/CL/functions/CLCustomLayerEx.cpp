/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

/*
 * Copyright (c) 2018-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "arm_compute/runtime/CL/functions/CLCustomLayerEx.h"
#include "arm_compute/core/CL/kernels/CLCustomLayerKernelEx.h"
#include "arm_compute/core/CL/ICLTensor.h"

namespace arm_compute
{
// namespace
// {
// } // namespace

CLCustomLayerEx::CLCustomLayerEx(std::shared_ptr<IMemoryManager> memory_manager)
  : _memory_group(std::move(memory_manager))
{
}

Status CLCustomLayerEx::validate(const ITensorInfo *input, 
                                          const ITensorInfo *output)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  return Status{};
}

void CLCustomLayerEx::configure(const ICLTensor *inputs,
                                         ICLTensor *outputs )
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(inputs, outputs);

  auto k = support::cpp14::make_unique<CLCustomLayerKernelEx>();
  k->configure(inputs, outputs);
  _kernel = std::move(k);
}

// void CLCustomLayerEx::run(/*const Window &window, cl::CommandQueue &queue*/)
// {
  
// }
} // namespace arm_compute

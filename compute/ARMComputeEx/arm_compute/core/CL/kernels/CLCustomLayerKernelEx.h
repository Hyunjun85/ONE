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
 * Copyright (c) 2019-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_CLARGMINMAXLAYERKERNELEX_H
#define ARM_COMPUTE_CLARGMINMAXLAYERKERNELEX_H

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the reduction operation kernel
 *
 */
class CLCustomLayerKernelEx : public ICLKernel
{
public:
  /** Default constructor */
  CLCustomLayerKernelEx();

  /** Default destructor */
  ~CLCustomLayerKernelEx() = default;

 /** Set the input and output tensors.
   *
   * @param[in]  input       Source tensor. Data types supported: S32/F16/F32.
   * @param[out] output      Destination tensor. Data types supported: U32/S32
   *                         Output will have the same number of dimensions as input.
   */
  void configure(const ICLTensor *input, ICLTensor *output);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLArgMinMaxLayerKernelEx.
   *
   * @param[in] input       Source tensor info. Data types supported: S32/F16/F32.
   * @param[in] output      Destination tensor info. Data types supported: U32/S32
   *                        Output will have the same number of dimensions as input.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLCUSTOMERKERNELEX_H */

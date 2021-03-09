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
 * Copyright (c) 2018-2020 ARM Limited.
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

/**
 * @file      CLCastBoolKernel.h
 * @ingroup   COM_AI_RUNTIME
 * @brief     This file defines CLCastBoolKernel class
 */

#ifndef __ARM_COMPUTE_CLCASTBOOLKERNEL_H__
#define __ARM_COMPUTE_CLCASTBOOLKERNEL_H__

#include "src/core/CL/ICLSimple3DKernel.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class for the kernel converting boolean type
 */
class CLCastBoolKernel : public ICLSimple3DKernel
{
public:
  /**
   * @brief Initialise the kernel's input and output.
   * @param[in]  input  Input tensor. Data types supported: U8
   * @param[in]  output Output tensor. Data types supported: U8/QASYMM8/S16/S32/F16/F32.
   * @return N/A
   */
  void configure(const ICLTensor *input, ICLTensor *output);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLCastBoolKernel
   *
   * @param[in] input  Source tensor info. Data types supported: U8.
   * @param[in] output Destination tensor info. Data type supported: U8/S8/U16/S16/U32/S32/F16/F32.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLCASTBOOLKERNEL_H__ */

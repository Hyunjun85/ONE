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

#include "ClKernel.h"

#include "ClProgram.h"
#include "Util.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace
{

bool GetKernelMaxWorkGroupSize(cl_kernel kernel, cl_device_id device_id, int *result)
{
  size_t max_work_group_size;
  cl_int error_code = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                               sizeof(size_t), &max_work_group_size, nullptr);
  if (error_code != CL_SUCCESS)
  {
    return false;
  }
  *result = static_cast<int>(max_work_group_size);
  return true;
}

bool GetKernelPrivateMemorySize(cl_kernel kernel, cl_device_id device_id, int *result)
{
  cl_ulong private_mem_size;
  cl_int error_code = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PRIVATE_MEM_SIZE,
                                               sizeof(cl_ulong), &private_mem_size, nullptr);
  if (error_code != CL_SUCCESS)
  {
    return false;
  }
  *result = static_cast<int>(private_mem_size);
  return true;
}

} // namespace

CLKernel::CLKernel(CLKernel &&kernel)
  : info_(kernel.info_), binding_counter_(kernel.binding_counter_),
    function_name_(std::move(kernel.function_name_)), program_(kernel.program_),
    kernel_(kernel.kernel_)
{
  kernel.kernel_ = nullptr;
}

CLKernel &CLKernel::operator=(CLKernel &&kernel)
{
  if (this != &kernel)
  {
    Release();
    std::swap(info_, kernel.info_);
    std::swap(binding_counter_, kernel.binding_counter_);
    function_name_ = std::move(kernel.function_name_);
    std::swap(program_, kernel.program_);
    std::swap(kernel_, kernel.kernel_);
  }
  return *this;
}

CLKernel::~CLKernel() { Release(); }

bool CLKernel::ReInit() const
{
  clReleaseKernel(kernel_);
  cl_kernel *kern_ptr = const_cast<cl_kernel *>(&kernel_);
  int error_code;
  *kern_ptr = clCreateKernel(program_, function_name_.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS)
  {
    *kern_ptr = nullptr;
    return false;
  }
  return true;
}

void CLKernel::Release()
{
  if (kernel_)
  {
    clReleaseKernel(kernel_);
    clReleaseProgram(program_);
    kernel_ = nullptr;
  }
}

bool CLKernel::CreateFromProgram(const CLProgram &program, const std::string &function_name)
{
  int error_code;
  function_name_ = function_name;
  kernel_ = clCreateKernel(program.program(), function_name.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS)
  {
    kernel_ = nullptr;
    return false;
  }

  program_ = program.program();
  clRetainProgram(program_);
  if (GetKernelPrivateMemorySize(kernel_, program.GetDeviceId(), &info_.private_memory_size))
  {
    return false;
  }
  if (GetKernelMaxWorkGroupSize(kernel_, program.GetDeviceId(), &info_.max_work_group_size))
  {
    return false;
  }
  return true;
}

bool CLKernel::SetMemory(int index, cl_mem memory)
{
  return SetBytes(index, &memory, sizeof(cl_mem));
}

bool CLKernel::SetMemoryAuto(cl_mem memory) { return SetBytesAuto(&memory, sizeof(cl_mem)); }

bool CLKernel::SetBytes(int index, const void *ptr, int length) const
{
  const int error_code = clSetKernelArg(kernel_, index, length, ptr);
  if (error_code != CL_SUCCESS)
  {
    return false;
  }
  return true;
}

bool CLKernel::SetBytesAuto(const void *ptr, int length)
{
  const int error_code = clSetKernelArg(kernel_, binding_counter_, length, ptr);
  if (error_code != CL_SUCCESS)
  {
    return false;
  }
  binding_counter_++;
  return true;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

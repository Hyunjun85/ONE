/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_EINSUM_H__
#define __NNFW_CKER_EINSUM_H__

#include "cker/operation/EinsumHelper.h"

namespace nnfw
{
namespace cker
{

  // ToDo :: Add Input & Output
  // parse equation
  inline void einsum(const std::string &equation,
                      const Shape &input1_shape, const float *input1_data,
                      const Shape &input2_shape, const float *input2_data,
                      const Shape &output_shape, float *output_data)
  {
    std::vector<std::vector<int>> input_labels;
    std::vector<int> output_labels;

    std::vector<std::vector<int>> input_label_counts;
    std::vector<int> output_label_counts;
    std::vector<bool> input_has_ellipsis;
    std::vector<EinsumHelper::DimensionType> label_types;
    bool output_has_ellipsis = false;

    std::vector<int64_t> label_to_dim_sizes;

    // Parse Equation.
    EinsumHelper::parseEquation(equation, input_labels, output_labels, label_types,
                                input_label_counts, output_label_counts,
                                input_has_ellipsis, output_has_ellipsis);

    // Make Input Shape List to ProcessDimensions.
    std::vector<Shape> inputs;
    if (input1_data != NULL)
    {
      inputs.push_back(input1_shape);
    }
    if (input2_data != NULL)
    {
      inputs.push_back(input2_shape);
    }

    EinsumHelper::ProcessDimensions(inputs, input_has_ellipsis, output_has_ellipsis,
                                    input_labels, output_labels, label_types, input_label_counts,
                                    output_label_counts, label_to_dim_sizes);

    // The reduction phase (a) sums across reduction dimensions, (b) takes
    // generalized diagonals, and (c) reshapes it into shape
    //   [(broadcasting) batch shape] + [F,C]
    // where F and C denote the total (compacted) size of free and contract
    // dimensions, respectively.
    const int num_inputs = inputs.size();
    std::vector<std::vector<int>> free_labels(num_inputs);
    std::vector<Shape> inputs_reduced(num_inputs);
    std::vector<bool> swap_free_and_contract(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      EinsumHelper::ReduceOperand<Device, T>(
                         inputs[i], label_types, input_label_counts[i],
                         &input_labels[i], &free_labels[i],
                         &swap_free_and_contract[i], &inputs_reduced[i]));
    }
  }

} // namespace cker
} // namespace nnfw



#endif
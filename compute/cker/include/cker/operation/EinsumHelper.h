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

#ifndef __NNFW_CKER_EINSUM_HELPER_H__
#define __NNFW_CKER_EINSUM_HELPER_H__

#include <iostream>
#include <cassert>
#include <numeric>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "cker/Shape.h"
#include "cker/Types.h"

const int kEllipsisLabel = -1;
struct EinsumHelper
{
  enum DimensionType
  {
    // Batch dimensions are those present in two inputs as well as the output.
    // They are part of the batch dimensions during Tensor contraction.
    // Such dimensions may be broadcasting dimensions (those mapping to
    // ellipsis)
    // or explicit batch dimensions corresponding to named axis labels.
    kBroadcasting = 0,
    kBatch = 1,
    // Free dimensions are present in exactly one of the inputs, and also the
    // output. These are non-contracted axes in the Tensor contraction.
    kFree = 2,
    // Contract dimensions are present in two inputs, but not the output. These
    // dimensions are contracted in Tensor contraction.
    kContract = 3,
    // Reduce dimensions are present in exactly one input; and not in the output
    // and are summed over prior to Tensor contraction.
    kReduce = 4,
  };

  static DimensionType GetDimensionType(bool is_removed, bool is_unique)
  {
    if (!is_removed && !is_unique)
      return kBatch;
    else if (!is_removed && is_unique)
      return kFree;
    else if (is_removed && !is_unique)
      return kContract;
    else // is_removed && is_unique
      return kReduce;
  }

  static void mapToLabels(const std::string &subscript, std::vector<int> &labels,
                          std::unordered_map<char, int> &label_map)
  {
    for (int i = 0; i < subscript.size(); ++i)
    {
      const char label_char = subscript[i];
      if (label_char == '.') // means ellipsis
      {
        labels.push_back(kEllipsisLabel);
        i += 2;
        continue;
      }
      if (label_map.find(label_char) == label_map.end())
      {
        const int next_label = label_map.size();
        label_map[label_char] = next_label;
      }

      const int mapped_label = label_map[label_char];
      labels.push_back(mapped_label);
    }
  }

  static void parseEinsumEquation(const std::string &equation,
                                  std::vector<std::string> &input_subscript,
                                  std::string &output_subscript)
  {
    int arrow_pos = 0;
    if ((arrow_pos = equation.find("->")) == std::string::npos)
    {
      // error
      assert(0);
    }

    int comma_pos = 0;
    if ((comma_pos = equation.find(",")) == std::string::npos)
    {
      input_subscript.push_back(std::move(equation.substr(0, arrow_pos)));
    }
    else
    {
      input_subscript.push_back(std::move(equation.substr(0, comma_pos)));
      input_subscript.push_back(std::move(equation.substr(comma_pos + 1, arrow_pos)));
    }
    output_subscript = equation.substr(arrow_pos + 2);
  }

  // static void parseEquation(const string& equation,
  //                             OperandLabels* input_labels,
  //                             Labels* output_labels,
  //                             std::vector<DimensionType>* label_types,
  //                             OperandLabelCounts* input_label_counts,
  //                             LabelCounts* output_label_counts,
  //                             gtl::InlinedVector<bool, 2>* input_has_ellipsis,
  //                             bool* output_has_ellipsis)
  static void parseEquation(const std::string &equation,
                            std::vector<std::vector<int>> &input_labels,
                            std::vector<int> &output_labels,
                            std::vector<DimensionType> &label_types,
                            std::vector<std::vector<int>> &input_label_counts,
                            std::vector<int> &output_label_counts,
                            std::vector<bool> &input_has_ellipsis,
                            bool &output_has_ellipsis)
  {
    std::vector<std::string> input_subscription;
    std::string output_subscript;

    EinsumHelper::parseEinsumEquation(equation, input_subscription, output_subscript);

    std::unordered_map<char, int> label_map;
    int num_inputs = input_subscription.size();
    input_labels.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i)
    {
#ifdef _DEBUG
      std::cout << "input subscript : " << input_subscription[i] << "" << input_subscription[i].size() << std::endl;
#endif
      EinsumHelper::mapToLabels(input_subscription[i], input_labels.at(i), label_map);
    }
#ifdef _DEBUG
    std::cout << "output subscript : " << output_subscript << std::endl;
#endif
    EinsumHelper::mapToLabels(output_subscript, output_labels, label_map);

#ifdef _DEBUG
    std::cout << "mapped" << std::endl;
    for (auto it = label_map.begin(); it != label_map.end(); ++it)
    {
      std::cout << it->first << " " << it->second << std::endl;
    }
#endif
    // Compute counts for input aond output labels.
    int num_labels = label_map.size();

    input_label_counts.resize(num_inputs);
    input_has_ellipsis.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i)
    {
      input_label_counts.at(i).resize(num_labels);
      for (const auto label : input_labels.at(i))
      {
        if (label != kEllipsisLabel)
          input_label_counts.at(i)[label] += 1;
        else
          input_has_ellipsis.at(i) = true;
      }
    }
    output_label_counts.resize(num_labels);
    for (const int &label : output_labels)
    {
      if (label != kEllipsisLabel)
        output_label_counts.at(label) += 1;
      else
        output_has_ellipsis = true;
    }
    // // Map each label to a unique DimensionType.
    label_types.resize(num_labels);
    for (int label = 0; label < num_labels; ++label)
    {
      if (label == kEllipsisLabel)
        continue;
      bool removed = output_label_counts[label] == 0;
      bool unique = num_inputs == 1 || input_label_counts[0][label] == 0 ||
                    input_label_counts[1][label] == 0;
      label_types[label] = GetDimensionType(removed, unique);
#ifdef _DEBUG
      std::cout << "label_types[" << label << "]" << label_types[label] << std::endl;
#endif
    }
  }

  static void InsertBroadcastLabels(int num_bcast_dims, int num_named_labels,
                                    int ellipsis_axis, std::vector<int>& labels,
                                    std::vector<int>& label_counts)
  {
    labels.erase(labels.begin() + ellipsis_axis);
    labels.insert(labels.begin() + ellipsis_axis, num_bcast_dims, 0);
    std::iota(labels.begin() + ellipsis_axis,
              labels.begin() + ellipsis_axis + num_bcast_dims,
              num_named_labels);
    // Increment label counts. Since these are new labels, the count is set
    // to 1.
    label_counts->resize(num_named_labels + num_bcast_dims, 1);
  }

  static void RecordLabelToDimension(const int label, const int axis,
                                       const Shape& input,
                                       std::vector<int64_t>& label_to_dim_sizes)
  {
    const int64 input_dim = input.Dims(axis);
    // We know that label_to_dim_sizes has the size to accommodate named labels.
    if (label_to_dim_sizes.at(label) != 0 &&
        label_to_dim_sizes.at(label) != input_dim) {
      // return errors::InvalidArgument(
      //     "Expected dimension ", label_to_dim_sizes.at(label), " at axis ",
      //     axis, " of the input shaped ", input.shape().DebugString(),
      //     " but got dimension ", input_dim);
    }
    label_to_dim_sizes[label] = input_dim;
  }

  static void ProcessDimensions(std::vector<nnfw::cker::Shape>& inputs, const std::vector<bool>& input_has_ellipsis,
                                  const bool& output_has_ellipsis, std::vector<std::vector<int>>& input_labels,
                                  std::vector<int>& output_labels, std::vector<DimensionType>& label_types,
                                  std::vector<std::vector<int>>& input_label_counts, std::vector<int>& output_label_counts,
                                  std::vector<int64_t>& label_to_dim_sizes)
  {
    if (inputs.size() != input_lebels.size())
    {
      // error()
    }
    const int num_inputs = inputs.size();

    // We infer the number of broadcasting dimensions by taking the maximum rank
    // among the broadcasting subshapes of the input.
    int max_bcast_dims = 0;
    const int num_named_labels = label_types.size();
    label_to_dim_sizes.resize(num_named_labels);

    for (int i = 0; i < num_inputs; ++i)
    {
      std::vector<int>& labels = input_labels[i];

      if (!input_has_ellipsis[i])
      {
        if (inputs[i].DimensionsCount() != labels.size())
        {
          // return errors::InvalidArgument("Expected input ", i, " to have rank ",
          //                                labels->size(),
          //                                " but got: ", inputs[i].DimensionsCount());
        }
        for (int label_idx = 0; label_idx < labels.size(); ++label_idx)
        {
          const int label = labels[label_idx];
          RecordLabelToDimension(label, label_idx, inputs[i],
                                                    label_to_dim_sizes);
        }
        continue;
      }

      // Input has an ellipsis.
      if (inputs[i].DimensionsCount() + 1 < labels.size()) {
        // return errors::InvalidArgument(
        //     "Expected input ", i, " to have rank at least ", labels->size() - 1,
        //     " but got: ", inputs[i].DimensionsCount());
      }
      int ellipsis_axis = -1;
      const int num_bcast_dims = inputs[i].DimensionsCount() - labels.size() + 1;
      for (int label_idx = 0; label_idx < labels.size(); ++label_idx) {
        const int label = labels[label_idx];
        if (label == kEllipsisLabel) {
          ellipsis_axis = label_idx;
          continue;
        }
        // Current label is not an ellipsis.
        const int axis =
            label_idx + (ellipsis_axis == -1 ? 0 : num_bcast_dims - 1);
        RecordLabelToDimension(label, axis, inputs[i], label_to_dim_sizes);
      }
      // Found an ellipsis. Replace 'kEllipsisLabel' with broadcasting
      // dimensions.
      if (ellipsis_axis != -1) {
        // InsertBroadcastLabels(num_bcast_dims, num_named_labels, ellipsis_axis,
        //                       labels, &input_label_counts->at(i));
        max_bcast_dims = std::max(max_bcast_dims, num_bcast_dims);
      }
    }
    if ((std::find(input_has_ellipsis.begin(), input_has_ellipsis.end(), true) == input_has_ellipsis.end()) &&
        !output_has_ellipsis)
    {
      // return Status::OK();
      return;
    }

    auto it = std::find(output_labels.begin(), output_labels.end(), kEllipsisLabel);
    if (it != output_labels.end()) {
      const int ellipsis_axis = it - output_labels.begin();
      InsertBroadcastLabels(max_bcast_dims, num_named_labels, ellipsis_axis,
                            output_labels, output_label_counts);
    }
    else if (max_bcast_dims > 0)
    {
      return; // with ERROR.
      // return errors::InvalidArgument(
      //     "Output contains ", max_bcast_dims,
      //     " broadcasting dimension(s) but no ellipsis "
      //     "(...) was found in the output subscripts.");
    }
    // Populate DimensionType for the new broadcasting labels.
    label_types.resize(num_named_labels + max_bcast_dims, kBroadcasting);
    // return Status::OK();
  }

  template <typename Device, typename T>
  static Status ReduceOperand(OpKernelContext* ctx, const Tensor& input,
                              const std::vector<DimensionType>& label_types,
                              const LabelCounts& label_counts, Labels* labels,
                              Labels* free_labels, bool* swap_free_and_contract,
                              Tensor* output) {
    // Find the permutation to transpose the input dimensions in the order of
    // DimensionType; i.e. batch, free, contract and reduce dimensions. This
    // makes it more convenient to invoke Reduce/Contract operations.
    std::vector<int> permutation(input.dims());
    absl::c_iota(permutation, 0);
    Tensor input_transposed;
    // Check if we can avoid the transpose. We need to flip the adj_x (or adj_y)
    // flag during BatchMatMul. This is an extra optimization not necessary for
    // correctness.
    if (ShouldSwapFreeAndContract(*labels, label_types)) {
      *swap_free_and_contract = true;
    } else {
      absl::c_sort(permutation, [&](int i, int j) {
        int label_i = (*labels)[i];
        int label_j = (*labels)[j];
        return std::tie(label_types[label_i], label_i) <
               std::tie(label_types[label_j], label_j);
      });
    }
    // Transpose the input so that DimensionTypes are in order.
    TF_RETURN_IF_ERROR(TransposeOperand<Device, T>(ctx, input, permutation,
                                                   &input_transposed));
    PermuteLabels(permutation, labels);

    // Take the generalized diagonal for dimensions with repeated axis labels.
    Tensor input_deduped;
    labels->erase(std::unique(labels->begin(), labels->end()), labels->end());
    TF_RETURN_IF_ERROR(
        StrideOrInflate<Device, T>(ctx, input_transposed, *labels, label_counts,
                                   false /* should_inflate */, &input_deduped));

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each DimensionType.
    gtl::InlinedVector<int64, 5> reshape(5, 1);
    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.
    TensorShape output_shape;
    for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
      const int label = labels->at(label_idx);
      int64 dim = input_deduped.dim_size(label_idx);
      if (label_types[label] == kBroadcasting || label_types[label] == kBatch) {
        output_shape.AddDim(dim);
      } else if (label_types[label] == kFree) {
        free_labels->push_back(label);
      }
      reshape[label_types[label]] *= dim;
    }
    if (*swap_free_and_contract) std::swap(reshape[kFree], reshape[kContract]);
    output_shape.AddDim(reshape[kFree]);
    output_shape.AddDim(reshape[kContract]);

    if (reshape[kReduce] == 1) {  // No need to actually reduce.
      return CopyFrom(input_deduped, output_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    using Reducer = Eigen::internal::SumReducer<T>;
    using Index = typename TTypes<T>::Tensor::Index;
    // Reduce along the last axis (i.e axis 1) of the rank-2 Tensor.
    const int64 output_size = reshape[kBroadcasting] * reshape[kBatch] *
                              reshape[kFree] * reshape[kContract];
    functor::ReduceFunctor<Device, Reducer>::Reduce(
        ctx, output->shaped<T, 1>({output_size}),
        const_cast<const Tensor&>(input_deduped)
            .shaped<T, 2>({output_size, reshape[kReduce]}),
        Eigen::array<Index, 1>({1}), Reducer());
    return Status::OK();
  }
};

#endif


/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "moco/Import/Nodes/Add.h"

#include <moco/IR/Nodes/TFAdd.h>

#include <loco.h>

#include <memory>

namespace
{

using namespace moco;

/**
 * @brief GraphUpdate for TF Add node
 */
class TFAddGraphUpdate final : public GraphUpdate
{
public:
  TFAddGraphUpdate(TFAdd *node, std::vector<TensorName> names) : _node(node), _names(names) {}

  void input(const SymbolTable *) const override;

private:
  TFAdd *_node;
  std::vector<TensorName> _names;
};

void TFAddGraphUpdate::input(const SymbolTable *tensor_names) const
{
  assert(_names.size() == 2);

  _node->x(tensor_names->node(_names[0]));
  _node->y(tensor_names->node(_names[1]));
}

} // namespace

namespace moco
{

bool AddGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  return node.input_size() == 2;
}

void AddGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();
  UpdateQueue *updates = context->updates();

  // creating TF dialect Add node
  auto tf_add = graph->nodes()->create<TFAdd>();
  tf_add->name(node.name());

  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, tf_add);

  std::vector<TensorName> add_input_names;
  add_input_names.push_back(TensorName(node.input(0))); // x
  add_input_names.push_back(TensorName(node.input(1))); // y

  auto tf_add_update = std::make_unique<TFAddGraphUpdate>(tf_add, add_input_names);
  updates->enroll(std::move(tf_add_update));
}

} // namespace moco

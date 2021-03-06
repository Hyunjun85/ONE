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

#ifndef NNCC_FUSE_ARITHMETIC_OPS_H
#define NNCC_FUSE_ARITHMETIC_OPS_H

#include "pass/Pass.h"
#include "pass/PassData.h"

namespace nnc
{

/**
 * @brief Main purpose of this pass - is to fuse 'Conv->BatchNorm' into 'Conv'
 * Currently 'BatchNorm' split by NNC frontends into 'Scale->Scale->BiasAdd'
 * This optimization performs in two steps (repeated while graph changing):
 *   1. Fuse two successive operations with constant weights into one (ex: 'Scale->Scale' becomes
 * 'Scale')
 *   2. Sink 'BiasAdd' through 'Scale' (so 'Conv->BiasAdd->Scale' becomes 'Conv->Scale->BiasAdd')
 */
class FuseArithmeticOps : public Pass
{
public:
  PassData run(PassData data) override;

  std::string getName() override { return "FuseArithmeticOps"; }
};

} // namespace nnc

#endif // NNCC_FUSE_ARITHMETIC_OPS_H

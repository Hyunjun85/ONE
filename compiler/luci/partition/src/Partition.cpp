/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PartitionIR.h"
#include "PartitionIRDump.h"
#include "PartitionPGroups.h"
#include "PartitionMerge.h"
#include "PartitionCleanup.h"
#include "PartitionPModules.h"
#include "PartitionPModulesDump.h"

#include "luci/Partition.h"
#include "luci/Log.h"

#include <cassert>

namespace luci
{

/**
 * @brief This will return Partitioned Modules object
 */
PartedModules apply(Module *source, const PartitionTable &partition)
{
  assert(source != nullptr);

  LOGGER(l);

  auto pgroups = produce_pgroups(source, partition);
  INFO(l) << "--- Partition Graph (1)------------------------";
  INFO(l) << pgroups.get();

  auto mpgroups = merge_pgroups(pgroups.get());
  INFO(l) << "--- Partition Graph (2)------------------------";
  INFO(l) << mpgroups.get();

  remove_unused_inputoutputs(mpgroups.get(), source);
  INFO(l) << "--- Partition Graph (3)------------------------";
  INFO(l) << mpgroups.get();

  auto pmodules = produce_pmodules(mpgroups.get());
  INFO(l) << "--- Modules -----------------------------------";
  INFO(l) << &pmodules;

  return pmodules;
}

} // namespace luci

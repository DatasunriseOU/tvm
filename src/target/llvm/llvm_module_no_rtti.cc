/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file llvm_module_no_rtti.cc
 * \brief Isolates construction of an LLVM type whose base has no RTTI.
 */
#ifdef TVM_LLVM_VERSION

#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <utility>

namespace tvm {
namespace codegen {

std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler> CreateTMOwningSimpleCompilerWithoutRtti(
    std::unique_ptr<llvm::TargetMachine> target_machine) {
  return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(target_machine));
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION

<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> Open Machine Learning Compiler Framework
==============================================
[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

Apache TVM is an open machine learning compilation framework,
following the following principles:

- Python-first development that enables quick customization of machine learning compiler pipelines.
- Universal deployment to bring models into minimum deployable modules.

CPPMEGA Local Patches (branch `tilelang-apache-tvm-migration`)
--------------------------------------------------------------

This fork at `apstenku123/tvm` carries narrow local patches that the
TileLang fork at `DatasunriseOU/tilelang` (branch `metal-gemm-upstream-rebase`)
depends on. All patches are marked `// CPPMEGA:` in the source so they can be
greppen, audited, and removed once upstream covers the same surface.

Current patch set:

1. **Restored `Analyzer::EnterConstraint` shim**
   (`include/tvm/arith/analyzer.h`, `src/arith/analyzer.cc`).  Apache TVM
   replaced the public `Analyzer::EnterConstraint(PrimExpr) -> std::function<void()>`
   API with the RAII `With<ConstraintContext>` form. TileLang's
   `src/transform/common/constr_visitor.h::Populate` still relies on the
   function-cleanup form, so we expose a thin shim that aggregates the
   five sub-analyzer `EnterConstraint` calls and returns a single recovery
   closure executed in reverse order.

2. **Bind / EnterConstraint hook table on `Analyzer`** (commit
   `601f8fec50`).  External sub-analyzers (the TileLang vendored
   `Z3Prover` for partial-sync proofs) need to observe every
   `Analyzer::Bind(var, expr)`, `Analyzer::Bind(var, range)`, and every
   constraint pushed by `ConstraintContext::EnterWithScope`.  In the
   classic TileLang+TVM fork, `Analyzer` carried a `z3_prover` member and
   forwarded these calls inline.  Apache TVM does not, and we are not
   allowed to add a hard dependency from `libtvm_compiler` to
   `libtilelang`.  The patch introduces a callback hook table:

       using BindExprHook = void(*)(Analyzer*, const Var&, const PrimExpr&, bool);
       using BindRangeHook = void(*)(Analyzer*, const Var&, const Range&, bool);
       using EnterConstraintHook = std::function<void()>(*)(Analyzer*, const PrimExpr&);
       static void RegisterBindExprHook(BindExprHook);
       static void RegisterBindRangeHook(BindRangeHook);
       static void RegisterEnterConstraintHook(EnterConstraintHook);

   `Analyzer::Bind(var, expr, ...)` and `Analyzer::Bind(var, range, ...)`
   call their hook (if registered) at the end of the body.
   `ConstraintContext::EnterWithScope` and the public
   `Analyzer::EnterConstraint` shim push the hook's recovery closure
   onto their `recovery_functions_` stack so the constraint is unwound
   in the right order.

   Default hook value is `nullptr`, so a build that does not link
   libtilelang behaves identically to upstream Apache TVM (the hook
   call sites collapse to a single null-pointer test under `-O2`).

   `libtilelang.dylib` registers the three hooks at static init via the
   `Z3HookRegistrar` in `src/transform/vendored/z3_prover.cc`.  Each hook
   forwards to the per-Analyzer cached `tilelang::tlz3::Z3Prover`
   instance, restoring the constraint-aware partial-sync semantics from
   stack-c / tl_pr_c.

License
-------
TVM is licensed under the [Apache-2.0](LICENSE) license.

Getting Started
---------------
Check out the [TVM Documentation](https://tvm.apache.org/docs/) site for installation instructions, tutorials, examples, and more.
The [Getting Started with TVM](https://tvm.apache.org/docs/get_started/overview.html) tutorial is a great
place to start.

Contribute to TVM
-----------------
TVM adopts the Apache committer model. We aim to create an open-source project maintained and owned by the community.
Check out the [Contributor Guide](https://tvm.apache.org/docs/contribute/).

History and Acknowledgement
---------------------------
TVM started as a research project for deep learning compilation.
The first version of the project benefited a lot from the following projects:

- [Halide](https://github.com/halide/Halide): Part of TVM's TIR and arithmetic simplification module
 originates from Halide. We also learned and adapted some parts of the lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.

Since then, the project has gone through several rounds of redesigns.
The current design is also drastically different from the initial design, following the
development trend of the ML compiler community.

The most recent version focuses on a cross-level design with TensorIR as the tensor-level representation
and Relax as the graph-level representation and Python-first transformations.
The project's current design goal is to make the ML compiler accessible by enabling most
transformations to be customizable in Python and bringing a cross-level representation that can jointly
optimize computational graphs, tensor programs, and libraries. The project is also a foundation
infra for building Python-first vertical compilers for domains, such as LLMs.

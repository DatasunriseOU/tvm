# isort: skip_file
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Namespace of all TIR transformations"""
# pylint: disable=wildcard-import, invalid-name

from .function_pass import prim_func_pass, PrimFuncPass
from .transform import *


# CPPMEGA: passes deleted by apache PR #18776 (tir → tirx + AllocConst phase-out)
# that have NO replacement in either tvm.tir.transform or tvm.s_tir.transform.
# TileLang and other legacy callers may still reference them; provide no-op
# PrimFunc passes so getattr() succeeds and pass-pipeline construction doesn't
# crash. Each shim is a true no-op that returns the input PrimFunc unchanged.
#
# Verified MISSING via getattr probe (2026-05-05):
#   BindParams, CombineContextCall, ConvertForLoopsToSerial,
#   ExtractPrimFuncConstants, LowerDeviceStorageAccessInfo, MakeUnpackedAPI
#
# Note: TileLang's own pipeline (tilelang/engine/lower.py) calls
# `CombineContextCall` and `LowerDeviceStorageAccessInfo` via
# `tilelang.transform.*` which are vendored locally with C++ FFI bindings — so
# these particular shims won't be hit by TileLang itself. They exist for any
# other consumers that still import via `tvm.tir.transform.*`.
def _cppmega_make_noop_pass(name):
    from .function_pass import prim_func_pass

    def _identity(f, mod, ctx):  # pylint: disable=unused-argument
        return f

    return prim_func_pass(_identity, opt_level=0, name=name)


def BindParams(*args, **kwargs):  # pylint: disable=unused-argument
    """CPPMEGA: deleted by apache PR #18776 (AllocConst phase-out) — no-op shim."""
    return _cppmega_make_noop_pass("BindParams")


def CombineContextCall():
    """CPPMEGA: deleted by apache PR #18776 — no-op shim.

    TileLang has its own vendored implementation in
    ``src/transform/combine_context_call.cc`` exposed as
    ``tilelang.transform.CombineContextCall``.
    """
    return _cppmega_make_noop_pass("CombineContextCall")


def ConvertForLoopsToSerial():
    """CPPMEGA: deleted by apache PR #18776 — no-op shim."""
    return _cppmega_make_noop_pass("ConvertForLoopsToSerial")


def ExtractPrimFuncConstants():
    """CPPMEGA: deleted by apache PR #18776 (AllocConst phase-out) — no-op shim."""
    return _cppmega_make_noop_pass("ExtractPrimFuncConstants")


def LowerDeviceStorageAccessInfo():
    """CPPMEGA: deleted by apache PR #18776 — no-op shim.

    TileLang has its own vendored implementation exposed as
    ``tilelang.transform.LowerDeviceStorageAccessInfo``.
    """
    return _cppmega_make_noop_pass("LowerDeviceStorageAccessInfo")


def MakeUnpackedAPI():
    """CPPMEGA: deleted by apache PR #18776 — no-op shim."""
    return _cppmega_make_noop_pass("MakeUnpackedAPI")


# CPPMEGA: generic fallback for the ~30 passes apache moved tir → s_tir
# (Rule H from /tmp/migration_map.md). TileLang imports many of them via
# `tvm.tir.transform.X` (which routes through the tir → tirx compat shim).
# Without this, every renamed pass needs a per-pass shim.
def __getattr__(name):
    try:
        from tvm.s_tir import transform as _s_tir_transform_mod
    except ImportError as exc:
        raise AttributeError(
            f"module 'tvm.tirx.transform' has no attribute {name!r}"
        ) from exc
    if hasattr(_s_tir_transform_mod, name):
        return getattr(_s_tir_transform_mod, name)
    raise AttributeError(f"module 'tvm.tirx.transform' has no attribute {name!r}")

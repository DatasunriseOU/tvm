# CPPMEGA: compat shim — apache/tvm latest renamed tvm.tir -> tvm.tirx.
# This module re-exports tvm.tirx so TileLang's `from tvm.tir import X` keeps working.
# The C++ side already aliases `tvm::tir = tvm::tirx` via src/transform/vendored/tl_compat.h.
"""CPPMEGA compat shim: re-export tvm.tirx as tvm.tir for backward compatibility."""
# pylint: disable=wildcard-import, unused-wildcard-import, unused-import
from tvm.tirx import *  # noqa: F401, F403
from tvm.tirx import _ffi_api  # noqa: F401

# Register `tir` as a script-dialect alias to `tvm.tirx.script` so that
# legacy paths like `tvm.script.parser.tir` resolve through the redirect
# finder (which appends `.parser`/`.builder`/etc.).  This keeps deep
# statement-form imports such as
#     `from tvm.script.parser.tir import parser as p`
# working without modifying TileLang.
def _cppmega_register_tir_dialect():
    import importlib
    import tvm.script as _ts
    _ts.register_dialect("tir", "tvm.tirx.script")

    # Patch _DialectRedirectFinder.find_spec to be tolerant of attribute-style
    # imports like `from tvm.script.ir_builder.tir import evaluate`, where
    # `evaluate` is a function (not a submodule).  Upstream raises
    # ModuleNotFoundError; we return None so Python falls back to attribute
    # lookup on the already-imported parent module.
    finder_cls = _ts._DialectRedirectFinder
    if not getattr(finder_cls, "_cppmega_patched", False):
        _orig_find_spec = finder_cls.find_spec.__func__

        def _patched_find_spec(cls, fullname, path, target=None):
            try:
                return _orig_find_spec(cls, fullname, path, target)
            except ModuleNotFoundError:
                return None

        finder_cls.find_spec = classmethod(_patched_find_spec)
        finder_cls._cppmega_patched = True

    # Eagerly resolve `tvm.script.ir_builder.tir` and `tvm.script.parser.tir`
    # (and selected sub-submodules) and ALIAS them in sys.modules to the
    # canonical tirx module object.  Without this, the dialect-redirect finder
    # ends up creating *distinct* module objects loaded from the same file
    # (one under each import path), so attributes set on one don't appear on
    # the other.
    import sys as _sys
    alias_pairs = [
        ("tvm.script.ir_builder.tir", "tvm.tirx.script.builder"),
        ("tvm.script.ir_builder.tir.frame", "tvm.tirx.script.builder.frame"),
        ("tvm.script.ir_builder.tir.ir", "tvm.tirx.script.builder.ir"),
        ("tvm.script.ir_builder.tir.utils", "tvm.tirx.script.builder.utils"),
        ("tvm.script.parser.tir", "tvm.tirx.script.parser"),
        ("tvm.script.parser.tir.parser", "tvm.tirx.script.parser.parser"),
        ("tvm.script.parser.tir.entry", "tvm.tirx.script.parser.entry"),
        ("tvm.script.parser.tir.operation", "tvm.tirx.script.parser.operation"),
        ("tvm.script.tir", "tvm.tirx.script"),
    ]
    for legacy_name, tirx_name in alias_pairs:
        try:
            real = importlib.import_module(tirx_name)
            _sys.modules[legacy_name] = real
        except Exception:  # pylint: disable=broad-except
            pass


try:
    _cppmega_register_tir_dialect()
except Exception:  # pylint: disable=broad-except
    pass
finally:
    del _cppmega_register_tir_dialect


# Inject legacy aliases for renamed classes/functions that TileLang still references.
# Each entry maps a (module_name, legacy_attr) to a list of candidate new names
# to try.  The first matching attr is aliased onto the module.
def _cppmega_inject_legacy_aliases():
    import sys as _sys
    rename_map = {
        # (module_name): [(legacy, [candidates])]
        "tvm.tirx.script.builder.frame": [("BlockFrame", ["SBlockFrame"])],
        "tvm.tirx.script.builder": [
            ("block_attr", ["sblock_attr"]),
            ("BlockFrame", ["SBlockFrame"]),
        ],
        "tvm.tirx.script.parser": [
            ("block_attr", ["sblock_attr"]),
        ],
    }
    for modname, items in rename_map.items():
        m = _sys.modules.get(modname)
        if m is None:
            continue
        for legacy, candidates in items:
            if hasattr(m, legacy):
                continue
            for cand in candidates:
                if hasattr(m, cand):
                    setattr(m, legacy, getattr(m, cand))
                    break

    # Inject an `allocate` compat shim into tvm.tirx.script.parser/builder.
    # Apache TVM removed the statement-level `allocate(extents, dtype, scope)`
    # builder; the closest replacement is `alloc_buffer(shape, dtype, scope)`,
    # which returns a Buffer instead of an AllocateFrame.  TileLang only uses
    # T.allocate(...) inside test/utility prim funcs (e.g., pass_config.py);
    # forwarding to alloc_buffer keeps imports + most code paths working.
    def _make_allocate_shim(builder_mod):
        if not hasattr(builder_mod, "alloc_buffer"):
            return None
        _alloc_buffer = builder_mod.alloc_buffer

        def allocate(extents, dtype, scope="global", condition=True, annotations=None):
            """CPPMEGA shim: forwards T.allocate(...) to T.alloc_buffer(...).

            Note: the legacy `allocate` returned an AllocateFrame; this shim
            returns a Buffer (matching the new alloc_buffer semantics).
            """
            # Drop unsupported kwargs silently (legacy API had condition).
            return _alloc_buffer(extents, dtype, scope, annotations)

        return allocate

    for modname in ("tvm.tirx.script.parser", "tvm.tirx.script.builder"):
        m = _sys.modules.get(modname)
        if m is None or hasattr(m, "allocate"):
            continue
        shim = _make_allocate_shim(m)
        if shim is not None:
            setattr(m, "allocate", shim)

    # CPPMEGA: install a `LetStmt` compatibility shim.
    #
    # Legacy apache/tvm exposed ``T.LetStmt(value)`` as a frame-style builder:
    # caller did ``frame = T.LetStmt(value); var = frame.var; with frame: ...``.
    # Apache's tirx replaced this with ``bind(value)`` which directly emits a
    # Bind statement and returns the freshly-bound Var (no frame, no context
    # manager).  TileLang's eager builder
    # (``tilelang/language/eager/builder.py::Builder.bind_immutable``) still
    # uses the legacy frame-with-``.var`` shape, so we fabricate a no-op
    # context manager whose ``.var`` is the bound variable.  ``bind()`` already
    # emits the statement on construction; the ``__enter__``/``__exit__`` are
    # purely to satisfy ``Builder.enter_frame``'s context-manager contract.
    def _make_let_stmt_shim(builder_mod):
        if not hasattr(builder_mod, "bind"):
            return None
        _bind = builder_mod.bind

        class _LetStmtFrame:
            __slots__ = ("var",)

            def __init__(self, value, type_annotation=None, var=None):
                self.var = _bind(value, type_annotation, var=var)

            def __enter__(self):
                return self.var

            def __exit__(self, exc_type, exc, tb):
                return False

        def LetStmt(value, type_annotation=None, *, var=None):
            """CPPMEGA shim: wrap ``bind(value)`` in a no-op frame.

            Restores the legacy ``T.LetStmt(value)`` shape (object exposes
            ``.var`` and is usable as a context manager) on top of apache's
            new flat ``bind`` builder.
            """
            return _LetStmtFrame(value, type_annotation, var)

        return LetStmt

    for modname in ("tvm.tirx.script.parser", "tvm.tirx.script.builder"):
        m = _sys.modules.get(modname)
        if m is None or hasattr(m, "LetStmt"):
            continue
        shim = _make_let_stmt_shim(m)
        if shim is not None:
            setattr(m, "LetStmt", shim)


_cppmega_inject_legacy_aliases()
del _cppmega_inject_legacy_aliases


# Patch tvm_ffi type-key resolution so legacy type-keys like
# "script.ir_builder.tir.LetFrame" first try the renamed tirx equivalent
# ("script.ir_builder.tirx.LetFrame").  TileLang still registers Python
# wrappers under the old key strings; the C++ side registers them under the
# new ones.  When neither exists (e.g., LetFrame was removed entirely), we
# silently skip the registration via the existing _SKIP_UNKNOWN_OBJECTS flag.
#
# CPPMEGA: extended to also handle BARE top-level legacy keys such as
# ``tir.Block`` (no leading dot), which apache renamed to ``tirx.SBlock`` etc.
# The previous version only rewrote dotted middle-of-string occurrences
# (``.tir.``), so bare keys produced "Cannot find type tir.Block" failures
# during IR construction / pickling.
def _cppmega_patch_ffi_typekey():
    try:
        import tvm_ffi.registry as _reg
        from tvm_ffi import core as _core
        if getattr(_reg, "_cppmega_typekey_patched", False):
            return
        _orig_resolve = _core._object_type_key_to_index

        # CPPMEGA: bare-key rename map.  Most apache renames are simple
        # ``tir.X`` -> ``tirx.X``, but a handful changed name as well.
        # See docs/migration_map.md.
        _BARE_TIR_RENAMES = {
            "tir.Block": "tirx.SBlock",
            "tir.BlockRealize": "tirx.SBlockRealize",
            "tir.LetStmt": "tirx.Bind",
            # ``tir.Allocate`` was removed in favour of ``tirx.AllocBuffer``
            # (which returns a Buffer rather than an AllocateFrame).  The
            # closest type-key match is AllocBuffer; if a caller actually
            # needs the legacy Allocate semantics it can also fall back to
            # the TileLang-vendored ``tilelang.Allocate`` below.
            "tir.Allocate": "tirx.AllocBuffer",
        }

        # CPPMEGA: TileLang vendored some statement nodes back under
        # ``tilelang.*`` (see src/transform/vendored/*.cc).  These are only
        # registered once the tilelang C++ library is loaded, so we look
        # them up dynamically rather than caching the index.
        _TILELANG_VENDORED = {
            "tir.LetStmt": "tilelang.LetStmt",
            "tir.Allocate": "tilelang.Allocate",
        }

        def _patched_resolve(key):
            idx = _orig_resolve(key)
            if idx is not None:
                return idx
            # Try ".tir." -> ".tirx." rewrite (dotted middle-of-string).
            if ".tir." in key:
                idx2 = _orig_resolve(key.replace(".tir.", ".tirx.", 1))
                if idx2 is not None:
                    return idx2
            # CPPMEGA: bare top-level ``tir.X`` keys.
            if isinstance(key, str) and key.startswith("tir."):
                # 1) TileLang vendored fallback (preferred for LetStmt/Allocate
                #    so we keep legacy frame-style semantics where possible).
                vendored = _TILELANG_VENDORED.get(key)
                if vendored is not None:
                    idx3 = _orig_resolve(vendored)
                    if idx3 is not None:
                        return idx3
                # 2) Explicit rename map (Block -> SBlock, BlockRealize ->
                #    SBlockRealize, LetStmt -> Bind, Allocate -> AllocBuffer).
                renamed = _BARE_TIR_RENAMES.get(key)
                if renamed is not None:
                    idx4 = _orig_resolve(renamed)
                    if idx4 is not None:
                        return idx4
                # 3) Generic ``tir.X`` -> ``tirx.X`` rewrite for everything
                #    else (For, Buffer, BufferLoad, IfThenElse, Var, ...).
                idx5 = _orig_resolve("tirx." + key[4:])
                if idx5 is not None:
                    return idx5
            return None

        _core._object_type_key_to_index = _patched_resolve
        _reg._SKIP_UNKNOWN_OBJECTS = True
        _reg._cppmega_typekey_patched = True
    except Exception:  # pylint: disable=broad-except
        pass


_cppmega_patch_ffi_typekey()
del _cppmega_patch_ffi_typekey


# Patch Op.get to fall back from "tir.X" to "tirx.X" since apache renamed the op
# namespace.  TileLang code (and several internal TVM utilities) still call
# Op.get("tir.if_then_else"), Op.get("tir.address_of"), etc.
def _cppmega_patch_op_get():
    try:
        from tvm.ir.op import Op
        if getattr(Op, "_cppmega_get_patched", False):
            return
        _orig_get = Op.get  # staticmethod -> raw fn already unbound

        def _patched_get(op_name):
            try:
                return _orig_get(op_name)
            except Exception:  # pylint: disable=broad-except
                if isinstance(op_name, str) and op_name.startswith("tir."):
                    return _orig_get("tirx." + op_name[4:])
                raise

        Op.get = staticmethod(_patched_get)
        Op._cppmega_get_patched = True
    except Exception:  # pylint: disable=broad-except
        pass


_cppmega_patch_op_get()
del _cppmega_patch_op_get


# Patch tvm_ffi._dunder._make_init_signature to deduplicate parameter names.
# Apache's stricter FFI registers ancestor fields when a child re-registers them
# (e.g., tl.GemmSPWarpPolicy re-registers policy_type/m_warp/n_warp from
# tl.GemmWarpPolicy), which produces duplicate inspect.Parameter entries and
# crashes inspect.Signature.  We dedupe by keeping the first occurrence.
def _cppmega_patch_dunder():
    try:
        from tvm_ffi import _dunder as _d
        if getattr(_d, "_cppmega_dedupe_patched", False):
            return
        _orig = _d._make_init_signature

        def _patched(type_info):
            try:
                return _orig(type_info)
            except ValueError as e:
                if "duplicate parameter name" not in str(e):
                    raise
                # Rebuild with dedup
                import inspect as _ins
                # Replicate logic from _orig but dedupe.
                # We'll call _orig manually inlined: easier to just patch
                # inspect.Signature to dedupe.
                raise

        # Simpler: wrap inspect.Signature called within _make_init_signature
        # by patching inside the module.  Instead, replace the function with
        # a dedupe-aware reimpl that mirrors the original.
        from tvm_ffi import core as _core
        import inspect as _ins

        def _new_make_init_signature(type_info):
            all_fields = []
            ti = type_info
            chain = []
            while ti is not None:
                chain.append(ti)
                ti = ti.parent_type_info
            for ancestor_info in reversed(chain):
                all_fields.extend(ancestor_info.fields)
            seen = set()
            positional = []
            kw_only = []
            for field in all_fields:
                if not field.c_init:
                    continue
                if field.name in seen:
                    continue
                seen.add(field.name)
                if field.c_kw_only:
                    kw_only.append((field.name, field.c_has_default))
                else:
                    positional.append((field.name, field.c_has_default))
            pos_required = [(n, d) for n, d in positional if not d]
            pos_default = [(n, d) for n, d in positional if d]
            kw_required = [(n, d) for n, d in kw_only if not d]
            kw_default = [(n, d) for n, d in kw_only if d]
            params = [_ins.Parameter("self", _ins.Parameter.POSITIONAL_OR_KEYWORD)]
            for name, _ in pos_required:
                params.append(_ins.Parameter(name, _ins.Parameter.POSITIONAL_OR_KEYWORD))
            for name, _ in pos_default:
                params.append(_ins.Parameter(name, _ins.Parameter.POSITIONAL_OR_KEYWORD,
                                              default=_core.MISSING))
            for name, _ in kw_required:
                params.append(_ins.Parameter(name, _ins.Parameter.KEYWORD_ONLY))
            for name, _ in kw_default:
                params.append(_ins.Parameter(name, _ins.Parameter.KEYWORD_ONLY,
                                              default=_core.MISSING))
            return _ins.Signature(params)

        _d._make_init_signature = _new_make_init_signature
        _d._cppmega_dedupe_patched = True
    except Exception:  # pylint: disable=broad-except
        pass


_cppmega_patch_dunder()
del _cppmega_patch_dunder

# Submodules referenced as tvm.tir.<sub> — also alias them in sys.modules so
# that `from tvm.tir.stmt import X` works (Python's import machinery checks
# sys.modules['tvm.tir.stmt'] before falling back to attribute lookup).
def _cppmega_alias_submodules():
    import sys as _sys
    import importlib as _il
    sub_map = {
        "tvm.tir.expr": "tvm.tirx.expr",
        "tvm.tir.stmt": "tvm.tirx.stmt",
        "tvm.tir.function": "tvm.tirx.function",
        "tvm.tir.op": "tvm.tirx.op",
        "tvm.tir.buffer": "tvm.tirx.buffer",
        "tvm.tir.generic": "tvm.tirx.generic",
        "tvm.tir.transform": "tvm.tirx.transform",
        "tvm.tir.analysis": "tvm.tirx.analysis",
        "tvm.tir.functor": "tvm.tirx.functor",
        "tvm.tir.stmt_functor": "tvm.tirx.stmt_functor",
    }
    for legacy, target in sub_map.items():
        try:
            mod = _il.import_module(target)
            _sys.modules.setdefault(legacy, mod)
            # also expose as attribute on `tvm.tir`
            globals()[legacy.rsplit(".", 1)[-1]] = mod
        except Exception:  # pylint: disable=broad-except
            pass


_cppmega_alias_submodules()
del _cppmega_alias_submodules

# transform / analysis exist as real subpackages under tvm/tir/ already; prefer tirx versions
try:
    from tvm.tirx import transform  # noqa: F401
except ImportError:
    pass
try:
    from tvm.tirx import analysis  # noqa: F401
except ImportError:
    pass

# CPPMEGA: ``tvm.tir.schedule`` and the ``Schedule`` / ``BlockRV`` classes were
# moved out of ``tvm.tir`` entirely — apache renamed the package to
# ``tvm.s_tir`` (statement-form TIR with the new schedule API) and renamed
# ``BlockRV`` to ``SBlockRV``.  TileLang's carver code still imports
# ``from tvm.tir.schedule.schedule import BlockRV`` and uses
# ``tir.Schedule`` / ``tir.schedule.BlockRV`` annotations, so we wire the
# legacy paths back through the new locations and inject a ``BlockRV`` alias.
def _cppmega_alias_schedule():
    import sys as _sys
    import importlib as _il
    try:
        s_sched = _il.import_module("tvm.s_tir.schedule")
        s_sched_inner = _il.import_module("tvm.s_tir.schedule.schedule")
    except Exception:  # pylint: disable=broad-except
        return
    # Inject ``BlockRV`` alias on the inner module so
    # ``from tvm.tir.schedule.schedule import BlockRV`` resolves.
    if not hasattr(s_sched_inner, "BlockRV") and hasattr(s_sched_inner, "SBlockRV"):
        s_sched_inner.BlockRV = s_sched_inner.SBlockRV
    if not hasattr(s_sched, "BlockRV") and hasattr(s_sched, "SBlockRV"):
        s_sched.BlockRV = s_sched.SBlockRV
    # Alias the legacy submodule paths in sys.modules.
    _sys.modules.setdefault("tvm.tir.schedule", s_sched)
    _sys.modules.setdefault("tvm.tir.schedule.schedule", s_sched_inner)
    # Expose at module level for ``tir.schedule`` / ``tir.Schedule`` /
    # ``tir.BlockRV`` attribute access.
    g = globals()
    g["schedule"] = s_sched
    if hasattr(s_sched, "Schedule"):
        g["Schedule"] = s_sched.Schedule
    if hasattr(s_sched_inner, "SBlockRV"):
        g["BlockRV"] = s_sched_inner.SBlockRV


_cppmega_alias_schedule()
del _cppmega_alias_schedule

# Hot symbols TileLang uses (explicit list for clarity / IDE resolution).
# Wrap each in try/except so a single missing symbol does not break the shim.
def _reexport():
    import tvm.tirx as _tirx
    g = globals()
    names = [
        "BufferLoad", "BufferStore", "PrimFunc", "Var", "SizeVar", "IterVar",
        "IntImm", "FloatImm", "StringImm", "Cast",
        "Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod",
        "Min", "Max", "EQ", "NE", "LT", "LE", "GT", "GE",
        "And", "Or", "Not", "Select", "Call", "Let",
        "Buffer", "BufferRegion", "For", "While", "IfThenElse",
        "Evaluate", "AttrStmt", "AssertStmt", "SeqStmt", "Stmt", "PrimExpr",
        "IndexMap", "TensorIntrin", "Reduce", "Broadcast", "Ramp", "Shuffle",
        "ProducerLoad", "CommReducer", "DeclBuffer", "AllocBuffer",
        "MatchBufferRegion", "decl_buffer", "DataProducer",
        "PyStmtExprMutator", "PyExprMutator", "PyExprVisitor", "PyStmtExprVisitor",
        "StmtFunctor", "ExprFunctor", "StmtVisitor", "ExprVisitor",
        "StmtMutator", "ExprMutator", "functor", "stmt_functor",
    ]
    for n in names:
        if hasattr(_tirx, n):
            g[n] = getattr(_tirx, n)

    # Legacy aliases: LetStmt was unified into Let; Allocate -> AllocBuffer.
    if "LetStmt" not in g and hasattr(_tirx, "Let"):
        g["LetStmt"] = _tirx.Let
    if "Allocate" not in g and hasattr(_tirx, "AllocBuffer"):
        g["Allocate"] = _tirx.AllocBuffer
    # CPPMEGA: Apache renamed tir.Block -> tirx.SBlock and
    # tir.BlockRealize -> tirx.SBlockRealize.  TileLang's metal_fragment_to_simdgroup
    # transform still uses these as Python class refs (isinstance / constructor),
    # so expose the renamed classes under the legacy names at the tvm.tir module
    # level.  Constructor signatures appear unchanged.
    if "Block" not in g and hasattr(_tirx, "SBlock"):
        g["Block"] = _tirx.SBlock
    if "BlockRealize" not in g and hasattr(_tirx, "SBlockRealize"):
        g["BlockRealize"] = _tirx.SBlockRealize

_reexport()
del _reexport


# CPPMEGA: TileLang passes ``DeclBuffer(buffer, body)`` (legacy 3-arg form,
# ``DeclBuffer(buffer, body, span)``) but apache/tvm dropped the ``body`` field
# from DeclBufferNode — the new signature is ``DeclBuffer(buffer, span=None)``.
# Body now lives in the surrounding SeqStmt (same architectural pattern as
# Bind / AllocBuffer).  This shim distinguishes a Span 2nd arg from a Stmt 2nd
# arg: when ``body`` is a Stmt we return ``SeqStmt([DeclBuffer(buffer), body])``
# instead, restoring the legacy semantics.  Only one TileLang call site is
# affected today (``tilelang/transform/decouple_type_cast.py``), but installing
# the shim at the tvm.tir layer protects any future call sites too.
def _cppmega_install_decl_buffer_shim():
    import tvm.tirx as _tirx_mod
    from tvm.ir import Span as _Span
    _orig_DeclBuffer = _tirx_mod.DeclBuffer
    if getattr(_orig_DeclBuffer, "_cppmega_shim", False):
        return

    class _DeclBufferShim:
        _cppmega_shim = True

        def __new__(cls, buffer, body=None, span=None):
            # 1-arg form: DeclBuffer(buffer)
            if body is None and span is None:
                return _orig_DeclBuffer(buffer)
            # 2-arg form where 2nd arg is actually a Span (legacy positional span)
            if span is None and (body is None or isinstance(body, _Span)):
                return _orig_DeclBuffer(buffer, body)
            # 3-arg form: DeclBuffer(buffer, body, span)
            if body is None:
                return _orig_DeclBuffer(buffer, span)
            # Body is a Stmt — emit SeqStmt([DeclBuffer(buffer, span), body])
            from tvm.tirx import SeqStmt as _SeqStmt
            decl = _orig_DeclBuffer(buffer, span)
            return _SeqStmt([decl, body])

    _tirx_mod.DeclBuffer = _DeclBufferShim
    globals()["DeclBuffer"] = _DeclBufferShim


_cppmega_install_decl_buffer_shim()
del _cppmega_install_decl_buffer_shim


# CPPMEGA: TileLang still emits the legacy 6-arg form
#   Allocate(buffer_var, dtype, extents, condition, body, annotations=None)
# but apache renamed Allocate -> AllocBuffer with a brand-new signature
#   AllocBuffer(buffer: Buffer, annotations=None, span=None)
# where ``buffer`` is a fully-formed Buffer object and ``body`` lives in the
# enclosing SeqStmt (same architectural pattern as Bind / DeclBuffer).
#
# This shim detects the legacy positional shape (>= 5 positional args, 3rd arg
# is a list/tuple of extents — i.e. shape) and:
#   1) constructs a Buffer via ``decl_buffer(shape, dtype, data=buffer_var)``
#      so the existing data Var is reused;
#   2) emits ``SeqStmt([AllocBuffer(buf, annotations), body])``.
# Condition is dropped: TileLang only ever passes ``tir.const(True)`` here
# (see ``tilelang/transform/decouple_type_cast.py::_wrap_with_allocations``);
# if a future call site needs a real condition, wrap the SeqStmt in
# IfThenElse externally.
def _cppmega_install_alloc_buffer_shim():
    import tvm.tirx as _tirx_mod
    _orig_AllocBuffer = _tirx_mod.AllocBuffer
    if getattr(_orig_AllocBuffer, "_cppmega_shim", False):
        return

    class _AllocBufferShim:
        _cppmega_shim = True

        def __new__(cls, *args, **kwargs):
            # Legacy 5/6-arg form:
            #   AllocBuffer(buffer_var, dtype, extents, condition, body[, annotations])
            if len(args) >= 5 and isinstance(args[1], str):
                buffer_var = args[0]
                dtype = args[1]
                extents = args[2]
                # condition = args[3]  # ignored — TileLang only passes const(True)
                body = args[4]
                annotations = args[5] if len(args) > 5 else kwargs.get("annotations")
                from tvm.tirx import decl_buffer as _decl_buffer
                from tvm.tirx import SeqStmt as _SeqStmt
                name = getattr(buffer_var, "name_hint", "buf") or "buf"
                buf = _decl_buffer(extents, dtype, name=name, data=buffer_var)
                alloc = _orig_AllocBuffer(buf, annotations)
                return _SeqStmt([alloc, body])
            # Apache form: AllocBuffer(buffer, annotations=None, span=None)
            return _orig_AllocBuffer(*args, **kwargs)

    _tirx_mod.AllocBuffer = _AllocBufferShim
    globals()["AllocBuffer"] = _AllocBufferShim
    # Allocate is the legacy alias TileLang imports — keep it pointing at the shim.
    globals()["Allocate"] = _AllocBufferShim


_cppmega_install_alloc_buffer_shim()
del _cppmega_install_alloc_buffer_shim


# CPPMEGA: apache/tvm split the legacy ``tir.LetStmt(var, value, body)`` (a
# statement-with-body) from the expression-level ``tir.Let(var, value, body)``
# (PrimExpr in / PrimExpr out).  ``tirx.LetStmt`` was renamed to ``tirx.Bind``
# (no body — body lives in the enclosing SeqStmt, same architectural pattern as
# AllocBuffer / DeclBuffer), and ``tirx.Let`` kept the expression-only meaning.
#
# TileLang code (e.g. ``tilelang/transform/hoist_broadcast_values.py``) still
# does ``LetStmt(var, value, stmt_body)`` — and the re-export above aliased
# ``LetStmt`` to ``tirx.Let``, so the body (a ``BufferStore``) is rejected by
# the FFI as not-a-PrimExpr.
#
# This shim wraps both ``tvm.tirx.Let`` and the legacy ``LetStmt`` alias so a
# Stmt body is rewritten to ``SeqStmt([Bind(var, value, span), body])``, while
# a PrimExpr body falls through to the real expression-Let constructor.
def _cppmega_install_let_stmt_shim():
    import tvm.tirx as _tirx_mod
    _orig_Let = _tirx_mod.Let
    if getattr(_orig_Let, "_cppmega_shim", False):
        return

    class _LetShim:
        _cppmega_shim = True

        def __new__(cls, var, value, body, span=None):
            # Stmt body -> emit SeqStmt([Bind(var, value, span), body])
            if isinstance(body, _tirx_mod.Stmt):
                from tvm.tirx import Bind as _Bind, SeqStmt as _SeqStmt
                return _SeqStmt([_Bind(var, value, span), body])
            # PrimExpr body -> real expression-Let
            return _orig_Let(var, value, body, span)

    _tirx_mod.Let = _LetShim
    globals()["Let"] = _LetShim
    globals()["LetStmt"] = _LetShim


_cppmega_install_let_stmt_shim()
del _cppmega_install_let_stmt_shim

# CPPMEGA: tvm.relay was removed entirely in apache/tvm latest (only relax remains).
# TileLang's parser.py:460 has a legacy isinstance(res, (tvm.relay.Call, tvm.relax.Call))
# check that's almost-dead but would AttributeError if reached. This shim provides a
# sentinel `Call` class that no real instance is ever an instance of.
class _NeverMatch:
    """Sentinel — isinstance(x, _NeverMatch) is always False."""
    pass

Call = _NeverMatch

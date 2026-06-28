"""Probe cute.struct API on the installed cutlass (version drift between e8ecfad
examples and the box's pinned wheel). Prints the supported struct field forms."""

import cutlass
import cutlass.cute as cute

print("VERSION:", cutlass.__version__)
print("STRUCT_MEMBERS:", [x for x in dir(cute.struct) if not x.startswith("_")])
for name in ("Array", "MemRange", "MemRangeArray", "Pointer"):
    print(f"  {name}:", getattr(cute.struct, name, "MISSING"))
# How does cutlass.pipeline expect barrier storage? introspect a couple of helpers.
import cutlass.pipeline as pipeline

print("PIPELINE:", [x for x in dir(pipeline) if not x.startswith("_")][:40])

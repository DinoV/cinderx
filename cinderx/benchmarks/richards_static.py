# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys

import cinderx.jit

try:
    from .richards_static_lib import Richards  # buck
except ImportError:
    # pyre-ignore[21]: Undefined import
    from richards_static_lib import Richards  # OSS standalone script


if __name__ == "__main__":
    cinderx.jit.auto()

    import time
    s = time.perf_counter()
    num_iterations = 100
    if len(sys.argv) > 1:
        num_iterations = int(sys.argv[1])
    Richards().run(num_iterations)
    e = time.perf_counter()
    print(e-s)

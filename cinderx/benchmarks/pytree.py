# pyre-strict

from __future__ import annotations

from cinderx.compiler.strict import loader as static_python_loader

static_python_loader.install()

import numpy

import os
import random
import time
from typing import Callable, Iterable

from jax.tree_util import (
    register_pytree_node as jax_register_pytree_node,
    tree_flatten as jax_tree_flatten,
    tree_leaves as jax_tree_leaves,
    tree_map as jax_tree_map,
    tree_unflatten as jax_tree_unflatten,
)
from torch.utils._cxx_pytree import (
    tree_flatten as cxx_pytree_tree_flatten,
    tree_leaves as cxx_pytree_tree_leaves,
    tree_map as cxx_pytree_tree_map,
    tree_unflatten as cxx_pytree_tree_unflatten,
)
from torch.utils._pytree import (
    Context,
    PyTree,
    register_pytree_node as pytree_register_pytree_node,
    tree_flatten as pytree_tree_flatten,
    tree_leaves as pytree_tree_leaves,
    tree_map as pytree_tree_map,
    tree_unflatten as pytree_tree_unflatten,
)


class Node:
    def __init__(self, children: list[PyTree]) -> None:
        if isinstance(children, tuple):
            raise Exception("tuple")
        self.children = children

    def __repr__(self) -> str:
        return f"Node({self.children})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.children == other.children

    def __hash__(self) -> int:
        return hash(self.children)

    @staticmethod
    def flatten(node: Node) -> tuple[list[PyTree], Context]:
        return node.children, None

    @staticmethod
    def jax_unflatten(context: Context, children: Iterable[PyTree]) -> Node:
        return Node(list(children))

    @staticmethod
    def pytree_unflatten(children: Iterable[PyTree], context: Context) -> Node:
        return Node(list(children))


def make_pytree(depth: int = 5, breadth: int = 4) -> PyTree:
    if depth == 0:
        return random.randint(0, 1000)
    else:
        node_type = random.choice(["list", "tuple", "dict", "node"])
        if node_type == "list":
            return [make_pytree(depth - 1, breadth) for _ in range(breadth)]
        elif node_type == "tuple":
            return tuple(make_pytree(depth - 1, breadth) for _ in range(breadth))
        elif node_type == "node":
            return Node([make_pytree(depth - 1, breadth) for _ in range(breadth)])
        else:
            return {f"k{i}": make_pytree(depth - 1, breadth) for i in range(breadth)}


def iterations(func: Callable[[], object], duration: int) -> float:
    count = 0
    start = time.perf_counter()
    end = start + duration

    while time.perf_counter() < end:
        func()
        count += 1

    elapsed = time.perf_counter() - start
    return count / elapsed


def benchmark(
    name: str,
    funcs: dict[str, Callable[[], object]],
    warmup: int = 2,
    duration: int = 4,
) -> None:
    print(f"{name}:")
    [iterations(func, warmup) for func in funcs.values()]

    results = [(kind, iterations(func, duration)) for kind, func in funcs.items()]
    results.sort(key=lambda tuple: tuple[1], reverse=True)

    maximum = results[0][1]
    for index, (kind, ips) in enumerate(results):
        pct = (maximum - ips) / maximum * 100
        print(f"    {kind:<20} {ips:.2f} ips", "" if index == 0 else f" (-{pct:.2f}%)")


def main() -> None:
    seed = os.environ.get("SEED", random.randint(100, 1000))
    random.seed(int(seed))

    print(f"pytree/benchmark (SEED={seed})")
    print("===========================")

    jax_register_pytree_node(Node, Node.flatten, Node.jax_unflatten)
    pytree_register_pytree_node(Node, Node.flatten, Node.pytree_unflatten)
    pytree = make_pytree(depth=2, breadth=3)

    jax_pytree_flat, jax_pytree_spec = jax_tree_flatten(pytree)
    pytree_flat, pytree_spec = pytree_tree_flatten(pytree)
    cxx_pytree_flat, cxx_pytree_spec = cxx_pytree_tree_flatten(pytree)

    assert jax_pytree_flat == pytree_flat
    assert pytree_flat == cxx_pytree_flat

    jax_pytree_unflat = jax_tree_unflatten(jax_pytree_spec, jax_pytree_flat)
    pytree_unflat = pytree_tree_unflatten(pytree_flat, pytree_spec)
    cxx_pytree_unflat = cxx_pytree_tree_unflatten(cxx_pytree_flat, cxx_pytree_spec)

    assert jax_pytree_unflat == pytree_unflat
    assert pytree_unflat == cxx_pytree_unflat

    def mapping(x: int) -> int:
        return x + 1

    jax_pytree_map = jax_tree_map(mapping, pytree)
    pytree_map = pytree_tree_map(mapping, pytree)
    cxx_pytree_map = cxx_pytree_tree_map(mapping, pytree)

    assert jax_pytree_map == pytree_map
    assert pytree_map == cxx_pytree_map

    jax_pytree_leaves = jax_tree_leaves(pytree)
    pytree_leaves = pytree_tree_leaves(pytree)
    cxx_pytree_leaves = cxx_pytree_tree_leaves(pytree)

    assert jax_pytree_leaves == pytree_leaves
    assert pytree_leaves == cxx_pytree_leaves

    benchmark(
        "tree_flatten",
        {
            "jax": lambda: jax_tree_flatten(pytree),
            "_pytree": lambda: pytree_tree_flatten(pytree),
            "_cxx_pytree": lambda: cxx_pytree_tree_flatten(pytree),
        },
    )

    benchmark(
        "tree_unflatten",
        {
            "jax": lambda: jax_tree_unflatten(jax_pytree_spec, jax_pytree_flat),
            "_pytree": lambda: pytree_tree_unflatten(pytree_flat, pytree_spec),
            "_cxx_pytree": lambda: cxx_pytree_tree_unflatten(
                cxx_pytree_flat, cxx_pytree_spec
            ),
        },
    )

    benchmark(
        "tree_map",
        {
            "jax": lambda: jax_tree_map(mapping, pytree),
            "_pytree": lambda: pytree_tree_map(mapping, pytree),
            "_cxx_pytree": lambda: cxx_pytree_tree_map(mapping, pytree),
        },
    )

    benchmark(
        "tree_leaves",
        {
            "jax": lambda: jax_tree_leaves(pytree),
            "_pytree": lambda: pytree_tree_leaves(pytree),
            "_cxx_pytree": lambda: cxx_pytree_tree_leaves(pytree),
        },
    )


if __name__ == "__main__":
    main()

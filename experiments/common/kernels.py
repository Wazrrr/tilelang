"""Build a kernel through its owning family."""

from experiments.families import family_module


def make_case(workload):
    return family_module(workload.op, "kernel").make_case(workload)

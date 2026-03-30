from ... import context
from ...utils import Instruction, PIPE, PipeInst


def barrier(pipe: PipeInst):
    active_mod = context.active_cube or context.active_vec
    assert active_mod is not None, 'Barrier should run in either cube forward or vec forward'
    active_mod.append(Instruction('BAR', pipe=pipe))


def bar_m():
    barrier(PIPE.M)

def bar_v():
    barrier(PIPE.V)

def bar_mte2():
    barrier(PIPE.MTE2)

def bar_mte1():
    barrier(PIPE.MTE1)

def bar_mte3():
    barrier(PIPE.MTE3)

def bar_fix():
    barrier(PIPE.FIX)

def bar_all():
    barrier(PIPE.ALL)


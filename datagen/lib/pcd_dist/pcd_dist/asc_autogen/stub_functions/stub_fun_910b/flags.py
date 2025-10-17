from ... import context
from ...utils import Instruction, PIPE, PipeInst, Var
from typing import Union


def setflag(src: PipeInst, dst: PipeInst, event_id: Union[int, Var]):
    active_mod = context.active_cube or context.active_vec
    assert active_mod is not None, 'setflag should run in either cube forward or vec forward'
    active_mod.append(Instruction('SETFLAG', src=src, dst=dst, idd=event_id))

def waitflag(src: PipeInst, dst: PipeInst, event_id: Union[int, Var]):
    active_mod = context.active_cube or context.active_vec
    assert active_mod is not None, 'waitflag should run in either cube forward or vec forward'
    active_mod.append(Instruction('WAITFLAG', src=src, dst=dst, idd=event_id))


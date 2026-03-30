from .... import context
from ....utils import Reg, Tensor, Var, MaskReg, Instruction, Position, sizeof, CastConfig
from typing import Union 


def cast(dst: Reg, src: Reg, config: CastConfig, mask: MaskReg):
    g_micro = context.active_micro
    assert g_micro is not None, 'OP must be called within forward function'
    assert isinstance(dst, Reg)
    assert isinstance(src, Reg)
    assert isinstance(config, CastConfig)
    assert isinstance(mask, MaskReg)
    assert config.name!='', 'cast config must be set in initialize function'
    dsrc = src.dtype
    ddst = dst.dtype
    
    g_micro.append(Instruction('CAST', dst=dst, src=src, config=config, mask=mask, ddst=ddst, dsrc=dsrc))

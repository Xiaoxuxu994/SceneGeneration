from .. import context

class RoundModeInst:
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        if context.device_type.startswith('910b') and self.mode=='UNKNOWN':
            return 'CAST_NONE'
        return self.mode
    
    def __eq__(self, other: 'RoundModeInst'):
        return self.mode==other.mode


class ROUNDMODE:
    NONE = RoundModeInst('UNKNOWN')
    TO_EVEN = RoundModeInst('CAST_RINT')              # R
    AWAY_FROM_ZERO = RoundModeInst('CAST_ROUND')      # A
    FLOOR = RoundModeInst('CAST_FLOOR')               # F
    CEIL = RoundModeInst('CAST_CEIL')                 # C
    TRUNC = RoundModeInst('CAST_TRUNC')               # Z


class RegLayoutInst:
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        return self.mode
    
    def __eq__(self, other: 'RoundModeInst'):
        return self.mode==other.mode


class RegLayout:
    ZERO = RegLayoutInst('ZERO')
    ONE = RegLayoutInst('ONE')
    TWO = RegLayoutInst('TWO')
    THREE = RegLayoutInst('THREE')


class CastConfig():
    def __init__(self, reg_layout: RegLayoutInst, round_mode: RoundModeInst, saturate: bool=False):
        self.reg_layout = reg_layout
        self.round_mode = round_mode
        self.saturate = saturate
        self.name = ''
    
    def __str__(self):
        return self.name

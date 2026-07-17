class LoadDistInst:
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        return self.mode 
    
    def __eq__(self, other: 'LoadDistInst'):
        return self.mode==other.mode


class LOADDIST:
    NORMAL = LoadDistInst('DIST_NORM')
    SINGLE_VALUE = LoadDistInst('SINGLE')
    UPSAMPLE = LoadDistInst('UPSAMPLE')
    DOWNSAMPLE = LoadDistInst('DOWNSAMPLE')
    BRCB = LoadDistInst('BRCB')
    UNPACK = LoadDistInst('UNPACK')
    UNPACK4 = LoadDistInst('UNPACK4')


class StoreDistInst:
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        return self.mode 
    
    def __eq__(self, other: 'StoreDistInst'):
        return self.mode==other.mode


class STOREDIST:
    NORMAL = StoreDistInst('DIST_NORM')
    SINGLE_VALUE = StoreDistInst('SINGLE')
    DOWNSAMPLE = StoreDistInst('DOWNSAMPLE')

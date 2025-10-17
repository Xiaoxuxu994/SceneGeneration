class DualModeInst:
    def __init__(self, mode: int):
        self.mode = mode 
    
    def __str__(self):
        return str(self.mode)


class DUALMODE:
    SINGLE = DualModeInst(0)
    SPLITM = DualModeInst(1)
    SPLITN = DualModeInst(2)


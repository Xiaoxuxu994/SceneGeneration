class PipeInst:
    def __init__(self, s: str):
        self.s = s 
    
    def __str__(self):
        return self.s 
    
    def __repr__(self):
        return self.s 
    
    def __eq__(self, other: 'PipeInst'):
        if self.s==other.s:
            return True 
        else:
            return False 



class PIPE:
    M = PipeInst('M')
    V = PipeInst('V')
    MTE1 = PipeInst('MTE1')
    MTE2 = PipeInst('MTE2')
    MTE3 = PipeInst('MTE3')
    FIX = PipeInst('FIX')
    S = PipeInst('S')
    ALL = PipeInst('ALL')



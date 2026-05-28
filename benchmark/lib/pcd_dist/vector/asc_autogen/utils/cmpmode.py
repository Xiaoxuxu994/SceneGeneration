class CompareMode:
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        return self.mode 
    

class CMPMODE:
    LT = CompareMode('LT')
    GT = CompareMode('GT')
    GE = CompareMode('GE')
    LE = CompareMode('LE')
    EQ = CompareMode('EQ')
    NE = CompareMode('NE')


class SelectMode:
    def __init__(self, mode: str):
        self.mode = mode 
    
    def __str__(self):
        return self.mode 


class SELMODE: 
    CMPMASK = SelectMode('CMPMASK_SPR')
    TENSOR_SCALAR = SelectMode('TENSOR_SCALAR_MODE')
    TENSOR_TENSOR = SelectMode('TENSOR_TENSOR_MODE')


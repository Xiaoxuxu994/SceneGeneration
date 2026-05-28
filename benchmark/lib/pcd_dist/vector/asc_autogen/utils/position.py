POS_TO_ASC_MAPPING: dict[str, str] = {
    'UB'  : 'VECCALC',
    'L1'  : 'A1',
    'L0A' : 'A2',
    'L0B' : 'B2',
    'L0C' : 'CO1',
}

class Pos():
    def __init__(self, pos: str):
        self.posstr = pos 

    def asc_pos(self):
        return POS_TO_ASC_MAPPING[self.posstr]
    
    def __eq__(self, other: 'Pos'):
        if self.posstr==other.posstr:
            return True 
        return False 
    
    def __str__(self):
        return self.posstr


class Position:
    UB = Pos('UB')
    L1 = Pos('L1')
    L0A = Pos('L0A')
    L0B = Pos('L0B')
    L0C = Pos('L0C')

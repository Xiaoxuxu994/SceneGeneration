class CodeHelper():
    def __init__(self):
        self._indent: int = 0
        self.result = ''

    def ir(self): # indent right 
        self._indent += 4
    
    def il(self): # indent left 
        self._indent -= 4 
        if self._indent<0:
            raise ValueError('Indent cannot be less than 0')
    
    def __call__(self, v: str=''):
        self.result += ' '*self._indent + v + '\n'
    
    def __str__(self):
        return self.result
    

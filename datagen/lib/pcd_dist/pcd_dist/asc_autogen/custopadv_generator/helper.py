class CodeHelper():
    def __init__(self):
        self.code = ''
        self.indent = 0 
    
    def ir(self):
        self.indent+=4
    
    def il(self):
        self.indent-=4
    
    def __call__(self, code: str=''):
        self.code += ' '*self.indent + code + '\n'
from typing import Any 


class Instruction:
    def __init__(self, inst: str, **kwargs: Any):
        self._inst = inst 
        self._kwargs = kwargs
    
    def __getattribute__(self, name: str) -> Any:
        if name in ['_inst', '_kwargs']:
            return super().__getattribute__(name)
        if name in self._kwargs:
            return self._kwargs[name]
        raise Exception(f'{name} not in instruction params')

    def __str__(self):
        return '[Instruction] ' + self._inst



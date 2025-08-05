'''
Implementation of a conditional associative array (dict) using the strategy
pattern.
'''
from typing import Any, Callable, Dict


class cdict(dict):
    '''
    A dictionary that conditionally accepts attributes and/or values.

    This implementation uses the strategy pattern such that a function is
    provided on initialization for validating items that are set. If an
    attribute and/or value is not accepted during an add operation, the
    set operation will fail and the key/value will be added to the "rejected"
    property.
    '''
    
    def __init__(
            self,
            accept_fn:Callable[[Dict, Any, Any], bool],
            *args, **kwargs
    ):
        '''
        :param accept_fn: A fn(d, key, value) that returns True to accept
            the key/value into this dict d, or False to reject.
        '''
        super().__init__()
        self._rejected = dict()
        self.accept_fn = accept_fn
        #super().__init__(*args, **kwargs)
        self.update(*args, **kwargs)

    @property
    def rejected(self) -> Dict:
        return self._rejected

    def __setitem__(self, key, value) -> bool:
        if self.accept_fn(self, key, value):
            super().__setitem__(key, value)
        else:
            self._rejected[key] = value

    def setdefault(self, key, default=None):
        rv = None
        if key not in self:
            if self.accept_fn(self, key, default):
                super().__setitem__(key, default)
                rv = default
            else:
                self._rejected[key] = default
        else:
            rv = self[key]
        return rv

    def update(self, E=None, **F):
        if E is not None:
            if 'keys' in dir(E):
                for k in E:
                    self.__setitem__(k, E[k])
            else:
                for k, v in E:
                    self.__setitem__(k, v)
        if F is not None:
            for k in F:
                self.__setitem__(k, F[k])

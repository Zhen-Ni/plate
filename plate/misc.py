#!/usr/bin/env python3


class ClassConstant:
    def __call__(self, func):
        cache_name = '_' + func.__name__
        def new_func(self):
            if hasattr(self, cache_name):
                return getattr(self, cache_name)
            value = func(self)
            setattr(self, cache_name, value)
            return value
        return new_func


def round_array(array, atol=1e-9, rtol=1e-15):
    threshold = max(abs(array).max() * rtol, atol)
    array[abs(array) < threshold] = 0

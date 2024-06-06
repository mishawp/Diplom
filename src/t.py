import inspect


class foo:
    def __init__(self, c, a=1, b=None):
        pass


d = {"a": 1, "b": 2}
d2 = {k: v.default for k, v in (inspect.signature(foo)).parameters.items()}
print(f"parameters: {d2}")




class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

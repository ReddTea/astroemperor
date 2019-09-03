"""also.py."""

# TODO: reformat into a more useful name?


class Accumulator(object):
    """Accumulates conditions."""

    none = None

    def also(self, condition):
        """Accumulates conditions."""
        self.none = not condition and (self.none is None or self.none)
        return condition

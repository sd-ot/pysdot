#
class RadialFuncEntropy:
    def __init__(self, eps):
        self.s = "exp((w-r**2)/{})".format(eps)

    def name(self):
        return self.s

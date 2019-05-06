#
class RadialFuncR2:
    def __init__(self, eps):
        self.s = "r**2"

    def name(self):
        return self.s

    def need_rb_corr(self):
        return False
        
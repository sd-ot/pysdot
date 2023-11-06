#
class RadialFuncEntropy:
    def __init__(self, eps):
        self.s = "exp((w-r**2)/{})".format(eps)

    def name(self):
        return self.s

    def need_rb_corr(self):
        return False
        
    def ball_cut( self ):
        return False

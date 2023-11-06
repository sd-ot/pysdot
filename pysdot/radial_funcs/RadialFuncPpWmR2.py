#
class RadialFuncPpWmR2:
    def name(self):
        return "pos_part(w-r**2)"

    def need_rb_corr(self):
        return False

    def second_order_moment_name(self):
        return "r^2*pos_part(w-r**2)"

    def ball_cut( self ):
        return True

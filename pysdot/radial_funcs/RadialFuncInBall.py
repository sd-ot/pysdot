#
class RadialFuncInBall:
    def name(self):
        return "in_ball(weight**0.5)"

    def need_rb_corr(self):
        return False

    def second_order_moment_name(self):
        return "r^2*in_ball(weight**0.5)"

    def ball_cut( self ):
        return True

import pybind_sdot_Arfd

#
class RadialFuncArfd:
    def __init__(self, values, inp_scaling, out_scaling, der_values, der_inp_scaling, der_out_scaling, stops, prec=1e-7):
        self.func = pybind_sdot_Arfd.Arfd(values, inp_scaling, out_scaling, der_values, der_inp_scaling, der_out_scaling, stops, prec)

    def name(self):
        return self.func

    def need_rb_corr(self):
        return False

    def second_order_moment_name(self):
        return ""

    def nb_polynomials( self ):
        return self.func.nb_polynomials()

    def ball_cut( self ):
        return False
        
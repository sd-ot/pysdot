from .radial_funcs import RadialFuncUnit
from .cpp import cpp_module
import numpy as np
import os


class PowerDiagram:
    def __init__(self, domain=None, radial_func=RadialFuncUnit()):
        self.radial_func = radial_func
        self.domain = domain

        self.positions = None
        self.weights = None

        self._positions_are_new = True
        self._weights_are_new = True
        self._domain_is_new = True
        self._inst = None

    def set_positions(self, positions):
        self._positions_are_new = True
        self.positions = positions

    def set_weights(self, weights):
        self._weights_are_new = True
        self.weights = weights

    def set_domain(self, domain):
        self._domain_is_new = True
        self.domain = domain

    def integrals(self):
        inst = self._updated_grid()
        return inst.integrals(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def centroids(self):
        inst = self._updated_grid()
        return inst.centroids(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def der_integrals_wrt_weights(self):
        inst = self._updated_grid()
        return inst.der_integrals_wrt_weights(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def der_centroids_and_integrals_wrt_weight_and_positions(self):
        inst = self._updated_grid()
        return inst.der_centroids_and_integrals_wrt_weight_and_positions(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def display_vtk(self, filename, points=False, centroids=False):
        dn = os.path.dirname(filename)
        if len(dn):
            os.makedirs(dn, exist_ok=True)
        inst = self._updated_grid()
        return inst.display_vtk(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name(),
            filename,
            points,
            centroids
        )

    def display_vtk_points(self, filename, points=None):
        if points is None:
            return self.display_vtk_points(filename, self.positions)
        dn = os.path.dirname(filename)
        if len(dn):
            os.makedirs(dn, exist_ok=True)
        inst = self._updated_grid()
        return inst.display_vtk_points(
            self.positions,
            filename
        )

    # make a .asy file for a representation of the power diagram
    def display_asy(self, filename, preamble="", closing="", output_format="pdf", linewidth=0.02, dotwidth=0.0, values=np.array([]), colormap="inferno", avoid_bounds=False, min_rf=1, max_rf=0):
        dn = os.path.dirname( filename )
        if len( dn ):
            os.makedirs( dn, exist_ok = True )

        p = "settings.outformat = \"{}\";\nunitsize(1cm);\n".format( output_format )
        if linewidth > 0:
            p += "defaultpen({}cm);\n".format( linewidth )
        elif dotwidth > 0:
            p += "defaultpen({}cm);\n".format( dotwidth / 6 )
        p += preamble

        inst = self._updated_grid()
        inst.display_asy(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name(),
            filename,
            p,
            values,
            colormap,
            linewidth,
            dotwidth,
            avoid_bounds,
            closing,
            min_rf,
            max_rf
        )

    #
    def display_jupyter( self ):
        import IPython
        from urllib.parse import quote
        return IPython.display.IFrame( 'data:text/html,' + quote( self.display_html() ), width="95%", height=500)
        
    # return a string
    def display_html( self ):
        inst = self._updated_grid()
        path = inst.display_html_canvas(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

        beg = """
            <!doctype html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Hello World</title>
                <style>* {padding: 0; margin: 0}</style>    
                <script type="text/javascript">
                    let path = new Path2D();
        """

        end = """
                    var cr = 0.52 * Math.max( max_x - min_x, max_y - min_y );
                    var cx = 0.5 * ( max_x + min_x );
                    var cy = 0.5 * ( max_y + min_y );
                    var orig_click_x = 0;
                    var orig_click_y = 0;
                    var pos_click_x = 0;
                    var pos_click_y = 0;

                    function draw() {
                        var canvas = document.getElementById( 'my_canvas' );
                        var w = canvas.width, h = canvas.height;
                        var m = 0.5 * Math.min( w, h );
                        var s = m / cr;

                        var ctx = canvas.getContext( '2d' );
                        ctx.setTransform( 1, 0, 0, 1, 0, 0 );
                        ctx.clearRect( 0, 0, w, h );

                        ctx.translate( 0.5 * w, 0.5 * h );
                        ctx.scale( s, s );
                        ctx.translate( - cx, - cy );

                        ctx.lineWidth = 1.0 / s;
                        ctx.stroke( path );
                    }

                    function resize() {
                        var canvas = document.getElementById( 'my_canvas' );
                        canvas.width  = window.innerWidth;
                        canvas.height = window.innerHeight;
                        draw();
                    }

                    function init() {
                        window.addEventListener( "resize", function( event ) { 
                            resize();
                        });

                        window.addEventListener( "wheel", function( e ) {  
                            if ( e.shiftKey ) {
                                var canvas = document.getElementById( 'my_canvas' );
                                var w = canvas.width, h = canvas.height;
                                var m = 0.5 * Math.min( w, h );
                                var s = m / cr;

                                var d = Math.pow( 2, ( e.deltaY || - e.wheelDeltaY ) / 200.0 );
                                cx -= ( e.x - 0.5 * w ) * ( d - 1 ) / s;
                                cy -= ( e.y - 0.5 * h ) * ( d - 1 ) / s;
                                cr *= d;

                                draw();
                                return false;
                            }
                        }, false );

                        window.addEventListener( "mousedown", function( e ) {  
                            orig_click_x = e.x;
                            orig_click_y = e.y;
                            pos_click_x = e.x;
                            pos_click_y = e.y;
                        } );

                        window.addEventListener( "mousemove", function( e ) {  
                            if ( e.buttons == 1 || e.buttons == 4 ) {
                                var canvas = document.getElementById( 'my_canvas' );
                                var w = canvas.width, h = canvas.height;
                                var m = 0.5 * Math.min( w, h );
                                var s = m / cr;

                                cx -= ( e.x - pos_click_x ) / s;
                                cy -= ( e.y - pos_click_y ) / s;
                                pos_click_x = e.x;
                                pos_click_y = e.y;

                                draw();
                            }
                            // if ( e.buttons == 4 ) {
                            //     var canvas = document.getElementById( 'my_canvas' );
                            //     var w = canvas.width, h = canvas.height;
                            //     var m = 0.5 * Math.min( w, h );
                            //     var s = m / cr;

                            //     var d = Math.pow( 2, ( pos_click_y - e.y ) / 200.0 );
                            //     pos_click_x = e.x;
                            //     pos_click_y = e.y;

                            //     cx -= ( orig_click_x - 0.5 * w ) * ( d - 1 ) / s;
                            //     cy -= ( orig_click_y - 0.5 * h ) * ( d - 1 ) / s;
                            //     cr *= d;

                            //     draw();
                            // }
                        } );

                        resize();
                    }
                </script>
            </head>
            <body onload="init();">
            <canvas id="my_canvas" style="position: absolute; display: block"></canvas>
            </body>
            </html>
        """

        return beg + path + end

    def _updated_grid(self):
        # check types
        if not isinstance(self.positions, np.ndarray):
            self.positions = np.array(self.positions)
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.array(self.weights)

        # instantiation of PowerDiagram
        if not self._inst:
            assert(self.positions.dtype == self.domain._type)
            assert(self.weights.dtype == self.domain._type)
            module = cpp_module.module_for_type_and_dim(
                self.domain._type, self.positions.shape[1]
            )
            self._inst = module.PowerDiagramZGrid(11)

        self._inst.update(
            self.positions,
            self.weights,
            self._positions_are_new or self._domain_is_new,
            self._weights_are_new or self._domain_is_new,
            self.radial_func.name()
        )
        self._positions_are_new = False
        self._weights_are_new = False
        self._domain_is_new = False

        return self._inst
        
    # def boundary_integral( self ):
    #     inst = self._updated_grid()
    #     return inst.boundary_integral(
    #         self.positions,
    #         self.weights,
    #         self.domain._inst,
    #         self.radial_func.name()
    #     )

    # def der_boundary_integral( self ):
    #     inst = self._updated_grid()
    #     return inst.der_boundary_integral(
    #         self.positions,
    #         self.weights,
    #         self.domain._inst,
    #         self.radial_func.name()
    #     )

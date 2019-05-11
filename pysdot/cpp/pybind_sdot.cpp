// #include "../../ext/sdot/src/sdot/Visitors/internal/ZCoords.cpp"
#include "../../ext/sdot/src/sdot/Support/ThreadPool.cpp"
#include "../../ext/sdot/src/sdot/Support/CbQueue.cpp"
#include "../../ext/sdot/src/sdot/Support/Assert.cpp"
#include "../../ext/sdot/src/sdot/Support/Mpi.cpp"

#include "../../ext/sdot/src/sdot/Domains/ConvexPolyhedronAssembly.h"
#include "../../ext/sdot/src/sdot/Domains/ScaledImage.h"

#include "../../ext/sdot/src/sdot/PowerDiagram/Visitors/SpZGrid.h"

#include "../../ext/sdot/src/sdot/PowerDiagram/get_der_centroids_and_integrals_wrt_weight_and_positions.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_der_integrals_wrt_weights.h"
// #include "../../ext/sdot/src/sdot/PowerDiagram/get_der_boundary_integral.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_boundary_integral.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_integrals.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_centroids.h"

#include "../../ext/sdot/src/sdot/Display/VtkOutput.h"
#include "inferno_color_map.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>

namespace {
    constexpr int dim = PD_DIM;
    using TF = PD_TYPE;

    template<class FU>
    void find_radial_func( const std::string &func, const FU &fu ) {
        if ( func == "1" || func == "unit" ) {
            fu( sdot::FunctionEnum::Unit() );
            return;
        }

        if ( func.size() > 13 && func.substr( 0, 13 ) == "exp((w-r**2)/" ) {
            PD_TYPE eps;
            std::istringstream is( func.substr( 13, func.size() - 14 ) );
            is >> eps;
            fu( sdot::FunctionEnum::ExpWmR2db<PD_TYPE>{ eps } );
            return;
        }

        if ( func == "r**2" || func == "r^2" ) {
            fu( sdot::FunctionEnum::R2() );
            return;
        }

        if ( func == "in_ball(weight**0.5)" ) {
            fu( sdot::FunctionEnum::InBallW05() );
            return;
        }

        throw pybind11::value_error( "unknown function type" );
    }

    template<class Domain,class Grid>
    pybind11::array_t<TF> get_integrals( pybind11::array_t<TF> &positions, pybind11::array_t<TF> &weights, Domain &domain, Grid &grid, const std::string &func ) {
        auto buf_positions = positions.request();
        auto buf_weights = weights.request();

        auto ptr_positions = reinterpret_cast<const typename Grid::Pt *>( buf_positions.ptr );
        auto ptr_weights = reinterpret_cast<const typename Grid::TF *>( buf_weights.ptr );

        pybind11::array_t<TF> res;
        res.resize( { positions.shape( 0 ) } );
        auto buf_res = res.request();
        auto ptr_res = (TF *)buf_res.ptr;

        find_radial_func( func, [&]( auto ft ) {
            sdot::get_integrals( ptr_res, grid, domain, ptr_positions, ptr_weights, positions.shape( 0 ), ft );
        } );

        return res;
    }

    template<class Domain,class Grid>
    pybind11::array_t<TF> get_centroids( pybind11::array_t<TF> &positions, pybind11::array_t<TF> &weights, Domain &domain, Grid &grid, const std::string &func ) {
        auto buf_positions = positions.request();
        auto buf_weights = weights.request();

        auto ptr_positions = reinterpret_cast<const typename Grid::Pt *>( buf_positions.ptr );
        auto ptr_weights = reinterpret_cast<const typename Grid::TF *>( buf_weights.ptr );

        pybind11::array_t<TF> res;
        res.resize( { positions.shape( 0 ), pybind11::ssize_t( PD_DIM ) } );
        auto buf_res = res.request();
        auto ptr_res = (TF *)buf_res.ptr;

        find_radial_func( func, [&]( auto ft ) {
            sdot::get_centroids( grid, domain, ptr_positions, ptr_weights, positions.shape( 0 ), ft, [&]( auto centroid, auto, auto num ) {
                for( int d = 0; d < PD_DIM; ++d )
                    ptr_res[ PD_DIM * num + d ] = centroid[ d ];
            } );
        } );

        return res;
    }

    template<class T>
    void vcp( pybind11::array_t<T> &dst, const std::vector<T> &src ) {
        dst.resize( { src.size() } );
        auto buf = dst.request();
        auto ptr = reinterpret_cast<T *>( buf.ptr );
        for( std::size_t i = 0; i < src.size(); ++i )
            ptr[ i ] = src[ i ];
    }

    template<int dim,class TF>
    struct PyDerResult {
        pybind11::array_t<std::size_t> m_offsets;
        pybind11::array_t<std::size_t> m_columns;
        pybind11::array_t<TF>          m_values;
        pybind11::array_t<TF>          v_values;
        int                            error;
    };

    struct PyPc {
        static constexpr int allow_translations = 0;
        static constexpr int nb_bits_per_axis   = 31;
        static constexpr int allow_ball_cut     = 1;
        static constexpr int dim                = PD_DIM;
        using                TI                 = std::size_t;
        using                CI                 = std::size_t;
        using                TF                 = PD_TYPE;
    };

    template<int dim,class TF>
    struct PyConvexPolyhedraAssembly {
        using TB = sdot::ConvexPolyhedronAssembly<PyPc>;
        using Pt = TB::Pt;

        PyConvexPolyhedraAssembly() {
        }

        void add_box( pybind11::array_t<PD_TYPE> &min_pos, pybind11::array_t<PD_TYPE> &max_pos, PD_TYPE coeff, std::size_t cut_id ) {
            auto buf_min_pos = min_pos.request(); auto ptr_min_pos = (PD_TYPE *)buf_min_pos.ptr;
            auto buf_max_pos = max_pos.request(); auto ptr_max_pos = (PD_TYPE *)buf_max_pos.ptr;
            if ( min_pos.size() != PyPc::dim )
                throw pybind11::value_error( "wrong dimensions for min_pos" );
            if ( max_pos.size() != PyPc::dim )
                throw pybind11::value_error( "wrong dimensions for max_pos" );
            bounds.add_box( ptr_min_pos, ptr_max_pos, coeff, cut_id );
        }

        void add_convex_polyhedron( pybind11::array_t<PD_TYPE> &positions_and_normals, PD_TYPE coeff, std::size_t cut_id ) {
            auto buf_pan = positions_and_normals.request(); auto ptr_pan = (PD_TYPE *)buf_pan.ptr;
            if ( positions_and_normals.shape( 1 ) != 2 * PyPc::dim )
                throw pybind11::value_error( "wrong dimensions for positions_and_normals" );
            std::vector<Pt> positions, normals;
            for( pybind11::ssize_t i = 0; i < positions_and_normals.shape( 0 ); ++i ) {
                positions.push_back( ptr_pan + PyPc::dim * ( 2 * i + 0 ) );
                normals  .push_back( ptr_pan + PyPc::dim * ( 2 * i + 1 ) );
            }
            bounds.add_convex_polyhedron( positions, normals, coeff, cut_id );
        }

        void normalize() {
            bounds.normalize();
        }

        void display_boundaries_vtk( const char *filename ) {
            sdot::VtkOutput<1,TF> vo;
            bounds.display_boundaries( vo );
            vo.save( filename );
        }

        PD_TYPE coeff_at( pybind11::array_t<PD_TYPE> &point ) {
            auto buf_point = point.request(); auto ptr_buf_point = (PD_TYPE *)buf_point.ptr;
            if ( point.size() != PyPc::dim )
                throw pybind11::value_error( "wrong dimensions for point" );
            return bounds.coeff_at( ptr_buf_point );
        }

        PD_TYPE measure() {
            return bounds.measure();
        }

        pybind11::array_t<PD_TYPE> min_position() {
            pybind11::array_t<PD_TYPE> res;
            res.resize( { PyPc::dim } );
            auto buf_res = res.request();
            auto ptr_res = (PD_TYPE *)buf_res.ptr;

            auto p = bounds.min_position();
            for( std::size_t d = 0; d < PyPc::dim; ++d )
                ptr_res[ d ] = p[ d ];
            return res;
        }

        pybind11::array_t<PD_TYPE> max_position() {
            pybind11::array_t<PD_TYPE> res;
            res.resize( { PyPc::dim } );
            auto buf_res = res.request();
            auto ptr_res = (PD_TYPE *)buf_res.ptr;

            auto p = bounds.max_position();
            for( std::size_t d = 0; d < PyPc::dim; ++d )
                ptr_res[ d ] = p[ d ];
            return res;
        }

        //    void display_asy( const char *filename, py::array_t<PD_TYPE> &positions, py::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly &domain, PyZGrid &py_grid, const std::string &radial_func, const char *preamble, py::array_t<PD_TYPE> &values, std::string colormap, double linewidth, double dotwidth, bool avoid_bounds, const char *closing, double min_rf, double max_rf ) {
        //        auto buf_positions = positions.request();
        //        auto buf_weights = weights.request();
        //        auto buf_values = values.request();

        //        auto ptr_positions = reinterpret_cast<const PyZGrid::Pt *>( buf_positions.ptr );
        //        auto ptr_weights = reinterpret_cast<const PyZGrid::TF *>( buf_weights.ptr );
        //        auto ptr_values = reinterpret_cast<const PyZGrid::TF *>( buf_values.ptr );

        //        auto get_rgb = [&]( double &r, double &g, double &b, PD_TYPE v ) {
        //            int p = std::min( std::max( v, PD_TYPE( 0 ) ), PD_TYPE( 1 ) ) * 255;
        //            r = inferno_color_map[ 3 * p + 0 ];
        //            g = inferno_color_map[ 3 * p + 1 ];
        //            b = inferno_color_map[ 3 * p + 2 ];
        //        };

        //        std::ofstream f( filename );
        //        f << preamble;

        //        if ( linewidth <= 0 && dotwidth ) {
        //            if ( values.size() ) {
        //                double r, g, b;
        //                for( int n = 0; n < positions.shape( 0 ); ++n ) {
        //                    get_rgb( r, g, b, ptr_values[ n ] );
        //                    f << "dot((" << ptr_positions[ n ][ 0 ] << "," << ptr_positions[ n ][ 1 ] << "),rgb(" << r << "," << g << "," << b << "));\n";
        //                }
        //            } else {
        //                for( int n = 0; n < positions.shape( 0 ); ++n )
        //                    f << "dot((" << ptr_positions[ n ][ 0 ] << "," << ptr_positions[ n ][ 1 ] << "));\n";
        //            }
        //        } else {
        //            std::vector<std::ostringstream> outputs( thread_pool.nb_threads() );
        //            find_radial_func( radial_func, [&]( auto ft ) {
        //                py_grid.grid.for_each_laguerre_cell( [&]( auto &lc, std::size_t num_dirac_0, int num_thread ) {
        //                        domain.bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
        //                            if ( values.size() ) {
        //                                if ( min_rf < max_rf ) {
        //                                    ft.span_for_viz( [&]( double r0, double r1, double val ) {
        //                                        auto ncp = cp;
        //                                        for( std::size_t na = 0, tn = 20; na < tn; ++na ) {
        //                                            double a = 2 * M_PI * na / tn;
        //                                            Point2<PD_TYPE> d{ cos( a ), sin( a ) };
        //                                            ncp.plane_cut( cp.sphere_center + r1 * d, d, 0 );
        //                                        }
        //                                        double r, g, b;
        //                                        std::ostringstream os;
        //                                        get_rgb( r, g, b, ( val - min_rf ) / ( max_rf - min_rf ) );
        //                                        os << "rgb(" << r << "," << g << "," << b << ")";
        //                                        ncp.display_asy( outputs[ num_thread ], "", os.str(), true, avoid_bounds, false );
        //                                    }, ptr_weights[ num_dirac_0 ] );

        //                                    // boundaries
        //                                    cp.display_asy( outputs[ num_thread ] );
        //                                } else {
        //                                    double r, g, b;
        //                                    std::ostringstream os;
        //                                    get_rgb( r, g, b, ptr_values[ num_dirac_0 ] );
        //                                    os << "rgb(" << r << "," << g << "," << b << ")";
        //                                    cp.display_asy( outputs[ num_thread ], "", os.str(), true, avoid_bounds );
        //                                }
        //                            } else {
        //                                cp.display_asy( outputs[ num_thread ], "", "", false, avoid_bounds );
        //                            }
        //                        } );
        //                    },
        //                    domain.bounds.englobing_convex_polyhedron(),
        //                    ptr_positions,
        //                    ptr_weights,
        //                    positions.shape( 0 ),
        //                    false,
        //                    ft.need_ball_cut()
        //                );
        //            } );

        //            for( auto &os : outputs )
        //                f << os.str();
        //        }

        //        f << closing;
        //    }

        TB bounds;
    };

    template<int dim,class TF>
    struct PyScaledImage {
        using TB = sdot::ScaledImage<PyPc>;
        using Pt = typename TB::Pt;
        using TI = typename TB::TI;

        PyScaledImage( pybind11::array_t<PD_TYPE> &min_pt, pybind11::array_t<PD_TYPE> &max_pt, pybind11::array_t<PD_TYPE> &img ) {
            if ( min_pt.size() != PyPc::dim )
                throw pybind11::value_error( "wrong dimensions for point" );
            if ( max_pt.size() != PyPc::dim )
                throw pybind11::value_error( "wrong dimensions for point" );
            std::array<TI,dim> sizes;
            for( std::size_t d = 0; d < dim; ++d )
                sizes[ d ] = img.shape( dim - 1 - d );
            bounds = { Pt( min_pt.data() ), Pt( max_pt.data() ), img.data(), sizes };
        }

        // PD_TYPE coeff_at( pybind11::array_t<PD_TYPE> &point ) {
        //     auto buf_point = point.request(); auto ptr_buf_point = (PD_TYPE *)buf_point.ptr;
        // }

        void display_boundaries_vtk( const char *filename ) {
            sdot::VtkOutput<1,TF> vo;
            bounds.display_boundaries( vo );
            vo.save( filename );
        }

        PD_TYPE coeff_at( pybind11::array_t<PD_TYPE> &point ) {
            auto buf_point = point.request(); auto ptr_buf_point = (PD_TYPE *)buf_point.ptr;
            if ( point.size() != PyPc::dim )
                throw pybind11::value_error( "wrong dimensions for point" );
            return bounds.coeff_at( ptr_buf_point );
        }

        PD_TYPE measure() {
            return bounds.measure();
        }

        pybind11::array_t<PD_TYPE> min_position() {
            pybind11::array_t<PD_TYPE> res;
            res.resize( { PyPc::dim } );
            auto buf_res = res.request();
            auto ptr_res = (PD_TYPE *)buf_res.ptr;

            auto p = bounds.min_position();
            for( std::size_t d = 0; d < PyPc::dim; ++d )
                ptr_res[ d ] = p[ d ];
            return res;
        }

        pybind11::array_t<PD_TYPE> max_position() {
            pybind11::array_t<PD_TYPE> res;
            res.resize( { PyPc::dim } );
            auto buf_res = res.request();
            auto ptr_res = (PD_TYPE *)buf_res.ptr;

            auto p = bounds.max_position();
            for( std::size_t d = 0; d < PyPc::dim; ++d )
                ptr_res[ d ] = p[ d ];
            return res;
        }

        TB bounds;
    };


    // template<int dim,class TF>
    struct PyPowerDiagramZGrid {
        using Grid = sdot::SpZGrid<PyPc>;
        using Pt   = typename Grid::Pt;

        PyPowerDiagramZGrid( int max_dirac_per_cell ) : grid( max_dirac_per_cell ) {
        }

        void update( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, bool positions_have_changed, bool weights_have_changed, std::string radial_func ) {
            auto buf_positions = positions.request();
            auto buf_weights = weights.request();
            if ( buf_positions.shape[ 1 ] != PyPc::dim )
                throw pybind11::value_error( "dim does not correspond to shape[ 1 ] of positions" );

            grid.update(
                reinterpret_cast<const Pt *>( buf_positions.ptr ),
                reinterpret_cast<const TF *>( buf_weights.ptr ),
                positions.shape( 0 ),
                positions_have_changed,
                weights_have_changed,
                radial_func == "in_ball(weight**0.5)"
            );
        }

        pybind11::array_t<PD_TYPE> integrals_acp( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func ) {
            return get_integrals( positions, weights, domain.bounds, grid, func );
        }

        pybind11::array_t<PD_TYPE> integrals_img( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyScaledImage<dim,TF> &domain, const std::string &func ) {
            return get_integrals( positions, weights, domain.bounds, grid, func );
        }

        pybind11::array_t<PD_TYPE> centroids_acp( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func ) {
            return get_centroids( positions, weights, domain.bounds, grid, func );
        }

        pybind11::array_t<PD_TYPE> centroids_img( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyScaledImage<dim,TF> &domain, const std::string &func ) {
            return get_centroids( positions, weights, domain.bounds, grid, func );
        }

        template<class Domain>
        PyDerResult<dim,TF> der_integrals_wrt_weights( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const std::string &func, bool stop_if_void ) {
            auto buf_positions = positions.request();
            auto buf_weights = weights.request();

            auto ptr_positions = reinterpret_cast<const Pt *>( buf_positions.ptr );
            auto ptr_weights = reinterpret_cast<const TF *>( buf_weights.ptr );

            std::vector<std::size_t> w_m_offsets;
            std::vector<std::size_t> w_m_columns;
            std::vector<PD_TYPE    > w_m_values;
            std::vector<PD_TYPE    > w_v_values;

            PyDerResult<dim,TF> res;
            find_radial_func( func, [&]( auto ft ) {
                res.error = sdot::get_der_integrals_wrt_weights( w_m_offsets, w_m_columns, w_m_values, w_v_values, grid, domain.bounds, ptr_positions, ptr_weights, std::size_t( positions.shape( 0 ) ), ft, stop_if_void );
            } );

            vcp( res.m_offsets, w_m_offsets );
            vcp( res.m_columns, w_m_columns );
            vcp( res.m_values , w_m_values  );
            vcp( res.v_values , w_v_values  );

            return res;
        }

        PyDerResult<dim,TF> der_integrals_wrt_weights_acp( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func, bool stop_if_void ) {
            return der_integrals_wrt_weights( positions, weights, domain, func, stop_if_void );
        }   

        PyDerResult<dim,TF> der_integrals_wrt_weights_img( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyScaledImage<dim,TF> &domain, const std::string &func, bool stop_if_void ) {
            return der_integrals_wrt_weights( positions, weights, domain, func, stop_if_void );
        }   
    
        PyDerResult<dim,TF> der_centroids_and_integrals_wrt_weight_and_positions( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func ) {
            auto buf_positions = positions.request();
            auto buf_weights = weights.request();

            auto ptr_positions = reinterpret_cast<const Pt *>( buf_positions.ptr );
            auto ptr_weights = reinterpret_cast<const TF *>( buf_weights.ptr );

            std::vector<std::size_t> w_m_offsets;
            std::vector<std::size_t> w_m_columns;
            std::vector<PD_TYPE    > w_m_values;
            std::vector<PD_TYPE    > w_v_values;

            PyDerResult<dim,TF> res;
            find_radial_func( func, [&]( auto ft ) {
                res.error = sdot::get_der_centroids_and_integrals_wrt_weight_and_positions( w_m_offsets, w_m_columns, w_m_values, w_v_values, grid, domain.bounds, ptr_positions, ptr_weights, std::size_t( positions.shape( 0 ) ), ft );
            } );

            vcp( res.m_offsets, w_m_offsets );
            vcp( res.m_columns, w_m_columns );
            vcp( res.m_values , w_m_values  );
            vcp( res.v_values , w_v_values  );

            return res;
        }

        template<class Domain>
        void display_vtk( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const std::string &func, const char *filename, bool points, bool centroids ) {
            //        sdot::VtkOutput<1,TF> vo({ "num" });
            //        grid.display( vo );
            //        vo.save( filename );
            sdot::VtkOutput<3> vtk_output( { "weight", "num", "kind" } );

            // auto buf_positions = positions.request();
            // auto buf_weights = weights.request();
            auto ptr_positions = reinterpret_cast<const Pt *>( positions.data() );
            auto ptr_weights = weights.data();

            find_radial_func( func, [&]( auto ft ) {
                grid.for_each_laguerre_cell(
                    [&]( auto &lc, std::size_t num_dirac_0, int ) {
                        domain.bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
                            cp.display( vtk_output, { ptr_weights[ num_dirac_0 ], TF( num_dirac_0 ), TF( 0 ) } );
                        } );
                    },
                    domain.bounds.englobing_convex_polyhedron(),
                    ptr_positions,
                    ptr_weights,
                    positions.shape( 0 ),
                    false,
                    ft.need_ball_cut()
                );
            } );

            if ( points ) {
                for( int n = 0; n < positions.shape( 0 ); ++n )
                    vtk_output.add_point( ptr_positions[ n ], { ptr_weights[ n ], TF( n ), TF( 1 ) } );
            }

            if ( centroids ) {
                std::vector<Pt> c( positions.shape( 0 ) );
                find_radial_func( func, [&]( auto ft ) {
                    sdot::get_centroids( grid, domain.bounds, ptr_positions, ptr_weights, positions.shape( 0 ), ft, [&]( auto centroid, auto, auto num ) {
                        c[ num ] = centroid;
                    } );
                } );

                for( int n = 0; n < positions.shape( 0 ); ++n )
                    vtk_output.add_point( c[ n ], { ptr_weights[ n ], TF( n ), TF( 2 ) } );
            }

            vtk_output.save( filename );
        }

        void display_vtk_acp( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func, const char *filename, bool points, bool centroids ) {
            display_vtk( positions, weights, domain, func, filename, points, centroids );
        }

        void display_vtk_img( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyScaledImage<dim,TF> &domain, const std::string &func, const char *filename, bool points, bool centroids ) {
            display_vtk( positions, weights, domain, func, filename, points, centroids );
        }

        template<class Domain>
        std::string display_html_canvas( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const std::string &func ) {
            auto ptr_positions = reinterpret_cast<const Pt *>( positions.data() );
            auto ptr_weights = reinterpret_cast<const TF *>( weights.data() );

            // output variable
            std::ostringstream out;
            out << "(function() {\n";

            // diracs
            out << "var diracs = [\n";
            for( int n = 0; n < positions.shape( 0 ); ++n )
                out << "    [" << ptr_positions[ n ][ 0 ] << ", " << ptr_positions[ n ][ 1 ] << "],\n";
            out << "];\n";

            // centroids
            std::vector<Pt> c( positions.shape( 0 ) );
            find_radial_func( func, [&]( auto ft ) {
                sdot::get_centroids( grid, domain.bounds, ptr_positions, ptr_weights, positions.shape( 0 ), ft, [&]( auto centroid, auto, auto num ) {
                    c[ num ] = centroid;
                } );
            } );

            out << "var centroids = [\n";
            for( int n = 0; n < positions.shape( 0 ); ++n )
                out << "    [" << c[ n ][ 0 ] << ", " << c[ n ][ 1 ] << "],\n";
            out << "];\n";

            // get contributions for min/max and the path
            std::vector<Pt> min_pts( thread_pool.nb_threads(), Pt( + std::numeric_limits<TF>::max() ) );
            std::vector<Pt> max_pts( thread_pool.nb_threads(), Pt( - std::numeric_limits<TF>::max() ) );
            std::vector<std::ostringstream> os( thread_pool.nb_threads() );
            find_radial_func( func, [&]( auto ft ) {
                grid.for_each_laguerre_cell(
                    [&]( auto &lc, std::size_t num_dirac_0, int num_thread ) {
                        domain.bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
                            cp.display_html_canvas( os[ num_thread ], ptr_weights[ num_dirac_0 ] );

                            cp.for_each_node( [&]( Pt v ) {
                                min_pts[ num_thread ] = min( min_pts[ num_thread ], v );
                                max_pts[ num_thread ] = max( max_pts[ num_thread ], v );
                            } );
                        } );
                    },
                    domain.bounds.englobing_convex_polyhedron(),
                    ptr_positions,
                    ptr_weights,
                    positions.shape( 0 ),
                    false,
                    ft.need_ball_cut()
                );
            } );
            out << "var path = new Path2D();\n";
            for( std::ostringstream &o : os )
                out << o.str();

            // display min/max
            Pt min_pt = min_pts[ 0 ];
            Pt max_pt = max_pts[ 0 ];
            for( Pt &p : min_pts )
                min_pt = min( min_pt, p );
            for( Pt &p : max_pts )
                max_pt = max( max_pt, p );

            out << "var min_x = " << min_pt[ 0 ] << ";\n";
            out << "var min_y = " << min_pt[ 1 ] << ";\n";
            out << "var max_x = " << max_pt[ 0 ] << ";\n";
            out << "var max_y = " << max_pt[ 1 ] << ";\n";

            out << "pd_list.push({ path, min_x, min_y, max_x, max_y, diracs, centroids });\n";
            out << "})();\n";
            return out.str();
        }

        std::string display_html_canvas_acp( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func ) {
            return display_html_canvas( positions, weights, domain, func );
        }

        std::string display_html_canvas_img( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyScaledImage<dim,TF> &domain, const std::string &func ) {
            return display_html_canvas( positions, weights, domain, func );
        }

        void display_vtk_points( pybind11::array_t<PD_TYPE> &positions, const char *filename ) {
            sdot::VtkOutput<1> vtk_output( { "num" } );

            auto buf_positions = positions.request();
            auto ptr_positions = reinterpret_cast<const Pt *>( buf_positions.ptr );
            for( int n = 0; n < positions.shape( 0 ); ++n )
                vtk_output.add_point( ptr_positions[ n ], { TF( n ) } );

            vtk_output.save( filename );
        }

        void display_asy( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &radial_func, const char *filename, const char *preamble, pybind11::array_t<PD_TYPE> &values, std::string colormap, double linewidth, double dotwidth, bool avoid_bounds, const char *closing, double min_rf, double max_rf ) {
            auto ptr_positions = reinterpret_cast<const Pt *>( positions.data() );
            #if PD_DIM==2
            auto ptr_weights = weights.data();
            #endif
            auto ptr_values = values.data();

            // vtk_output.save( filename );
            auto get_rgb = [&]( double &r, double &g, double &b, TF v ) {
                int p = std::min( std::max( v, TF( 0 ) ), TF( 1 ) ) * 255;
                r = inferno_color_map[ 3 * p + 0 ];
                g = inferno_color_map[ 3 * p + 1 ];
                b = inferno_color_map[ 3 * p + 2 ];
            };

            std::ofstream f( filename );
            f << preamble;

            if ( linewidth <= 0 && dotwidth ) {
                if ( values.size() ) {
                    double r, g, b;
                    for( int n = 0; n < positions.shape( 0 ); ++n ) {
                        get_rgb( r, g, b, ptr_values[ n ] );
                        f << "dot((" << ptr_positions[ n ][ 0 ] << "," << ptr_positions[ n ][ 1 ] << "),rgb(" << r << "," << g << "," << b << "));\n";
                    }
                } else {
                    for( int n = 0; n < positions.shape( 0 ); ++n )
                        f << "dot((" << ptr_positions[ n ][ 0 ] << "," << ptr_positions[ n ][ 1 ] << "));\n";
                }
            } else {
                #if PD_DIM==2
                std::vector<std::ostringstream> outputs( thread_pool.nb_threads() );
                find_radial_func( radial_func, [&]( auto ft ) {
                    grid.for_each_laguerre_cell(
                        [&]( auto &lc, std::size_t num_dirac_0, int num_thread ) {
                            domain.bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
                                if ( values.size() ) {
                                    if ( min_rf < max_rf ) {
                                        ft.span_for_viz( [&]( double r0, double r1, double val ) {
                                            auto ncp = cp;
                                            for( std::size_t na = 0, tn = 20; na < tn; ++na ) {
                                                double a = 2 * M_PI * na / tn;
                                                sdot::Point2<TF> d{ cos( a ), sin( a ) };
                                                ncp.plane_cut( cp.sphere_center + r1 * d, d, 0 );
                                            }
                                            double r, g, b;
                                            std::ostringstream os;
                                            get_rgb( r, g, b, ( val - min_rf ) / ( max_rf - min_rf ) );
                                            os << "rgb(" << r << "," << g << "," << b << ")";
                                            ncp.display_asy( outputs[ num_thread ], "", os.str(), true, avoid_bounds, false );
                                        }, ptr_weights[ num_dirac_0 ] );

                                        // boundaries
                                        cp.display_asy( outputs[ num_thread ] );
                                    } else {
                                        double r, g, b;
                                        std::ostringstream os;
                                        get_rgb( r, g, b, ptr_values[ num_dirac_0 ] );
                                        os << "rgb(" << r << "," << g << "," << b << ")";
                                        cp.display_asy( outputs[ num_thread ], "", os.str(), true, avoid_bounds );
                                    }
                                } else {
                                    cp.display_asy( outputs[ num_thread ], "", "", false, avoid_bounds );
                                }
                            } );
                        },
                        domain.bounds.englobing_convex_polyhedron(),
                        ptr_positions,
                        ptr_weights,
                        positions.shape( 0 ),
                        false,
                        ft.need_ball_cut()
                    );
                } );

                for( auto &os : outputs )
                    f << os.str();
                #endif
            }

            f << closing;
        }

        Grid grid;
    };
}

PYBIND11_MODULE( PD_MODULE_NAME, m ) {
    m.doc() = "Semi-discrete optimal transportation";

    using DerResult = PyDerResult<PD_DIM,PD_TYPE>;
    pybind11::class_<DerResult>( m, "DerResult" )
        .def_readwrite( "m_offsets"                                 , &DerResult::m_offsets                                                   , "" )
        .def_readwrite( "m_columns"                                 , &DerResult::m_columns                                                   , "" )
        .def_readwrite( "m_values"                                  , &DerResult::m_values                                                    , "" )
        .def_readwrite( "v_values"                                  , &DerResult::v_values                                                    , "" )
        .def_readwrite( "error"                                     , &DerResult::error                                                       , "" )
    ;

    using ConvexPolyhedraAssembly = PyConvexPolyhedraAssembly<PD_DIM,PD_TYPE>;
    pybind11::class_<ConvexPolyhedraAssembly>( m, "ConvexPolyhedraAssembly" )
        .def( pybind11::init<>()                                                                                                              , "" )
        .def( "add_convex_polyhedron"                               , &ConvexPolyhedraAssembly::add_convex_polyhedron                         , "" )
        .def( "add_box"                                             , &ConvexPolyhedraAssembly::add_box                                       , "" )
        .def( "normalize"                                           , &ConvexPolyhedraAssembly::normalize                                     , "" )
        .def( "display_boundaries_vtk"                              , &ConvexPolyhedraAssembly::display_boundaries_vtk                        , "" )
        .def( "min_position"                                        , &ConvexPolyhedraAssembly::min_position                                  , "" )
        .def( "max_position"                                        , &ConvexPolyhedraAssembly::max_position                                  , "" )
        .def( "coeff_at"                                            , &ConvexPolyhedraAssembly::coeff_at                                      , "" )
        .def( "measure"                                             , &ConvexPolyhedraAssembly::measure                                       , "" )
    ;

    using ScaledImage = PyScaledImage<PD_DIM,PD_TYPE>;
    pybind11::class_<ScaledImage>( m, "ScaledImage" )
        .def( pybind11::init<pybind11::array_t<PD_TYPE> &, pybind11::array_t<PD_TYPE> &, pybind11::array_t<PD_TYPE> &>()                     , "" )
        .def( "display_boundaries_vtk"                              , &ScaledImage::display_boundaries_vtk                                   , "" )
        .def( "min_position"                                        , &ScaledImage::min_position                                             , "" )
        .def( "max_position"                                        , &ScaledImage::max_position                                             , "" )
        .def( "coeff_at"                                            , &ScaledImage::coeff_at                                                 , "" )
        .def( "measure"                                             , &ScaledImage::measure                                                  , "" )
    ;

    using PowerDiagramZGrid = PyPowerDiagramZGrid;
    pybind11::class_<PowerDiagramZGrid>( m, "PowerDiagramZGrid" )
        .def( pybind11::init<int>()                                                                                                           , "" )
        .def( "update"                                              , &PowerDiagramZGrid::update                                              , "" )
        .def( "integrals"                                           , &PowerDiagramZGrid::integrals_acp                                       , "" )
        .def( "integrals"                                           , &PowerDiagramZGrid::integrals_img                                       , "" )
        .def( "der_integrals_wrt_weights"                           , &PowerDiagramZGrid::der_integrals_wrt_weights_acp                       , "" )
        .def( "der_integrals_wrt_weights"                           , &PowerDiagramZGrid::der_integrals_wrt_weights_img                       , "" )
        .def( "der_centroids_and_integrals_wrt_weight_and_positions", &PowerDiagramZGrid::der_centroids_and_integrals_wrt_weight_and_positions, "" )
        .def( "centroids"                                           , &PowerDiagramZGrid::centroids_acp                                       , "" )
        .def( "centroids"                                           , &PowerDiagramZGrid::centroids_img                                       , "" )
        .def( "display_vtk"                                         , &PowerDiagramZGrid::display_vtk_acp                                     , "" )
        .def( "display_vtk"                                         , &PowerDiagramZGrid::display_vtk_img                                     , "" )
        .def( "display_html_canvas"                                 , &PowerDiagramZGrid::display_html_canvas_acp                             , "" )
        .def( "display_html_canvas"                                 , &PowerDiagramZGrid::display_html_canvas_img                             , "" )
        .def( "display_vtk_points"                                  , &PowerDiagramZGrid::display_vtk_points                                  , "" )
        .def( "display_asy"                                         , &PowerDiagramZGrid::display_asy                                         , "" )
    ;
}

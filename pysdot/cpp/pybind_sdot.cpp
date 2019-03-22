// #include "../../ext/sdot/src/sdot/Visitors/internal/ZCoords.cpp"
#include "../../ext/sdot/src/sdot/Support/ThreadPool.cpp"
#include "../../ext/sdot/src/sdot/Support/CbQueue.cpp"
#include "../../ext/sdot/src/sdot/Support/Assert.cpp"
#include "../../ext/sdot/src/sdot/Support/Mpi.cpp"

#include "../../ext/sdot/src/sdot/Domains/ConvexPolyhedronAssembly.h"

#include "../../ext/sdot/src/sdot/PowerDiagram/Visitors/SpZGrid.h"

#include "../../ext/sdot/src/sdot/PowerDiagram/get_der_integrals_wrt_weights.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_integrals.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_centroids.h"

#include "../../ext/sdot/src/sdot/Display/VtkOutput.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>

namespace {
    template<class FU>
    void find_radial_func( const std::string &func, const FU &fu ) {
        if ( func == "1" || func == "unit" ) {
            fu( FunctionEnum::Unit() );
            return;
        }

        if ( func.size() > 13 && func.substr( 0, 13 ) == "exp((w-r**2)/" ) {
            PD_TYPE eps;
            std::istringstream is( func.substr( 13, func.size() - 14 ) );
            is >> eps;
            fu( FunctionEnum::ExpWmR2db<PD_TYPE>{ eps } );
            return;
        }

        if ( func == "r**2" || func == "r^2" ) {
            fu( FunctionEnum::R2() );
            return;
        }

        if ( func == "in_ball(weight**0.5)" ) {
            fu( FunctionEnum::InBallW05() );
            return;
        }

        throw pybind11::value_error( "unknown function type" );
    }

    template<class Domain,class Grid>
    pybind11::array_t<PD_TYPE> get_integrals( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, Grid &grid, const std::string &func ) {
        auto buf_positions = positions.request();
        auto buf_weights = weights.request();

        auto ptr_positions = reinterpret_cast<const typename Grid::Pt *>( buf_positions.ptr );
        auto ptr_weights = reinterpret_cast<const typename Grid::TF *>( buf_weights.ptr );

        pybind11::array_t<PD_TYPE> res;
        res.resize( { positions.shape( 0 ) } );
        auto buf_res = res.request();
        auto ptr_res = (PD_TYPE *)buf_res.ptr;

        find_radial_func( func, [&]( auto ft ) {
            sdot::get_integrals( ptr_res, grid, domain, ptr_positions, ptr_weights, positions.shape( 0 ), ft );
        } );

        return res;
    }

    template<class Domain,class Grid>
    pybind11::array_t<PD_TYPE> get_centroids( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, Grid &grid, const std::string &func ) {
        auto buf_positions = positions.request();
        auto buf_weights = weights.request();

        auto ptr_positions = reinterpret_cast<const typename Grid::Pt *>( buf_positions.ptr );
        auto ptr_weights = reinterpret_cast<const typename Grid::TF *>( buf_weights.ptr );

        pybind11::array_t<PD_TYPE> res;
        res.resize( { positions.shape( 0 ), pybind11::ssize_t( PD_DIM ) } );
        auto buf_res = res.request();
        auto ptr_res = (PD_TYPE *)buf_res.ptr;

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
        pybind11::array_t<PD_TYPE>     m_values;
        pybind11::array_t<PD_TYPE>     v_values;
        int                            error;
    };
}

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

    pybind11::array_t<PD_TYPE> integrals( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func ) {
        return get_integrals( positions, weights, domain.bounds, grid, func );
    }

    pybind11::array_t<PD_TYPE> centroids( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func ) {
        return get_centroids( positions, weights, domain.bounds, grid, func );
    }

    PyDerResult<dim,TF> der_integrals_wrt_weights( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func ) {
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
            res.error = sdot::get_der_integrals_wrt_weights( w_m_offsets, w_m_columns, w_m_values, w_v_values, grid, domain.bounds, ptr_positions, ptr_weights, std::size_t( positions.shape( 0 ) ), ft );
        } );

        vcp( res.m_offsets, w_m_offsets );
        vcp( res.m_columns, w_m_columns );
        vcp( res.m_values , w_m_values  );
        vcp( res.v_values , w_v_values  );

        return res;
    }

    void display_vtk( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, PyConvexPolyhedraAssembly<dim,TF> &domain, const std::string &func, const char *filename ) {
        //        sdot::VtkOutput<1,TF> vo({ "num" });
        //        grid.display( vo );
        //        vo.save( filename );
        sdot::VtkOutput<2> vtk_output( { "weight", "num" } );

        auto buf_positions = positions.request();
        auto buf_weights = weights.request();

        auto ptr_positions = reinterpret_cast<const Pt *>( buf_positions.ptr );
        auto ptr_weights = reinterpret_cast<const TF *>( buf_weights.ptr );

        find_radial_func( func, [&]( auto ft ) {
            grid.for_each_laguerre_cell(
                [&]( auto &lc, std::size_t num_dirac_0, int ) {
                    domain.bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
                        cp.display( vtk_output, { ptr_weights[ num_dirac_0 ], TF( num_dirac_0 ) } );
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

        vtk_output.save( filename );
    }

    void display_vtk_points( pybind11::array_t<PD_TYPE> &positions, const char *filename ) {
        sdot::VtkOutput<1> vtk_output( { "num" } );

        auto buf_positions = positions.request();
        auto ptr_positions = reinterpret_cast<const Pt *>( buf_positions.ptr );
        for( int n = 0; n < positions.shape( 0 ); ++n )
            vtk_output.add_point( ptr_positions[ n ], { TF( n ) } );

        vtk_output.save( filename );
    }

    Grid grid;
};

PYBIND11_MODULE( PD_MODULE_NAME, m ) {
    m.doc() = "Semi-discrete optimal transportation";

    using DerResult = PyDerResult<PD_DIM,PD_TYPE>;
    pybind11::class_<DerResult>( m, "DerResult" )
        .def_readwrite( "m_offsets"      , &DerResult::m_offsets                           , "" )
        .def_readwrite( "m_columns"      , &DerResult::m_columns                           , "" )
        .def_readwrite( "m_values"       , &DerResult::m_values                            , "" )
        .def_readwrite( "v_values"       , &DerResult::v_values                            , "" )
        .def_readwrite( "error"          , &DerResult::error                               , "" )
    ;

    using ConvexPolyhedraAssembly = PyConvexPolyhedraAssembly<PD_DIM,PD_TYPE>;
    pybind11::class_<ConvexPolyhedraAssembly>( m, "ConvexPolyhedraAssembly" )
        .def( pybind11::init<>()                                                           , "" )
        .def( "add_convex_polyhedron"    , &ConvexPolyhedraAssembly::add_convex_polyhedron , "" )
        .def( "add_box"                  , &ConvexPolyhedraAssembly::add_box               , "" )
        .def( "normalize"                , &ConvexPolyhedraAssembly::normalize             , "" )
        .def( "display_boundaries_vtk"   , &ConvexPolyhedraAssembly::display_boundaries_vtk, "" )
        .def( "min_position"             , &ConvexPolyhedraAssembly::min_position          , "" )
        .def( "max_position"             , &ConvexPolyhedraAssembly::max_position          , "" )
        .def( "coeff_at"                 , &ConvexPolyhedraAssembly::coeff_at              , "" )
        .def( "measure"                  , &ConvexPolyhedraAssembly::measure               , "" )
    ;

    using PowerDiagramZGrid = PyPowerDiagramZGrid<PD_DIM,PD_TYPE>;
    pybind11::class_<PowerDiagramZGrid>( m, "PowerDiagramZGrid" )
        .def( pybind11::init<int>()                                                        , "" )
        .def( "update"                   , &PowerDiagramZGrid::update                      , "" )
        .def( "integrals"                , &PowerDiagramZGrid::integrals                   , "" )
        .def( "der_integrals_wrt_weights", &PowerDiagramZGrid::der_integrals_wrt_weights   , "" )
        .def( "centroids"                , &PowerDiagramZGrid::centroids                   , "" )
        .def( "display_vtk"              , &PowerDiagramZGrid::display_vtk                 , "" )
        .def( "display_vtk_points"       , &PowerDiagramZGrid::display_vtk_points          , "" )
    ;

    //    m.def( "display_asy"                  , &display_asy                   );
    //    m.def( "display_vtk"                  , &display_vtk                   );
    //    m.def( "get_centroids"                , &get_centroids                 );
    //    m.def( "get_integrals"                , &get_integrals                 );
    //    m.def( "get_der_integrals_wrt_weights", &get_der_integrals_wrt_weights );
}

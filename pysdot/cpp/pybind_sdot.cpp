// #include "../../ext/sdot/src/sdot/Visitors/internal/ZCoords.cpp"
#include "../../ext/sdot/src/sdot/Support/ThreadPool.cpp"
#include "../../ext/sdot/src/sdot/Support/CbQueue.cpp"
#include "../../ext/sdot/src/sdot/Support/Assert.cpp"
#include "../../ext/sdot/src/sdot/Support/Mpi.cpp"

#include "../../ext/sdot/src/sdot/Integration/Arfd.cpp"

#ifdef PD_WANT_STAT
#include "../../ext/sdot/src/sdot/Support/Stat.cpp"
#endif

#include "../../ext/sdot/src/sdot/Domains/ConvexPolyhedronAssembly.h"
#include "../../ext/sdot/src/sdot/Domains/ScaledImage.h"

#include "../../ext/sdot/src/sdot/PowerDiagram/Visitors/SpZGrid.h"

#include "../../ext/sdot/src/sdot/PowerDiagram/get_der_centroids_and_integrals_wrt_weight_and_positions.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_der_integrals_wrt_weights.h"
// #include "../../ext/sdot/src/sdot/PowerDiagram/get_der_boundary_integral.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_distances_from_boundaries.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_boundary_integral.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_image_integrals.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_integrals.h"
#include "../../ext/sdot/src/sdot/PowerDiagram/get_centroids.h"

#include "../../ext/sdot/src/sdot/Display/VtkOutput.h"
#include "inferno_color_map.h"

#include <pybind11/functional.h>
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
            sdot::FunctionEnum::Unit f;
            fu( f );
            return;
        }

        if ( func.size() > 13 && func.substr( 0, 13 ) == "exp((w-r**2)/" ) {
            PD_TYPE eps;
            std::istringstream is( func.substr( 13, func.size() - 14 ) );

            is >> eps;
            sdot::FunctionEnum::ExpWmR2db<PD_TYPE> f{ eps };
            fu( f );
            return;
        }

        if ( func == "r**2" || func == "r^2" ) {
            sdot::FunctionEnum::R2 f;
            fu( f );
            return;
        }

        if ( func == "pos_part(w-r**2)" || func == "pos_part(w-r^2)" ) {
            sdot::FunctionEnum::PpWmR2 f;
            fu( f );
            return;
        }

        if ( func == "in_ball(weight**0.5)" ) {
            sdot::FunctionEnum::InBallW05 f;
            fu( f );
            return;
        }

        if ( func == "r**2*in_ball(weight**0.5)" || func == "r^2*in_ball(weight**0.5)" ) {
            sdot::FunctionEnum::R2InBallW05 f;
            fu( f );
            return;
        }

        throw pybind11::value_error( "unknown function type" );
    }

    template<class RF,class FU>
    void find_radial_func( const RF &func, const FU &fu ) {
        pybind11::gil_scoped_release release{};
        fu( func );
    }

    template<class Domain,class Grid,class FUNC>
    pybind11::array_t<TF> get_integrals( pybind11::array_t<TF> &positions, pybind11::array_t<TF> &weights, Domain &domain, Grid &grid, const FUNC &func ) {
        auto ptr_positions = reinterpret_cast<const typename Grid::Pt *>( positions.data() );
        auto ptr_weights = reinterpret_cast<const typename Grid::TF *>( weights.data() );

        pybind11::array_t<TF> res;
        res.resize( { positions.shape( 0 ) } );
        auto buf_res = res.request();
        auto ptr_res = (TF *)buf_res.ptr;

        find_radial_func( func, [&]( const auto &ft ) {
            sdot::get_integrals( ptr_res, grid, domain, ptr_positions, ptr_weights, positions.shape( 0 ), ft );
        } );

        return res;
    }

    template<class Domain,class Grid,class FUNC>
    pybind11::array_t<TF> get_image_integrals( pybind11::array_t<TF> &positions, pybind11::array_t<TF> &weights, Domain &domain, Grid &grid, const FUNC &func, pybind11::array_t<TF> &beg, pybind11::array_t<TF> &end, pybind11::array_t<std::size_t> &nbp ) {
        auto ptr_positions = reinterpret_cast<const typename Grid::Pt *>( positions.data() );
        auto ptr_weights = reinterpret_cast<const typename Grid::TF *>( weights.data() );
        auto ptr_nbp = reinterpret_cast<const std::size_t *>( nbp.data() );
        auto ptr_beg = reinterpret_cast<const TF *>( beg.data() );
        auto ptr_end = reinterpret_cast<const TF *>( end.data() );
        using Pt = typename Grid::Pt;
        using ST = std::size_t;

        Pt a_beg;
        Pt a_end;
        std::array<ST,Grid::dim> a_nbp;
        for( int d = 0; d < Grid::dim; ++d ) {
            a_beg[ d ] = ptr_beg[ Grid::dim - 1 - d ];
            a_end[ d ] = ptr_end[ Grid::dim - 1 - d ];
            a_nbp[ d ] = ptr_nbp[ Grid::dim - 1 - d ];
        }

        std::array<std::size_t,Grid::dim+1> shape;
        for( int d = 0; d < Grid::dim; ++d )
            shape[ d ] = a_nbp[ d ];
        shape[ Grid::dim ] = Grid::dim + 1;

        pybind11::array_t<TF> res;
        res.resize( shape );

        find_radial_func( func, [&]( const auto &ft ) {
            sdot::get_image_integrals( res.mutable_data(), grid, domain, ptr_positions, ptr_weights, positions.shape( 0 ), ft, a_beg, a_end, a_nbp );
        } );

        return res;
    }

    template<class Domain,class Grid,class FUNC>
    pybind11::array_t<TF> get_centroids( pybind11::array_t<TF> &positions, pybind11::array_t<TF> &weights, Domain &domain, Grid &grid, const FUNC &func, TF rand_ratio ) {
        auto buf_positions = positions.request();
        auto buf_weights = weights.request();

        auto ptr_positions = reinterpret_cast<const typename Grid::Pt *>( buf_positions.ptr );
        auto ptr_weights = reinterpret_cast<const typename Grid::TF *>( buf_weights.ptr );

        pybind11::array_t<TF> res;
        res.resize( { positions.shape( 0 ), pybind11::ssize_t( PD_DIM ) } );
        auto buf_res = res.request();
        auto ptr_res = (TF *)buf_res.ptr;

        find_radial_func( func, [&]( const auto &ft ) {
            sdot::get_centroids( grid, domain, ptr_positions, ptr_weights, positions.shape( 0 ), ft, [&]( auto centroid, auto, auto num ) {
                for( int d = 0; d < PD_DIM; ++d )
                    ptr_res[ PD_DIM * num + d ] = centroid[ d ];
            }, rand_ratio );
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
        static constexpr int allow_translations = 1;
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
            if ( min_pt.size() != dim )
                throw pybind11::value_error( "wrong dimensions for point" );
            if ( max_pt.size() != dim )
                throw pybind11::value_error( "wrong dimensions for point" );

            std::array<TI,dim> sizes;
            for( std::size_t d = 0; d < dim; ++d )
                sizes[ d ] = img.shape( img.ndim() - 1 - d );

            TI nb_coeffs = 1;
            if ( img.ndim() == dim + 1 )
                nb_coeffs = img.shape( 0 );

            bounds = { Pt( min_pt.data() ), Pt( max_pt.data() ), img.data(), sizes, nb_coeffs };
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

    #define PyPowerDiagramZGrid PyPowerDiagramZGrid_##PD_DIM

    struct PyPowerDiagramZGrid {
        using Grid = sdot::SpZGrid<PyPc>;
        using Pt   = typename Grid::Pt;

        PyPowerDiagramZGrid( int max_dirac_per_cell ) : grid( max_dirac_per_cell ) {
        }

        void add_replication( pybind11::array_t<PD_TYPE> &positions ) {
            Pt p;
            for( int d = 0; d < PD_DIM; ++d )
                p[ d ] = positions.at( d );
            grid.translations.push_back( p );
        }

        void update( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, bool positions_have_changed, bool weights_have_changed, bool ball_cut ) {
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
                ball_cut
            );
        }

        template<class Domain,class FUNC>
        PyDerResult<dim,TF> der_integrals_wrt_weights( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const FUNC &func, bool stop_if_void ) {
            auto buf_positions = positions.request();
            auto buf_weights = weights.request();

            auto ptr_positions = reinterpret_cast<const Pt *>( buf_positions.ptr );
            auto ptr_weights = reinterpret_cast<const TF *>( buf_weights.ptr );

            std::vector<std::size_t> w_m_offsets;
            std::vector<std::size_t> w_m_columns;
            std::vector<PD_TYPE    > w_m_values;
            std::vector<PD_TYPE    > w_v_values;

            PyDerResult<dim,TF> res;
            find_radial_func( func, [&]( const auto &ft ) {
                res.error = sdot::get_der_integrals_wrt_weights( w_m_offsets, w_m_columns, w_m_values, w_v_values, grid, domain.bounds, ptr_positions, ptr_weights, std::size_t( positions.shape( 0 ) ), ft, stop_if_void );
            } );

            vcp( res.m_offsets, w_m_offsets );
            vcp( res.m_columns, w_m_columns );
            vcp( res.m_values , w_m_values  );
            vcp( res.v_values , w_v_values  );

            return res;
        }

        template<class Domain,class FUNC>
        pybind11::array_t<TF> distances_from_boundaries( pybind11::array_t<PD_TYPE> &points, pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const FUNC &func, bool count_domain_boundaries ) {
            auto ptr_positions = reinterpret_cast<const Pt *>( positions.data() );
            auto ptr_points = reinterpret_cast<const Pt *>( points.data() );
            auto ptr_weights = weights.data();

            pybind11::array_t<PD_TYPE> res;
            res.resize( { points.shape( 0 ) } );

            find_radial_func( func, [&]( const auto &ft ) {
                sdot::get_distances_from_boundaries( res.mutable_data(), ptr_points, std::size_t( points.shape( 0 ) ), grid, domain.bounds, ptr_positions, ptr_weights, std::size_t( positions.shape( 0 ) ), ft, count_domain_boundaries );
            } );

            return res;
        }

        template<class Domain,class FUNC>
        PyDerResult<dim,TF> der_centroids_and_integrals_wrt_weight_and_positions( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const FUNC &func ) {
            auto buf_positions = positions.request();
            auto buf_weights = weights.request();

            auto ptr_positions = reinterpret_cast<const Pt *>( buf_positions.ptr );
            auto ptr_weights = reinterpret_cast<const TF *>( buf_weights.ptr );

            std::vector<std::size_t> w_m_offsets;
            std::vector<std::size_t> w_m_columns;
            std::vector<PD_TYPE    > w_m_values;
            std::vector<PD_TYPE    > w_v_values;

            PyDerResult<dim,TF> res;
            find_radial_func( func, [&]( const auto &ft ) {
                res.error = sdot::get_der_centroids_and_integrals_wrt_weight_and_positions( w_m_offsets, w_m_columns, w_m_values, w_v_values, grid, domain.bounds, ptr_positions, ptr_weights, std::size_t( positions.shape( 0 ) ), ft );
            } );

            vcp( res.m_offsets, w_m_offsets );
            vcp( res.m_columns, w_m_columns );
            vcp( res.m_values , w_m_values  );
            vcp( res.v_values , w_v_values  );

            return res;
        }

        template<class Domain,class FUNC>
        void display_vtk( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const FUNC &func, const char *filename, bool points, bool centroids ) {
            //        sdot::VtkOutput<1,TF> vo({ "num" });
            //        grid.display( vo );
            //        vo.save( filename );
            sdot::VtkOutput<3> vtk_output( { "weight", "num", "kind" } );

            // auto buf_positions = positions.request();
            // auto buf_weights = weights.request();
            auto ptr_positions = reinterpret_cast<const Pt *>( positions.data() );
            auto ptr_weights = weights.data();

            find_radial_func( func, [&]( const auto &ft ) {
                grid.for_each_laguerre_cell(
                    [&]( auto &lc, std::size_t num_dirac_0, int ) {
                        domain.bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
                            if ( space_func )
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
                find_radial_func( func, [&]( const auto &ft ) {
                    sdot::get_centroids( grid, domain.bounds, ptr_positions, ptr_weights, positions.shape( 0 ), ft, [&]( auto centroid, auto, auto num ) {
                        c[ num ] = centroid;
                    } );
                } );

                for( int n = 0; n < positions.shape( 0 ); ++n )
                    vtk_output.add_point( c[ n ], { ptr_weights[ n ], TF( n ), TF( 2 ) } );
            }

            vtk_output.save( filename );
        }

        template<class Domain,class FUNC>
        std::string display_html_canvas( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const FUNC &func, int hide_after ) {
            std::size_t nd = hide_after >= 0 ? std::min( hide_after, int( positions.shape( 0 ) ) ) : positions.shape( 0 );
            auto ptr_positions = reinterpret_cast<const Pt *>( positions.data() );
            auto ptr_weights = reinterpret_cast<const TF *>( weights.data() );

            // output variable
            std::ostringstream out;
            out << "(function() {\n";

            // diracs
            out << "var diracs = [\n";
            for( std::size_t n = 0; n < nd; ++n )
                out << "    [" << ptr_positions[ n ][ 0 ] << ", " << ptr_positions[ n ][ 1 ] << "],\n";
            out << "];\n";

            // centroids
            std::vector<Pt> c( positions.shape( 0 ) );
            find_radial_func( func, [&]( const auto &ft ) {
                sdot::get_centroids( grid, domain.bounds, ptr_positions, ptr_weights, positions.shape( 0 ), ft, [&]( auto centroid, auto, auto num ) {
                    c[ num ] = centroid;
                } );
            } );

            out << "var centroids = [\n";
            for( std::size_t n = 0; n < nd; ++n )
                out << "    [" << c[ n ][ 0 ] << ", " << c[ n ][ 1 ] << "],\n";
            out << "];\n";

            // get contributions for min/max and the path
            std::vector<Pt> min_pts( thread_pool.nb_threads(), Pt( + std::numeric_limits<TF>::max() ) );
            std::vector<Pt> max_pts( thread_pool.nb_threads(), Pt( - std::numeric_limits<TF>::max() ) );
            std::vector<std::ostringstream> os_int( thread_pool.nb_threads() );
            std::vector<std::ostringstream> os_ext( thread_pool.nb_threads() );
            find_radial_func( func, [&]( const auto &ft ) {
                grid.for_each_laguerre_cell(
                    [&]( auto &lc, std::size_t num_dirac_0, int num_thread ) {
                        domain.bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
                            if ( space_func ) {
                                if ( num_dirac_0 < nd ) {
                                    cp.display_html_canvas( os_int[ num_thread ], ptr_weights[ num_dirac_0 ], 0 );
                                    cp.display_html_canvas( os_ext[ num_thread ], ptr_weights[ num_dirac_0 ], 1 );
                                }

                                cp.for_each_node( [&]( Pt v ) {
                                    min_pts[ num_thread ] = min( min_pts[ num_thread ], v );
                                    max_pts[ num_thread ] = max( max_pts[ num_thread ], v );
                                } );
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
            out << "var path_int = new Path2D();\n";
            for( std::ostringstream &o : os_int )
                out << o.str();

            out << "var path_ext = new Path2D();\n";
            for( std::ostringstream &o : os_ext )
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

            out << "pd_list.push({ path_int, path_ext, min_x, min_y, max_x, max_y, diracs, centroids });\n";
            out << "})();\n";
            return out.str();
        }

        void display_vtk_points( pybind11::array_t<PD_TYPE> &positions, const char *filename ) {
            sdot::VtkOutput<1> vtk_output( { "num" } );

            auto buf_positions = positions.request();
            auto ptr_positions = reinterpret_cast<const Pt *>( buf_positions.ptr );
            for( int n = 0; n < positions.shape( 0 ); ++n )
                vtk_output.add_point( ptr_positions[ n ], { TF( n ) } );

            vtk_output.save( filename );
        }

        template<class Domain,class Func>
        void display_asy( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, Domain &domain, const Func &radial_func, const char *filename, const char *preamble, pybind11::array_t<PD_TYPE> &values, std::string colormap, double linewidth, double dotwidth, bool avoid_bounds, const char *closing, double min_rf, double max_rf ) {
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
                find_radial_func( radial_func, [&]( const auto &ft ) {
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

        #define DEF_FOR( NAME, DOMAIN, FUNC ) \
            pybind11::array_t<PD_TYPE> integrals_##NAME( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, FUNC &func ) { \
                return get_integrals( positions, weights, domain.bounds, grid, func ); \
            } \
            pybind11::array_t<PD_TYPE> image_integrals_##NAME( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, const FUNC &func, pybind11::array_t<TF> &beg, pybind11::array_t<TF> &end, pybind11::array_t<std::size_t> &nb_pixels ) { \
                return get_image_integrals( positions, weights, domain.bounds, grid, func, beg, end, nb_pixels ); \
            } \
            pybind11::array_t<PD_TYPE> centroids_##NAME( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, const FUNC &func, TF rand_ratio ) { \
                return get_centroids( positions, weights, domain.bounds, grid, func, rand_ratio ); \
            } \
            PyDerResult<dim,TF> der_integrals_wrt_weights_##NAME( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, const FUNC &func, bool stop_if_void ) { \
                return der_integrals_wrt_weights( positions, weights, domain, func, stop_if_void ); \
            } \
            pybind11::array_t<TF> distances_from_boundaries_##NAME( pybind11::array_t<PD_TYPE> &points, pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, const FUNC &func, bool count_domain_boundaries ) { \
                return distances_from_boundaries( points, positions, weights, domain, func, count_domain_boundaries ); \
            } \
            void display_vtk_##NAME( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, const FUNC &func, const char *filename, bool points, bool centroids ) { \
                display_vtk( positions, weights, domain, func, filename, points, centroids ); \
            } \
            std::string display_html_canvas_##NAME( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, const FUNC &func, int hide_after ) { \
                return display_html_canvas( positions, weights, domain, func, hide_after ); \
            } \
            PyDerResult<dim,TF> der_centroids_and_integrals_wrt_weight_and_positions_##NAME( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, const FUNC &func ) { \
                return der_centroids_and_integrals_wrt_weight_and_positions( positions, weights, domain, func ); \
            } \
            void display_asy_##NAME( pybind11::array_t<PD_TYPE> &positions, pybind11::array_t<PD_TYPE> &weights, DOMAIN<dim,TF> &domain, const FUNC &radial_func, const char *filename, const char *preamble, pybind11::array_t<PD_TYPE> &values, std::string colormap, double linewidth, double dotwidth, bool avoid_bounds, const char *closing, double min_rf, double max_rf ) { \
                display_asy( positions, weights, domain, radial_func, filename, preamble, values, colormap, linewidth, dotwidth, avoid_bounds, closing, min_rf, max_rf ); \
            }
        #include "comb_types.h"
        #undef DEF_FOR

        Grid grid;
    };
}

PYBIND11_MODULE( PD_MODULE_NAME, m ) {
    m.doc() = "Semi-discrete optimal transportation";

    using DerResult = PyDerResult<PD_DIM,PD_TYPE>;
    pybind11::class_<DerResult>( m, "DerResult" )
        .def_readwrite( "m_offsets"                                 , &DerResult::m_offsets                                                                  , "" )
        .def_readwrite( "m_columns"                                 , &DerResult::m_columns                                                                  , "" )
        .def_readwrite( "m_values"                                  , &DerResult::m_values                                                                   , "" )
        .def_readwrite( "v_values"                                  , &DerResult::v_values                                                                   , "" )
        .def_readwrite( "error"                                     , &DerResult::error                                                                      , "" )
    ;

    using ConvexPolyhedraAssembly = PyConvexPolyhedraAssembly<PD_DIM,PD_TYPE>;
    pybind11::class_<ConvexPolyhedraAssembly>( m, "ConvexPolyhedraAssembly" )
        .def( pybind11::init<>()                                                                                                                             , "" )
        .def( "add_convex_polyhedron"                               , &ConvexPolyhedraAssembly::add_convex_polyhedron                                        , "" )
        .def( "add_box"                                             , &ConvexPolyhedraAssembly::add_box                                                      , "" )
        .def( "normalize"                                           , &ConvexPolyhedraAssembly::normalize                                                    , "" )
        .def( "display_boundaries_vtk"                              , &ConvexPolyhedraAssembly::display_boundaries_vtk                                       , "" )
        .def( "min_position"                                        , &ConvexPolyhedraAssembly::min_position                                                 , "" )
        .def( "max_position"                                        , &ConvexPolyhedraAssembly::max_position                                                 , "" )
        .def( "coeff_at"                                            , &ConvexPolyhedraAssembly::coeff_at                                                     , "" )
        .def( "measure"                                             , &ConvexPolyhedraAssembly::measure                                                      , "" )
    ;

    using ScaledImage = PyScaledImage<PD_DIM,PD_TYPE>;
    pybind11::class_<ScaledImage>( m, "ScaledImage" )
        .def( pybind11::init<pybind11::array_t<PD_TYPE> &, pybind11::array_t<PD_TYPE> &, pybind11::array_t<PD_TYPE> &>()                                     , "" )
        .def( "display_boundaries_vtk"                                      , &ScaledImage::display_boundaries_vtk                                           , "" )
        .def( "min_position"                                                , &ScaledImage::min_position                                                     , "" )
        .def( "max_position"                                                , &ScaledImage::max_position                                                     , "" )
        .def( "coeff_at"                                                    , &ScaledImage::coeff_at                                                         , "" )
        .def( "measure"                                                     , &ScaledImage::measure                                                          , "" )
    ;


    using PowerDiagramZGrid = PyPowerDiagramZGrid;
    pybind11::class_<PowerDiagramZGrid>( m, "PowerDiagramZGrid" )
        .def( pybind11::init<int>()                                                                                                                          , "" )
        .def( "update"                                                      , &PowerDiagramZGrid::update                                                     , "" )
        .def( "add_replication"                                             , &PowerDiagramZGrid::add_replication                                            , "" )
        #define DEF_FOR( NAME, DOMAIN, FUNC ) \
                .def( "integrals"                                           , &PowerDiagramZGrid::integrals_##NAME                                           , "" ) \
                .def( "image_integrals"                                     , &PowerDiagramZGrid::image_integrals_##NAME                                     , "" ) \
                .def( "der_integrals_wrt_weights"                           , &PowerDiagramZGrid::der_integrals_wrt_weights_##NAME                           , "" ) \
                .def( "distances_from_boundaries"                           , &PowerDiagramZGrid::distances_from_boundaries_##NAME                           , "" ) \
                .def( "centroids"                                           , &PowerDiagramZGrid::centroids_##NAME                                           , "" ) \
                .def( "display_vtk"                                         , &PowerDiagramZGrid::display_vtk_##NAME                                         , "" ) \
                .def( "display_html_canvas"                                 , &PowerDiagramZGrid::display_html_canvas_##NAME                                 , "" ) \
                .def( "der_centroids_and_integrals_wrt_weight_and_positions", &PowerDiagramZGrid::der_centroids_and_integrals_wrt_weight_and_positions_##NAME, "" ) \
                .def( "display_asy"                                         , &PowerDiagramZGrid::display_asy_##NAME                                         , "" )
        #include "comb_types.h"
        #undef DEF_FOR
        .def( "display_vtk_points"                                  , &PowerDiagramZGrid::display_vtk_points                                                 , "" )
    ;

}

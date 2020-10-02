// #include "../../ext/sdot/src/sdot/Visitors/internal/ZCoords.cpp"
#include "../../ext/sdot/src/sdot/Integration/Arfd.cpp"
#include "../../ext/sdot/src/sdot/Support/Assert.cpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
using TF = double;

//     std::size_t nb_polynomials() const { return approximations.size(); }

PYBIND11_MODULE( pybind_sdot_Arfd, m ) {
    using FFF = std::function<TF( TF )>;
    m.doc() = "Arfd";

    pybind11::class_<sdot::FunctionEnum::Arfd>( m, "Arfd" )
        .def( pybind11::init<FFF,FFF,FFF,FFF,FFF,FFF,std::vector<TF>,TF>(), "" )
        .def( "nb_polynomials", &sdot::FunctionEnum::Arfd::nb_polynomials );
}

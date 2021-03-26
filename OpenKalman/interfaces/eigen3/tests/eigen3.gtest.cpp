/* Created by Christopher Lee Ogden <ogden@gatech.edu> in 2018.
 * Any copyright as to this file is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

#include "eigen3.gtest.hpp"

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

using namespace OpenKalman;

using M = native_matrix_t<double, 3, 3>;
using M1 = native_matrix_t<double, 3, 1>;
using M3 = native_matrix_t<double, 3, 2>;
using M4 = native_matrix_t<double, 4, 2>;
template<typename Mat> using SAl = SelfAdjointMatrix<Mat, TriangleType::lower>;
template<typename Mat> using SAu = SelfAdjointMatrix<Mat, TriangleType::upper>;
template<typename Mat> using Tl = TriangularMatrix<Mat, TriangleType::lower>;
template<typename Mat> using Tu = TriangularMatrix<Mat, TriangleType::upper>;
template<typename Mat> using D = DiagonalMatrix<Mat>;
using C3 = Coefficients<Axis, angle::Radians, Axis>;
template<typename C, typename Mat> using To = ToEuclideanExpr<C, Mat>;
template<typename C, typename Mat> using From = FromEuclideanExpr<C, Mat>;


static_assert(modifiable<M, M>);
static_assert(modifiable<M, ZeroMatrix<double, 3, 3>>);
static_assert(modifiable<M, IdentityMatrix<M>>);
static_assert(not modifiable<M, M1>);
static_assert(not modifiable<M, native_matrix_t<int, 3, 3>>);
static_assert(not modifiable<const M, M>);
static_assert(not modifiable<M::ConstantReturnType, M>);
static_assert(not modifiable<M::IdentityReturnType, M>);
static_assert(not modifiable<decltype(-std::declval<M>()), M>);
static_assert(not modifiable<decltype(std::declval<M>() + std::declval<M>()), M>);
static_assert(not modifiable<M, int>);
static_assert(not modifiable<M::ConstantReturnType, M::ConstantReturnType>);
static_assert(not modifiable<M::IdentityReturnType, M::IdentityReturnType>);
static_assert(not modifiable<decltype(M::Identity() * 2), decltype(M::Identity() * 2)>);
static_assert(modifiable<Eigen::Block<M, 3, 1, true>, M1>);
static_assert(modifiable<Eigen::Block<M, 3, 1, true>, Eigen::Block<M, 3, 1, true>>);
static_assert(not modifiable<Eigen::Block<M, 3, 2, true>, M1>);
static_assert(not modifiable<Eigen::Block<M, 3, 2, true>, Eigen::Block<M, 3, 1, true>>);

static_assert(not OpenKalman::internal::has_const<SAl<M>>::value);
static_assert(OpenKalman::internal::has_const<SAl<const M>>::value);
static_assert(OpenKalman::internal::has_same_matrix_shape<SAl<M>, ZeroMatrix<double, 3, 3>>::value);
static_assert(not OpenKalman::internal::has_same_matrix_shape<SAl<M>, ZeroMatrix<double, 2, 2>>::value);

static_assert(modifiable<SAl<M>, ZeroMatrix<double, 3, 3>>);
static_assert(modifiable<SAu<M>, ZeroMatrix<double, 3, 3>>);
static_assert(modifiable<Tl<M>, ZeroMatrix<double, 3, 3>>);
static_assert(modifiable<Tu<M>, ZeroMatrix<double, 3, 3>>);
static_assert(modifiable<D<M1>, ZeroMatrix<double, 3, 3>>);
static_assert(modifiable<To<C3, M3>, M4>);
static_assert(modifiable<From<C3, M4>, M3>);

static_assert(modifiable<SAl<M>, IdentityMatrix<M>>);
static_assert(modifiable<SAu<M>, IdentityMatrix<M>>);
static_assert(modifiable<Tl<M>, IdentityMatrix<M>>);
static_assert(modifiable<Tu<M>, IdentityMatrix<M>>);
static_assert(modifiable<D<M1>, IdentityMatrix<M>>);

static_assert(modifiable<SAl<M>, D<M1>>);
static_assert(modifiable<SAu<M>, D<M1>>);
static_assert(modifiable<Tl<M>, D<M1>>);
static_assert(modifiable<Tu<M>, D<M1>>);
static_assert(modifiable<To<C3, M3>, M4>);
static_assert(modifiable<From<C3, M4>, M3>);

static_assert(not modifiable<SAl<M>, M>);
static_assert(not modifiable<SAu<M>, M>);
static_assert(not modifiable<Tl<M>, M>);
static_assert(not modifiable<Tu<M>, M>);
static_assert(not modifiable<D<M1>, M1>);
static_assert(not modifiable<To<C3, M3>, M3>);
static_assert(not modifiable<From<C3, M4>, M4>);

static_assert(modifiable<SAl<M>, SAl<M>>);
static_assert(modifiable<SAu<M>, SAu<M>>);
static_assert(modifiable<Tl<M>, Tl<M>>);
static_assert(modifiable<Tu<M>, Tu<M>>);
static_assert(modifiable<D<M1>, D<M1>>);
static_assert(modifiable<To<C3, M3>, To<C3, M3>>);
static_assert(modifiable<From<C3, M4>, From<C3, M4>>);
static_assert(not modifiable<SAl<M::ConstantReturnType>, SAl<M::ConstantReturnType>>);
static_assert(not modifiable<Tu<M::IdentityReturnType>, Tu<M::IdentityReturnType>>);
static_assert(not modifiable<SAu<decltype(M::Identity() * 2)>, SAu<decltype(M::Identity() * 2)>>);
static_assert(modifiable<To<C3, Eigen::Block<M3, 3, 1, true>>, To<C3, M1>>);

static_assert(modifiable<SAl<M>, const SAl<M>>);
static_assert(modifiable<SAu<M>, const SAu<M>>);
static_assert(modifiable<Tl<M>, const Tl<M>>);
static_assert(modifiable<Tu<M>, const Tu<M>>);
static_assert(modifiable<D<M1>, const D<M1>>);
static_assert(modifiable<To<C3, M3>, const To<C3, M3>>);
static_assert(modifiable<From<C3, M4>, const From<C3, M4>>);

static_assert(modifiable<SAl<M>, SAl<const M>>);
static_assert(modifiable<SAu<M>, SAu<const M>>);
static_assert(modifiable<Tl<M>, Tl<const M>>);
static_assert(modifiable<Tu<M>, Tu<const M>>);
static_assert(modifiable<D<M1>, D<const M1>>);
static_assert(modifiable<To<C3, M3>, To<C3, const M3>>);
static_assert(modifiable<From<C3, M4>, From<C3, const M4>>);

static_assert(not modifiable<SAl<const M>, SAl<M>>);
static_assert(not modifiable<SAu<const M>, SAu<M>>);
static_assert(not modifiable<Tl<const M>, Tl<M>>);
static_assert(not modifiable<Tu<const M>, Tu<M>>);
static_assert(not modifiable<D<const M1>, D<M1>>);
static_assert(not modifiable<To<C3, const M3>, To<C3, M3>>);
static_assert(not modifiable<From<C3, const M4>, From<C3, M4>>);
static_assert(not modifiable<SAl<decltype(M::Constant(9))>, M>);

static_assert(not modifiable<SAl<M>, Tl<M>>);
static_assert(not modifiable<SAu<M>, Tu<M>>);
static_assert(not modifiable<Tl<M>, SAl<M>>);
static_assert(not modifiable<Tu<M>, SAu<M>>);
static_assert(not modifiable<D<M1>, SAu<M>>);

static_assert(not modifiable<SAl<M>, SAl<native_matrix_t<double, 2, 2>>>);
static_assert(not modifiable<SAu<M>, SAu<native_matrix_t<double, 2, 2>>>);
static_assert(not modifiable<Tl<M>, Tl<native_matrix_t<double, 2, 2>>>);
static_assert(not modifiable<Tu<M>, Tu<native_matrix_t<double, 2, 2>>>);
static_assert(not modifiable<D<M1>, D<native_matrix_t<double, 2, 1>>>);
static_assert(not modifiable<To<Axes<3>, M3>, To<Axes<4>, M4>>);
static_assert(not modifiable<From<Axes<3>, M3>, From<Axes<4>, M4>>);

static_assert(modifiable<SAl<M>&, SAl<M>>);
static_assert(modifiable<SAu<M>&, SAu<M>>);
static_assert(modifiable<Tl<M>&, Tl<M>>);
static_assert(modifiable<Tu<M>&, Tu<M>>);
static_assert(modifiable<D<M1>&, D<M1>>);
static_assert(modifiable<To<C3, M3>&, To<C3, M3>>);
static_assert(modifiable<From<C3, M4>&, From<C3, M4>>);

static_assert(modifiable<SAl<M&>, SAl<M>>);
static_assert(modifiable<SAu<M&>, SAu<M>>);
static_assert(modifiable<Tl<M&>, Tl<M>>);
static_assert(modifiable<Tu<M&>, Tu<M>>);
static_assert(modifiable<D<M1&>, D<M1>>);
static_assert(modifiable<To<C3, M3&>, To<C3, M3>>);
static_assert(modifiable<From<C3, M4&>, From<C3, M4>>);

static_assert(not modifiable<SAl<M&>, M>);
static_assert(not modifiable<SAu<M&>, M>);
static_assert(not modifiable<Tl<M&>, M>);
static_assert(not modifiable<Tu<M&>, M>);
static_assert(not modifiable<D<M1&>, M1>);
static_assert(not modifiable<To<C3, M3&>, M3>);
static_assert(not modifiable<From<C3, M4&>, M4>);

static_assert(not modifiable<const SAl<M>&, SAl<M>>);
static_assert(not modifiable<const SAu<M>&, SAu<M>>);
static_assert(not modifiable<const Tl<M>&, Tl<M>>);
static_assert(not modifiable<const Tu<M>&, Tu<M>>);
static_assert(not modifiable<const D<M1>&, D<M1>>);
static_assert(not modifiable<const To<C3, M3>&, To<C3, M3>>);
static_assert(not modifiable<const From<C3, M4>&, From<C3, M4>>);

static_assert(not modifiable<SAl<const M&>, SAl<M>>);
static_assert(not modifiable<SAu<const M&>, SAu<M>>);
static_assert(not modifiable<Tl<const M&>, Tl<M>>);
static_assert(not modifiable<Tu<const M&>, Tu<M>>);
static_assert(not modifiable<D<const M1&>, D<M1>>);
static_assert(not modifiable<To<C3, const M3&>, To<C3, M3>>);
static_assert(not modifiable<From<C3, const M4&>, From<C3, M4>>);

static_assert(not modifiable<SAl<const M>&, SAl<M>>);
static_assert(not modifiable<SAu<const M>&, SAu<M>>);
static_assert(not modifiable<Tl<const M>&, Tl<M>>);
static_assert(not modifiable<Tu<const M>&, Tu<M>>);
static_assert(not modifiable<D<const M1>&, D<M1>>);
static_assert(not modifiable<To<C3, const M3>&, To<C3, M3>>);
static_assert(not modifiable<From<C3, const M4>&, From<C3, M4>>);


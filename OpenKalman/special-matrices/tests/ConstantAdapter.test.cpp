/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "special-matrices.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using numbers::pi;

namespace
{
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M12 = eigen_matrix_t<double, 1, 2>;
  using M13 = eigen_matrix_t<double, 1, 3>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M42 = eigen_matrix_t<double, 4, 2>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using M10 = eigen_matrix_t<double, 1, dynamic_size>;
  using M01 = eigen_matrix_t<double, dynamic_size, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_size>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_size>;
  using M03 = eigen_matrix_t<double, dynamic_size, 3>;
  using M04 = eigen_matrix_t<double, dynamic_size, 4>;
  using M40 = eigen_matrix_t<double, 4, dynamic_size>;
  using M50 = eigen_matrix_t<double, 5, dynamic_size>;
  using M05 = eigen_matrix_t<double, dynamic_size, 5>;

  using cdouble = std::complex<double>;

  using CM11 = eigen_matrix_t<cdouble, 1, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM33 = eigen_matrix_t<cdouble, 3, 3>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using CM02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using CM00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using Axis2 = TypedIndex<Axis, Axis>;

  using ZA11 = ZeroAdapter<M11>;
  using ZA10 = ZeroAdapter<M10>;
  using ZA01 = ZeroAdapter<M01>;
  using ZA00 = ZeroAdapter<M00>;

  using ZA21 = ZeroAdapter<M21>;
  using ZA12 = ZeroAdapter<M12>;
  using ZA22 = ZeroAdapter<M22>;
  using ZA23 = ZeroAdapter<M23>;
  using ZA20 = ZeroAdapter<M20>;
  using ZA02 = ZeroAdapter<M02>;
  using ZA03 = ZeroAdapter<M03>;

  using ZA13 = ZeroAdapter<M13>;
  using ZA31 = ZeroAdapter<M31>;
  using ZA33 = ZeroAdapter<M33>;
  using ZA30 = ZeroAdapter<M30>;

  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
}


TEST(eigen3, ConstantAdapter_traits)
{
  static_assert(indexible<ConstantAdapter<M22, double, 1>>);
  static_assert(indexible<ZA31>);

  static_assert(self_contained<ConstantAdapter<M22, double, 1>>);
  static_assert(self_contained<ConstantAdapter<M22, std::integral_constant<int, 1>>>);
  static_assert(self_contained<ZA31>);

  static_assert(constant_matrix<ConstantAdapter<M22, double, 1>, CompileTimeStatus::known>);
  static_assert(not constant_matrix<ConstantAdapter<M22, double, 1>, CompileTimeStatus::unknown>);
  static_assert(constant_matrix<ConstantAdapter<M22>, CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<ConstantAdapter<M22>, CompileTimeStatus::known>);

  static_assert(constant_diagonal_matrix<ZA33>);
  static_assert(not constant_diagonal_matrix<ZA31, Likelihood::maybe>);

  static_assert(zero_matrix<ConstantAdapter<M22, double, 0>>);
  static_assert(zero_matrix<ConstantAdapter<M00, double, 0>>);
  static_assert(not zero_matrix<ConstantAdapter<M22, double, 1>>);
  static_assert(zero_matrix<ZA31>);

  static_assert(diagonal_matrix<ConstantAdapter<M22, double, 0>>);
  static_assert(diagonal_matrix<ConstantAdapter<M11, double, 5>>);
  static_assert(diagonal_matrix<ConstantAdapter<M11>>);
  static_assert(not diagonal_matrix<ConstantAdapter<M00, double, 5>>);
  static_assert(not diagonal_matrix<ConstantAdapter<CM22, cdouble, 5>>);

  static_assert(diagonal_matrix<ZA33>);
  static_assert(not diagonal_adapter<ZA33>);
  static_assert(not diagonal_matrix<ZA31, Likelihood::maybe>);
  static_assert(not diagonal_matrix<ZA00>);
  static_assert(diagonal_matrix<ZA00, Likelihood::maybe>);

  static_assert(hermitian_matrix<ConstantAdapter<M22, double, 0>>);
  static_assert(hermitian_matrix<ConstantAdapter<M11, double, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<CM11, cdouble, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<M22, double, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<CM22, cdouble, 5>>);
  static_assert(not hermitian_matrix<ConstantAdapter<M34, double, 5>>);
  static_assert(not hermitian_matrix<ConstantAdapter<CM34, cdouble, 5>>);

  static_assert(hermitian_matrix<ZA33>);
  static_assert(hermitian_matrix<ZeroAdapter<CM33>>);
  static_assert(not hermitian_matrix<ZA31, Likelihood::maybe>);
  static_assert(not hermitian_matrix<ZA00>);
  static_assert(hermitian_matrix<ZA00, Likelihood::maybe>);

  static_assert(triangular_matrix<ConstantAdapter<M22, double, 0>>);
  static_assert(triangular_matrix<ConstantAdapter<M11, double, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M22, double, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M00, double, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M34, double, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M34, double, 0>>);

  static_assert(triangular_matrix<ZA33, TriangleType::upper>);
  static_assert(not triangular_matrix<ZA31, TriangleType::upper>);
  static_assert(not triangular_matrix<ZA00, TriangleType::upper>);
  static_assert(triangular_matrix<ZA00, TriangleType::upper, Likelihood::maybe>);

  static_assert(triangular_matrix<ZA33, TriangleType::lower>);
  static_assert(not triangular_matrix<ZA31, TriangleType::lower>);
  static_assert(not triangular_matrix<ZA00, TriangleType::lower>);
  static_assert(triangular_matrix<ZA00, TriangleType::lower, Likelihood::maybe>);

  static_assert(square_matrix<ConstantAdapter<M22, double, 0>, Likelihood::maybe>);
  static_assert(not square_matrix<ConstantAdapter<M34, double, 5>, Likelihood::maybe>);
  static_assert(square_matrix<ConstantAdapter<M30, double, 5>, Likelihood::maybe>);
  static_assert(square_matrix<ConstantAdapter<M00, double, 5>, Likelihood::maybe>);

  static_assert(square_matrix<ConstantAdapter<M22, double, 0>>);
  static_assert(square_matrix<ConstantAdapter<M22, double, 5>>);
  static_assert(not square_matrix<ConstantAdapter<M00, double, 5>>);
  static_assert(not square_matrix<ConstantAdapter<M34, double, 5>>);

  static_assert(not square_matrix<ZA31, Likelihood::maybe>);
  static_assert(square_matrix<ZA33>);
  static_assert(not square_matrix<ZA30>);
  static_assert(square_matrix<ZA30, Likelihood::maybe>);
  static_assert(not square_matrix<ZA03>);
  static_assert(square_matrix<ZA03, Likelihood::maybe>);
  static_assert(not square_matrix<ZA00>);
  static_assert(square_matrix<ZA00, Likelihood::maybe>);

  static_assert(one_by_one_matrix<ConstantAdapter<M11, double, 5>>);
  static_assert(one_by_one_matrix<ConstantAdapter<M10, double, 5>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<ConstantAdapter<M10, double, 5>>);
  static_assert(one_by_one_matrix<ConstantAdapter<M00, double, 5>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<ConstantAdapter<M00, double, 5>>);

  static_assert(not one_by_one_matrix<ZA31, Likelihood::maybe>);
  static_assert(one_by_one_matrix<ZA11>);
  static_assert(not one_by_one_matrix<ZA10>);
  static_assert(one_by_one_matrix<ZA10, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<ZA01>);
  static_assert(one_by_one_matrix<ZA01, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<ZA00>);
  static_assert(one_by_one_matrix<ZA00, Likelihood::maybe>);

  static_assert(element_gettable<ConstantAdapter<M22, double, 3>, 2>);
  static_assert(element_gettable<ConstantAdapter<M20, double, 3>, 2>);
  static_assert(element_gettable<ConstantAdapter<M02, double, 3>, 2>);
  static_assert(element_gettable<ConstantAdapter<M00, double, 3>, 2>);

  static_assert(element_gettable<ZA33, 2>);
  static_assert(element_gettable<ZA30, 2>);
  static_assert(element_gettable<ZA03, 2>);
  static_assert(element_gettable<ZA00, 2>);

  static_assert(not element_settable<ConstantAdapter<M22, double, 3>&, 2>);
  static_assert(not element_settable<ConstantAdapter<M20, double, 3>&, 2>);
  static_assert(not element_settable<ConstantAdapter<M02, double, 3>&, 2>);
  static_assert(not element_settable<ConstantAdapter<M20, double, 3>&, 2>);

  static_assert(not element_settable<ZA33&, 2>);
  static_assert(not element_settable<ZA30&, 2>);
  static_assert(not element_settable<ZA03&, 2>);
  static_assert(not element_settable<ZA00&, 2>);

  static_assert(dynamic_rows<ConstantAdapter<M00, double, 5>>);
  static_assert(dynamic_columns<ConstantAdapter<M00, double, 5>>);
  static_assert(dynamic_rows<ConstantAdapter<M01, double, 5>>);
  static_assert(not dynamic_columns<ConstantAdapter<M01, double, 5>>);
  static_assert(not dynamic_rows<ConstantAdapter<M10, double, 5>>);
  static_assert(dynamic_columns<ConstantAdapter<M10, double, 5>>);

  static_assert(dynamic_rows<ZA00>);
  static_assert(dynamic_rows<ZA03>);
  static_assert(not dynamic_rows<ZA30>);

  static_assert(dynamic_columns<ZA00>);
  static_assert(not dynamic_columns<ZA03>);
  static_assert(dynamic_columns<ZA30>);

  static_assert(not writable<ConstantAdapter<M33, double, 7>>);
  static_assert(not writable<ZA33>);

  static_assert(modifiable<M31, ConstantAdapter<M31, double, 7>>);
  static_assert(modifiable<M33, ZA33>);
}


TEST(eigen3, ConstantAdapter_class)
{
  ConstantAdapter<M23, double, 3> c323 {};
  ConstantAdapter<M20, double, 3> c320 {3};
  ConstantAdapter<M03, double, 3> c303 {2};
  ConstantAdapter<M00, double, 3> c300 {2,3};

  ZA23 z23;
  ZA20 z20 {3};
  ZA03 z03 {2};
  ZA00 z00 {2, 3};

  EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  EXPECT_TRUE(is_near(c300, M23::Constant(3)));

  EXPECT_TRUE(is_near(z23, M23::Zero()));
  EXPECT_TRUE(is_near(z20, M23::Zero()));
  EXPECT_TRUE(is_near(z03, M23::Zero()));
  EXPECT_TRUE(is_near(z00, M23::Zero()));

  EXPECT_TRUE(is_near(ConstantAdapter {z23}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantAdapter {z20}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantAdapter {z03}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantAdapter {z00}, M23::Zero()));

  EXPECT_TRUE(is_near(ConstantAdapter {ZA23 {N2{}, N3{}}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantAdapter {ZA20 {N2{}, 3}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantAdapter {ZA03 {2, N3{}}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantAdapter {ZA00 {2, 3}}, M23::Zero()));

  EXPECT_NEAR(std::real(ConstantAdapter<CM22, cdouble, 3> {}(0, 1)), 3, 1e-6);

  EXPECT_TRUE(is_near(ConstantAdapter {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {c300}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M23, double, 3> {N2{}, N3{}}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M20, double, 3> {N2{}, 3}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M03, double, 3> {2, N3{}}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M00, double, 3> {2, 3}}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M23, double, 2> {N2{}, N3{}}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M20, double, 2> {N2{}, 3}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M20, double, 2> {N2{}, N2{}, 3}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M03, double, 2> {2, N3{}}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M00, double, 2> {2, 3}}, M23::Constant(2)));

  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M23, double, 0>{}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M20, double, 0>{3}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M03, double, 0>{2}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M00, double, 0>{2,3}}(1, 2)), 0, 1e-6);

  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, double, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, double, 3> {c303}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {ConstantAdapter {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, double, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, double, 3> {ConstantAdapter {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, double, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, double, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, double, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, double, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, double, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, double, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, double, 3> {ConstantAdapter {c303}}, M23::Constant(3)));

  c320 = c323; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = c323; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = c323; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = c320; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c303 = c320; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = c320; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = c303; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = c303; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c300 = c303; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = c300; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = c300; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = c300; EXPECT_TRUE(is_near(c303, M23::Constant(3)));

  c323 = ConstantAdapter {c323}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantAdapter {c323}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantAdapter {c323}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantAdapter {c323}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantAdapter {c320}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantAdapter {c320}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantAdapter {c320}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantAdapter {c320}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantAdapter {c303}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantAdapter {c303}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantAdapter {c303}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantAdapter {c303}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantAdapter {c300}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantAdapter {c300}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantAdapter {c300}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantAdapter {c300}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));

  auto nc11 = M11::Identity() + M11::Identity() + M11::Identity(); using NC11 = decltype(nc11);
  auto nc23 = nc11.replicate<2,3>();
  auto nc20 = Eigen::Replicate<NC11, 2, Eigen::Dynamic> {nc11, 2, 3};
  auto nc03 = Eigen::Replicate<NC11, Eigen::Dynamic, 3> {nc11, 2, 3};
  auto nc00 = Eigen::Replicate<NC11, Eigen::Dynamic, Eigen::Dynamic> {nc11, 2, 3};

  c320 = nc23; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = nc23; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = nc23; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = nc20; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c303 = nc20; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = nc20; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = nc03; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = nc03; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c300 = nc03; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = nc00; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = nc00; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = nc00; EXPECT_TRUE(is_near(c303, M23::Constant(3)));

  c323 = ConstantAdapter {nc23}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantAdapter {nc23}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantAdapter {nc23}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantAdapter {nc23}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantAdapter {nc20}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantAdapter {nc20}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantAdapter {nc20}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantAdapter {nc20}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantAdapter {nc03}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantAdapter {nc03}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantAdapter {nc03}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantAdapter {nc03}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantAdapter {nc00}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantAdapter {nc00}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantAdapter {nc00}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantAdapter {nc00}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));

  EXPECT_NEAR((ConstantAdapter<M22, double, 3> {}(0, 0)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M22, double, 3> {}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M20, double, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M02, double, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M00, double, 3> {2,2}(0, 1)), 3, 1e-6);

  EXPECT_NEAR((ConstantAdapter<M31, double, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M01, double, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M13, double, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M10, double, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M31, double, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M01, double, 3> {3}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M13, double, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M10, double, 3> {3}[1]), 3, 1e-6);

  auto nz11 = M11::Identity() - M11::Identity(); using Z11 = decltype(nz11);
  auto nz23 = nz11.replicate<2,3>();
  auto nz20 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {nz11, 2, 3};
  auto nz03 = Eigen::Replicate<Z11, Eigen::Dynamic, 3> {nz11, 2, 3};
  auto nz00 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {nz11, 2, 3};

  z20 = nz23; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = nz23; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = nz23; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = nz20; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z03 = nz20; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = nz20; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = nz03; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = nz03; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z00 = nz03; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = nz00; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = nz00; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = nz00; EXPECT_TRUE(is_near(z03, M23::Zero()));

  z23 = ConstantAdapter {nz23}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = ConstantAdapter {nz23}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = ConstantAdapter {nz23}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = ConstantAdapter {nz23}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = ConstantAdapter {nz20}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = ConstantAdapter {nz20}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = ConstantAdapter {nz20}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = ConstantAdapter {nz20}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = ConstantAdapter {nz03}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = ConstantAdapter {nz03}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = ConstantAdapter {nz03}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = ConstantAdapter {nz03}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = ConstantAdapter {nz00}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = ConstantAdapter {nz00}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = ConstantAdapter {nz00}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = ConstantAdapter {nz00}; EXPECT_TRUE(is_near(z00, M23::Zero()));

  EXPECT_NEAR((ZA23 {}(0, 0)), 0, 1e-6);
  EXPECT_NEAR((ZA23 {}(0, 1)), 0, 1e-6);
  EXPECT_NEAR((ZA20 {2}(0, 1)), 0, 1e-6);
  EXPECT_NEAR((ZA03 {2}(0, 1)), 0, 1e-6);
  EXPECT_NEAR((ZA00 {2,2}(0, 1)), 0, 1e-6);

  EXPECT_NEAR((ZA31 {}(1)), 0, 1e-6);
  EXPECT_NEAR((ZA01 {3}(1)), 0, 1e-6);
  EXPECT_NEAR((ZA13 {}(1)), 0, 1e-6);
  EXPECT_NEAR((ZA10 {3}(1)), 0, 1e-6);
  EXPECT_NEAR((ZA31 {}[1]), 0, 1e-6);
  EXPECT_NEAR((ZA01 {3}[1]), 0, 1e-6);
  EXPECT_NEAR((ZA13 {}[1]), 0, 1e-6);
  EXPECT_NEAR((ZA10 {3}[1]), 0, 1e-6);
}


TEST(eigen3, make_dense_writable_matrix_from)
{
  ConstantAdapter<M34, double, 5> c534 {};
  ConstantAdapter<M30, double, 5> c530_4 {4};
  ConstantAdapter<M04, double, 5> c504_3 {3};
  ConstantAdapter<M00, double, 5> c500_34 {3, 4};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c534), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c530_4), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c504_3), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c500_34), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(ConstantAdapter<CM34, cdouble, 5> {}), CM34::Constant(cdouble(5,0))));
}


TEST(eigen3, make_constant_matrix_like)
{
  auto m23 = make_dense_writable_matrix_from<M23>(0, 0, 0, 0, 0, 0);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  ConstantAdapter<M34, double, 5> c534 {};
  ConstantAdapter<M30, double, 5> c530_4 {4};
  ConstantAdapter<M04, double, 5> c504_3 {3};
  ConstantAdapter<M00, double, 5> c500_34 {3, 4};

  using C534 = decltype(c534);

  constexpr internal::ScalarConstant<double, 5> nd5;

  EXPECT_TRUE(is_near(make_constant_matrix_like<M23>(nd5, Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00>(nd5, Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00>(nd5, 2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00>(nd5, 2, 3), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00>(nd5, Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00>(nd5, Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00>(nd5, 2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00>(nd5, 2, 3))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00>(nd5, Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00>(nd5, Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00>(nd5, 2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<M00>(nd5, 2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00>(nd5, 2, 3)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<M00>(nd5, 2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00>(nd5, Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00>(nd5, Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<M00>(nd5, 2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00>(nd5, 2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00>(nd5, 2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<M00>(nd5, 2, 3)), 3);

  EXPECT_TRUE(is_near(make_constant_matrix_like<M23>(5., Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00>(5., Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00>(5., 2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00>(5., 2, 3), M23::Constant(5)));

  EXPECT_TRUE(is_near(make_constant_matrix_like<M23>(nd5), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M23>(nd5))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M23>(nd5)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M23>(nd5)), 1> == 3);

  EXPECT_TRUE(is_near(make_constant_matrix_like<C534>(Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<C534>(Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<C534>(2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<C534>(2, 3), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534>(Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534>(Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534>(2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534>(2, 3))> == 5);

  EXPECT_TRUE(is_near(make_constant_matrix_like(m23, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like(m20_3, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like(m03_2, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like(m00_23, nd5), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like(m23, nd5))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like(m20_3, nd5))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like(m03_2, nd5))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like(m00_23, nd5))> == 5);

  EXPECT_TRUE(is_near(make_constant_matrix_like(m23, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like(m20_3, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like(m03_2, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like(m00_23, 5.), M23::Constant(5)));

  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, double, 5>(Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, double, 5>(Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, double, 5>(2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, double, 5>(2, 3), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, double, 5>(Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, double, 5>(Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, double, 5>(2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, double, 5>(2, 3))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, double, 5>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, double, 5>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, double, 5>(2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<M00, double, 5>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, double, 5>(2, 3)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<M00, double, 5>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, double, 5>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, double, 5>(Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<M00, double, 5>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, double, 5>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M00, double, 5>(2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<M00, double, 5>(2, 3)), 3);

  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, cdouble, 3, 4>(Dimensions<2>{}, Dimensions<3>{}), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, cdouble, 3, 4>(Dimensions<2>{}, 3), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, cdouble, 3, 4>(2, Dimensions<3>{}), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant_matrix_like<M00, cdouble, 3, 4>(2, 3), CM23::Constant(cdouble{3, 4})));

  EXPECT_TRUE(is_near(make_constant_matrix_like<M23, double, 5>(), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M23, double, 5>())> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M23, double, 5>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<M23, double, 5>()), 1> == 3);

  EXPECT_TRUE(is_near(make_constant_matrix_like<M23, cdouble, 3, 4>(), CM23::Constant(cdouble{3, 4})));

  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, 5>(2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<M00, 5>(2, 3))> == 5);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant_matrix_like<M00, 5>(Dimensions<2>(), 3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant_matrix_like<M00, 5>(2, Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant_matrix_like<M00, 5>(2, 3))>::value_type>);

  EXPECT_TRUE(is_near(make_constant_matrix_like<double, 5>(m23), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<double, 5>(m20_3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<double, 5>(m03_2), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant_matrix_like<double, 5>(m00_23), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<double, 5>(m23))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<double, 5>(m20_3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<double, 5>(m03_2))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<double, 5>(m00_23))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<double, 5>(m23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<double, 5>(m20_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<double, 5>(m03_2)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<double, 5>(m03_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<double, 5>(m00_23)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant_matrix_like<double, 5>(m00_23)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<double, 5>(m20_3)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<double, 5>(m20_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<double, 5>(m03_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<double, 5>(m00_23)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_constant_matrix_like<double, 5>(m00_23)), 3);

  EXPECT_TRUE(is_near(make_constant_matrix_like<cdouble, 3, 4>(m23), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant_matrix_like<cdouble, 3, 4>(m20_3), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant_matrix_like<cdouble, 3, 4>(m03_2), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant_matrix_like<cdouble, 3, 4>(m00_23), CM23::Constant(cdouble{3, 4})));

  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(m23))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(m20_3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(m03_2))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(m00_23))> == 5);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant_matrix_like<5>(m23))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant_matrix_like<5>(m20_3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant_matrix_like<5>(m03_2))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant_matrix_like<5>(m00_23))>::value_type>);
}


TEST(eigen3, make_zero_matrix_like)
{
  auto m23 = make_dense_writable_matrix_from<M23>(0, 0, 0, 0, 0, 0);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  EXPECT_TRUE(is_near(make_zero_matrix_like<M23>(Dimensions<2>{}, Dimensions<3>{}), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<M00>(Dimensions<2>{}, 3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<M00>(2, Dimensions<3>{}), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<M00>(2, 3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like(m23), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like(m20_3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like(m03_2), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like(m00_23), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<M23>(), M23::Zero()));

  static_assert(zero_matrix<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), Dimensions<3>()))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), 3))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like<M00>(2, Dimensions<3>()))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like<M00>(2, 3))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like(m23))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like(m20_3))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like(m03_2))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like(m00_23))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like<M23>())>);

  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero_matrix_like<M00>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(2, 3)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero_matrix_like<M00>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m20_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m03_2)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero_matrix_like(m03_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m00_23)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero_matrix_like(m00_23)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M23>()), 0> == 2);

  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_zero_matrix_like<M00>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M00>(2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_zero_matrix_like<M00>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m23)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m20_3)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_zero_matrix_like(m20_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m03_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(m00_23)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_zero_matrix_like(m00_23)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<M23>()), 1> == 3);

  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<M00, int>(Dimensions<2>(), Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<M00, int>(Dimensions<2>(), 3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<M00, int>(2, Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<M00, int>(2, 3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<int>(m23))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<int>(m20_3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<int>(m03_2))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<int>(m00_23))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero_matrix_like<M23, int>())>::value_type>);
}


TEST(eigen3, ConstantAdapter_equality)
{
  ConstantAdapter<M23, double, 3> ca23;
  ConstantAdapter<M20, double, 3> ca20 {3};
  ConstantAdapter<M03, double, 3> ca03 {2};
  ConstantAdapter<M00, double, 3> ca00 {2, 3};

  auto nc11 = M11::Identity() + M11::Identity() + M11::Identity(); using NC11 = decltype(nc11);
  auto nc23 = nc11.replicate<2,3>();
  auto nc20 = Eigen::Replicate<NC11, 2, Eigen::Dynamic> {nc11, 2, 3};
  auto nc03 = Eigen::Replicate<NC11, Eigen::Dynamic, 3> {nc11, 2, 3};
  auto nc00 = Eigen::Replicate<NC11, Eigen::Dynamic, Eigen::Dynamic> {nc11, 2, 3};

  auto mc23 = M23::Constant(3);
  auto mc20 = M20::Constant(2, 3, 3);
  auto mc03 = M03::Constant(2, 3, 3);
  auto mc00 = M00::Constant(2, 3, 3);

  static_assert(ca23 == ca23);
  static_assert(ca23 == nc23);
  static_assert(nc23 == ca23);

  EXPECT_TRUE(ca23 == nc20);
  EXPECT_TRUE(ca23 == nc03);
  EXPECT_TRUE(ca23 == nc00);
  EXPECT_TRUE(ca20 == nc20);
  EXPECT_TRUE(ca20 == nc03);
  EXPECT_TRUE(ca20 == nc00);
  EXPECT_TRUE(ca03 == nc20);
  EXPECT_TRUE(ca03 == nc03);
  EXPECT_TRUE(ca03 == nc00);
  EXPECT_TRUE(ca00 == nc20);
  EXPECT_TRUE(ca00 == nc03);
  EXPECT_TRUE(ca00 == nc00);

  EXPECT_TRUE(nc20 == ca23);
  EXPECT_TRUE(nc03 == ca23);
  EXPECT_TRUE(nc00 == ca23);
  EXPECT_TRUE(nc20 == ca20);
  EXPECT_TRUE(nc03 == ca20);
  EXPECT_TRUE(nc00 == ca20);
  EXPECT_TRUE(nc20 == ca03);
  EXPECT_TRUE(nc03 == ca03);
  EXPECT_TRUE(nc00 == ca03);
  EXPECT_TRUE(nc20 == ca00);
  EXPECT_TRUE(nc03 == ca00);
  EXPECT_TRUE(nc00 == ca00);

  EXPECT_TRUE(ca23 == mc23);
  EXPECT_TRUE(ca23 == mc20);
  EXPECT_TRUE(ca23 == mc03);
  EXPECT_TRUE(ca23 == mc00);
  EXPECT_TRUE(ca20 == mc20);
  EXPECT_TRUE(ca20 == mc23);
  EXPECT_TRUE(ca20 == mc03);
  EXPECT_TRUE(ca20 == mc00);
  EXPECT_TRUE(ca03 == mc20);
  EXPECT_TRUE(ca03 == mc23);
  EXPECT_TRUE(ca03 == mc03);
  EXPECT_TRUE(ca03 == mc00);
  EXPECT_TRUE(ca00 == mc20);
  EXPECT_TRUE(ca00 == mc23);
  EXPECT_TRUE(ca00 == mc03);
  EXPECT_TRUE(ca00 == mc00);

  EXPECT_TRUE(mc20 == ca23);
  EXPECT_TRUE(mc23 == ca23);
  EXPECT_TRUE(mc03 == ca23);
  EXPECT_TRUE(mc00 == ca23);
  EXPECT_TRUE(mc23 == ca20);
  EXPECT_TRUE(mc20 == ca20);
  EXPECT_TRUE(mc03 == ca20);
  EXPECT_TRUE(mc00 == ca20);
  EXPECT_TRUE(mc23 == ca03);
  EXPECT_TRUE(mc20 == ca03);
  EXPECT_TRUE(mc03 == ca03);
  EXPECT_TRUE(mc00 == ca03);
  EXPECT_TRUE(mc23 == ca00);
  EXPECT_TRUE(mc20 == ca00);
  EXPECT_TRUE(mc03 == ca00);
  EXPECT_TRUE(mc00 == ca00);

  ZeroAdapter<M23> za23;
  ZeroAdapter<M20> za20 {3};
  ZeroAdapter<M03> za03 {2};
  ZA00 za00 {2, 3};

  auto nz11 = M11::Identity() - M11::Identity(); using NZ11 = decltype(nz11);
  auto nz23 = nz11.replicate<2,3>();
  auto nz20 = Eigen::Replicate<NZ11, 2, Eigen::Dynamic> {nz11, 2, 3};
  auto nz03 = Eigen::Replicate<NZ11, Eigen::Dynamic, 3> {nz11, 2, 3};
  auto nz00 = Eigen::Replicate<NZ11, Eigen::Dynamic, Eigen::Dynamic> {nz11, 2, 3};

  auto mz23 = M23::Zero();
  auto mz20 = M20::Zero(2, 3);
  auto mz03 = M03::Zero(2, 3);
  auto mz00 = M00::Zero(2, 3);

  static_assert(za23 == za23);
  static_assert(za23 == nz23);
  static_assert(nz23 == za23);

  EXPECT_TRUE(za23 == nz20);
  EXPECT_TRUE(za23 == nz03);
  EXPECT_TRUE(za23 == nz00);
  EXPECT_TRUE(za20 == nz20);
  EXPECT_TRUE(za20 == nz03);
  EXPECT_TRUE(za20 == nz00);
  EXPECT_TRUE(za03 == nz20);
  EXPECT_TRUE(za03 == nz03);
  EXPECT_TRUE(za03 == nz00);
  EXPECT_TRUE(za00 == nz20);
  EXPECT_TRUE(za00 == nz03);
  EXPECT_TRUE(za00 == nz00);

  EXPECT_TRUE(nz20 == za23);
  EXPECT_TRUE(nz03 == za23);
  EXPECT_TRUE(nz00 == za23);
  EXPECT_TRUE(nz20 == za20);
  EXPECT_TRUE(nz03 == za20);
  EXPECT_TRUE(nz00 == za20);
  EXPECT_TRUE(nz20 == za03);
  EXPECT_TRUE(nz03 == za03);
  EXPECT_TRUE(nz00 == za03);
  EXPECT_TRUE(nz20 == za00);
  EXPECT_TRUE(nz03 == za00);
  EXPECT_TRUE(nz00 == za00);

  EXPECT_TRUE(za23 == mz23);
  EXPECT_TRUE(za23 == mz20);
  EXPECT_TRUE(za23 == mz03);
  EXPECT_TRUE(za23 == mz00);
  EXPECT_TRUE(za20 == mz20);
  EXPECT_TRUE(za20 == mz23);
  EXPECT_TRUE(za20 == mz03);
  EXPECT_TRUE(za20 == mz00);
  EXPECT_TRUE(za03 == mz20);
  EXPECT_TRUE(za03 == mz23);
  EXPECT_TRUE(za03 == mz03);
  EXPECT_TRUE(za03 == mz00);
  EXPECT_TRUE(za00 == mz20);
  EXPECT_TRUE(za00 == mz23);
  EXPECT_TRUE(za00 == mz03);
  EXPECT_TRUE(za00 == mz00);

  EXPECT_TRUE(mz20 == za23);
  EXPECT_TRUE(mz23 == za23);
  EXPECT_TRUE(mz03 == za23);
  EXPECT_TRUE(mz00 == za23);
  EXPECT_TRUE(mz23 == za20);
  EXPECT_TRUE(mz20 == za20);
  EXPECT_TRUE(mz03 == za20);
  EXPECT_TRUE(mz00 == za20);
  EXPECT_TRUE(mz23 == za03);
  EXPECT_TRUE(mz20 == za03);
  EXPECT_TRUE(mz03 == za03);
  EXPECT_TRUE(mz00 == za03);
  EXPECT_TRUE(mz23 == za00);
  EXPECT_TRUE(mz20 == za00);
  EXPECT_TRUE(mz03 == za00);
  EXPECT_TRUE(mz00 == za00);
}


TEST(eigen3, ConstantAdapter_arithmetic)
{
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} + ConstantAdapter<M22, double, 5> {}, ConstantAdapter<M22, double, 8> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} - ConstantAdapter<M22, double, 5> {}, ConstantAdapter<M22, double, -2> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {} * ConstantAdapter<M32, double, 5> {}, ConstantAdapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M34, double, 4> {} * ConstantAdapter<M42, double, 7> {}, ConstantAdapter<M32, double, 112> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 0> {} * 2.0, ConstantAdapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} * -2.0, ConstantAdapter<M22, double, -6> {}));
  EXPECT_TRUE(is_near(3.0 * ConstantAdapter<M22, double, 0> {}, ConstantAdapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(-3.0 * ConstantAdapter<M22, double, 3> {}, ConstantAdapter<M22, double, -9> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 0> {} / 2.0, ConstantAdapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 8> {} / -2.0, ConstantAdapter<M22, double, -4> {}));
  EXPECT_TRUE(is_near(-ConstantAdapter<M22, double, 7> {}, ConstantAdapter<M22, double, -7> {}));

  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} + M22::Constant(5), ConstantAdapter<M22, double, 8> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) + ConstantAdapter<M22, double, 3> {}, ConstantAdapter<M22, double, 8> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} - M22::Constant(5), ConstantAdapter<M22, double, -2> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) - ConstantAdapter<M22, double, 3> {}, ConstantAdapter<M22, double, 2> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {} * M32::Constant(5), ConstantAdapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M34, double, 4> {} * M42::Constant(7), ConstantAdapter<M32, double, 112> {}));
  EXPECT_TRUE(is_near(M23::Constant(3) * ConstantAdapter<M32, double, 5> {}, ConstantAdapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(M34::Constant(4) * ConstantAdapter<M42, double, 7> {}, ConstantAdapter<M32, double, 112> {}));

  EXPECT_EQ((ConstantAdapter<M43, double, 3>{}.rows()), 4);
  EXPECT_EQ((ConstantAdapter<M43, double, 3>{}.cols()), 3);

  ZA00 z00 {2, 2};

  auto m22y = make_eigen_matrix<double, 2, 2>(1, 2, 3, 4);
  EXPECT_TRUE(is_near(z00 + m22y, m22y, 1e-6));
  EXPECT_TRUE(is_near(m22y + z00, m22y, 1e-6));
  static_assert(zero_matrix<decltype(z00 + z00)>);
  EXPECT_TRUE(is_near(m22y - z00, m22y, 1e-6));
  EXPECT_TRUE(is_near(z00 - m22y, -m22y, 1e-6));
  EXPECT_TRUE(is_near(z00 - m22y.Identity(), -m22y.Identity(), 1e-6));
  //static_assert(diagonal_matrix<decltype(z00 - decltype(m22y)::Identity())>);
  static_assert(zero_matrix<decltype(z00 - z00)>);
  EXPECT_TRUE(is_near(z00 * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 * m22y, z00, 1e-6));
  EXPECT_TRUE(is_near(m22y * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 * 2, z00, 1e-6));
  EXPECT_TRUE(is_near(2 * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 / 2, z00, 1e-6));
  EXPECT_TRUE(is_near(-z00, z00, 1e-6));

  EXPECT_EQ((z00.rows()), 2);
  EXPECT_EQ((z00.cols()), 2);
}


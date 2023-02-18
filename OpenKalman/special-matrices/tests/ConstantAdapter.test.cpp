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
  static_assert(constant_matrix<ConstantAdapter<M22, 1>, Likelihood::definitely, CompileTimeStatus::known>);
  static_assert(not constant_matrix<ConstantAdapter<M22, 1>, Likelihood::definitely, CompileTimeStatus::unknown>);
  static_assert(constant_matrix<ConstantAdapter<M22>, Likelihood::definitely, CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<ConstantAdapter<M22>, Likelihood::definitely, CompileTimeStatus::known>);

  static_assert(indexible<ConstantAdapter<M22, 1>>);

  static_assert(zero_matrix<ConstantAdapter<M22, 0>>);
  static_assert(zero_matrix<ConstantAdapter<M00, 0>>);
  static_assert(not zero_matrix<ConstantAdapter<M22, 1>>);

  static_assert(diagonal_matrix<ConstantAdapter<M22, 0>>);
  static_assert(diagonal_matrix<ConstantAdapter<M11, 5>>);
  static_assert(diagonal_matrix<ConstantAdapter<M11>>);
  static_assert(not diagonal_matrix<ConstantAdapter<M00, 5>>);
  static_assert(not diagonal_matrix<ConstantAdapter<CM22, 5>>);

  static_assert(hermitian_matrix<ConstantAdapter<M22, 0>>);
  static_assert(hermitian_matrix<ConstantAdapter<M11, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<CM11, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<M22, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<CM22, 5>>);
  static_assert(not hermitian_matrix<ConstantAdapter<M34, 5>>);
  static_assert(not hermitian_matrix<ConstantAdapter<CM34, 5>>);

  static_assert(triangular_matrix<ConstantAdapter<M22, 0>>);
  static_assert(triangular_matrix<ConstantAdapter<M11, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M22, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M00, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M34, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M34, 0>>);

  static_assert(square_matrix<ConstantAdapter<M22, 0>, Likelihood::maybe>);
  static_assert(not square_matrix<ConstantAdapter<M34, 5>, Likelihood::maybe>);
  static_assert(square_matrix<ConstantAdapter<M30, 5>, Likelihood::maybe>);
  static_assert(square_matrix<ConstantAdapter<M00, 5>, Likelihood::maybe>);

  static_assert(square_matrix<ConstantAdapter<M22, 0>>);
  static_assert(square_matrix<ConstantAdapter<M22, 5>>);
  static_assert(not square_matrix<ConstantAdapter<M00, 5>>);
  static_assert(not square_matrix<ConstantAdapter<M34, 5>>);

  static_assert(one_by_one_matrix<ConstantAdapter<M11, 5>>);
  static_assert(one_by_one_matrix<ConstantAdapter<M10, 5>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<ConstantAdapter<M10, 5>>);
  static_assert(one_by_one_matrix<ConstantAdapter<M00, 5>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<ConstantAdapter<M00, 5>>);

  static_assert(element_gettable<ConstantAdapter<M22, 3>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantAdapter<M20, 3>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantAdapter<M02, 3>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantAdapter<M00, 3>, std::size_t, std::size_t>);

  static_assert(not element_settable<ConstantAdapter<M22, 3>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantAdapter<M20, 3>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantAdapter<M02, 3>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantAdapter<M20, 3>&, std::size_t, std::size_t>);

  static_assert(dynamic_rows<ConstantAdapter<M00, 5>>);
  static_assert(dynamic_columns<ConstantAdapter<M00, 5>>);
  static_assert(dynamic_rows<ConstantAdapter<M01, 5>>);
  static_assert(not dynamic_columns<ConstantAdapter<M01, 5>>);
  static_assert(not dynamic_rows<ConstantAdapter<M10, 5>>);
  static_assert(dynamic_columns<ConstantAdapter<M10, 5>>);

  static_assert(not writable<ConstantAdapter<M33, 7>>);
  static_assert(modifiable<M31, ConstantAdapter<M31, 7>>);
}


TEST(eigen3, ZeroAdapter_traits)
{
  static_assert(indexible<ZA31>);

  static_assert(zero_matrix<ZA31>);
  static_assert(constant_diagonal_matrix<ZA33>);
  static_assert(not constant_diagonal_matrix<ZA31, Likelihood::maybe>);
  static_assert(diagonal_matrix<ZA33>);
  static_assert(not diagonal_adapter<ZA33>);
  static_assert(not diagonal_matrix<ZA31, Likelihood::maybe>);
  static_assert(not diagonal_matrix<ZA00>);
  static_assert(diagonal_matrix<ZA00, Likelihood::maybe>);

  static_assert(hermitian_matrix<ZA33>);
  static_assert(hermitian_matrix<ZeroAdapter<CM33>>);
  static_assert(not hermitian_matrix<ZA31, Likelihood::maybe>);
  static_assert(not hermitian_matrix<ZA00>);
  static_assert(hermitian_matrix<ZA00, Likelihood::maybe>);

  static_assert(upper_triangular_matrix<ZA33>);
  static_assert(not upper_triangular_matrix<ZA31>);
  static_assert(not upper_triangular_matrix<ZA00>);
  static_assert(upper_triangular_matrix<ZA00, Likelihood::maybe>);

  static_assert(lower_triangular_matrix<ZA33>);
  static_assert(not lower_triangular_matrix<ZA31>);
  static_assert(not lower_triangular_matrix<ZA00>);
  static_assert(lower_triangular_matrix<ZA00, Likelihood::maybe>);

  static_assert(not square_matrix<ZA31, Likelihood::maybe>);
  static_assert(square_matrix<ZA33>);
  static_assert(not square_matrix<ZA30>);
  static_assert(square_matrix<ZA30, Likelihood::maybe>);
  static_assert(not square_matrix<ZA03>);
  static_assert(square_matrix<ZA03, Likelihood::maybe>);
  static_assert(not square_matrix<ZA00>);
  static_assert(square_matrix<ZA00, Likelihood::maybe>);

  static_assert(not one_by_one_matrix<ZA31, Likelihood::maybe>);
  static_assert(one_by_one_matrix<ZA11>);
  static_assert(not one_by_one_matrix<ZA10>);
  static_assert(one_by_one_matrix<ZA10, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<ZA01>);
  static_assert(one_by_one_matrix<ZA01, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<ZA00>);
  static_assert(one_by_one_matrix<ZA00, Likelihood::maybe>);

  static_assert(element_gettable<ZA33, std::size_t, std::size_t>);
  static_assert(element_gettable<ZA30, std::size_t, std::size_t>);
  static_assert(element_gettable<ZA03, std::size_t, std::size_t>);
  static_assert(element_gettable<ZA00, std::size_t, std::size_t>);

  static_assert(not element_settable<ZA33&, std::size_t, std::size_t>);
  static_assert(not element_settable<ZA30&, std::size_t, std::size_t>);
  static_assert(not element_settable<ZA03&, std::size_t, std::size_t>);
  static_assert(not element_settable<ZA00&, std::size_t, std::size_t>);

  static_assert(dynamic_rows<ZA00>);
  static_assert(dynamic_rows<ZA03>);
  static_assert(not dynamic_rows<ZA30>);

  static_assert(dynamic_columns<ZA00>);
  static_assert(not dynamic_columns<ZA03>);
  static_assert(dynamic_columns<ZA30>);

  static_assert(not writable<ZA33>);

  static_assert(modifiable<M33, ZA33>);
}


TEST(eigen3, ConstantAdapter_class)
{
  ConstantAdapter<M23, 3> c323 {};
  ConstantAdapter<M20, 3> c320 {3};
  ConstantAdapter<M03, 3> c303 {2};
  ConstantAdapter<M00, 3> c300 {2,3};

  EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  EXPECT_TRUE(is_near(c300, M23::Constant(3)));

  EXPECT_NEAR(std::real(ConstantAdapter<CM22, 3> {}(0, 1)), 3, 1e-6);

  EXPECT_TRUE(is_near(ConstantAdapter {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {c300}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M23, 3> {N2{}, N3{}}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M20, 3> {N2{}, 3}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M03, 3> {2, N3{}}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M00, 3> {2, 3}}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantAdapter<M23, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, 3> {c303}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantAdapter<M23, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, 3> {ConstantAdapter {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, 3> {ConstantAdapter {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M20, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M03, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M00, 3> {ConstantAdapter {c303}}, M23::Constant(3)));

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

  EXPECT_NEAR((ConstantAdapter<M22, 3> {}(0, 0)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M22, 3> {}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M20, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M02, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M00, 3> {2,2}(0, 1)), 3, 1e-6);

  EXPECT_NEAR((ConstantAdapter<M31, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M01, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M13, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M10, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M31, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M01, 3> {3}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M13, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M10, 3> {3}[1]), 3, 1e-6);
}


TEST(eigen3, ZeroAdapter_class)
{
  ZA23 z23;
  ZA20 z20 {3};
  ZA03 z03 {2};
  ZA00 z00 {2, 3};

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

  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M23, 0>{}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M20, 0>{3}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M03, 0>{2}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M00, 0>{2,3}}(1, 2)), 0, 1e-6);

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


TEST(eigen3, ConstantAdapter_make_functions)
{
  ConstantAdapter<M34, 5> c534 {};
  ConstantAdapter<M30, 5> c530_4 {4};
  ConstantAdapter<M04, 5> c504_3 {3};
  ConstantAdapter<M00, 5> c500_34 {3, 4};

  ConstantAdapter<M33, 5> c533 {};
  ConstantAdapter<M30, 5> c530_3 {3};
  ConstantAdapter<M03, 5> c503_3 {3};
  ConstantAdapter<M00, 5> c500_33 {3, 3};

  ConstantAdapter<M31, 5> c531 {};
  ConstantAdapter<M30, 5> c530_1 {1};
  ConstantAdapter<M01, 5> c501_3 {3};
  ConstantAdapter<M00, 5> c500_31 {3, 1};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c534), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c530_4), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c504_3), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c500_34), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(ConstantAdapter<CM34, 5> {}), CM34::Constant(cdouble(5,0))));

  using C533 = decltype(c533);
  using C534 = decltype(c534);
  using C500 = decltype(c500_34);

  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>())> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>(Dimensions<3>(), Dimensions<4>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>(Dimensions<3>(), 4))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>(3, Dimensions<4>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>(3, 4))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(c534))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(c530_4))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(c504_3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(c500_34))> == 5);

  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>(Dimensions<3>(), Dimensions<4>())), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>(Dimensions<3>(), 4)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>(3, Dimensions<4>())), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like<C500>(3, Dimensions<4>())), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>(3, 4)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like<C500>(3, 4)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(c534)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(c530_4)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(c504_3)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_constant_matrix_like(c504_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(c500_34)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_constant_matrix_like(c500_34)), 4);

  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>()), 1> == 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>(Dimensions<3>(), Dimensions<4>())), 1> == 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>(Dimensions<3>(), 4)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like<C500>(3, 4)), 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>(3, Dimensions<4>())), 1> == 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>(3, 4)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like<C500>(3, 4)), 4);

  static_assert(identity_matrix<decltype(make_identity_matrix_like<C533>())>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<C500>(Dimensions<3>()))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<C500>(3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c533))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c530_3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c503_3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c500_33))>);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C533>()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(Dimensions<3>())), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(3)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like<C500>(3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c533)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c530_3)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c503_3)), 0> == 3); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like(c503_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c500_33)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like(c500_33)), 3);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C533>()), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(3)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like<C500>(3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c533)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c530_3)), 1> == 3); EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like(c530_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c503_3)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c500_33)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like(c500_33)), 3);

  EXPECT_TRUE(is_near(make_self_contained(ConstantAdapter {c534}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantAdapter {c530_4}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantAdapter {c504_3}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantAdapter {c500_34}), M34::Constant(5)));
}


TEST(eigen3, ZeroAdapter_make_functions)
{
  ZA23 z23 {Dimensions<2>(), Dimensions<3>()};
  ZA20 z20_3 {Dimensions<2>(), 3};
  ZA03 z03_2 {2, Dimensions<3>()};
  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z20_2 {Dimensions<2>(), 2};
  ZA02 z02_2 {2, Dimensions<2>()};
  ZA00 z00_23 {2, 3};
  ZA00 z00_22 {2, 2};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(z23), M23::Zero()));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(z20_3), M23::Zero()));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(z03_2), M23::Zero()));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(z00_23), M23::Zero()));

  static_assert(zero_matrix<decltype(make_zero_matrix_like<ZA23>())>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like<ZA00>(Dimensions<2>(), Dimensions<3>()))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like<ZA00>(Dimensions<2>(), 3))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like<ZA00>(2, Dimensions<3>()))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like<ZA00>(2, 3))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like(z23))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like(z20_3))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like(z03_2))>);
  static_assert(zero_matrix<decltype(make_zero_matrix_like(z00_23))>);

  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA23>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA00>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA00>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA00>(2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like<ZA00>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA00>(2, 3)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like<ZA00>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z20_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z03_2)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like(z03_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z00_23)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like(z00_23)), 2);

  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA23>()), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA00>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA00>(Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like<ZA00>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA00>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like<ZA00>(2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like<ZA00>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z23)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z20_3)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like(z20_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z03_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero_matrix_like(z00_23)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like(z00_23)), 3);

  static_assert(identity_matrix<decltype(make_identity_matrix_like<ZA22>())>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<ZA00>(Dimensions<2>()))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<ZA00>(2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z22))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z20_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z02_2))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(z00_22))>);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA22>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA00>(Dimensions<2>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA00>(2)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like<ZA00>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z22)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z20_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z02_2)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z00_22)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like(z00_22)), 2);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA22>()), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA00>(Dimensions<2>())), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<ZA00>(2)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like<ZA00>(2)), 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z22)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z20_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z02_2)), 1> == 2);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(z00_22)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like(z00_22)), 2);

  EXPECT_TRUE(is_near(make_self_contained(ConstantAdapter {z23}), M23::Zero()));
  EXPECT_TRUE(is_near(make_self_contained(ConstantAdapter {z20_3}), M23::Zero()));
  EXPECT_TRUE(is_near(make_self_contained(ConstantAdapter {z03_2}), M23::Zero()));
  EXPECT_TRUE(is_near(make_self_contained(ConstantAdapter {z00_23}), M23::Zero()));
}


TEST(eigen3, ConstantAdapter_functions)
{
  ConstantAdapter<M34, 5> c534 {};
  ConstantAdapter<M30, 5> c530_4 {4};
  ConstantAdapter<M04, 5> c504_3 {3};
  ConstantAdapter<M00, 5> c500_34 {3, 4};

  ConstantAdapter<M33, 5> c533 {};
  ConstantAdapter<M30, 5> c530_3 {3};
  ConstantAdapter<M03, 5> c503_3 {3};
  ConstantAdapter<M00, 5> c500_33 {3, 3};

  ConstantAdapter<M31, 5> c531 {};
  ConstantAdapter<M30, 5> c530_1 {1};
  ConstantAdapter<M01, 5> c501_3 {3};
  ConstantAdapter<M00, 5> c500_31 {3, 1};

  EXPECT_EQ(get_dimensions_of<0>(c534), 3);
  EXPECT_EQ(get_dimensions_of<0>(c530_4), 3);
  EXPECT_EQ(get_dimensions_of<0>(c504_3), 3);
  EXPECT_EQ(get_dimensions_of<0>(c500_34), 3);

  EXPECT_EQ(get_dimensions_of<1>(c534), 4);
  EXPECT_EQ(get_dimensions_of<1>(c530_4), 4);
  EXPECT_EQ(get_dimensions_of<1>(c504_3), 4);
  EXPECT_EQ(get_dimensions_of<1>(c500_34), 4);

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  auto m33_d5 = make_dense_writable_matrix_from<M33>(5, 0, 0, 0, 5, 0, 0, 0, 5);

  EXPECT_TRUE(is_near(to_diagonal(c531), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c530_1), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c501_3), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c500_31), m33_d5));

  EXPECT_TRUE(is_near(diagonal_of(c533), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c530_3), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c503_3), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c500_33), M31::Constant(5)));
  static_assert(eigen_constant_expr<decltype(diagonal_of(c500_34))>);

  EXPECT_TRUE(is_near(transpose(c534), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c530_4), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c504_3), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c500_34), M43::Constant(5)));
  static_assert(eigen_constant_expr<decltype(transpose(c500_34))>);

  EXPECT_TRUE(is_near(adjoint(c534), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c530_4), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c504_3), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c500_34), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(ConstantAdapter<CM34, 5> {}), CM43::Constant(cdouble(5,0))));
  static_assert(eigen_constant_expr<decltype(adjoint(c500_34))>);

  EXPECT_NEAR(determinant(c533), 0, 1e-6);
  EXPECT_NEAR(determinant(c530_3), 0, 1e-6);
  EXPECT_NEAR(determinant(c530_3), 0, 1e-6);
  EXPECT_NEAR(determinant(c500_33), 0, 1e-6);

  EXPECT_NEAR(trace(c533), 15, 1e-6);
  EXPECT_NEAR(trace(c530_3), 15, 1e-6);
  EXPECT_NEAR(trace(c530_3), 15, 1e-6);
  EXPECT_NEAR(trace(c500_33), 15, 1e-6);

  // \todo rank_update

  M23 m23_66 = make_eigen_matrix<double, 2, 3>(6, 14, 22, 6, 14, 22);
  M20 m20_3_66 {2,3}; m20_3_66 = m23_66;
  M03 m03_2_66 {2,3}; m03_2_66 = m23_66;
  M00 m00_23_66 {2,3}; m00_23_66 = m23_66;
  auto m23_12 = make_eigen_matrix<double, 2, 3>(1.5, 3.5, 5.5, 1.5, 3.5, 5.5);

  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, m00_23_66), m23_12));

  ConstantAdapter<M23, 8> c23_8;
  ConstantAdapter<M20, 8> c20_3_8 {3};
  ConstantAdapter<M03, 8> c03_2_8 {2};
  ConstantAdapter<M00, 8> c00_23_8 {2, 3};
  ConstantAdapter<M23, 2> c23_2;

  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, c00_23_8), c23_2));

  ConstantAdapter<M23, 6> c23_6;
  ConstantAdapter<M20, 6> c20_3_6 {3};
  ConstantAdapter<M03, 6> c03_2_6 {2};
  ConstantAdapter<M00, 6> c00_23_6 {2, 3};
  auto m23_15 = make_eigen_matrix<double, 2, 3>(1.5, 1.5, 1.5, 1.5, 1.5, 1.5);

  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M22, 2> {}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M20, 2> {2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M02, 2> {2}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {2, 2}, c00_23_6), m23_15));

  EXPECT_TRUE(is_near(solve(ConstantAdapter<M11, 2> {}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M11, 2> {}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M11, 2> {}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M11, 2> {}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M10, 2> {1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M10, 2> {1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M10, 2> {1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M10, 2> {1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M01, 2> {1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M01, 2> {1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M01, 2> {1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M01, 2> {1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {1, 1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {1, 1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {1, 1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantAdapter<M00, 2> {1, 1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));

  EXPECT_TRUE(is_near(solve(M11::Identity(), make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(8)));

  auto colzc34 = average_reduce<1>(c500_34);
  EXPECT_TRUE(is_near(average_reduce<1>(ConstantAdapter<M23, 3> ()), (M21::Constant(3))));
  EXPECT_EQ(colzc34, (ConstantAdapter<M31, 5> {}));
  EXPECT_EQ(get_dimensions_of<0>(colzc34), 3);
  EXPECT_EQ(get_dimensions_of<1>(colzc34), 1);
  static_assert(eigen_constant_expr<decltype(colzc34)>);

  auto rowzc34 = average_reduce<0>(c500_34);
  EXPECT_TRUE(is_near(average_reduce<0>(ConstantAdapter<M23, 3> ()), (M13::Constant(3))));
  EXPECT_EQ(rowzc34, (ConstantAdapter<eigen_matrix_t<double, 1, 4>, 5> {}));
  EXPECT_EQ(get_dimensions_of<1>(rowzc34), 4);
  EXPECT_EQ(get_dimensions_of<0>(rowzc34), 1);
  static_assert(eigen_constant_expr<decltype(rowzc34)>);

  EXPECT_TRUE(is_near(LQ_decomposition(ConstantAdapter<eigen_matrix_t<double, 5, 3>, 7> ()), LQ_decomposition(make_eigen_matrix<double, 5, 3>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  auto lq332 = make_self_contained(LQ_decomposition(make_eigen_matrix<double, 3, 2>(3, 3, 3, 3, 3, 3)).cwiseAbs());
  EXPECT_TRUE(is_near(LQ_decomposition(ConstantAdapter<M32, 3> ()), lq332));
  auto lqzc30_2 = LQ_decomposition(ConstantAdapter<M30, 3> {2});
  EXPECT_TRUE(is_near(lqzc30_2, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc30_2), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc30_2), 3);
  auto lqzc02_3 = LQ_decomposition(ConstantAdapter<M02, 3> {3});
  EXPECT_TRUE(is_near(lqzc02_3, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc02_3), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc02_3), 3);
  auto lqzc00_32 = LQ_decomposition(ConstantAdapter<M00, 3> {3, 2});
  EXPECT_TRUE(is_near(lqzc00_32, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc00_32), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc00_32), 3);

  EXPECT_TRUE(is_near(QR_decomposition(ConstantAdapter<eigen_matrix_t<double, 3, 5>, 7> ()), QR_decomposition(make_eigen_matrix<double, 3, 5>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  auto qr323 = make_self_contained(QR_decomposition(make_eigen_matrix<double, 2, 3>(3, 3, 3, 3, 3, 3)).cwiseAbs());
  EXPECT_TRUE(is_near(QR_decomposition(ConstantAdapter<M23, 3> ()), qr323));
  auto qrzc20_3 = QR_decomposition(ConstantAdapter<M20, 3> {3});
  EXPECT_TRUE(is_near(qrzc20_3, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc20_3), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc20_3), 3);
  auto qrzc03_2 = QR_decomposition(ConstantAdapter<M03, 3> {2});
  EXPECT_TRUE(is_near(qrzc03_2, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc03_2), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc03_2), 3);
  auto qrzc00_23 = QR_decomposition(ConstantAdapter<M00, 3> {2, 3});
  EXPECT_TRUE(is_near(qrzc00_23, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc00_23), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc00_23), 3);

  // \todo concatenate_vertical
  // \todo concatenate_horizontal
  // \todo concatenate_vertical
  // \todo split_vertical
  // \todo split_horizontal
  // \todo vertical_vertical

  EXPECT_NEAR(get_element(ConstantAdapter<M22, 5> {}, 1, 0), 5, 1e-8);

  ConstantAdapter<M00, 5> c00 {2, 2};

  EXPECT_NEAR((get_element(c00, 0, 0)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 0, 1)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 1, 0)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 1, 1)), 5, 1e-6);
  EXPECT_NEAR((get_element(ConstantAdapter<M01, 7> {3}, 0)), 7, 1e-6);

  EXPECT_TRUE(is_near(get_chip<1>(ConstantAdapter<M23, 6> {}, 1), (M21::Constant(6))));
  EXPECT_TRUE(is_near(get_chip<1>(ConstantAdapter<M23, 7> {}, N1{}), (M21::Constant(7))));
  auto czc34 = get_chip<1>(c500_34, 1);
  EXPECT_EQ(get_dimensions_of<0>(czc34), 3);
  static_assert(get_dimensions_of<1>(czc34) == 1);
  static_assert(eigen_constant_expr<decltype(czc34)>);
  auto czv34 = get_chip<1>(ConstantAdapter<M04, 5> {3}, N1{});
  EXPECT_EQ(get_dimensions_of<0>(czv34), 3);
  static_assert(get_dimensions_of<1>(czv34) == 1);
  static_assert(eigen_constant_expr<decltype(czv34)>);

  EXPECT_TRUE(is_near(get_chip<0>(ConstantAdapter<M32, 6> {}, 1), (M12::Constant(6))));
  EXPECT_TRUE(is_near(get_chip<0>(ConstantAdapter<M32, 7> {}, N1{}), (M12::Constant(7))));
  auto rzc34 = get_chip<0>(c500_34, 1);
  EXPECT_EQ(get_dimensions_of<1>(rzc34), 4);
  static_assert(get_dimensions_of<0>(rzc34) == 1);
  static_assert(eigen_constant_expr<decltype(rzc34)>);
  auto rzv34 = get_chip<0>(ConstantAdapter<M30, 5> {4}, N1{});
  EXPECT_EQ(get_dimensions_of<1>(rzv34), 4);
  static_assert(get_dimensions_of<0>(rzv34) == 1);
  static_assert(eigen_constant_expr<decltype(rzv34)>);

  // \todo apply_columnwise
  // \todo apply_rowwise
  // \todo apply_coefficientwise
}


TEST(eigen3, ZeroAdapter_functions)
{
  ZA23 z23 {Dimensions<2>(), Dimensions<3>()};
  ZA20 z20_3 {Dimensions<2>(), 3};
  ZA03 z03_2 {2, Dimensions<3>()};
  ZA00 z00_23 {2, 3};

  ZA22 z22 {Dimensions<2>(), Dimensions<2>()};
  ZA20 z20_2 {Dimensions<2>(), 2};
  ZA02 z02_2 {2, Dimensions<2>()};
  ZA00 z00_22 {2, 2};

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  ZA21 z21;
  ZA20 z20_1 {1};
  ZA01 z01_2 {2};
  ZA00 z00_21 {2, 1};

  ZA12 z12;
  ZA10 z10_2 {2};
  ZA02 z02_1 {1};
  ZA00 z00_12 {1, 2};

  EXPECT_TRUE(is_near(to_diagonal(z21), z22));
  EXPECT_TRUE(is_near(to_diagonal(z20_1), z22));
  EXPECT_TRUE(is_near(to_diagonal(z01_2), z22));
  EXPECT_TRUE(is_near(to_diagonal(z00_21), z22));
  static_assert(diagonal_adapter<decltype(to_diagonal(z21))>);
  static_assert(diagonal_adapter<decltype(to_diagonal(z01_2))>);
  static_assert(diagonal_adapter<decltype(to_diagonal(z20_1)), Likelihood::maybe>);
  static_assert(diagonal_matrix<decltype(to_diagonal(z20_1))>);
  static_assert(diagonal_adapter<decltype(to_diagonal(z00_21)), Likelihood::maybe>);
  static_assert(diagonal_matrix<decltype(to_diagonal(z00_21))>);
  static_assert(zero_matrix<decltype(to_diagonal(z00_21))>);

  EXPECT_TRUE(is_near(diagonal_of(z22), z21));
  EXPECT_TRUE(is_near(diagonal_of(z20_2), z21));
  EXPECT_TRUE(is_near(diagonal_of(z02_2), z21));
  EXPECT_TRUE(is_near(diagonal_of(z00_22), z21));
  static_assert(zero_matrix<decltype(diagonal_of(z00_22))>);

  EXPECT_TRUE(is_near(transpose(z23), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z20_3), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z03_2), M32::Zero()));
  EXPECT_TRUE(is_near(transpose(z00_23), M32::Zero()));
  static_assert(zero_matrix<decltype(transpose(z00_23))>);

  EXPECT_TRUE(is_near(adjoint(z23), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z20_3), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z03_2), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(z00_23), M32::Zero()));
  EXPECT_TRUE(is_near(adjoint(ZeroAdapter<CM23> {}), M32::Zero()));
  static_assert(zero_matrix<decltype(adjoint(z00_23))>);

  EXPECT_NEAR(determinant(z22), 0, 1e-6);
  EXPECT_NEAR(determinant(z20_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z02_2), 0, 1e-6);
  EXPECT_NEAR(determinant(z00_22), 0, 1e-6);

  EXPECT_NEAR(trace(z22), 0, 1e-6);
  EXPECT_NEAR(trace(z20_2), 0, 1e-6);
  EXPECT_NEAR(trace(z02_2), 0, 1e-6);
  EXPECT_NEAR(trace(z00_22), 0, 1e-6);

  auto m1034 = make_eigen_matrix<double, 2, 2>(1, 0, 3, 4);
  auto m1034_2 = m1034 * adjoint(m1034);

  EXPECT_TRUE(is_near(rank_update_triangular(z22, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(z20_2, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(z02_2, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_triangular(z00_22, m1034, 0.25), 0.5*m1034));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z22, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z20_2, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z02_2, m1034, 0.25), 0.25*m1034_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z00_22, m1034, 0.25), 0.25*m1034_2));

  auto di5 = M22::Identity() * 5;
  auto di5_2 = di5 * di5;

  EXPECT_TRUE(is_near(rank_update_triangular(z22, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(z20_2, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(z02_2, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_triangular(z00_22, di5, 0.25), 0.5*di5));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z22, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z20_2, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z02_2, di5, 0.25), 0.25*di5_2));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(z00_22, di5, 0.25), 0.25*di5_2));

  EXPECT_TRUE(is_near(solve(z22, z23), z23));
  EXPECT_TRUE(is_near(solve(z22, z00_23), z23));
  EXPECT_TRUE(is_near(solve(z22, z20_3), z23));
  EXPECT_TRUE(is_near(solve(z22, z03_2), z23));
  EXPECT_TRUE(is_near(solve(z20_2, z23), z23));
  EXPECT_TRUE(is_near(solve(z20_2, z00_23), z23));
  EXPECT_TRUE(is_near(solve(z20_2, z20_3), z23));
  EXPECT_TRUE(is_near(solve(z20_2, z03_2), z23));
  EXPECT_TRUE(is_near(solve(z02_2, z23), z23));
  EXPECT_TRUE(is_near(solve(z02_2, z20_3), z23));
  EXPECT_TRUE(is_near(solve(z02_2, z03_2), z23));
  EXPECT_TRUE(is_near(solve(z02_2, z00_23), z23));
  EXPECT_TRUE(is_near(solve(z00_22, z23), z23));
  EXPECT_TRUE(is_near(solve(z00_22, z20_3), z23));
  EXPECT_TRUE(is_near(solve(z00_22, z03_2), z23));
  EXPECT_TRUE(is_near(solve(z00_22, z00_23), z23));

  EXPECT_TRUE(is_near(average_reduce<1>(z22), z21));
  EXPECT_TRUE(is_near(average_reduce<1>(z20_2), z21));
  EXPECT_TRUE(is_near(average_reduce<1>(z02_2), z21));
  EXPECT_TRUE(is_near(average_reduce<1>(z00_22), z21));
  static_assert(zero_matrix<decltype(average_reduce<1>(z00_22))>);

  EXPECT_TRUE(is_near(average_reduce<0>(z22), z12));
  EXPECT_TRUE(is_near(average_reduce<0>(z20_2), z12));
  EXPECT_TRUE(is_near(average_reduce<0>(z02_2), z12));
  EXPECT_TRUE(is_near(average_reduce<0>(z00_22), z12));
  static_assert(zero_matrix<decltype(average_reduce<0>(z00_22))>);

  EXPECT_TRUE(is_near(LQ_decomposition(z23), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z20_3), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z03_2), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z00_23), z22));
  static_assert(zero_matrix<decltype(LQ_decomposition(z00_23))>);

  ZeroAdapter<M32> z32;
  ZeroAdapter<M30> z30_2 {2};
  ZeroAdapter<M02> z02_3 {3};
  ZA00 z00_32 {3, 2};

  EXPECT_TRUE(is_near(QR_decomposition(z32), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z30_2), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z02_3), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z00_32), z22));
  static_assert(zero_matrix<decltype(QR_decomposition(z00_32))>);

  auto tup_z33_z23 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M23>()};

  EXPECT_TRUE(is_near(split<0>(make_zero_matrix_like<M00>(Dimensions<5>{}, Dimensions<3>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero_matrix_like<M00>(Dimensions<5>{}, 3), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero_matrix_like<M00>(5, Dimensions<3>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));
  EXPECT_TRUE(is_near(split<0>(make_zero_matrix_like<M00>(5, 3), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z23));

  auto tup_z33_z32 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M32>()};

  EXPECT_TRUE(is_near(split<1>(make_zero_matrix_like<M00>(Dimensions<3>{}, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero_matrix_like<M00>(Dimensions<3>{}, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero_matrix_like<M00>(3, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));
  EXPECT_TRUE(is_near(split<1>(make_zero_matrix_like<M00>(3, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z32));

  auto tup_z33_z22 = std::tuple {make_zero_matrix_like<M33>(), make_zero_matrix_like<M22>()};

  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M55>(Dimensions<5>{}, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M50>(Dimensions<5>{}, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M05>(5, Dimensions<5>{}), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M00>(5, 5), Dimensions<3>{}, Dimensions<2>{}), tup_z33_z22));

  auto tup_z32_z23 = std::tuple {make_zero_matrix_like<M32>(), make_zero_matrix_like<M23>()};

  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M55>(Dimensions<5>{}, Dimensions<5>{}), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M50>(Dimensions<5>{}, 5), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M05>(5, Dimensions<5>{}), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));
  EXPECT_TRUE(is_near(split<0, 1>(make_zero_matrix_like<M00>(5, 5), std::tuple{Dimensions<3>{}, Dimensions<2>{}}, std::tuple{Dimensions<2>{}, Dimensions<3>{}}), tup_z32_z23));

  EXPECT_NEAR(get_element(z22, 1, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z20_2, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z02_2, 0, 1), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_22, 1, 0), 0, 1e-6);

  EXPECT_NEAR(get_element(z21, 0, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z20_1, 1, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z01_2, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_21, 1, 0), 0, 1e-6);

  EXPECT_NEAR(get_element(z21, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z20_1, 1), 0, 1e-6);
  EXPECT_NEAR(get_element(z01_2, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_21, 1), 0, 1e-6);

  EXPECT_NEAR(get_element(z12, 0, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z10_2, 0, 1), 0, 1e-6);
  EXPECT_NEAR(get_element(z02_1, 0, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_12, 0, 1), 0, 1e-6);

  EXPECT_NEAR(get_element(z12, 0), 0, 1e-8);
  EXPECT_NEAR(get_element(z10_2, 1), 0, 1e-6);
  EXPECT_NEAR(get_element(z02_1, 0), 0, 1e-6);
  EXPECT_NEAR(get_element(z00_12, 1), 0, 1e-6);

  std::integral_constant<std::size_t, 1> N1;

  auto zc34 = ZeroAdapter<M34> {};

  EXPECT_TRUE(is_near(get_column(ZeroAdapter<M23>(), N1), (M21::Zero())));
  EXPECT_TRUE(is_near(get_column(ZeroAdapter<M23>(), 1), (M21::Zero())));
  auto czc34 = get_column(zc34, 1);
  EXPECT_EQ(get_dimensions_of<0>(czc34), 3);
  static_assert(get_dimensions_of<1>(czc34) == 1);
  static_assert(zero_matrix<decltype(czc34)>);
  auto czv34 = get_column(ZeroAdapter<M04> {3}, N1);
  EXPECT_EQ(get_dimensions_of<0>(czv34), 3);
  static_assert(get_dimensions_of<1>(czv34) == 1);
  static_assert(zero_matrix<decltype(czv34)>);

  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), N1), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M30>(Dimensions<3>{}, 3), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M03>(3, Dimensions<3>{}), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_column(make_zero_matrix_like<M00>(3,3), 2), make_dense_writable_matrix_from<M31>(0., 0, 0)));

  EXPECT_TRUE(is_near(get_row(ZeroAdapter<M32>(Dimensions<3>{}, Dimensions<2>{}), N1), (M12::Zero())));
  EXPECT_TRUE(is_near(get_row(ZeroAdapter<M32>(Dimensions<3>{}, Dimensions<2>{}), 1), (M12::Zero())));
  auto rzc34 = get_row(zc34, 1);
  EXPECT_EQ(get_dimensions_of<1>(rzc34), 4);
  static_assert(get_dimensions_of<0>(rzc34) == 1);
  static_assert(zero_matrix<decltype(rzc34)>);
  auto rzv34 = get_row(ZeroAdapter<M30> {Dimensions<3>{}, 4}, N1);
  EXPECT_EQ(get_dimensions_of<1>(rzv34), 4);
  static_assert(get_dimensions_of<0>(rzv34) == 1);
  static_assert(zero_matrix<decltype(rzv34)>);

  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), N1), make_dense_writable_matrix_from<M13>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{}), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M30>(Dimensions<3>{}, 3), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M03>(3, Dimensions<3>{}), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));
  EXPECT_TRUE(is_near(get_row(make_zero_matrix_like<M00>(3,3), 2), make_eigen_matrix<double, 1, 3>(0., 0, 0)));

  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + make_constant_matrix_like<1>(row)); }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ return make_self_contained(row + make_constant_matrix_like<1>(row)); }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));

  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + make_constant_matrix_like<1>(col)); }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ return make_self_contained(col + make_constant_matrix_like<1>(col)); }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero_matrix_like<M33>(Dimensions<3>{}, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero_matrix_like<M30>(Dimensions<3>{}, 3)), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero_matrix_like<M03>(3, Dimensions<3>{})), M33::Constant(1)));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x){ return x + 1; }, make_zero_matrix_like<M00>(3,3)), M33::Constant(1)));
}


TEST(eigen3, ConstantAdapter_equality)
{
  ConstantAdapter<M23, 3> ca23;
  ConstantAdapter<M20, 3> ca20 {3};
  ConstantAdapter<M03, 3> ca03 {2};
  ConstantAdapter<M00, 3> ca00 {2, 3};

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
}


TEST(eigen3, ZeroAdapter_equality)
{
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
  EXPECT_TRUE(is_near(ConstantAdapter<M22, 3> {} + ConstantAdapter<M22, 5> {}, ConstantAdapter<M22, 8> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, 3> {} - ConstantAdapter<M22, 5> {}, ConstantAdapter<M22, -2> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, 3> {} * ConstantAdapter<M32, 5> {}, ConstantAdapter<M22, 45> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M34, 4> {} * ConstantAdapter<M42, 7> {}, ConstantAdapter<M32, 112> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, 0> {} * 2.0, ConstantAdapter<M22, 0> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, 3> {} * -2.0, ConstantAdapter<M22, -6> {}));
  EXPECT_TRUE(is_near(3.0 * ConstantAdapter<M22, 0> {}, ConstantAdapter<M22, 0> {}));
  EXPECT_TRUE(is_near(-3.0 * ConstantAdapter<M22, 3> {}, ConstantAdapter<M22, -9> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, 0> {} / 2.0, ConstantAdapter<M22, 0> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, 8> {} / -2.0, ConstantAdapter<M22, -4> {}));
  EXPECT_TRUE(is_near(-ConstantAdapter<M22, 7> {}, ConstantAdapter<M22, -7> {}));

  EXPECT_TRUE(is_near(ConstantAdapter<M22, 3> {} + M22::Constant(5), ConstantAdapter<M22, 8> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) + ConstantAdapter<M22, 3> {}, ConstantAdapter<M22, 8> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, 3> {} - M22::Constant(5), ConstantAdapter<M22, -2> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) - ConstantAdapter<M22, 3> {}, ConstantAdapter<M22, 2> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, 3> {} * M32::Constant(5), ConstantAdapter<M22, 45> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M34, 4> {} * M42::Constant(7), ConstantAdapter<M32, 112> {}));
  EXPECT_TRUE(is_near(M23::Constant(3) * ConstantAdapter<M32, 5> {}, ConstantAdapter<M22, 45> {}));
  EXPECT_TRUE(is_near(M34::Constant(4) * ConstantAdapter<M42, 7> {}, ConstantAdapter<M32, 112> {}));

  EXPECT_EQ((ConstantAdapter<M43, 3>{}.rows()), 4);
  EXPECT_EQ((ConstantAdapter<M43, 3>{}.cols()), 3);
  EXPECT_TRUE(is_near(make_zero_matrix_like<ConstantAdapter<M23, 3>>(), M23::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<ConstantAdapter<M22, 3>>(), M22::Identity()));
}


TEST(eigen3, ZeroAdapter_arithmetic)
{
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
  EXPECT_TRUE(is_near(make_identity_matrix_like<ZeroAdapter<M22>>(), M22::Identity()));
}


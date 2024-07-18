/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "adapters.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using numbers::pi;

namespace
{
  using Axis2 = FixedDescriptor<Axis, Axis>;

  using ZA11 = ZeroAdapter<M11>;
  using ZA10 = ZeroAdapter<M1x>;
  using ZA01 = ZeroAdapter<Mx1>;
  using ZA00 = ZeroAdapter<Mxx>;

  using ZA21 = ZeroAdapter<M21>;
  using ZA12 = ZeroAdapter<M12>;
  using ZA22 = ZeroAdapter<M22>;
  using ZA23 = ZeroAdapter<M23>;
  using ZA20 = ZeroAdapter<M2x>;
  using ZA02 = ZeroAdapter<Mx2>;
  using ZA03 = ZeroAdapter<Mx3>;

  using ZA13 = ZeroAdapter<M13>;
  using ZA31 = ZeroAdapter<M31>;
  using ZA33 = ZeroAdapter<M33>;
  using ZA30 = ZeroAdapter<M3x>;

  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
}


TEST(adapters, ConstantAdapter_traits)
{
  static_assert(indexible<ConstantAdapter<M22, double, 1>>);
  static_assert(indexible<ZA31>);

  static_assert(self_contained<ConstantAdapter<M22, double, 1>>);
  static_assert(self_contained<ConstantAdapter<M22, std::integral_constant<int, 1>>>);
  static_assert(self_contained<ZA31>);

  static_assert(constant_matrix<ConstantAdapter<M22, double, 1>, ConstantType::static_constant>);
  static_assert(not constant_matrix<ConstantAdapter<M22, double, 1>, ConstantType::dynamic_constant>);
  static_assert(constant_matrix<ConstantAdapter<M22>, ConstantType::dynamic_constant>);
  static_assert(not constant_matrix<ConstantAdapter<M22>, ConstantType::static_constant>);

  static_assert(constant_diagonal_matrix<ZA33>);
  static_assert(constant_diagonal_matrix<ZA31>);
  static_assert(constant_diagonal_matrix<ZA13>);

  static_assert(zero<ConstantAdapter<M22, double, 0>>);
  static_assert(zero<ConstantAdapter<Mxx, double, 0>>);
  static_assert(not zero<ConstantAdapter<M22, double, 1>>);
  static_assert(zero<ZA31>);

  static_assert(diagonal_matrix<ConstantAdapter<M22, double, 0>>);
  static_assert(diagonal_matrix<ConstantAdapter<M11, double, 5>>);
  static_assert(diagonal_matrix<ConstantAdapter<M11>>);
  static_assert(not diagonal_matrix<ConstantAdapter<Mxx, double, 5>>);
  static_assert(not diagonal_matrix<ConstantAdapter<CM22, cdouble, 5>>);

  static_assert(diagonal_matrix<ZA33>);
  static_assert(not diagonal_adapter<ZA33>);
  static_assert(diagonal_matrix<ZA31>);
  static_assert(diagonal_matrix<ZA00>);

  static_assert(hermitian_matrix<ConstantAdapter<M22, double, 0>>);
  static_assert(hermitian_matrix<ConstantAdapter<M11, double, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<CM11, cdouble, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<M22, double, 5>>);
  static_assert(hermitian_matrix<ConstantAdapter<CM22, cdouble, 5>>);
  static_assert(not hermitian_matrix<ConstantAdapter<M34, double, 5>>);
  static_assert(not hermitian_matrix<ConstantAdapter<CM34, cdouble, 5>>);

  static_assert(hermitian_matrix<ZA33>);
  static_assert(hermitian_matrix<ZeroAdapter<CM33>>);
  static_assert(not hermitian_matrix<ZA31, Qualification::depends_on_dynamic_shape>);
  static_assert(not hermitian_matrix<ZA00>);
  static_assert(hermitian_matrix<ZA00, Qualification::depends_on_dynamic_shape>);

  static_assert(triangular_matrix<ConstantAdapter<M22, double, 0>>);
  static_assert(triangular_matrix<ConstantAdapter<M11, double, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M22, double, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<Mxx, double, 5>>);
  static_assert(not triangular_matrix<ConstantAdapter<M34, double, 5>>);
  static_assert(triangular_matrix<ConstantAdapter<M34, double, 0>>); // becaues it's a zero matrix and thus diagonal

  static_assert(triangular_matrix<ZA33, TriangleType::upper>);
  static_assert(triangular_matrix<ZA31, TriangleType::upper>);
  static_assert(triangular_matrix<ZA00, TriangleType::upper>);

  static_assert(triangular_matrix<ZA33, TriangleType::lower>);
  static_assert(triangular_matrix<ZA31, TriangleType::lower>);
  static_assert(triangular_matrix<ZA00, TriangleType::lower>);

  static_assert(square_shaped<ConstantAdapter<M22, double, 0>, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<ConstantAdapter<M34, double, 5>, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<ConstantAdapter<M3x, double, 5>, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<ConstantAdapter<Mxx, double, 5>, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<ConstantAdapter<M22, double, 0>>);
  static_assert(square_shaped<ConstantAdapter<M22, double, 5>>);
  static_assert(not square_shaped<ConstantAdapter<Mxx, double, 5>>);
  static_assert(not square_shaped<ConstantAdapter<M34, double, 5>>);

  static_assert(not square_shaped<ZA31, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<ZA33>);
  static_assert(not square_shaped<ZA30>);
  static_assert(square_shaped<ZA30, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<ZA03>);
  static_assert(square_shaped<ZA03, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<ZA00>);
  static_assert(square_shaped<ZA00, Qualification::depends_on_dynamic_shape>);

  static_assert(one_dimensional<ConstantAdapter<M11, double, 5>>);
  static_assert(one_dimensional<ConstantAdapter<M1x, double, 5>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<ConstantAdapter<M1x, double, 5>>);
  static_assert(one_dimensional<ConstantAdapter<Mxx, double, 5>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<ConstantAdapter<Mxx, double, 5>>);

  static_assert(not one_dimensional<ZA31, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<ZA11>);
  static_assert(not one_dimensional<ZA10>);
  static_assert(one_dimensional<ZA10, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<ZA01>);
  static_assert(one_dimensional<ZA01, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<ZA00>);
  static_assert(one_dimensional<ZA00, Qualification::depends_on_dynamic_shape>);

  static_assert(element_gettable<ConstantAdapter<M22, double, 3>, 2>);
  static_assert(element_gettable<ConstantAdapter<M2x, double, 3>, 2>);
  static_assert(element_gettable<ConstantAdapter<Mx2, double, 3>, 2>);
  static_assert(element_gettable<ConstantAdapter<Mxx, double, 3>, 2>);

  static_assert(element_gettable<ZA33, 2>);
  static_assert(element_gettable<ZA30, 2>);
  static_assert(element_gettable<ZA03, 2>);
  static_assert(element_gettable<ZA00, 2>);

  static_assert(not writable_by_component<ConstantAdapter<M22, double, 3>&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ConstantAdapter<M2x, double, 3>&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ConstantAdapter<Mx2, double, 3>&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ConstantAdapter<M2x, double, 3>&, std::array<std::size_t, 2>>);

  static_assert(not writable_by_component<ZA33&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ZA30&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ZA03&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ZA00&, std::array<std::size_t, 2>>);

  static_assert(dynamic_dimension<ConstantAdapter<Mxx, double, 5>, 0>);
  static_assert(dynamic_dimension<ConstantAdapter<Mxx, double, 5>, 1>);
  static_assert(dynamic_dimension<ConstantAdapter<Mx1, double, 5>, 0>);
  static_assert(not dynamic_dimension<ConstantAdapter<Mx1, double, 5>, 1>);
  static_assert(not dynamic_dimension<ConstantAdapter<M1x, double, 5>, 0>);
  static_assert(dynamic_dimension<ConstantAdapter<M1x, double, 5>, 1>);

  static_assert(dynamic_dimension<ZA00, 0>);
  static_assert(dynamic_dimension<ZA03, 0>);
  static_assert(not dynamic_dimension<ZA30, 0>);

  static_assert(dynamic_dimension<ZA00, 1>);
  static_assert(not dynamic_dimension<ZA03, 1>);
  static_assert(dynamic_dimension<ZA30, 1>);

  static_assert(not writable<ConstantAdapter<M33, double, 7>>);
  static_assert(not writable<ZA33>);

  static_assert(modifiable<M31, ConstantAdapter<M31, double, 7>>);
  static_assert(modifiable<M33, ZA33>);
}


TEST(adapters, ConstantAdapter_class)
{
  ConstantAdapter<M23, double, 3> c323 {};
  ConstantAdapter<M2x, double, 3> c320 {3};
  ConstantAdapter<Mx3, double, 3> c303 {2};
  ConstantAdapter<Mxx, double, 3> c300 {2,3};

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
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M2x, double, 3> {N2{}, 3}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<Mx3, double, 3> {2, N3{}}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<Mxx, double, 3> {2, 3}}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M23, double, 2> {N2{}, N3{}}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M2x, double, 2> {N2{}, 3}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<M2x, double, 2> {N2{}, N2{}, 3}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<Mx3, double, 2> {2, N3{}}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(ConstantAdapter {ConstantAdapter<Mxx, double, 2> {2, 3}}, M23::Constant(2)));

  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M23, double, 0>{}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<M2x, double, 0>{3}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<Mx3, double, 0>{2}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ConstantAdapter {ConstantAdapter<Mxx, double, 0>{2,3}}(1, 2)), 0, 1e-6);

  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M2x, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M2x, double, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M2x, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mx3, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mx3, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mx3, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mxx, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mxx, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mxx, double, 3> {c303}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {ConstantAdapter {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M2x, double, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M2x, double, 3> {ConstantAdapter {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<M2x, double, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mx3, double, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mx3, double, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mx3, double, 3> {ConstantAdapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mxx, double, 3> {ConstantAdapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mxx, double, 3> {ConstantAdapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantAdapter<Mxx, double, 3> {ConstantAdapter {c303}}, M23::Constant(3)));

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
  EXPECT_NEAR((ConstantAdapter<M2x, double, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<Mx2, double, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<Mxx, double, 3> {2,2}(0, 1)), 3, 1e-6);

  EXPECT_NEAR((ConstantAdapter<M31, double, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<Mx1, double, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M13, double, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M1x, double, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M31, double, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<Mx1, double, 3> {3}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M13, double, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantAdapter<M1x, double, 3> {3}[1]), 3, 1e-6);

  auto nz11 = M11::Identity() - M11::Identity(); using Z11e = decltype(nz11);
  auto nz23 = nz11.replicate<2,3>();
  auto nz20 = Eigen::Replicate<Z11e, 2, Eigen::Dynamic> {nz11, 2, 3};
  auto nz03 = Eigen::Replicate<Z11e, Eigen::Dynamic, 3> {nz11, 2, 3};
  auto nz00 = Eigen::Replicate<Z11e, Eigen::Dynamic, Eigen::Dynamic> {nz11, 2, 3};

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


TEST(adapters, make_dense_object_from)
{
  ConstantAdapter<M34, double, 5> c534 {};
  ConstantAdapter<M3x, double, 5> c530_4 {4};
  ConstantAdapter<Mx4, double, 5> c504_3 {3};
  ConstantAdapter<Mxx, double, 5> c500_34 {3, 4};

  EXPECT_TRUE(is_near(to_dense_object(c534), M34::Constant(5)));
  EXPECT_TRUE(is_near(to_dense_object(c530_4), M34::Constant(5)));
  EXPECT_TRUE(is_near(to_dense_object(c504_3), M34::Constant(5)));
  EXPECT_TRUE(is_near(to_dense_object(c500_34), M34::Constant(5)));
  EXPECT_TRUE(is_near(to_dense_object(ConstantAdapter<CM34, cdouble, 5> {}), CM34::Constant(cdouble(5,0))));
}


TEST(adapters, make_constant)
{
  auto m23 = make_dense_object_from<M23>(0, 0, 0, 0, 0, 0);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  ConstantAdapter<M34, double, 5> c534 {};
  ConstantAdapter<M3x, double, 5> c530_4 {4};
  ConstantAdapter<Mx4, double, 5> c504_3 {3};
  ConstantAdapter<Mxx, double, 5> c500_34 {3, 4};

  using C534 = decltype(c534);

  constexpr values::ScalarConstant<double, 5> nd5;

  EXPECT_TRUE(is_near(make_constant<M23>(nd5, Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(nd5, Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(nd5, 2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(nd5, 2, 3), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx>(nd5, 2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx>(nd5, 2, 3))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, 2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant<Mxx>(nd5, 2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, 2, 3)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant<Mxx>(nd5, 2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant<Mxx>(nd5, 2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, 2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, 2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant<Mxx>(nd5, 2, 3)), 3);

  EXPECT_TRUE(is_near(make_constant<M23>(5., Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(5., Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(5., 2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(5., 2, 3), M23::Constant(5)));

  EXPECT_TRUE(is_near(make_constant<M23>(nd5), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant<M23>(nd5))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<M23>(nd5)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<M23>(nd5)), 1> == 3);

  EXPECT_TRUE(is_near(make_constant<C534>(5., Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<C534>(5., Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<C534>(5., 2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<C534>(5., 2, 3), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant<C534>(values::ScalarConstant<double, 5>{}, Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<C534>(values::ScalarConstant<double, 5>{}, Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<C534>(values::ScalarConstant<double, 5>{}, 2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<C534>(values::ScalarConstant<double, 5>{}, 2, 3))> == 5);

  EXPECT_TRUE(is_near(make_constant(m23, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(m2x_3, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(mx3_2, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(mxx_23, nd5), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant(m23, nd5))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant(m2x_3, nd5))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant(mx3_2, nd5))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant(mxx_23, nd5))> == 5);

  EXPECT_TRUE(is_near(make_constant(m23, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(m2x_3, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(mx3_2, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(mxx_23, 5.), M23::Constant(5)));

  EXPECT_TRUE(is_near(make_constant<Mxx, double, 5>(Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx, double, 5>(Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx, double, 5>(2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx, double, 5>(2, 3), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx, double, 5>(2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx, double, 5>(2, 3))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant<Mxx, double, 5>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(2, 3)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant<Mxx, double, 5>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant<Mxx, double, 5>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_constant<Mxx, double, 5>(2, 3)), 3);

  EXPECT_TRUE(is_near(make_constant<Mxx, cdouble, 3, 4>(Dimensions<2>{}, Dimensions<3>{}), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<Mxx, cdouble, 3, 4>(Dimensions<2>{}, 3), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<Mxx, cdouble, 3, 4>(2, Dimensions<3>{}), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<Mxx, cdouble, 3, 4>(2, 3), CM23::Constant(cdouble{3, 4})));

  EXPECT_TRUE(is_near(make_constant<M23, double, 5>(), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant<M23, double, 5>())> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<M23, double, 5>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<M23, double, 5>()), 1> == 3);

  EXPECT_TRUE(is_near(make_constant<M23, cdouble, 3, 4>(), CM23::Constant(cdouble{3, 4})));

  static_assert(constant_coefficient_v<decltype(make_constant<Mxx, 5>(Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx, 5>(Dimensions<2>(), 3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx, 5>(2, Dimensions<3>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<Mxx, 5>(2, 3))> == 5);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant<Mxx, 5>(Dimensions<2>(), Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant<Mxx, 5>(Dimensions<2>(), 3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant<Mxx, 5>(2, Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant<Mxx, 5>(2, 3))>::value_type>);

  EXPECT_TRUE(is_near(make_constant<double, 5>(m23), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<double, 5>(m2x_3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<double, 5>(mx3_2), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<double, 5>(mxx_23), M23::Constant(5)));
  static_assert(constant_coefficient_v<decltype(make_constant<double, 5>(m23))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<double, 5>(m2x_3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<double, 5>(mx3_2))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<double, 5>(mxx_23))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(m23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(m2x_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(mx3_2)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant<double, 5>(mx3_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(mxx_23)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_constant<double, 5>(mxx_23)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(m2x_3)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_constant<double, 5>(m2x_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(mx3_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(mxx_23)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_constant<double, 5>(mxx_23)), 3);

  EXPECT_TRUE(is_near(make_constant<cdouble, 3, 4>(m23), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<cdouble, 3, 4>(m2x_3), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<cdouble, 3, 4>(mx3_2), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<cdouble, 3, 4>(mxx_23), CM23::Constant(cdouble{3, 4})));

  static_assert(constant_coefficient_v<decltype(make_constant<5>(m23))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<5>(m2x_3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<5>(mx3_2))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant<5>(mxx_23))> == 5);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant<5>(m23))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant<5>(m2x_3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant<5>(mx3_2))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_constant<5>(mxx_23))>::value_type>);
}


TEST(adapters, make_zero)
{
  auto m23 = make_dense_object_from<M23>(0, 0, 0, 0, 0, 0);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  EXPECT_TRUE(is_near(make_zero<M23>(Dimensions<2>{}, Dimensions<3>{}), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero<Mxx>(Dimensions<2>{}, 3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero<Mxx>(2, Dimensions<3>{}), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero<Mxx>(2, 3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero(m23), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero(m2x_3), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero(mx3_2), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero(mxx_23), M23::Zero()));
  EXPECT_TRUE(is_near(make_zero<M23>(), M23::Zero()));

  static_assert(zero<decltype(make_zero<Mxx>(Dimensions<2>(), Dimensions<3>()))>);
  static_assert(zero<decltype(make_zero<Mxx>(Dimensions<2>(), 3))>);
  static_assert(zero<decltype(make_zero<Mxx>(2, Dimensions<3>()))>);
  static_assert(zero<decltype(make_zero<Mxx>(2, 3))>);
  static_assert(zero<decltype(make_zero(m23))>);
  static_assert(zero<decltype(make_zero(m2x_3))>);
  static_assert(zero<decltype(make_zero(mx3_2))>);
  static_assert(zero<decltype(make_zero(mxx_23))>);
  static_assert(zero<decltype(make_zero<M23>())>);

  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(2, Dimensions<3>())), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero<Mxx>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(2, 3)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero<Mxx>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero(m23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero(m2x_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero(mx3_2)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero(mx3_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero(mxx_23)), 0> == dynamic_size); EXPECT_EQ(get_index_dimension_of<0>(make_zero(mxx_23)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero<M23>()), 0> == 2);

  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(Dimensions<2>(), 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_zero<Mxx>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(2, 3)), 1> == dynamic_size);  EXPECT_EQ(get_index_dimension_of<1>(make_zero<Mxx>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero(m23)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero(m2x_3)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_zero(m2x_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero(mx3_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero(mxx_23)), 1> == dynamic_size); EXPECT_EQ(get_index_dimension_of<1>(make_zero(mxx_23)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero<M23>()), 1> == 3);

  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<Mxx, int>(Dimensions<2>(), Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<Mxx, int>(Dimensions<2>(), 3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<Mxx, int>(2, Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<Mxx, int>(2, 3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<int>(m23))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<int>(m2x_3))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<int>(mx3_2))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<int>(mxx_23))>::value_type>);
  static_assert(std::is_integral_v<constant_coefficient<decltype(make_zero<M23, int>())>::value_type>);
}


TEST(adapters, diagonal_of_constant)
{
  // Note: ConstantAdapter is only created when the constant is known at compile time.
  // dynamic one-by-one, known at compile time:

  static_assert(one_dimensional<decltype(diagonal_of(M11::Identity()))>);
  static_assert(not one_dimensional<decltype(diagonal_of(M1x::Identity(1, 1)))>);
  static_assert(not one_dimensional<decltype(diagonal_of(Mx1::Identity(1, 1)))>);
  static_assert(not one_dimensional<decltype(diagonal_of(Mxx::Identity(1, 1)))>);
  static_assert(one_dimensional<decltype(diagonal_of(M1x::Identity(1, 1))), Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<decltype(diagonal_of(Mx1::Identity(1, 1))), Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<decltype(diagonal_of(Mxx::Identity(1, 1))), Qualification::depends_on_dynamic_shape>);

  static_assert(not has_dynamic_dimensions<decltype(diagonal_of(M11::Identity()))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(M1x::Identity(1, 1)))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(Mx1::Identity(1, 1)))>);

  static_assert(dynamic_dimension<decltype(diagonal_of(Mxx::Identity(1, 1))), 0>);
  static_assert(dimension_size_of_index_is<decltype(diagonal_of(Mxx::Identity(1, 1))), 1, 1>);

  static_assert(constant_coefficient_v<decltype(diagonal_of(M11::Identity()))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(M1x::Identity()))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(Mx1::Identity(1, 1)))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(Mxx::Identity(1, 1)))> == 1);

  auto i22 = M22::Identity();
  auto i2x_2 = M2x::Identity(2, 2);
  auto ix2_2 = Mx2::Identity(2, 2);
  auto ixx_22 = Mxx::Identity(2, 2);

  static_assert(not has_dynamic_dimensions<decltype(diagonal_of(i22))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(i2x_2))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(ix2_2))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(ixx_22))>);

  static_assert(constant_coefficient_v<decltype(diagonal_of(i22))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(i2x_2))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(ix2_2))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(ixx_22))> == 1);

  EXPECT_TRUE(is_near(diagonal_of(i22), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(i2x_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(ix2_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(ixx_22), M21::Constant(1)));
  EXPECT_TRUE(is_near(Eigen3::make_eigen_wrapper(diagonal_of(i22)), M21::Constant(1)));

  static_assert(constant_coefficient_v<decltype(diagonal_of(M22::Identity()))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(M2x::Identity(2, 2)))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(Mx2::Identity(2, 2)))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(Mxx::Identity(2, 2)))> == 1);

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity()), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2, 2)), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(Mx2::Identity(2, 2)), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(Mxx::Identity(2, 2)), M21::Constant(1)));

  static_assert(constant_adapter<decltype(diagonal_of(std::declval<Eigen3::IdentityMatrix<M33>>()))>);
  static_assert(index_dimension_of_v<decltype(diagonal_of(std::declval<Eigen3::IdentityMatrix<M33>>())), 0> == 3);
  static_assert(index_dimension_of_v<decltype(diagonal_of(std::declval<Eigen3::IdentityMatrix<M33>>())), 1> == 1);

  auto z11 = M11::Identity() - M11::Identity();

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  EXPECT_TRUE(is_near(diagonal_of(z22.template triangularView<Eigen::Upper>()), M21::Zero())); static_assert(zero<decltype(diagonal_of(z22.template triangularView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z20_2.template triangularView<Eigen::Lower>()), M21::Zero())); static_assert(zero<decltype(diagonal_of(z20_2.template triangularView<Eigen::Lower>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z02_2.template triangularView<Eigen::Upper>()), M21::Zero())); static_assert(zero<decltype(diagonal_of(z02_2.template triangularView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z00_22.template triangularView<Eigen::Lower>()), M21::Zero())); static_assert(zero<decltype(diagonal_of(z00_22.template triangularView<Eigen::Lower>()))>);
  EXPECT_TRUE(is_near(diagonal_of(Eigen3::make_eigen_wrapper(z22.template triangularView<Eigen::Upper>())), M21::Zero()));

  auto c11_2 {M11::Identity() + M11::Identity()};

  auto c22_2 = c11_2.replicate<2, 2>();
  auto c20_2_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 2);
  auto c02_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2>(c11_2, 2, 2);
  auto c00_22_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 2);

  EXPECT_TRUE(is_near(diagonal_of(c22_2.template triangularView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c22_2.template triangularView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c20_2_2.template triangularView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c20_2_2.template triangularView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c02_2_2.template triangularView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c02_2_2.template triangularView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c00_22_2.template triangularView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c00_22_2.template triangularView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(Eigen3::make_eigen_wrapper(c22_2.template triangularView<Eigen::Upper>())), M21::Constant(2)));

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity().template triangularView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M22::Identity().template triangularView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Lower>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Lower>()))> == 1);
}


TEST(adapters, incidental_creation)
{
  EXPECT_NEAR(trace(M0x{M00 {}}), 0, 1e-6); // creates ConstantAdapter
  EXPECT_NEAR(trace(Mx0{M00 {}}), 0, 1e-6); // creates ConstantAdapter
}


TEST(adapters, ConstantAdapter_equality)
{
  ConstantAdapter<M23, double, 3> ca23;
  ConstantAdapter<M2x, double, 3> ca20 {3};
  ConstantAdapter<Mx3, double, 3> ca03 {2};
  ConstantAdapter<Mxx, double, 3> ca00 {2, 3};

  auto nc11 = M11::Identity() + M11::Identity() + M11::Identity(); using NC11 = decltype(nc11);
  auto nc23 = nc11.replicate<2,3>();
  auto nc20 = Eigen::Replicate<NC11, 2, Eigen::Dynamic> {nc11, 2, 3};
  auto nc03 = Eigen::Replicate<NC11, Eigen::Dynamic, 3> {nc11, 2, 3};
  auto nc00 = Eigen::Replicate<NC11, Eigen::Dynamic, Eigen::Dynamic> {nc11, 2, 3};

  auto mc23 = M23::Constant(3);
  auto mc20 = M2x::Constant(2, 3, 3);
  auto mc03 = Mx3::Constant(2, 3, 3);
  auto mc00 = Mxx::Constant(2, 3, 3);

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
  ZeroAdapter<M2x> za20 {3};
  ZeroAdapter<Mx3> za03 {2};
  ZA00 za00 {2, 3};

  auto nz11 = M11::Identity() - M11::Identity(); using NZ11 = decltype(nz11);
  auto nz23 = nz11.replicate<2,3>();
  auto nz20 = Eigen::Replicate<NZ11, 2, Eigen::Dynamic> {nz11, 2, 3};
  auto nz03 = Eigen::Replicate<NZ11, Eigen::Dynamic, 3> {nz11, 2, 3};
  auto nz00 = Eigen::Replicate<NZ11, Eigen::Dynamic, Eigen::Dynamic> {nz11, 2, 3};

  auto mz23 = M23::Zero();
  auto mz20 = M2x::Zero(2, 3);
  auto mz03 = Mx3::Zero(2, 3);
  auto mz00 = Mxx::Zero(2, 3);

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


TEST(adapters, ConstantAdapter_arithmetic)
{
  EXPECT_TRUE(is_near(-ConstantAdapter<M22, double, 7> {}, ConstantAdapter<M22, double, -7> {}));
  static_assert(constant_adapter<decltype(-ConstantAdapter<M22, double, 7> {})>);

  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 0> {} * 2.0, ConstantAdapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} * -2.0, ConstantAdapter<M22, double, -6> {}));
  static_assert(constant_adapter<decltype(ConstantAdapter<M22, double, 0> {} * 2.0)>);
  static_assert(constant_matrix<decltype(ConstantAdapter<M22, double, 0> {} * 2.0), ConstantType::static_constant>);
  static_assert(constant_matrix<decltype(ConstantAdapter<M22, double, 3> {} * 2.0), ConstantType::dynamic_constant>);
  static_assert(constant_matrix<decltype(ConstantAdapter<M22, double, 3> {} * N2{}), ConstantType::static_constant>);

  EXPECT_TRUE(is_near(3.0 * ConstantAdapter<M22, double, 0> {}, ConstantAdapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(-3.0 * ConstantAdapter<M22, double, 3> {}, ConstantAdapter<M22, double, -9> {}));
  static_assert(constant_adapter<decltype(3.0 * ConstantAdapter<M22, double, 0> {})>);
  static_assert(constant_matrix<decltype(3.0 * ConstantAdapter<M22, double, 0> {}), ConstantType::static_constant>);
  static_assert(constant_matrix<decltype(3.0 * ConstantAdapter<M22, double, 2> {}), ConstantType::dynamic_constant>);
  static_assert(constant_matrix<decltype(N2{} * ConstantAdapter<M22, double, 2> {}), ConstantType::static_constant>);

  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 0> {} / 2.0, ConstantAdapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 8> {} / -2.0, ConstantAdapter<M22, double, -4> {}));
  static_assert(not constant_adapter<decltype(ConstantAdapter<M22, double, 8> {} / -2.0)>);
  static_assert(constant_matrix<decltype(ConstantAdapter<M22, double, 8> {} / -2.0)>);

  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} + ConstantAdapter<M22, double, 5> {}, ConstantAdapter<M22, double, 8> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} + M22::Constant(5), ConstantAdapter<M22, double, 8> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) + ConstantAdapter<M22, double, 3> {}, ConstantAdapter<M22, double, 8> {}));
  static_assert(constant_matrix<decltype(ConstantAdapter<M22, double, 3> {} + ConstantAdapter<M22, double, 5> {}), ConstantType::static_constant>);
  static_assert(not constant_adapter<decltype(ConstantAdapter<M22, double, 3> {} + ConstantAdapter<M22, double, 5> {})>);
  static_assert(constant_matrix<decltype(M22::Constant(5) + ConstantAdapter<M22, double, 3> {}), ConstantType::dynamic_constant>);
  static_assert(not constant_adapter<decltype(M22::Constant(5) + ConstantAdapter<M22, double, 3> {})>);

  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} - ConstantAdapter<M22, double, 5> {}, ConstantAdapter<M22, double, -2> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M22, double, 3> {} - M22::Constant(5), ConstantAdapter<M22, double, -2> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) - ConstantAdapter<M22, double, 3> {}, ConstantAdapter<M22, double, 2> {}));
  static_assert(constant_matrix<decltype(ConstantAdapter<M22, double, 3> {} - ConstantAdapter<M22, double, 5> {}), ConstantType::static_constant>);
  static_assert(not constant_adapter<decltype(ConstantAdapter<M22, double, 3> {} - ConstantAdapter<M22, double, 5> {})>);
  static_assert(constant_matrix<decltype(M22::Constant(5) - ConstantAdapter<M22, double, 3> {}), ConstantType::dynamic_constant>);
  static_assert(not constant_adapter<decltype(M22::Constant(5) - ConstantAdapter<M22, double, 3> {})>);

  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {} * ConstantAdapter<M32, double, 5> {}, ConstantAdapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M34, double, 4> {} * ConstantAdapter<M42, double, 7> {}, ConstantAdapter<M32, double, 112> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M23, double, 3> {} * M32::Constant(5), ConstantAdapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(ConstantAdapter<M34, double, 4> {} * M42::Constant(7), ConstantAdapter<M32, double, 112> {}));
  EXPECT_TRUE(is_near(M23::Constant(3) * ConstantAdapter<M32, double, 5> {}, ConstantAdapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(M34::Constant(4) * ConstantAdapter<M42, double, 7> {}, ConstantAdapter<M32, double, 112> {}));
  static_assert(constant_matrix<decltype(ConstantAdapter<M23, double, 3> {} * ConstantAdapter<M32, double, 5> {}), ConstantType::static_constant>);
  static_assert(not constant_adapter<decltype(ConstantAdapter<M23, double, 3> {} * ConstantAdapter<M32, double, 5> {})>);
  static_assert(constant_matrix<decltype(M23::Constant(3) * ConstantAdapter<M32, double, 5> {}), ConstantType::dynamic_constant>);
  static_assert(not constant_adapter<decltype(M23::Constant(3) * ConstantAdapter<M32, double, 5> {})>);

  EXPECT_EQ((ConstantAdapter<M43, double, 3>{}.rows()), 4);
  EXPECT_EQ((ConstantAdapter<M43, double, 3>{}.cols()), 3);

  ZA00 z00 {2, 2};

  EXPECT_TRUE(is_near(-z00, z00, 1e-6));
  static_assert(zero<decltype(-z00)>);
  static_assert(constant_adapter<decltype(-z00)>);

  auto m22y = make_eigen_matrix<double, 2, 2>(1, 2, 3, 4);
  EXPECT_TRUE(is_near(z00 + m22y, m22y, 1e-6));
  EXPECT_TRUE(is_near(m22y + z00, m22y, 1e-6));
  static_assert(zero<decltype(z00 + z00)>);
  EXPECT_TRUE(is_near(m22y - z00, m22y, 1e-6));
  EXPECT_TRUE(is_near(z00 - m22y, -m22y, 1e-6));
  EXPECT_TRUE(is_near(z00 - m22y.Identity(), -m22y.Identity(), 1e-6));
  //static_assert(diagonal_matrix<decltype(z00 - decltype(m22y)::Identity())>);
  static_assert(zero<decltype(z00 - z00)>);
  EXPECT_TRUE(is_near(z00 * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 * m22y, z00, 1e-6));
  EXPECT_TRUE(is_near(m22y * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 * 2, z00, 1e-6));
  static_assert(zero<decltype(z00 * 2)>);
  static_assert(constant_adapter<decltype(z00 * 2)>);
  EXPECT_TRUE(is_near(2 * z00, z00, 1e-6));
  static_assert(zero<decltype(2 * z00)>);
  static_assert(constant_adapter<decltype(2 * z00)>);
  EXPECT_TRUE(is_near(z00 / 2, z00, 1e-6));

  EXPECT_EQ((z00.rows()), 2);
  EXPECT_EQ((z00.cols()), 2);
}


/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen.gtest.hpp"
#include "linear-algebra/adapters/constant_adapter.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using stdex::numbers::pi;

namespace
{
  using Axis2 = Dimensions<2>;

  using ZA11 = zero_adapter<M11>;
  using ZA10 = zero_adapter<M1x>;
  using ZA01 = zero_adapter<Mx1>;
  using ZA00 = zero_adapter<Mxx>;

  using ZA21 = zero_adapter<M21>;
  using ZA12 = zero_adapter<M12>;
  using ZA22 = zero_adapter<M22>;
  using ZA23 = zero_adapter<M23>;
  using ZA20 = zero_adapter<M2x>;
  using ZA02 = zero_adapter<Mx2>;
  using ZA03 = zero_adapter<Mx3>;

  using ZA13 = zero_adapter<M13>;
  using ZA31 = zero_adapter<M31>;
  using ZA33 = zero_adapter<M33>;
  using ZA30 = zero_adapter<M3x>;

  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
}


TEST(eigen3, constant_adapter_traits)
{
  static_assert(indexible<constant_adapter<M22, double, 1>>);
  static_assert(indexible<ZA31>);

  static_assert(values::fixed<constant_value<constant_adapter<M22, double, 1>>>);
  static_assert(not values::dynamic<constant_value<constant_adapter<M22, double, 1>>>);
  static_assert(values::dynamic<constant_value<constant_adapter<M22>>>);
  static_assert(not values::fixed<constant_value<constant_adapter<M22>>>);

  static_assert(constant_diagonal_matrix<ZA33>);
  static_assert(constant_diagonal_matrix<ZA31>);
  static_assert(constant_diagonal_matrix<ZA13>);

  static_assert(zero<constant_adapter<M22, double, 0>>);
  static_assert(zero<constant_adapter<Mxx, double, 0>>);
  static_assert(not zero<constant_adapter<M22, double, 1>>);
  static_assert(zero<ZA31>);

  static_assert(diagonal_matrix<constant_adapter<M22, double, 0>>);
  static_assert(diagonal_matrix<constant_adapter<M11, double, 5>>);
  static_assert(diagonal_matrix<constant_adapter<M11>>);
  static_assert(not diagonal_matrix<constant_adapter<Mxx, double, 5>>);
  static_assert(not diagonal_matrix<constant_adapter<CM22, cdouble, 5>>);

  static_assert(diagonal_matrix<ZA33>);
  static_assert(not internal::has_nested_vector<ZA33>);
  static_assert(diagonal_matrix<ZA31>);
  static_assert(diagonal_matrix<ZA00>);

  static_assert(hermitian_matrix<constant_adapter<M22, double, 0>>);
  static_assert(hermitian_matrix<constant_adapter<M11, double, 5>>);
  static_assert(hermitian_matrix<constant_adapter<CM11, cdouble, 5>>);
  static_assert(hermitian_matrix<constant_adapter<M22, double, 5>>);
  static_assert(hermitian_matrix<constant_adapter<CM22, cdouble, 5>>);
  static_assert(not hermitian_matrix<constant_adapter<M34, double, 5>>);
  static_assert(not hermitian_matrix<constant_adapter<CM34, cdouble, 5>>);

  static_assert(hermitian_matrix<ZA33>);
  static_assert(hermitian_matrix<zero_adapter<CM33>>);
  static_assert(not hermitian_matrix<ZA31, applicability::permitted>);
  static_assert(not hermitian_matrix<ZA00>);
  static_assert(hermitian_matrix<ZA00, applicability::permitted>);

  static_assert(triangular_matrix<constant_adapter<M22, double, 0>>);
  static_assert(triangular_matrix<constant_adapter<M11, double, 5>>);
  static_assert(not triangular_matrix<constant_adapter<M22, double, 5>>);
  static_assert(not triangular_matrix<constant_adapter<Mxx, double, 5>>);
  static_assert(not triangular_matrix<constant_adapter<M34, double, 5>>);
  static_assert(triangular_matrix<constant_adapter<M34, double, 0>>); // becaues it's a zero matrix and thus diagonal

  static_assert(triangular_matrix<ZA33, triangle_type::upper>);
  static_assert(triangular_matrix<ZA31, triangle_type::upper>);
  static_assert(triangular_matrix<ZA00, triangle_type::upper>);

  static_assert(triangular_matrix<ZA33, triangle_type::lower>);
  static_assert(triangular_matrix<ZA31, triangle_type::lower>);
  static_assert(triangular_matrix<ZA00, triangle_type::lower>);

  static_assert(square_shaped<constant_adapter<M22, double, 0>, applicability::permitted>);
  static_assert(not square_shaped<constant_adapter<M34, double, 5>, applicability::permitted>);
  static_assert(square_shaped<constant_adapter<M3x, double, 5>, applicability::permitted>);
  static_assert(square_shaped<constant_adapter<Mxx, double, 5>, applicability::permitted>);

  static_assert(square_shaped<constant_adapter<M22, double, 0>>);
  static_assert(square_shaped<constant_adapter<M22, double, 5>>);
  static_assert(not square_shaped<constant_adapter<Mxx, double, 5>>);
  static_assert(not square_shaped<constant_adapter<M34, double, 5>>);

  static_assert(not square_shaped<ZA31, applicability::permitted>);
  static_assert(square_shaped<ZA33>);
  static_assert(not square_shaped<ZA30>);
  static_assert(square_shaped<ZA30, applicability::permitted>);
  static_assert(not square_shaped<ZA03>);
  static_assert(square_shaped<ZA03, applicability::permitted>);
  static_assert(not square_shaped<ZA00>);
  static_assert(square_shaped<ZA00, applicability::permitted>);

  static_assert(one_dimensional<constant_adapter<M11, double, 5>>);
  static_assert(one_dimensional<constant_adapter<M1x, double, 5>, applicability::permitted>);
  static_assert(not one_dimensional<constant_adapter<M1x, double, 5>>);
  static_assert(one_dimensional<constant_adapter<Mxx, double, 5>, applicability::permitted>);
  static_assert(not one_dimensional<constant_adapter<Mxx, double, 5>>);

  static_assert(not one_dimensional<ZA31, applicability::permitted>);
  static_assert(one_dimensional<ZA11>);
  static_assert(not one_dimensional<ZA10>);
  static_assert(one_dimensional<ZA10, applicability::permitted>);
  static_assert(not one_dimensional<ZA01>);
  static_assert(one_dimensional<ZA01, applicability::permitted>);
  static_assert(not one_dimensional<ZA00>);
  static_assert(one_dimensional<ZA00, applicability::permitted>);

  static_assert(element_gettable<constant_adapter<M22, double, 3>, 2>);
  static_assert(element_gettable<constant_adapter<M2x, double, 3>, 2>);
  static_assert(element_gettable<constant_adapter<Mx2, double, 3>, 2>);
  static_assert(element_gettable<constant_adapter<Mxx, double, 3>, 2>);

  static_assert(element_gettable<ZA33, 2>);
  static_assert(element_gettable<ZA30, 2>);
  static_assert(element_gettable<ZA03, 2>);
  static_assert(element_gettable<ZA00, 2>);

  static_assert(not writable_by_component<constant_adapter<M22, double, 3>&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<constant_adapter<M2x, double, 3>&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<constant_adapter<Mx2, double, 3>&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<constant_adapter<M2x, double, 3>&, std::array<std::size_t, 2>>);

  static_assert(not writable_by_component<ZA33&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ZA30&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ZA03&, std::array<std::size_t, 2>>);
  static_assert(not writable_by_component<ZA00&, std::array<std::size_t, 2>>);

  static_assert(dynamic_dimension<constant_adapter<Mxx, double, 5>, 0>);
  static_assert(dynamic_dimension<constant_adapter<Mxx, double, 5>, 1>);
  static_assert(dynamic_dimension<constant_adapter<Mx1, double, 5>, 0>);
  static_assert(not dynamic_dimension<constant_adapter<Mx1, double, 5>, 1>);
  static_assert(not dynamic_dimension<constant_adapter<M1x, double, 5>, 0>);
  static_assert(dynamic_dimension<constant_adapter<M1x, double, 5>, 1>);

  static_assert(get_pattern_collection(constant_adapter<M23, double, 5>{}, std::integral_constant<std::size_t, 0>{}) == 2);
  static_assert(get_pattern_collection(constant_adapter<M23, double, 5>{}, std::integral_constant<std::size_t, 1>{}) == 3);
  static_assert(get_pattern_collection(constant_adapter<M23, double, 5>{}, std::integral_constant<std::size_t, 2>{}) == 1);

  static_assert(dynamic_dimension<ZA00, 0>);
  static_assert(dynamic_dimension<ZA03, 0>);
  static_assert(not dynamic_dimension<ZA30, 0>);

  static_assert(dynamic_dimension<ZA00, 1>);
  static_assert(not dynamic_dimension<ZA03, 1>);
  static_assert(dynamic_dimension<ZA30, 1>);

  static_assert(not writable<constant_adapter<M33, double, 7>>);
  static_assert(not writable<ZA33>);
}


TEST(eigen3, constant_adapter_class)
{
  constant_adapter<M23, double, 3> c323 {};
  constant_adapter<M2x, double, 3> c320 {3};
  constant_adapter<Mx3, double, 3> c303 {2};
  constant_adapter<Mxx, double, 3> c300 {2,3};

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

  EXPECT_TRUE(is_near(constant_adapter {z23}, M23::Zero()));
  EXPECT_TRUE(is_near(constant_adapter {z20}, M23::Zero()));
  EXPECT_TRUE(is_near(constant_adapter {z03}, M23::Zero()));
  EXPECT_TRUE(is_near(constant_adapter {z00}, M23::Zero()));

  EXPECT_TRUE(is_near(constant_adapter {ZA23 {N2{}, N3{}}}, M23::Zero()));
  EXPECT_TRUE(is_near(constant_adapter {ZA20 {N2{}, 3}}, M23::Zero()));
  EXPECT_TRUE(is_near(constant_adapter {ZA03 {2, N3{}}}, M23::Zero()));
  EXPECT_TRUE(is_near(constant_adapter {ZA00 {2, 3}}, M23::Zero()));

  EXPECT_NEAR(std::real(constant_adapter<CM22, cdouble, 3> {}(0, 1)), 3, 1e-6);

  EXPECT_TRUE(is_near(constant_adapter {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter {c300}, M23::Constant(3)));

  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<M23, double, 3> {N2{}, N3{}}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<M2x, double, 3> {N2{}, 3}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<Mx3, double, 3> {2, N3{}}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<Mxx, double, 3> {2, 3}}, M23::Constant(3)));

  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<M23, double, 2> {N2{}, N3{}}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<M2x, double, 2> {N2{}, 3}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<M2x, double, 2> {N2{}, N2{}, 3}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<Mx3, double, 2> {2, N3{}}}, M23::Constant(2)));
  EXPECT_TRUE(is_near(constant_adapter {constant_adapter<Mxx, double, 2> {2, 3}}, M23::Constant(2)));

  EXPECT_NEAR((constant_adapter {constant_adapter<M23, double, 0>{}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((constant_adapter {constant_adapter<M2x, double, 0>{3}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((constant_adapter {constant_adapter<Mx3, double, 0>{2}}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((constant_adapter {constant_adapter<Mxx, double, 0>{2,3}}(1, 2)), 0, 1e-6);

  EXPECT_TRUE(is_near(constant_adapter<M23, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M23, double, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M23, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M2x, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M2x, double, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M2x, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mx3, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mx3, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mx3, double, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mxx, double, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mxx, double, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mxx, double, 3> {c303}, M23::Constant(3)));

  EXPECT_TRUE(is_near(constant_adapter<M23, double, 3> {constant_adapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M23, double, 3> {constant_adapter {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M23, double, 3> {constant_adapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M2x, double, 3> {constant_adapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M2x, double, 3> {constant_adapter {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<M2x, double, 3> {constant_adapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mx3, double, 3> {constant_adapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mx3, double, 3> {constant_adapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mx3, double, 3> {constant_adapter {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mxx, double, 3> {constant_adapter {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mxx, double, 3> {constant_adapter {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(constant_adapter<Mxx, double, 3> {constant_adapter {c303}}, M23::Constant(3)));

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

  c323 = constant_adapter {c323}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = constant_adapter {c323}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = constant_adapter {c323}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = constant_adapter {c323}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = constant_adapter {c320}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = constant_adapter {c320}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = constant_adapter {c320}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = constant_adapter {c320}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = constant_adapter {c303}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = constant_adapter {c303}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = constant_adapter {c303}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = constant_adapter {c303}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = constant_adapter {c300}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = constant_adapter {c300}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = constant_adapter {c300}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = constant_adapter {c300}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));

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

  c323 = constant_adapter {nc23}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = constant_adapter {nc23}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = constant_adapter {nc23}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = constant_adapter {nc23}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = constant_adapter {nc20}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = constant_adapter {nc20}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = constant_adapter {nc20}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = constant_adapter {nc20}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = constant_adapter {nc03}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = constant_adapter {nc03}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = constant_adapter {nc03}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = constant_adapter {nc03}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = constant_adapter {nc00}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = constant_adapter {nc00}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = constant_adapter {nc00}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = constant_adapter {nc00}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));

  EXPECT_NEAR((constant_adapter<M22, double, 3> {}(0, 0)), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<M22, double, 3> {}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<M2x, double, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<Mx2, double, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<Mxx, double, 3> {2,2}(0, 1)), 3, 1e-6);

  EXPECT_NEAR((constant_adapter<M31, double, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<Mx1, double, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<M13, double, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<M1x, double, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<M31, double, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<Mx1, double, 3> {3}[1]), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<M13, double, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((constant_adapter<M1x, double, 3> {3}[1]), 3, 1e-6);

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

  z23 = constant_adapter {nz23}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = constant_adapter {nz23}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = constant_adapter {nz23}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = constant_adapter {nz23}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = constant_adapter {nz20}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = constant_adapter {nz20}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = constant_adapter {nz20}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = constant_adapter {nz20}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = constant_adapter {nz03}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = constant_adapter {nz03}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = constant_adapter {nz03}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = constant_adapter {nz03}; EXPECT_TRUE(is_near(z00, M23::Zero()));
  z23 = constant_adapter {nz00}; EXPECT_TRUE(is_near(z23, M23::Zero()));
  z20 = constant_adapter {nz00}; EXPECT_TRUE(is_near(z20, M23::Zero()));
  z03 = constant_adapter {nz00}; EXPECT_TRUE(is_near(z03, M23::Zero()));
  z00 = constant_adapter {nz00}; EXPECT_TRUE(is_near(z00, M23::Zero()));

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


TEST(eigen3, make_dense_object_from)
{
  constant_adapter<M34, double, 5> c534 {};
  constant_adapter<M3x, double, 5> c530_4 {4};
  constant_adapter<Mx4, double, 5> c504_3 {3};
  constant_adapter<Mxx, double, 5> c500_34 {3, 4};

  EXPECT_TRUE(is_near(to_dense_object(c534), M34::Constant(5)));
  EXPECT_TRUE(is_near(to_dense_object(c530_4), M34::Constant(5)));
  EXPECT_TRUE(is_near(to_dense_object(c504_3), M34::Constant(5)));
  EXPECT_TRUE(is_near(to_dense_object(c500_34), M34::Constant(5)));
  EXPECT_TRUE(is_near(to_dense_object(constant_adapter<CM34, cdouble, 5> {}), CM34::Constant(cdouble(5,0))));
}


TEST(eigen3, make_constant)
{
  auto m23 = make_dense_object_from<M23>(0, 0, 0, 0, 0, 0);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  constant_adapter<M34, double, 5> c534 {};
  constant_adapter<M3x, double, 5> c530_4 {4};
  constant_adapter<Mx4, double, 5> c504_3 {3};
  constant_adapter<Mxx, double, 5> c500_34 {3, 4};

  using C534 = decltype(c534);

  constexpr values::fixed_value<double, 5> nd5;

  EXPECT_TRUE(is_near(make_constant<M23>(nd5, Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(nd5, Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(nd5, 2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(nd5, 2, 3), M23::Constant(5)));
  static_assert(constant_value_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), 3))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx>(nd5, 2, Dimensions<3>()))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx>(nd5, 2, 3))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, 2, Dimensions<3>())), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_constant<Mxx>(nd5, 2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, 2, 3)), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_constant<Mxx>(nd5, 2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, Dimensions<2>(), 3)), 1> == stdex::dynamic_extent);  EXPECT_EQ(get_index_dimension_of<1>(make_constant<Mxx>(nd5, 2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, 2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx>(nd5, 2, 3)), 1> == stdex::dynamic_extent);  EXPECT_EQ(get_index_dimension_of<1>(make_constant<Mxx>(nd5, 2, 3)), 3);

  EXPECT_TRUE(is_near(make_constant<M23>(5., Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(5., Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(5., 2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx>(5., 2, 3), M23::Constant(5)));

  EXPECT_TRUE(is_near(make_constant<M23>(nd5), M23::Constant(5)));
  static_assert(constant_value_v<decltype(make_constant<M23>(nd5))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<M23>(nd5)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<M23>(nd5)), 1> == 3);

  EXPECT_TRUE(is_near(make_constant<C534>(5., Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<C534>(5., Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<C534>(5., 2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<C534>(5., 2, 3), M23::Constant(5)));
  static_assert(constant_value_v<decltype(make_constant<C534>(values::fixed_value<double, 5>{}, Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_value_v<decltype(make_constant<C534>(values::fixed_value<double, 5>{}, Dimensions<2>(), 3))> == 5);
  static_assert(constant_value_v<decltype(make_constant<C534>(values::fixed_value<double, 5>{}, 2, Dimensions<3>()))> == 5);
  static_assert(constant_value_v<decltype(make_constant<C534>(values::fixed_value<double, 5>{}, 2, 3))> == 5);

  EXPECT_TRUE(is_near(make_constant(m23, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(m2x_3, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(mx3_2, nd5), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(mxx_23, nd5), M23::Constant(5)));
  static_assert(constant_value_v<decltype(make_constant(m23, nd5))> == 5);
  static_assert(constant_value_v<decltype(make_constant(m2x_3, nd5))> == 5);
  static_assert(constant_value_v<decltype(make_constant(mx3_2, nd5))> == 5);
  static_assert(constant_value_v<decltype(make_constant(mxx_23, nd5))> == 5);

  EXPECT_TRUE(is_near(make_constant(m23, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(m2x_3, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(mx3_2, 5.), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant(mxx_23, 5.), M23::Constant(5)));

  EXPECT_TRUE(is_near(make_constant<Mxx, double, 5>(Dimensions<2>{}, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx, double, 5>(Dimensions<2>{}, 3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx, double, 5>(2, Dimensions<3>{}), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<Mxx, double, 5>(2, 3), M23::Constant(5)));
  static_assert(constant_value_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), 3))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx, double, 5>(2, Dimensions<3>()))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx, double, 5>(2, 3))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), Dimensions<3>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), 3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(2, Dimensions<3>())), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_constant<Mxx, double, 5>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(2, 3)), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_constant<Mxx, double, 5>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(Dimensions<2>(), 3)), 1> == stdex::dynamic_extent);  EXPECT_EQ(get_index_dimension_of<1>(make_constant<Mxx, double, 5>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<Mxx, double, 5>(2, 3)), 1> == stdex::dynamic_extent);  EXPECT_EQ(get_index_dimension_of<1>(make_constant<Mxx, double, 5>(2, 3)), 3);

  EXPECT_TRUE(is_near(make_constant<Mxx, cdouble, 3, 4>(Dimensions<2>{}, Dimensions<3>{}), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<Mxx, cdouble, 3, 4>(Dimensions<2>{}, 3), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<Mxx, cdouble, 3, 4>(2, Dimensions<3>{}), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<Mxx, cdouble, 3, 4>(2, 3), CM23::Constant(cdouble{3, 4})));

  EXPECT_TRUE(is_near(make_constant<M23, double, 5>(), M23::Constant(5)));
  static_assert(constant_value_v<decltype(make_constant<M23, double, 5>())> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<M23, double, 5>()), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<M23, double, 5>()), 1> == 3);

  EXPECT_TRUE(is_near(make_constant<M23, cdouble, 3, 4>(), CM23::Constant(cdouble{3, 4})));

  static_assert(constant_value_v<decltype(make_constant<Mxx, 5>(Dimensions<2>(), Dimensions<3>()))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx, 5>(Dimensions<2>(), 3))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx, 5>(2, Dimensions<3>()))> == 5);
  static_assert(constant_value_v<decltype(make_constant<Mxx, 5>(2, 3))> == 5);
  static_assert(std::is_integral_v<constant_value<decltype(make_constant<Mxx, 5>(Dimensions<2>(), Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_constant<Mxx, 5>(Dimensions<2>(), 3))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_constant<Mxx, 5>(2, Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_constant<Mxx, 5>(2, 3))>::value_type>);

  EXPECT_TRUE(is_near(make_constant<double, 5>(m23), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<double, 5>(m2x_3), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<double, 5>(mx3_2), M23::Constant(5)));
  EXPECT_TRUE(is_near(make_constant<double, 5>(mxx_23), M23::Constant(5)));
  static_assert(constant_value_v<decltype(make_constant<double, 5>(m23))> == 5);
  static_assert(constant_value_v<decltype(make_constant<double, 5>(m2x_3))> == 5);
  static_assert(constant_value_v<decltype(make_constant<double, 5>(mx3_2))> == 5);
  static_assert(constant_value_v<decltype(make_constant<double, 5>(mxx_23))> == 5);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(m23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(m2x_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(mx3_2)), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_constant<double, 5>(mx3_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(mxx_23)), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_constant<double, 5>(mxx_23)), 2);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(m2x_3)), 1> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<1>(make_constant<double, 5>(m2x_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(mx3_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant<double, 5>(mxx_23)), 1> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<1>(make_constant<double, 5>(mxx_23)), 3);

  EXPECT_TRUE(is_near(make_constant<cdouble, 3, 4>(m23), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<cdouble, 3, 4>(m2x_3), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<cdouble, 3, 4>(mx3_2), CM23::Constant(cdouble{3, 4})));
  EXPECT_TRUE(is_near(make_constant<cdouble, 3, 4>(mxx_23), CM23::Constant(cdouble{3, 4})));

  static_assert(constant_value_v<decltype(make_constant<5>(m23))> == 5);
  static_assert(constant_value_v<decltype(make_constant<5>(m2x_3))> == 5);
  static_assert(constant_value_v<decltype(make_constant<5>(mx3_2))> == 5);
  static_assert(constant_value_v<decltype(make_constant<5>(mxx_23))> == 5);
  static_assert(std::is_integral_v<constant_value<decltype(make_constant<5>(m23))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_constant<5>(m2x_3))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_constant<5>(mx3_2))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_constant<5>(mxx_23))>::value_type>);
}


TEST(eigen3, make_zero)
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
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(2, Dimensions<3>())), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_zero<Mxx>(2, Dimensions<3>())), 2);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(2, 3)), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_zero<Mxx>(2, 3)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero(m23)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero(m2x_3)), 0> == 2);
  static_assert(index_dimension_of_v<decltype(make_zero(mx3_2)), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_zero(mx3_2)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero(mxx_23)), 0> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<0>(make_zero(mxx_23)), 2);
  static_assert(index_dimension_of_v<decltype(make_zero<M23>()), 0> == 2);

  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(Dimensions<2>(), Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(Dimensions<2>(), 3)), 1> == stdex::dynamic_extent);  EXPECT_EQ(get_index_dimension_of<1>(make_zero<Mxx>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(2, Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero<Mxx>(2, 3)), 1> == stdex::dynamic_extent);  EXPECT_EQ(get_index_dimension_of<1>(make_zero<Mxx>(2, 3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero(m23)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero(m2x_3)), 1> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<1>(make_zero(m2x_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero(mx3_2)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_zero(mxx_23)), 1> == stdex::dynamic_extent); EXPECT_EQ(get_index_dimension_of<1>(make_zero(mxx_23)), 3);
  static_assert(index_dimension_of_v<decltype(make_zero<M23>()), 1> == 3);

  static_assert(std::is_integral_v<constant_value<decltype(make_zero<Mxx, int>(Dimensions<2>(), Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_zero<Mxx, int>(Dimensions<2>(), 3))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_zero<Mxx, int>(2, Dimensions<3>()))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_zero<Mxx, int>(2, 3))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_zero<int>(m23))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_zero<int>(m2x_3))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_zero<int>(mx3_2))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_zero<int>(mxx_23))>::value_type>);
  static_assert(std::is_integral_v<constant_value<decltype(make_zero<M23, int>())>::value_type>);
}


TEST(eigen3, make_identity_matrix_like)
{
  EXPECT_TRUE(is_near(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions<3>{}, Dimensions<1>{}), M33::Identity()));
  static_assert(identity_matrix<decltype(make_identity_matrix_like<M00>(Dimensions<3>{}, Dimensions<3>{}, Dimensions<1>{}))>);
}


TEST(eigen3, diagonal_of_constant)
{
  // Note: constant_adapter is only created when the constant is known at compile time.
  // dynamic one-by-one, known at compile time:

  static_assert(one_dimensional<decltype(diagonal_of(M11::Identity()))>);
  static_assert(not one_dimensional<decltype(diagonal_of(M1x::Identity(1, 1)))>);
  static_assert(not one_dimensional<decltype(diagonal_of(Mx1::Identity(1, 1)))>);
  static_assert(not one_dimensional<decltype(diagonal_of(Mxx::Identity(1, 1)))>);
  static_assert(one_dimensional<decltype(diagonal_of(M1x::Identity(1, 1))), applicability::permitted>);
  static_assert(one_dimensional<decltype(diagonal_of(Mx1::Identity(1, 1))), applicability::permitted>);
  static_assert(one_dimensional<decltype(diagonal_of(Mxx::Identity(1, 1))), applicability::permitted>);

  static_assert(not has_dynamic_dimensions<decltype(diagonal_of(M11::Identity()))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(M1x::Identity(1, 1)))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(Mx1::Identity(1, 1)))>);

  static_assert(dynamic_dimension<decltype(diagonal_of(Mxx::Identity(1, 1))), 0>);
  static_assert(dimension_size_of_index_is<decltype(diagonal_of(Mxx::Identity(1, 1))), 1, 1>);

  static_assert(constant_value_v<decltype(diagonal_of(M11::Identity()))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(M1x::Identity()))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(Mx1::Identity(1, 1)))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(Mxx::Identity(1, 1)))> == 1);

  auto i22 = M22::Identity();
  auto i2x_2 = M2x::Identity(2, 2);
  auto ix2_2 = Mx2::Identity(2, 2);
  auto ixx_22 = Mxx::Identity(2, 2);

  static_assert(not has_dynamic_dimensions<decltype(diagonal_of(i22))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(i2x_2))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(ix2_2))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(ixx_22))>);

  static_assert(constant_value_v<decltype(diagonal_of(i22))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(i2x_2))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(ix2_2))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(ixx_22))> == 1);

  EXPECT_TRUE(is_near(diagonal_of(i22), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(i2x_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(ix2_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(ixx_22), M21::Constant(1)));
  EXPECT_TRUE(is_near(Eigen3::make_eigen_wrapper(diagonal_of(i22)), M21::Constant(1)));

  static_assert(constant_value_v<decltype(diagonal_of(M22::Identity()))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(M2x::Identity(2, 2)))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(Mx2::Identity(2, 2)))> == 1);
  static_assert(constant_value_v<decltype(diagonal_of(Mxx::Identity(2, 2)))> == 1);

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity()), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2, 2)), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(Mx2::Identity(2, 2)), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(Mxx::Identity(2, 2)), M21::Constant(1)));

  static_assert(constant_matrix<decltype(diagonal_of(std::declval<Eigen3::IdentityMatrix<M33>>()))>);
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

  EXPECT_TRUE(is_near(diagonal_of(c22_2.template triangularView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_value_v<decltype(diagonal_of(c22_2.template triangularView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c20_2_2.template triangularView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_value_v<decltype(diagonal_of(c20_2_2.template triangularView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c02_2_2.template triangularView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_value_v<decltype(diagonal_of(c02_2_2.template triangularView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c00_22_2.template triangularView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_value_v<decltype(diagonal_of(c00_22_2.template triangularView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(Eigen3::make_eigen_wrapper(c22_2.template triangularView<Eigen::Upper>())), M21::Constant(2)));

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity().template triangularView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_value_v<decltype(diagonal_of(M22::Identity().template triangularView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_value_v<decltype(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Lower>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_value_v<decltype(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_value_v<decltype(diagonal_of(M2x::Identity(2,2).template triangularView<Eigen::Lower>()))> == 1);
}


TEST(eigen3, trace)
{
  EXPECT_NEAR(trace(M0x{M00 {}}), 0, 1e-6); // creates constant_adapter
  EXPECT_NEAR(trace(Mx0{M00 {}}), 0, 1e-6); // creates constant_adapter
}


TEST(eigen3, scalar_product)
{
  auto m23a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  auto c22_2 = (M11::Identity() + M11::Identity()).replicate<2,2>();
  auto cxx_22_2 = (M11::Identity() + M11::Identity()).replicate(2,2);

  // Constant * compile-time value
  static_assert(constant_value_v<decltype(scalar_product(std::declval<C22_2>(), std::integral_constant<int, 5>{}))> == 10);
  static_assert(constant_value_v<decltype(scalar_product(std::declval<Cxx_2>(), std::integral_constant<int, 5>{}))> == 10);

  // Constant diagonal * anything
  static_assert(values::dynamic<constant_diagonal_value<decltype(scalar_product(std::declval<Cd22_2>(), std::declval<double>()))>>);
  static_assert(values::dynamic<constant_diagonal_value<decltype(scalar_product(std::declval<Cdxx_2>(), std::declval<double>()))>>);
  EXPECT_TRUE(constant_diagonal_value{scalar_product(M22::Identity() + M22::Identity(), 5)} == 10);
  EXPECT_TRUE(constant_diagonal_value{scalar_product(Mxx::Identity(2, 2) + Mxx::Identity(2, 2), 5)} == 10);
  static_assert(constant_diagonal_value_v<decltype(scalar_product(std::declval<Cd22_2>(), std::integral_constant<int, 5>{}))> == 10);
  static_assert(constant_diagonal_value_v<decltype(scalar_product(std::declval<Cdxx_2>(), std::integral_constant<int, 5>{}))> == 10);

  // Any object * compile-time 0
  static_assert(zero<decltype(scalar_product(std::declval<M23>(), std::integral_constant<int, 0>{}))>);
  EXPECT_TRUE(constant_value{scalar_product(m23a, std::integral_constant<int, 0>{})} == 0);
  EXPECT_TRUE(constant_value{scalar_product(M23{m23a}, std::integral_constant<int, 0>{})} == 0);
  static_assert(constant_value_v<decltype(scalar_product(std::declval<C22_2>(), std::integral_constant<int, 2>{}))> == 4);

  // Any object * compile-time 1
  EXPECT_TRUE(is_near(scalar_product(m23a, std::integral_constant<int, 1>{}), m23a));
  static_assert(constant_value_v<decltype(scalar_product(std::declval<C22_2>(), std::integral_constant<int, 1>{}))> == 2);

  // Any object * compile-time constant
  EXPECT_TRUE(is_near(scalar_product(m23a, std::integral_constant<int, 5>{}), m23a * 5));
  EXPECT_TRUE(is_near(scalar_product(M23{m23a}, std::integral_constant<int, 5>{}), m23a * 5));
}


TEST(eigen3, scalar_quotient)
{
  auto m23a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  auto c22_2 = (M11::Identity() + M11::Identity()).replicate<2,2>();
  auto cxx_22_2 = (M11::Identity() + M11::Identity()).replicate(2,2);

  // Constant / compile-time value
  static_assert(constant_value_v<decltype(scalar_quotient(std::declval<C22_2>(), std::integral_constant<int, 2>{}))> == 1);
  static_assert(constant_value_v<decltype(scalar_quotient(std::declval<Cxx_2>(), std::integral_constant<int, 2>{}))> == 1);

  // Constant diagonal / anything
  static_assert(values::dynamic<constant_diagonal_value<decltype(scalar_quotient(std::declval<Cd22_2>(), std::declval<double>()))>>);
  static_assert(values::dynamic<constant_diagonal_value<decltype(scalar_quotient(std::declval<Cdxx_2>(), std::declval<double>()))>>);
  EXPECT_TRUE(constant_diagonal_value{scalar_quotient(M22::Identity() + M22::Identity(), 2)} == 1);
  EXPECT_TRUE(constant_diagonal_value{scalar_quotient(Mxx::Identity(2, 2) + Mxx::Identity(2, 2), 2)} == 1);
  static_assert(constant_diagonal_value_v<decltype(scalar_quotient(std::declval<Cd22_2>(), std::integral_constant<int, 2>{}))> == 1);
  static_assert(constant_diagonal_value_v<decltype(scalar_quotient(std::declval<Cdxx_2>(), std::integral_constant<int, 2>{}))> == 1);

  // Any object / compile-time 1
  EXPECT_TRUE(is_near(scalar_quotient(m23a, std::integral_constant<int, 1>{}), m23a));
  static_assert(constant_value_v<decltype(scalar_quotient(std::declval<C22_2>(), std::integral_constant<int, 1>{}))> == 2);

  // Any object / compile-time constant
  EXPECT_TRUE(is_near(scalar_quotient(m23a, std::integral_constant<int, 5>{}), m23a / 5));
  EXPECT_TRUE(is_near(scalar_quotient(M23{m23a}, std::integral_constant<int, 5>{}), m23a / 5));
}


TEST(eigen3, constant_adapter_equality)
{
  constant_adapter<M23, double, 3> ca23;
  constant_adapter<M2x, double, 3> ca20 {3};
  constant_adapter<Mx3, double, 3> ca03 {2};
  constant_adapter<Mxx, double, 3> ca00 {2, 3};

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

  zero_adapter<M23> za23;
  zero_adapter<M2x> za20 {3};
  zero_adapter<Mx3> za03 {2};
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


TEST(eigen3, constant_adapter_arithmetic)
{
  EXPECT_TRUE(is_near(-constant_adapter<M22, double, 7> {}, constant_adapter<M22, double, -7> {}));
  static_assert(constant_matrix<decltype(-constant_adapter<M22, double, 7> {})>);

  EXPECT_TRUE(is_near(constant_adapter<M22, double, 0> {} * 2.0, constant_adapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(constant_adapter<M22, double, 3> {} * -2.0, constant_adapter<M22, double, -6> {}));
  static_assert(constant_matrix<decltype(constant_adapter<M22, double, 0> {} * 2.0)>);
  static_assert(values::fixed<constant_value<decltype(constant_adapter<M22, double, 0> {} * 2.0)>>);
  static_assert(values::dynamic<constant_value<decltype(constant_adapter<M22, double, 3> {} * 2.0)>>);
  static_assert(values::fixed<constant_value<decltype(constant_adapter<M22, double, 3> {} * N2{})>>);

  EXPECT_TRUE(is_near(3.0 * constant_adapter<M22, double, 0> {}, constant_adapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(-3.0 * constant_adapter<M22, double, 3> {}, constant_adapter<M22, double, -9> {}));
  static_assert(constant_matrix<decltype(3.0 * constant_adapter<M22, double, 0> {})>);
  static_assert(values::fixed<constant_value<decltype(3.0 * constant_adapter<M22, double, 0> {})>>);
  static_assert(values::dynamic<constant_value<decltype(3.0 * constant_adapter<M22, double, 2> {})>>);
  static_assert(values::fixed<constant_value<decltype(N2{} * constant_adapter<M22, double, 2> {})>>);

  EXPECT_TRUE(is_near(constant_adapter<M22, double, 0> {} / 2.0, constant_adapter<M22, double, 0> {}));
  EXPECT_TRUE(is_near(constant_adapter<M22, double, 8> {} / -2.0, constant_adapter<M22, double, -4> {}));
  static_assert(constant_matrix<decltype(constant_adapter<M22, double, 8> {} / -2.0)>);

  EXPECT_TRUE(is_near(constant_adapter<M22, double, 3> {} + constant_adapter<M22, double, 5> {}, constant_adapter<M22, double, 8> {}));
  EXPECT_TRUE(is_near(constant_adapter<M22, double, 3> {} + M22::Constant(5), constant_adapter<M22, double, 8> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) + constant_adapter<M22, double, 3> {}, constant_adapter<M22, double, 8> {}));
  static_assert(values::fixed<constant_value<decltype(constant_adapter<M22, double, 3> {} + constant_adapter<M22, double, 5> {})>>);
  static_assert(constant_matrix<decltype(constant_adapter<M22, double, 3> {} + constant_adapter<M22, double, 5> {})>);
  static_assert(values::dynamic<constant_value<decltype(M22::Constant(5) + constant_adapter<M22, double, 3> {})>>);
  static_assert(constant_matrix<decltype(M22::Constant(5) + constant_adapter<M22, double, 3> {})>);

  EXPECT_TRUE(is_near(constant_adapter<M22, double, 3> {} - constant_adapter<M22, double, 5> {}, constant_adapter<M22, double, -2> {}));
  EXPECT_TRUE(is_near(constant_adapter<M22, double, 3> {} - M22::Constant(5), constant_adapter<M22, double, -2> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) - constant_adapter<M22, double, 3> {}, constant_adapter<M22, double, 2> {}));
  static_assert(values::fixed<constant_value<decltype(constant_adapter<M22, double, 3> {} - constant_adapter<M22, double, 5> {})>>);
  static_assert(constant_matrix<decltype(constant_adapter<M22, double, 3> {} - constant_adapter<M22, double, 5> {})>);
  static_assert(values::dynamic<constant_value<decltype(M22::Constant(5) - constant_adapter<M22, double, 3> {})>>);
  static_assert(constant_matrix<decltype(M22::Constant(5) - constant_adapter<M22, double, 3> {})>);

  EXPECT_TRUE(is_near(constant_adapter<M23, double, 3> {} * constant_adapter<M32, double, 5> {}, constant_adapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(constant_adapter<M34, double, 4> {} * constant_adapter<M42, double, 7> {}, constant_adapter<M32, double, 112> {}));
  EXPECT_TRUE(is_near(constant_adapter<M23, double, 3> {} * M32::Constant(5), constant_adapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(constant_adapter<M34, double, 4> {} * M42::Constant(7), constant_adapter<M32, double, 112> {}));
  EXPECT_TRUE(is_near(M23::Constant(3) * constant_adapter<M32, double, 5> {}, constant_adapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(M34::Constant(4) * constant_adapter<M42, double, 7> {}, constant_adapter<M32, double, 112> {}));
  static_assert(values::fixed<constant_value<decltype(constant_adapter<M23, double, 3> {} * constant_adapter<M32, double, 5> {})>>);
  static_assert(constant_matrix<decltype(constant_adapter<M23, double, 3> {} * constant_adapter<M32, double, 5> {})>);
  static_assert(values::dynamic<constant_value<decltype(M23::Constant(3) * constant_adapter<M32, double, 5> {})>>);
  static_assert(constant_matrix<decltype(M23::Constant(3) * constant_adapter<M32, double, 5> {})>);

  EXPECT_EQ((constant_adapter<M43, double, 3>{}.rows()), 4);
  EXPECT_EQ((constant_adapter<M43, double, 3>{}.cols()), 3);

  ZA00 z00 {2, 2};

  EXPECT_TRUE(is_near(-z00, z00, 1e-6));
  static_assert(zero<decltype(-z00)>);

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
  EXPECT_TRUE(is_near(2 * z00, z00, 1e-6));
  static_assert(zero<decltype(2 * z00)>);
  EXPECT_TRUE(is_near(z00 / 2, z00, 1e-6));

  EXPECT_EQ((z00.rows()), 2);
  EXPECT_EQ((z00.cols()), 2);
}


TEST(eigen3, constant_objects)
{
  static_assert(not constant_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>>);

  static_assert(values::fixed<constant_value<Z21>>);
  static_assert(values::fixed<constant_value<Z21>>);
  static_assert(not values::dynamic<constant_value<Z21>>);

  static_assert(values::dynamic<constant_value<M11>>);
  static_assert(not constant_matrix<M1x>);
  static_assert(not constant_matrix<Mx1>);
  static_assert(not constant_matrix<Mxx>);
  static_assert(not constant_matrix<M21>);
  EXPECT_EQ(constant_value{make_dense_object_from<M11>(5.5)}(), 5.5);

  static_assert(constant_value_v<Z21> == 0);
  static_assert(constant_value_v<Z12> == 0);
  static_assert(constant_value_v<Z23> == 0);
  static_assert(constant_value_v<Z2x> == 0);
  static_assert(constant_value_v<Zx2> == 0);

  static_assert(constant_value_v<Zxx> == 0);
  static_assert(constant_value_v<Zx1> == 0);
  static_assert(constant_value_v<C11_1> == 1);
  static_assert(constant_value_v<C11_m1> == -1);
  static_assert(constant_value_v<C11_2> == 2);
  static_assert(constant_value_v<C11_m2> == -2);
  static_assert(constant_value_v<C11_3> == 3);
  static_assert(constant_value_v<C2x_2> == 2);
  static_assert(constant_value_v<Cx2_2> == 2);
  static_assert(constant_value_v<Cxx_2> == 2);
  static_assert(constant_value_v<B22_true> == true);
  static_assert(constant_value_v<B22_false> == false);
  static_assert(not constant_matrix<Cd22_2>);
  static_assert(not constant_matrix<Cd2x_2>);
  static_assert(not constant_matrix<Cdx2_2>);
  static_assert(not constant_matrix<Cd22_3>);
  static_assert(not constant_matrix<Cd22_2>);
  static_assert(not constant_matrix<Cd2x_2>);
  static_assert(not constant_matrix<Cdx2_2>);
  static_assert(not constant_matrix<Cd22_3>);

  constant_value<C21_3> c3;
  static_assert(std::decay_t<decltype(+c3)>::value == 3);
  static_assert(std::decay_t<decltype(-c3)>::value == -3);
  static_assert(std::decay_t<decltype(c3 + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(c3 - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(c3 * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(c3 / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + c3 == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - c3 == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * c3 == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 9>{})>::value / c3 == 3);

  constant_value c3u {M21::Constant(3)};
  EXPECT_EQ(+c3u, 3);
  EXPECT_EQ(-c3u, -3);
  EXPECT_EQ((c3u + std::integral_constant<int, 2>{}), 5);
  EXPECT_EQ((c3u - std::integral_constant<int, 2>{}), 1);
  EXPECT_EQ((c3u * std::integral_constant<int, 2>{}), 6);
  EXPECT_EQ((c3u / std::integral_constant<int, 2>{}), 1.5);

  EXPECT_EQ((std::integral_constant<int, 2>{} + c3u), 5);
  EXPECT_EQ((std::integral_constant<int, 2>{} - c3u), -1);
  EXPECT_EQ((std::integral_constant<int, 2>{} * c3u), 6);
  EXPECT_EQ((std::integral_constant<int, 6>{} / c3u), 2);

  static_assert(zero<Z21>);
  static_assert(zero<Eigen::DiagonalWrapper<Z21>>);
  static_assert(zero<Z23>);
  static_assert(zero<Z2x>);
  static_assert(zero<Zx2>);
  static_assert(zero<B22_false>);
  static_assert(not zero<Cd22_2>);
  static_assert(zero<Z11>);
  static_assert(zero<Zxx>);
}


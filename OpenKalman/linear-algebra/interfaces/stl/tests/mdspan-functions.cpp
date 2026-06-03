/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "patterns/patterns.hpp"
#include "linear-algebra/functions/copy_from.hpp"
#include "linear-algebra/tests/tests.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/array-object.hpp"

using namespace OpenKalman;

using namespace OpenKalman;
using namespace OpenKalman::test;
using namespace OpenKalman::patterns;

using stdex::numbers::pi;

namespace
{
  using N0 = std::integral_constant<std::size_t, 0>;
  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
  using N4 = std::integral_constant<std::size_t, 3>;

  using F0 = values::fixed_value<double, 0>;
  using F1 = values::fixed_value<double, 1>;
  using F2 = values::fixed_value<double, 2>;
  using F3 = values::fixed_value<double, 3>;
  using F4 = values::fixed_value<double, 4>;
  using F5 = values::fixed_value<double, 5>;
}


#include "linear-algebra/functions/attach_patterns.hpp"


/*
#include "linear-algebra/functions/trace.hpp"

TEST(stl_interfaces, trace)
{
  EXPECT_NEAR(trace(M0x{M00 {}}), 0, 1e-6);
  EXPECT_NEAR(trace(Mx0{M00 {}}), 0, 1e-6);
}

TEST(stl_interfaces, scalar_product)
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


TEST(stl_interfaces, scalar_quotient)
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


TEST(stl_interfaces, constant_adapter_equality)
{
  constant_adapter<F3, double[2][3]> ca23;
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


TEST(stl_interfaces, constant_adapter_arithmetic)
{
  EXPECT_TRUE(is_near(-constant_adapter<F7, double[2][2]> {}, constant_adapter<M22, double, -7> {}));
  static_assert(constant_object<decltype(-constant_adapter<F7, double[2][2]> {})>);

  EXPECT_TRUE(is_near(constant_adapter<F0, double[2][2]> {} * 2.0, constant_adapter<F0, double[2][2]> {}));
  EXPECT_TRUE(is_near(constant_adapter<F3, double[2][2]> {} * -2.0, constant_adapter<M22, double, -6> {}));
  static_assert(constant_object<decltype(constant_adapter<F0, double[2][2]> {} * 2.0)>);
  static_assert(values::fixed<constant_value<decltype(constant_adapter<F0, double[2][2]> {} * 2.0)>>);
  static_assert(values::dynamic<constant_value<decltype(constant_adapter<F3, double[2][2]> {} * 2.0)>>);
  static_assert(values::fixed<constant_value<decltype(constant_adapter<F3, double[2][2]> {} * N2{})>>);

  EXPECT_TRUE(is_near(3.0 * constant_adapter<F0, double[2][2]> {}, constant_adapter<F0, double[2][2]> {}));
  EXPECT_TRUE(is_near(-3.0 * constant_adapter<F3, double[2][2]> {}, constant_adapter<M22, double, -9> {}));
  static_assert(constant_object<decltype(3.0 * constant_adapter<F0, double[2][2]> {})>);
  static_assert(values::fixed<constant_value<decltype(3.0 * constant_adapter<F0, double[2][2]> {})>>);
  static_assert(values::dynamic<constant_value<decltype(3.0 * constant_adapter<F2, double[2][2]> {})>>);
  static_assert(values::fixed<constant_value<decltype(N2{} * constant_adapter<F2, double[2][2]> {})>>);

  EXPECT_TRUE(is_near(constant_adapter<F0, double[2][2]> {} / 2.0, constant_adapter<F0, double[2][2]> {}));
  EXPECT_TRUE(is_near(constant_adapter<F8, double[2][2]> {} / -2.0, constant_adapter<M22, double, -4> {}));
  static_assert(constant_object<decltype(constant_adapter<F8, double[2][2]> {} / -2.0)>);

  EXPECT_TRUE(is_near(constant_adapter<F3, double[2][2]> {} + constant_adapter<F5, double[2][2]> {}, constant_adapter<F8, double[2][2]> {}));
  EXPECT_TRUE(is_near(constant_adapter<F3, double[2][2]> {} + M22::Constant(5), constant_adapter<F8, double[2][2]> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) + constant_adapter<F3, double[2][2]> {}, constant_adapter<F8, double[2][2]> {}));
  static_assert(values::fixed<constant_value<decltype(constant_adapter<F3, double[2][2]> {} + constant_adapter<F5, double[2][2]> {})>>);
  static_assert(constant_object<decltype(constant_adapter<F3, double[2][2]> {} + constant_adapter<F5, double[2][2]> {})>);
  static_assert(values::dynamic<constant_value<decltype(M22::Constant(5) + constant_adapter<F3, double[2][2]> {})>>);
  static_assert(constant_object<decltype(M22::Constant(5) + constant_adapter<F3, double[2][2]> {})>);

  EXPECT_TRUE(is_near(constant_adapter<F3, double[2][2]> {} - constant_adapter<F5, double[2][2]> {}, constant_adapter<M22, double, -2> {}));
  EXPECT_TRUE(is_near(constant_adapter<F3, double[2][2]> {} - M22::Constant(5), constant_adapter<M22, double, -2> {}));
  EXPECT_TRUE(is_near(M22::Constant(5) - constant_adapter<F3, double[2][2]> {}, constant_adapter<F2, double[2][2]> {}));
  static_assert(values::fixed<constant_value<decltype(constant_adapter<F3, double[2][2]> {} - constant_adapter<F5, double[2][2]> {})>>);
  static_assert(constant_object<decltype(constant_adapter<F3, double[2][2]> {} - constant_adapter<F5, double[2][2]> {})>);
  static_assert(values::dynamic<constant_value<decltype(M22::Constant(5) - constant_adapter<F3, double[2][2]> {})>>);
  static_assert(constant_object<decltype(M22::Constant(5) - constant_adapter<F3, double[2][2]> {})>);

  EXPECT_TRUE(is_near(constant_adapter<F3, double[2][3]> {} * constant_adapter<F5, double[3][2]> {}, constant_adapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(constant_adapter<F4, double[3][4]> {} * constant_adapter<F7, double[4][2]> {}, constant_adapter<M32, double, 112> {}));
  EXPECT_TRUE(is_near(constant_adapter<F3, double[2][3]> {} * M32::Constant(5), constant_adapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(constant_adapter<F4, double[3][4]> {} * M42::Constant(7), constant_adapter<M32, double, 112> {}));
  EXPECT_TRUE(is_near(M23::Constant(3) * constant_adapter<F5, double[3][2]> {}, constant_adapter<M22, double, 45> {}));
  EXPECT_TRUE(is_near(M34::Constant(4) * constant_adapter<F7, double[4][2]> {}, constant_adapter<M32, double, 112> {}));
  static_assert(values::fixed<constant_value<decltype(constant_adapter<F3, double[2][3]> {} * constant_adapter<F5, double[3][2]> {})>>);
  static_assert(constant_object<decltype(constant_adapter<F3, double[2][3]> {} * constant_adapter<F5, double[3][2]> {})>);
  static_assert(values::dynamic<constant_value<decltype(M23::Constant(3) * constant_adapter<F5, double[3][2]> {})>>);
  static_assert(constant_object<decltype(M23::Constant(3) * constant_adapter<F5, double[3][2]> {})>);

  EXPECT_EQ((constant_adapter<F3, double[4][3]>{}.rows()), 4);
  EXPECT_EQ((constant_adapter<F3, double[4][3]>{}.cols()), 3);

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

*/

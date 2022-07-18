/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

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
  using M50 = eigen_matrix_t<double, 5, dynamic_size>;
  using M05 = eigen_matrix_t<double, dynamic_size, 5>;

  using cdouble = std::complex<double>;

  using CM21 = eigen_matrix_t<cdouble, 2, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;
}


TEST(eigen3, reduce_matrix)
{
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, make_dense_writable_matrix_from<M13>(1, 2, 3)), make_dense_writable_matrix_from<M13>(1, 2, 3)));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, make_dense_writable_matrix_from<M21>(1, 4)), make_dense_writable_matrix_from<M21>(1, 4)));

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  auto f23_cust = [](const auto& arg1, const auto& arg2) { return arg1 + arg2 + 1; };

  auto m13_sum = make_dense_writable_matrix_from<M13>(5, 7, 9);
  auto m13_prod = make_dense_writable_matrix_from<M13>(4, 10, 18);
  auto m13_cust = make_dense_writable_matrix_from<M13>(6, 8, 10);

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, m23), m13_sum));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, m03_2), m13_sum));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, m20_3), m13_prod));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, m00_23), m13_prod));
  EXPECT_TRUE(is_near(reduce<0>(f23_cust, m23), m13_cust));
  EXPECT_TRUE(is_near(reduce<0>(f23_cust, m00_23), m13_cust));

  auto m21_sum = make_dense_writable_matrix_from<M21>(6, 15);
  auto m21_prod = make_dense_writable_matrix_from<M21>(6, 120);
  auto m21_cust = make_dense_writable_matrix_from<M21>(8, 17);

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, m23), m21_sum));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, m03_2), m21_sum));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, m20_3), m21_prod));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, m00_23), m21_prod));
  EXPECT_TRUE(is_near(reduce<1>(f23_cust, m23), m21_cust));
  EXPECT_TRUE(is_near(reduce<1>(f23_cust, m00_23), m21_cust));

  double m11_sum = 21;
  double m11_prod = 720;
  double m11_cust = 26;

  EXPECT_EQ((reduce(std::plus<double>{}, m23)), m11_sum);
  EXPECT_EQ((reduce(std::plus<double>{}, m03_2)), m11_sum);
  EXPECT_EQ((reduce<1, 0>(std::plus<double>{}, m23)), m11_sum);
  EXPECT_EQ((reduce(std::multiplies<double>{}, m00_23)), m11_prod);
  EXPECT_EQ((reduce<1, 0>(std::multiplies<double>{}, m20_3)), m11_prod);
  EXPECT_EQ((reduce(f23_cust, m00_23)), m11_cust);
  EXPECT_EQ((reduce<0, 1>(f23_cust, m23)), m11_cust);
  EXPECT_EQ((reduce<1, 0>(f23_cust, m00_23)), m11_cust);
}


TEST(eigen3, average_matrix)
{
  EXPECT_TRUE(is_near(average_reduce<1>(M21 {1, 4}), M21 {1, 4}));
  EXPECT_TRUE(is_near(average_reduce<0>(make_eigen_matrix<double, 1, 3>(1, 2, 3)), make_eigen_matrix<double, 1, 3>(1, 2, 3)));

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  auto m21_25 = make_eigen_matrix<double, 2, 1>(2, 5);

  EXPECT_TRUE(is_near(average_reduce<1>(m23), m21_25));
  EXPECT_TRUE(is_near(average_reduce<1>(m03_2), m21_25));
  EXPECT_TRUE(is_near(average_reduce<1>(m20_3), m21_25));
  EXPECT_TRUE(is_near(average_reduce<1>(m00_23), m21_25));

  auto i20_2 = M20::Identity(2, 2);
  auto i02_2 = M02::Identity(2, 2);
  auto i00_22 = M00::Identity(2, 2);

  EXPECT_TRUE(is_near(average_reduce<1>(i20_2), M21::Constant(0.5)));
  auto rci02_2 = average_reduce<1>(i02_2);
  EXPECT_TRUE(is_near(rci02_2, M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i02_2), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i00_22), M21::Constant(0.5)));

  EXPECT_TRUE(is_near(average_reduce<1>(M20::Identity(2, 2)), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(M02::Identity(2, 2)), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(M00::Identity(2, 2)), M21::Constant(0.5)));

  auto m13_234 = make_eigen_matrix<double, 1, 3>(2.5, 3.5, 4.5);

  EXPECT_TRUE(is_near(average_reduce<0>(m23), m13_234));
  EXPECT_TRUE(is_near(average_reduce<0>(m03_2), m13_234));
  EXPECT_TRUE(is_near(average_reduce<0>(m20_3), m13_234));
  EXPECT_TRUE(is_near(average_reduce<0>(m00_23), m13_234));

  EXPECT_EQ(average_reduce(m23), 3.5);
  EXPECT_EQ(average_reduce(m03_2), 3.5);
  EXPECT_EQ(average_reduce(m20_3), 3.5);
  EXPECT_EQ(average_reduce(m00_23), 3.5);

  auto cm23 = make_dense_writable_matrix_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});

  EXPECT_TRUE(is_near(average_reduce<1>(cm23), make_eigen_matrix<cdouble, 2, 1>(cdouble {2,5}, cdouble {5,2})));

  EXPECT_TRUE(is_near(average_reduce<0>(cm23), make_eigen_matrix<cdouble, 1, 3>(cdouble {2.5,4.5}, cdouble{3.5,3.5}, cdouble {4.5,2.5})));
}


TEST(eigen3, reduce_zero)
{
  auto z11 = M11::Identity() - M11::Identity();
  using Z11 = decltype(z11);

  auto z23 = Eigen::Replicate<Z11, 2, 3> {z11};
  auto z20_3 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {z11, 2, 3};
  auto z03_2 = Eigen::Replicate<Z11, Eigen::Dynamic, 3> {z11, 2, 3};
  auto z00_23 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 3};

  auto z13 = Eigen::Replicate<Z11, 1, 3> {z11};
  auto z21 = Eigen::Replicate<Z11, 2, 1> {z11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, z23), z13)); static_assert(zero_matrix<decltype(reduce<0>(std::plus<double>{}, z23))>);
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, z03_2), z13)); static_assert(zero_matrix<decltype(reduce<0>(std::plus<double>{}, z03_2))>);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, z20_3), z13)); static_assert(zero_matrix<decltype(reduce<0>(std::plus<double>{}, z20_3))>);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, z00_23), z13)); static_assert(zero_matrix<decltype(reduce<0>(std::plus<double>{}, z00_23))>);

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, z23), z21)); static_assert(zero_matrix<decltype(reduce<1>(std::plus<double>{}, z23))>);
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, z03_2), z21)); static_assert(zero_matrix<decltype(reduce<1>(std::plus<double>{}, z03_2))>);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, z20_3), z21)); static_assert(zero_matrix<decltype(reduce<1>(std::plus<double>{}, z20_3))>);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, z00_23), z21)); static_assert(zero_matrix<decltype(reduce<1>(std::plus<double>{}, z00_23))>);

  EXPECT_EQ((reduce(std::plus<double>{}, z23)), 0);
  EXPECT_EQ((reduce(std::plus<double>{}, z03_2)), 0);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, z23), z11));
  EXPECT_EQ((reduce(std::multiplies<double>{}, z00_23)), 0);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, z20_3), z11));

  static_assert(reduce(std::plus<double>{}, z11) == 0);
  static_assert(reduce(std::plus<double>{}, z13) == 0);
  static_assert(reduce(std::plus<double>{}, z21) == 0);
  static_assert(reduce(std::plus<double>{}, z23) == 0);
  static_assert(reduce(std::multiplies<double>{}, z23) == 0);
}


TEST(eigen3, average_zero)
{
  auto z11 = M11::Identity() - M11::Identity();

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  auto z21 = (M22::Identity() - M22::Identity()).diagonal();

  EXPECT_TRUE(is_near(average_reduce<1>(z22), z21)); static_assert(zero_matrix<decltype(average_reduce<1>(z22))>);
  EXPECT_TRUE(is_near(average_reduce<1>(z20_2), z21)); static_assert(zero_matrix<decltype(average_reduce<1>(z20_2))>);
  EXPECT_TRUE(is_near(average_reduce<1>(z02_2), z21)); static_assert(zero_matrix<decltype(average_reduce<1>(z02_2))>);
  EXPECT_TRUE(is_near(average_reduce<1>(z00_22), z21)); static_assert(zero_matrix<decltype(average_reduce<1>(z00_22))>);

  auto z12 = Eigen::Replicate<decltype(z11), 1, 2> {z11, 1, 2};

  EXPECT_TRUE(is_near(average_reduce<0>(z22), z12)); static_assert(zero_matrix<decltype(average_reduce<0>(z22))>);
  EXPECT_TRUE(is_near(average_reduce<0>(z20_2), z12)); static_assert(zero_matrix<decltype(average_reduce<0>(z20_2))>);
  EXPECT_TRUE(is_near(average_reduce<0>(z02_2), z12)); static_assert(zero_matrix<decltype(average_reduce<0>(z02_2))>);
  EXPECT_TRUE(is_near(average_reduce<0>(z00_22), z12)); static_assert(zero_matrix<decltype(average_reduce<0>(z00_22))>);

  static_assert(average_reduce(z22) == 0);
  static_assert(average_reduce(z20_2) == 0);
  static_assert(average_reduce(z02_2) == 0);
  static_assert(average_reduce(z00_22) == 0);
}


TEST(eigen3, reduce_constant)
{
  auto c11 = M11::Identity() + M11::Identity();
  using C11 = decltype(c11);

  auto c23 = Eigen::Replicate<C11, 2, 3> {c11};
  auto c20_3 = Eigen::Replicate<C11, 2, Eigen::Dynamic> {c11, 2, 3};
  auto c03_2 = Eigen::Replicate<C11, Eigen::Dynamic, 3> {c11, 2, 3};
  auto c00_23 = Eigen::Replicate<C11, Eigen::Dynamic, Eigen::Dynamic> {c11, 2, 3};

  auto c13 = Eigen::Replicate<C11, 1, 3> {c11};
  auto c21 = Eigen::Replicate<C11, 2, 1> {c11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, c23), 2 * c13)); static_assert(constant_coefficient_v<decltype(reduce<0>(std::plus<double>{}, c23))> == 4);
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, c03_2), 2 * c13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, c20_3), 2 * c13)); static_assert(constant_coefficient_v<decltype(reduce<0>(std::plus<double>{}, c20_3))> == 4);
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, c00_23), 2 * c13));

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, c23), 3 * c21)); static_assert(constant_coefficient_v<decltype(reduce<1>(std::plus<double>{}, c23))> == 6);
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, c03_2), 3 * c21)); static_assert(constant_coefficient_v<decltype(reduce<1>(std::plus<double>{}, c03_2))> == 6);
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, c20_3), 4 * c21));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, c00_23), 4 * c21));

  EXPECT_EQ((reduce(std::plus<double>{}, c23)), 12);
  EXPECT_EQ((reduce(std::plus<double>{}, c03_2)), 12);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, c23), 6 * c11));
  EXPECT_EQ((reduce(std::multiplies<double>{}, c00_23)), 64);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, c20_3), 32 * c11));

  static_assert(reduce(std::plus<double>{}, c11) == 2);
  static_assert(reduce(std::multiplies<double>{}, c11) == 2);
  static_assert(reduce(std::plus<double>{}, c13) == 6);
  static_assert(reduce(std::multiplies<double>{}, c13) == 8);
  static_assert(reduce(std::plus<double>{}, c21) == 4);
  static_assert(reduce(std::multiplies<double>{}, c21) == 4);
  static_assert(reduce(std::plus<double>{}, c23) == 12);
  static_assert(reduce(std::multiplies<double>{}, c23) == 64);
}


TEST(eigen3, reduce_constant_frac)
{
  auto c11 = (M11::Identity() + M11::Identity()).array() / (M11::Identity() + M11::Identity() + M11::Identity()).array();
  using C11 = decltype(c11);
  static_assert(are_within_tolerance(constant_coefficient_v<C11>, 2./3));

  auto c23 = Eigen::Replicate<C11, 2, 3> {c11};
  auto c20_3 = Eigen::Replicate<C11, 2, Eigen::Dynamic> {c11, 2, 3};
  auto c03_2 = Eigen::Replicate<C11, Eigen::Dynamic, 3> {c11, 2, 3};
  auto c00_23 = Eigen::Replicate<C11, Eigen::Dynamic, Eigen::Dynamic> {c11, 2, 3};

  auto c13 = Eigen::Replicate<C11, 1, 3> {c11};
  auto c21 = Eigen::Replicate<C11, 2, 1> {c11};

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, c23), 2 * c13));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, c03_2), 2 * c13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, c20_3), 2./3 * c13));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, c00_23), 2./3 * c13));

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, c23), 3 * c21)); static_assert(are_within_tolerance(constant_coefficient_v<decltype(reduce<1>(std::plus<double>{}, c23))>, 2));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, c03_2), 3 * c21)); static_assert(are_within_tolerance(constant_coefficient_v<decltype(reduce<1>(std::plus<double>{}, c03_2))>, 2));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, c20_3), 4./9 * c21));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, c00_23), 4./9 * c21));

  EXPECT_EQ((reduce(std::plus<double>{}, c23)), 4);
  EXPECT_EQ((reduce(std::plus<double>{}, c03_2)), 4);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::plus<double>{}, c23), 6 * c11));
  EXPECT_NEAR((reduce(std::multiplies<double>{}, c00_23)), 64./729, 1e-9);
  EXPECT_TRUE(is_near(reduce<1, 0>(std::multiplies<double>{}, c20_3), 32./243 * c11));

  static_assert(are_within_tolerance(reduce(std::plus<double>{}, c11), 2./3));
  static_assert(are_within_tolerance(reduce(std::multiplies<double>{}, c11), 2./3));
  static_assert(are_within_tolerance(reduce(std::plus<double>{}, c13), 2));
  static_assert(are_within_tolerance(reduce(std::multiplies<double>{}, c13), 8./27));
  static_assert(are_within_tolerance(reduce(std::plus<double>{}, c21), 4./3));
  static_assert(are_within_tolerance(reduce(std::multiplies<double>{}, c21), 4./9));
  static_assert(are_within_tolerance(reduce(std::plus<double>{}, c23), 4));
  static_assert(are_within_tolerance(reduce(std::multiplies<double>{}, c23), 64./729));
}


TEST(eigen3, average_constant)
{
  auto c11 = M11::Identity() + M11::Identity();
  using C11 = decltype(c11);

  auto c23 = Eigen::Replicate<C11, 2, 3> {c11};
  auto c20_3 = Eigen::Replicate<C11, 2, Eigen::Dynamic> {c11, 2, 3};
  auto c03_2 = Eigen::Replicate<C11, Eigen::Dynamic, 3> {c11, 2, 3};
  auto c00_23 = Eigen::Replicate<C11, Eigen::Dynamic, Eigen::Dynamic> {c11, 2, 3};

  EXPECT_TRUE(is_near(average_reduce<1>(c23), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<1>(c23))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(c20_3), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<1>(c20_3))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(c03_2), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<1>(c03_2))> == 2);
  EXPECT_TRUE(is_near(average_reduce<1>(c00_23), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<1>(c00_23))> == 2);

  EXPECT_TRUE(is_near(average_reduce<0>(c23), M13::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<0>(c23))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(c20_3), M13::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<0>(c20_3))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(c03_2), M13::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<0>(c03_2))> == 2);
  EXPECT_TRUE(is_near(average_reduce<0>(c00_23), M13::Constant(2))); static_assert(constant_coefficient_v<decltype(average_reduce<0>(c00_23))> == 2);

  static_assert(average_reduce(c23) == 2);
  static_assert(average_reduce(c20_3) == 2);
  static_assert(average_reduce(c03_2) == 2);
  static_assert(average_reduce(c00_23) == 2);
}


TEST(eigen3, average_identity)
{
  auto i22 = M22::Identity();
  auto i20_2 = Eigen::Replicate<typename M11::IdentityReturnType, 2, Eigen::Dynamic> {M11::Identity(), 2, 1}.asDiagonal();
  auto i02_2 = Eigen::Replicate<typename M11::IdentityReturnType, Eigen::Dynamic, 1> {M11::Identity(), 2, 1}.asDiagonal();
  auto i00_22 = Eigen::Replicate<typename M11::IdentityReturnType, Eigen::Dynamic, Eigen::Dynamic> {M11::Identity(), 2, 1}.asDiagonal();
  static_assert(constant_diagonal_coefficient_v<decltype(i00_22)>);

  EXPECT_TRUE(is_near(average_reduce<1>(i22), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i20_2), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i02_2), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i00_22), M21::Constant(0.5)));

  EXPECT_TRUE(is_near(average_reduce<0>(i22), M12::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(i20_2), M12::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(i02_2), M12::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(i00_22), M12::Constant(0.5)));

#if __cpp_nontype_template_args >= 201911L
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(average_reduce<0>(i22))>, 0.5));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(average_reduce<1>(i22))>, 0.5));
  static_assert(average_reduce(i22) == 0.5);
  static_assert(average_reduce(i20_2) == 0.5);
  static_assert(average_reduce(i02_2)== 0.5);
  static_assert(average_reduce(i00_22) == 0.5);
#endif

}


TEST(eigen3, average_constant_diagonal)
{
  auto c11_2 {M11::Identity() + M11::Identity()};

  auto d21_2 = c11_2.replicate<2, 1>().asDiagonal();
  auto d20_1_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();
  auto d01_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 1> {c11_2, 2, 1}.asDiagonal();
  auto d00_21_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();

  static_assert(constant_coefficient_v<decltype(average_reduce<1>(d21_2))> == 1);
  EXPECT_TRUE(is_near(average_reduce<1>(d20_1_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<1>(d01_2_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<1>(d00_21_2), M21::Constant(1)));

  static_assert(constant_coefficient_v<decltype(average_reduce<0>(d21_2))> == 1);
  EXPECT_TRUE(is_near(average_reduce<0>(d20_1_2), M12::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<0>(d01_2_2), M12::Constant(1)));
  EXPECT_TRUE(is_near(average_reduce<0>(d00_21_2), M12::Constant(1)));

#if __cpp_nontype_template_args >= 201911L
  static_assert(average_reduce(d21_2) == 1);
  static_assert(average_reduce(d20_1_2) == 1);
  static_assert(average_reduce(d01_2_2) == 1);
  static_assert(average_reduce(d00_21_2) == 1);
#else
  EXPECT_EQ(average_reduce(d21_2), 1);
  EXPECT_EQ(average_reduce(d20_1_2), 1);
  EXPECT_EQ(average_reduce(d01_2_2), 1);
  EXPECT_EQ(average_reduce(d00_21_2), 1);
#endif
}


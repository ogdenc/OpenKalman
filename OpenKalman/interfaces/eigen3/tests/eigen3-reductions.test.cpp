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

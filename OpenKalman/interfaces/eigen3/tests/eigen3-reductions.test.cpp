/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, reduce_matrix)
{
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, make_dense_writable_matrix_from<M13>(1, 2, 3)), make_dense_writable_matrix_from<M13>(1, 2, 3)));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, make_dense_writable_matrix_from<M21>(1, 4)), make_dense_writable_matrix_from<M21>(1, 4)));

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  auto f23_cust = [](const auto& arg1, const auto& arg2) { return arg1 + arg2 + 1; };

  auto m13_sum = make_dense_writable_matrix_from<M13>(5, 7, 9);
  auto m13_prod = make_dense_writable_matrix_from<M13>(4, 10, 18);
  auto m13_cust = make_dense_writable_matrix_from<M13>(6, 8, 10);

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, m23), m13_sum));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, EigenWrapper {m23}), m13_sum));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, mx3_2), m13_sum));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, m2x_3), m13_prod));
  EXPECT_TRUE(is_near(reduce<0>(std::multiplies<double>{}, mxx_23), m13_prod));
  EXPECT_TRUE(is_near(reduce<0>(f23_cust, m23), m13_cust));
  EXPECT_TRUE(is_near(reduce<0>(f23_cust, mxx_23), m13_cust));

  auto m21_sum = make_dense_writable_matrix_from<M21>(6, 15);
  auto m21_prod = make_dense_writable_matrix_from<M21>(6, 120);
  auto m21_cust = make_dense_writable_matrix_from<M21>(8, 17);

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, m23), m21_sum));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, EigenWrapper {m23}), m21_sum));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, mx3_2), m21_sum));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, m2x_3), m21_prod));
  EXPECT_TRUE(is_near(reduce<1>(std::multiplies<double>{}, mxx_23), m21_prod));
  EXPECT_TRUE(is_near(reduce<1>(f23_cust, m23), m21_cust));
  EXPECT_TRUE(is_near(reduce<1>(f23_cust, mxx_23), m21_cust));

  double m11_sum = 21;
  double m11_prod = 720;
  double m11_cust = 26;

  EXPECT_EQ((reduce(std::plus<double>{}, m23)), m11_sum);
  EXPECT_EQ((reduce(std::plus<double>{}, mx3_2)), m11_sum);
  EXPECT_EQ((reduce<1, 0>(std::plus<double>{}, m23)), m11_sum);
  EXPECT_EQ((reduce(std::multiplies<double>{}, mxx_23)), m11_prod);
  EXPECT_EQ((reduce<1, 0>(std::multiplies<double>{}, m2x_3)), m11_prod);
  EXPECT_EQ((reduce(f23_cust, mxx_23)), m11_cust);
  EXPECT_EQ((reduce<0, 1>(f23_cust, m23)), m11_cust);
  EXPECT_EQ((reduce<1, 0>(f23_cust, mxx_23)), m11_cust);
}


TEST(eigen3, reduce_eigen_general)
{
  const auto m33 = make_dense_writable_matrix_from<M33>(1, 2, 3, 4, 5, 6, 7, 8, 9);

  auto m13_sum_u = make_dense_writable_matrix_from<M13>(1, 7, 18);
  auto m13_sum_l = make_dense_writable_matrix_from<M13>(12, 13, 9);
  auto m13_sum_hl = make_dense_writable_matrix_from<M13>(12, 17, 24);
  auto m13_sum_hu = make_dense_writable_matrix_from<M13>(6, 13, 18);

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, m33.triangularView<Eigen::Upper>()), m13_sum_u));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, m33.triangularView<Eigen::Lower>()), m13_sum_l));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, m33.selfadjointView<Eigen::Upper>()), m13_sum_hu));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, m33.selfadjointView<Eigen::Lower>()), m13_sum_hl));

  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, EigenWrapper {m33.triangularView<Eigen::Upper>()}), m13_sum_u));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, EigenWrapper {m33.triangularView<Eigen::Lower>()}), m13_sum_l));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, EigenWrapper {m33.selfadjointView<Eigen::Upper>()}), m13_sum_hu));
  EXPECT_TRUE(is_near(reduce<0>(std::plus<double>{}, EigenWrapper {m33.selfadjointView<Eigen::Lower>()}), m13_sum_hl));

  auto m31_sum_u = make_dense_writable_matrix_from<M31>(6, 11, 9);
  auto m31_sum_l = make_dense_writable_matrix_from<M31>(1, 9, 24);
  auto m31_sum_hl = make_dense_writable_matrix_from<M31>(12, 17, 24);
  auto m31_sum_hu = make_dense_writable_matrix_from<M31>(6, 13, 18);

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, m33.triangularView<Eigen::Upper>()), m31_sum_u));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, m33.triangularView<Eigen::Lower>()), m31_sum_l));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, m33.selfadjointView<Eigen::Upper>()), m31_sum_hu));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, m33.selfadjointView<Eigen::Lower>()), m31_sum_hl));

  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, EigenWrapper {m33.triangularView<Eigen::Upper>()}), m31_sum_u));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, EigenWrapper {m33.triangularView<Eigen::Lower>()}), m31_sum_l));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, EigenWrapper {m33.selfadjointView<Eigen::Upper>()}), m31_sum_hu));
  EXPECT_TRUE(is_near(reduce<1>(std::plus<double>{}, EigenWrapper {m33.selfadjointView<Eigen::Lower>()}), m31_sum_hl));

  double m11_sum_u = 26;
  double m11_sum_l = 34;
  double m11_sum_hu = 37;
  double m11_sum_hl = 53;

  EXPECT_EQ((reduce(std::plus<double>{}, m33.triangularView<Eigen::Upper>())), m11_sum_u);
  EXPECT_EQ((reduce(std::plus<double>{}, m33.triangularView<Eigen::Lower>())), m11_sum_l);
  EXPECT_EQ((reduce(std::plus<double>{}, m33.selfadjointView<Eigen::Upper>())), m11_sum_hu);
  EXPECT_EQ((reduce(std::plus<double>{}, m33.selfadjointView<Eigen::Lower>())), m11_sum_hl);

  EXPECT_EQ((reduce(std::plus<double>{}, EigenWrapper {m33.triangularView<Eigen::Upper>()})), m11_sum_u);
  EXPECT_EQ((reduce(std::plus<double>{}, EigenWrapper {m33.triangularView<Eigen::Lower>()})), m11_sum_l);
  EXPECT_EQ((reduce(std::plus<double>{}, EigenWrapper {m33.selfadjointView<Eigen::Upper>()})), m11_sum_hu);
  EXPECT_EQ((reduce(std::plus<double>{}, EigenWrapper {m33.selfadjointView<Eigen::Lower>()})), m11_sum_hl);
}


TEST(eigen3, average_matrix)
{
  EXPECT_TRUE(is_near(average_reduce<1>(M21 {1, 4}), M21 {1, 4}));
  EXPECT_TRUE(is_near(average_reduce<0>(make_eigen_matrix<double, 1, 3>(1, 2, 3)), make_eigen_matrix<double, 1, 3>(1, 2, 3)));

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  auto m21_25 = make_eigen_matrix<double, 2, 1>(2, 5);

  EXPECT_TRUE(is_near(average_reduce<1>(m23), m21_25));
  EXPECT_TRUE(is_near(average_reduce<1>(EigenWrapper {m23}), m21_25));
  EXPECT_TRUE(is_near(average_reduce<1>(mx3_2), m21_25));
  EXPECT_TRUE(is_near(average_reduce<1>(m2x_3), m21_25));
  EXPECT_TRUE(is_near(average_reduce<1>(mxx_23), m21_25));

  auto i20_2 = M2x::Identity(2, 2);
  auto i02_2 = Mx2::Identity(2, 2);
  auto i00_22 = Mxx::Identity(2, 2);

  EXPECT_TRUE(is_near(average_reduce<1>(i20_2), M21::Constant(0.5)));
  auto rci02_2 = average_reduce<1>(i02_2);
  EXPECT_TRUE(is_near(rci02_2, M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i02_2), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(i00_22), M21::Constant(0.5)));

  EXPECT_TRUE(is_near(average_reduce<1>(M2x::Identity(2, 2)), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(Mx2::Identity(2, 2)), M21::Constant(0.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(Mxx::Identity(2, 2)), M21::Constant(0.5)));

  auto m13_234 = make_eigen_matrix<double, 1, 3>(2.5, 3.5, 4.5);

  EXPECT_TRUE(is_near(average_reduce<0>(m23), m13_234));
  EXPECT_TRUE(is_near(average_reduce<0>(mx3_2), m13_234));
  EXPECT_TRUE(is_near(average_reduce<0>(m2x_3), m13_234));
  EXPECT_TRUE(is_near(average_reduce<0>(mxx_23), m13_234));

  EXPECT_EQ(average_reduce(m23), 3.5);
  EXPECT_EQ(average_reduce(mx3_2), 3.5);
  EXPECT_EQ(average_reduce(m2x_3), 3.5);
  EXPECT_EQ(average_reduce(mxx_23), 3.5);

  auto cm23 = make_dense_writable_matrix_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});

  EXPECT_TRUE(is_near(average_reduce<1>(cm23), make_eigen_matrix<cdouble, 2, 1>(cdouble {2,5}, cdouble {5,2})));

  EXPECT_TRUE(is_near(average_reduce<0>(cm23), make_eigen_matrix<cdouble, 1, 3>(cdouble {2.5,4.5}, cdouble{3.5,3.5}, cdouble {4.5,2.5})));
}

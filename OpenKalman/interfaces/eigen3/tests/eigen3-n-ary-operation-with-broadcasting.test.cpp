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
}


TEST(eigen3, nullary_operation)
{
  // One operation for the entire matrix
  auto m23 = make_dense_writable_matrix_from<M23>(5.5, 5.5, 5.5, 5.5, 5.5, 5.5);
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23>([]{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions{2}, Dimensions<3>{}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions<2>{}, Dimensions{3}}, []{return 5.5;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions{2}, Dimensions{3}}, []{return 5.5;}), m23));

  auto m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 1, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M23>([](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::tuple {Dimensions{2}, Dimensions{3}}, [](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));

  // One operation for each element
  m23 = make_dense_writable_matrix_from<M23>(5.4, 5.5, 5.6, 5.7, 5.8, 5.9);
  EXPECT_TRUE(is_near(n_ary_operation<M23>(std::index_sequence<0, 1>{}, []{return 5.4;}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}, []{return 5.8;}, []{return 5.9;}), m23));

  m23p = make_dense_writable_matrix_from<M23>(0, 0, 0, 1, 2, 3);
  EXPECT_TRUE(is_near(n_ary_operation<M23>(std::index_sequence<0, 1>{}, []{return 5.4;}, []{return 5.5;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.7 + r + c;}, [](std::size_t r, std::size_t c){return 5.8 + r + c;}, [](std::size_t r, std::size_t c){return 5.9 + r + c;}), m23 + m23p));

  // One operation for each row
  m23 = make_dense_writable_matrix_from<M23>(5.5, 5.5, 5.5, 5.6, 5.6, 5.6);
  EXPECT_TRUE(is_near(n_ary_operation<M23>(std::index_sequence<0>{}, []{return 5.5;}, []{return 5.6;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::index_sequence<0>{}, std::tuple {Dimensions<2>{}, Dimensions{3}}, []{return 5.5;}, []{return 5.6;}), m23));

  m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 0, 1, 2);
  EXPECT_TRUE(is_near(n_ary_operation<M23>(std::index_sequence<0>{}, [](std::size_t r, std::size_t c){return 5.5 + c;}, [](std::size_t r, std::size_t c){return 5.6 + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::index_sequence<0>{}, std::tuple {Dimensions<2>{}, Dimensions{3}}, [](std::size_t r, std::size_t c){return 5.5 + c;}, [](std::size_t r, std::size_t c){return 5.6 + c;}), m23 + m23p));

  // One operation for each column
  m23 = make_dense_writable_matrix_from<M23>(5.5, 5.6, 5.7, 5.5, 5.6, 5.7);
  EXPECT_TRUE(is_near(n_ary_operation<M23>(std::index_sequence<1>{}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::index_sequence<1>{}, std::tuple {Dimensions{2}, Dimensions<3>{}}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}), m23));

  m23p = make_dense_writable_matrix_from<M23>(0, 0, 0, 1, 0, 1);
  EXPECT_TRUE(is_near(n_ary_operation<M23>(std::index_sequence<1>{}, [](std::size_t r, std::size_t c){return 5.5 + r;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.7 + r;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M00>(std::index_sequence<1>{}, std::tuple {Dimensions{2}, Dimensions<3>{}}, [](std::size_t r, std::size_t c){return 5.5 + r;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.7 + r;}), m23 + m23p));
}


TEST(eigen3, unary_operation)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = make_dense_writable_matrix_from<M20>(m23);
  auto m03_2 = make_dense_writable_matrix_from<M03>(m23);
  auto m00_23 = make_dense_writable_matrix_from<M00>(m23);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg) {return arg + arg;}, m23), m23 * 2));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg){return 3 * arg;}, m00_23), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg){return 3 * arg;}, m20_3), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg){return 3 * arg;}, m03_2), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg){return 3 * arg;}, m20_3), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg) {return -arg;}, m23), -m23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, std::negate<double>{}, m23), -m23));

  auto m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 1, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg, std::size_t r, std::size_t c) {return arg + arg + r + c;}, m23), 2*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m00_23), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m20_3), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m03_2), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m20_3), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg, std::size_t r, std::size_t c) {return -arg + r + c;}, m23), -m23 + m23p));

  auto b23 = make_dense_writable_matrix_from<eigen_matrix_t<bool, 2, 3>>(true, false, true, false, true, false);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg) {return not arg;}, b23), not b23.array()));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, std::logical_not<bool>{}, b23), not b23.array()));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg){return arg + arg;}, m23), m23 * 2));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg){return 3 * arg;}, m20_3), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg){return 3 * arg;}, m03_2), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg){return 3 * arg;}, m00_23), m23 * 3));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg, std::size_t r, std::size_t c){return arg + arg + r + c;}, m23), 2*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m20_3), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m03_2), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m00_23), 3*m23 + m23p));

  auto m31 = make_dense_writable_matrix_from<M31>(1, 2, 3);
  auto m30_1 = make_dense_writable_matrix_from<M30>(m31);
  auto m01_3 = make_dense_writable_matrix_from<M01>(m31);
  auto m00_31 = make_dense_writable_matrix_from<M00>(m31);
  auto m32 = make_dense_writable_matrix_from<M32>(1, 1, 2, 2, 3, 3);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions<2>{}}, [](const auto& arg){return arg + arg + arg;}, m31), 3 * m32));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions<2>{}}, [](const auto& arg){return 3 * arg;}, m00_31), 3 * m32));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{3}, Dimensions<2>{}}, [](const auto& arg){return 3 * arg;}, m00_31), 3 * m32));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions{2}}, [](const auto& arg){return 3 * arg;}, m01_3), 3 * m32));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{3}, Dimensions{2}}, [](const auto& arg){return 3 * arg;}, m30_1), 3 * m32));

  auto m32p = make_dense_writable_matrix_from<M32>(0, 1, 1, 2, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions<2>{}}, [](const auto& arg, std::size_t r, std::size_t c){return arg + arg + arg + r + c;}, m31), 3*m32 + m32p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions<2>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m00_31), 3*m32 + m32p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{3}, Dimensions<2>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m00_31), 3*m32 + m32p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions{2}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m01_3), 3*m32 + m32p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{3}, Dimensions{2}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m30_1), 3*m32 + m32p));

  auto m13 = make_dense_writable_matrix_from<M13>(1, 2, 3);
  auto m10_3 = make_dense_writable_matrix_from<M10>(m13);
  auto m03_1 = make_dense_writable_matrix_from<M03>(m13);
  auto m00_13 = make_dense_writable_matrix_from<M00>(m13);
  auto m23a = make_dense_writable_matrix_from<M23>(1, 2, 3, 1, 2, 3);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg){return arg + arg + arg;}, m13), 3 * m23a));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg){return 3 * arg;}, m00_13), 3 * m23a));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg){return 3 * arg;}, m00_13), 3 * m23a));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg){return 3 * arg;}, m03_1), 3 * m23a));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg){return 3 * arg;}, m10_3), 3 * m23a));

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg, std::size_t r, std::size_t c){return arg + arg + arg + r + c;}, m13), 3*m23a + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m00_13), 3*m23a + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m00_13), 3*m23a + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m03_1), 3*m23a + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m10_3), 3*m23a + m23p));
}


TEST(eigen3, unary_operation_in_place)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = make_dense_writable_matrix_from<M20>(m23);
  auto m03_2 = make_dense_writable_matrix_from<M03>(m23);
  auto m00_23 = make_dense_writable_matrix_from<M00>(m23);

  EXPECT_TRUE(is_near(n_ary_operation([](auto& arg){return arg *= 3;}, m20_3), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation([](auto& arg){return arg *= 4;}, m03_2), m23 * 4));
  EXPECT_TRUE(is_near(n_ary_operation([](auto& arg){return arg *= 5;}, m00_23), m23 * 5));

  const auto m33_123 = make_dense_writable_matrix_from<M33>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  const auto m33_234 = make_dense_writable_matrix_from<M33>(
    2, 1, 1,
    1, 3, 1,
    1, 1, 4);

  auto m33 = M33 {m33_123}; n_ary_operation([](auto& x){ x += 1; }, m33); EXPECT_TRUE(is_near(m33, m33_234));
  auto m30 = M30 {m33_123}; n_ary_operation([](auto& x){ x += 1; }, m30); EXPECT_TRUE(is_near(m30, m33_234));
  auto m03 = M03 {m33_123}; n_ary_operation([](auto& x){ x += 1; }, m03); EXPECT_TRUE(is_near(m03, m33_234));
  auto m00 = M00 {m33_123}; n_ary_operation([](auto& x){ x += 1; }, m00); EXPECT_TRUE(is_near(m00, m33_234));

  EXPECT_TRUE(is_near(n_ary_operation([](double& x){ return x + 1; }, M33 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(n_ary_operation([](const double& x){ return x + 1; }, M30 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(n_ary_operation([](double x){ return x + 1; }, M03 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(n_ary_operation([](const double x){ return x + 1; }, M00 {m33_123}), m33_234));

  EXPECT_TRUE(is_near(n_ary_operation([](const double& x){ return x + 1; }, m33_123), m33_234));
  EXPECT_TRUE(is_near(m33_123 + M33::Constant(1), m33_234));

  auto m33_147 = make_eigen_matrix<double, 3, 3>(
    1, 1, 2,
    1, 4, 3,
    2, 3, 7);

  m33 = m33_123; n_ary_operation([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m33); EXPECT_TRUE(is_near(m33, m33_147));
  m30 = m33_123; n_ary_operation([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m30); EXPECT_TRUE(is_near(m30, m33_147));
  m03 = m33_123; n_ary_operation([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m03); EXPECT_TRUE(is_near(m03, m33_147));
  m00 = m33_123; n_ary_operation([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m00); EXPECT_TRUE(is_near(m00, m33_147));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, M33 {m33_123}), m33_147));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, M30 {m33_123}), m33_147));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, M03 {m33_123}), m33_147));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, M00 {m33_123}), m33_147));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, m33_123), m33_147));
}


TEST(eigen3, binary_operation)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m20_3 = make_dense_writable_matrix_from<M20>(m23);
  auto m03_2 = make_dense_writable_matrix_from<M03>(m23);
  auto m00_23 = make_dense_writable_matrix_from<M00>(m23);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2) {return arg1 + arg2;}, m23, 2 * m23), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m00_23, 2 * m23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m20_3, 2 * m23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m03_2, 2 * m23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m20_3, 2 * m23), m23 * 5));

  auto m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 1, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c) {return arg1 + arg2 + r + c;}, m23, 2 * m23), m23 * 3 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m00_23, 2 * m23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m20_3, 2 * m23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m03_2, 2 * m23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m20_3, 2 * m23), m23 * 5 + m23p));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m23, 2 * m00_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m20_3, 2 * m00_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m03_2, 2 * m00_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m00_23, 2 * m00_23), m23 * 5));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m23, 2 * m00_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m20_3, 2 * m00_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m03_2, 2 * m00_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m00_23, 2 * m00_23), m23 * 5 + m23p));

  auto m13 = make_dense_writable_matrix_from<M13>(1, 2, 3);
  auto m10_3 = make_dense_writable_matrix_from<M10>(m13);
  auto m03_1 = make_dense_writable_matrix_from<M03>(m13);
  auto m00_13 = make_dense_writable_matrix_from<M00>(m13);
  m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 1, 2, 3);
  m20_3 = m23;
  m03_2 = m23;
  m00_23 = m23;

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2) {return arg1 + arg2;}, m13, 2 * m23), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, std::plus<double>{}, m13, 2 * m23), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, std::minus<double>{}, m13, 2 * m23), -m23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m23, 2 * m13), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m00_23, 2 * m13), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return arg1 + 3 * arg2;}, 2 * m10_3, m23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, 2 * m03_2, 2 * m13), m23 * 8));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2){return arg1 * arg2;}, m10_3, m13), m23.array().square()));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::multiplies<double>{}, m10_3, m13), m23.array().square()));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::divides<double>{}, m10_3, m23), M23::Constant(1)));

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c) {return arg1 + arg2 + r + c;}, m13, 2 * m23), m23 * 3 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m23, 2 * m13), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m00_23, 2 * m13), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return arg1 + 3 * arg2 + r + c;}, 2 * m10_3, m23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, 2 * m03_2, 2 * m13), m23 * 8 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return arg1 * arg2 + r + c;}, m10_3, m13), m23.array().square().matrix() + m23p));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m13, 2 * m00_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m20_3, 2 * m00_13), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m03_1, 2 * m00_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m00_23, 2 * m00_13), m23 * 5));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m13, 2 * m00_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m20_3, 2 * m00_13), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m03_1, 2 * m00_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m00_23, 2 * m00_13), m23 * 5 + m23p));

  auto m13_eq = make_dense_writable_matrix_from<M13>(1, 3, 3);
  auto m13_gt = make_dense_writable_matrix_from<M13>(2, 2, 4);
  auto m13_ge = make_dense_writable_matrix_from<M13>(2, 1, 3);
  auto b23 = make_dense_writable_matrix_from<eigen_matrix_t<bool, 2, 3>>(true, false, true, true, false, true);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::equal_to<double>{}, m10_3, m13_eq), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::not_equal_to<double>{}, m10_3, m13_gt), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::greater<double>{}, m13_gt, m10_3), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::less<double>{}, m10_3, m13_gt), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::greater_equal<double>{}, m13_ge, m10_3), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::less_equal<double>{}, m10_3, m13_ge), b23));

  auto b13 = make_dense_writable_matrix_from<eigen_matrix_t<bool, 1, 3>>(true, false, true);
  auto b13_and = make_dense_writable_matrix_from<eigen_matrix_t<bool, 1, 3>>(true, true, true);
  auto b13_or = make_dense_writable_matrix_from<eigen_matrix_t<bool, 1, 3>>(false, false, false);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::logical_and<bool>{}, b13, b13_and), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::logical_or<bool>{}, b13, b13_or), b23));
}


TEST(eigen3, ternary_operation)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 1, 2, 3);
  auto m20_3 = make_dense_writable_matrix_from<M20>(m23);
  auto m03_2 = make_dense_writable_matrix_from<M03>(m23);
  auto m00_23 = make_dense_writable_matrix_from<M00>(m23);
  auto m13 = make_dense_writable_matrix_from<M13>(1, 2, 3);
  auto m10_3 = make_dense_writable_matrix_from<M10>(m13);
  auto m03_1 = make_dense_writable_matrix_from<M03>(m13);
  auto m00_13 = make_dense_writable_matrix_from<M00>(m13);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3) {return arg1 + arg2 + arg3;}, m23, 2 * m23, m00_23), m23 * 4));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m00_23, 2 * m23, m13), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m20_3, 2 * m23, m03_1), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m03_2, 2 * m23, m10_3), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m20_3, 2 * m23, m00_13), m23 * 6));

  auto m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 1, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c) {return arg1 + arg2 + arg3 + r + c;}, m23, 2 * m23, m00_23), m23 * 4 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m00_23, 2 * m23, m13), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m20_3, 2 * m23, m03_1), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m03_2, 2 * m23, m10_3), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m20_3, 2 * m23, m00_13), m23 * 6 + m23p));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m23, 2 * m00_23, m00_23), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m20_3, 2 * m00_23, m03_1), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m03_2, 2 * m00_23, m10_3), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m00_23, 2 * m00_23, m13), m23 * 6));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m23, 2 * m00_23, m00_23), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m20_3, 2 * m00_23, m03_1), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m03_2, 2 * m00_23, m10_3), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m00_23, 2 * m00_23, m13), m23 * 6 + m23p));
}


TEST(eigen3, randomize)
{
  using N = std::normal_distribution<double>;

  M22 m22, m22_true;
  M23 m23, m23_true;
  M32 m32, m32_true;
  auto m20_2 = make_default_dense_writable_matrix_like<M22>(Dimensions<2>{}, 2);
  auto m20_3 = make_default_dense_writable_matrix_like<M23>(Dimensions<2>{}, 3);
  auto m02_2 = make_default_dense_writable_matrix_like<M22>(2, Dimensions<2>{});
  auto m02_3 = make_default_dense_writable_matrix_like<M32>(3, Dimensions<2>{});
  auto m30_2 = make_default_dense_writable_matrix_like<M32>(Dimensions<3>{}, 2);
  auto m03_2 = make_default_dense_writable_matrix_like<M23>(2, Dimensions<3>{});
  auto m00_22 = make_default_dense_writable_matrix_like<M22>(2, 2);
  auto m00_32 = make_default_dense_writable_matrix_like<M32>(3, 2);
  auto m00_23 = make_default_dense_writable_matrix_like<M23>(2, 3);

  // Test just using the parameters, rather than a constructed distribution.
  m22 = randomize<M22>(N {0.0, 0.7});
  m20_2 = randomize<M20>(std::tuple {Dimensions<2>{}, 2}, N {0.0, 1.0});
  m20_3 = randomize<M20>(std::tuple {Dimensions<2>{}, 3}, N {0.0, 0.7});
  m02_2 = randomize<M02>(std::tuple {2, Dimensions<2>{}}, N {0.0, 1.0});
  m02_3 = randomize<M02>(std::tuple {3, Dimensions<2>{}}, N {0.0, 0.7});
  m00_22 = randomize<M00>(std::tuple{2, 2}, N {0.0, 1.0});

  // Single distribution for the entire matrix.
  m22 = M22::Zero();
  m20_2 = M20::Zero(2, 2);
  m02_2 = M02::Zero(2, 2);
  m00_22 = M00::Zero(2, 2);

  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<M22>(N {1.0, 0.3})) / (i + 1);
    m20_2 = (m20_2 * i + randomize<M20>(std::tuple {Dimensions<2>{}, 2}, N {1.0, 0.3})) / (i + 1);
    m02_2 = (m02_2 * i + randomize<M02>(std::tuple {2, Dimensions<2>{}}, N {1.0, 0.3})) / (i + 1);
    m00_22 = (m00_22 * i + randomize<M00>(std::tuple{2, 2}, N {1.0, 0.3})) / (i + 1);
  }

  m22_true = M22::Constant(1);
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m20_2, m22_true, 0.1));
  EXPECT_FALSE(is_near(m20_2, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m02_2, m22_true, 0.1));
  EXPECT_FALSE(is_near(m02_2, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m00_22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m00_22, m22_true, 1e-8));

  // One distribution for each element.
  m22 = M22::Zero();

  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<M22>(std::index_sequence<0, 1>{}, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})) / (i + 1);
  }

  m22_true = MatrixTraits<M22>::make(1, 2, 3, 4);
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));

  // One distribution for each row.
  m32 = M32::Zero();
  m30_2 = M30::Zero(3, 2);

  m23 = M23::Zero();
  m20_3 = M20::Zero(2, 3);

  for (int i=0; i<10; i++)
  {
    m32 = (m32 * i + randomize<M32>(std::index_sequence<0>{}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
    m30_2 = (m30_2 * i + randomize<M30>(std::index_sequence<0>{}, std::tuple {Dimensions<3>{}, 2}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);

    m23 = (m23 * i + randomize<M23>(std::index_sequence<0>{}, N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    m20_3 = (m20_3 * i + randomize<M20>(std::index_sequence<0>{}, std::tuple {Dimensions<2>{}, 3}, N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
  }

  m32_true = MatrixTraits<M32>::make(1, 1, 2, 2, 3, 3);
  EXPECT_TRUE(is_near(m32, m32_true, 0.1));
  EXPECT_FALSE(is_near(m32, m32_true, 1e-8));
  EXPECT_TRUE(is_near(m30_2, m32_true, 0.1));
  EXPECT_FALSE(is_near(m30_2, m32_true, 1e-8));

  m23_true = MatrixTraits<M23>::make(1, 1, 1, 2, 2, 2);
  EXPECT_TRUE(is_near(m23, m23_true, 0.1));
  EXPECT_FALSE(is_near(m23, m23_true, 1e-8));
  EXPECT_TRUE(is_near(m20_3, m23_true, 0.1));
  EXPECT_FALSE(is_near(m20_3, m23_true, 1e-8));

  // One distribution for each column.
  m32 = M32::Zero();
  m02_3 = M02::Zero(3, 2);

  m23 = M23::Zero();
  m03_2 = M03::Zero(2, 3);

  for (int i=0; i<10; i++)
  {
    m32 = (m32 * i + randomize<M32>(std::index_sequence<1>{}, N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    m02_3 = (m02_3 * i + randomize<M02>(std::index_sequence<1>{}, std::tuple {3, Dimensions<2>{}}, N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);

    m23 = (m23 * i + randomize<M23>(std::index_sequence<1>{}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
    m03_2 = (m03_2 * i + randomize<M03>(std::index_sequence<1>{}, std::tuple {2, Dimensions<3>{}}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
  }

  m32_true = MatrixTraits<M32>::make(1, 2, 1, 2, 1, 2);
  EXPECT_TRUE(is_near(m32, m32_true, 0.1));
  EXPECT_FALSE(is_near(m32, m32_true, 1e-8));
  EXPECT_TRUE(is_near(m02_3, m32_true, 0.1));
  EXPECT_FALSE(is_near(m02_3, m32_true, 1e-8));

  m23_true = MatrixTraits<M23>::make(1, 2, 3, 1, 2, 3);
  EXPECT_TRUE(is_near(m23, m23_true, 0.1));
  EXPECT_FALSE(is_near(m23, m23_true, 1e-8));
  EXPECT_TRUE(is_near(m03_2, m23_true, 0.1));
  EXPECT_FALSE(is_near(m03_2, m23_true, 1e-8));
}


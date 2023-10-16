/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, nullary_operation)
{
  // One operation for the entire matrix
  auto m23 = make_dense_writable_matrix_from<M23>(5.5, 5.5, 5.5, 5.5, 5.5, 5.5);
  auto m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 1, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation<Mxx>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<Mxx>(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<Mxx>(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<Mxx>(std::tuple {Dimensions{2}, Dimensions{3}}, [](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M23>([](std::size_t r, std::size_t c){return 5.5 + r + c;}), m23 + m23p));

  // One operation for each element
  m23 = make_dense_writable_matrix_from<M23>(5.4, 5.5, 5.6, 5.7, 5.8, 5.9);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 0, 1>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, []{return 5.4;}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}, []{return 5.8;}, []{return 5.9;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 0, 1>([]{return 5.4;}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}, []{return 5.8;}, []{return 5.9;}), m23));

  m23p = make_dense_writable_matrix_from<M23>(0, 0, 0, 1, 2, 3);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 1, 0>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, []{return 5.4;}, [](std::size_t r, std::size_t c){return 5.7 + r + c;}, []{return 5.5;}, [](std::size_t r, std::size_t c){return 5.8 + r + c;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.9 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 1, 0>([]{return 5.4;}, [](std::size_t r, std::size_t c){return 5.7 + r + c;}, []{return 5.5;}, [](std::size_t r, std::size_t c){return 5.8 + r + c;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.9 + r + c;}), m23 + m23p));

  // One operation for each row
  m23 = make_dense_writable_matrix_from<M23>(5.5, 5.5, 5.5, 5.6, 5.6, 5.6);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 0>(std::tuple {Dimensions<2>{}, Dimensions{3}}, []{return 5.5;}, []{return 5.6;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 0>([]{return 5.5;}, []{return 5.6;}), m23));

  m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 0, 1, 2);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 0>(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](std::size_t r, std::size_t c){return 5.5 + c;}, [](std::size_t r, std::size_t c){return 5.6 + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 0>([](std::size_t r, std::size_t c){return 5.5 + c;}, [](std::size_t r, std::size_t c){return 5.6 + c;}), m23 + m23p));

  // One operation for each column
  m23 = make_dense_writable_matrix_from<M23>(5.5, 5.6, 5.7, 5.5, 5.6, 5.7);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 1>(std::tuple {Dimensions{2}, Dimensions<3>{}}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 1>([]{return 5.5;}, []{return 5.6;}, []{return 5.7;}), m23));

  m23p = make_dense_writable_matrix_from<M23>(0, 0, 0, 1, 0, 1);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 1>(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](std::size_t r, std::size_t c){return 5.5 + r;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.7 + r;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 1>([](std::size_t r, std::size_t c){return 5.5 + r;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.7 + r;}), m23 + m23p));
}


TEST(eigen3, unary_operation)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  M2x m2x_3 {m23};
  Mx3 mx3_2 {m23};
  Mxx mxx_23 {m23};
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg) {return arg + arg;}, m23), m23 * 2));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg) {return arg + arg;}, Eigen3::make_eigen_wrapper(m23)), m23 * 2));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg){return 3 * arg;}, mxx_23), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg){return 3 * arg;}, m2x_3), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg){return 3 * arg;}, mx3_2), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg){return 3 * arg;}, m2x_3), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg) {return -arg;}, m23), -m23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, std::negate<double>{}, m23), -m23));

  auto m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 1, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg, std::size_t r, std::size_t c) {return arg + arg + r + c;}, m23), 2*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg, std::size_t r, std::size_t c) {return arg + arg + r + c;}, Eigen3::make_eigen_wrapper(m23)), 2*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mxx_23), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m2x_3), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mx3_2), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m2x_3), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](auto arg, std::size_t r, std::size_t c) {return -arg + r + c;}, m23), -m23 + m23p));

  auto b23 = make_dense_writable_matrix_from<eigen_matrix_t<bool, 2, 3>>(true, false, true, false, true, false);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg) {return not arg;}, b23), not b23.array()));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, std::logical_not<bool>{}, b23), not b23.array()));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg){return arg + arg;}, m23), m23 * 2));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg){return 3 * arg;}, m2x_3), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg){return 3 * arg;}, mx3_2), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg){return 3 * arg;}, mxx_23), m23 * 3));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg, std::size_t r, std::size_t c){return arg + arg + r + c;}, m23), 2*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m2x_3), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mx3_2), 3*m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mxx_23), 3*m23 + m23p));

  auto m31 = make_dense_writable_matrix_from<M31>(1, 2, 3);
  M3x m3x_1 {m31};
  Mx1 mx1_3 {m31};
  Mxx mxx_31 {m31};
  auto m32 = make_dense_writable_matrix_from<M32>(1, 1, 2, 2, 3, 3);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions<2>{}}, [](const auto& arg){return arg + arg + arg;}, m31), 3 * m32));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions<2>{}}, [](const auto& arg){return 3 * arg;}, mxx_31), 3 * m32));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{3}, Dimensions<2>{}}, [](const auto& arg){return 3 * arg;}, mxx_31), 3 * m32));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions{2}}, [](const auto& arg){return 3 * arg;}, mx1_3), 3 * m32));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{3}, Dimensions{2}}, [](const auto& arg){return 3 * arg;}, m3x_1), 3 * m32));

  auto m32p = make_dense_writable_matrix_from<M32>(0, 1, 1, 2, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions<2>{}}, [](const auto& arg, std::size_t r, std::size_t c){return arg + arg + arg + r + c;}, m31), 3*m32 + m32p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions<2>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mxx_31), 3*m32 + m32p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{3}, Dimensions<2>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mxx_31), 3*m32 + m32p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<3>{}, Dimensions{2}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mx1_3), 3*m32 + m32p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{3}, Dimensions{2}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m3x_1), 3*m32 + m32p));

  auto m13 = make_dense_writable_matrix_from<M13>(1, 2, 3);
  M1x m1x_3 {m13};
  Mx3 mx3_1 {m13};
  Mxx mxx_13 {m13};
  auto m23a = make_dense_writable_matrix_from<M23>(1, 2, 3, 1, 2, 3);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg){return arg + arg + arg;}, m13), 3 * m23a));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg){return 3 * arg;}, mxx_13), 3 * m23a));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg){return 3 * arg;}, mxx_13), 3 * m23a));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg){return 3 * arg;}, mx3_1), 3 * m23a));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg){return 3 * arg;}, m1x_3), 3 * m23a));

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg, std::size_t r, std::size_t c){return arg + arg + arg + r + c;}, m13), 3*m23a + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mxx_13), 3*m23a + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mxx_13), 3*m23a + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, mx3_1), 3*m23a + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg, std::size_t r, std::size_t c){return 3 * arg + r + c;}, m1x_3), 3*m23a + m23p));
}


TEST(eigen3, unary_operation_in_place)
{
  const auto m23c = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m23 {m23c};
  M2x m2x_3 {m23};
  Mx3 mx3_2 {m23};
  Mxx mxx_23 {m23};
  auto ewm23 = Eigen3::make_eigen_wrapper(m23);

  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& arg){return arg *= 2;}, m23), m23c * 2));
  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& arg){return arg *= 3;}, ewm23), m23c * 6));
  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& arg){return arg *= 4;}, Eigen3::make_eigen_wrapper(m23)), m23c * 24));

  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& arg){return arg *= 3;}, m2x_3), m23c * 3));
  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& arg){return arg *= 4;}, mx3_2), m23c * 4));
  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& arg){return arg *= 5;}, mxx_23), m23c * 5));

  const auto m33_123 = make_dense_writable_matrix_from<M33>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  const auto m33_234 = make_dense_writable_matrix_from<M33>(
    2, 1, 1,
    1, 3, 1,
    1, 1, 4);

  auto m33 = M33 {m33_123}; unary_operation_in_place([](auto& x){ return x += 1; }, m33); EXPECT_TRUE(is_near(m33, m33_234));
  auto m30 = M3x {m33_123}; unary_operation_in_place([](auto& x){ return x += 1; }, m30); EXPECT_TRUE(is_near(m30, m33_234));
  auto m03 = Mx3 {m33_123}; unary_operation_in_place([](auto& x){ return x += 1; }, m03); EXPECT_TRUE(is_near(m03, m33_234));
  auto mxx = Mxx {m33_123}; unary_operation_in_place([](auto& x){ return x += 1; }, mxx); EXPECT_TRUE(is_near(mxx, m33_234));
  auto ewm33 = Eigen3::make_eigen_wrapper(M33 {m33_123}); unary_operation_in_place([](auto& x){ return x += 1; }, ewm33); EXPECT_TRUE(is_near(ewm33, m33_234));

  EXPECT_TRUE(is_near(unary_operation_in_place([](double& x){ ++x; }, M33 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(unary_operation_in_place([](double& x){ ++x; }, M3x {m33_123}), m33_234));
  EXPECT_TRUE(is_near(unary_operation_in_place([](double& x){ ++x; }, Mx3 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(unary_operation_in_place([](double& x){ ++x; }, Mxx {m33_123}), m33_234));

  m33 = m33_123;
  EXPECT_TRUE(is_near(unary_operation_in_place([](double& x){ ++x; }, m33), m33_234));
  EXPECT_TRUE(is_near(m33, m33_234));

  const auto m33_147 = make_eigen_matrix<double, 3, 3>(
    1, 1, 2,
    1, 4, 3,
    2, 3, 7);

  m33 = m33_123; unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m33); EXPECT_TRUE(is_near(m33, m33_147));
  m30 = m33_123; unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m30); EXPECT_TRUE(is_near(m30, m33_147));
  m03 = m33_123; unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m03); EXPECT_TRUE(is_near(m03, m33_147));
  mxx = m33_123; unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, mxx); EXPECT_TRUE(is_near(mxx, m33_147));

  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ return x += i + j; }, M33 {m33_123}), m33_147));
  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ return x += i + j; }, M3x {m33_123}), m33_147));
  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ return x += i + j; }, Mx3 {m33_123}), m33_147));
  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ return x += i + j; }, Mxx {m33_123}), m33_147));

  m33 = m33_123;
  EXPECT_TRUE(is_near(unary_operation_in_place([](auto& x, std::size_t i, std::size_t j){ return x += i + j; }, m33), m33_147));
  EXPECT_TRUE(is_near(m33, m33_147));
}


TEST(eigen3, binary_operation)
{
  auto m11_2 = make_dense_writable_matrix_from<M11>(2);
  auto m11_5 = make_dense_writable_matrix_from<M11>(5);
  auto m11_7 = make_dense_writable_matrix_from<M11>(7);
  auto m11_10 = make_dense_writable_matrix_from<M11>(10);

  EXPECT_TRUE(is_near(n_ary_operation(std::multiplies<double>{}, m11_2, m11_5), m11_10));
  EXPECT_TRUE(is_near(n_ary_operation(std::multiplies<double>{}, M1x{m11_2}, m11_5), m11_10));
  EXPECT_TRUE(is_near(n_ary_operation(std::multiplies<double>{}, Mx1{m11_2}, m11_5), m11_10));
  EXPECT_TRUE(is_near(n_ary_operation(std::multiplies<double>{}, Mxx{m11_2}, m11_5), m11_10));
  EXPECT_TRUE(is_near(n_ary_operation(std::multiplies<double>{}, m11_2, M1x{m11_5}), m11_10));
  EXPECT_TRUE(is_near(n_ary_operation(std::multiplies<double>{}, M1x{m11_2}, M1x{m11_5}), m11_10));
  EXPECT_TRUE(is_near(n_ary_operation(std::multiplies<double>{}, Mx1{m11_2}, M1x{m11_5}), m11_10));
  EXPECT_TRUE(is_near(n_ary_operation(std::multiplies<double>{}, Mxx{m11_2}, M1x{m11_5}), m11_10));
  EXPECT_TRUE(is_near(n_ary_operation(std::plus<double>{}, m11_2, Mx1{m11_5}), m11_7));
  EXPECT_TRUE(is_near(n_ary_operation(std::plus<double>{}, M1x{m11_2}, Mx1{m11_5}), m11_7));
  EXPECT_TRUE(is_near(n_ary_operation(std::plus<double>{}, Mx1{m11_2}, Mx1{m11_5}), m11_7));
  EXPECT_TRUE(is_near(n_ary_operation(std::plus<double>{}, Mxx{m11_2}, Mx1{m11_5}), m11_7));
  EXPECT_TRUE(is_near(n_ary_operation(std::plus<double>{}, m11_2, Mxx{m11_5}), m11_7));
  EXPECT_TRUE(is_near(n_ary_operation(std::plus<double>{}, M1x{m11_2}, Mxx{m11_5}), m11_7));
  EXPECT_TRUE(is_near(n_ary_operation(std::plus<double>{}, Mx1{m11_2}, Mxx{m11_5}), m11_7));
  EXPECT_TRUE(is_near(n_ary_operation(std::plus<double>{}, Mxx{m11_2}, Mxx{m11_5}), m11_7));

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  M2x m2x_3 {m23};
  Mx3 mx3_2 {m23};
  Mxx mxx_23 {m23};
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2) {return arg1 + arg2;}, m23, 2 * m23), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, mxx_23, 2 * m23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m2x_3, 2 * m23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, mx3_2, 2 * m23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m2x_3, 2 * m23), m23 * 5));

  auto m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 1, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c) {return arg1 + arg2 + r + c;}, m23, 2 * m23), m23 * 3 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, mxx_23, 2 * m23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m2x_3, 2 * m23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, mx3_2, 2 * m23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m2x_3, 2 * m23), m23 * 5 + m23p));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m23, 2 * mxx_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m2x_3, 2 * mxx_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, mx3_2, 2 * mxx_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, mxx_23, 2 * mxx_23), m23 * 5));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m23, 2 * mxx_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m2x_3, 2 * mxx_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, mx3_2, 2 * mxx_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, mxx_23, 2 * mxx_23), m23 * 5 + m23p));

  auto m13 = make_dense_writable_matrix_from<M13>(1, 2, 3);
  M1x m1x_3 {m13};
  Mx3 mx3_1 {m13};
  Mxx mxx_13 {m13};
  m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 1, 2, 3);
  m2x_3 = m23;
  mx3_2 = m23;
  mxx_23 = m23;

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2) {return arg1 + arg2;}, m13, 2 * m23), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, std::plus<double>{}, m13, 2 * m23), m23 * 3));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, std::minus<double>{}, m13, 2 * m23), -m23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m23, 2 * m13), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, mxx_23, 2 * m13), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2){return arg1 + 3 * arg2;}, 2 * m1x_3, m23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, 2 * mx3_2, 2 * m13), m23 * 8));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2){return arg1 * arg2;}, m1x_3, m13), m23.array().square()));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::multiplies<double>{}, m1x_3, m13), m23.array().square()));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::divides<double>{}, m1x_3, m23), M23::Constant(1)));

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c) {return arg1 + arg2 + r + c;}, m13, 2 * m23), m23 * 3 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m23, 2 * m13), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, mxx_23, 2 * m13), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return arg1 + 3 * arg2 + r + c;}, 2 * m1x_3, m23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, 2 * mx3_2, 2 * m13), m23 * 8 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return arg1 * arg2 + r + c;}, m1x_3, m13), m23.array().square().matrix() + m23p));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m13, 2 * mxx_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, m2x_3, 2 * mxx_13), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, mx3_1, 2 * mxx_23), m23 * 5));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2){return 3 * arg1 + arg2;}, mxx_23, 2 * mxx_13), m23 * 5));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m13, 2 * mxx_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, m2x_3, 2 * mxx_13), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, mx3_1, 2 * mxx_23), m23 * 5 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + r + c;}, mxx_23, 2 * mxx_13), m23 * 5 + m23p));

  auto m13_eq = make_dense_writable_matrix_from<M13>(1, 3, 3);
  auto m13_gt = make_dense_writable_matrix_from<M13>(2, 2, 4);
  auto m13_ge = make_dense_writable_matrix_from<M13>(2, 1, 3);
  auto b23 = make_dense_writable_matrix_from<eigen_matrix_t<bool, 2, 3>>(true, false, true, true, false, true);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::equal_to<double>{}, m1x_3, m13_eq), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::not_equal_to<double>{}, m1x_3, m13_gt), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::greater<double>{}, m13_gt, m1x_3), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::less<double>{}, m1x_3, m13_gt), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::greater_equal<double>{}, m13_ge, m1x_3), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::less_equal<double>{}, m1x_3, m13_ge), b23));

  auto b13 = make_dense_writable_matrix_from<eigen_matrix_t<bool, 1, 3>>(true, false, true);
  auto b13_and = make_dense_writable_matrix_from<eigen_matrix_t<bool, 1, 3>>(true, true, true);
  auto b13_or = make_dense_writable_matrix_from<eigen_matrix_t<bool, 1, 3>>(false, false, false);
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::logical_and<bool>{}, b13, b13_and), b23));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, std::logical_or<bool>{}, b13, b13_or), b23));
}


TEST(eigen3, ternary_operation)
{
  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 1, 2, 3);
  M2x m2x_3 {m23};
  Mx3 mx3_2 {m23};
  Mxx mxx_23 {m23};
  auto m13 = make_dense_writable_matrix_from<M13>(1, 2, 3);
  M1x m1x_3 {m13};
  Mx3 mx3_1 {m13};
  Mxx mxx_13 {m13};
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3) {return arg1 + arg2 + arg3;}, m23, 2 * m23, mxx_23), m23 * 4));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, mxx_23, 2 * m23, m13), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m2x_3, 2 * m23, mx3_1), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, mx3_2, 2 * m23, m1x_3), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m2x_3, 2 * m23, mxx_13), m23 * 6));

  auto m23p = make_dense_writable_matrix_from<M23>(0, 1, 2, 1, 2, 3);

  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c) {return arg1 + arg2 + arg3 + r + c;}, m23, 2 * m23, mxx_23), m23 * 4 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, mxx_23, 2 * m23, m13), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m2x_3, 2 * m23, mx3_1), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, mx3_2, 2 * m23, m1x_3), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation(std::tuple {Dimensions{2}, Dimensions{3}}, [](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m2x_3, 2 * m23, mxx_13), m23 * 6 + m23p));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m23, 2 * mxx_23, mxx_23), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, m2x_3, 2 * mxx_23, mx3_1), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, mx3_2, 2 * mxx_23, m1x_3), m23 * 6));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3){return 3 * arg1 + arg2 + arg3;}, mxx_23, 2 * mxx_23, m13), m23 * 6));

  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m23, 2 * mxx_23, mxx_23), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, m2x_3, 2 * mxx_23, mx3_1), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, mx3_2, 2 * mxx_23, m1x_3), m23 * 6 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation([](const auto& arg1, const auto& arg2, const auto& arg3, std::size_t r, std::size_t c){return 3 * arg1 + arg2 + arg3 + r + c;}, mxx_23, 2 * mxx_23, m13), m23 * 6 + m23p));
}


TEST(eigen3, randomize)
{
  using N = std::normal_distribution<double>;

  M22 m22, m22_true;
  M23 m23, m23_true;
  M32 m32, m32_true;
  auto m2x_2 = make_default_dense_writable_matrix_like<M22>(Dimensions<2>{}, 2);
  auto m2x_3 = make_default_dense_writable_matrix_like<M23>(Dimensions<2>{}, 3);
  auto mx2_2 = make_default_dense_writable_matrix_like<M22>(2, Dimensions<2>{});
  auto mx2_3 = make_default_dense_writable_matrix_like<M32>(3, Dimensions<2>{});
  auto m3x_2 = make_default_dense_writable_matrix_like<M32>(Dimensions<3>{}, 2);
  auto mx3_2 = make_default_dense_writable_matrix_like<M23>(2, Dimensions<3>{});
  auto mxx_22 = make_default_dense_writable_matrix_like<M22>(2, 2);
  auto mxx_32 = make_default_dense_writable_matrix_like<M32>(3, 2);
  auto mxx_23 = make_default_dense_writable_matrix_like<M23>(2, 3);

  // Test just using the parameters, rather than a constructed distribution.
  m22 = randomize<M22>(N {0.0, 0.7});
  m2x_2 = randomize<M2x>(std::tuple {Dimensions<2>{}, 2}, N {0.0, 1.0});
  m2x_3 = randomize<M2x>(std::tuple {Dimensions<2>{}, 3}, N {0.0, 0.7});
  mx2_2 = randomize<Mx2>(std::tuple {2, Dimensions<2>{}}, N {0.0, 1.0});
  mx2_3 = randomize<Mx2>(std::tuple {3, Dimensions<2>{}}, N {0.0, 0.7});
  mxx_22 = randomize<Mxx>(std::tuple{2, 2}, N {0.0, 1.0});

  // Single distribution for the entire matrix.
  m22 = M22::Zero();
  m2x_2 = M2x::Zero(2, 2);
  mx2_2 = Mx2::Zero(2, 2);
  mxx_22 = Mxx::Zero(2, 2);

  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<M22>(N {1.0, 0.3})) / (i + 1);
    m2x_2 = (m2x_2 * i + randomize<M2x>(std::tuple {Dimensions<2>{}, 2}, N {1.0, 0.3})) / (i + 1);
    mx2_2 = (mx2_2 * i + randomize<Mx2>(std::tuple {2, Dimensions<2>{}}, N {1.0, 0.3})) / (i + 1);
    mxx_22 = (mxx_22 * i + randomize<Mxx>(std::tuple{2, 2}, N {1.0, 0.3})) / (i + 1);
  }

  m22_true = M22::Constant(1);
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m2x_2, m22_true, 0.1));
  EXPECT_FALSE(is_near(m2x_2, m22_true, 1e-8));
  EXPECT_TRUE(is_near(mx2_2, m22_true, 0.1));
  EXPECT_FALSE(is_near(mx2_2, m22_true, 1e-8));
  EXPECT_TRUE(is_near(mxx_22, m22_true, 0.1));
  EXPECT_FALSE(is_near(mxx_22, m22_true, 1e-8));

  // One distribution for each element.
  m22 = M22::Zero();

  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<M22, 0, 1>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})) / (i + 1);
  }

  m22_true = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));

  // One distribution for each row.
  m32 = M32::Zero();
  m3x_2 = M3x::Zero(3, 2);

  m23 = M23::Zero();
  m2x_3 = M2x::Zero(2, 3);

  for (int i=0; i<10; i++)
  {
    m32 = (m32 * i + randomize<M32, 0>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
    m3x_2 = (m3x_2 * i + randomize<M3x, 0>(std::tuple {Dimensions<3>{}, 2}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);

    m23 = (m23 * i + randomize<M23, 0>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    m2x_3 = (m2x_3 * i + randomize<M2x, 0>(std::tuple {Dimensions<2>{}, 3}, N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
  }

  m32_true = make_dense_writable_matrix_from<M32>(1, 1, 2, 2, 3, 3);
  EXPECT_TRUE(is_near(m32, m32_true, 0.1));
  EXPECT_FALSE(is_near(m32, m32_true, 1e-8));
  EXPECT_TRUE(is_near(m3x_2, m32_true, 0.1));
  EXPECT_FALSE(is_near(m3x_2, m32_true, 1e-8));

  m23_true = make_dense_writable_matrix_from<M23>(1, 1, 1, 2, 2, 2);
  EXPECT_TRUE(is_near(m23, m23_true, 0.1));
  EXPECT_FALSE(is_near(m23, m23_true, 1e-8));
  EXPECT_TRUE(is_near(m2x_3, m23_true, 0.1));
  EXPECT_FALSE(is_near(m2x_3, m23_true, 1e-8));

  // One distribution for each column.
  m32 = M32::Zero();
  mx2_3 = Mx2::Zero(3, 2);

  m23 = M23::Zero();
  mx3_2 = Mx3::Zero(2, 3);

  for (int i=0; i<10; i++)
  {
    m32 = (m32 * i + randomize<M32, 1>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    mx2_3 = (mx2_3 * i + randomize<Mx2, 1>(std::tuple {3, Dimensions<2>{}}, N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);

    m23 = (m23 * i + randomize<M23, 1>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
    mx3_2 = (mx3_2 * i + randomize<Mx3, 1>(std::tuple {2, Dimensions<3>{}}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
  }

  m32_true = make_dense_writable_matrix_from<M32>(1, 2, 1, 2, 1, 2);
  EXPECT_TRUE(is_near(m32, m32_true, 0.1));
  EXPECT_FALSE(is_near(m32, m32_true, 1e-8));
  EXPECT_TRUE(is_near(mx2_3, m32_true, 0.1));
  EXPECT_FALSE(is_near(mx2_3, m32_true, 1e-8));

  m23_true = make_dense_writable_matrix_from<M23>(1, 2, 3, 1, 2, 3);
  EXPECT_TRUE(is_near(m23, m23_true, 0.1));
  EXPECT_FALSE(is_near(m23, m23_true, 1e-8));
  EXPECT_TRUE(is_near(mx3_2, m23_true, 0.1));
  EXPECT_FALSE(is_near(mx3_2, m23_true, 1e-8));
}


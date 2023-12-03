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
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, chipwise_nullary)
{
  auto N1 = std::integral_constant<std::size_t, 1>{};
  auto N3 = std::integral_constant<std::size_t, 3>{};
  auto N4 = std::integral_constant<std::size_t, 4>{};

  const auto m14 = make_dense_writable_matrix_from<M14>(1, 2, 3, 4);
  const auto m34r = make_dense_writable_matrix_from<M34>(
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4);

  const auto m31 = make_dense_writable_matrix_from<M31>(1, 2, 3);
  const auto m34c = make_dense_writable_matrix_from<M34>(
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3);

  const auto m11 = make_dense_writable_matrix_from<M11>(3);
  const auto m34rc = make_dense_writable_matrix_from<M34>(
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3);

  EXPECT_TRUE(is_near(chipwise_operation([&]{ return m34r; }), m34r));

  EXPECT_TRUE(is_near(chipwise_operation<0>([&]{ return m14; }, N3), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&]{ return m14; }, 3), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&]{ return m14; }, 1), m14));

  EXPECT_TRUE(is_near(chipwise_operation<1>([&]{ return m31; }, N4), m34c));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&]{ return m31; }, 4), m34c));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&]{ return m31; }, N1), m31));

  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([&]{ return m11; }, N3, N4), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([&]{ return m11; }, 3, N4), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([&]{ return m11; }, N1, 1), m11));

  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([&]{ return m11; }, 4, N3), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([&]{ return m11; }, 4, 3), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([&]{ return m11; }, N1, 1), m11));
}


TEST(eigen3, chipwise_nullary_w_indices)
{
  auto N1 = std::integral_constant<std::size_t, 1>{};
  auto N3 = std::integral_constant<std::size_t, 3>{};
  auto N4 = std::integral_constant<std::size_t, 4>{};

  const auto m14 = make_dense_writable_matrix_from<M14>(1, 2, 3, 4);
  const auto m34r = make_dense_writable_matrix_from<M34>(
    2, 4,  6,  8,
    3, 6,  9, 12,
    4, 8, 12, 16);

  const auto m31 = make_dense_writable_matrix_from<M31>(1, 2, 3);
  const auto m34c = make_dense_writable_matrix_from<M34>(
    2, 3,  4,  5,
    4, 6,  8, 10,
    6, 9, 12, 15);

  const auto m11 = make_dense_writable_matrix_from<M11>(3);
  const auto m34rc = make_dense_writable_matrix_from<M34>(
     6,  9, 12, 15,
    12, 18, 24, 30,
    18, 27, 36, 45);

  EXPECT_TRUE(is_near(chipwise_operation<0>([&](std::size_t i){ return m14 * (i + 2); }, N3), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&](std::size_t i){ return m14 * (i + 2); }, 3), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&](std::size_t i){ return m14 * (i + 2); }, 1), m14 * 2));

  EXPECT_TRUE(is_near(chipwise_operation<1>([&](std::size_t j){ return m31 * (j + 2); }, N4), m34c));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&](std::size_t j){ return m31 * (j + 2); }, 4), m34c));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&](std::size_t j){ return m31 * (j + 2); }, N1), m31 * 2));

  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([&](std::size_t i, std::size_t j){ return m11 * (i + 1) * (j + 2); }, N3, N4), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([&](std::size_t i, std::size_t j){ return m11 * (i + 1) * (j + 2); }, 3, N4), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([&](std::size_t i, std::size_t j){ return m11 * (i + 3) * (j + 2); }, N1, 1), m11 * 6));

  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([&](std::size_t j, std::size_t i){ return m11 * (i + 1) * (j + 2); }, 4, N3), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([&](std::size_t j, std::size_t i){ return m11 * (i + 1) * (j + 2); }, 4, 3), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([&](std::size_t j, std::size_t i){ return m11 * (i + 4) * (j + 5); }, N1, 1), m11 * 20));
}


TEST(eigen3, chipwise_unary)
{
  const auto m34 = make_dense_writable_matrix_from<M34>(
    1,  2,  3,  4,
    5,  6,  7,  8,
    9, 10, 11, 12);

  EXPECT_TRUE(is_near(chipwise_operation([](const auto& mat){ static_assert(dimension_size_of_index_is<decltype(mat), 0, 3>); return mat * 2; }, m34), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation([](const auto& mat){ static_assert(dimension_size_of_index_is<decltype(mat), 1, 4>); return mat * 2; }, M34{m34}), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation([](const auto& mat){ static_assert(dimension_size_of_index_is<decltype(mat), 0, 3> and dynamic_dimension<decltype(mat), 1>); return mat * 3; }, M3x{m34}), m34 * 3));
  EXPECT_TRUE(is_near(chipwise_operation([](const auto& mat){ static_assert(dynamic_dimension<decltype(mat), 0> and dimension_size_of_index_is<decltype(mat), 1, 4>); return mat * 4; }, Mx4{m34}), m34 * 4));
  EXPECT_TRUE(is_near(chipwise_operation([](const auto& mat){ static_assert(dynamic_dimension<decltype(mat), 0> and dynamic_dimension<decltype(mat), 1>); return mat * 5; }, Mxx{m34}), m34 * 5));

  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ static_assert(vector<decltype(row), 1>); return row * 2; }, m34), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ static_assert(vector<decltype(row), 1>); return row * 2; }, M34{m34}), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ static_assert(vector<decltype(row), 1>); return row * 3; }, M3x{m34}), m34 * 3));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ static_assert(vector<decltype(row), 1>); return row * 4; }, Mx4{m34}), m34 * 4));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row){ static_assert(vector<decltype(row), 1>); return row * 5; }, Mxx{m34}), m34 * 5));

  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ static_assert(vector<decltype(col), 0>); return col * 2; }, m34), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ static_assert(vector<decltype(col), 0>); return col * 2; }, M34{m34}), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ static_assert(vector<decltype(col), 0>); return col * 3; }, M3x{m34}), m34 * 3));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ static_assert(vector<decltype(col), 0>); return col * 4; }, Mx4{m34}), m34 * 4));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col){ static_assert(vector<decltype(col), 0>); return col * 5; }, Mxx{m34}), m34 * 5));

  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& elem){ static_assert(one_by_one_matrix<decltype(elem)>); return elem * 2; }, m34), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& elem){ static_assert(one_by_one_matrix<decltype(elem)>); return elem * 2; }, M34{m34}), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& elem){ static_assert(one_by_one_matrix<decltype(elem)>); return elem * 4; }, Mx4{m34}), m34 * 4));

  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([](const auto& elem){ static_assert(one_by_one_matrix<decltype(elem)>); return elem * 3; }, M3x{m34}), m34 * 3));
  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([](const auto& elem){ static_assert(one_by_one_matrix<decltype(elem)>); return elem * 5; }, Mxx{m34}), m34 * 5));
}


TEST(eigen3, chipwise_unary_w_indices)
{
  const auto m34 = make_dense_writable_matrix_from<M34>(
    1,  2,  3,  4,
    5,  6,  7,  8,
    9, 10, 11, 12);
  const auto m34r = make_dense_writable_matrix_from<M34>(
     2,  4,  6,  8,
    15, 18, 21, 24,
    36, 40, 44, 48);
  const auto m34c = make_dense_writable_matrix_from<M34>(
     2,  6, 12, 20,
    10, 18, 28, 40,
    18, 30, 44, 60);
  const auto m34rc = make_dense_writable_matrix_from<M34>(
     2,  6,  12,  20,
    20, 36,  56,  80,
    54, 90, 132, 180);

  EXPECT_TRUE(is_near(chipwise_operation<2>([](const auto& mat, std::size_t k){ return mat * (k + 2); }, m34), m34 * 2)); // k is always 0 b/c the third index is 1D
  EXPECT_TRUE(is_near(chipwise_operation<2>([](const auto& mat, std::size_t k){ return mat * (k + 2); }, M34{m34}), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<2>([](const auto& mat, std::size_t k){ return mat * (k + 2); }, M3x{m34}), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<2>([](const auto& mat, std::size_t k){ return mat * (k + 2); }, Mx4{m34}), m34 * 2));
  EXPECT_TRUE(is_near(chipwise_operation<2>([](const auto& mat, std::size_t k){ return mat * (k + 2); }, Mxx{m34}), m34 * 2));

  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row, std::size_t i){ return row * (i + 2); }, m34), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row, std::size_t i){ return row * (i + 2); }, M34{m34}), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row, std::size_t i){ return row * (i + 2); }, M3x{m34}), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row, std::size_t i){ return row * (i + 2); }, Mx4{m34}), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0>([](const auto& row, std::size_t i){ return row * (i + 2); }, Mxx{m34}), m34r));

  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col, std::size_t j){ return col * (j + 2); }, m34), m34c));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col, std::size_t j){ return col * (j + 2); }, M34{m34}), m34c));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col, std::size_t j){ return col * (j + 2); }, M3x{m34}), m34c));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col, std::size_t j){ return col * (j + 2); }, Mx4{m34}), m34c));
  EXPECT_TRUE(is_near(chipwise_operation<1>([](const auto& col, std::size_t j){ return col * (j + 2); }, Mxx{m34}), m34c));

  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& elem, std::size_t i, std::size_t j){ return elem * (i + 1) * (j + 2); }, m34), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& elem, std::size_t i, std::size_t j){ return elem * (i + 1) * (j + 2); }, M34{m34}), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& elem, std::size_t i, std::size_t j){ return elem * (i + 1) * (j + 2); }, Mx4{m34}), m34rc));

  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([](const auto& elem, std::size_t j, std::size_t i){ return elem * (i + 1) * (j + 2); }, M3x{m34}), m34rc));
  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([](const auto& elem, std::size_t j, std::size_t i){ return elem * (i + 1) * (j + 2); }, Mxx{m34}), m34rc));
}


TEST(eigen3, chipwise_binary)
{
  const auto m34a = make_dense_writable_matrix_from<M34>(
    1,  2,  3,  4,
    5,  6,  7,  8,
    9, 10, 11, 12);

  const auto m34b = make_dense_writable_matrix_from<M34>(
    1, 4, 7, 10,
    2, 5, 8, 11,
    3, 6, 9, 12);

  const auto mt3 = make_dense_writable_matrix_from<M33>(
    8, 7, 6,
    5, 4, 3,
    3, 1, 0);

  const auto mt4 = make_dense_writable_matrix_from<M44>(
    16, 15, 14, 13,
    12, 11, 10,  9,
     8,  7,  6,  5,
     4,  3,  1,  0);

  EXPECT_TRUE(is_near(chipwise_operation([&](const auto& a, const auto& b){ return mt3 * a + b * mt4; }, m34a, m34b), mt3 * m34a + m34b * mt4));
  EXPECT_TRUE(is_near(chipwise_operation([&](const auto& a, const auto& b){ return mt3 * a + b * mt4; }, M3x{m34a}, m34b), mt3 * m34a + m34b * mt4));
  EXPECT_TRUE(is_near(chipwise_operation([&](const auto& a, const auto& b){ return mt3 * a + b * mt4; }, m34a, Mx4{m34b}), mt3 * m34a + m34b * mt4));
  EXPECT_TRUE(is_near(chipwise_operation([&](const auto& a, const auto& b){ return mt3 * a + b * mt4; }, Mxx{m34a}, Mxx{m34b}), mt3 * m34a + m34b * mt4));

  EXPECT_TRUE(is_near(chipwise_operation<0>([&](const auto& a, const auto& b){ return a * mt4 + b * mt4; }, m34a, m34b), m34a * mt4 + m34b * mt4));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&](const auto& a, const auto& b){ return a * mt4 + b * mt4; }, Mx4{m34a}, m34b), m34a * mt4 + m34b * mt4));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&](const auto& a, const auto& b){ return a * mt4 + b * mt4; }, m34a, M3x{m34b}), m34a * mt4 + m34b * mt4));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&](const auto& a, const auto& b){ return a * mt4 + b * mt4; }, Mxx{m34a}, Mxx{m34b}), m34a * mt4 + m34b * mt4));

  EXPECT_TRUE(is_near(chipwise_operation<1>([&](const auto& a, const auto& b){ return mt3 * a + mt3 * b; }, m34a, m34b), mt3 * m34a + mt3 * m34b));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&](const auto& a, const auto& b){ return mt3 * a + mt3 * b; }, M3x{m34a}, m34b), mt3 * m34a + mt3 * m34b));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&](const auto& a, const auto& b){ return mt3 * a + mt3 * b; }, m34a, Mx4{m34b}), mt3 * m34a + mt3 * m34b));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&](const auto& a, const auto& b){ return mt3 * a + mt3 * b; }, Mxx{m34a}, Mxx{m34b}), mt3 * m34a + mt3 * m34b));

  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& a, const auto& b){ return 3 * a + 2 * b; }, m34a, m34b), 3 * m34a + 2 * m34b));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& a, const auto& b){ return 3 * a + 2 * b; }, Mx4{m34a}, m34b), 3 * m34a + 2 * m34b));

  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([](const auto& a, const auto& b){ return 3 * a + 2 * b; }, m34a, M3x{m34b}), 3 * m34a + 2 * m34b));
  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([](const auto& a, const auto& b){ return 3 * a + 2 * b; }, Mxx{m34a}, Mxx{m34b}), 3 * m34a + 2 * m34b));
}


TEST(eigen3, chipwise_binary_w_indices)
{
  const auto m34a = make_dense_writable_matrix_from<M34>(
    1,  2,  3,  4,
    5,  6,  7,  8,
    9, 10, 11, 12);

  const auto m34b = make_dense_writable_matrix_from<M34>(
    1, 4, 7, 10,
    2, 5, 8, 11,
    3, 6, 9, 12);

  const auto mt3 = make_dense_writable_matrix_from<M33>(
    8, 7, 6,
    5, 4, 3,
    3, 1, 0);

  const auto mt4 = make_dense_writable_matrix_from<M44>(
    16, 15, 14, 13,
    12, 11, 10,  9,
     8,  7,  6,  5,
     4,  3,  1,  0);

  const M33 d1 {make_dense_writable_matrix_from<M31>(1, 2, 3).asDiagonal()};
  const M33 d2 {make_dense_writable_matrix_from<M31>(2, 3, 4).asDiagonal()};
  const M44 e1 {make_dense_writable_matrix_from<M41>(1, 2, 3, 4).asDiagonal()};
  const M44 e2 {make_dense_writable_matrix_from<M41>(2, 3, 4, 5).asDiagonal()};
  const M44 e3 {make_dense_writable_matrix_from<M41>(3, 4, 5, 6).asDiagonal()};

  const auto m34p = d1 * m34a * mt4 + d2 * m34b * mt4;
  const auto m34q = mt3 * m34a * e2 + mt3 * m34b * e1;
  const auto m34r = d2 * m34a + m34b * e3;

  EXPECT_TRUE(is_near(chipwise_operation<0>([&](const auto& a, const auto& b, std::size_t i){ return a * mt4 * (i + 1) + b * mt4 * (i + 2); }, m34a, m34b), m34p));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&](const auto& a, const auto& b, std::size_t i){ return a * mt4 * (i + 1) + b * mt4 * (i + 2); }, Mx4{m34a}, m34b), m34p));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&](const auto& a, const auto& b, std::size_t i){ return a * mt4 * (i + 1) + b * mt4 * (i + 2); }, m34a, M3x{m34b}), m34p));
  EXPECT_TRUE(is_near(chipwise_operation<0>([&](const auto& a, const auto& b, std::size_t i){ return a * mt4 * (i + 1) + b * mt4 * (i + 2); }, Mxx{m34a}, Mxx{m34b}), m34p));

  EXPECT_TRUE(is_near(chipwise_operation<1>([&](const auto& a, const auto& b, std::size_t j){ return mt3 * a * (j + 2) + mt3 * b * (j + 1); }, m34a, m34b), m34q));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&](const auto& a, const auto& b, std::size_t j){ return mt3 * a * (j + 2) + mt3 * b * (j + 1); }, M3x{m34a}, m34b), m34q));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&](const auto& a, const auto& b, std::size_t j){ return mt3 * a * (j + 2) + mt3 * b * (j + 1); }, m34a, Mx4{m34b}), m34q));
  EXPECT_TRUE(is_near(chipwise_operation<1>([&](const auto& a, const auto& b, std::size_t j){ return mt3 * a * (j + 2) + mt3 * b * (j + 1); }, Mxx{m34a}, Mxx{m34b}), m34q));

  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& a, const auto& b, std::size_t i, std::size_t j){ return (i + 2) * a + (j + 3) * b; }, m34a, m34b), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<0, 1>([](const auto& a, const auto& b, std::size_t i, std::size_t j){ return (i + 2) * a + (j + 3) * b; }, Mx4{m34a}, m34b), m34r));

  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([](const auto& a, const auto& b, std::size_t j, std::size_t i){ return (i + 2) * a + (j + 3) * b; }, m34a, M3x{m34b}), m34r));
  EXPECT_TRUE(is_near(chipwise_operation<1, 0>([](const auto& a, const auto& b, std::size_t j, std::size_t i){ return (i + 2) * a + (j + 3) * b; }, Mxx{m34a}, Mxx{m34b}), m34r));
}


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
#include <complex>

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


TEST(eigen3, apply_columnwise)
{
  const auto m21_12 = make_dense_writable_matrix_from<M21>(1, 2);

  auto m21 = M21 {m21_12};
  auto m20_1 = M20 {m21_12};
  auto m01_2 = M01 {m21_12};
  auto m00_21 = M00 {m21_12};

  const auto m21_23 = make_dense_writable_matrix_from<M21>(2, 3);

  apply_columnwise([](auto& col){ col += col.Constant(1); }, m21); EXPECT_TRUE(is_near(m21, m21_23));
  apply_columnwise([](auto& col){ col += col.Constant(1); }, m20_1); EXPECT_TRUE(is_near(m20_1, m21_23));
  apply_columnwise([](auto& col){ col += col.Constant(2, 1, 1); }, m01_2); EXPECT_TRUE(is_near(m01_2, m21_23));
  apply_columnwise([](auto& col){ col += col.Constant(2, 1, 1); }, m00_21); EXPECT_TRUE(is_near(m00_21, m21_23));

  const auto m33_123 = make_eigen_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  auto m33 = M33 {m33_123};
  auto m30 = M30 {m33_123};
  auto m03 = M03 {m33_123};
  auto m00 = M00 {m33_123};

  const auto m33_234 = make_dense_writable_matrix_from<M33>(
    2, 1, 1,
    1, 3, 1,
    1, 1, 4);

  apply_columnwise([](auto& col){ col += col.Constant(1); }, m33); EXPECT_TRUE(is_near(m33, m33_234));
  apply_columnwise([](auto& col){ col += col.Constant(1); }, m30); EXPECT_TRUE(is_near(m30, m33_234));
  apply_columnwise([](auto& col){ col += col.Constant(3, 1, 1); }, m03); EXPECT_TRUE(is_near(m03, m33_234));
  apply_columnwise([](auto& col){ col += col.Constant(3, 1, 1); }, m00); EXPECT_TRUE(is_near(m00, m33_234));

  m33 = m33_123; EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, m33), m33_234));
  m30 = m33_123; EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, m30), m33_234));
  m03 = m33_123; EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(3, 1, 1)); }, m03), m33_234));
  m00 = m33_123; EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(3, 1, 1)); }, m00), m33_234));

  EXPECT_TRUE(is_near(m33, m33_123));
  EXPECT_TRUE(is_near(m30, m33_123));
  EXPECT_TRUE(is_near(m03, m33_123));
  EXPECT_TRUE(is_near(m00, m33_123));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, M33 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, M30 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(3, 1, 1)); }, M03 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(3, 1, 1)); }, M00 {m33_123}), m33_234));

  auto m33_135_col = make_dense_writable_matrix_from<M33>(
    1, 1, 2,
    0, 3, 2,
    0, 1, 5);

  m33 = m33_123; apply_columnwise([](auto& col, std::size_t i){ col += col.Constant(i); }, m33); EXPECT_TRUE(is_near(m33, m33_135_col));
  m30 = m33_123; apply_columnwise([](auto& col, std::size_t i){ col += col.Constant(i); }, m30); EXPECT_TRUE(is_near(m30, m33_135_col));
  m03 = m33_123; apply_columnwise([](auto& col, std::size_t i){ col += col.Constant(3, 1, i); }, m03); EXPECT_TRUE(is_near(m03, m33_135_col));
  m00 = m33_123; apply_columnwise([](auto& col, std::size_t i){ col += col.Constant(3, 1, i); }, m00); EXPECT_TRUE(is_near(m00, m33_135_col));

  m33 = m33_123; EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, m33), m33_135_col));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, m33_123), m33_135_col));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, M33 {m33_123}), m33_135_col));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, M30 {m33_123}), m33_135_col));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(3, 1, i)); }, M03 {m33_123}), m33_135_col));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(3, 1, i)); }, M00 {m33_123}), m33_135_col));

  auto m33_123_h = make_dense_writable_matrix_from<M33>(
    1, 1, 1,
    2, 2, 2,
    3, 3, 3);

  auto m31_123 = make_dense_writable_matrix_from<M31>(1, 2, 3);

  EXPECT_TRUE(is_near(apply_columnwise<3>([&] { return m31_123; }), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise<3>([&] { return M30 {m31_123}; }), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise<3>([&] { return M01 {m31_123}; }), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise<3>([&] { return M00 {m31_123}; }), m33_123_h));

  EXPECT_TRUE(is_near(apply_columnwise([&] { return m31_123; }, 3), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise([&] { return M30 {m31_123}; }, 3), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise([&] { return M01 {m31_123}; }, 3), m33_123_h));
  EXPECT_TRUE(is_near(apply_columnwise([&] { return M00 {m31_123}; }, 3), m33_123_h));

  auto m33_123_d = make_dense_writable_matrix_from<M33>(
    1, 2, 3,
    2, 3, 4,
    3, 4, 5);

  EXPECT_TRUE(is_near(apply_columnwise<3>([](std::size_t i) { return make_dense_writable_matrix_from<M31>(1 + i, 2 + i, 3 + i); }), m33_123_d));

  EXPECT_TRUE(is_near(apply_columnwise([](std::size_t i) { return make_dense_writable_matrix_from<M31>(1 + i, 2 + i, 3 + i); }, 3), m33_123_d));
}


TEST(eigen3, apply_rowwise)
{
  const auto m12_12 = make_dense_writable_matrix_from<M12>(1, 2);

  auto m12_vol = M12 {m12_12};
  auto m10_2_vol = M10 {m12_12};
  auto m02_1_vol = M02 {m12_12};
  auto m00_12_vol = M00 {m12_12};

  const auto m12_23 = make_dense_writable_matrix_from<M12>(2, 3);

  apply_rowwise([](auto& row){ row += row.Constant(1); }, m12_vol); EXPECT_TRUE(is_near(m12_vol, m12_23));
  apply_rowwise([](auto& row){ row += row.Constant(1, 2, 1); }, m10_2_vol); EXPECT_TRUE(is_near(m10_2_vol, m12_23));
  apply_rowwise([](auto& row){ row += row.Constant(1); }, m02_1_vol); EXPECT_TRUE(is_near(m02_1_vol, m12_23));
  apply_rowwise([](auto& row){ row += row.Constant(1, 2, 1); }, m00_12_vol); EXPECT_TRUE(is_near(m00_12_vol, m12_23));

  const auto m33_123 = make_eigen_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  auto m33 = M33 {m33_123};
  auto m30 = M30 {m33_123};
  auto m03 = M03 {m33_123};
  auto m00 = M00 {m33_123};

  const auto m33_234 = make_dense_writable_matrix_from<M33>(
    2, 1, 1,
    1, 3, 1,
    1, 1, 4);

  apply_rowwise([](auto& row){ row += row.Constant(1); }, m33); EXPECT_TRUE(is_near(m33, m33_234));
  apply_rowwise([](auto& row){ row += row.Constant(1, 3, 1); }, m30); EXPECT_TRUE(is_near(m30, m33_234));
  apply_rowwise([](auto& row){ row += row.Constant(1); }, m03); EXPECT_TRUE(is_near(m03, m33_234));
  apply_rowwise([](auto& row){ row += row.Constant(1, 3, 1); }, m00); EXPECT_TRUE(is_near(m00, m33_234));

  m33 = m33_123; EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, m33), m33_234));
  m30 = m33_123; EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1, 3, 1)); }, m30), m33_234));
  m03 = m33_123; EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, m03), m33_234));
  m00 = m33_123; EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1, 3, 1)); }, m00), m33_234));

  EXPECT_TRUE(is_near(m33, m33_123));
  EXPECT_TRUE(is_near(m30, m33_123));
  EXPECT_TRUE(is_near(m03, m33_123));
  EXPECT_TRUE(is_near(m00, m33_123));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, M33 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1, 3, 1)); }, M30 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, M03 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1, 3, 1)); }, M00 {m33_123}), m33_234));

  auto m33_135_row = make_eigen_matrix<double, 3, 3>(
    1, 0, 0,
    1, 3, 1,
    2, 2, 5);

  m33 = m33_123; apply_rowwise([](auto& row, std::size_t i){ row += row.Constant(i); }, m33); EXPECT_TRUE(is_near(m33, m33_135_row));
  m30 = m33_123; apply_rowwise([](auto& row, std::size_t i){ row += row.Constant(1, 3, i); }, m30); EXPECT_TRUE(is_near(m30, m33_135_row));
  m03 = m33_123; apply_rowwise([](auto& row, std::size_t i){ row += row.Constant(i); }, m03); EXPECT_TRUE(is_near(m03, m33_135_row));
  m00 = m33_123; apply_rowwise([](auto& row, std::size_t i){ row += row.Constant(1, 3, i); }, m00); EXPECT_TRUE(is_near(m00, m33_135_row));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }, m33_123), m33_135_row));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }, M33 {m33_123}), m33_135_row));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(1, 3, i)); }, M30 {m33_123}), m33_135_row));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }, M03 {m33_123}), m33_135_row));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(1, 3, i)); }, M00 {m33_123}), m33_135_row));

  auto m33_123_v = make_eigen_matrix<double, 3, 3>(
    1, 2, 3,
    1, 2, 3,
    1, 2, 3);

  auto m13_123 = make_dense_writable_matrix_from<M13>(1, 2, 3);

  EXPECT_TRUE(is_near(apply_rowwise<3>([&] { return m13_123; }), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise<3>([&] { return M10 {m13_123}; }), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise<3>([&] { return M03 {m13_123}; }), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise<3>([&] { return M00 {m13_123}; }), m33_123_v));

  EXPECT_TRUE(is_near(apply_rowwise([&] { return m13_123; }, 3), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise([&] { return M10 {m13_123}; }, 3), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise([&] { return M03 {m13_123}; }, 3), m33_123_v));
  EXPECT_TRUE(is_near(apply_rowwise([&] { return M00 {m13_123}; }, 3), m33_123_v));

  auto m33_123_d = make_dense_writable_matrix_from<M33>(
    1, 2, 3,
    2, 3, 4,
    3, 4, 5);

  EXPECT_TRUE(is_near(apply_rowwise<3>([](std::size_t i) { return make_dense_writable_matrix_from<M13>(1 + i, 2 + i, 3 + i); }), m33_123_d));

  EXPECT_TRUE(is_near(apply_rowwise([](std::size_t i) { return make_dense_writable_matrix_from<M13>(1 + i, 2 + i, 3 + i); }, 3), m33_123_d));
}


TEST(eigen3, apply_coefficientwise)
{
  const auto m33_123 = make_eigen_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3);

  const auto m33_234 = make_dense_writable_matrix_from<M33>(
    2, 1, 1,
    1, 3, 1,
    1, 1, 4);

  auto m33 = M33 {m33_123}; apply_coefficientwise([](auto& x){ x += 1; }, m33); EXPECT_TRUE(is_near(m33, m33_234));
  auto m30 = M30 {m33_123}; apply_coefficientwise([](auto& x){ x += 1; }, m30); EXPECT_TRUE(is_near(m30, m33_234));
  auto m03 = M03 {m33_123}; apply_coefficientwise([](auto& x){ x += 1; }, m03); EXPECT_TRUE(is_near(m03, m33_234));
  auto m00 = M00 {m33_123}; apply_coefficientwise([](auto& x){ x += 1; }, m00); EXPECT_TRUE(is_near(m00, m33_234));

  EXPECT_TRUE(is_near(apply_coefficientwise([](double& x){ return x + 1; }, M33 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const double& x){ return x + 1; }, M30 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_coefficientwise([](double x){ return x + 1; }, M03 {m33_123}), m33_234));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const double x){ return x + 1; }, M00 {m33_123}), m33_234));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const double& x){ return x + 1; }, m33_123), m33_234));
  EXPECT_TRUE(is_near(m33_123 + M33::Constant(1), m33_234));

  auto m33_147 = make_eigen_matrix<double, 3, 3>(
    1, 1, 2,
    1, 4, 3,
    2, 3, 7);

  m33 = m33_123; apply_coefficientwise([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m33); EXPECT_TRUE(is_near(m33, m33_147));
  m30 = m33_123; apply_coefficientwise([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m30); EXPECT_TRUE(is_near(m30, m33_147));
  m03 = m33_123; apply_coefficientwise([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m03); EXPECT_TRUE(is_near(m03, m33_147));
  m00 = m33_123; apply_coefficientwise([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, m00); EXPECT_TRUE(is_near(m00, m33_147));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, M33 {m33_123}), m33_147));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, M30 {m33_123}), m33_147));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, M03 {m33_123}), m33_147));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, M00 {m33_123}), m33_147));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, m33_123), m33_147));



  auto m33_135 = make_eigen_matrix<double, 3, 3>(
    1, 2, 3,
    2, 3, 4,
    3, 4, 5);

  EXPECT_TRUE(is_near(apply_coefficientwise<3,3>([](std::size_t i, std::size_t j){ return 1. + i + j; }), m33_135));
  EXPECT_TRUE(is_near(apply_coefficientwise<3,dynamic_size>([](std::size_t i, std::size_t j){ return 1. + i + j; }, 3), m33_135));
  EXPECT_TRUE(is_near(apply_coefficientwise<dynamic_size,3>([](std::size_t i, std::size_t j){ return 1. + i + j; }, 3), m33_135));
  EXPECT_TRUE(is_near(apply_coefficientwise<dynamic_size, dynamic_size>([](std::size_t i, std::size_t j){ return 1. + i + j; }, 3, 3), m33_135));
  EXPECT_TRUE(is_near(apply_coefficientwise([](std::size_t i, std::size_t j){ return 1. + i + j; }, 3, 3), m33_135));

  EXPECT_TRUE(is_near(apply_coefficientwise<3,3>([]{ return 5.; }), M33::Constant(5)));
  EXPECT_TRUE(is_near(apply_coefficientwise<3,dynamic_size>([]{ return 5.; }, 3), M33::Constant(5)));
  EXPECT_TRUE(is_near(apply_coefficientwise<dynamic_size,3>([]{ return 5.; }, 3), M33::Constant(5)));
  EXPECT_TRUE(is_near(apply_coefficientwise<dynamic_size,dynamic_size>([]{ return 5.; }, 3, 3), M33::Constant(5)));
  EXPECT_TRUE(is_near(apply_coefficientwise([]{ return 5.; }, 3, 3), M33::Constant(5)));
}

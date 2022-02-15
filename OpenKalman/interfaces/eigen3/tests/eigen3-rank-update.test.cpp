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

  using M00 = eigen_matrix_t<double, dynamic_extent, dynamic_extent>;
  using M10 = eigen_matrix_t<double, 1, dynamic_extent>;
  using M01 = eigen_matrix_t<double, dynamic_extent, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_extent>;
  using M02 = eigen_matrix_t<double, dynamic_extent, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_extent>;
  using M03 = eigen_matrix_t<double, dynamic_extent, 3>;
  using M04 = eigen_matrix_t<double, dynamic_extent, 4>;
  using M50 = eigen_matrix_t<double, 5, dynamic_extent>;
  using M05 = eigen_matrix_t<double, dynamic_extent, 5>;

  using cdouble = std::complex<double>;

  using CM21 = eigen_matrix_t<cdouble, 2, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;
}


TEST(eigen3, rank_update_1x1)
{
  const auto m11_2 = M11(2);
  const auto m10_1_2 = M10 {m11_2};
  const auto m01_1_2 = M01 {m11_2};
  const auto m00_11_2 = M00 {m11_2};

  const auto m11_5 = M11(5);

  auto m11 = M11(3);
  auto m10_1 = M10 {m11};
  auto m01_1 = M01 {m11};
  auto m00_11 = M00 {m11};

  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(M10 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M01 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::diagonal>(M00 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(M10 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M01 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::diagonal>(M00 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(M10 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M01 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::diagonal>(M00 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(M10 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M01 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::diagonal>(M00 {m11}, M00 {m11_2}, 4), m11_5));

  rank_update_triangular(m11, m11_2, 4); EXPECT_TRUE(is_near(m11, m11_5));
  rank_update_triangular(m10_1, m11_2, 4); EXPECT_TRUE(is_near(m10_1, m11_5));
  rank_update_triangular(m01_1, m11_2, 4); EXPECT_TRUE(is_near(m01_1, m11_5));
  rank_update_triangular(m00_11, m11_2, 4); EXPECT_TRUE(is_near(m00_11, m11_5));

  m11 = M11(9);
  m10_1 = m11;
  m01_1 = m11;
  m00_11 = m11;

  const auto m11_25 = M11(25);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(M10 {m11}, m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M01 {m11}, m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::diagonal>(M00 {m11}, m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, M10 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(M10 {m11}, M10 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M01 {m11}, M10 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::diagonal>(M00 {m11}, M10 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, M01 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(M10 {m11}, M01 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M01 {m11}, M01 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::diagonal>(M00 {m11}, M01 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, M00 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(M10 {m11}, M00 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M01 {m11}, M00 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::diagonal>(M00 {m11}, M00 {m11_2}, 4), m11_25));

  rank_update_self_adjoint(m11, m11_2, 4); EXPECT_TRUE(is_near(m11, m11_25));
  rank_update_self_adjoint(m10_1, m11_2, 4); EXPECT_TRUE(is_near(m10_1, m11_25));
  rank_update_self_adjoint(m01_1, m11_2, 4); EXPECT_TRUE(is_near(m01_1, m11_25));
  rank_update_self_adjoint(m00_11, m11_2, 4); EXPECT_TRUE(is_near(m00_11, m11_25));

  m11 = M11(3);
  m10_1 = m11;
  m01_1 = m11;
  m00_11 = m11;

  EXPECT_TRUE(is_near(rank_update(M11 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M10 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M01 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M00 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M11 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M10 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M01 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M00 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M11 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M10 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M01 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M00 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M11 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M10 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M01 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update(M00 {m11}, M00 {m11_2}, 4), m11_5));
  static_assert(one_by_one_matrix<decltype(rank_update(M00 {m11}, M00 {m11_2}, 4))>);

  rank_update(m11, m11_2, 4); EXPECT_TRUE(is_near(m11, m11_5));
  rank_update(m10_1, m11_2, 4); EXPECT_TRUE(is_near(m10_1, m11_5));
  rank_update(m01_1, m11_2, 4); EXPECT_TRUE(is_near(m01_1, m11_5));
  rank_update(m00_11, m11_2, 4); EXPECT_TRUE(is_near(m00_11, m11_5));
}


TEST(eigen3, rank_update_diagonal)
{
  auto c11_2 {M11::Identity() + M11::Identity()};

  auto d21_2 = Eigen::Replicate<decltype(c11_2), 2, 1> {c11_2, 2, 1}.asDiagonal();
  auto d20_1_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();
  auto d01_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 1> {c11_2, 2, 1}.asDiagonal();
  auto d00_21_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 1}.asDiagonal();

  auto c11_3 {M11::Identity() + M11::Identity() + M11::Identity()};

  auto d21_3 = Eigen::Replicate<decltype(c11_3), 2, 1> {c11_3, 2, 1}.asDiagonal();
  auto d20_1_3 = Eigen::Replicate<decltype(c11_3), 2, Eigen::Dynamic> {c11_3, 2, 1}.asDiagonal();
  auto d01_2_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, 1> {c11_3, 2, 1}.asDiagonal();
  auto d00_21_3 = Eigen::Replicate<decltype(c11_3), Eigen::Dynamic, Eigen::Dynamic> {c11_3, 2, 1}.asDiagonal();

  auto m22_5005 = make_native_matrix<M22>(5, 0, 0, 5);

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(d20_1_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d01_2_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::diagonal>(d00_21_3, d21_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(d21_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d20_1_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::diagonal>(d01_2_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d00_21_3, d20_1_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular(d21_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(d20_1_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d01_2_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::diagonal>(d00_21_3, d01_2_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(d21_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d20_1_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::diagonal>(d01_2_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update_triangular(d00_21_3, d00_21_2, 4), m22_5005));

  auto c11_9 {c11_3 + c11_3 + c11_3};

  auto d21_9 = Eigen::Replicate<decltype(c11_9), 2, 1> {c11_9, 2, 1}.asDiagonal();
  auto d20_1_9 = Eigen::Replicate<decltype(c11_9), 2, Eigen::Dynamic> {c11_9, 2, 1}.asDiagonal();
  auto d01_2_9 = Eigen::Replicate<decltype(c11_9), Eigen::Dynamic, 1> {c11_9, 2, 1}.asDiagonal();
  auto d00_21_9 = Eigen::Replicate<decltype(c11_9), Eigen::Dynamic, Eigen::Dynamic> {c11_9, 2, 1}.asDiagonal();

  auto m22_25 = make_native_matrix<M22>(25, 0, 0, 25);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(d21_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(d20_1_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d01_2_9, d21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::diagonal>(d00_21_9, d21_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(d21_9, d20_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d20_1_9, d20_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::diagonal>(d01_2_9, d20_1_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d00_21_9, d20_1_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_self_adjoint(d21_9, d01_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(d20_1_9, d01_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d01_2_9, d01_2_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::diagonal>(d00_21_9, d01_2_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(d21_9, d00_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d20_1_9, d00_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::diagonal>(d01_2_9, d00_21_2, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d00_21_9, d00_21_2, 4), m22_25));

  EXPECT_TRUE(is_near(rank_update(d21_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_1_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d01_2_3, d21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_21_3, d21_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d21_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_1_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d01_2_3, d20_1_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_21_3, d20_1_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d21_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_1_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d01_2_3, d01_2_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_21_3, d01_2_2, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d21_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_1_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d01_2_3, d00_21_2, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_21_3, d00_21_2, 4), m22_5005));

  auto m22_2002 = make_native_matrix<M22>(2, 0, 0, 2);
  auto m20_2002 = M20 {m22_2002};
  auto m02_2002 = M02 {m22_2002};
  auto m00_2002 = M00 {m22_2002};

  EXPECT_TRUE(is_near(rank_update(d21_3, m22_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_1_3, m22_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d01_2_3, m22_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_21_3, m22_2002, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d21_3, m20_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_1_3, m20_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d01_2_3, m20_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_21_3, m20_2002, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d21_3, m02_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_1_3, m02_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d01_2_3, m02_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_21_3, m02_2002, 4), m22_5005));

  EXPECT_TRUE(is_near(rank_update(d21_3, m00_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d20_1_3, m00_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d01_2_3, m00_2002, 4), m22_5005));
  EXPECT_TRUE(is_near(rank_update(d00_21_3, m00_2002, 4), m22_5005));

  auto m21_33 = make_native_matrix<M21>(3, 3);
  auto m20_33 = make_native_matrix<M20>(3, 3);
  auto m01_33 = make_native_matrix<M01>(3, 3);
  auto m00_33 = make_native_matrix<M00>(3, 3);

  auto m22_2012 = make_native_matrix<M22>(2, 0, 1, 2);
  auto m22_501626 = make_native_matrix<M22>(5., 0, 1.6, std::sqrt(26.44));

  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalMatrix<double, 2>(m21_33), m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalMatrix<double, Eigen::Dynamic>(m21_33), m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m21_33}, m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m20_33}, m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m01_33}, m22_2012, 4), m22_501626));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m00_33}, m22_2012, 4), m22_501626));

  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_triangular(Eigen::DiagonalMatrix<double, 2>(m21_33), m22_2012, 4))>);
  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_triangular(Eigen::DiagonalMatrix<double, Eigen::Dynamic>(m21_33), m22_2012, 4))>);
  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_triangular(Eigen::DiagonalWrapper {m00_33}, m22_2012, 4))>);

  auto m21_37 = make_native_matrix<M21>(3, 7);
  auto m20_37 = make_native_matrix<M20>(m21_37);
  auto m01_37 = make_native_matrix<M01>(m21_37);
  auto m00_37 = make_native_matrix<M00>(m21_37);

  auto d21_16 = Eigen::DiagonalMatrix<double, 2> {make_native_matrix<M21>(1, 6)};

  const auto m22_525 = make_native_matrix<M22>(5, 0, 0, 25);

  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalMatrix<double, 2> {m21_37}, d21_16, 16), m22_525));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m00_37}, d21_16, 16), m22_525));

  auto d21_37 = Eigen::DiagonalMatrix<double, 2> {m21_37};
  auto d00_37 = Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21_37};

  EXPECT_TRUE(is_near(rank_update(d21_37, d21_16, 16), m22_525));
  EXPECT_TRUE(is_near(rank_update(d00_37, d21_16, 16), m22_525));

  EXPECT_TRUE(is_near(d21_37, m22_525));
  EXPECT_TRUE(is_near(d00_37, m22_525));

  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m21_37}, d21_16, 16), m22_525));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m20_37}, d21_16, 16), m22_525));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m01_37}, d21_16, 16), m22_525));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalWrapper {m00_37}, d21_16, 16), m22_525));

  auto dw21_37 = Eigen::DiagonalWrapper {m21_37};
  auto dw20_37 = Eigen::DiagonalWrapper {m20_37};
  auto dw01_37 = Eigen::DiagonalWrapper {m01_37};
  auto dw00_37 = Eigen::DiagonalWrapper {m00_37};

  EXPECT_TRUE(is_near(rank_update(dw21_37, d21_16, 16), m22_525));
  EXPECT_TRUE(is_near(rank_update(dw20_37, d21_16, 16), m22_525));
  EXPECT_TRUE(is_near(rank_update(dw01_37, d21_16, 16), m22_525));
  EXPECT_TRUE(is_near(rank_update(dw00_37, d21_16, 16), m22_525));

  EXPECT_TRUE(is_near(m21_37, make_native_matrix<M21>(3, 7)));
  EXPECT_TRUE(is_near(m20_37, m21_37));
  EXPECT_TRUE(is_near(m01_37, m21_37));
  EXPECT_TRUE(is_near(m00_37, m21_37));

  auto cm21_3 = make_native_matrix<CM21>(cdouble {3, 3}, cdouble {3, -3});
  auto cm22_2002 = make_native_matrix<CM22>(cdouble {4, 4}, 0, 0, cdouble {4, -4});
  double s73 = std::sqrt(73);
  auto cm22_8008 = make_native_matrix<CM22>(cdouble {s73, s73}, 0, 0, cdouble {s73, -s73});

  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalMatrix<cdouble, 2> {cm21_3}, cm22_2002, 4), cm22_8008));
  EXPECT_TRUE(is_near(rank_update(Eigen::DiagonalMatrix<cdouble, Eigen::Dynamic> {cm21_3}, cm22_2002, 4), cm22_8008));
}


TEST(eigen3, rank_update_self_adjoint)
{
  auto m22_2022 = make_native_matrix<M22>(2, 0, 2, 2);
  auto m22_25 = make_native_matrix<M22>(25, 25, 25, 50);

  auto m22 = make_native_matrix<M22>(9, 9, 9, 18);
  auto m20 = make_native_matrix<M20>(m22);
  auto m02 = make_native_matrix<M02>(m22);
  auto m00 = make_native_matrix<M00>(m22);

  rank_update_self_adjoint(m22, m22_2022, 4); EXPECT_TRUE(is_near(m22.selfadjointView<Eigen::Lower>(), m22_25));
  rank_update_self_adjoint(m20, m22_2022, 4); EXPECT_TRUE(is_near(m20.selfadjointView<Eigen::Lower>(), m22_25));
  rank_update_self_adjoint(m02, m22_2022, 4); EXPECT_TRUE(is_near(m02.selfadjointView<Eigen::Lower>(), m22_25));
  rank_update_self_adjoint(m00, m22_2022, 4); EXPECT_TRUE(is_near(m00.selfadjointView<Eigen::Lower>(), m22_25));

  m22 = make_native_matrix<M22>(9, 9, 9, 18);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(M22 {m22}, m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M20 {m22}, m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M02 {m22}, m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M00 {m22}, m22_2022, 4), m22_25));

  static_assert(lower_self_adjoint_matrix<decltype(rank_update_self_adjoint(M22 {m22}, m22_2022, 4))>);
  static_assert(lower_self_adjoint_matrix<decltype(rank_update_self_adjoint(M20 {m22}, m22_2022, 4))>);
  static_assert(lower_self_adjoint_matrix<decltype(rank_update_self_adjoint(M02 {m22}, m22_2022, 4))>);
  static_assert(lower_self_adjoint_matrix<decltype(rank_update_self_adjoint(M00 {m22}, m22_2022, 4))>);

  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(M22 {m22}, m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(M20 {m22}, m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(M02 {m22}, m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(M00 {m22}, m22_2022, 4), m22_25));

  static_assert(upper_self_adjoint_matrix<decltype(rank_update_self_adjoint<TriangleType::upper>(M22 {m22}, m22_2022, 4))>);
  static_assert(upper_self_adjoint_matrix<decltype(rank_update_self_adjoint<TriangleType::upper>(M20 {m22}, m22_2022, 4))>);
  static_assert(upper_self_adjoint_matrix<decltype(rank_update_self_adjoint<TriangleType::upper>(M02 {m22}, m22_2022, 4))>);
  static_assert(upper_self_adjoint_matrix<decltype(rank_update_self_adjoint<TriangleType::upper>(M00 {m22}, m22_2022, 4))>);

  auto z22 = M22::Identity() - M22::Identity(); static_assert(zero_matrix<decltype(z22)>);

  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::upper>(z22, m22_2022, 4), rank_update_self_adjoint<TriangleType::upper>(M22::Zero(), m22_2022, 4)));
  EXPECT_TRUE(is_near(rank_update_self_adjoint<TriangleType::lower>(z22, m22_2022, 4), rank_update_self_adjoint<TriangleType::lower>(M22::Zero(), m22_2022, 4)));

  auto m22_2012 = make_native_matrix<M22>(2, 0, 1, 2);
  auto m20_2012 = M20 {m22_2012};
  auto m02_2012 = M02 {m22_2012};
  auto m00_2012 = M00 {m22_2012};

  auto m22_2102 = make_native_matrix<M22>(2, 1, 0, 2);
  auto m20_2102 = M20 {m22_2102};
  auto m02_2102 = M02 {m22_2102};
  auto m00_2102 = M00 {m22_2102};

  auto m22_93310 = make_native_matrix<M22>(9, 3, 3, 10);
  auto m20_93310 = M20 {m22_93310};
  auto m02_93310 = M02 {m22_93310};
  auto m00_93310 = M00 {m22_93310};

  auto m22_25111130 = make_native_matrix<M22>(25, 11, 11, 30);
  auto m22_29111126 = make_native_matrix<M22>(29, 11, 11, 26);

  auto ru_93310_2102_4_rvalue = rank_update(Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310}, m22_2102, 4);
  EXPECT_TRUE(is_near(ru_93310_2102_4_rvalue, m22_29111126));
  static_assert(eigen_self_adjoint_expr<decltype(ru_93310_2102_4_rvalue)>);
  static_assert(upper_self_adjoint_matrix<decltype(ru_93310_2102_4_rvalue)>);
  static_assert(not std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru_93310_2102_4_rvalue)>>);

  auto sa_93310_2012_4 = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310};

  auto ru_93310_2012_4_lvalue = rank_update(sa_93310_2012_4, m22_2012, 4);
  EXPECT_TRUE(is_near(ru_93310_2012_4_lvalue, m22_25111130));
  EXPECT_TRUE(is_near(sa_93310_2012_4, m22_25111130));
  EXPECT_TRUE(is_near(m22_93310.template selfadjointView<Eigen::Lower>(), m22_25111130));
  static_assert(eigen_self_adjoint_expr<decltype(ru_93310_2012_4_lvalue)>);
  static_assert(lower_self_adjoint_matrix<decltype(ru_93310_2012_4_lvalue)>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru_93310_2012_4_lvalue)>>);

  m22_93310 = make_native_matrix<M22>(9, 3, 3, 10);

  auto ru_93310_2012_4_const_lvalue = rank_update(std::as_const(sa_93310_2012_4), m22_2012, 4);
  EXPECT_TRUE(is_near(ru_93310_2012_4_const_lvalue, m22_25111130));
  EXPECT_TRUE(is_near(sa_93310_2012_4, make_native_matrix<M22>(9, 3, 3, 10)));
  static_assert(eigen_self_adjoint_expr<decltype(ru_93310_2012_4_const_lvalue)>);
  static_assert(not std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru_93310_2012_4_const_lvalue)>>);

  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310}, m22_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M20, Eigen::Upper> {m20_93310}, m22_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M02, Eigen::Lower> {m02_93310}, m22_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M00, Eigen::Upper> {m00_93310}, m22_2102, 4), m22_29111126));

  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310}, m20_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M20, Eigen::Lower> {m20_93310}, m20_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M02, Eigen::Upper> {m02_93310}, m20_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M00, Eigen::Lower> {m00_93310}, m20_2012, 4), m22_25111130));

  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310}, m02_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M20, Eigen::Upper> {m20_93310}, m02_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M02, Eigen::Lower> {m02_93310}, m02_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M00, Eigen::Upper> {m00_93310}, m02_2102, 4), m22_29111126));

  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310}, m00_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M20, Eigen::Lower> {m20_93310}, m00_2012, 4), m22_25111130));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M02, Eigen::Upper> {m02_93310}, m00_2102, 4), m22_29111126));
  EXPECT_TRUE(is_near(rank_update(Eigen::SelfAdjointView<M00, Eigen::Lower> {m00_93310}, m00_2012, 4), m22_25111130));
}


TEST(eigen3, rank_update_triangular)
{
  auto m22_3033 = make_native_matrix<M22>(3, 0, 3, 3);
  auto m20_3033 = make_native_matrix<M20>(m22_3033);
  auto m02_3033 = make_native_matrix<M02>(m22_3033);
  auto m00_3033 = make_native_matrix<M00>(m22_3033);

  auto m22_3303 = make_native_matrix<M22>(3, 3, 0, 3);
  auto m20_3303 = make_native_matrix<M20>(m22_3303);
  auto m02_3303 = make_native_matrix<M02>(m22_3303);
  auto m00_3303 = make_native_matrix<M00>(m22_3303);

  auto m22_2022 = make_native_matrix<M22>(2, 0, 2, 2);
  auto m22_5055 = make_native_matrix<M22>(5, 0, 5, 5);
  auto m22_5505 = make_native_matrix<M22>(5, 5, 0, 5);

  auto m22 = m22_3033;
  auto m20 = m20_3033;
  auto m02 = m02_3033;
  auto m00 = m00_3033;

  rank_update_triangular(m22, m22_2022, 4); EXPECT_TRUE(is_near(m22, m22_5055));
  rank_update_triangular(m20, m22_2022, 4); EXPECT_TRUE(is_near(m20, m22_5055));
  rank_update_triangular(m02, m22_2022, 4); EXPECT_TRUE(is_near(m02, m22_5055));
  rank_update_triangular(m00, m22_2022, 4); EXPECT_TRUE(is_near(m00, m22_5055));

  EXPECT_TRUE(is_near(rank_update_triangular(M22 {m22_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update_triangular(M20 {m22_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update_triangular(M02 {m22_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update_triangular(M00 {m22_3033}, m22_2022, 4), m22_5055));

  static_assert(lower_triangular_matrix<decltype(rank_update_triangular(M22 {m22_3033}, m22_2022, 4))>);
  static_assert(lower_triangular_matrix<decltype(rank_update_triangular(M20 {m22_3033}, m22_2022, 4))>);
  static_assert(lower_triangular_matrix<decltype(rank_update_triangular(M02 {m22_3033}, m22_2022, 4))>);
  static_assert(lower_triangular_matrix<decltype(rank_update_triangular(M00 {m22_3033}, m22_2022, 4))>);

  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(M22 {m22_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(M20 {m22_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(M02 {m22_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(M00 {m22_3303}, m22_2022, 4), m22_5505));

  static_assert(upper_triangular_matrix<decltype(rank_update_triangular<TriangleType::upper>(M22 {m22_3303}, m22_2022, 4))>);
  static_assert(upper_triangular_matrix<decltype(rank_update_triangular<TriangleType::upper>(M20 {m22_3303}, m22_2022, 4))>);
  static_assert(upper_triangular_matrix<decltype(rank_update_triangular<TriangleType::upper>(M02 {m22_3303}, m22_2022, 4))>);
  static_assert(upper_triangular_matrix<decltype(rank_update_triangular<TriangleType::upper>(M00 {m22_3303}, m22_2022, 4))>);

  auto z22 = M22::Identity() - M22::Identity(); static_assert(zero_matrix<decltype(z22)>);

  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::upper>(z22, m22_2022, 4), rank_update_triangular<TriangleType::upper>(M22::Zero(), m22_2022, 4)));
  EXPECT_TRUE(is_near(rank_update_triangular<TriangleType::lower>(z22, m22_2022, 4), rank_update_triangular<TriangleType::lower>(M22::Zero(), m22_2022, 4)));

  m22 = m22_3033;
  m20 = m20_3033;
  m02 = m02_3033;
  m00 = m00_3033;

  auto tl22_lvalue = Eigen::TriangularView<M22, Eigen::Lower> {m22};
  auto tl20_lvalue = Eigen::TriangularView<M20, Eigen::Lower> {m20};
  auto tl02_lvalue = Eigen::TriangularView<M02, Eigen::Lower> {m02};
  auto tl00_lvalue = Eigen::TriangularView<M00, Eigen::Lower> {m00};

  auto ru22_lvalue = rank_update(tl22_lvalue, m22_2022, 4); EXPECT_TRUE(is_near(ru22_lvalue, m22_5055));
  auto ru20_lvalue = rank_update(tl20_lvalue, m22_2022, 4); EXPECT_TRUE(is_near(ru20_lvalue, m22_5055));
  auto ru02_lvalue = rank_update(tl02_lvalue, m22_2022, 4); EXPECT_TRUE(is_near(ru02_lvalue, m22_5055));
  auto ru00_lvalue = rank_update(tl00_lvalue, m22_2022, 4); EXPECT_TRUE(is_near(ru00_lvalue, m22_5055));

  EXPECT_TRUE(is_near(tl22_lvalue, m22_5055));
  EXPECT_TRUE(is_near(tl20_lvalue, m22_5055));
  EXPECT_TRUE(is_near(tl02_lvalue, m22_5055));
  EXPECT_TRUE(is_near(tl00_lvalue, m22_5055));

  EXPECT_TRUE(is_near(m22, m22_5055));
  EXPECT_TRUE(is_near(m20, m22_5055));
  EXPECT_TRUE(is_near(m02, m22_5055));
  EXPECT_TRUE(is_near(m00, m22_5055));

  static_assert(triangular_matrix<decltype(ru22_lvalue)>); static_assert(Eigen3::eigen_TriangularView<decltype(ru22_lvalue)>);
  static_assert(triangular_matrix<decltype(ru20_lvalue)>); static_assert(Eigen3::eigen_TriangularView<decltype(ru20_lvalue)>);
  static_assert(triangular_matrix<decltype(ru02_lvalue)>); static_assert(Eigen3::eigen_TriangularView<decltype(ru02_lvalue)>);
  static_assert(triangular_matrix<decltype(ru00_lvalue)>); static_assert(Eigen3::eigen_TriangularView<decltype(ru00_lvalue)>);

  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru22_lvalue)>>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru20_lvalue)>>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru02_lvalue)>>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru00_lvalue)>>);

  m22 = m22_3303;
  auto ru22_rvalue = rank_update(Eigen::TriangularView<M22, Eigen::Upper> {m22}, m22_2022, 4);
  EXPECT_TRUE(is_near(ru22_rvalue, m22_5505));
  static_assert(eigen_triangular_expr<decltype(ru22_rvalue)>);
  static_assert(upper_triangular_matrix<decltype(ru22_rvalue)>);
  static_assert(not std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru22_rvalue)>>);

  m22 = m22_3033;
  auto t22_lvalue = Eigen::TriangularView<M22, Eigen::Lower> {m22};
  auto ru22_const_lvalue = rank_update(std::as_const(t22_lvalue), m22_2022, 4);
  EXPECT_TRUE(is_near(ru22_const_lvalue, m22_5055));
  EXPECT_TRUE(is_near(t22_lvalue, m22_3033));
  static_assert(eigen_triangular_expr<decltype(ru22_const_lvalue)>);
  static_assert(lower_triangular_matrix<decltype(ru22_const_lvalue)>);
  static_assert(not std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru22_const_lvalue)>>);

  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M22, Eigen::Lower> {m22_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M20, Eigen::Upper> {m20_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M02, Eigen::Lower> {m02_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M00, Eigen::Upper> {m00_3303}, m22_2022, 4), m22_5505));

  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M22, Eigen::Upper> {m22_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M20, Eigen::Lower> {m20_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M02, Eigen::Upper> {m02_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M00, Eigen::Lower> {m00_3033}, m22_2022, 4), m22_5055));

  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M22, Eigen::Lower> {m22_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M20, Eigen::Upper> {m20_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M02, Eigen::Lower> {m02_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M00, Eigen::Upper> {m00_3303}, m22_2022, 4), m22_5505));

  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M22, Eigen::Upper> {m22_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M20, Eigen::Lower> {m20_3033}, m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M02, Eigen::Upper> {m02_3303}, m22_2022, 4), m22_5505));
  EXPECT_TRUE(is_near(rank_update(Eigen::TriangularView<M00, Eigen::Lower> {m00_3033}, m22_2022, 4), m22_5055));
}


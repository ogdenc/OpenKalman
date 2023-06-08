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
  using M41 = eigen_matrix_t<double, 4, 1>;
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


TEST(eigen3, to_diagonal)
{
  auto m11 = M11 {3};

  EXPECT_TRUE(is_near(to_diagonal(m11), m11));
  EXPECT_TRUE(is_near(to_diagonal(M10 {m11}), m11));
  EXPECT_TRUE(is_near(to_diagonal(M01 {m11}), m11));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
  EXPECT_TRUE(is_near(to_diagonal(M00 {m11}), m11));
#pragma GCC diagnostic pop
  EXPECT_TRUE(is_near(to_diagonal(Eigen::DiagonalWrapper {m11}), m11));
  EXPECT_TRUE(is_near(to_diagonal(Eigen::DiagonalMatrix<double, 1> {m11}), m11));
  EXPECT_TRUE(is_near(to_diagonal(m11.selfadjointView<Eigen::Lower>()), m11));
  EXPECT_TRUE(is_near(to_diagonal(m11.triangularView<Eigen::Upper>()), m11));
  EXPECT_TRUE(is_near(to_diagonal(EigenWrapper {m11.triangularView<Eigen::Upper>()}), m11));
}


TEST(eigen3, diagonal_of_matrix)
{
  auto m11 = M11 {3};
  auto m10_1 = M10 {m11};
  auto m01_1 = M01 {m11};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
  auto m00_11 = M00 {m11};
#pragma GCC diagnostic pop

  EXPECT_TRUE(is_near(diagonal_of(m11), m11));
  EXPECT_TRUE(is_near(diagonal_of(m10_1), m11)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(m10_1))>);
  EXPECT_TRUE(is_near(diagonal_of(m01_1), m11)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(m01_1))>);
  EXPECT_TRUE(is_near(diagonal_of(m00_11), m11)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(m00_11))>);

  auto m21 = M21 {1, 4};
  auto m20_1 = M20 {m21};
  auto m01_2 = M01 {m21};
  auto m00_21 = M00 {m21};

  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m20_2 = M20 {m22};
  auto m02_2 = M02 {m22};
  auto m00_22 = M00 {m22};

  EXPECT_TRUE(is_near(diagonal_of(m22), m21));
  EXPECT_TRUE(is_near(diagonal_of(m20_2), m21)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(m20_2))>);
  EXPECT_TRUE(is_near(diagonal_of(m02_2), m21)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(m02_2))>);
  EXPECT_TRUE(is_near(diagonal_of(m00_22), m21)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(m00_22))>);
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {m22}), m21));

  EXPECT_TRUE(is_near(diagonal_of(M22 {m22}), m21));
  EXPECT_TRUE(is_near(diagonal_of(M20 {m20_2}), m21)); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(M20 {m20_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(M02 {m02_2}), m21)); static_assert(not has_dynamic_dimensions<decltype(diagonal_of(M02 {m02_2}))>);
  EXPECT_TRUE(is_near(diagonal_of(M00 {m00_22}), m21)); static_assert(has_dynamic_dimensions<decltype(diagonal_of(M00 {m00_22}))>);
}


TEST(eigen3, diagonal_of_eigen_diagonal)
{
  auto m21 = M21 {1, 4};
  auto m20_1 = M20 {m21};
  auto m01_2 = M01 {m21};
  auto m00_21 = M00 {m21};

  auto dm2 = Eigen::DiagonalMatrix<double, 2> {m21};
  auto dm0_2 = Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m01_2};

  EXPECT_TRUE(is_near(diagonal_of(dm2), m21));
  EXPECT_TRUE(is_near(diagonal_of(dm0_2), m01_2));

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalMatrix<double, 2> {dm2}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {dm0_2}), m21));
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {Eigen::DiagonalMatrix<double, Eigen::Dynamic> {dm0_2}}), m21));

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m21}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m20_1}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m01_2}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m00_21}), m21));
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {Eigen::DiagonalWrapper {m21}}), m21));

  static_assert(std::is_assignable_v<decltype(nested_matrix(Eigen::DiagonalWrapper {m21})), M21>);
  static_assert(std::is_assignable_v<decltype(diagonal_of(Eigen::DiagonalWrapper {m21})), M21>);

  auto m12 = M12 {1, 4};
  auto m10_2 = M10 {m12};
  auto m02_1 = M02 {m12};
  auto m00_12 = M00 {m12};

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m12}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m10_2}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m02_1}), m21));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m00_12}), m21));

  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m20_2 = M20 {m22};
  auto m02_2 = M02 {m22};
  auto m00_22 = M00 {m22};

  const auto m41 = make_dense_writable_matrix_from<M41>(1, 3, 2, 4);

  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m22}), m41));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m20_2}), m41));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m02_2}), m41));
  EXPECT_TRUE(is_near(diagonal_of(Eigen::DiagonalWrapper {m00_22}), m41));
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {Eigen::DiagonalWrapper {m22}}), m41));
}


TEST(eigen3, diagonal_of_identity)
{
  auto i22 = M22::Identity();
  auto i20_2 = M20::Identity(2, 2);
  auto i02_2 = M02::Identity(2, 2);
  auto i00_22 = M00::Identity(2, 2);

  static_assert(not has_dynamic_dimensions<decltype(diagonal_of(i22))>);
  static_assert(not has_dynamic_dimensions<decltype(diagonal_of(i20_2))>);
  static_assert(not has_dynamic_dimensions<decltype(diagonal_of(i02_2))>);
  static_assert(has_dynamic_dimensions<decltype(diagonal_of(i00_22))>);

  static_assert(constant_coefficient_v<decltype(diagonal_of(i22))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(i20_2))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(i02_2))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(i00_22))> == 1);

  EXPECT_TRUE(is_near(diagonal_of(i22), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(i20_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(i02_2), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(i00_22), M21::Constant(1)));
  EXPECT_TRUE(is_near(EigenWrapper {diagonal_of(i22)}, M21::Constant(1)));

  static_assert(constant_coefficient_v<decltype(diagonal_of(M22::Identity()))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2, 2)))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(M02::Identity(2, 2)))> == 1);
  static_assert(constant_coefficient_v<decltype(diagonal_of(M00::Identity(2, 2)))> == 1);

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity()), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2, 2)), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(M02::Identity(2, 2)), M21::Constant(1)));
  EXPECT_TRUE(is_near(diagonal_of(M00::Identity(2, 2)), M21::Constant(1)));

  static_assert(constant_adapter<decltype(diagonal_of(std::declval<Eigen3::IdentityMatrix<M33>>()))>);
  static_assert(row_dimension_of_v<decltype(diagonal_of(std::declval<Eigen3::IdentityMatrix<M33>>()))> == 3);
  static_assert(column_dimension_of_v<decltype(diagonal_of(std::declval<Eigen3::IdentityMatrix<M33>>()))> == 1);
}


TEST(eigen3, diagonal_of_self_adjoint)
{
  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M20 {m22_93310};
  auto m02_93310 = M02 {m22_93310};
  auto m00_93310 = M00 {m22_93310};

  auto m21_910 = make_dense_writable_matrix_from<M21>(9, 10);

  EXPECT_TRUE(is_near(diagonal_of(m22_93310.template selfadjointView<Eigen::Upper>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m20_93310.template selfadjointView<Eigen::Lower>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m02_93310.template selfadjointView<Eigen::Upper>()), m21_910));
  EXPECT_TRUE(is_near(diagonal_of(m00_93310.template selfadjointView<Eigen::Lower>()), m21_910));
  EXPECT_TRUE(is_near(EigenWrapper {diagonal_of(m22_93310.template selfadjointView<Eigen::Upper>())}, m21_910));
}


TEST(eigen3, diagonal_of_triangular)
{
  auto m22_3013 = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  auto m20_3013 = M20 {m22_3013};
  auto m02_3013 = M02 {m22_3013};
  auto m00_3013 = M00 {m22_3013};

  auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m20_3103 = M20 {m22_3103};
  auto m02_3103 = M02 {m22_3103};
  auto m00_3103 = M00 {m22_3103};

  auto m21_33 = make_dense_writable_matrix_from<M21>(3, 3);

  EXPECT_TRUE(is_near(diagonal_of(m22_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m20_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m02_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m00_3013.template triangularView<Eigen::Lower>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {m22_3013.template triangularView<Eigen::Lower>()}), m21_33));

  EXPECT_TRUE(is_near(diagonal_of(m22_3103.template triangularView<Eigen::Upper>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m20_3103.template triangularView<Eigen::Upper>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m02_3103.template triangularView<Eigen::Upper>()), m21_33));
  EXPECT_TRUE(is_near(diagonal_of(m00_3103.template triangularView<Eigen::Upper>()), m21_33));

  auto z11 = M11::Identity() - M11::Identity();

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  EXPECT_TRUE(is_near(diagonal_of(z22.template triangularView<Eigen::Upper>()), M21::Zero())); static_assert(zero_matrix<decltype(diagonal_of(z22.template triangularView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z20_2.template triangularView<Eigen::Lower>()), M21::Zero())); static_assert(zero_matrix<decltype(diagonal_of(z20_2.template triangularView<Eigen::Lower>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z02_2.template triangularView<Eigen::Upper>()), M21::Zero())); static_assert(zero_matrix<decltype(diagonal_of(z02_2.template triangularView<Eigen::Upper>()))>);
  EXPECT_TRUE(is_near(diagonal_of(z00_22.template triangularView<Eigen::Lower>()), M21::Zero())); static_assert(zero_matrix<decltype(diagonal_of(z00_22.template triangularView<Eigen::Lower>()))>);
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {z22.template triangularView<Eigen::Upper>()}), M21::Zero()));

  auto c11_2 {M11::Identity() + M11::Identity()};

  auto c22_2 = c11_2.replicate<2, 2>();
  auto c20_2_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic>(c11_2, 2, 2);
  auto c02_2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2>(c11_2, 2, 2);
  auto c00_22_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic>(c11_2, 2, 2);

  EXPECT_TRUE(is_near(diagonal_of(c22_2.template triangularView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c22_2.template triangularView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c20_2_2.template triangularView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c20_2_2.template triangularView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c02_2_2.template triangularView<Eigen::Upper>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c02_2_2.template triangularView<Eigen::Upper>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(c00_22_2.template triangularView<Eigen::Lower>()), M21::Constant(2))); static_assert(constant_coefficient_v<decltype(diagonal_of(c00_22_2.template triangularView<Eigen::Lower>()))> == 2);
  EXPECT_TRUE(is_near(diagonal_of(EigenWrapper {c22_2.template triangularView<Eigen::Upper>()}), M21::Constant(2)));

  EXPECT_TRUE(is_near(diagonal_of(M22::Identity().template triangularView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M22::Identity().template triangularView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Lower>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Upper>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Upper>()))> == 1);
  EXPECT_TRUE(is_near(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Lower>()), M21::Constant(1))); static_assert(constant_coefficient_v<decltype(diagonal_of(M20::Identity(2,2).template triangularView<Eigen::Lower>()))> == 1);
}


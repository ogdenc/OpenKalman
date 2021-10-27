/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests relating to Eigen3::DiagonalMatrix.
 */

#include "eigen3.gtest.hpp"

#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using Mat2 = eigen_matrix_t<double, 2, 2>;
using Mat3 = eigen_matrix_t<double, 3, 3>;
using Axis2 = Coefficients<Axis, Axis>;
using cdouble = std::complex<double>;

TEST(eigen3, Diagonal_class)
{
  DiagonalMatrix<eigen_matrix_t<double, 3, 1>> d1; // default constructor
  d1 << 1, 2, 3;
  EXPECT_TRUE(is_near(d1.nested_matrix(), make_native_matrix<double, 3, 1>(1, 2, 3)));
  DiagonalMatrix d2 = DiagonalMatrix {make_native_matrix(1., 2, 3)}; // native matrix and move constructors
  EXPECT_TRUE(is_near(d2, d1));
  DiagonalMatrix d3 {d2}; // copy constructor
  EXPECT_TRUE(is_near(d3, d1));
  DiagonalMatrix d4 {1., 2, 3}; // column scalar constructor
  EXPECT_TRUE(is_near(d4, d1));
  DiagonalMatrix<eigen_matrix_t<double, 3, 1>> d4a {1., 0, 0, 0, 2, 0, 0, 0, 3}; // square matrix scalar constructor
  EXPECT_TRUE(is_near(d4a.nested_matrix(), d1.nested_matrix()));
  DiagonalMatrix d5 {(DiagonalMatrix {ZeroMatrix<double, 3, 3>()}).nested_matrix()}; // zero matrix constructor
  EXPECT_TRUE(is_near(d5, ZeroMatrix<double, 3, 3>()));
  DiagonalMatrix d6 = MatrixTraits<DiagonalMatrix<eigen_matrix_t<double, 3, 1>>>::identity();
  EXPECT_TRUE(is_near(d6, Mat3::Identity()));
  DiagonalMatrix d7 {0.7 * Mat3::Identity()};
  EXPECT_TRUE(is_near(d7, Mat3::Identity() * 0.7));
  DiagonalMatrix d7a {((0.7 * Mat3::Identity()) * (0.3 * Mat3::Identity() * 0.7 + 0.7 * Mat3::Identity()) - Mat3::Identity() * 0.3)};
  EXPECT_TRUE(is_near(d7a, Mat3::Identity() * 0.337));
  DiagonalMatrix<eigen_matrix_t<double, 3, 1>> d8 = ((0.7 * Mat3::Identity()) * (0.3 * Mat3::Identity() * 0.7 + 0.7 * Mat3::Identity()) - Mat3::Identity() * 0.3);
  EXPECT_TRUE(is_near(d8, Mat3::Identity() * 0.337));
  DiagonalMatrix<eigen_matrix_t<double, 3, 1>> d9 = ZeroMatrix<double, 3, 3>();
  EXPECT_TRUE(is_near(d9, Mat3::Zero()));
  DiagonalMatrix<eigen_matrix_t<double, 3, 1>> d10 = eigen_matrix_t<double, 3, 3>::Identity();
  EXPECT_TRUE(is_near(d10, Mat3::Identity()));
  EXPECT_TRUE(is_near(DiagonalMatrix<eigen_matrix_t<double, 3, 1>>(ZeroMatrix<double, 3, 1>()), Mat3::Zero()));
  //
  DiagonalMatrix d11a_s = SelfAdjointMatrix<Mat2, TriangleType::diagonal>{9, 10};
  EXPECT_TRUE(is_near(d11a_s, make_native_matrix<Mat2>(9, 0, 0, 10)));
  DiagonalMatrix d11b_s = SelfAdjointMatrix<DiagonalMatrix<eigen_matrix_t<double, 2, 1>>, TriangleType::diagonal>{9, 10};
  EXPECT_TRUE(is_near(d11b_s, make_native_matrix<Mat2>(9, 0, 0, 10)));
  DiagonalMatrix d11c_s = SelfAdjointMatrix<DiagonalMatrix<eigen_matrix_t<double, 2, 1>>, TriangleType::lower>{9, 10};
  EXPECT_TRUE(is_near(d11c_s, make_native_matrix<Mat2>(9, 0, 0, 10)));
  DiagonalMatrix d11a_t = TriangularMatrix<Mat2, TriangleType::diagonal>{3, 3};
  EXPECT_TRUE(is_near(d11a_t, make_native_matrix<Mat2>(3, 0, 0, 3)));
  DiagonalMatrix d11b_t = TriangularMatrix<DiagonalMatrix<eigen_matrix_t<double, 2, 1>>, TriangleType::diagonal>{3, 3};
  EXPECT_TRUE(is_near(d11b_t, make_native_matrix<Mat2>(3, 0, 0, 3)));
  DiagonalMatrix d11c_t = TriangularMatrix<DiagonalMatrix<eigen_matrix_t<double, 2, 1>>, TriangleType::lower>{3, 3};
  EXPECT_TRUE(is_near(d11c_t, make_native_matrix<Mat2>(3, 0, 0, 3)));
  //
  d3 = d2; // copy assignment.
  EXPECT_TRUE(is_near(d3, d1));
  d4 = DiagonalMatrix {1., 2, 3}; // move assignment.
  EXPECT_TRUE(is_near(d4, d1));
  d4 = ZeroMatrix<double, 3, 3>();
  EXPECT_TRUE(is_near(d4, ZeroMatrix<double, 3, 3>()));
  d4 = Mat3::Identity();
  EXPECT_TRUE(is_near(d4, Mat3::Identity()));
  d4 = DiagonalMatrix {make_native_matrix<double, 3, 3>(4, 0, 0, 0, 5, 0, 0, 0, 6)};
  EXPECT_TRUE(is_near(d4, DiagonalMatrix {4., 5, 6}));
  d4 = DiagonalMatrix<eigen_matrix_t<double, 3, 1>> {1., 0, 0, 0, 2, 0, 0, 0, 3};
  EXPECT_TRUE(is_near(d4, d1));
  //
  d11a_s = SelfAdjointMatrix<Mat2, TriangleType::diagonal>{9, 10};
  EXPECT_TRUE(is_near(d11a_s, make_native_matrix<Mat2>(9, 0, 0, 10)));
  d11a_s = SelfAdjointMatrix<DiagonalMatrix<eigen_matrix_t<double, 2, 1>>, TriangleType::diagonal>{9, 10};
  EXPECT_TRUE(is_near(d11b_s, make_native_matrix<Mat2>(9, 0, 0, 10)));
  d11a_s = SelfAdjointMatrix<DiagonalMatrix<eigen_matrix_t<double, 2, 1>>, TriangleType::lower>{9, 10};
  EXPECT_TRUE(is_near(d11c_s, make_native_matrix<Mat2>(9, 0, 0, 10)));
  d11a_s = TriangularMatrix<Mat2, TriangleType::diagonal>{3, 3};
  EXPECT_TRUE(is_near(d11a_t, make_native_matrix<Mat2>(3, 0, 0, 3)));
  d11a_s = TriangularMatrix<DiagonalMatrix<eigen_matrix_t<double, 2, 1>>, TriangleType::diagonal>{3, 3};
  EXPECT_TRUE(is_near(d11b_t, make_native_matrix<Mat2>(3, 0, 0, 3)));
  d11a_s = TriangularMatrix<DiagonalMatrix<eigen_matrix_t<double, 2, 1>>, TriangleType::lower>{3, 3};
  EXPECT_TRUE(is_near(d11c_t, make_native_matrix<Mat2>(3, 0, 0, 3)));
  //
  d1 += d2;
  EXPECT_TRUE(is_near(d1, DiagonalMatrix {2., 4, 6}));
  d1 -= DiagonalMatrix {1., 2, 3};
  EXPECT_TRUE(is_near(d1, DiagonalMatrix {1., 2, 3}));
  d1 *= 3;
  EXPECT_TRUE(is_near(d1, DiagonalMatrix {3., 6, 9}));
  d1 /= 3;
  EXPECT_TRUE(is_near(d1, DiagonalMatrix {1., 2, 3}));
  d1 *= d2;
  EXPECT_TRUE(is_near(d1, DiagonalMatrix {1., 4, 9}));
  EXPECT_TRUE(is_near(d1.square_root(), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(d1.square(), DiagonalMatrix {1., 16, 81}));

  EXPECT_EQ((DiagonalMatrix<eigen_matrix_t<double, 3, 1>>::rows()), 3);
  EXPECT_EQ((DiagonalMatrix<eigen_matrix_t<double, 3, 1>>::cols()), 3);
  EXPECT_TRUE(is_near(DiagonalMatrix<eigen_matrix_t<double, 3, 1>>::zero(), Mat3::Zero()));
  EXPECT_TRUE(is_near(DiagonalMatrix<eigen_matrix_t<double, 3, 1>>::identity(), Mat3::Identity()));
}

TEST(eigen3, Diagonal_subscripts)
{
  auto el = DiagonalMatrix {1., 2, 3};
  set_element(el, 5.5, 1);
  EXPECT_NEAR(get_element(el, 1), 5.5, 1e-8);
  set_element(el, 6.5, 2, 2);
  EXPECT_NEAR(get_element(el, 2), 6.5, 1e-8);
  bool test = false; try { set_element(el, 7.5, 2, 0); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_NEAR(get_element(el, 2, 0), 0, 1e-8);

  DiagonalMatrix<eigen_matrix_t<double, 3, 1>> d1 {1, 4, 9};

  static_assert(element_gettable<DiagonalMatrix<eigen_matrix_t<double, 3, 1>>, 2>);
  static_assert(element_gettable<decltype(d1), 2>);
  static_assert(element_gettable<decltype(d1), 1>);
  static_assert(not element_gettable<decltype(d1), 3>);
  static_assert(element_settable<decltype(d1), 2>);
  static_assert(element_settable<decltype(d1), 1>);
  EXPECT_EQ(d1(2), 9);
  EXPECT_EQ(d1(0), 1);
  EXPECT_EQ(d1(0, 1), 0);
  EXPECT_EQ(d1(1, 1), 4);

  d1(0,0) = 5;
  d1(1) = 6;
  d1(2) = 7;
  test = false; try { d1(1, 0) = 3; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_EQ(d1(1, 0), 0);

  EXPECT_TRUE(is_near(d1, DiagonalMatrix {5., 6, 7}));
  EXPECT_NEAR(d1(0), 5, 1e-6);
  EXPECT_NEAR(d1(1), 6, 1e-6);
  EXPECT_NEAR(d1(2), 7, 1e-6);
  EXPECT_NEAR(d1[0], 5, 1e-6);
  EXPECT_NEAR(d1[1], 6, 1e-6);
  EXPECT_NEAR(d1[2], 7, 1e-6);
  EXPECT_NEAR(d1(0, 0), 5, 1e-6);
  EXPECT_NEAR(d1(0, 1), 0, 1e-6);
  EXPECT_NEAR(d1(0, 2), 0, 1e-6);
  EXPECT_NEAR(d1(1, 0), 0, 1e-6);
  EXPECT_NEAR(d1(1, 1), 6, 1e-6);
  EXPECT_NEAR(d1(1, 2), 0, 1e-6);
  EXPECT_NEAR(d1(2, 0), 0, 1e-6);
  EXPECT_NEAR(d1(2, 1), 0, 1e-6);
  EXPECT_NEAR(d1(2, 2), 7, 1e-6);
  //
  EXPECT_NEAR((DiagonalMatrix {1., 2, 3}).nested_matrix()[0], 1, 1e-6);
  EXPECT_NEAR((DiagonalMatrix {1., 2, 3}).nested_matrix()[1], 2, 1e-6);
  EXPECT_NEAR((DiagonalMatrix {1., 2, 3}).nested_matrix()[2], 3, 1e-6);
}

TEST(eigen3, Diagonal_traits)
{
  static_assert(diagonal_matrix<decltype(DiagonalMatrix{2.})>);
  static_assert(diagonal_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(self_adjoint_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(triangular_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(lower_triangular_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(upper_triangular_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(not identity_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(not zero_matrix<decltype(DiagonalMatrix{2, 3})>);
  static_assert(covariance_nestable<decltype(DiagonalMatrix{2, 3})>);
  static_assert(covariance_nestable<decltype(Mat3::Identity())>);
  static_assert(identity_matrix<decltype(Mat3::Identity() * Mat3::Identity())>);
  static_assert(diagonal_matrix<decltype(0.3 * Mat3::Identity() + 0.7 * Mat3::Identity() * 0.7)>);
  static_assert(diagonal_matrix<decltype(0.7 * Mat3::Identity() * 0.7 - Mat3::Identity() * 0.3)>);
  static_assert(diagonal_matrix<decltype((0.7 * Mat3::Identity()) * (Mat3::Identity() + Mat3::Identity()))>);
  // MatrixTraits
  using D = DiagonalMatrix<eigen_matrix_t<double, 3, 1>>;
  EXPECT_TRUE(is_near(MatrixTraits<D>::make(make_native_matrix<double, 3, 1>(1, 2, 3)),
    make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3)));
  EXPECT_TRUE(is_near(MatrixTraits<D>::make(1, 2, 3), make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3)));
  EXPECT_TRUE(is_near(make_native_matrix<D>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3), make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3)));
  EXPECT_TRUE(is_near(MatrixTraits<D>::zero(), eigen_matrix_t<double, 3, 3>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<D>::identity(), eigen_matrix_t<double, 3, 3>::Identity()));
}

TEST(eigen3, Diagonal_overloads)
{
  EXPECT_TRUE(is_near(make_native_matrix(DiagonalMatrix(1., 2, 3)), make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3)));
  EXPECT_TRUE(is_near(Cholesky_square(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 4, 9}));
  EXPECT_TRUE(is_near(Cholesky_factor(DiagonalMatrix {1., 4, 9}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(Cholesky_square(eigen_matrix_t<double, 1, 1>(4)), eigen_matrix_t<double, 1, 1>(16)));
  EXPECT_TRUE(is_near(Cholesky_factor(eigen_matrix_t<double, 1, 1>(4)), eigen_matrix_t<double, 1, 1>(2)));
  EXPECT_TRUE(is_near(Cholesky_square(Mat2::Identity() * 0.1), DiagonalMatrix {0.01, 0.01}));
  EXPECT_TRUE(is_near(Cholesky_factor(Mat2::Identity() * 0.01), DiagonalMatrix {0.1, 0.1}));
  EXPECT_TRUE(is_near(Cholesky_square(DiagonalMatrix {9.}), eigen_matrix_t<double, 1, 1>(81)));
  EXPECT_TRUE(is_near(Cholesky_factor(DiagonalMatrix {9.}), eigen_matrix_t<double, 1, 1>(3)));

  EXPECT_TRUE(is_near(diagonal_of(DiagonalMatrix {1., 2, 3}), make_native_matrix<double, 3, 1>(1., 2, 3)));

  EXPECT_TRUE(is_near(transpose(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(transpose(DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}), DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}));

  EXPECT_TRUE(is_near(adjoint(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(adjoint(DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}), DiagonalMatrix {cdouble(1,-2), cdouble(2,-3), 3}));

  EXPECT_NEAR(determinant(DiagonalMatrix {2., 3, 4}), 24, 1e-6);

  EXPECT_NEAR(trace(DiagonalMatrix {2., 3, 4}), 9, 1e-6);
  //
  auto s2 = DiagonalMatrix {3., 3};
  rank_update(s2, DiagonalMatrix {2., 2}, 4);
  EXPECT_TRUE(is_near(s2, MatrixTraits<Mat2>::make(5., 0, 0, 5)));
  s2 = DiagonalMatrix {3., 3};
  rank_update(s2, 2 * Mat2::Identity(), 4);
  EXPECT_TRUE(is_near(s2, MatrixTraits<Mat2>::make(5., 0, 0, 5)));
  //
  using M1by1 = eigen_matrix_t<double, 1, 1>;
  auto s2a = M1by1 {3.};
  rank_update(s2a, M1by1 {2.}, 4);
  EXPECT_TRUE(is_near(s2a, M1by1 {5.}));
  //
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, DiagonalMatrix {2., 2}, 4), make_native_matrix<double, 2, 2>(5., 0, 0, 5.)));
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, 2 * Mat2::Identity(), 4), make_native_matrix<Mat2>(5., 0, 0, 5.)));
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, make_native_matrix<Mat2>(2, 0, 0, 2.), 4), make_native_matrix<Mat2>(5., 0, 0, 5.)));
  //
  EXPECT_TRUE(is_near(rank_update(M1by1 {3.}, M1by1 {2.}, 4), M1by1 {5.}));
  //
  EXPECT_TRUE(is_near(solve(DiagonalMatrix {1., 2, 3}, make_native_matrix<double, 3, 1>(4., 10, 18)),
    make_native_matrix(4., 5, 6)));
  EXPECT_TRUE(is_near(reduce_columns(DiagonalMatrix {1., 2, 3}), make_native_matrix(1., 2, 3)));
  EXPECT_TRUE(is_near(reduce_rows(DiagonalMatrix {1., 2, 3}), make_native_matrix(1., 2, 3)));
  EXPECT_TRUE(is_near(LQ_decomposition(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(QR_decomposition(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(LQ_decomposition(DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}), DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}));
  EXPECT_TRUE(is_near(QR_decomposition(DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}), DiagonalMatrix {cdouble(1,2), cdouble(2,3), 3}));
  EXPECT_TRUE(is_near(LQ_decomposition(M1by1 {4}), DiagonalMatrix {4.}));
  EXPECT_TRUE(is_near(QR_decomposition(M1by1 {4}), DiagonalMatrix {4.}));

  using N = std::normal_distribution<double>;
  using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
  using D3 = DiagonalMatrix<eigen_matrix_t<double, 3, 1>>;
  using D0 = DiagonalMatrix<eigen_matrix_t<double, 0, 1>>;

  D2 d2 = MatrixTraits<D2>::zero();
  D0 d0_2 {MatrixTraits<D0>::zero(2, 2)};
  D0 d0_3 {MatrixTraits<D0>::zero(3, 3)};
  for (int i=0; i<100; i++)
  {
    d2 = (d2 * i + randomize<D2>(N {1.0, 0.3}, 2.0)) / (i + 1);
    d0_2 = (d0_2 * i + randomize<D2>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    d0_3 = (d0_3 * i + randomize<D0>(3, 3, N {1.0, 0.3})) / (i + 1);
  }
  D2 d2_offset = {1, 2};
  D3 d0_3_offset = {1, 1, 1};
  EXPECT_TRUE(is_near(d2, d2_offset, 0.1));
  EXPECT_FALSE(is_near(d2, d2_offset, 1e-6));
  EXPECT_TRUE(is_near(d0_2, d2_offset, 0.1));
  EXPECT_FALSE(is_near(d0_2, d2_offset, 1e-6));
  EXPECT_TRUE(is_near(d0_3, d0_3_offset, 0.1));
  EXPECT_FALSE(is_near(d0_3, d0_3_offset, 1e-6));
}

TEST(eigen3, Diagonal_blocks)
{
  EXPECT_TRUE(is_near(concatenate_diagonal(DiagonalMatrix {1., 2, 3}, DiagonalMatrix {4., 5}), DiagonalMatrix {1., 2, 3, 4, 5}));
  EXPECT_TRUE(is_near(concatenate_vertical(DiagonalMatrix {1., 2}, DiagonalMatrix {3., 4}),
    make_native_matrix<4,2>(1., 0, 0, 2, 3, 0, 0, 4)));
  EXPECT_TRUE(is_near(concatenate_horizontal(DiagonalMatrix {1., 2}, DiagonalMatrix {3., 4}),
    make_native_matrix<2,4>(1., 0, 3, 0, 0, 2, 0, 4)));
  EXPECT_TRUE(is_near(split_diagonal(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple {DiagonalMatrix{1., 2, 3}, DiagonalMatrix{4., 5}}));
  const auto a1 = DiagonalMatrix<const eigen_matrix_t<double, 5, 1>>{1., 2, 3, 4, 5};
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(a1), std::tuple {DiagonalMatrix{1., 2, 3}, DiagonalMatrix{4., 5}}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple {DiagonalMatrix{1., 2}, DiagonalMatrix{3., 4}}));
  EXPECT_TRUE(is_near(split_vertical(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(DiagonalMatrix{1., 2, 3, 4, 5}),
    std::tuple {make_native_matrix<double, 3, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0),
               make_native_matrix<double, 2, 5>(
                 0, 0, 0, 4, 0,
                 0, 0, 0, 0, 5)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(DiagonalMatrix{1., 2, 3, 4, 5}),
    std::tuple {make_native_matrix<double, 2, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0),
               make_native_matrix<double, 2, 5>(
                 0, 0, 3, 0, 0,
                 0, 0, 0, 4, 0)}));
  EXPECT_TRUE(is_near(split_horizontal(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(DiagonalMatrix{1., 2, 3, 4, 5}),
    std::tuple {make_native_matrix<double, 5, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      0, 0, 0,
      0, 0, 0),
               make_native_matrix<double, 5, 2>(
                 0, 0,
                 0, 0,
                 0, 0,
                 4, 0,
                 0, 5)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(DiagonalMatrix{1., 2, 3, 4, 5}),
    std::tuple {make_native_matrix<double, 5, 2>(
      1, 0,
      0, 2,
      0, 0,
      0, 0,
      0, 0),
               make_native_matrix<double, 5, 2>(
                 0, 0,
                 0, 0,
                 3, 0,
                 0, 4,
                 0, 0)}));

  EXPECT_TRUE(is_near(column(DiagonalMatrix{1., 2, 3}, 2), make_native_matrix(0., 0, 3)));
  EXPECT_TRUE(is_near(column<1>(DiagonalMatrix{1., 2, 3}), make_native_matrix(0., 2, 0)));
  EXPECT_TRUE(is_near(row(DiagonalMatrix{1., 2, 3}, 2), eigen_matrix_t<double, 1, 3>(0., 0, 3)));
  EXPECT_TRUE(is_near(row<1>(DiagonalMatrix{1., 2, 3}), eigen_matrix_t<double, 1, 3>(0., 2, 0)));

  EXPECT_TRUE(is_near(apply_columnwise(DiagonalMatrix{1., 2, 3}, [](const auto& col){ return make_self_contained(col + col.Constant(1)); }),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_columnwise(DiagonalMatrix{1., 2, 3}, [](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      0, 3, 2,
      0, 1, 5)));

  EXPECT_TRUE(is_near(apply_rowwise(DiagonalMatrix{1., 2, 3}, [](const auto& row){ return make_self_contained(row + row.Constant(1)); }),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_rowwise(DiagonalMatrix{1., 2, 3}, [](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }),
    make_native_matrix<double, 3, 3>(
      1, 0, 0,
      1, 3, 1,
      2, 2, 5)));

  EXPECT_TRUE(is_near(apply_coefficientwise(DiagonalMatrix{1., 2, 3}, [](const auto& x){ return x + 1; }),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_coefficientwise(DiagonalMatrix{1., 2, 3}, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      1, 4, 3,
      2, 3, 7)));
}

TEST(eigen3, Diagonal_arithmetic)
{
  auto d1 = DiagonalMatrix {1., 2, 3};
  auto d2 = DiagonalMatrix {4., 5, 6};
  auto i = Mat3::Identity();
  auto z = ZeroMatrix<double, 3, 3> {};
  EXPECT_TRUE(is_near(d1 + d2, DiagonalMatrix {5., 7, 9})); static_assert(eigen_diagonal_expr<decltype(d1 + d2)>);
  EXPECT_TRUE(is_near(d1 + i, DiagonalMatrix {2., 3, 4})); static_assert(eigen_diagonal_expr<decltype(d1 + i)>);
  EXPECT_TRUE(is_near(d1 + z, d1)); static_assert(eigen_diagonal_expr<decltype(d1 + z)>);
  EXPECT_TRUE(is_near(i + i, DiagonalMatrix {2., 2, 2})); static_assert(diagonal_matrix<decltype(i + i)>);
  EXPECT_TRUE(is_near(i + z, i)); static_assert(identity_matrix<decltype(i + z)>);
  EXPECT_TRUE(is_near(z + i, i)); static_assert(identity_matrix<decltype(z + i)>);
  EXPECT_TRUE(is_near(z + z, z)); static_assert(eigen_zero_expr<decltype(z + z)>);

  EXPECT_TRUE(is_near(d1 - d2, DiagonalMatrix {-3., -3, -3})); static_assert(diagonal_matrix<decltype(d1 - d2)>);
  EXPECT_TRUE(is_near(d1 - i, DiagonalMatrix {0., 1, 2})); static_assert(diagonal_matrix<decltype(d1 - i)>);
  EXPECT_TRUE(is_near(d1 - z, d1)); static_assert(diagonal_matrix<decltype(d1 - z)>);
  EXPECT_TRUE(is_near(i - i, z)); static_assert(zero_matrix<decltype(i - i)>);
  EXPECT_TRUE(is_near(i * (i - i), z)); static_assert(zero_matrix<decltype(i - i)>);
  EXPECT_TRUE(is_near(z * (i - i), z)); static_assert(eigen_zero_expr<decltype(z * (i - i))>);
  EXPECT_TRUE(is_near(i - z, i)); static_assert(identity_matrix<decltype(i - z)>);
  EXPECT_TRUE(is_near(z - i, -i)); static_assert(diagonal_matrix<decltype(z - i)>);
  EXPECT_TRUE(is_near(z - z, z)); static_assert(eigen_zero_expr<decltype(z - z)>);

  EXPECT_TRUE(is_near(d1 * 4, DiagonalMatrix {4., 8, 12})); static_assert(eigen_diagonal_expr<decltype(d1 * 4)>);
  EXPECT_TRUE(is_near(4 * d1, DiagonalMatrix {4., 8, 12})); static_assert(eigen_diagonal_expr<decltype(4 * d1)>);
  EXPECT_TRUE(is_near(d1 / 2, DiagonalMatrix {0.5, 1, 1.5})); static_assert(diagonal_matrix<decltype(d1 / 2)>);
  static_assert(diagonal_matrix<decltype(d1 / 0)>);
  EXPECT_TRUE(is_near(i * 4, DiagonalMatrix {4., 4, 4})); static_assert(diagonal_matrix<decltype(i * 4)>);
  EXPECT_TRUE(is_near(4 * i, DiagonalMatrix {4., 4, 4})); static_assert(diagonal_matrix<decltype(4 * i)>);
  static_assert(diagonal_matrix<decltype(i * 0)>);
  static_assert(not zero_matrix<decltype(i * 0)>);
  EXPECT_TRUE(is_near(i / 2, DiagonalMatrix {0.5, 0.5, 0.5})); static_assert(diagonal_matrix<decltype(i / 2)>);
  EXPECT_TRUE(is_near(z * 4, z)); static_assert(zero_matrix<decltype(z * 4)>);
  EXPECT_TRUE(is_near(4 * z, z)); static_assert(zero_matrix<decltype(4 * z)>);
  EXPECT_TRUE(is_near(z / 4, z)); static_assert(zero_matrix<decltype(z / 4)>);
  static_assert(zero_matrix<decltype(z / 0)>); // This is not technically true, but it's indeterminate at compile time.
  EXPECT_TRUE(is_near((i - i) * 4, z)); static_assert(zero_matrix<decltype((i - i) * 4)>);
  EXPECT_TRUE(is_near((i - i) / 4, z)); static_assert(not zero_matrix<decltype((i - i) / 4)>);
  static_assert(diagonal_matrix<decltype(i / 0)>);

  EXPECT_TRUE(is_near(-d1, DiagonalMatrix {-1., -2, -3})); static_assert(eigen_diagonal_expr<decltype(-d1)>);
  EXPECT_TRUE(is_near(-i, DiagonalMatrix {-1., -1, -1})); static_assert(diagonal_matrix<decltype(-i)>);
  EXPECT_TRUE(is_near(-z, z)); static_assert(eigen_zero_expr<decltype(z)>);

  EXPECT_TRUE(is_near(d1 * d2, DiagonalMatrix {4., 10, 18})); static_assert(eigen_diagonal_expr<decltype(d1 * d2)>);
  EXPECT_TRUE(is_near(d1 * i, d1)); static_assert(eigen_diagonal_expr<decltype(d1 * i)>);
  EXPECT_TRUE(is_near(i * d1, d1)); static_assert(eigen_diagonal_expr<decltype(i * d1)>);
  EXPECT_TRUE(is_near(d1 * z, z)); static_assert(eigen_zero_expr<decltype(d1 * z)>);
  EXPECT_TRUE(is_near(z * d1, z)); static_assert(eigen_zero_expr<decltype(z * d1)>);
  EXPECT_TRUE(is_near(i * i, i)); static_assert(identity_matrix<decltype(i * i)>);
  EXPECT_TRUE(is_near(z * z, z)); static_assert(eigen_zero_expr<decltype(z * z)>);

  EXPECT_TRUE(is_near(d1 * SelfAdjointMatrix {
    1., 2, 3,
    2, 4, 5,
    3, 5, 6}, make_native_matrix<3,3>(
      1., 2, 3,
      4, 8, 10,
      9, 15, 18)));
  EXPECT_TRUE(is_near(SelfAdjointMatrix {
    1., 2, 3,
    2, 4, 5,
    3, 5, 6} * DiagonalMatrix {1., 2, 3}, make_native_matrix<3,3>(
    1., 4, 9,
    2, 8, 15,
    3, 10, 18)));
  EXPECT_TRUE(is_near(DiagonalMatrix {1., 2, 3} * TriangularMatrix {
    1., 0, 0,
    2, 3, 0,
    4, 5, 6}, make_native_matrix<3,3>(
    1., 0, 0,
    4, 6, 0,
    12, 15, 18)));
  EXPECT_TRUE(is_near(TriangularMatrix {
    1., 0, 0,
    2, 3, 0,
    4, 5, 6} * DiagonalMatrix {1., 2, 3}, make_native_matrix<3,3>(
    1., 0, 0,
    2, 6, 0,
    4, 10, 18)));
  EXPECT_TRUE(is_near(DiagonalMatrix {1., 2, 3} * make_native_matrix<double, 3, 3>(
    1, 2, 3,
    4, 5, 6,
    7, 8, 9), make_native_matrix<3,3>(
    1., 2, 3,
    8, 10, 12,
    21, 24, 27)));
  EXPECT_TRUE(is_near(make_native_matrix<double, 3, 3>(
    1, 2, 3,
    4, 5, 6,
    7, 8, 9) * DiagonalMatrix {1., 2, 3}, make_native_matrix<3,3>(
    1., 4, 9,
    4, 10, 18,
    7, 16, 27)));
}

TEST(eigen3, Diagonal_references)
{
  using M3 = eigen_matrix_t<double, 3, 1>;
  DiagonalMatrix<M3> m {1, 2, 3};
  DiagonalMatrix<const M3> n {4, 5, 6};
  DiagonalMatrix<M3> x = m;
  DiagonalMatrix<M3&> x_l {x};
  EXPECT_TRUE(is_near(x_l, m));
  DiagonalMatrix x_l2 {x_l};
  static_assert(std::is_lvalue_reference_v<nested_matrix_t<decltype(x_l2)>>);
  EXPECT_TRUE(is_near(x_l, m));
  DiagonalMatrix<const M3&> x_lc = x_l;
  EXPECT_TRUE(is_near(x_lc, m));
  x = n;
  EXPECT_TRUE(is_near(x_l, n));
  EXPECT_TRUE(is_near(x_l2, n));
  EXPECT_TRUE(is_near(x_lc, n));
  x_l2[0] = 1;
  x_l2[1] = 2;
  x_l2[2] = 3;
  EXPECT_TRUE(is_near(x, m));
  EXPECT_TRUE(is_near(x_l, m));
  EXPECT_TRUE(is_near(x_lc, m));
  EXPECT_TRUE(is_near(DiagonalMatrix<M3&> {m}.nested_matrix(), (M3 {} << 1, 2, 3).finished() ));

  M3 p; p << 10, 11, 12;
  M3 q; q << 13, 14, 15;
  DiagonalMatrix yl {p};
  static_assert(std::is_lvalue_reference_v<nested_matrix_t<decltype(yl)>>);
  EXPECT_TRUE(is_near(diagonal_of(yl), p));
  DiagonalMatrix yr {(M3 {} << 13, 14, 15).finished() * 1.0};
  static_assert(not std::is_reference_v<nested_matrix_t<decltype(yr)>>);
  EXPECT_TRUE(is_near(diagonal_of(yr), q));
  yl = DiagonalMatrix {q};
  EXPECT_TRUE(is_near(p, q));
  p = (M3 {} << 16, 17, 18).finished();
  EXPECT_TRUE(is_near(diagonal_of(yl), p));
}

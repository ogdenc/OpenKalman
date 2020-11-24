/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"

using namespace OpenKalman;

using Mat2 = Eigen::Matrix<double, 2, 2>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Axis2 = Coefficients<Axis, Axis>;

TEST_F(eigen3, Diagonal_class)
{
  DiagonalMatrix<Eigen::Matrix<double, 3, 1>> d1;
  d1 << 1, 2, 3;
  EXPECT_TRUE(is_near(d1.nested_matrix(), make_native_matrix<double, 3, 1>(1, 2, 3)));
  DiagonalMatrix d2 = make_native_matrix(1., 2, 3);
  EXPECT_TRUE(is_near(d2, d1));
  DiagonalMatrix d3(d2);
  EXPECT_TRUE(is_near(d3, d1));
  DiagonalMatrix d4(DiagonalMatrix<Eigen::Matrix<double, 3, 1>>{1., 2, 3});
  EXPECT_TRUE(is_near(d4, d1));
  DiagonalMatrix d5 = MatrixTraits<DiagonalMatrix<Eigen::Matrix<double, 3, 1>>>::zero();
  EXPECT_TRUE(is_near(d5, Eigen::Matrix<double, 3, 3>::Zero()));
  DiagonalMatrix d6 = MatrixTraits<DiagonalMatrix<Eigen::Matrix<double, 3, 1>>>::identity();
  EXPECT_TRUE(is_near(d6, Eigen::Matrix<double, 3, 3>::Identity()));
  DiagonalMatrix d7 = ((0.7 * Mat3::Identity()) * (0.3 * Mat3::Identity() * 0.7 + 0.7 * Mat3::Identity()) - Mat3::Identity() * 0.3);
  EXPECT_TRUE(is_near(d7, Eigen::Matrix<double, 3, 3>::Identity() * 0.337));
  DiagonalMatrix<Eigen::Matrix<double, 3, 1>> d8 = ((0.7 * Mat3::Identity()) * (0.3 * Mat3::Identity() * 0.7 + 0.7 * Mat3::Identity()) - Mat3::Identity() * 0.3);
  EXPECT_TRUE(is_near(d8, Eigen::Matrix<double, 3, 3>::Identity() * 0.337));
  DiagonalMatrix<Eigen::Matrix<double, 3, 1>> d9 = ZeroMatrix<Eigen::Matrix<double, 3, 3>>();
  EXPECT_TRUE(is_near(d9, Eigen::Matrix<double, 3, 3>::Zero()));
  DiagonalMatrix<Eigen::Matrix<double, 3, 1>> d10 = Eigen::Matrix<double, 3, 3>::Identity();
  EXPECT_TRUE(is_near(d10, Eigen::Matrix<double, 3, 3>::Identity()));
  EXPECT_TRUE(is_near(DiagonalMatrix<Eigen::Matrix<double, 3, 1>>(ZeroMatrix<Eigen::Matrix<double, 3, 1>>()), Eigen::Matrix<double, 3, 3>::Zero()));
  d3 = d2;
  EXPECT_TRUE(is_near(d3, d1));
  d4 = DiagonalMatrix {1., 2, 3};
  EXPECT_TRUE(is_near(d4, d1));
  d4 = {1., 2, 3};
  EXPECT_TRUE(is_near(d4, d1));
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
}

TEST_F(eigen3, Diagonal_subscripts)
{
  auto el = DiagonalMatrix {1., 2, 3};
  set_element(el, 5.5, 1);
  EXPECT_NEAR(get_element(el, 1), 5.5, 1e-8);
  set_element(el, 6.5, 2, 2);
  EXPECT_NEAR(get_element(el, 2), 6.5, 1e-8);
  bool test = false; try { set_element(el, 7.5, 2, 0); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_NEAR(get_element(el, 2, 0), 0, 1e-8);

  DiagonalMatrix<Eigen::Matrix<double, 3, 1>> d1 {1, 4, 9};

  static_assert(element_gettable<DiagonalMatrix<Eigen::Matrix<double, 3, 1>>, 2>);
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
}

TEST_F(eigen3, Diagonal_traits)
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
  using D = DiagonalMatrix<Eigen::Matrix<double, 3, 1>>;
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
  EXPECT_TRUE(is_near(MatrixTraits<D>::zero(), Eigen::Matrix<double, 3, 3>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<D>::identity(), Eigen::Matrix<double, 3, 3>::Identity()));
}

TEST_F(eigen3, Diagonal_overloads)
{
  EXPECT_TRUE(is_near(make_native_matrix(DiagonalMatrix(1., 2, 3)), make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3)));
  EXPECT_TRUE(is_near(Cholesky_square(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 4, 9}));
  EXPECT_TRUE(is_near(Cholesky_factor(DiagonalMatrix {1., 4, 9}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(Cholesky_square(Eigen::Matrix<double, 1, 1>(4)), Eigen::Matrix<double, 1, 1>(16)));
  EXPECT_TRUE(is_near(Cholesky_factor(Eigen::Matrix<double, 1, 1>(4)), Eigen::Matrix<double, 1, 1>(2)));
  EXPECT_TRUE(is_near(Cholesky_square(Mat2::Identity() * 0.1), DiagonalMatrix {0.01, 0.01}));
  EXPECT_TRUE(is_near(Cholesky_factor(Mat2::Identity() * 0.01), DiagonalMatrix {0.1, 0.1}));
  EXPECT_TRUE(is_near(Cholesky_square(DiagonalMatrix {9.}), Eigen::Matrix<double, 1, 1>(81)));
  EXPECT_TRUE(is_near(Cholesky_factor(DiagonalMatrix {9.}), Eigen::Matrix<double, 1, 1>(3)));

  EXPECT_TRUE(is_near(transpose(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(adjoint(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 2, 3}));
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
  using M1by1 = Eigen::Matrix<double, 1, 1>;
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
  EXPECT_TRUE(is_near(LQ_decomposition(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(QR_decomposition(DiagonalMatrix {1., 2, 3}), DiagonalMatrix {1., 2, 3}));
  EXPECT_TRUE(is_near(LQ_decomposition(M1by1 {4}), DiagonalMatrix {4.}));
  EXPECT_TRUE(is_near(QR_decomposition(M1by1 {4}), DiagonalMatrix {4.}));

  using N = std::normal_distribution<double>::param_type;
  using Mat = DiagonalMatrix<Eigen::Matrix<double, 2, 1>>;
  Mat m = MatrixTraits<Mat>::zero();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat>(N {1.0, 0.3}, 2.0)) / (i + 1);
  }
  Mat offset = {1, 2};
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));
}

TEST_F(eigen3, Diagonal_blocks)
{
  EXPECT_TRUE(is_near(concatenate_diagonal(DiagonalMatrix {1., 2, 3}, DiagonalMatrix {4., 5}), DiagonalMatrix {1., 2, 3, 4, 5}));
  EXPECT_TRUE(is_near(concatenate_vertical(DiagonalMatrix {1., 2}, DiagonalMatrix {3., 4}),
    make_native_matrix<4,2>(1., 0, 0, 2, 3, 0, 0, 4)));
  EXPECT_TRUE(is_near(concatenate_horizontal(DiagonalMatrix {1., 2}, DiagonalMatrix {3., 4}),
    make_native_matrix<2,4>(1., 0, 3, 0, 0, 2, 0, 4)));
  EXPECT_TRUE(is_near(split_diagonal(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple{}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple{DiagonalMatrix{1., 2, 3}, DiagonalMatrix{4., 5}}));
  const auto a1 = DiagonalMatrix<const Eigen::Matrix<double, 5, 1>>{1., 2, 3, 4, 5};
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(a1), std::tuple{DiagonalMatrix{1., 2, 3}, DiagonalMatrix{4., 5}}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple{DiagonalMatrix{1., 2}, DiagonalMatrix{3., 4}}));
  EXPECT_TRUE(is_near(split_vertical(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(DiagonalMatrix{1., 2, 3, 4, 5}),
    std::tuple{make_native_matrix<double, 3, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0),
               make_native_matrix<double, 2, 5>(
                 0, 0, 0, 4, 0,
                 0, 0, 0, 0, 5)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(DiagonalMatrix{1., 2, 3, 4, 5}),
    std::tuple{make_native_matrix<double, 2, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0),
               make_native_matrix<double, 2, 5>(
                 0, 0, 3, 0, 0,
                 0, 0, 0, 4, 0)}));
  EXPECT_TRUE(is_near(split_horizontal(DiagonalMatrix{1., 2, 3, 4, 5}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(DiagonalMatrix{1., 2, 3, 4, 5}),
    std::tuple{make_native_matrix<double, 5, 3>(
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
    std::tuple{make_native_matrix<double, 5, 2>(
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
  EXPECT_TRUE(is_near(apply_columnwise(DiagonalMatrix{1., 2, 3}, [](const auto& col){ return col + col.Constant(1); }),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_columnwise(DiagonalMatrix{1., 2, 3}, [](const auto& col, std::size_t i){ return col + col.Constant(i); }),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      0, 3, 2,
      0, 1, 5)));
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

TEST_F(eigen3, Diagonal_arithmetic)
{
  auto d1 = DiagonalMatrix {1., 2, 3};
  auto d2 = DiagonalMatrix {4., 5, 6};
  auto i = Mat3::Identity();
  auto z = ZeroMatrix<Mat3> {};
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
  EXPECT_TRUE(is_near(z - i, -i)); static_assert(eigen_diagonal_expr<decltype(z - i)>);
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
  EXPECT_TRUE(is_near(z / 4, z)); static_assert(not zero_matrix<decltype(z / 4)>);
  static_assert(not zero_matrix<decltype(z / 0)>);
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

TEST_F(eigen3, Diagonal_references)
{
  using M3 = Eigen::Matrix<double, 3, 1>;
  const DiagonalMatrix<const M3> m {1, 2, 3};
  const DiagonalMatrix<const M3> n {4, 5, 6};
  DiagonalMatrix<M3> x = m;
  DiagonalMatrix<M3&> x_l = x;
  EXPECT_TRUE(is_near(x_l, m));
  x = n;
  EXPECT_TRUE(is_near(x_l, n));
  x_l = m;
  EXPECT_TRUE(is_near(x, m));
  DiagonalMatrix<M3&&> x_r = std::move(x);
  EXPECT_TRUE(is_near(x_r, m));
  x_r = n;
  EXPECT_TRUE(is_near(x_r, n));
}

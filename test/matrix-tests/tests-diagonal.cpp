/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "matrix_tests.h"

using namespace OpenKalman;

using Mat2 = Eigen::Matrix<double, 2, 2>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Axis2 = Coefficients<Axis, Axis>;

TEST_F(matrix_tests, Diagonal_class)
{
  EigenDiagonal<Eigen::Matrix<double, 3, 1>> d1;
  d1 << 1, 2, 3;
  EXPECT_TRUE(is_near(d1.base_matrix(), (Eigen::Matrix<double, 3, 1>() << 1, 2, 3).finished()));
  EigenDiagonal d2 = (Eigen::Matrix<double, 3, 1>() << 1, 2, 3).finished();
  EXPECT_TRUE(is_near(d2, d1));
  EigenDiagonal d3(d2);
  EXPECT_TRUE(is_near(d3, d1));
  EigenDiagonal d4(EigenDiagonal<Eigen::Matrix<double, 3, 1>>{1., 2, 3});
  EXPECT_TRUE(is_near(d4, d1));
  EigenDiagonal d5 = MatrixTraits<EigenDiagonal<Eigen::Matrix<double, 3, 1>>>::zero();
  EXPECT_TRUE(is_near(d5, Eigen::Matrix<double, 3, 3>::Zero()));
  EigenDiagonal d6 = MatrixTraits<EigenDiagonal<Eigen::Matrix<double, 3, 1>>>::identity();
  EXPECT_TRUE(is_near(d6, Eigen::Matrix<double, 3, 3>::Identity()));
  EigenDiagonal d7 = ((0.7 * Mat3::Identity()) * (0.3 * Mat3::Identity() * 0.7 + 0.7 * Mat3::Identity()) - Mat3::Identity() * 0.3);
  EXPECT_TRUE(is_near(d7, Eigen::Matrix<double, 3, 3>::Identity() * 0.337));
  EigenDiagonal<Eigen::Matrix<double, 3, 1>> d8 = ((0.7 * Mat3::Identity()) * (0.3 * Mat3::Identity() * 0.7 + 0.7 * Mat3::Identity()) - Mat3::Identity() * 0.3);
  EXPECT_TRUE(is_near(d8, Eigen::Matrix<double, 3, 3>::Identity() * 0.337));
  EigenDiagonal<Eigen::Matrix<double, 3, 1>> d9 = EigenZero<Eigen::Matrix<double, 3, 3>>();
  EXPECT_TRUE(is_near(d9, Eigen::Matrix<double, 3, 3>::Zero()));
  EigenDiagonal<Eigen::Matrix<double, 3, 1>> d10 = Eigen::Matrix<double, 3, 3>::Identity();
  EXPECT_TRUE(is_near(d10, Eigen::Matrix<double, 3, 3>::Identity()));
  EXPECT_TRUE(is_near(EigenDiagonal<Eigen::Matrix<double, 3, 1>>(EigenZero<Eigen::Matrix<double, 3, 1>>()), Eigen::Matrix<double, 3, 3>::Zero()));
  d3 = d2;
  EXPECT_TRUE(is_near(d3, d1));
  d4 = EigenDiagonal {1., 2, 3};
  EXPECT_TRUE(is_near(d4, d1));
  d4 = {1., 2, 3};
  EXPECT_TRUE(is_near(d4, d1));
  d1 += d2;
  EXPECT_TRUE(is_near(d1, EigenDiagonal {2., 4, 6}));
  d1 -= EigenDiagonal {1., 2, 3};
  EXPECT_TRUE(is_near(d1, EigenDiagonal {1., 2, 3}));
  d1 *= 3;
  EXPECT_TRUE(is_near(d1, EigenDiagonal {3., 6, 9}));
  d1 /= 3;
  EXPECT_TRUE(is_near(d1, EigenDiagonal {1., 2, 3}));
  d1 *= d2;
  EXPECT_TRUE(is_near(d1, EigenDiagonal {1., 4, 9}));
  EXPECT_TRUE(is_near(d1.square_root(), EigenDiagonal {1., 2, 3}));
  EXPECT_TRUE(is_near(d1.square(), EigenDiagonal {1., 16, 81}));
  EXPECT_EQ(d1(2), 9);
  EXPECT_EQ(d1(0), 1);
  EXPECT_EQ(d1(0, 1), 0);
  EXPECT_EQ(d1(1, 1), 4);
  d1(0,0) = 5;
  d1(1) = 6;
  d1(2) = 7;
  d1(1, 0) = 3; // Should have no effect.
  EXPECT_EQ(d1(1, 0), 0);
  EXPECT_TRUE(is_near(d1, EigenDiagonal {5., 6, 7}));
  EXPECT_NEAR(d1(0), 5, 1e-6);
  EXPECT_NEAR(d1(1), 6, 1e-6);
  EXPECT_NEAR(d1(2), 7, 1e-6);
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

TEST_F(matrix_tests, Diagonal_traits)
{
  static_assert(is_diagonal_v<decltype(EigenDiagonal{2, 3})>);
  static_assert(is_self_adjoint_v<decltype(EigenDiagonal{2, 3})>);
  static_assert(is_triangular_v<decltype(EigenDiagonal{2, 3})>);
  static_assert(is_lower_triangular_v<decltype(EigenDiagonal{2, 3})>);
  static_assert(is_upper_triangular_v<decltype(EigenDiagonal{2, 3})>);
  static_assert(not is_identity_v<decltype(EigenDiagonal{2, 3})>);
  static_assert(not is_zero_v<decltype(EigenDiagonal{2, 3})>);
  static_assert(is_covariance_base_v<decltype(EigenDiagonal{2, 3})>);
  static_assert(is_covariance_base_v<decltype(Mat3::Identity())>);
  static_assert(is_identity_v<decltype(Mat3::Identity() * Mat3::Identity())>);
  static_assert(is_diagonal_v<decltype(0.3 * Mat3::Identity() + 0.7 * Mat3::Identity() * 0.7)>);
  static_assert(is_diagonal_v<decltype(0.7 * Mat3::Identity() * 0.7 - Mat3::Identity() * 0.3)>);
  static_assert(is_diagonal_v<decltype((0.7 * Mat3::Identity()) * (Mat3::Identity() + Mat3::Identity()))>);
  // MatrixTraits
  using D = EigenDiagonal<Eigen::Matrix<double, 3, 1>>;
  EXPECT_TRUE(is_near(MatrixTraits<D>::make((Eigen::Matrix<double, 3, 1>() << 1, 2, 3).finished()),
    (Eigen::Matrix<double, 3, 3>() <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3).finished()));
  EXPECT_TRUE(is_near(MatrixTraits<D>::make(1, 2, 3), (Eigen::Matrix<double, 3, 3>() <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3).finished()));
  EXPECT_TRUE(is_near(MatrixTraits<D>::make(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3), (Eigen::Matrix<double, 3, 3>() <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3).finished()));
  EXPECT_TRUE(is_near(MatrixTraits<D>::zero(), Eigen::Matrix<double, 3, 3>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<D>::identity(), Eigen::Matrix<double, 3, 3>::Identity()));
}

TEST_F(matrix_tests, Diagonal_overloads)
{
  EXPECT_TRUE(is_near(strict_matrix(EigenDiagonal(1., 2, 3)), (Eigen::Matrix<double, 3, 3>() <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3).finished()));
  EXPECT_TRUE(is_near(Cholesky_square(EigenDiagonal {1., 2, 3}), EigenDiagonal {1., 4, 9}));
  EXPECT_TRUE(is_near(Cholesky_factor(EigenDiagonal {1., 4, 9}), EigenDiagonal {1., 2, 3}));
  EXPECT_TRUE(is_near(Cholesky_square(Eigen::Matrix<double, 1, 1>(4)), Eigen::Matrix<double, 1, 1>(16)));
  EXPECT_TRUE(is_near(Cholesky_factor(Eigen::Matrix<double, 1, 1>(4)), Eigen::Matrix<double, 1, 1>(2)));
  EXPECT_TRUE(is_near(transpose(EigenDiagonal {1., 2, 3}), EigenDiagonal {1., 2, 3}));
  EXPECT_TRUE(is_near(adjoint(EigenDiagonal {1., 2, 3}), EigenDiagonal {1., 2, 3}));
  EXPECT_NEAR(determinant(EigenDiagonal {2., 3, 4}), 24, 1e-6);
  EXPECT_NEAR(trace(EigenDiagonal {2., 3, 4}), 9, 1e-6);
  EXPECT_TRUE(is_near(solve(EigenDiagonal {1., 2, 3}, (Eigen::Matrix<double, 3, 1>() << 4., 10, 18).finished()),
    Mean {4., 5, 6}));
  EXPECT_TRUE(is_near(reduce_columns(EigenDiagonal {1., 2, 3}), Mean {1., 2, 3}));
  EXPECT_TRUE(is_near(LQ_decomposition(EigenDiagonal {1., 2, 3}), EigenDiagonal {1., 2, 3}));
  EXPECT_TRUE(is_near(QR_decomposition(EigenDiagonal {1., 2, 3}), EigenDiagonal {1., 2, 3}));

  using Mat = EigenDiagonal<Eigen::Matrix<double, 2, 1>>;
  Mat m = MatrixTraits<Mat>::zero();
  Mat offset = {1, 1};
  for (int i=0; i<100; i++)
  {
    m = (m * i + offset + randomize<Mat>(0.7)) / (i + 1);
  }
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));
}

TEST_F(matrix_tests, Diagonal_blocks)
{
  EXPECT_TRUE(is_near(concatenate(EigenDiagonal {1., 2, 3}, EigenDiagonal {4., 5}), EigenDiagonal {1., 2, 3, 4, 5}));
  EXPECT_TRUE(is_near(concatenate_vertical(EigenDiagonal {1., 2}, EigenDiagonal {3., 4}),
    TypedMatrix<Axes<4>, Axes<2>> {1., 0, 0, 2, 3, 0, 0, 4}));
  EXPECT_TRUE(is_near(concatenate_horizontal(EigenDiagonal {1., 2}, EigenDiagonal {3., 4}),
    TypedMatrix<Axes<2>, Axes<4>> {1., 0, 3, 0, 0, 2, 0, 4}));
  EXPECT_TRUE(is_near(split(EigenDiagonal{1., 2, 3, 4, 5}), std::tuple{}));
  EXPECT_TRUE(is_near(split<3, 2>(EigenDiagonal{1., 2, 3, 4, 5}), std::tuple{EigenDiagonal{1., 2, 3}, EigenDiagonal{4., 5}}));
  EXPECT_TRUE(is_near(split_diagonal(EigenDiagonal{1., 2, 3, 4, 5}), std::tuple{}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(EigenDiagonal{1., 2, 3, 4, 5}), std::tuple{EigenDiagonal{1., 2, 3}, EigenDiagonal{4., 5}}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(EigenDiagonal{1., 2, 3, 4, 5}), std::tuple{EigenDiagonal{1., 2}, EigenDiagonal{3., 4}}));
  EXPECT_TRUE(is_near(split_vertical(EigenDiagonal{1., 2, 3, 4, 5}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(EigenDiagonal{1., 2, 3, 4, 5}),
    std::tuple{(Eigen::Matrix<double, 3, 5>() <<
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0).finished(),
               (Eigen::Matrix<double, 2, 5>() <<
                 0, 0, 0, 4, 0,
                 0, 0, 0, 0, 5).finished()}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(EigenDiagonal{1., 2, 3, 4, 5}),
    std::tuple{(Eigen::Matrix<double, 2, 5>() <<
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0).finished(),
               (Eigen::Matrix<double, 2, 5>() <<
                 0, 0, 3, 0, 0,
                 0, 0, 0, 4, 0).finished()}));
  EXPECT_TRUE(is_near(split_horizontal(EigenDiagonal{1., 2, 3, 4, 5}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(EigenDiagonal{1., 2, 3, 4, 5}),
    std::tuple{(Eigen::Matrix<double, 5, 3>() <<
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      0, 0, 0,
      0, 0, 0).finished(),
               (Eigen::Matrix<double, 5, 2>() <<
                 0, 0,
                 0, 0,
                 0, 0,
                 4, 0,
                 0, 5).finished()}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(EigenDiagonal{1., 2, 3, 4, 5}),
    std::tuple{(Eigen::Matrix<double, 5, 2>() <<
      1, 0,
      0, 2,
      0, 0,
      0, 0,
      0, 0).finished(),
               (Eigen::Matrix<double, 5, 2>() <<
                 0, 0,
                 0, 0,
                 3, 0,
                 0, 4,
                 0, 0).finished()}));
  EXPECT_TRUE(is_near(column(EigenDiagonal{1., 2, 3}, 2), Mean{0., 0, 3}));
  EXPECT_TRUE(is_near(column<1>(EigenDiagonal{1., 2, 3}), Mean{0., 2, 0}));
  EXPECT_TRUE(is_near(apply_columnwise(EigenDiagonal{1., 2, 3}, [](const auto& col){ return col + col.Constant(1); }),
    (Eigen::Matrix<double, 3, 3>() <<
      2, 1, 1,
      1, 3, 1,
      1, 1, 4).finished()));
  EXPECT_TRUE(is_near(apply_columnwise(EigenDiagonal{1., 2, 3}, [](const auto& col, std::size_t i){ return col + col.Constant(i); }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 1, 2,
      0, 3, 2,
      0, 1, 5).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(EigenDiagonal{1., 2, 3}, [](const auto& x){ return x + 1; }),
    (Eigen::Matrix<double, 3, 3>() <<
      2, 1, 1,
      1, 3, 1,
      1, 1, 4).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(EigenDiagonal{1., 2, 3}, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 1, 2,
      1, 4, 3,
      2, 3, 7).finished()));
}

TEST_F(matrix_tests, Diagonal_arithmetic)
{
  auto d1 = EigenDiagonal {1., 2, 3};
  auto d2 = EigenDiagonal {4., 5, 6};
  auto i = Mat3::Identity();
  auto z = EigenZero<Mat3> {};
  EXPECT_TRUE(is_near(d1 + d2, EigenDiagonal {5., 7, 9})); static_assert(is_EigenDiagonal_v<decltype(d1 + d2)>);
  EXPECT_TRUE(is_near(d1 + i, EigenDiagonal {2., 3, 4})); static_assert(is_EigenDiagonal_v<decltype(d1 + i)>);
  EXPECT_TRUE(is_near(d1 + z, d1)); static_assert(is_EigenDiagonal_v<decltype(d1 + z)>);
  EXPECT_TRUE(is_near(i + i, EigenDiagonal {2., 2, 2})); static_assert(is_diagonal_v<decltype(i + i)>);
  EXPECT_TRUE(is_near(i + z, i)); static_assert(is_identity_v<decltype(i + z)>);
  EXPECT_TRUE(is_near(z + i, i)); static_assert(is_identity_v<decltype(z + i)>);
  EXPECT_TRUE(is_near(z + z, z)); static_assert(is_EigenZero_v<decltype(z + z)>);

  EXPECT_TRUE(is_near(d1 - d2, EigenDiagonal {-3., -3, -3})); static_assert(is_diagonal_v<decltype(d1 - d2)>);
  EXPECT_TRUE(is_near(d1 - i, EigenDiagonal {0., 1, 2})); static_assert(is_diagonal_v<decltype(d1 - i)>);
  EXPECT_TRUE(is_near(d1 - z, d1)); static_assert(is_diagonal_v<decltype(d1 - z)>);
  EXPECT_TRUE(is_near(i - i, z)); static_assert(is_zero_v<decltype(i - i)>);
  EXPECT_TRUE(is_near(i * (i - i), z)); static_assert(is_zero_v<decltype(i - i)>);
  EXPECT_TRUE(is_near(z * (i - i), z)); static_assert(is_EigenZero_v<decltype(z * (i - i))>);
  EXPECT_TRUE(is_near(i - z, i)); static_assert(is_identity_v<decltype(i - z)>);
  EXPECT_TRUE(is_near(z - i, -i)); static_assert(is_EigenDiagonal_v<decltype(z - i)>);
  EXPECT_TRUE(is_near(z - z, z)); static_assert(is_EigenZero_v<decltype(z - z)>);

  EXPECT_TRUE(is_near(d1 * 4, EigenDiagonal {4., 8, 12})); static_assert(is_EigenDiagonal_v<decltype(d1 * 4)>);
  EXPECT_TRUE(is_near(4 * d1, EigenDiagonal {4., 8, 12})); static_assert(is_EigenDiagonal_v<decltype(4 * d1)>);
  EXPECT_TRUE(is_near(d1 / 2, EigenDiagonal {0.5, 1, 1.5})); static_assert(is_diagonal_v<decltype(d1 / 2)>);
  static_assert(is_diagonal_v<decltype(d1 / 0)>);
  EXPECT_TRUE(is_near(i * 4, EigenDiagonal {4., 4, 4})); static_assert(is_diagonal_v<decltype(i * 4)>);
  EXPECT_TRUE(is_near(4 * i, EigenDiagonal {4., 4, 4})); static_assert(is_diagonal_v<decltype(4 * i)>);
  static_assert(is_diagonal_v<decltype(i * 0)>);
  static_assert(not is_zero_v<decltype(i * 0)>);
  EXPECT_TRUE(is_near(i / 2, EigenDiagonal {0.5, 0.5, 0.5})); static_assert(is_diagonal_v<decltype(i / 2)>);
  EXPECT_TRUE(is_near(z * 4, z)); static_assert(is_zero_v<decltype(z * 4)>);
  EXPECT_TRUE(is_near(4 * z, z)); static_assert(is_zero_v<decltype(4 * z)>);
  EXPECT_TRUE(is_near(z / 4, z)); static_assert(not is_zero_v<decltype(z / 4)>);
  static_assert(not is_zero_v<decltype(z / 0)>);
  EXPECT_TRUE(is_near((i - i) * 4, z)); static_assert(is_zero_v<decltype((i - i) * 4)>);
  EXPECT_TRUE(is_near((i - i) / 4, z)); static_assert(not is_zero_v<decltype((i - i) / 4)>);
  static_assert(is_diagonal_v<decltype(i / 0)>);

  EXPECT_TRUE(is_near(-d1, EigenDiagonal {-1., -2, -3})); static_assert(is_EigenDiagonal_v<decltype(-d1)>);
  EXPECT_TRUE(is_near(-i, EigenDiagonal {-1., -1, -1})); static_assert(is_diagonal_v<decltype(-i)>);
  EXPECT_TRUE(is_near(-z, z)); static_assert(is_EigenZero_v<decltype(z)>);

  EXPECT_TRUE(is_near(d1 * d2, EigenDiagonal {4., 10, 18})); static_assert(is_EigenDiagonal_v<decltype(d1 * d2)>);
  EXPECT_TRUE(is_near(d1 * i, d1)); static_assert(is_EigenDiagonal_v<decltype(d1 * i)>);
  EXPECT_TRUE(is_near(i * d1, d1)); static_assert(is_EigenDiagonal_v<decltype(i * d1)>);
  EXPECT_TRUE(is_near(d1 * z, z)); static_assert(is_EigenZero_v<decltype(d1 * z)>);
  EXPECT_TRUE(is_near(z * d1, z)); static_assert(is_EigenZero_v<decltype(z * d1)>);
  EXPECT_TRUE(is_near(i * i, i)); static_assert(is_identity_v<decltype(i * i)>);
  EXPECT_TRUE(is_near(z * z, z)); static_assert(is_EigenZero_v<decltype(z * z)>);

  EXPECT_TRUE(is_near(d1 * EigenSelfAdjointMatrix {
    1., 2, 3,
    2, 4, 5,
    3, 5, 6}, make_Matrix<Axes<3>,Axes<3>>(
      1., 2, 3,
      4, 8, 10,
      9, 15, 18)));
  EXPECT_TRUE(is_near(EigenSelfAdjointMatrix {
    1., 2, 3,
    2, 4, 5,
    3, 5, 6} * EigenDiagonal {1., 2, 3}, make_Matrix<Axes<3>,Axes<3>>(
    1., 4, 9,
    2, 8, 15,
    3, 10, 18)));
  EXPECT_TRUE(is_near(EigenDiagonal {1., 2, 3} * EigenTriangularMatrix {
    1., 0, 0,
    2, 3, 0,
    4, 5, 6}, make_Matrix<Axes<3>,Axes<3>>(
    1., 0, 0,
    4, 6, 0,
    12, 15, 18)));
  EXPECT_TRUE(is_near(EigenTriangularMatrix {
    1., 0, 0,
    2, 3, 0,
    4, 5, 6} * EigenDiagonal {1., 2, 3}, make_Matrix<Axes<3>,Axes<3>>(
    1., 0, 0,
    2, 6, 0,
    4, 10, 18)));
  EXPECT_TRUE(is_near(EigenDiagonal {1., 2, 3} * (Eigen::Matrix<double, 3, 3>() <<
    1, 2, 3,
    4, 5, 6,
    7, 8, 9).finished(), make_Matrix<Axes<3>,Axes<3>>(
    1., 2, 3,
    8, 10, 12,
    21, 24, 27)));
  EXPECT_TRUE(is_near((Eigen::Matrix<double, 3, 3>() <<
    1, 2, 3,
    4, 5, 6,
    7, 8, 9).finished() * EigenDiagonal {1., 2, 3}, make_Matrix<Axes<3>,Axes<3>>(
    1., 4, 9,
    4, 10, 18,
    7, 16, 27)));
}

TEST_F(matrix_tests, Diagonal_references)
{
  using M3 = Eigen::Matrix<double, 3, 1>;
  const EigenDiagonal<const M3> m {1, 2, 3};
  const EigenDiagonal<const M3> n {4, 5, 6};
  EigenDiagonal<M3> x = m;
  EigenDiagonal<M3&> x_l = x;
  EXPECT_TRUE(is_near(x_l, m));
  x = n;
  EXPECT_TRUE(is_near(x_l, n));
  x_l = m;
  EXPECT_TRUE(is_near(x, m));
  EigenDiagonal<M3&&> x_r = std::move(x);
  EXPECT_TRUE(is_near(x_r, m));
  x_r = n;
  EXPECT_TRUE(is_near(x_r, n));
}

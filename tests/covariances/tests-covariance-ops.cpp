/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "covariance_tests.hpp"

using namespace OpenKalman;

using M2 = native_matrix_t<double, 2, 2>;
using C = Coefficients<angle::Radians, Axis>;
using Mat2 = Matrix<C, C, M2>;
using Mat2col = Matrix<C, Axis, native_matrix_t<double, 2, 1>>;
using SA2l = SelfAdjointMatrix<M2, TriangleType::lower>;
using SA2u = SelfAdjointMatrix<M2, TriangleType::upper>;
using T2l = TriangularMatrix<M2, TriangleType::lower>;
using T2u = TriangularMatrix<M2, TriangleType::upper>;
using D2 = DiagonalMatrix<native_matrix_t<double, 2, 1>>;
using I2 = IdentityMatrix<M2>;
using Z2 = ZeroMatrix<double, 2, 2>;
using CovSA2l = Covariance<C, SA2l>;
using CovSA2u = Covariance<C, SA2u>;
using CovT2l = Covariance<C, T2l>;
using CovT2u = Covariance<C, T2u>;
using CovD2 = Covariance<C, D2>;
using CovI2 = Covariance<C, I2>;
using CovZ2 = Covariance<C, Z2>;
using SqCovSA2l = SquareRootCovariance<C, SA2l>;
using SqCovSA2u = SquareRootCovariance<C, SA2u>;
using SqCovT2l = SquareRootCovariance<C, T2l>;
using SqCovT2u = SquareRootCovariance<C, T2u>;
using SqCovD2 = SquareRootCovariance<C, D2>;
using SqCovI2 = SquareRootCovariance<C, I2>;
using SqCovZ2 = SquareRootCovariance<C, Z2>;

inline I2 i2 = M2::Identity();
inline Z2 z2 = ZeroMatrix<double, 2, 2>();
inline auto covi2 = CovI2 {i2};
inline auto covz2 = CovZ2 {z2};
inline auto sqcovi2 = SqCovI2 {i2};
inline auto sqcovz2 = SqCovZ2 {z2};


TEST_F(covariance_tests, Covariance_addition)
{
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + covi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + covz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + CovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + covi2).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + covi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + covz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + CovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + covi2).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + covi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + covz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} + CovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} + covi2).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} + covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + covi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + covz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} + CovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} + covi2).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} + covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 2, 2, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 2, 2, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 2, 2, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 2, 2, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovD2 {4, 5}, Mat2 {13, 0, 0, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + covi2, Mat2 {10, 0, 0, 11}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + covz2, Mat2 {9, 0, 0, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovD2 {9, 10} + CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovD2 {9, 10} + CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovD2 {9, 10} + CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovD2 {9, 10} + CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} + CovD2 {4, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} + covi2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} + covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covi2 + CovSA2l {4, 2, 2, 5}, Mat2 {5, 2, 2, 6}));
  EXPECT_TRUE(is_near(covi2 + CovSA2u {4, 2, 2, 5}, Mat2 {5, 2, 2, 6}));
  EXPECT_TRUE(is_near(covi2 + CovT2l {4, 2, 2, 5}, Mat2 {5, 2, 2, 6}));
  EXPECT_TRUE(is_near(covi2 + CovT2u {4, 2, 2, 5}, Mat2 {5, 2, 2, 6}));
  EXPECT_TRUE(is_near(covi2 + CovD2 {4, 5}, Mat2 {5, 0, 0, 6}));
  EXPECT_TRUE(is_near(covi2 + covi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(covi2 + covz2, Mat2 {1, 0, 0, 1}));
  static_assert(Eigen3::lower_triangular_storage<decltype((covi2 + CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covi2 + CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((covi2 + CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covi2 + CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 + CovD2 {4, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 + covi2).nested_matrix())>);
  static_assert(identity_matrix<decltype((covi2 + covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covz2 + CovSA2l {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covz2 + CovSA2u {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covz2 + CovT2l {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covz2 + CovT2u {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covz2 + CovD2 {4, 5}, Mat2 {4, 0, 0, 5}));
  EXPECT_TRUE(is_near(covz2 + covi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covz2 + covz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((covz2 + CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covz2 + CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((covz2 + CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((covz2 + CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covz2 + CovD2 {4, 5}).nested_matrix())>);
  static_assert(identity_matrix<decltype((covz2 + covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 + covz2).nested_matrix())>);
}


TEST_F(covariance_tests, Covariance_addition_mixed)
{
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + SqCovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + sqcovi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + sqcovi2).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} + sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + sqcovi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + sqcovi2).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} + sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + sqcovi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} + SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} + sqcovi2).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} + sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + sqcovi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} + SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} + sqcovi2).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} + sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 1, 0, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 1, 0, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovD2 {4, 5}, Mat2 {13, 0, 0, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + sqcovi2, Mat2 {10, 0, 0, 11}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + sqcovz2, Mat2 {9, 0, 0, 10}));
  static_assert(diagonal_matrix<decltype((CovD2 {9, 3, 3, 10} + SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 3, 3, 10} + sqcovi2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 3, 3, 10} + sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covi2 + SqCovSA2l {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covi2 + SqCovSA2u {2, 1, 0, 2}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(covi2 + SqCovT2l {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covi2 + SqCovT2u {2, 1, 0, 2}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(covi2 + SqCovD2 {4, 5}, Mat2 {5, 0, 0, 6}));
  EXPECT_TRUE(is_near(covi2 + sqcovi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(covi2 + sqcovz2, Mat2 {1, 0, 0, 1}));
  static_assert(diagonal_matrix<decltype((covi2 + SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 + sqcovi2).nested_matrix())>);
  static_assert(identity_matrix<decltype((covi2 + sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covz2 + SqCovSA2l {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(covz2 + SqCovSA2u {2, 1, 0, 2}, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(covz2 + SqCovT2l {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(covz2 + SqCovT2u {2, 1, 0, 2}, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(covz2 + SqCovD2 {4, 5}, Mat2 {4, 0, 0, 5}));
  EXPECT_TRUE(is_near(covz2 + sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covz2 + sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((covz2 + SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(identity_matrix<decltype((covz2 + sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 + sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + Mat2 {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + Mat2 {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + Mat2 {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + Mat2 {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + Mat2 {2, 0, 1, 2}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(covi2 + Mat2 {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covz2 + Mat2 {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));

  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} + CovD2 {9, 10}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} + covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} + covz2, Mat2 {2, 0, 1, 2}));
}


TEST_F(covariance_tests, Covariance_subtraction)
{
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - CovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - covi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - covz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - CovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - covi2).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - covi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - covz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - CovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - covi2).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - covi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - covz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} - CovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} - covi2).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} - covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - covi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - covz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} - CovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} - covi2).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} - covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, -2, -2, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, -2, -2, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, -2, -2, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, -2, -2, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovD2 {4, 5}, Mat2 {5, 0, 0, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - covi2, Mat2 {8, 0, 0, 9}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - covz2, Mat2 {9, 0, 0, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovD2 {9, 10} - CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovD2 {9, 10} - CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovD2 {9, 10} - CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovD2 {9, 10} - CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} - CovD2 {4, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} - covi2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} - covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covi2 - CovSA2l {4, 2, 2, 5}, Mat2 {-3, -2, -2, -4}));
  EXPECT_TRUE(is_near(covi2 - CovSA2u {4, 2, 2, 5}, Mat2 {-3, -2, -2, -4}));
  EXPECT_TRUE(is_near(covi2 - CovT2l {4, 2, 2, 5}, Mat2 {-3, -2, -2, -4}));
  EXPECT_TRUE(is_near(covi2 - CovT2u {4, 2, 2, 5}, Mat2 {-3, -2, -2, -4}));
  EXPECT_TRUE(is_near(covi2 - CovD2 {4, 5}, Mat2 {-3, 0, 0, -4}));
  EXPECT_TRUE(is_near(covi2 - covi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covi2 - covz2, Mat2 {1, 0, 0, 1}));
  static_assert(Eigen3::lower_triangular_storage<decltype((covi2 - CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covi2 - CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((covi2 - CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covi2 - CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 - CovD2 {4, 5}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 - covz2).nested_matrix())>);
  static_assert(identity_matrix<decltype((covi2 - covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covz2 - CovSA2l {4, 2, 2, 5}, Mat2 {-4, -2, -2, -5}));
  EXPECT_TRUE(is_near(covz2 - CovSA2u {4, 2, 2, 5}, Mat2 {-4, -2, -2, -5}));
  EXPECT_TRUE(is_near(covz2 - CovT2l {4, 2, 2, 5}, Mat2 {-4, -2, -2, -5}));
  EXPECT_TRUE(is_near(covz2 - CovT2u {4, 2, 2, 5}, Mat2 {-4, -2, -2, -5}));
  EXPECT_TRUE(is_near(covz2 - CovD2 {4, 5}, Mat2 {-4, 0, 0, -5}));
  EXPECT_TRUE(is_near(covz2 - covi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(covz2 - covz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((covz2 - CovSA2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covz2 - CovSA2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((covz2 - CovT2l {4, 2, 2, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covz2 - CovT2u {4, 2, 2, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covz2 - CovD2 {4, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covz2 - covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 - covz2).nested_matrix())>);
}


TEST_F(covariance_tests, Covariance_subtraction_mixed)
{
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - SqCovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - sqcovi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - sqcovi2).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} - sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - sqcovi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - sqcovi2).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} - sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - sqcovi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} - SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype((CovT2l {9, 3, 3, 10} - sqcovi2).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} - sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - sqcovi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} - SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovT2u {9, 3, 3, 10} - sqcovi2).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} - sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 0, -1, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, -1, 0, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 0, -1, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, -1, 0, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovD2 {4, 5}, Mat2 {5, 0, 0, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - sqcovi2, Mat2 {8, 0, 0, 9}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - sqcovz2, Mat2 {9, 0, 0, 10}));
  static_assert(diagonal_matrix<decltype((CovD2 {9, 3, 3, 10} - SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 3, 3, 10} - sqcovi2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 3, 3, 10} - sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covi2 - SqCovSA2l {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(covi2 - SqCovSA2u {2, 1, 0, 2}, Mat2 {-1, -1, 0, -1}));
  EXPECT_TRUE(is_near(covi2 - SqCovT2l {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(covi2 - SqCovT2u {2, 1, 0, 2}, Mat2 {-1, -1, 0, -1}));
  EXPECT_TRUE(is_near(covi2 - SqCovD2 {4, 5}, Mat2 {-3, 0, 0, -4}));
  EXPECT_TRUE(is_near(covi2 - sqcovi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covi2 - sqcovz2, Mat2 {1, 0, 0, 1}));
  static_assert(diagonal_matrix<decltype((covi2 - SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covi2 - sqcovi2).nested_matrix())>);
  static_assert(identity_matrix<decltype((covi2 - sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covz2 - SqCovSA2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(covz2 - SqCovSA2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(covz2 - SqCovT2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(covz2 - SqCovT2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(covz2 - SqCovD2 {4, 5}, Mat2 {-4, 0, 0, -5}));
  EXPECT_TRUE(is_near(covz2 - sqcovi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(covz2 - sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((covz2 - SqCovD2 {4, 5}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covz2 - sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 - sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} - Mat2 {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - Mat2 {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - Mat2 {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - Mat2 {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - Mat2 {2, 0, 1, 2}, Mat2 {7, 0, -1, 8}));
  EXPECT_TRUE(is_near(covi2 - Mat2 {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(covz2 - Mat2 {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));

  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} - CovSA2l {9, 3, 3, 10}, Mat2 {-7, -3, -2, -8}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} - CovSA2u {9, 3, 3, 10}, Mat2 {-7, -3, -2, -8}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} - CovT2l {9, 3, 3, 10}, Mat2 {-7, -3, -2, -8}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} - CovT2u {9, 3, 3, 10}, Mat2 {-7, -3, -2, -8}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} - CovD2 {9, 10}, Mat2 {-7, 0, 1, -8}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} - covi2, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(Mat2 {2, 0, 1, 2} - covz2, Mat2 {2, 0, 1, 2}));
}


TEST_F(covariance_tests, Covariance_mult_covariance)
{
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * CovT2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * CovT2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * CovD2 {9, 10}, Mat2 {81, 30, 27, 100}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * covi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovSA2l {9, 3, 3, 10} * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovD2 {3, 3}, Mat2 {27, 9, 9, 30}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * sqcovi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l {9, 3, 3, 10} * sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovSA2l {9, 3, 3, 10} * sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovT2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovT2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovD2 {9, 10}, Mat2 {81, 30, 27, 100}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * covi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovSA2u {9, 3, 3, 10} * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovD2 {3, 3}, Mat2 {27, 9, 9, 30}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * sqcovi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u {9, 3, 3, 10} * sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovSA2u {9, 3, 3, 10} * sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovT2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovT2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovD2 {9, 10}, Mat2 {81, 30, 27, 100}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * covi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovT2l {9, 3, 3, 10} * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovD2 {3, 3}, Mat2 {27, 9, 9, 30}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * sqcovi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(lower_triangular_matrix<decltype((CovT2l {9, 3, 3, 10} * sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovT2l {9, 3, 3, 10} * sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovT2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovT2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovD2 {9, 10}, Mat2 {81, 30, 27, 100}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * covi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovT2u {9, 3, 3, 10} * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovD2 {3, 3}, Mat2 {27, 9, 9, 30}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * sqcovi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(upper_triangular_matrix<decltype((CovT2u {9, 3, 3, 10} * sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovT2u {9, 3, 3, 10} * sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovT2l {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovT2u {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovD2 {9, 10}, Mat2 {81, 0, 0, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * covi2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} * CovD2 {9, 10}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovD2 {9, 10} * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {27, 0, 10, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 9, 0, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {27, 0, 10, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 9, 0, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovD2 {3, 3}, Mat2 {27, 0, 0, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * sqcovi2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} * SqCovD2 {9, 10}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {9, 10} * sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((CovD2 {9, 10} * sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covi2 * CovSA2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covi2 * CovSA2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covi2 * CovT2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covi2 * CovT2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covi2 * CovD2 {9, 10}, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(covi2 * covi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covi2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((covi2 * CovSA2l {9, 3, 3, 10}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covi2 * CovSA2u {9, 3, 3, 10}).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((covi2 * CovT2l {9, 3, 3, 10}).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((covi2 * CovT2u {9, 3, 3, 10}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 * CovD2 {9, 10}).nested_matrix())>);
  static_assert(identity_matrix<decltype((covi2 * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covi2 * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covi2 * SqCovSA2l {3, 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covi2 * SqCovSA2u {3, 1, 0, 3}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(covi2 * SqCovT2l {3, 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covi2 * SqCovT2u {3, 1, 0, 3}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(covi2 * SqCovD2 {3, 3}, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(covi2 * sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covi2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((covi2 * SqCovSA2l {9, 3, 3, 10}).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((covi2 * SqCovSA2u {9, 3, 3, 10}).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((covi2 * SqCovT2l {9, 3, 3, 10}).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((covi2 * SqCovT2u {9, 3, 3, 10}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 * SqCovD2 {9, 10}).nested_matrix())>);
  static_assert(identity_matrix<decltype((covi2 * sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covi2 * sqcovz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covz2 * CovSA2l {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * CovSA2u {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * CovT2l {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * CovT2u {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * CovD2 {9, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * covi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero_matrix<decltype((covz2 * CovSA2l {9, 3, 3, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * CovSA2u {9, 3, 3, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * CovT2l {9, 3, 3, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * CovT2u {9, 3, 3, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * CovD2 {9, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(covz2 * SqCovSA2l {3, 0, 1, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * SqCovSA2u {3, 1, 0, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * SqCovT2l {3, 0, 1, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * SqCovT2u {3, 1, 0, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * SqCovD2 {3, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * sqcovi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero_matrix<decltype((covz2 * SqCovSA2l {9, 3, 3, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * SqCovSA2u {9, 3, 3, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * SqCovT2l {9, 3, 3, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * SqCovT2u {9, 3, 3, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * SqCovD2 {9, 10}).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * sqcovi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * sqcovz2).nested_matrix())>);
}


TEST_F(covariance_tests, Covariance_mult_TypedMatrix)
{
  using MatI2 = Matrix<C, C, I2>;
  using MatZ2 = Matrix<C, C, Z2>;
  auto mati2 = MatI2(i2);
  auto matz2 = MatZ2(z2);
  using Cx = Coefficients<Axis, angle::Radians>;
  using MatI2x = Matrix<C, Cx, I2>;
  auto mati2x = MatI2x(i2);

  auto covSA2l = CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(covSA2l * Mat2 {4, 2, 2, 5}, Mat2 {42, 33, 32, 56}));
  EXPECT_TRUE(is_near(covSA2l * mati2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covSA2l * mati2x, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covSA2l * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((covSA2l * mati2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covSA2l * matz2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covSA2l * mati2x)>::RowCoefficients, C>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covSA2l * mati2x)>::ColumnCoefficients, Cx>);

  auto covSA2u = CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(covSA2u * Mat2 {4, 2, 2, 5}, Mat2 {42, 33, 32, 56}));
  EXPECT_TRUE(is_near(covSA2u * mati2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covSA2u * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::upper_triangular_storage<decltype((covSA2u * mati2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covSA2u * matz2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covSA2u * mati2x)>::RowCoefficients, C>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covSA2u * mati2x)>::ColumnCoefficients, Cx>);

  auto covT2l = CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(covT2l * Mat2 {4, 2, 2, 5}, Mat2 {42, 33, 32, 56}));
  EXPECT_TRUE(is_near(covT2l * mati2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covT2l * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(lower_triangular_matrix<decltype((covT2l * mati2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covT2l * matz2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covT2l * mati2x)>::RowCoefficients, C>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covT2l * mati2x)>::ColumnCoefficients, Cx>);

  auto covT2u = CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(covT2u * Mat2 {4, 2, 2, 5}, Mat2 {42, 33, 32, 56}));
  EXPECT_TRUE(is_near(covT2u * mati2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covT2u * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(upper_triangular_matrix<decltype((covT2u * mati2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covT2u * matz2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covT2u * mati2x)>::RowCoefficients, C>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covT2u * mati2x)>::ColumnCoefficients, Cx>);

  auto covD2 = CovD2 {9, 10};
  EXPECT_TRUE(is_near(covD2 * Mat2 {4, 2, 2, 5}, Mat2 {36, 18, 20, 50}));
  EXPECT_TRUE(is_near(covD2 * mati2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(covD2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((covD2 * mati2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covD2 * matz2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covD2 * mati2x)>::RowCoefficients, C>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covD2 * mati2x)>::ColumnCoefficients, Cx>);

  EXPECT_TRUE(is_near(covi2 * Mat2 {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covi2 * mati2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covi2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(identity_matrix<decltype((covi2 * mati2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covi2 * matz2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covi2 * mati2x)>::RowCoefficients, C>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covi2 * mati2x)>::ColumnCoefficients, Cx>);

  EXPECT_TRUE(is_near(covz2 * Mat2 {4, 2, 2, 5}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * mati2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero_matrix<decltype((covz2 * mati2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * matz2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covz2 * mati2x)>::RowCoefficients, C>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(covz2 * mati2x)>::ColumnCoefficients, Cx>);
}


TEST_F(covariance_tests, Covariance_mult_scalar)
{
  Mat2 p1 {4, 2, 2, 5};
  EXPECT_TRUE(is_near(CovSA2l(p1) * 2, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovSA2u(p1) * 2, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovT2l(p1) * 2, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovT2u(p1) * 2, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovD2 {1, 2} * 2, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(covi2 * 2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(covz2 * 2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l(p1) * 2).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u(p1) * 2).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l(p1) * 2).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u(p1) * 2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {1, 2} * 2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 * 2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * 2).nested_matrix())>);

  EXPECT_TRUE(is_near(2 * CovSA2l(p1), Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(2 * CovSA2u(p1), Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(2 * CovT2l(p1), Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(2 * CovT2u(p1), Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(2 * CovD2 {1, 2}, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(2 * covi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((2 * CovSA2l(p1)).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((2 * CovSA2u(p1)).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((2 * CovT2l(p1)).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((2 * CovT2u(p1)).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((2 * CovD2 {1, 2}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((2 * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((2 * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l(p1) * -2, Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(CovSA2u(p1) * -2, Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(CovD2 {1, 2} * -2, Mat2 {-2, 0, 0, -4}));
  EXPECT_TRUE(is_near(covi2 * -2, Mat2 {-2, 0, 0, -2}));
  EXPECT_TRUE(is_near(covz2 * -2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l(p1) * -2).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u(p1) * -2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {1, 2} * -2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 * -2).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 * -2).nested_matrix())>);

  EXPECT_TRUE(is_near(-2 * CovSA2l(p1), Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(-2 * CovSA2u(p1), Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(-2 * CovD2 {1, 2}, Mat2 {-2, 0, 0, -4}));
  EXPECT_TRUE(is_near(-2 * covi2, Mat2 {-2, 0, 0, -2}));
  EXPECT_TRUE(is_near(-2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((-2 * CovSA2l(p1)).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((-2 * CovSA2u(p1)).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((-2 * CovD2 {1, 2}).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((-2 * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((-2 * covz2).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l::identity() + 2 * CovSA2l::identity(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(CovSA2u::identity() * 2 + CovSA2u::identity(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(CovT2l::identity() + 2 * CovT2l::identity(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(CovT2u::identity() * 2 + CovT2u::identity(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(CovD2::identity() + 2 * CovD2::identity(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(CovI2::identity() * 2 + CovI2::identity(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(CovZ2::identity() + 2 * CovZ2::identity(), Mat2 {3, 0, 0, 3}));

  // Scalar division
  EXPECT_TRUE(is_near(CovSA2l(p1) / 0.5, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovSA2u(p1) / 0.5, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovT2l(p1) / 0.5, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovT2u(p1) / 0.5, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovD2 {1, 2} / 0.5, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(covi2 / 0.5, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(covz2 / 0.5, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l(p1) / 0.5).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u(p1) / 0.5).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype((CovT2l(p1) / 0.5).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype((CovT2u(p1) / 0.5).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {1, 2} / 0.5).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 / 0.5).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 / 0.5).nested_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l(p1) / -0.5, Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(CovSA2u(p1) / -0.5, Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(CovD2 {1, 2} / -0.5, Mat2 {-2, 0, 0, -4}));
  EXPECT_TRUE(is_near(covi2 / -0.5, Mat2 {-2, 0, 0, -2}));
  EXPECT_TRUE(is_near(covz2 / -0.5, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((CovSA2l(p1) / -0.5).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype((CovSA2u(p1) / -0.5).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((CovD2 {1, 2} / -0.5).nested_matrix())>);
  static_assert(diagonal_matrix<decltype((covi2 / -0.5).nested_matrix())>);
  static_assert(zero_matrix<decltype((covz2 / -0.5).nested_matrix())>);
}


TEST_F(covariance_tests, Covariance_scale)
{
  Mat2 p1 {4, 2, 2, 5};
  EXPECT_TRUE(is_near(scale(CovSA2l(p1), 2), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(scale(CovSA2u(p1), 2), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(scale(CovT2l(p1), 2), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(scale(CovT2u(p1), 2), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(scale(CovD2 {1, 2}, 2), Mat2 {4, 0, 0, 8}));
  static_assert(Eigen3::lower_triangular_storage<decltype(scale(CovSA2l(p1), 2).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype(scale(CovSA2u(p1), 2).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(scale(CovT2l(p1), 2).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(scale(CovT2u(p1), 2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype(scale(CovD2 {1, 2}, 2).nested_matrix())>);

  EXPECT_TRUE(is_near(inverse_scale(CovSA2l(p1), 0.5), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(inverse_scale(CovSA2u(p1), 0.5), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(inverse_scale(CovT2l(p1), 0.5), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(inverse_scale(CovT2u(p1), 0.5), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(inverse_scale(CovD2 {4, 8}, 2), Mat2 {1, 0, 0, 2}));
  static_assert(Eigen3::lower_triangular_storage<decltype(inverse_scale(CovSA2l(p1), 2).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype(inverse_scale(CovSA2u(p1), 2).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(inverse_scale(CovT2l(p1), 2).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(inverse_scale(CovT2u(p1), 2).nested_matrix())>);
  static_assert(diagonal_matrix<decltype(inverse_scale(CovD2 {1, 2}, 2).nested_matrix())>);

  // Rank-deficient case
  using M3 = native_matrix_t<double, 3, 3>;
  using Mat3 = Matrix<Coefficients<angle::Radians, Axis, angle::Radians>, Coefficients<angle::Radians, Axis, angle::Radians>, M3>;
  Matrix<Coefficients<angle::Radians, Axis, angle::Radians>, C> a1 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(scale(CovSA2l(p1), a1), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(scale(CovSA2u(p1), a1), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(scale(CovT2l(p1), a1), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(scale(CovT2u(p1), a1), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(scale(CovD2 {1, 2}, a1), Mat3 {9, 19, 29, 19, 41, 63, 29, 63, 97}));
  static_assert(Eigen3::lower_triangular_storage<decltype(scale(CovSA2l(p1), a1).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype(scale(CovSA2u(p1), a1).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(scale(CovT2l(p1), a1).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(scale(CovT2u(p1), a1).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype(scale(CovD2 {1, 2}, a1).nested_matrix())>);

  // Rank-sufficient case
  using CovSA3l = Covariance<Coefficients<angle::Radians, Axis, angle::Radians>, SelfAdjointMatrix<M3, TriangleType::lower>>;
  using CovSA3u = Covariance<Coefficients<angle::Radians, Axis, angle::Radians>, SelfAdjointMatrix<M3, TriangleType::upper>>;
  using CovT3l = Covariance<Coefficients<angle::Radians, Axis, angle::Radians>, TriangularMatrix<M3, TriangleType::lower>>;
  using CovT3u = Covariance<Coefficients<angle::Radians, Axis, angle::Radians>, TriangularMatrix<M3, TriangleType::upper>>;
  using CovD3 = Covariance<Coefficients<angle::Radians, Axis, angle::Radians>, DiagonalMatrix<native_matrix_t<double, 3, 1>>>;
  Mat3 q1 {4, 2, 2,
           2, 5, 3,
           2, 3, 6};
  Matrix<C, Coefficients<angle::Radians, Axis, angle::Radians>> b1 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(scale(CovSA3l(q1), b1), Mat2 {134, 317, 317, 761}));
  EXPECT_TRUE(is_near(scale(CovSA3u(q1), b1), Mat2 {134, 317, 317, 761}));
  EXPECT_TRUE(is_near(scale(CovT3l(q1), b1), Mat2 {134, 317, 317, 761}));
  EXPECT_TRUE(is_near(scale(CovT3u(q1), b1), Mat2 {134, 317, 317, 761}));
  EXPECT_TRUE(is_near(scale(CovD3 {4, 5, 6}, b1), Mat2 {78, 174, 174, 405}));
  static_assert(Eigen3::lower_triangular_storage<decltype(scale(CovSA3l(q1), b1).nested_matrix())>);
  static_assert(Eigen3::upper_triangular_storage<decltype(scale(CovSA3u(q1), b1).nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(scale(CovT3l(q1), b1).nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(scale(CovT3u(q1), b1).nested_matrix())>);
  static_assert(Eigen3::lower_triangular_storage<decltype(scale(CovD3 {4, 5, 6}, b1).nested_matrix())>);
}


TEST_F(covariance_tests, TypedMatrix_mult_Covariance)
{
  using MatI2 = Matrix<C, C, I2>;
  using MatZ2 = Matrix<C, C, Z2>;
  auto mati2 = MatI2(i2);
  auto matz2 = MatZ2(z2);
  using Cx = Coefficients<Axis, angle::Radians>;
  using MatI2x = Matrix<Cx, C, I2>;
  auto mati2x = MatI2x(i2);

  auto covSA2l = CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covSA2l, Mat2 {42, 32, 33, 56}));
  EXPECT_TRUE(is_near(mati2 * covSA2l, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(mati2x * covSA2l, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(matz2 * covSA2l, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::lower_triangular_storage<decltype((mati2 * covSA2l).nested_matrix())>);
  static_assert(zero_matrix<decltype((matz2 * covSA2l).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covSA2l)>::RowCoefficients, Cx>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covSA2l)>::ColumnCoefficients, C>);

  auto covSA2u = CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covSA2u, Mat2 {42, 32, 33, 56}));
  EXPECT_TRUE(is_near(mati2 * covSA2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(mati2x * covSA2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(matz2 * covSA2u, Mat2 {0, 0, 0, 0}));
  static_assert(Eigen3::upper_triangular_storage<decltype((mati2 * covSA2u).nested_matrix())>);
  static_assert(zero_matrix<decltype((matz2 * covSA2u).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covSA2u)>::RowCoefficients, Cx>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covSA2u)>::ColumnCoefficients, C>);

  auto covT2l = CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covT2l, Mat2 {42, 32, 33, 56}));
  EXPECT_TRUE(is_near(mati2 * covT2l, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(matz2 * covT2l, Mat2 {0, 0, 0, 0}));
  static_assert(lower_triangular_matrix<decltype((mati2 * covT2l).nested_matrix())>);
  static_assert(zero_matrix<decltype((matz2 * covT2l).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covT2l)>::RowCoefficients, Cx>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covT2l)>::ColumnCoefficients, C>);

  auto covT2u = CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covT2u, Mat2 {42, 32, 33, 56}));
  EXPECT_TRUE(is_near(mati2 * covT2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(matz2 * covT2u, Mat2 {0, 0, 0, 0}));
  static_assert(upper_triangular_matrix<decltype((mati2 * covT2u).nested_matrix())>);
  static_assert(zero_matrix<decltype((matz2 * covT2u).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covT2u)>::RowCoefficients, Cx>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covT2u)>::ColumnCoefficients, C>);

  auto covD2 = CovD2 {9, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covD2, Mat2 {36, 20, 18, 50}));
  EXPECT_TRUE(is_near(mati2 * covD2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(matz2 * covD2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((mati2 * covD2).nested_matrix())>);
  static_assert(zero_matrix<decltype((matz2 * covD2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covD2)>::RowCoefficients, Cx>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covD2)>::ColumnCoefficients, C>);

  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covi2, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(mati2 * covi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(matz2 * covi2, Mat2 {0, 0, 0, 0}));
  static_assert(identity_matrix<decltype((mati2 * covi2).nested_matrix())>);
  static_assert(zero_matrix<decltype((matz2 * covi2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covi2)>::RowCoefficients, Cx>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covi2)>::ColumnCoefficients, C>);

  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covz2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(mati2 * covz2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(matz2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero_matrix<decltype((mati2 * covz2).nested_matrix())>);
  static_assert(zero_matrix<decltype((matz2 * covz2).nested_matrix())>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covz2)>::RowCoefficients, Cx>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(mati2x * covz2)>::ColumnCoefficients, C>);
}


TEST_F(covariance_tests, Covariance_other_operations)
{
  EXPECT_TRUE(is_near(-CovSA2l {9, 3, 3, 10}, Mat2 {-9, -3, -3, -10}));
  EXPECT_TRUE(is_near(-CovSA2u {9, 3, 3, 10}, Mat2 {-9, -3, -3, -10}));
  //EXPECT_TRUE(is_near(-CovT2l {9, 3, 3, 10}, Mat2 {-9, -3, -3, -10})); // Should not compile.
  //EXPECT_TRUE(is_near(-CovT2u {9, 3, 3, 10}, Mat2 {-9, -3, -3, -10})); // Should not compile.
  EXPECT_TRUE(is_near(-CovD2 {9, 10}, Mat2 {-9, 0, 0, -10}));
  EXPECT_TRUE(is_near(-covi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(-covz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero_matrix<decltype((-sqcovz2).nested_matrix())>);

  EXPECT_TRUE((CovSA2l {9, 3, 3, 10} == CovSA2l {9, 3, 3, 10}));
  EXPECT_TRUE((CovSA2l {9, 3, 3, 10} == CovT2u {9, 3, 3, 10}));
  EXPECT_TRUE((CovSA2u {9, 3, 3, 10} != CovSA2u {9, 2, 3, 10}));
  EXPECT_TRUE((CovSA2l {9, 3, 3, 10} != Covariance<Axes<2>, SA2l> {9, 3, 3, 10}));
}


/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "covariances.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;
using namespace OpenKalman::test;

using M2 = eigen_matrix_t<double, 2, 2>;
using C = std::tuple<angle::Radians, Axis>;
using Mat2 = Matrix<C, C, M2>;
using Mat2col = Matrix<C, Axis, eigen_matrix_t<double, 2, 1>>;
using SA2l = HermitianAdapter<M2, TriangleType::lower>;
using SA2u = HermitianAdapter<M2, TriangleType::upper>;
using T2l = TriangularAdapter<M2, TriangleType::lower>;
using T2u = TriangularAdapter<M2, TriangleType::upper>;
using D2 = DiagonalAdapter<eigen_matrix_t<double, 2, 1>>;
using I2 = Eigen3::IdentityMatrix<eigen_matrix_t<double, 2, 2>>;
using Z2 = ZeroAdapter<eigen_matrix_t<double, 2, 2>>;
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
inline Z2 z2 = ZeroAdapter<eigen_matrix_t<double, 2, 2>>();
inline auto covi2 = CovI2(i2);
inline auto covz2 = CovZ2(z2);
inline auto sqcovi2 = SqCovI2(i2);
inline auto sqcovz2 = SqCovZ2(z2);


TEST(covariance_tests, SquareRootCovariance_addition)
{
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 0, 2, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 0, 2, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovD2 {2, 2}, Mat2 {5, 0, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + sqcovi2, Mat2 {4, 0, 1, 4}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + sqcovz2, Mat2 {3, 0, 1, 3}));
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} + SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} + SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} + SqCovD2 {2, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} + sqcovi2), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} + sqcovz2), TriangleType::lower>);

  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 2, 0, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 2, 0, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovD2 {2, 2}, Mat2 {5, 1, 0, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + sqcovi2, Mat2 {4, 1, 0, 4}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + sqcovz2, Mat2 {3, 1, 0, 3}));
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} + SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} + SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} + SqCovD2 {2, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} + sqcovi2), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} + sqcovz2), TriangleType::upper>);

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 0, 2, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 0, 2, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovD2 {2, 2}, Mat2 {5, 0, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + sqcovi2, Mat2 {4, 0, 1, 4}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + sqcovz2, Mat2 {3, 0, 1, 3}));
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} + SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} + SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} + SqCovD2 {2, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} + sqcovi2), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} + sqcovz2), TriangleType::lower>);

  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 2, 0, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 2, 0, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovD2 {2, 2}, Mat2 {5, 1, 0, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + sqcovi2, Mat2 {4, 1, 0, 4}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + sqcovz2, Mat2 {3, 1, 0, 3}));
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} + SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} + SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} + SqCovD2 {2, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} + sqcovi2), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} + sqcovz2), TriangleType::upper>);

  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 0, 1, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 1, 0, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 0, 1, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 1, 0, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovD2 {2, 2}, Mat2 {5, 0, 0, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + sqcovi2, Mat2 {4, 0, 0, 4}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + sqcovz2, Mat2 {3, 0, 0, 3}));
  static_assert(triangular_matrix<decltype(SqCovD2 {3, 3} + SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovD2 {3, 3} + SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovD2 {3, 3} + SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovD2 {3, 3} + SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(SqCovD2 {3, 3} + SqCovD2 {2, 2})>);
  static_assert(diagonal_matrix<decltype(SqCovD2 {3, 3} + sqcovi2)>);
  static_assert(diagonal_matrix<decltype(SqCovD2 {3, 3} + sqcovz2)>);

  EXPECT_TRUE(is_near(sqcovi2 + SqCovSA2l {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + SqCovSA2u {2, 1, 0, 2}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + SqCovT2l {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + SqCovT2u {2, 1, 0, 2}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + SqCovD2 {2, 2}, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + sqcovi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(sqcovi2 + sqcovz2, Mat2 {1, 0, 0, 1}));
  static_assert(triangular_matrix<decltype(sqcovi2 + SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(sqcovi2 + SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(sqcovi2 + SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(sqcovi2 + SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(sqcovi2 + SqCovD2 {2, 2})>);
  static_assert(diagonal_matrix<decltype(sqcovi2 + sqcovi2)>);
  static_assert(identity_matrix<decltype(sqcovi2 + sqcovz2)>);

  EXPECT_TRUE(is_near(sqcovz2 + SqCovSA2l {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + SqCovSA2u {2, 1, 0, 2}, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + SqCovT2l {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + SqCovT2u {2, 1, 0, 2}, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + SqCovD2 {2, 2}, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovz2 + sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype(sqcovz2 + SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(sqcovz2 + SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(sqcovz2 + SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(sqcovz2 + SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(sqcovz2 + SqCovD2 {2, 2})>);
  static_assert(identity_matrix<decltype(sqcovz2 + sqcovi2)>);
  static_assert(zero<decltype(sqcovz2 + sqcovz2)>);
}


TEST(covariance_tests, SquareRootCovariance_addition_mixed)
{
  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovSA2l {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovSA2l {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovSA2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(hermitian_adapter<decltype((SqCovD2 {4, 5} + CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovi2 + CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovz2 + CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovSA2u {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovSA2u {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovSA2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(hermitian_adapter<decltype((SqCovD2 {4, 5} + CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype((sqcovi2 + CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype((sqcovz2 + CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovT2l {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovT2l {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovT2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(hermitian_adapter<decltype((SqCovD2 {4, 5} + CovT2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovi2 + CovT2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(triangular_matrix<decltype((sqcovz2 + CovT2l {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::lower>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovT2u {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovT2u {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovT2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(hermitian_adapter<decltype((SqCovD2 {4, 5} + CovT2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype((sqcovi2 + CovT2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype((sqcovz2 + CovT2u {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::upper>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovD2 {9, 10}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovD2 {9, 10}, Mat2 {11, 1, 0, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovD2 {9, 10}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovD2 {9, 10}, Mat2 {11, 1, 0, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovD2 {9, 10}, Mat2 {13, 0, 0, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovD2 {9, 10}, Mat2 {10, 0, 0, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovD2 {9, 10}, Mat2 {9, 0, 0, 10}));
  static_assert(diagonal_matrix<decltype((SqCovD2 {4, 5} + CovD2 {9, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((sqcovi2 + CovD2 {9, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((sqcovz2 + CovD2 {9, 10}).get_self_adjoint_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + covi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + covi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + covi2, Mat2 {5, 0, 0, 6}));
  EXPECT_TRUE(is_near(sqcovi2 + covi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + covi2, Mat2 {1, 0, 0, 1}));
  static_assert(diagonal_matrix<decltype((SqCovD2 {4, 5} + covi2).get_self_adjoint_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((sqcovi2 + covi2).get_self_adjoint_nested_matrix())>);
  static_assert(identity_matrix<decltype((sqcovz2 + covi2).get_self_adjoint_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + covz2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + covz2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + covz2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + covz2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + covz2, Mat2 {4, 0, 0, 5}));
  EXPECT_TRUE(is_near(sqcovi2 + covz2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovz2 + covz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((SqCovD2 {4, 5} + covz2).get_self_adjoint_nested_matrix())>);
  static_assert(identity_matrix<decltype((sqcovi2 + covz2).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 + covz2).get_self_adjoint_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + Mat2 {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + Mat2 {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + Mat2 {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + Mat2 {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + Mat2 {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + Mat2 {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + Mat2 {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));

  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} + SqCovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} + sqcovi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} + sqcovz2, Mat2 {9, 3, 3, 10}));
}


TEST(covariance_tests, SquareRootCovariance_subtraction)
{
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, -1, 1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, -1, 1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovD2 {2, 2}, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - sqcovi2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - sqcovz2, Mat2 {3, 0, 1, 3}));
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} - SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} - SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} - SqCovD2 {2, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} - sqcovi2), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovSA2l {3, 0, 1, 3} - sqcovz2), TriangleType::lower>);

  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 1, -1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 1, -1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovD2 {2, 2}, Mat2 {1, 1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - sqcovi2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - sqcovz2, Mat2 {3, 1, 0, 3}));
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} - SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} - SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} - SqCovD2 {2, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} - sqcovi2), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovSA2u {3, 1, 0, 3} - sqcovz2), TriangleType::upper>);

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, -1, 1, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, -1, 1, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovD2 {2, 2}, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - sqcovi2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - sqcovz2, Mat2 {3, 0, 1, 3}));
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} - SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} - SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} - SqCovD2 {2, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} - sqcovi2), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovT2l {3, 0, 1, 3} - sqcovz2), TriangleType::lower>);

  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 1, -1, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 1, -1, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovD2 {2, 2}, Mat2 {1, 1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - sqcovi2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - sqcovz2, Mat2 {3, 1, 0, 3}));
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} - SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} - SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} - SqCovD2 {2, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} - sqcovi2), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovT2u {3, 1, 0, 3} - sqcovz2), TriangleType::upper>);

  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 0, -1, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, -1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 0, -1, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, -1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovD2 {2, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - sqcovi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - sqcovz2, Mat2 {3, 0, 0, 3}));
  static_assert(triangular_matrix<decltype(SqCovD2 {3, 3} - SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovD2 {3, 3} - SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(SqCovD2 {3, 3} - SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(SqCovD2 {3, 3} - SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(SqCovD2 {3, 3} - SqCovD2 {2, 2})>);
  static_assert(diagonal_matrix<decltype(SqCovD2 {3, 3} - sqcovi2)>);
  static_assert(diagonal_matrix<decltype(SqCovD2 {3, 3} - sqcovz2)>);

  EXPECT_TRUE(is_near(sqcovi2 - SqCovSA2l {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - SqCovSA2u {2, 1, 0, 2}, Mat2 {-1, -1, 0, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - SqCovT2l {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - SqCovT2u {2, 1, 0, 2}, Mat2 {-1, -1, 0, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - SqCovD2 {2, 2}, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - sqcovi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovi2 - sqcovz2, Mat2 {1, 0, 0, 1}));
  static_assert(triangular_matrix<decltype(sqcovi2 - SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(sqcovi2 - SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(sqcovi2 - SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(sqcovi2 - SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(sqcovi2 - SqCovD2 {2, 2})>);
  static_assert(zero<decltype(sqcovi2 - sqcovi2)>);
  static_assert(identity_matrix<decltype(sqcovi2 - sqcovz2)>);

  EXPECT_TRUE(is_near(sqcovz2 - SqCovSA2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - SqCovSA2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - SqCovT2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - SqCovT2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - SqCovD2 {2, 2}, Mat2 {-2, 0, 0, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - sqcovi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(sqcovz2 - sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype(sqcovz2 - SqCovSA2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(sqcovz2 - SqCovSA2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(sqcovz2 - SqCovT2l {2, 0, 1, 2}), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(sqcovz2 - SqCovT2u {2, 1, 0, 2}), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(sqcovz2 - SqCovD2 {2, 2})>);
  static_assert(diagonal_matrix<decltype(sqcovz2 - sqcovi2)>);
  static_assert(zero<decltype(sqcovz2 - sqcovz2)>);}


TEST(covariance_tests, SquareRootCovariance_subtraction_mixed)
{
  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovSA2l {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovSA2l {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovSA2l {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovSA2l {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovSA2l {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovSA2l {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovSA2l {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));
  static_assert(hermitian_adapter<decltype((SqCovD2 {4, 5} - CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovi2 - CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovz2 - CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovSA2u {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovSA2u {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovSA2u {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovSA2u {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovSA2u {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovSA2u {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovSA2u {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));
  static_assert(hermitian_adapter<decltype((SqCovD2 {4, 5} - CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype((sqcovi2 - CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype((sqcovz2 - CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovT2l {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovT2l {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovT2l {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovT2l {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovT2l {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovT2l {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovT2l {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));
  static_assert(hermitian_adapter<decltype((SqCovD2 {4, 5} - CovT2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovi2 - CovT2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovz2 - CovT2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovT2u {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovT2u {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovT2u {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovT2u {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovT2u {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovT2u {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovT2u {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));
  static_assert(hermitian_adapter<decltype((SqCovD2 {4, 5} - CovT2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype((sqcovi2 - CovT2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype((sqcovz2 - CovT2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovD2 {9, 10}, Mat2 {-7, 0, 1, -8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovD2 {9, 10}, Mat2 {-7, 1, 0, -8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovD2 {9, 10}, Mat2 {-7, 0, 1, -8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovD2 {9, 10}, Mat2 {-7, 1, 0, -8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovD2 {9, 10}, Mat2 {-5, 0, 0, -5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovD2 {9, 10}, Mat2 {-8, 0, 0, -9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovD2 {9, 10}, Mat2 {-9, 0, 0, -10}));
  static_assert(diagonal_matrix<decltype((SqCovD2 {4, 5} - CovD2 {9, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((sqcovi2 - CovD2 {9, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((sqcovz2 - CovD2 {9, 10}).get_self_adjoint_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - covi2, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - covi2, Mat2 {1, 1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - covi2, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - covi2, Mat2 {1, 1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - covi2, Mat2 {3, 0, 0, 4}));
  EXPECT_TRUE(is_near(sqcovi2 - covi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 - covi2, Mat2 {-1, 0, 0, -1}));
  static_assert(diagonal_matrix<decltype((SqCovD2 {4, 5} - covi2).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovi2 - covi2).get_self_adjoint_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((sqcovz2 - covi2).get_self_adjoint_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - covz2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - covz2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - covz2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - covz2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - covz2, Mat2 {4, 0, 0, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - covz2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovz2 - covz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((SqCovD2 {4, 5} - covz2).get_self_adjoint_nested_matrix())>);
  static_assert(identity_matrix<decltype((sqcovi2 - covz2).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 - covz2).get_self_adjoint_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - Mat2 {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - Mat2 {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - Mat2 {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - Mat2 {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - Mat2 {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - Mat2 {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - Mat2 {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));

  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} - SqCovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} - sqcovi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(Mat2 {9, 3, 3, 10} - sqcovz2, Mat2 {9, 3, 3, 10}));
}


TEST(covariance_tests, SquareRootCovariance_mult_covariance)
{
  auto sqcovsa2l = SqCovSA2l {3, 0, 1, 3};
  auto sqcovsa2u = SqCovSA2u {3, 1, 0, 3};
  auto sqcovt2l = SqCovT2l {3, 0, 1, 3};
  auto sqcovt2u = SqCovT2u {3, 1, 0, 3};

  EXPECT_TRUE(is_near(sqcovsa2l * sqcovsa2l, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(sqcovsa2l * sqcovt2l, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(sqcovt2l * sqcovsa2l, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(sqcovt2l * sqcovt2l, Mat2 {9, 0, 6, 9})); //
  EXPECT_TRUE(is_near(sqcovsa2u * sqcovsa2u, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(sqcovsa2u * sqcovt2u, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(sqcovt2u * sqcovsa2u, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(sqcovt2u * sqcovt2u, Mat2 {9, 6, 0, 9}));

  EXPECT_TRUE(is_near(sqcovsa2l * sqcovsa2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovsa2l * sqcovt2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovt2l * sqcovsa2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovt2l * sqcovt2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovsa2u * sqcovsa2l, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovsa2u * sqcovt2l, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovt2u * sqcovsa2l, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovt2u * sqcovt2l, Mat2 {10, 3, 3, 9}));

  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovSA2l {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovSA2u {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovT2l {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovT2u {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovD2 {9, 10}, Mat2 {27, 0, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((SqCovSA2l {3, 0, 1, 3} * covi2).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(zero<decltype((SqCovSA2l {3, 0, 1, 3} * covz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovSA2l {3, 0, 1, 3}, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovSA2u {3, 1, 0, 3}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovT2l {3, 0, 1, 3}, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovT2u {3, 1, 0, 3}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovD2 {3, 3}, Mat2 {9, 0, 3, 9}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * sqcovi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((SqCovSA2l {3, 0, 1, 3} * sqcovi2).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(zero<decltype((SqCovSA2l {3, 0, 1, 3} * sqcovz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovSA2l {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovSA2u {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovT2l {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovT2u {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovD2 {9, 10}, Mat2 {27, 10, 0, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * covi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((SqCovSA2u {3, 1, 0, 3} * covi2).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(zero<decltype((SqCovSA2u {3, 1, 0, 3} * covz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovSA2l {3, 0, 1, 3}, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovSA2u {3, 1, 0, 3}, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovT2l {3, 0, 1, 3}, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovT2u {3, 1, 0, 3}, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovD2 {3, 3}, Mat2 {9, 3, 0, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * sqcovi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((SqCovSA2u {3, 1, 0, 3} * sqcovi2).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(zero<decltype((SqCovSA2u {3, 1, 0, 3} * sqcovz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovSA2l {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovSA2u {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovT2l {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovT2u {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovD2 {9, 10}, Mat2 {27, 0, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype((SqCovT2l {3, 0, 1, 3} * covi2).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(zero<decltype((SqCovT2l {3, 0, 1, 3} * covz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovSA2l {3, 0, 1, 3}, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovSA2u {3, 1, 0, 3}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovT2l {3, 0, 1, 3}, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovT2u {3, 1, 0, 3}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovD2 {3, 3}, Mat2 {9, 0, 3, 9}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * sqcovi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype((SqCovT2l {3, 0, 1, 3} * SqCovT2l {3, 0, 1, 3}).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype((SqCovT2l {3, 0, 1, 3} * sqcovi2).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(zero<decltype((SqCovT2l {3, 0, 1, 3} * sqcovz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovSA2l {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovSA2u {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovT2l {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovT2u {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovD2 {9, 10}, Mat2 {27, 10, 0, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * covi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype((SqCovT2u {3, 1, 0, 3} * covi2).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(zero<decltype((SqCovT2u {3, 1, 0, 3} * covz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovSA2l {3, 0, 1, 3}, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovSA2u {3, 1, 0, 3}, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovT2l {3, 0, 1, 3}, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovT2u {3, 1, 0, 3}, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovD2 {3, 3}, Mat2 {9, 3, 0, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * sqcovi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype((SqCovT2u {3, 1, 0, 3} * SqCovT2u {3, 1, 0, 3}).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype((SqCovT2u {3, 1, 0, 3} * sqcovi2).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(zero<decltype((SqCovT2u {3, 1, 0, 3} * sqcovz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovT2l {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovT2u {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovD2 {9, 10}, Mat2 {81, 0, 0, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * covi2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((SqCovD2 {9, 10} * CovD2 {9, 10}).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((SqCovD2 {9, 10} * covi2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((SqCovD2 {9, 10} * covz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {27, 0, 10, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 9, 0, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {27, 0, 10, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 9, 0, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovD2 {3, 3}, Mat2 {27, 0, 0, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * sqcovi2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((SqCovD2 {9, 10} * SqCovD2 {9, 10}).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((SqCovD2 {9, 10} * sqcovi2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((SqCovD2 {9, 10} * sqcovz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(sqcovi2 * CovSA2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * CovSA2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * CovT2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * CovT2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * CovD2 {9, 10}, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * covi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovi2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((sqcovi2 * CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovi2 * CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype((sqcovi2 * CovT2l {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype((sqcovi2 * CovT2u {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((sqcovi2 * CovD2 {9, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(identity_matrix<decltype((sqcovi2 * covi2).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovi2 * covz2).get_self_adjoint_nested_matrix())>);

  EXPECT_TRUE(is_near(sqcovi2 * SqCovSA2l {3, 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * SqCovSA2u {3, 1, 0, 3}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * SqCovT2l {3, 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * SqCovT2u {3, 1, 0, 3}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * SqCovD2 {3, 3}, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovi2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((sqcovi2 * SqCovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((sqcovi2 * SqCovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype((sqcovi2 * SqCovT2l {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype((sqcovi2 * SqCovT2u {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((sqcovi2 * SqCovD2 {9, 10}).get_triangular_nested_matrix())>);
  static_assert(identity_matrix<decltype((sqcovi2 * sqcovi2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovi2 * sqcovz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(sqcovz2 * CovSA2l {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * CovSA2u {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * CovT2l {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * CovT2u {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * CovD2 {9, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * covi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero<decltype((sqcovz2 * CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * CovT2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * CovT2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * CovD2 {9, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * covi2).get_self_adjoint_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * covz2).get_self_adjoint_nested_matrix())>);

  EXPECT_TRUE(is_near(sqcovz2 * SqCovSA2l {3, 0, 1, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * SqCovSA2u {3, 1, 0, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * SqCovT2l {3, 0, 1, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * SqCovT2u {3, 1, 0, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * SqCovD2 {3, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * sqcovi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero<decltype((sqcovz2 * SqCovSA2l {9, 3, 3, 10}).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * SqCovSA2u {9, 3, 3, 10}).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * SqCovT2l {9, 3, 3, 10}).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * SqCovT2u {9, 3, 3, 10}).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * SqCovD2 {9, 10}).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * sqcovi2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * sqcovz2).get_triangular_nested_matrix())>);
}


TEST(covariance_tests, SquareRootCovariance_mult_TypedMatrix)
{
  using MatI2 = Matrix<C, C, I2>;
  using MatZ2 = Matrix<C, C, Z2>;
  auto mati2 = MatI2(i2);
  auto matz2 = MatZ2(z2);
  using Cx = std::tuple<Axis, angle::Radians>;
  using MatI2x = Matrix<C, Cx, I2>;
  auto mati2x = MatI2x(i2);

  auto sqCovSA2l = SqCovSA2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(sqCovSA2l * Mat2 {4, 2, 2, 5}, Mat2 {12, 6, 10, 17}));
  EXPECT_TRUE(is_near(sqCovSA2l * mati2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqCovSA2l * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((sqCovSA2l * mati2).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(zero<decltype((sqCovSA2l * matz2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovSA2l * mati2x), 0>, C>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovSA2l * mati2x), 1>, Cx>);

  auto sqCovSA2u = SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(sqCovSA2u * Mat2 {4, 2, 2, 5}, Mat2 {14, 11, 6, 15}));
  EXPECT_TRUE(is_near(sqCovSA2u * mati2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqCovSA2u * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((sqCovSA2u * mati2).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(zero<decltype((sqCovSA2u * matz2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovSA2u * mati2x), 0>, C>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovSA2u * mati2x), 1>, Cx>);

  auto sqCovT2l = SqCovT2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(sqCovT2l * Mat2 {4, 2, 2, 5}, Mat2 {12, 6, 10, 17}));
  EXPECT_TRUE(is_near(sqCovT2l * mati2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqCovT2l * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype((sqCovT2l * mati2).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(zero<decltype((sqCovT2l * matz2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovT2l * mati2x), 0>, C>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovT2l * mati2x), 1>, Cx>);

  auto sqCovT2u = SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(sqCovT2u * Mat2 {4, 2, 2, 5}, Mat2 {14, 11, 6, 15}));
  EXPECT_TRUE(is_near(sqCovT2u * mati2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqCovT2u * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype((sqCovT2u * mati2).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(zero<decltype((sqCovT2u * matz2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovT2u * mati2x), 0>, C>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovT2u * mati2x), 1>, Cx>);

  auto sqCovD2 = SqCovD2 {3, 3};
  EXPECT_TRUE(is_near(sqCovD2 * Mat2 {4, 2, 2, 5}, Mat2 {12, 6, 6, 15}));
  EXPECT_TRUE(is_near(sqCovD2 * mati2, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(sqCovD2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((sqCovD2 * mati2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqCovD2 * matz2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovD2 * mati2x), 0>, C>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqCovD2 * mati2x), 1>, Cx>);

  EXPECT_TRUE(is_near(sqcovi2 * Mat2 {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(sqcovi2 * mati2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovi2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(identity_matrix<decltype((sqcovi2 * mati2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovi2 * matz2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqcovi2 * mati2x), 0>, C>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqcovi2 * mati2x), 1>, Cx>);

  EXPECT_TRUE(is_near(sqcovz2 * Mat2 {4, 2, 2, 5}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * mati2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero<decltype((sqcovz2 * mati2).nested_object())>);
  static_assert(zero<decltype((sqcovz2 * matz2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqcovz2 * mati2x), 0>, C>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(sqcovz2 * mati2x), 1>, Cx>);
}


TEST(covariance_tests, SquareRootCovariance_mult_scalar)
{
  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} * 2, Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} * 2, Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} * 2, Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} * 2, Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(SqCovD2 {1, 2} * 2, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(sqcovi2 * 2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 * 2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((SqCovSA2l {2, 0, 1, 2} * 2).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((SqCovSA2u {2, 1, 0, 2} * 2).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype((SqCovT2l {2, 0, 1, 2} * 2).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype((SqCovT2u {2, 1, 0, 2} * 2).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((SqCovD2 {1, 2} * 2).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((sqcovi2 * 2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 * 2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(2 * SqCovSA2l {2, 0, 1, 2}, Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(2 * SqCovSA2u {2, 1, 0, 2}, Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(2 * SqCovT2l {2, 0, 1, 2}, Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(2 * SqCovT2u {2, 1, 0, 2}, Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(2 * SqCovD2 {1, 2}, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(2 * sqcovi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((2 * SqCovSA2l {2, 0, 1, 2}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((2 * SqCovSA2u {2, 1, 0, 2}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype((2 * SqCovT2l {2, 0, 1, 2}).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype((2 * SqCovT2u {2, 1, 0, 2}).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((2 * SqCovD2 {1, 2}).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((2 * sqcovi2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((2 * sqcovz2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(make_identity_matrix_like<SqCovSA2l>() + 2 * make_identity_matrix_like<SqCovSA2l>(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(make_identity_matrix_like<SqCovSA2u>() * 2 + make_identity_matrix_like<SqCovSA2u>(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(make_identity_matrix_like<SqCovT2l>() + 2 * make_identity_matrix_like<SqCovT2l>(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(make_identity_matrix_like<SqCovT2u>() * 2 + make_identity_matrix_like<SqCovT2u>(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(make_identity_matrix_like<SqCovD2>() + 2 * make_identity_matrix_like<SqCovD2>(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(make_identity_matrix_like<SqCovI2>() * 2 + make_identity_matrix_like<SqCovI2>(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(make_identity_matrix_like<SqCovZ2>() + 2 * make_identity_matrix_like<SqCovZ2>(), Mat2 {3, 0, 0, 3}));

  // Scalar division
  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} / 0.5, Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} / 0.5, Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} / 0.5, Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} / 0.5, Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(SqCovD2 {1, 2} / 0.5, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(sqcovi2 / 0.5, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 / 0.5, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((SqCovSA2l {2, 0, 1, 2} / 0.5).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype((SqCovSA2u {2, 1, 0, 2} / 0.5).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype((SqCovT2l {2, 0, 1, 2} / 0.5).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype((SqCovT2u {2, 1, 0, 2} / 0.5).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype((SqCovD2 {1, 2} / 0.5).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype((sqcovi2 / 0.5).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((sqcovz2 / 0.5).get_triangular_nested_matrix())>);
}


TEST(covariance_tests, SquareRootCovariance_scale)
{
  Matrix<std::tuple<angle::Radians, Axis, angle::Radians>, C> a1 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(scale(SqCovSA2l {2, 0, 1, 2}, 2), Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(scale(SqCovSA2u {2, 1, 0, 2}, 2), Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(scale(SqCovT2l {2, 0, 1, 2}, 2), Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(scale(SqCovT2u {2, 1, 0, 2}, 2), Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(scale(SqCovD2 {1, 2}, 2), Mat2 {2, 0, 0, 4}));
  static_assert(hermitian_adapter<decltype(scale(SqCovSA2l {2, 0, 1, 2}, 2).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(scale(SqCovSA2u {2, 1, 0, 2}, 2).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(scale(SqCovT2l {2, 0, 1, 2}, 2).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(scale(SqCovT2u {2, 1, 0, 2}, 2).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(scale(SqCovD2 {1, 2}, 2).get_triangular_nested_matrix())>);

  EXPECT_TRUE(is_near(inverse_scale(SqCovSA2l {2, 0, 1, 2}, 2), Mat2 {1, 0, 0.5, 1}));
  EXPECT_TRUE(is_near(inverse_scale(SqCovSA2u {2, 1, 0, 2}, 2), Mat2 {1, 0.5, 0, 1}));
  EXPECT_TRUE(is_near(inverse_scale(SqCovT2l {2, 0, 1, 2}, 2), Mat2 {1, 0, 0.5, 1}));
  EXPECT_TRUE(is_near(inverse_scale(SqCovT2u {2, 1, 0, 2}, 2), Mat2 {1, 0.5, 0, 1}));
  EXPECT_TRUE(is_near(inverse_scale(SqCovD2 {2, 4}, 2), Mat2 {1, 0, 0, 2}));
  static_assert(hermitian_adapter<decltype(inverse_scale(SqCovSA2l {2, 0, 1, 2}, 2).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(inverse_scale(SqCovSA2u {2, 1, 0, 2}, 2).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(inverse_scale(SqCovT2l {2, 0, 1, 2}, 2).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(inverse_scale(SqCovT2u {2, 1, 0, 2}, 2).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(inverse_scale(SqCovD2 {1, 2}, 2).get_triangular_nested_matrix())>);

  // Rank-deficient case
  using M3 = eigen_matrix_t<double, 3, 3>;
  using Mat3 = Matrix<std::tuple<angle::Radians, Axis, angle::Radians>, std::tuple<angle::Radians, Axis, angle::Radians>, M3>;
  EXPECT_TRUE(is_near(square(scale(SqCovSA2l {2, 0, 1, 2}, a1)), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(square(scale(SqCovSA2u {2, 1, 0, 2}, a1)), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(square(scale(SqCovT2l {2, 0, 1, 2}, a1)), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(square(scale(SqCovT2u {2, 1, 0, 2}, a1)), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(square(scale(SqCovD2 {2, 3}, a1)), Mat3 {40, 84, 128, 84, 180, 276, 128, 276, 424}));
  static_assert(hermitian_adapter<decltype(scale(SqCovSA2l {2, 0, 1, 2}, a1).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(scale(SqCovSA2u {2, 1, 0, 2}, a1).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(scale(SqCovT2l {2, 0, 1, 2}, a1).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(scale(SqCovT2u {2, 1, 0, 2}, a1).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(hermitian_adapter<decltype(scale(SqCovD2 {1, 2}, a1).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);

  // Rank-sufficient case
  using SqCovSA3l = SquareRootCovariance<std::tuple<angle::Radians, Axis, angle::Radians>, HermitianAdapter<M3, TriangleType::lower>>;
  using SqCovSA3u = SquareRootCovariance<std::tuple<angle::Radians, Axis, angle::Radians>, HermitianAdapter<M3, TriangleType::upper>>;
  using SqCovT3l = SquareRootCovariance<std::tuple<angle::Radians, Axis, angle::Radians>, TriangularAdapter<M3, TriangleType::lower>>;
  using SqCovT3u = SquareRootCovariance<std::tuple<angle::Radians, Axis, angle::Radians>, TriangularAdapter<M3, TriangleType::upper>>;
  using SqCovD3 = SquareRootCovariance<std::tuple<angle::Radians, Axis, angle::Radians>, DiagonalAdapter<eigen_matrix_t<double, 3, 1>>>;
  Mat3 q1l {4, 0, 0,
            2, 5, 0,
            2, 3, 6};
  Mat3 q1u {4, 2, 2,
            0, 5, 3,
            0, 0, 6};
  Matrix<C, std::tuple<angle::Radians, Axis, angle::Radians>> b1 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(square(scale(SqCovSA3l(q1l), b1)), Mat2 {881, 1997, 1997, 4589}));
  SqCovSA3l sqcovsa3lq1l {q1l}; EXPECT_TRUE(is_near(square(scale(sqcovsa3lq1l, b1)), Mat2 {881, 1997, 1997, 4589}));
  EXPECT_TRUE(is_near(square(scale(SqCovSA3u(q1u), b1)), Mat2 {881, 1997, 1997, 4589}));
  SqCovSA3u sqcovsa3uq1u {q1u}; EXPECT_TRUE(is_near(square(scale(sqcovsa3uq1u, b1)), Mat2 {881, 1997, 1997, 4589}));
  EXPECT_TRUE(is_near(square(scale(SqCovT3l(q1l), b1)), Mat2 {881, 1997, 1997, 4589}));
  SqCovT3l sqcovt3lq1l {q1l}; EXPECT_TRUE(is_near(square(scale(sqcovt3lq1l, b1)), Mat2 {881, 1997, 1997, 4589}));
  EXPECT_TRUE(is_near(square(scale(SqCovT3u(q1u), b1)), Mat2 {881, 1997, 1997, 4589}));
  SqCovT3u sqcovt3uq1u {q1u}; EXPECT_TRUE(is_near(square(scale(sqcovt3uq1u, b1)), Mat2 {881, 1997, 1997, 4589}));
  EXPECT_TRUE(is_near(square(scale(SqCovD3 {4, 5, 6}, b1)), Mat2 {440, 962, 962, 2177}));
  SqCovD3 sqcovd3456 {4, 5, 6}; EXPECT_TRUE(is_near(square(scale(sqcovd3456, b1)), Mat2 {440, 962, 962, 2177}));
  static_assert(hermitian_adapter<decltype(scale(SqCovSA3l(q1l), b1).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(scale(SqCovSA3u(q1u), b1).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(scale(SqCovT3l(q1l), b1).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(scale(SqCovT3u(q1u), b1).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(hermitian_adapter<decltype(scale(SqCovD3 {4, 5, 6}, b1).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
}


TEST(covariance_tests, TypedMatrix_mult_SquareRootCovariance)
{
  using MatI2 = Matrix<C, C, I2>;
  using MatZ2 = Matrix<C, C, Z2>;
  auto mati2 = MatI2(i2);
  auto matz2 = MatZ2(z2);
  using Cx = std::tuple<Axis, angle::Radians>;
  using MatI2x = Matrix<Cx, C, I2>;
  auto mati2x = MatI2x(i2);

  auto sqCovSA2l = SqCovSA2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovSA2l, Mat2 {14, 6, 11, 15}));
  EXPECT_TRUE(is_near(mati2 * sqCovSA2l, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovSA2l, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((mati2 * sqCovSA2l).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(zero<decltype((matz2 * sqCovSA2l).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovSA2l), 0>, Cx>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovSA2l), 1>, C>);

  auto sqCovSA2u = SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovSA2u, Mat2 {12, 10, 6, 17}));
  EXPECT_TRUE(is_near(mati2 * sqCovSA2u, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovSA2u, Mat2 {0, 0, 0, 0}));
  static_assert(hermitian_adapter<decltype((mati2 * sqCovSA2u).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(zero<decltype((matz2 * sqCovSA2u).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovSA2u), 0>, Cx>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovSA2u), 1>, C>);

  auto sqCovT2l = SqCovT2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovT2l, Mat2 {14, 6, 11, 15}));
  EXPECT_TRUE(is_near(mati2 * sqCovT2l, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovT2l, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype((mati2 * sqCovT2l).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(zero<decltype((matz2 * sqCovT2l).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovT2l), 0>, Cx>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovT2l), 1>, C>);

  auto sqCovT2u = SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovT2u, Mat2 {12, 10, 6, 17}));
  EXPECT_TRUE(is_near(mati2 * sqCovT2u, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovT2u, Mat2 {0, 0, 0, 0}));
  static_assert(triangular_matrix<decltype((mati2 * sqCovT2u).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(zero<decltype((matz2 * sqCovT2u).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovT2u), 0>, Cx>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovT2u), 1>, C>);

  auto sqCovD2 = SqCovD2 {3, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovD2, Mat2 {12, 6, 6, 15}));
  EXPECT_TRUE(is_near(mati2 * sqCovD2, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovD2, Mat2 {0, 0, 0, 0}));
  static_assert(diagonal_matrix<decltype((mati2 * sqCovD2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((matz2 * sqCovD2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovD2), 0>, Cx>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqCovD2), 1>, C>);

  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqcovi2, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(mati2 * sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(matz2 * sqcovi2, Mat2 {0, 0, 0, 0}));
  static_assert(identity_matrix<decltype((mati2 * sqcovi2).get_triangular_nested_matrix())>);
  static_assert(zero<decltype((matz2 * sqcovi2).nested_object())>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqcovi2), 0>, Cx>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqcovi2), 1>, C>);

  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqcovz2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(mati2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(matz2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero<decltype(nested_object(mati2 * sqcovz2))>);
  static_assert(zero<decltype(nested_object(matz2 * sqcovz2))>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqcovz2), 0>, Cx>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(mati2x * sqcovz2), 1>, C>);
}


TEST(covariance_tests, SquareRootCovariance_other_operations)
{
  EXPECT_TRUE(is_near(-SqCovT2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(-SqCovT2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(-SqCovD2 {4, 5}, Mat2 {-4, 0, 0, -5}));
  EXPECT_TRUE(is_near(-sqcovi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(-sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(zero<decltype(nested_object(-sqcovz2))>);

  EXPECT_TRUE((SqCovT2l {3, 0, 1, 3} == SqCovT2l {3, 0, 1, 3}));
  EXPECT_TRUE((SqCovT2l {3, 0, 1, 3} == SqCovSA2l {3, 0, 1, 3}));
  EXPECT_TRUE((SqCovT2u {3, 1, 0, 3} != SqCovT2u {3, 2, 0, 3}));
  EXPECT_TRUE((SqCovT2l {3, 0, 1, 3} != SquareRootCovariance<Dimensions<2>, T2l> {3, 0, 1, 3}));
}


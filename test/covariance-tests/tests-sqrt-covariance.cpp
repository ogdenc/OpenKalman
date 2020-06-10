/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "covariance_tests.h"

using namespace OpenKalman;

using M2 = Eigen::Matrix<double, 2, 2>;
using C = Coefficients<Angle, Axis>;
using Mat2 = TypedMatrix<C, C, M2>;
using Mat2col = TypedMatrix<C, Axis, Eigen::Matrix<double, 2, 1>>;
using SA2l = EigenSelfAdjointMatrix<M2, TriangleType::lower>;
using SA2u = EigenSelfAdjointMatrix<M2, TriangleType::upper>;
using T2l = EigenTriangularMatrix<M2, TriangleType::lower>;
using T2u = EigenTriangularMatrix<M2, TriangleType::upper>;
using D2 = EigenDiagonal<Eigen::Matrix<double, 2, 1>>;
using I2 = EigenIdentity<Eigen::Matrix<double, 2, 2>>;
using Z2 = EigenZero<Eigen::Matrix<double, 2, 2>>;
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
using M3 = Eigen::Matrix<double, 3, 3>;
using Mat3 = TypedMatrix<C, C, M3>;

inline I2 i2 = M2::Identity();
inline Z2 z2 = EigenZero<M2>();
inline auto covi2 = CovI2(i2);
inline auto covz2 = CovZ2(z2);
inline auto sqcovi2 = SqCovI2(i2);
inline auto sqcovz2 = SqCovZ2(z2);

TEST_F(covariance_tests, SquareRootCovariance_class)
{
  // Default constructor and Eigen3 construction
  SqCovSA2l clsa1;
  clsa1 << 3, 0, 1, 3;
  EXPECT_TRUE(is_near(clsa1, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusa1;
  cusa1 << 2, 1, 0, 2;
  EXPECT_TRUE(is_near(cusa1, Mat2 {2, 1, 0, 2}));
  SqCovT2l clt1;
  clt1 << 3, 0, 1, 3;
  EXPECT_TRUE(is_near(clt1, Mat2 {3, 0, 1, 3}));
  SqCovT2u cut1;
  cut1 << 2, 1, 0, 2;
  EXPECT_TRUE(is_near(cut1, Mat2 {2, 1, 0, 2}));
  SqCovD2 cd1;
  cd1 << 1, 2;
  EXPECT_TRUE(is_near(cd1, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(base_matrix(cd1), Mat2 {1, 0, 0, 2}));
  SqCovI2 ci1 = i2;
  EXPECT_TRUE(is_near(ci1, Mat2 {1, 0, 0, 1}));

  // Copy constructor
  SqCovSA2l clsa2 = const_cast<const SqCovSA2l&>(clsa1);
  EXPECT_TRUE(is_near(clsa2, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusa2 = const_cast<const SqCovSA2u&>(cusa1);
  EXPECT_TRUE(is_near(cusa2, Mat2 {2, 1, 0, 2}));
  SqCovT2l clt2 = const_cast<const SqCovT2l&>(clt1);
  EXPECT_TRUE(is_near(clt2, Mat2 {3, 0, 1, 3}));
  SqCovT2u cut2 = const_cast<const SqCovT2u&>(cut1);
  EXPECT_TRUE(is_near(cut2, Mat2 {2, 1, 0, 2}));
  SqCovD2 cd2 = const_cast<const SqCovD2&>(cd1);
  EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(base_matrix(cd2), Mat2 {1, 0, 0, 2}));
  SqCovD2 cd2b = cd1; // Template constructor, not copy constructor.
  EXPECT_TRUE(is_near(cd2b, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(base_matrix(cd2b), Mat2 {1, 0, 0, 2}));
  SqCovI2 ci2 = const_cast<const SqCovI2&>(ci1);
  EXPECT_TRUE(is_near(ci2, Mat2 {1, 0, 0, 1}));
  SqCovI2 ci2b = ci1; // Template constructor, not copy constructor.
  EXPECT_TRUE(is_near(ci2b, Mat2 {1, 0, 0, 1}));

  // Move constructor
  SqCovSA2l clsa3(std::move(SqCovSA2l {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(clsa3, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusa3(std::move(SqCovSA2u {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(cusa3, Mat2 {2, 1, 0, 2}));
  SqCovT2l clt3(std::move(SqCovT2l {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(clt3, Mat2 {3, 0, 1, 3}));
  SqCovT2u cut3(std::move(SqCovT2u {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(cut3, Mat2 {2, 1, 0, 2}));
  SqCovD2 cd3(std::move(SqCovD2 {1, 2}));
  EXPECT_TRUE(is_near(cd3, Mat2 {1, 0, 0, 2}));
  SqCovI2 ci3(std::move(SqCovI2 {i2}));
  EXPECT_TRUE(is_near(ci3, Mat2 {1, 0, 0, 1}));

  // Convert from different square-root covariance type
  SqCovSA2l clsasa4(SqCovSA2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(clsasa4, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusasa4(SqCovSA2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cusasa4, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsat4(SqCovT2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(clsat4, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusat4(SqCovT2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cusat4, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsat4s(SqCovT2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(clsat4s, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusat4s(SqCovT2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cusat4s, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltt4(adjoint(SqCovT2u {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(cltt4, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutt4(adjoint(SqCovT2l {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(cutt4, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltsa4(SqCovSA2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(cltsa4, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutsa4(SqCovSA2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cutsa4, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltsa4s(SqCovSA2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(cltsa4s, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutsa4s(SqCovSA2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cutsa4s, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsa4d(SqCovD2 {1, 2});
  EXPECT_TRUE(is_near(clsa4d, Mat2 {1, 0, 0, 2}));
  SqCovSA2u cusa4d(SqCovD2 {1, 2});
  EXPECT_TRUE(is_near(cusa4d, Mat2 {1, 0, 0, 2}));
  SqCovT2l clt4d(SqCovD2 {1, 2});
  EXPECT_TRUE(is_near(clt4d, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(base_matrix(clt4d), Mat2 {1, 0, 0, 2}));
  SqCovT2u cut4d(SqCovD2 {1, 2});
  EXPECT_TRUE(is_near(cut4d, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(base_matrix(cut4d), Mat2 {1, 0, 0, 2}));
  SqCovT2u cut4i(SqCovI2 {i2});
  EXPECT_TRUE(is_near(cut4i, Mat2 {1, 0, 0, 1}));

  // Convert from different non-square-root covariance type
  SqCovSA2l clsasa4X(CovSA2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsasa4X, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusasa4X(CovSA2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusasa4X, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsat4X(CovT2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsat4X, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusat4X(CovT2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusat4X, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsat4sX(CovT2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsat4sX, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusat4sX(CovT2u {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusat4sX, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltt4X(CovT2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltt4X, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutt4X(CovT2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutt4X, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltsa4X(CovSA2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltsa4X, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutsa4X(CovSA2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutsa4X, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltsa4sX(CovSA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltsa4sX, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutsa4sX(CovSA2u {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutsa4sX, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsa4dX(CovD2 {4, 9});
  EXPECT_TRUE(is_near(clsa4dX, Mat2 {2, 0, 0, 3}));
  SqCovSA2u cusa4dX(CovD2 {4, 9});
  EXPECT_TRUE(is_near(cusa4dX, Mat2 {2, 0, 0, 3}));
  SqCovT2l clt4dX(CovD2 {4, 9});
  EXPECT_TRUE(is_near(clt4dX, Mat2 {2, 0, 0, 3}));
  SqCovT2u cut4dX(CovD2 {4, 9});
  EXPECT_TRUE(is_near(cut4dX, Mat2 {2, 0, 0, 3}));
  SqCovT2u cut4iX(CovI2 {i2});
  EXPECT_TRUE(is_near(cut4iX, Mat2 {1, 0, 0, 1}));

  // Construct from a covariance base
  SqCovSA2l clsasa5(SA2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsasa5, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusasa5(SA2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusasa5, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsat5(T2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(clsat5, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusat5(T2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cusat5, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsat5s(T2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(clsat5s, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusat5s(T2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cusat5s, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltt5(adjoint(T2u {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(cltt5, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutt5(adjoint(T2l {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(cutt5, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltsa5(SA2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltsa5, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutsa5(SA2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutsa5, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltsa5s(SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltsa5s, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutsa5s(SA2u {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutsa5s, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsa5d(D2 {1, 2});
  EXPECT_TRUE(is_near(clsa4d, Mat2 {1, 0, 0, 2}));
  SqCovSA2u cusa5d(D2 {1, 2});
  EXPECT_TRUE(is_near(cusa4d, Mat2 {1, 0, 0, 2}));
  SqCovT2l clt5d(D2 {1, 2});
  EXPECT_TRUE(is_near(clt4d, Mat2 {1, 0, 0, 2}));
  SqCovT2u cut5d(D2 {1, 2});
  EXPECT_TRUE(is_near(cut4d, Mat2 {1, 0, 0, 2}));
  SqCovT2l clt5i(i2);
  EXPECT_TRUE(is_near(clt5i, Mat2 {1, 0, 0, 1}));

  // Construct from a typed matrix
  SqCovSA2l clsa6(Mat2 {3, 7, 1, 3});
  EXPECT_TRUE(is_near(clsa6, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusa6(Mat2 {2, 1, 7, 2});
  EXPECT_TRUE(is_near(cusa6, Mat2 {2, 1, 0, 2}));
  SqCovT2l clt6(Mat2 {3, 7, 1, 3});
  EXPECT_TRUE(is_near(clt6, Mat2 {3, 0, 1, 3}));
  SqCovT2u cut6(Mat2 {2, 1, 7, 2});
  EXPECT_TRUE(is_near(cut6, Mat2 {2, 1, 0, 2}));
  SqCovD2 cd6(TypedMatrix<C, Axis, Eigen::Matrix<double, 2, 1>> {1, 2});
  EXPECT_TRUE(is_near(cd6, Mat2 {1, 0, 0, 2}));

  // Construct from a regular matrix
  SqCovSA2l clsa7((M2() << 3, 7, 1, 3).finished());
  EXPECT_TRUE(is_near(clsa7, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusa7((M2() << 2, 1, 7, 2).finished());
  EXPECT_TRUE(is_near(cusa7, Mat2 {2, 1, 0, 2}));
  SqCovT2l clt7((M2() << 3, 7, 1, 3).finished());
  EXPECT_TRUE(is_near(clt7, Mat2 {3, 0, 1, 3}));
  SqCovT2u cut7((M2() << 2, 1, 7, 2).finished());
  EXPECT_TRUE(is_near(cut7, Mat2 {2, 1, 0, 2}));
  SqCovD2 cd7((Eigen::Matrix<double, 2, 1>() << 1, 2).finished());
  EXPECT_TRUE(is_near(cd7, Mat2 {1, 0, 0, 2}));

  // Construct from a list of coefficients
  SqCovSA2l clsa8{2, 7, 1, 2};
  EXPECT_TRUE(is_near(clsa8, Mat2 {2, 0, 1, 2}));
  SqCovSA2u cusa8{3, 1, 7, 3};
  EXPECT_TRUE(is_near(cusa8, Mat2 {3, 1, 0, 3}));
  SqCovT2l clt8{2, 7, 1, 2};
  EXPECT_TRUE(is_near(clt8, Mat2 {2, 0, 1, 2}));
  SqCovT2u cut8{3, 1, 7, 3};
  EXPECT_TRUE(is_near(cut8, Mat2 {3, 1, 0, 3}));
  SqCovD2 cd8({1, 2});
  EXPECT_TRUE(is_near(cd8, Mat2 {1, 0, 0, 2}));

  // Copy assignment
  clsa2 = clsa8;
  EXPECT_TRUE(is_near(clsa2, Mat2 {2, 0, 1, 2}));
  cusa2 = cusa8;
  EXPECT_TRUE(is_near(cusa2, Mat2 {3, 1, 0, 3}));
  clt2 = clt8;
  EXPECT_TRUE(is_near(clt2, Mat2 {2, 0, 1, 2}));
  cut2 = cut8;
  EXPECT_TRUE(is_near(cut2, Mat2 {3, 1, 0, 3}));
  cd2 = cd8;
  EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // Move assignment
  clsa2 = std::move(SqCovSA2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(clsa2, Mat2 {3, 0, 1, 3}));
  cusa2 = std::move(SqCovSA2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cusa2, Mat2 {2, 1, 0, 2}));
  clt2 = std::move(SqCovT2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(clt2, Mat2 {3, 0, 1, 3}));
  cut2 = std::move(SqCovT2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cut2, Mat2 {2, 1, 0, 2}));
  cd2 = std::move(SqCovD2 {1, 2});
  EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // Assign from a list of coefficients (via move assignment operator)
  clsa8 = {3, 0, 1, 3};
  EXPECT_TRUE(is_near(clsa8, Mat2 {3, 0, 1, 3}));
  cusa8 = {2, 1, 0, 2};
  EXPECT_TRUE(is_near(cusa8, Mat2 {2, 1, 0, 2}));
  clt8 = {3, 0, 1, 3};
  EXPECT_TRUE(is_near(clt8, Mat2 {3, 0, 1, 3}));
  cut8 = {2, 1, 0, 2};
  EXPECT_TRUE(is_near(cut8, Mat2 {2, 1, 0, 2}));
  cd8 = {3, 4};
  EXPECT_TRUE(is_near(cd8, Mat2 {3, 0, 0, 4}));

  // Assign from different covariance type
  clsasa4 = SqCovSA2u {2, 1, 0, 2};
  EXPECT_TRUE(is_near(clsasa4, Mat2 {2, 0, 1, 2}));
  cusasa4 = SqCovSA2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(cusasa4, Mat2 {3, 1, 0, 3}));
  clsat4 = SqCovT2u {2, 1, 0, 2};
  EXPECT_TRUE(is_near(clsat4, Mat2 {2, 0, 1, 2}));
  cusat4 = SqCovT2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(cusat4, Mat2 {3, 1, 0, 3}));
  clsat4s = SqCovT2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clsat4s, Mat2 {2, 0, 1, 2}));
  cusat4s = SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cusat4s, Mat2 {3, 1, 0, 3}));
  cltsa4 = SqCovSA2u {2, 1, 0, 2};
  EXPECT_TRUE(is_near(cltsa4, Mat2 {2, 0, 1, 2}));
  cutsa4 = SqCovSA2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(cutsa4, Mat2 {3, 1, 0, 3}));
  cltsa4s = SqCovSA2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(cltsa4s, Mat2 {2, 0, 1, 2}));
  cutsa4s = SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cutsa4s, Mat2 {3, 1, 0, 3}));
  clsa4d = SqCovD2 {3, 4};
  EXPECT_TRUE(is_near(clsa4d, Mat2 {3, 0, 0, 4}));
  cusa4d = SqCovD2 {3, 4};
  EXPECT_TRUE(is_near(cusa4d, Mat2 {3, 0, 0, 4}));
  clt4d = SqCovD2 {3, 4};
  EXPECT_TRUE(is_near(clt4d, Mat2 {3, 0, 0, 4}));
  cut4d = SqCovD2 {3, 4};
  EXPECT_TRUE(is_near(cut4d, Mat2 {3, 0, 0, 4}));
  cut4i = SqCovI2 {i2};
  EXPECT_TRUE(is_near(cut4i, Mat2 {1, 0, 0, 1}));

  // Assign from different non-square-root covariance type
  clsasa5 = CovSA2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(clsasa5, Mat2 {2, 0, 1, 2}));
  cusasa5 = CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cusasa5, Mat2 {3, 1, 0, 3}));
  clsat5 = CovT2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(clsat5, Mat2 {2, 0, 1, 2}));
  cusat5 = CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cusat5, Mat2 {3, 1, 0, 3}));
  clsat5s = CovT2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(clsat5s, Mat2 {2, 0, 1, 2}));
  cusat5s = CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cusat5s, Mat2 {3, 1, 0, 3}));
  cltsa5 = CovSA2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cltsa5, Mat2 {2, 0, 1, 2}));
  cutsa5 = CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cutsa5, Mat2 {3, 1, 0, 3}));
  cltsa5s = CovSA2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cltsa5s, Mat2 {2, 0, 1, 2}));
  cutsa5s = CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cutsa5s, Mat2 {3, 1, 0, 3}));
  clsa5d = CovD2 {4, 9};
  EXPECT_TRUE(is_near(clsa5d, Mat2 {2, 0, 0, 3}));
  cusa5d = CovD2 {4, 9};
  EXPECT_TRUE(is_near(cusa5d, Mat2 {2, 0, 0, 3}));
  clt5d = CovD2 {4, 9};
  EXPECT_TRUE(is_near(clt5d, Mat2 {2, 0, 0, 3}));
  cut5d = CovD2 {4, 9};
  EXPECT_TRUE(is_near(cut5d, Mat2 {2, 0, 0, 3}));
  clt5i = CovI2 {i2};
  EXPECT_TRUE(is_near(clt5i, Mat2 {1, 0, 0, 1}));

  // Increment
  clsa8 += SqCovSA2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clsa8, Mat2 {5, 0, 2, 5}));
  cusa8 += SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cusa8, Mat2 {5, 2, 0, 5}));
  clt8 += SqCovT2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clt8, Mat2 {5, 0, 2, 5}));
  cut8 += SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cut8, Mat2 {5, 2, 0, 5}));
  cd8 += SqCovD2 {1, 2};
  EXPECT_TRUE(is_near(cd8, Mat2 {4, 0, 0, 6}));

  // Decrement
  clsa8 -= SqCovSA2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clsa8, Mat2 {3, 0, 1, 3}));
  cusa8 -= SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cusa8, Mat2 {2, 1, 0, 2}));
  clt8 -= SqCovT2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clt8, Mat2 {3, 0, 1, 3}));
  cut8 -= SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cut8, Mat2 {2, 1, 0, 2}));
  cd8 -= SqCovD2 {1, 2};
  EXPECT_TRUE(is_near(cd8, Mat2 {3, 0, 0, 4}));

  // Scalar multiplication
  clsa2 *= 2;
  EXPECT_TRUE(is_near(clsa2, Mat2 {6, 0, 2, 6}));
  cusa2 *= 2;
  EXPECT_TRUE(is_near(cusa2, Mat2 {4, 2, 0, 4}));
  clt2 *= 2;
  EXPECT_TRUE(is_near(clt2, Mat2 {6, 0, 2, 6}));
  cut2 *= 2;
  EXPECT_TRUE(is_near(cut2, Mat2 {4, 2, 0, 4}));
  cd2 *= 2;
  EXPECT_TRUE(is_near(cd2, Mat2 {2, 0, 0, 4}));

  // Scalar division
  clsa2 /= 2;
  EXPECT_TRUE(is_near(clsa2, Mat2 {3, 0, 1, 3}));
  cusa2 /= 2;
  EXPECT_TRUE(is_near(cusa2, Mat2 {2, 1, 0, 2}));
  clt2 /= 2;
  EXPECT_TRUE(is_near(clt2, Mat2 {3, 0, 1, 3}));
  cut2 /= 2;
  EXPECT_TRUE(is_near(cut2, Mat2 {2, 1, 0, 2}));
  cd2 /= 2;
  EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // Scalar multiplication, zero
  clsa2 *= 0;
  EXPECT_TRUE(is_near(clsa2, Mat2::zero()));
  cusa2 *= 0;
  EXPECT_TRUE(is_near(cusa2, Mat2::zero()));
  clt2 *= 0;
  EXPECT_TRUE(is_near(clt2, Mat2::zero()));
  cut2 *= 0;
  EXPECT_TRUE(is_near(cut2, Mat2::zero()));
  cd2 *= 0;
  EXPECT_TRUE(is_near(cd2, Mat2::zero()));

  // Matrix multiplication
  clsa2 = {3, 0, 1, 3}; EXPECT_TRUE(is_near(clsa2 *= SqCovSA2l {2, 0, 1, 2}, Mat2 {6, 0, 5, 6}));
  clsa2 = {3, 0, 1, 3}; EXPECT_TRUE(is_near(clsa2 *= SqCovT2l {2, 0, 1, 2}, Mat2 {6, 0, 5, 6}));
  cusa2 = {2, 1, 0, 2}; EXPECT_TRUE(is_near(cusa2 *= SqCovSA2u {3, 1, 0, 3}, Mat2 {6, 5, 0, 6}));
  cusa2 = {2, 1, 0, 2}; EXPECT_TRUE(is_near(cusa2 *= SqCovT2u {3, 1, 0, 3}, Mat2 {6, 5, 0, 6}));
  clt2 = {3, 0, 1, 3}; EXPECT_TRUE(is_near(clt2 *= SqCovSA2l {2, 0, 1, 2}, Mat2 {6, 0, 5, 6}));
  clt2 = {3, 0, 1, 3}; EXPECT_TRUE(is_near(clt2 *= SqCovT2l {2, 0, 1, 2}, Mat2 {6, 0, 5, 6}));
  cut2 = {2, 1, 0, 2}; EXPECT_TRUE(is_near(cut2 *= SqCovSA2u {3, 1, 0, 3}, Mat2 {6, 5, 0, 6}));
  cut2 = {2, 1, 0, 2}; EXPECT_TRUE(is_near(cut2 *= SqCovT2u {3, 1, 0, 3}, Mat2 {6, 5, 0, 6}));
  cd2 = {1, 2}; EXPECT_TRUE(is_near(cd2 *= SqCovD2 {3, 4}, Mat2 {3, 0, 0, 8}));

  // Zero
  EXPECT_TRUE(is_near(SqCovSA2l::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(SqCovSA2u::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(SqCovT2l::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(SqCovT2u::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(SqCovD2::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(square(SqCovSA2l::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(square(SqCovSA2u::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(square(SqCovT2l::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(square(SqCovT2u::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(square(SqCovD2::zero()), M2::Zero()));

  // Identity
  EXPECT_TRUE(is_near(SqCovSA2l::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(SqCovSA2u::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(SqCovT2l::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(SqCovT2u::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(SqCovD2::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(square(SqCovSA2l::identity()), M2::Identity()));
  EXPECT_TRUE(is_near(square(SqCovSA2u::identity()), M2::Identity()));
  EXPECT_TRUE(is_near(square(SqCovT2l::identity()), M2::Identity()));
  EXPECT_TRUE(is_near(square(SqCovT2u::identity()), M2::Identity()));
  EXPECT_TRUE(is_near(square(SqCovD2::identity()), M2::Identity()));

  // Subscripts
  EXPECT_NEAR((SqCovSA2l {3, 0, 1, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((SqCovSA2l {3, 0, 1, 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((SqCovSA2l {3, 0, 1, 3})(1, 0), 1, 1e-6);
  EXPECT_NEAR((SqCovSA2l {3, 0, 1, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((SqCovSA2u {3, 1, 0, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((SqCovSA2u {3, 1, 0, 3})(0, 1), 1, 1e-6);
  EXPECT_NEAR((SqCovSA2u {3, 1, 0, 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((SqCovSA2u {3, 1, 0, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((SqCovT2l {3, 0, 1, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((SqCovT2l {3, 0, 1, 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((SqCovT2l {3, 0, 1, 3})(1, 0), 1, 1e-6);
  EXPECT_NEAR((SqCovT2l {3, 0, 1, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((SqCovT2u {3, 1, 0, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((SqCovT2u {3, 1, 0, 3})(0, 1), 1, 1e-6);
  EXPECT_NEAR((SqCovT2u {3, 1, 0, 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((SqCovT2u {3, 1, 0, 3})(1, 1), 3, 1e-6);
}

TEST_F(covariance_tests, SquareRootCovariance_deduction_guides)
{
  EXPECT_TRUE(is_near(SquareRootCovariance(SqCovSA2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(SquareRootCovariance(SqCovSA2l {3, 0, 1, 3}))>::Coefficients, C>);

  EXPECT_TRUE(is_near(SquareRootCovariance(SqCovSA2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(SquareRootCovariance(SqCovSA2l {3, 1, 0, 3}))>::Coefficients, C>);

  EXPECT_TRUE(is_near(SquareRootCovariance(T2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(SquareRootCovariance(T2l {3, 0, 1, 3}))>::Coefficients, Axes<2>>);

  EXPECT_TRUE(is_near(SquareRootCovariance(T2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(SquareRootCovariance(T2u {3, 1, 0, 3}))>::Coefficients, Axes<2>>);

  EXPECT_TRUE(is_near(SquareRootCovariance(D2 {1, 2}), Mat2 {1, 0, 0, 2}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(SquareRootCovariance(D2 {1, 2}))>::Coefficients, Axes<2>>);

  EXPECT_TRUE(is_near(SquareRootCovariance(Mat2 {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(SquareRootCovariance(Mat2 {3, 0, 1, 3}))>::Coefficients, C>);

  EXPECT_TRUE(is_near(SquareRootCovariance((M2() << 3, 0, 1, 3).finished()), Mat2 {3, 0, 1, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(SquareRootCovariance((M2() << 3, 0, 1, 3).finished()))>::Coefficients, Axes<2>>);

  EXPECT_TRUE(is_near(SquareRootCovariance {3., 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(SquareRootCovariance {3., 0, 1, 3})>::Coefficients, Axes<2>>);
}

TEST_F(covariance_tests, SquareRootCovariance_make)
{
  // SquareRootCovariance bases:
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::lower>(SA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::upper>(SA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::lower>(SA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::upper>(SA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::lower>(T2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::upper>(T2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::lower>(T2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::upper>(T2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::lower>(D2 {1, 2}).base_matrix(), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::upper>(D2 {1, 2}).base_matrix(), Mat2 {1, 0, 0, 2}));

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<C, TriangleType::lower>(SA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<C, TriangleType::upper>(SA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<C, TriangleType::lower>(SA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<C, TriangleType::upper>(SA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower>(T2l {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper>(T2l {3, 1, 0, 3}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower>(T2u {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper>(T2u {3, 1, 0, 3}).base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_SquareRootCovariance<C, TriangleType::lower>(D2 {1, 2}).base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_SquareRootCovariance<C, TriangleType::upper>(D2 {1, 2}).base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<C, TriangleType::lower, SA2l>().base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<C, TriangleType::upper, SA2l>().base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<C, TriangleType::lower, SA2u>().base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<C, TriangleType::upper, SA2u>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower, T2l>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper, T2l>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower, T2u>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper, T2u>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower, D2>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper, D2>().base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::lower>(SA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<C>(SA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<TriangleType::lower, T2u>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, T2l>().base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_SquareRootCovariance<C, D2>().base_matrix())>);

  // Other covariance:
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(SqCovSA2l {3, 0, 1, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(SqCovSA2l {3, 0, 1, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(SqCovSA2u {3, 1, 0, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(SqCovSA2u {3, 1, 0, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(SqCovT2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(SqCovT2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(SqCovT2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(SqCovT2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(SqCovD2 {1, 2}).base_matrix(), Mat2 {1, 0, 0, 2}));

  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(CovSA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(CovSA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(CovSA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(CovSA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(CovT2l {9, 3, 3, 10}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(CovT2l {9, 3, 3, 10}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(CovT2u {9, 3, 3, 10}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(CovT2u {9, 3, 3, 10}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(CovD2 {1, 2}).base_matrix(), Mat2 {1, 0, 0, 2}));

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::lower>(SqCovSA2l {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::upper>(SqCovSA2l {3, 1, 0, 3}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::lower>(SqCovSA2u {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::upper>(SqCovSA2u {3, 1, 0, 3}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<TriangleType::lower>(SqCovT2l {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<TriangleType::upper>(SqCovT2l {3, 1, 0, 3}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<TriangleType::lower>(SqCovT2u {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<TriangleType::upper>(SqCovT2u {3, 1, 0, 3}).base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_SquareRootCovariance<TriangleType::upper>(SqCovD2 {1, 2}).base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::lower, SqCovSA2l>().base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::upper, SqCovSA2l>().base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::lower, SqCovSA2u>().base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_SquareRootCovariance<TriangleType::upper, SqCovSA2u>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<TriangleType::lower, SqCovT2l>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<TriangleType::upper, SqCovT2l>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<TriangleType::lower, SqCovT2u>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<TriangleType::upper, SqCovT2u>().base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_SquareRootCovariance<TriangleType::upper, SqCovD2>().base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<>(SqCovSA2l {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_SquareRootCovariance<SqCovSA2l>().base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_SquareRootCovariance<SqCovD2>().base_matrix())>);

  // Regular matrices:
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::lower>(Mat2 {3, 0, 1, 3}.base_matrix()).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::upper>(Mat2 {3, 1, 0, 3}.base_matrix()).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C>(Mat2 {3, 0, 1, 3}.base_matrix()).base_matrix(), Mat2 {3, 0, 1, 3}));

  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower>(Mat2 {3, 0, 1, 3}.base_matrix()).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper>(Mat2 {3, 1, 0, 3}.base_matrix()).base_matrix())>);

  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower, M2>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper, M2>().base_matrix())>);

  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C>(Mat2 {3, 0, 1, 3}.base_matrix()).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<TriangleType::upper>(Mat2 {3, 1, 0, 3}.base_matrix()).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, M2>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<TriangleType::upper, M2>().base_matrix())>);

  // Typed matrices:
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::lower>(Mat2 {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<TriangleType::upper>(Mat2 {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance(Mat2 {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));

  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<TriangleType::lower>(Mat2 {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<TriangleType::upper>(Mat2 {3, 1, 0, 3}).base_matrix())>);

  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<TriangleType::lower, Mat2>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<TriangleType::upper, Mat2>().base_matrix())>);

  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance(Mat2 {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<Mat2>().base_matrix())>);

  // Eigen defaults
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::lower>(3, 0, 1, 3).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C, TriangleType::upper>(3, 1, 0, 3).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_SquareRootCovariance<C>(3, 0, 1, 3).base_matrix(), Mat2 {3, 0, 1, 3}));

  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower>(3, 0, 1, 3).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper>(3, 1, 0, 3).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C>(9, 3, 3, 10).base_matrix())>);

  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<TriangleType::lower>(3, 0, 1, 3).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::lower>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_SquareRootCovariance<C, TriangleType::upper>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_SquareRootCovariance<C>().base_matrix())>);
  static_assert(MatrixTraits<decltype(make_SquareRootCovariance<C>())>::Coefficients::size == 2);
}

TEST_F(covariance_tests, SquareRootCovariance_traits)
{
  static_assert(is_square_root_v<SqCovSA2l>);
  static_assert(not is_diagonal_v<SqCovSA2l>);
  static_assert(not is_self_adjoint_v<SqCovSA2l>);
  static_assert(not is_Cholesky_v<SqCovSA2l>);
  static_assert(is_triangular_v<SqCovSA2l>);
  static_assert(is_triangular_v<SqCovSA2u>);
  static_assert(is_lower_triangular_v<SqCovSA2l>);
  static_assert(is_upper_triangular_v<SqCovSA2u>);
  static_assert(not is_upper_triangular_v<SqCovSA2l>);
  static_assert(not is_identity_v<SqCovSA2l>);
  static_assert(not is_zero_v<SqCovSA2l>);

  static_assert(is_square_root_v<SqCovT2l>);
  static_assert(not is_diagonal_v<SqCovT2l>);
  static_assert(not is_self_adjoint_v<SqCovT2l>);
  static_assert(is_Cholesky_v<SqCovT2l>);
  static_assert(is_triangular_v<SqCovT2l>);
  static_assert(is_triangular_v<SqCovT2u>);
  static_assert(is_lower_triangular_v<SqCovT2l>);
  static_assert(not is_upper_triangular_v<SqCovT2l>);
  static_assert(is_upper_triangular_v<SqCovT2u>);
  static_assert(not is_identity_v<SqCovT2l>);
  static_assert(not is_zero_v<SqCovT2l>);

  static_assert(is_square_root_v<SqCovD2>);
  static_assert(is_diagonal_v<SqCovD2>);
  static_assert(is_self_adjoint_v<SqCovD2>);
  static_assert(not is_Cholesky_v<SqCovD2>);
  static_assert(is_triangular_v<SqCovD2>);
  static_assert(is_lower_triangular_v<SqCovD2>);
  static_assert(is_upper_triangular_v<SqCovD2>);
  static_assert(not is_identity_v<SqCovD2>);
  static_assert(not is_zero_v<SqCovD2>);

  static_assert(is_square_root_v<SqCovI2>);
  static_assert(is_diagonal_v<SqCovI2>);
  static_assert(is_self_adjoint_v<SqCovI2>);
  static_assert(not is_Cholesky_v<SqCovI2>);
  static_assert(is_triangular_v<SqCovI2>);
  static_assert(is_lower_triangular_v<SqCovI2>);
  static_assert(is_upper_triangular_v<SqCovI2>);
  static_assert(is_identity_v<SqCovI2>);
  static_assert(not is_zero_v<SqCovI2>);

  static_assert(is_square_root_v<SqCovZ2>);
  static_assert(is_diagonal_v<SqCovZ2>);
  static_assert(is_self_adjoint_v<SqCovZ2>);
  static_assert(not is_Cholesky_v<SqCovZ2>);
  static_assert(is_triangular_v<SqCovZ2>);
  static_assert(is_lower_triangular_v<SqCovZ2>);
  static_assert(is_upper_triangular_v<SqCovZ2>);
  static_assert(not is_identity_v<SqCovZ2>);
  static_assert(is_zero_v<SqCovZ2>);

  // MatrixTraits
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2l>::make(SA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2u>::make(SA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2l>::make(T2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2u>::make(T2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2l>::zero(), Eigen::Matrix<double, 2, 2>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2l>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2l>::make(SA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2u>::make(SA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2l>::make(T2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2u>::make(T2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2l>::zero(), Eigen::Matrix<double, 2, 2>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2l>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
}

TEST_F(covariance_tests, SquareRootCovariance_overloads)
{
  EXPECT_TRUE(is_near(internal::convert_base_matrix(SqCovSA2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(SqCovSA2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(SqCovT2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(SqCovT2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(SqCovD2 {1, 2}).base_matrix(), Mean {1., 2}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(SqCovI2 {i2}), Mat2 {1, 0, 0, 1}));

  EXPECT_TRUE(is_near(base_matrix(SqCovSA2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(base_matrix(SqCovSA2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(base_matrix(SqCovT2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(base_matrix(SqCovT2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(square(SqCovSA2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovSA2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovT2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovT2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovD2 {2, 3}), Mat2 {4, 0, 0, 9}));
  EXPECT_TRUE(is_near(square(SqCovI2 {i2}), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(square(SquareRootCovariance<C, EigenZero<Eigen::Matrix<double, 2, 2>>>()), Mat2 {0, 0, 0, 0}));

  EXPECT_TRUE(is_near(to_Cholesky(SqCovSA2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_Cholesky(SqCovSA2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(from_Cholesky(SqCovT2l {3, 0, 1, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(from_Cholesky(SqCovT2u {3, 1, 0, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));

  EXPECT_TRUE(is_near(strict_matrix(SqCovSA2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(strict_matrix(SqCovSA2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(strict_matrix(SqCovT2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(strict_matrix(SqCovT2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));

  static_assert(std::is_same_v<std::decay_t<decltype(strict(SqCovSA2l {3, 0, 1, 3} * 2))>, SqCovSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(SqCovSA2u {3, 1, 0, 3} * 2))>, SqCovSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(SqCovT2l {3, 0, 1, 3} * 2))>, SqCovT2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(SqCovT2u {3, 1, 0, 3} * 2))>, SqCovT2u>);

  EXPECT_TRUE(is_near(transpose(SqCovSA2l {3, 0, 1, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(transpose(SqCovSA2u {3, 1, 0, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(transpose(SqCovT2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(transpose(SqCovT2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(SqCovSA2l {3, 0, 1, 3})))>, SqCovSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(SqCovSA2u {3, 1, 0, 3})))>, SqCovSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(SqCovT2l {3, 0, 1, 3})))>, SqCovT2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(SqCovT2u {3, 1, 0, 3})))>, SqCovT2l>);

  EXPECT_TRUE(is_near(adjoint(SqCovSA2l {3, 0, 1, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(adjoint(SqCovSA2u {3, 1, 0, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(adjoint(SqCovT2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(adjoint(SqCovT2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(SqCovSA2l {3, 0, 1, 3})))>, SqCovSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(SqCovSA2u {3, 1, 0, 3})))>, SqCovSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(SqCovT2l {3, 0, 1, 3})))>, SqCovT2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(SqCovT2u {3, 1, 0, 3})))>, SqCovT2l>);

  EXPECT_NEAR(determinant(SqCovSA2l {3, 0, 1, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(SqCovSA2u {3, 1, 0, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(SqCovT2l {3, 0, 1, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(SqCovT2u {3, 1, 0, 3}), 9, 1e-6);

  EXPECT_NEAR(trace(SqCovSA2l {3, 0, 1, 3}), 6, 1e-6);
  EXPECT_NEAR(trace(SqCovSA2u {3, 1, 0, 3}), 6, 1e-6);
  EXPECT_NEAR(trace(SqCovT2l {3, 0, 1, 3}), 6, 1e-6);
  EXPECT_NEAR(trace(SqCovT2u {3, 1, 0, 3}), 6, 1e-6);

  auto x1sal = SqCovSA2l {3, 0, 1, 3};
  auto x1sau = SqCovSA2u {3, 1, 0, 3};
  auto x1tl = SqCovT2l {3, 0, 1, 3};
  auto x1tu = SqCovT2u {3, 1, 0, 3};
  rank_update(x1sal, Mat2 {2, 0, 1, 2}, 4);
  rank_update(x1sau, Mat2 {2, 0, 1, 2}, 4);
  rank_update(x1tl, Mat2 {2, 0, 1, 2}, 4);
  rank_update(x1tu, Mat2 {2, 0, 1, 2}, 4);
  EXPECT_TRUE(is_near(x1sal, Mat2 {5., 0, 2.2, std::sqrt(25.16)}));
  EXPECT_TRUE(is_near(x1sau, Mat2 {5., 2.2, 0, std::sqrt(25.16)}));
  EXPECT_TRUE(is_near(x1tl, Mat2 {5., 0, 2.2, std::sqrt(25.16)}));
  EXPECT_TRUE(is_near(x1tu, Mat2 {5., 2.2, 0, std::sqrt(25.16)}));
  EXPECT_TRUE(is_near(rank_update(SqCovSA2l {3, 0, 1, 3}, Mat2 {2, 0, 1, 2}, 4), Mat2 {5., 0, 2.2, std::sqrt(25.16)}));
  EXPECT_TRUE(is_near(rank_update(SqCovSA2u {3, 1, 0, 3}, Mat2 {2, 0, 1, 2}, 4), Mat2 {5., 2.2, 0, std::sqrt(25.16)}));
  EXPECT_TRUE(is_near(rank_update(SqCovT2l {3, 0, 1, 3}, Mat2 {2, 0, 1, 2}, 4), Mat2 {5., 0, 2.2, std::sqrt(25.16)}));
  EXPECT_TRUE(is_near(rank_update(SqCovT2u {3, 1, 0, 3}, Mat2 {2, 0, 1, 2}, 4), Mat2 {5., 2.2, 0, std::sqrt(25.16)}));

  EXPECT_TRUE(is_near(solve(SqCovSA2l {3, 0, 1, 3}, Mat2col {3, 7}), Mat2col {1, 2}));
  EXPECT_TRUE(is_near(solve(SqCovSA2u {3, 1, 0, 3}, Mat2col {5, 6}), Mat2col {1, 2}));
  EXPECT_TRUE(is_near(solve(SqCovT2l {3, 0, 1, 3}, Mat2col {3, 7}), Mat2col {1, 2}));
  EXPECT_TRUE(is_near(solve(SqCovT2u {3, 1, 0, 3}, Mat2col {5, 6}), Mat2col {1, 2}));

  EXPECT_TRUE(is_near(reduce_columns(SqCovSA2l {3, 0, 1, 3}), Mat2col {1.5, 2}));
  EXPECT_TRUE(is_near(reduce_columns(SqCovSA2u {3, 1, 0, 3}), Mat2col {2, 1.5}));
  EXPECT_TRUE(is_near(reduce_columns(SqCovT2l {3, 0, 1, 3}), Mat2col {1.5, 2}));
  EXPECT_TRUE(is_near(reduce_columns(SqCovT2u {3, 1, 0, 3}), Mat2col {2, 1.5}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(SqCovSA2l {3, 0, 1, 3})), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(SqCovSA2u {3, 1, 0, 3})), Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(SqCovT2l {3, 0, 1, 3})), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(SqCovT2u {3, 1, 0, 3})), Mat2 {10, 3, 3, 9}));

  EXPECT_TRUE(is_near(square(QR_decomposition(SqCovSA2l {3, 0, 1, 3})), Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(square(QR_decomposition(SqCovSA2u {3, 1, 0, 3})), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(QR_decomposition(SqCovT2l {3, 0, 1, 3})), Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(square(QR_decomposition(SqCovT2u {3, 1, 0, 3})), Mat2 {9, 3, 3, 10}));
}

TEST_F(covariance_tests, SquareRootCovariance_blocks)
{
  using C4 = Concatenate<C, C>;
  using M4 = Eigen::Matrix<double, 4, 4>;
  using Mat4 = TypedMatrix<C4, C4, M4>;
  using SqCovSA4l = SquareRootCovariance<C4, EigenSelfAdjointMatrix<M4, TriangleType::lower>>;
  using SqCovSA4u = SquareRootCovariance<C4, EigenSelfAdjointMatrix<M4, TriangleType::upper>>;
  using SqCovT4l = SquareRootCovariance<C4, EigenTriangularMatrix<M4, TriangleType::lower>>;
  using SqCovT4u = SquareRootCovariance<C4, EigenTriangularMatrix<M4, TriangleType::upper>>;
  Mat2 m1l {3, 0, 1, 3}, m2l {2, 0, 1, 2}, m1u {3, 1, 0, 3}, m2u {2, 1, 0, 2};
  Mat4 nl {3, 0, 0, 0,
          1, 3, 0, 0,
          0, 0, 2, 0,
          0, 0, 1, 2};
  Mat4 nu {3, 1, 0, 0,
           0, 3, 0, 0,
           0, 0, 2, 1,
           0, 0, 0, 2};
  EXPECT_TRUE(is_near(concatenate(SqCovSA2l(m1l), SqCovSA2l(m2l)), nl));
  EXPECT_TRUE(is_near(concatenate(SqCovSA2u(m1u), SqCovSA2u(m2u)), nu));
  EXPECT_TRUE(is_near(concatenate(SqCovT2l(m1l), SqCovT2l(m2l)), nl));
  EXPECT_TRUE(is_near(concatenate(SqCovT2u(m1u), SqCovT2u(m2u)), nu));

  EXPECT_TRUE(is_near(split(SqCovSA4l(nl)), std::tuple {}));
  EXPECT_TRUE(is_near(split<C, C>(SqCovSA4l(nl)), std::tuple {m1l, m2l}));
  EXPECT_TRUE(is_near(split<C, C>(SqCovSA4u(nu)), std::tuple {m1u, m2u}));
  EXPECT_TRUE(is_near(split<C, C>(SqCovT4l(nl)), std::tuple {m1l, m2l}));
  EXPECT_TRUE(is_near(split<C, C>(SqCovT4u(nu)), std::tuple {m1u, m2u}));
  EXPECT_TRUE(is_near(split<C, C::Take<1>>(SqCovSA4l(nl)), std::tuple {m1l, TypedMatrix<Angle, Angle>{2}}));
  EXPECT_TRUE(is_near(split<C, C::Take<1>>(SqCovSA4u(nu)), std::tuple {m1u, TypedMatrix<Angle, Angle>{2}}));
  EXPECT_TRUE(is_near(split<C, C::Take<1>>(SqCovT4l(nl)), std::tuple {m1l, TypedMatrix<Angle, Angle>{2}}));
  EXPECT_TRUE(is_near(split<C, C::Take<1>>(SqCovT4u(nu)), std::tuple {m1u, TypedMatrix<Angle, Angle>{2}}));

  EXPECT_TRUE(is_near(split_vertical(SqCovSA4l(nl)), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(SqCovSA4l(nl)), std::tuple {concatenate_horizontal(m1l, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2l)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(SqCovSA4u(nu)), std::tuple {concatenate_horizontal(m1u, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2u)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(SqCovT4l(nl)), std::tuple {concatenate_horizontal(m1l, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2l)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(SqCovT4u(nu)), std::tuple {concatenate_horizontal(m1u, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2u)}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(SqCovSA4l(nl)), std::tuple {concatenate_horizontal(m1l, Mat2::zero()), TypedMatrix<Angle, C4>{0, 0, 2, 0}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(SqCovSA4u(nu)), std::tuple {concatenate_horizontal(m1u, Mat2::zero()), TypedMatrix<Angle, C4>{0, 0, 2, 1}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(SqCovT4l(nl)), std::tuple {concatenate_horizontal(m1l, Mat2::zero()), TypedMatrix<Angle, C4>{0, 0, 2, 0}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(SqCovT4u(nu)), std::tuple {concatenate_horizontal(m1u, Mat2::zero()), TypedMatrix<Angle, C4>{0, 0, 2, 1}}));

  EXPECT_TRUE(is_near(split_horizontal(SqCovSA4l(nl)), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(SqCovSA4l(nl)), std::tuple {concatenate_vertical(m1l, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2l)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(SqCovSA4u(nu)), std::tuple {concatenate_vertical(m1u, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2u)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(SqCovT4l(nl)), std::tuple {concatenate_vertical(m1l, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2l)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(SqCovT4u(nu)), std::tuple {concatenate_vertical(m1u, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2u)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(SqCovSA4l(nl)), std::tuple {concatenate_vertical(m1l, Mat2::zero()), TypedMatrix<C4, Angle>{0, 0, 2, 1}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(SqCovSA4u(nu)), std::tuple {concatenate_vertical(m1u, Mat2::zero()), TypedMatrix<C4, Angle>{0, 0, 2, 0}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(SqCovT4l(nl)), std::tuple {concatenate_vertical(m1l, Mat2::zero()), TypedMatrix<C4, Angle>{0, 0, 2, 1}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(SqCovT4u(nu)), std::tuple {concatenate_vertical(m1u, Mat2::zero()), TypedMatrix<C4, Angle>{0, 0, 2, 0}}));

  EXPECT_TRUE(is_near(column(SqCovSA4l(nl), 2), Mean{0., 0, 2, 1}));
  EXPECT_TRUE(is_near(column(SqCovSA4u(nu), 2), Mean{0., 0, 2, 0}));
  EXPECT_TRUE(is_near(column(SqCovT4l(nl), 2), Mean{0., 0, 2, 1}));
  EXPECT_TRUE(is_near(column(SqCovT4u(nu), 2), Mean{0., 0, 2, 0}));

  EXPECT_TRUE(is_near(column<1>(SqCovSA4l(nl)), Mean{0., 3, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(SqCovSA4u(nu)), Mean{1., 3, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(SqCovT4l(nl)), Mean{0., 3, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(SqCovT4u(nu)), Mean{1., 3, 0, 0}));

  EXPECT_TRUE(is_near(apply_columnwise(SqCovSA2l(m1l), [](auto col){ return col * 2; }), Mat2 {6, 0, 2, 6}));
  EXPECT_TRUE(is_near(apply_columnwise(SqCovSA2u(m1u), [](const auto col){ return col * 2; }), Mat2 {6, 2, 0, 6}));
  EXPECT_TRUE(is_near(apply_columnwise(SqCovT2l(m1l), [](auto&& col){ return col * 2; }), Mat2 {6, 0, 2, 6}));
  EXPECT_TRUE(is_near(apply_columnwise(SqCovT2u(m1u), [](const auto& col){ return col * 2; }), Mat2 {6, 2, 0, 6}));

  EXPECT_TRUE(is_near(apply_columnwise(SqCovSA2l(m1l), [](auto col, std::size_t i){ return col * i; }), Mat2 {0, 0, 0, 3}));
  EXPECT_TRUE(is_near(apply_columnwise(SqCovSA2u(m1u), [](const auto col, std::size_t i){ return col * i; }), Mat2 {0, 1, 0, 3}));
  EXPECT_TRUE(is_near(apply_columnwise(SqCovT2l(m1l), [](auto&& col, std::size_t i){ return col * i; }), Mat2 {0, 0, 0, 3}));
  EXPECT_TRUE(is_near(apply_columnwise(SqCovT2u(m1u), [](const auto& col, std::size_t i){ return col * i; }), Mat2 {0, 1, 0, 3}));

  EXPECT_TRUE(is_near(apply_coefficientwise(SqCovSA2l(m1l), [](auto x){ return x + 1; }), Mat2 {4, 1, 2, 4}));
  EXPECT_TRUE(is_near(apply_coefficientwise(SqCovSA2u(m1u), [](const auto x){ return x + 1; }), Mat2 {4, 2, 1, 4}));
  EXPECT_TRUE(is_near(apply_coefficientwise(SqCovT2l(m1l), [](auto&& x){ return x + 1; }), Mat2 {4, 1, 2, 4}));
  EXPECT_TRUE(is_near(apply_coefficientwise(SqCovT2u(m1u), [](const auto& x){ return x + 1; }), Mat2 {4, 2, 1, 4}));

  EXPECT_TRUE(is_near(apply_coefficientwise(SqCovSA2l(m1l), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat2 {3, 1, 2, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise(SqCovSA2u(m1u), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat2 {3, 2, 1, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise(SqCovT2l(m1l), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat2 {3, 1, 2, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise(SqCovT2u(m1u), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat2 {3, 2, 1, 5}));
}


TEST_F(covariance_tests, SquareRootCovariance_addition)
{
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 0, 2, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 0, 2, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + SqCovD2 {2, 2}, Mat2 {5, 0, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + sqcovi2, Mat2 {4, 0, 1, 4}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} + sqcovz2, Mat2 {3, 0, 1, 3}));
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} + SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} + SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} + SqCovD2 {2, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} + sqcovi2)>);
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} + sqcovz2)>);

  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 2, 0, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 2, 0, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + SqCovD2 {2, 2}, Mat2 {5, 1, 0, 5}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + sqcovi2, Mat2 {4, 1, 0, 4}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} + sqcovz2, Mat2 {3, 1, 0, 3}));
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} + SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} + SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} + SqCovD2 {2, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} + sqcovi2)>);
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} + sqcovz2)>);

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 0, 2, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 0, 2, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + SqCovD2 {2, 2}, Mat2 {5, 0, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + sqcovi2, Mat2 {4, 0, 1, 4}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} + sqcovz2, Mat2 {3, 0, 1, 3}));
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} + SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} + SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} + SqCovD2 {2, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} + sqcovi2)>);
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} + sqcovz2)>);

  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 2, 0, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 2, 0, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + SqCovD2 {2, 2}, Mat2 {5, 1, 0, 5}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + sqcovi2, Mat2 {4, 1, 0, 4}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} + sqcovz2, Mat2 {3, 1, 0, 3}));
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} + SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} + SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} + SqCovD2 {2, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} + sqcovi2)>);
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} + sqcovz2)>);

  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovSA2l {2, 0, 1, 2}, Mat2 {5, 0, 1, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovSA2u {2, 1, 0, 2}, Mat2 {5, 1, 0, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovT2l {2, 0, 1, 2}, Mat2 {5, 0, 1, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovT2u {2, 1, 0, 2}, Mat2 {5, 1, 0, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + SqCovD2 {2, 2}, Mat2 {5, 0, 0, 5}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + sqcovi2, Mat2 {4, 0, 0, 4}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} + sqcovz2, Mat2 {3, 0, 0, 3}));
  static_assert(is_lower_triangular_v<decltype(SqCovD2 {3, 3} + SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovD2 {3, 3} + SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovD2 {3, 3} + SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovD2 {3, 3} + SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_diagonal_v<decltype(SqCovD2 {3, 3} + SqCovD2 {2, 2})>);
  static_assert(is_diagonal_v<decltype(SqCovD2 {3, 3} + sqcovi2)>);
  static_assert(is_diagonal_v<decltype(SqCovD2 {3, 3} + sqcovz2)>);

  EXPECT_TRUE(is_near(sqcovi2 + SqCovSA2l {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + SqCovSA2u {2, 1, 0, 2}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + SqCovT2l {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + SqCovT2u {2, 1, 0, 2}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + SqCovD2 {2, 2}, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 + sqcovi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(sqcovi2 + sqcovz2, Mat2 {1, 0, 0, 1}));
  static_assert(is_lower_triangular_v<decltype(sqcovi2 + SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(sqcovi2 + SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_lower_triangular_v<decltype(sqcovi2 + SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(sqcovi2 + SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_diagonal_v<decltype(sqcovi2 + SqCovD2 {2, 2})>);
  static_assert(is_diagonal_v<decltype(sqcovi2 + sqcovi2)>);
  static_assert(is_identity_v<decltype(sqcovi2 + sqcovz2)>);

  EXPECT_TRUE(is_near(sqcovz2 + SqCovSA2l {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + SqCovSA2u {2, 1, 0, 2}, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + SqCovT2l {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + SqCovT2u {2, 1, 0, 2}, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + SqCovD2 {2, 2}, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovz2 + sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype(sqcovz2 + SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(sqcovz2 + SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_lower_triangular_v<decltype(sqcovz2 + SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(sqcovz2 + SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_diagonal_v<decltype(sqcovz2 + SqCovD2 {2, 2})>);
  static_assert(is_identity_v<decltype(sqcovz2 + sqcovi2)>);
  static_assert(is_zero_v<decltype(sqcovz2 + sqcovz2)>);
}


TEST_F(covariance_tests, SquareRootCovariance_addition_mixed)
{
  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovSA2l {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovSA2l {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovSA2l {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovSA2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((SqCovD2 {4, 5} + CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovi2 + CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovz2 + CovSA2l {9, 3, 3, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovSA2u {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovSA2u {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovSA2u {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovSA2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((SqCovD2 {4, 5} + CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovi2 + CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovz2 + CovSA2u {9, 3, 3, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovT2l {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovT2l {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovT2l {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovT2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((SqCovD2 {4, 5} + CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovi2 + CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((sqcovz2 + CovT2l {9, 3, 3, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovT2u {9, 3, 3, 10}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovT2u {9, 3, 3, 10}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovT2u {9, 3, 3, 10}, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovT2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((SqCovD2 {4, 5} + CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovi2 + CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((sqcovz2 + CovT2u {9, 3, 3, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + CovD2 {9, 10}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + CovD2 {9, 10}, Mat2 {11, 1, 0, 12}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + CovD2 {9, 10}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + CovD2 {9, 10}, Mat2 {11, 1, 0, 12}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + CovD2 {9, 10}, Mat2 {13, 0, 0, 15}));
  EXPECT_TRUE(is_near(sqcovi2 + CovD2 {9, 10}, Mat2 {10, 0, 0, 11}));
  EXPECT_TRUE(is_near(sqcovz2 + CovD2 {9, 10}, Mat2 {9, 0, 0, 10}));
  static_assert(is_diagonal_v<decltype((SqCovD2 {4, 5} + CovD2 {9, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((sqcovi2 + CovD2 {9, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((sqcovz2 + CovD2 {9, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + covi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + covi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + covi2, Mat2 {5, 0, 0, 6}));
  EXPECT_TRUE(is_near(sqcovi2 + covi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(sqcovz2 + covi2, Mat2 {1, 0, 0, 1}));
  static_assert(is_diagonal_v<decltype((SqCovD2 {4, 5} + covi2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((sqcovi2 + covi2).base_matrix())>);
  static_assert(is_identity_v<decltype((sqcovz2 + covi2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} + covz2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} + covz2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} + covz2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} + covz2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} + covz2, Mat2 {4, 0, 0, 5}));
  EXPECT_TRUE(is_near(sqcovi2 + covz2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovz2 + covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((SqCovD2 {4, 5} + covz2).base_matrix())>);
  static_assert(is_identity_v<decltype((sqcovi2 + covz2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 + covz2).base_matrix())>);

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


TEST_F(covariance_tests, SquareRootCovariance_subtraction)
{
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, -1, 1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, -1, 1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - SqCovD2 {2, 2}, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - sqcovi2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} - sqcovz2, Mat2 {3, 0, 1, 3}));
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} - SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} - SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} - SqCovD2 {2, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} - sqcovi2)>);
  static_assert(is_lower_triangular_v<decltype(SqCovSA2l {3, 0, 1, 3} - sqcovz2)>);

  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 1, -1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 1, -1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - SqCovD2 {2, 2}, Mat2 {1, 1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - sqcovi2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} - sqcovz2, Mat2 {3, 1, 0, 3}));
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} - SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} - SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} - SqCovD2 {2, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} - sqcovi2)>);
  static_assert(is_upper_triangular_v<decltype(SqCovSA2u {3, 1, 0, 3} - sqcovz2)>);

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, -1, 1, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, -1, 1, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - SqCovD2 {2, 2}, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - sqcovi2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} - sqcovz2, Mat2 {3, 0, 1, 3}));
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} - SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} - SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} - SqCovD2 {2, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} - sqcovi2)>);
  static_assert(is_lower_triangular_v<decltype(SqCovT2l {3, 0, 1, 3} - sqcovz2)>);

  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 1, -1, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 1, -1, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - SqCovD2 {2, 2}, Mat2 {1, 1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - sqcovi2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} - sqcovz2, Mat2 {3, 1, 0, 3}));
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} - SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} - SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} - SqCovD2 {2, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} - sqcovi2)>);
  static_assert(is_upper_triangular_v<decltype(SqCovT2u {3, 1, 0, 3} - sqcovz2)>);

  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovSA2l {2, 0, 1, 2}, Mat2 {1, 0, -1, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovSA2u {2, 1, 0, 2}, Mat2 {1, -1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovT2l {2, 0, 1, 2}, Mat2 {1, 0, -1, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovT2u {2, 1, 0, 2}, Mat2 {1, -1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - SqCovD2 {2, 2}, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - sqcovi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3} - sqcovz2, Mat2 {3, 0, 0, 3}));
  static_assert(is_lower_triangular_v<decltype(SqCovD2 {3, 3} - SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovD2 {3, 3} - SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_lower_triangular_v<decltype(SqCovD2 {3, 3} - SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(SqCovD2 {3, 3} - SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_diagonal_v<decltype(SqCovD2 {3, 3} - SqCovD2 {2, 2})>);
  static_assert(is_diagonal_v<decltype(SqCovD2 {3, 3} - sqcovi2)>);
  static_assert(is_diagonal_v<decltype(SqCovD2 {3, 3} - sqcovz2)>);

  EXPECT_TRUE(is_near(sqcovi2 - SqCovSA2l {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - SqCovSA2u {2, 1, 0, 2}, Mat2 {-1, -1, 0, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - SqCovT2l {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - SqCovT2u {2, 1, 0, 2}, Mat2 {-1, -1, 0, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - SqCovD2 {2, 2}, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(sqcovi2 - sqcovi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovi2 - sqcovz2, Mat2 {1, 0, 0, 1}));
  static_assert(is_lower_triangular_v<decltype(sqcovi2 - SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(sqcovi2 - SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_lower_triangular_v<decltype(sqcovi2 - SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(sqcovi2 - SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_diagonal_v<decltype(sqcovi2 - SqCovD2 {2, 2})>);
  static_assert(is_zero_v<decltype(sqcovi2 - sqcovi2)>);
  static_assert(is_identity_v<decltype(sqcovi2 - sqcovz2)>);

  EXPECT_TRUE(is_near(sqcovz2 - SqCovSA2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - SqCovSA2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - SqCovT2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - SqCovT2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - SqCovD2 {2, 2}, Mat2 {-2, 0, 0, -2}));
  EXPECT_TRUE(is_near(sqcovz2 - sqcovi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(sqcovz2 - sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype(sqcovz2 - SqCovSA2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(sqcovz2 - SqCovSA2u {2, 1, 0, 2})>);
  static_assert(is_lower_triangular_v<decltype(sqcovz2 - SqCovT2l {2, 0, 1, 2})>);
  static_assert(is_upper_triangular_v<decltype(sqcovz2 - SqCovT2u {2, 1, 0, 2})>);
  static_assert(is_diagonal_v<decltype(sqcovz2 - SqCovD2 {2, 2})>);
  static_assert(is_diagonal_v<decltype(sqcovz2 - sqcovi2)>);
  static_assert(is_zero_v<decltype(sqcovz2 - sqcovz2)>);}


TEST_F(covariance_tests, SquareRootCovariance_subtraction_mixed)
{
  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovSA2l {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovSA2l {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovSA2l {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovSA2l {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovSA2l {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovSA2l {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovSA2l {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((SqCovD2 {4, 5} - CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovi2 - CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovz2 - CovSA2l {9, 3, 3, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovSA2u {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovSA2u {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovSA2u {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovSA2u {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovSA2u {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovSA2u {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovSA2u {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((SqCovD2 {4, 5} - CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovi2 - CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovz2 - CovSA2u {9, 3, 3, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovT2l {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovT2l {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovT2l {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovT2l {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovT2l {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovT2l {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovT2l {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((SqCovD2 {4, 5} - CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovi2 - CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovz2 - CovT2l {9, 3, 3, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovT2u {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovT2u {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovT2u {9, 3, 3, 10}, -Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovT2u {9, 3, 3, 10}, -Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovT2u {9, 3, 3, 10}, -Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovT2u {9, 3, 3, 10}, -Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovT2u {9, 3, 3, 10}, -Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((SqCovD2 {4, 5} - CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovi2 - CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovz2 - CovT2u {9, 3, 3, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - CovD2 {9, 10}, Mat2 {-7, 0, 1, -8}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - CovD2 {9, 10}, Mat2 {-7, 1, 0, -8}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - CovD2 {9, 10}, Mat2 {-7, 0, 1, -8}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - CovD2 {9, 10}, Mat2 {-7, 1, 0, -8}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - CovD2 {9, 10}, Mat2 {-5, 0, 0, -5}));
  EXPECT_TRUE(is_near(sqcovi2 - CovD2 {9, 10}, Mat2 {-8, 0, 0, -9}));
  EXPECT_TRUE(is_near(sqcovz2 - CovD2 {9, 10}, Mat2 {-9, 0, 0, -10}));
  static_assert(is_diagonal_v<decltype((SqCovD2 {4, 5} - CovD2 {9, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((sqcovi2 - CovD2 {9, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((sqcovz2 - CovD2 {9, 10}).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - covi2, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - covi2, Mat2 {1, 1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - covi2, Mat2 {1, 0, 1, 1}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - covi2, Mat2 {1, 1, 0, 1}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - covi2, Mat2 {3, 0, 0, 4}));
  EXPECT_TRUE(is_near(sqcovi2 - covi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 - covi2, Mat2 {-1, 0, 0, -1}));
  static_assert(is_diagonal_v<decltype((SqCovD2 {4, 5} - covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovi2 - covi2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((sqcovz2 - covi2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {2, 0, 1, 2} - covz2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovSA2u {2, 1, 0, 2} - covz2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovT2l {2, 0, 1, 2} - covz2, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(SqCovT2u {2, 1, 0, 2} - covz2, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(SqCovD2 {4, 5} - covz2, Mat2 {4, 0, 0, 5}));
  EXPECT_TRUE(is_near(sqcovi2 - covz2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovz2 - covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((SqCovD2 {4, 5} - covz2).base_matrix())>);
  static_assert(is_identity_v<decltype((sqcovi2 - covz2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 - covz2).base_matrix())>);

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


TEST_F(covariance_tests, SquareRootCovariance_mult_covariance)
{
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovSA2l {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovSA2u {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovT2l {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovT2u {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * CovD2 {9, 10}, Mat2 {27, 0, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((SqCovSA2l {3, 0, 1, 3} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovSA2l {3, 0, 1, 3} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovSA2l {3, 0, 1, 3}, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovSA2u {3, 1, 0, 3}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovT2l {3, 0, 1, 3}, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovT2u {3, 1, 0, 3}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * SqCovD2 {3, 3}, Mat2 {9, 0, 3, 9}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * sqcovi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((SqCovSA2l {3, 0, 1, 3} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovSA2l {3, 0, 1, 3} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovSA2l {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovSA2u {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovT2l {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovT2u {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * CovD2 {9, 10}, Mat2 {27, 10, 0, 30}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * covi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((SqCovSA2u {3, 1, 0, 3} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovSA2u {3, 1, 0, 3} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovSA2l {3, 0, 1, 3}, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovSA2u {3, 1, 0, 3}, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovT2l {3, 0, 1, 3}, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovT2u {3, 1, 0, 3}, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * SqCovD2 {3, 3}, Mat2 {9, 3, 0, 9}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * sqcovi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((SqCovSA2u {3, 1, 0, 3} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovSA2u {3, 1, 0, 3} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovSA2l {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovSA2u {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovT2l {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovT2u {9, 3, 3, 10}, Mat2 {27, 9, 18, 33}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * CovD2 {9, 10}, Mat2 {27, 0, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * covi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype((SqCovT2l {3, 0, 1, 3} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovT2l {3, 0, 1, 3} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovSA2l {3, 0, 1, 3}, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovSA2u {3, 1, 0, 3}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovT2l {3, 0, 1, 3}, Mat2 {9, 0, 6, 9}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovT2u {3, 1, 0, 3}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * SqCovD2 {3, 3}, Mat2 {9, 0, 3, 9}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * sqcovi2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype((SqCovT2l {3, 0, 1, 3} * SqCovT2l {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((SqCovT2l {3, 0, 1, 3} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovT2l {3, 0, 1, 3} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovSA2l {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovSA2u {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovT2l {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovT2u {9, 3, 3, 10}, Mat2 {30, 19, 9, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * CovD2 {9, 10}, Mat2 {27, 10, 0, 30}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * covi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_upper_triangular_v<decltype((SqCovT2u {3, 1, 0, 3} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovT2u {3, 1, 0, 3} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovSA2l {3, 0, 1, 3}, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovSA2u {3, 1, 0, 3}, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovT2l {3, 0, 1, 3}, Mat2 {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovT2u {3, 1, 0, 3}, Mat2 {9, 6, 0, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * SqCovD2 {3, 3}, Mat2 {9, 3, 0, 9}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * sqcovi2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_upper_triangular_v<decltype((SqCovT2u {3, 1, 0, 3} * SqCovT2u {3, 1, 0, 3}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((SqCovT2u {3, 1, 0, 3} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovT2u {3, 1, 0, 3} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovT2l {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovT2u {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * CovD2 {9, 10}, Mat2 {81, 0, 0, 100}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * covi2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((SqCovD2 {9, 10} * CovD2 {9, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((SqCovD2 {9, 10} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovD2 {9, 10} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {27, 0, 10, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 9, 0, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {27, 0, 10, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 9, 0, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * SqCovD2 {3, 3}, Mat2 {27, 0, 0, 30}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * sqcovi2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(SqCovD2 {9, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((SqCovD2 {9, 10} * SqCovD2 {9, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((SqCovD2 {9, 10} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((SqCovD2 {9, 10} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(sqcovi2 * CovSA2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * CovSA2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * CovT2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * CovT2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * CovD2 {9, 10}, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(sqcovi2 * covi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovi2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovi2 * CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovi2 * CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((sqcovi2 * CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((sqcovi2 * CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((sqcovi2 * CovD2 {9, 10}).base_matrix())>);
  static_assert(is_identity_v<decltype((sqcovi2 * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovi2 * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(sqcovi2 * SqCovSA2l {3, 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * SqCovSA2u {3, 1, 0, 3}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * SqCovT2l {3, 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * SqCovT2u {3, 1, 0, 3}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * SqCovD2 {3, 3}, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(sqcovi2 * sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovi2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqcovi2 * SqCovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqcovi2 * SqCovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((sqcovi2 * SqCovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((sqcovi2 * SqCovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((sqcovi2 * SqCovD2 {9, 10}).base_matrix())>);
  static_assert(is_identity_v<decltype((sqcovi2 * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovi2 * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(sqcovz2 * CovSA2l {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * CovSA2u {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * CovT2l {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * CovT2u {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * CovD2 {9, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * covi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((sqcovz2 * CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * CovD2 {9, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(sqcovz2 * SqCovSA2l {3, 0, 1, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * SqCovSA2u {3, 1, 0, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * SqCovT2l {3, 0, 1, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * SqCovT2u {3, 1, 0, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * SqCovD2 {3, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * sqcovi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((sqcovz2 * SqCovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * SqCovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * SqCovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * SqCovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * SqCovD2 {9, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * sqcovz2).base_matrix())>);
}


TEST_F(covariance_tests, SquareRootCovariance_mult_TypedMatrix)
{
  using MatI2 = TypedMatrix<C, C, I2>;
  using MatZ2 = TypedMatrix<C, C, Z2>;
  auto mati2 = MatI2(i2);
  auto matz2 = MatZ2(z2);
  using Cx = Coefficients<Axis, Angle>;
  using MatI2x = TypedMatrix<C, Cx, I2>;
  auto mati2x = MatI2x(i2);

  auto sqCovSA2l = SqCovSA2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(sqCovSA2l * Mat2 {4, 2, 2, 5}, Mat2 {12, 6, 10, 17}));
  EXPECT_TRUE(is_near(sqCovSA2l * mati2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqCovSA2l * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((sqCovSA2l * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqCovSA2l * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovSA2l * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovSA2l * mati2x)>::ColumnCoefficients, Cx>);

  auto sqCovSA2u = SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(sqCovSA2u * Mat2 {4, 2, 2, 5}, Mat2 {14, 11, 6, 15}));
  EXPECT_TRUE(is_near(sqCovSA2u * mati2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqCovSA2u * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((sqCovSA2u * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqCovSA2u * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovSA2u * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovSA2u * mati2x)>::ColumnCoefficients, Cx>);

  auto sqCovT2l = SqCovT2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(sqCovT2l * Mat2 {4, 2, 2, 5}, Mat2 {12, 6, 10, 17}));
  EXPECT_TRUE(is_near(sqCovT2l * mati2, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqCovT2l * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype((sqCovT2l * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqCovT2l * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovT2l * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovT2l * mati2x)>::ColumnCoefficients, Cx>);

  auto sqCovT2u = SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(sqCovT2u * Mat2 {4, 2, 2, 5}, Mat2 {14, 11, 6, 15}));
  EXPECT_TRUE(is_near(sqCovT2u * mati2, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(sqCovT2u * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_upper_triangular_v<decltype((sqCovT2u * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqCovT2u * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovT2u * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovT2u * mati2x)>::ColumnCoefficients, Cx>);

  auto sqCovD2 = SqCovD2 {3, 3};
  EXPECT_TRUE(is_near(sqCovD2 * Mat2 {4, 2, 2, 5}, Mat2 {12, 6, 6, 15}));
  EXPECT_TRUE(is_near(sqCovD2 * mati2, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(sqCovD2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((sqCovD2 * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqCovD2 * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovD2 * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqCovD2 * mati2x)>::ColumnCoefficients, Cx>);

  EXPECT_TRUE(is_near(sqcovi2 * Mat2 {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(sqcovi2 * mati2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(sqcovi2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_identity_v<decltype((sqcovi2 * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovi2 * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqcovi2 * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqcovi2 * mati2x)>::ColumnCoefficients, Cx>);

  EXPECT_TRUE(is_near(sqcovz2 * Mat2 {4, 2, 2, 5}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * mati2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(sqcovz2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((sqcovz2 * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((sqcovz2 * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqcovz2 * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(sqcovz2 * mati2x)>::ColumnCoefficients, Cx>);

  // Scale
  TypedMatrix<Coefficients<Angle, Axis, Angle>, C> a1 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(scale(SqCovSA2l {2, 0, 1, 2}, 2), Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(scale(SqCovSA2u {2, 1, 0, 2}, 2), Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(scale(SqCovT2l {2, 0, 1, 2}, 2), Mat2 {4, 0, 2, 4}));
  EXPECT_TRUE(is_near(scale(SqCovT2u {2, 1, 0, 2}, 2), Mat2 {4, 2, 0, 4}));
  EXPECT_TRUE(is_near(scale(SqCovD2 {1, 2}, 2), Mat2 {2, 0, 0, 4}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(SqCovSA2l {2, 0, 1, 2}, 2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(scale(SqCovSA2u {2, 1, 0, 2}, 2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(scale(SqCovT2l {2, 0, 1, 2}, 2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(scale(SqCovT2u {2, 1, 0, 2}, 2).base_matrix())>);
  static_assert(is_diagonal_v<decltype(scale(SqCovD2 {1, 2}, 2).base_matrix())>);

  EXPECT_TRUE(is_near(inverse_scale(SqCovSA2l {2, 0, 1, 2}, 2), Mat2 {1, 0, 0.5, 1}));
  EXPECT_TRUE(is_near(inverse_scale(SqCovSA2u {2, 1, 0, 2}, 2), Mat2 {1, 0.5, 0, 1}));
  EXPECT_TRUE(is_near(inverse_scale(SqCovT2l {2, 0, 1, 2}, 2), Mat2 {1, 0, 0.5, 1}));
  EXPECT_TRUE(is_near(inverse_scale(SqCovT2u {2, 1, 0, 2}, 2), Mat2 {1, 0.5, 0, 1}));
  EXPECT_TRUE(is_near(inverse_scale(SqCovD2 {2, 4}, 2), Mat2 {1, 0, 0, 2}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(inverse_scale(SqCovSA2l {2, 0, 1, 2}, 2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(inverse_scale(SqCovSA2u {2, 1, 0, 2}, 2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(inverse_scale(SqCovT2l {2, 0, 1, 2}, 2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(inverse_scale(SqCovT2u {2, 1, 0, 2}, 2).base_matrix())>);
  static_assert(is_diagonal_v<decltype(inverse_scale(SqCovD2 {1, 2}, 2).base_matrix())>);

  // Rank-deficient case
  using M3 = Eigen::Matrix<double, 3, 3>;
  using Mat3 = TypedMatrix<Coefficients<Angle, Axis, Angle>, Coefficients<Angle, Axis, Angle>, M3>;
  EXPECT_TRUE(is_near(square(scale(SqCovSA2l {2, 0, 1, 2}, a1)), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(square(scale(SqCovSA2u {2, 1, 0, 2}, a1)), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(square(scale(SqCovT2l {2, 0, 1, 2}, a1)), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(square(scale(SqCovT2u {2, 1, 0, 2}, a1)), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(square(scale(SqCovD2 {2, 3}, a1)), Mat3 {40, 84, 128, 84, 180, 276, 128, 276, 424}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(SqCovSA2l {2, 0, 1, 2}, a1).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(scale(SqCovSA2u {2, 1, 0, 2}, a1).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(scale(SqCovT2l {2, 0, 1, 2}, a1).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(scale(SqCovT2u {2, 1, 0, 2}, a1).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(SqCovD2 {1, 2}, a1).base_matrix())>);

  // Rank-sufficient case
  using SqCovSA3l = SquareRootCovariance<Coefficients<Angle, Axis, Angle>, EigenSelfAdjointMatrix<M3, TriangleType::lower>>;
  using SqCovSA3u = SquareRootCovariance<Coefficients<Angle, Axis, Angle>, EigenSelfAdjointMatrix<M3, TriangleType::upper>>;
  using SqCovT3l = SquareRootCovariance<Coefficients<Angle, Axis, Angle>, EigenTriangularMatrix<M3, TriangleType::lower>>;
  using SqCovT3u = SquareRootCovariance<Coefficients<Angle, Axis, Angle>, EigenTriangularMatrix<M3, TriangleType::upper>>;
  using SqCovD3 = SquareRootCovariance<Coefficients<Angle, Axis, Angle>, EigenDiagonal<Eigen::Matrix<double, 3, 1>>>;
  Mat3 q1l {4, 0, 0,
            2, 5, 0,
            2, 3, 6};
  Mat3 q1u {4, 2, 2,
            0, 5, 3,
            0, 0, 6};
  TypedMatrix<C, Coefficients<Angle, Axis, Angle>> b1 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(square(scale(SqCovSA3l(q1l), b1)), Mat2 {881, 1997, 1997, 4589}));
  EXPECT_TRUE(is_near(square(scale(SqCovSA3u(q1u), b1)), Mat2 {881, 1997, 1997, 4589}));
  EXPECT_TRUE(is_near(square(scale(SqCovT3l(q1l), b1)), Mat2 {881, 1997, 1997, 4589}));
  EXPECT_TRUE(is_near(square(scale(SqCovT3u(q1u), b1)), Mat2 {881, 1997, 1997, 4589}));
  EXPECT_TRUE(is_near(square(scale(SqCovD3 {4, 5, 6}, b1)), Mat2 {440, 962, 962, 2177}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(SqCovSA3l(q1l), b1).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(scale(SqCovSA3u(q1u), b1).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(scale(SqCovT3l(q1l), b1).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(scale(SqCovT3u(q1u), b1).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(SqCovD3 {4, 5, 6}, b1).base_matrix())>);
}


TEST_F(covariance_tests, TypedMatrix_mult_SquareRootCovariance)
{
  using MatI2 = TypedMatrix<C, C, I2>;
  using MatZ2 = TypedMatrix<C, C, Z2>;
  auto mati2 = MatI2(i2);
  auto matz2 = MatZ2(z2);
  using Cx = Coefficients<Axis, Angle>;
  using MatI2x = TypedMatrix<Cx, C, I2>;
  auto mati2x = MatI2x(i2);

  auto sqCovSA2l = SqCovSA2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovSA2l, Mat2 {14, 6, 11, 15}));
  EXPECT_TRUE(is_near(mati2 * sqCovSA2l, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovSA2l, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((mati2 * sqCovSA2l).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * sqCovSA2l).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovSA2l)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovSA2l)>::ColumnCoefficients, C>);

  auto sqCovSA2u = SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovSA2u, Mat2 {12, 10, 6, 17}));
  EXPECT_TRUE(is_near(mati2 * sqCovSA2u, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovSA2u, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((mati2 * sqCovSA2u).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * sqCovSA2u).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovSA2u)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovSA2u)>::ColumnCoefficients, C>);

  auto sqCovT2l = SqCovT2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovT2l, Mat2 {14, 6, 11, 15}));
  EXPECT_TRUE(is_near(mati2 * sqCovT2l, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovT2l, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype((mati2 * sqCovT2l).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * sqCovT2l).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovT2l)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovT2l)>::ColumnCoefficients, C>);

  auto sqCovT2u = SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovT2u, Mat2 {12, 10, 6, 17}));
  EXPECT_TRUE(is_near(mati2 * sqCovT2u, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovT2u, Mat2 {0, 0, 0, 0}));
  static_assert(is_upper_triangular_v<decltype((mati2 * sqCovT2u).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * sqCovT2u).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovT2u)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovT2u)>::ColumnCoefficients, C>);

  auto sqCovD2 = SqCovD2 {3, 3};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqCovD2, Mat2 {12, 6, 6, 15}));
  EXPECT_TRUE(is_near(mati2 * sqCovD2, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(matz2 * sqCovD2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((mati2 * sqCovD2).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * sqCovD2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovD2)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqCovD2)>::ColumnCoefficients, C>);

  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqcovi2, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(mati2 * sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(matz2 * sqcovi2, Mat2 {0, 0, 0, 0}));
  static_assert(is_identity_v<decltype((mati2 * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * sqcovi2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqcovi2)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqcovi2)>::ColumnCoefficients, C>);

  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * sqcovz2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(mati2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(matz2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((mati2 * sqcovz2).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * sqcovz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqcovz2)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * sqcovz2)>::ColumnCoefficients, C>);
}


TEST_F(covariance_tests, SquareRootCovariance_other_operations)
{
  EXPECT_TRUE(is_near(-SqCovT2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(-SqCovT2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(-SqCovD2 {4, 5}, Mat2 {-4, 0, 0, -5}));
  EXPECT_TRUE(is_near(-sqcovi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(-sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((-sqcovz2).base_matrix())>);

  EXPECT_TRUE((SqCovT2l {3, 0, 1, 3} == SqCovT2l {3, 0, 1, 3}));
  EXPECT_TRUE((SqCovT2l {3, 0, 1, 3} == SqCovSA2l {3, 0, 1, 3}));
  EXPECT_TRUE((SqCovT2u {3, 1, 0, 3} != SqCovT2u {3, 2, 0, 3}));
  EXPECT_TRUE((SqCovT2l {3, 0, 1, 3} != SquareRootCovariance<Axes<2>, T2l> {3, 0, 1, 3}));
}


TEST_F(covariance_tests, SquareRootCovariance_references_triangular)
{
  using V = SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 0, 0,
        2, 4, 0,
        3, -6, -3};
  v1(0, 1) = 3.2;
  EXPECT_EQ(v1(0,1), 0); // Assigning to the upper right triangle does not change anything. It's still zero.
  EXPECT_EQ(v1(1,0), 2);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(2, 0) = 5.2;
  EXPECT_EQ(v1(2,0), 5.2);
  v1 = {1.4, 0, 0,
        2.4, 4.4, 0,
        3.4, -6.4, -3.4};
  EXPECT_NEAR(v2(1,0), 2.4, 1e-6);
  v2 = V {1.1, 0, 0,
          2.1, 4.1, 0,
          3.1, -6.1, -3.1};
  EXPECT_NEAR(v1(2,0), 3.1, 1e-6);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&&>> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 2.1);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<const Eigen::Matrix<double, 3, 3>&>> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  v3 = V {1.2, 0, 0,
          2.2, 4.2, 0,
          3.2, -6.2, -3.2};
  EXPECT_NEAR(v4(2,1), -6.2, 1e-6);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4.2);
  v5(1,1) = 8.5;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 8.5);
  v5 = {1.6, 0, 0,
        2.6, 4.6, 0,
        3.6, -6.6, -3.6};
  EXPECT_NEAR(v3(1,1), 8.4, 1e-6);
  EXPECT_NEAR(v5(1,1), 4.6, 1e-6);
}


TEST_F(covariance_tests, SquareRootCovariance_references_triangular2)
{
  using V = SquareRootCovariance<Coefficients<Angle, Axis, Axis>,
    EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::lower>>;
  V v1 {1., 0, 0,
        2, 4, 0,
        3, -6, -3};
  v1(0, 1) = 3.2;
  EXPECT_EQ(v1(0,1), 0); // Assigning to the upper right triangle does not change anything. It's still zero.
  EXPECT_EQ(v1(1,0), 2);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(2, 0) = 5.2;
  EXPECT_EQ(v1(2,0), 5.2);
  v1 = {1.4, 0, 0,
        2.4, 4.4, 0,
        3.4, -6.4, -3.4};
  EXPECT_NEAR(v2(1,0), 2.4, 1e-6);
  v2 = V {1.1, 0, 0,
          2.1, 4.1, 0,
          3.1, -6.1, -3.1};
  EXPECT_NEAR(v1(2,0), 3.1, 1e-6);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&&> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 2.1);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, const EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  EXPECT_TRUE(is_near(v4, V {
    1.1, 0, 0,
    2.1, 4.1, 0,
    3.1, 7.3, -3.1}));
  v3 = V {1.2, 0, 0,
          2.2, 4.2, 0,
          3.2, -6.2, -3.2};
  EXPECT_NEAR(v4(2,1), -6.2, 1e-6);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4.2);
  v5(1,1) = 8.5;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 8.5);
  v5 = {1.6, 0, 0,
        2.6, 4.6, 0,
        3.6, -6.6, -3.6};
  EXPECT_NEAR(v3(1,1), 8.4, 1e-6);
  EXPECT_NEAR(v5(1,1), 4.6, 1e-6);
}


TEST_F(covariance_tests, SquareRootCovariance_references_self_adjoint)
{
  using V = SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1, 2, 3,
        2, 4, -6,
        3, -6, -3};
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1 = {1.4, 0, 0,
        2.4, 4.4, 0,
        3.4, -6.4, -3.4};
  EXPECT_NEAR(v2(1,0), 2.4, 1e-6);
  v2 = V {1.1, 2.1, 3.1,
          2.1, 4.1, -6.1,
          3.1, -6.1, -3.1};
  EXPECT_NEAR(v1(2,0), 3.1, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>&&>> v3 = std::move(v2);
  EXPECT_NEAR(v3(1,0), 2.1, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<const Eigen::Matrix<double, 3, 3>&>> v4 = v3;
  v3 = V {1.2, 2.2, 3.2,
          2.2, 4.2, -6.2,
          3.2, -6.2, -3.2};
  EXPECT_NEAR(v4(2,1), -6.2, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v5 = {1.3, 2.3, 3.3,
        2.3, 4.3, -6.3,
        3.3, -6.3, -3.3};
  EXPECT_NEAR(v3(1,1), 4.2, 1e-6);
  EXPECT_NEAR(v5(1,1), 4.3, 1e-6);
}


TEST_F(covariance_tests, SquareRootCovariance_references_self_adjoint2)
{
  using V = SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 4, -6,
        3, -6, -3};
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1 = {1.4, 2.4, 3.4,
        2.4, 4.4, -6.4,
        3.4, -6.4, -3.4};
  EXPECT_NEAR(v2(1,0), 2.4, 1e-6);
  v2 = V {1.1, 2.1, 3.1,
          2.1, 4.1, -6.1,
          3.1, -6.1, -3.1};
  EXPECT_NEAR(v1(2,0), 3.1, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&&> v3 = std::move(v2);
  EXPECT_NEAR(v3(1,0), 2.1, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, const EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&> v4 = v3;
  v3 = V {1.2, 2.2, 3.2,
          2.2, 4.2, -6.2,
          3.2, -6.2, -3.2};
  EXPECT_NEAR(v4(2,1), -6.2, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v5 = {1.3, 2.3, 3.3,
        2.3, 4.3, -6.3,
        3.3, -6.3, -3.3};
  EXPECT_NEAR(v3(1,1), 4.2, 1e-6);
  EXPECT_NEAR(v5(1,1), 4.3, 1e-6);
}

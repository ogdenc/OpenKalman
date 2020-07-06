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
using I2 = EigenIdentity<M2>;
using Z2 = EigenZero<M2>;
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
inline Z2 z2 = EigenZero<M2>();
inline auto covi2 = CovI2(i2);
inline auto covz2 = CovZ2(z2);
inline auto sqcovi2 = SqCovI2(i2);
inline auto sqcovz2 = SqCovZ2(z2);


TEST_F(covariance_tests, Covariance_class)
{
  // Default constructor and Eigen3 construction
  CovSA2l clsa1;
  clsa1 << 9, 3, 3, 10;
  EXPECT_TRUE(is_near(clsa1, Mat2 {9, 3, 3, 10}));
  CovSA2u cusa1;
  cusa1 << 4, 2, 2, 5;
  EXPECT_TRUE(is_near(cusa1, Mat2 {4, 2, 2, 5}));
  CovT2l clt1;
  clt1 << 9, 3, 3, 10;
  EXPECT_TRUE(is_near(clt1, Mat2 {9, 3, 3, 10}));
  CovT2u cut1;
  cut1 << 4, 2, 2, 5;
  EXPECT_TRUE(is_near(cut1, Mat2 {4, 2, 2, 5}));
  CovD2 cd1;
  cd1 << 1, 2;
  EXPECT_TRUE(is_near(cd1, Mat2 {1, 0, 0, 2}));
  CovI2 ci1 = i2;
  EXPECT_TRUE(is_near(ci1, Mat2 {1, 0, 0, 1}));

  // Copy constructor
  CovSA2l clsa2 = const_cast<const CovSA2l&>(clsa1);
  EXPECT_TRUE(is_near(clsa2, Mat2 {9, 3, 3, 10}));
  CovSA2u cusa2 = const_cast<const CovSA2u&>(cusa1);
  EXPECT_TRUE(is_near(cusa2, Mat2 {4, 2, 2, 5}));
  CovT2l clt2 = const_cast<const CovT2l&>(clt1);
  EXPECT_TRUE(is_near(clt2, Mat2 {9, 3, 3, 10}));
  CovT2u cut2 = const_cast<const CovT2u&>(cut1);
  EXPECT_TRUE(is_near(cut2, Mat2 {4, 2, 2, 5}));
  CovD2 cd2 = const_cast<const CovD2&>(cd1);
  EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));
  CovD2 cd2b = cd1; // Template constructor, not copy constructor.
  EXPECT_TRUE(is_near(cd2b, Mat2 {1, 0, 0, 2}));
  CovI2 ci2 = const_cast<const CovI2&>(ci1);
  EXPECT_TRUE(is_near(ci2, Mat2 {1, 0, 0, 1}));
  CovI2 ci2b = ci1; // Template constructor, not copy constructor.
  EXPECT_TRUE(is_near(ci2b, Mat2 {1, 0, 0, 1}));

  // Move constructor
  CovSA2l clsa3(std::move(CovSA2l {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(clsa3, Mat2 {9, 3, 3, 10}));
  CovSA2u cusa3(std::move(CovSA2u {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(cusa3, Mat2 {4, 2, 2, 5}));
  CovT2l clt3(std::move(CovT2l {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(clt3, Mat2 {9, 3, 3, 10}));
  CovT2u cut3(std::move(CovT2u {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(cut3, Mat2 {4, 2, 2, 5}));
  CovD2 cd3(std::move(CovD2 {1, 2}));
  EXPECT_TRUE(is_near(cd3, Mat2 {1, 0, 0, 2}));
  CovI2 ci3(std::move(CovI2 {i2}));
  EXPECT_TRUE(is_near(ci3, Mat2 {1, 0, 0, 1}));

  // Convert from different covariance type
  CovSA2l clsasa4(CovSA2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsasa4, Mat2 {9, 3, 3, 10}));
  CovSA2u cusasa4(CovSA2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusasa4, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat4(CovT2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsat4, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat4(CovT2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusat4, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat4s(CovT2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsat4s, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat4s(CovT2u {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusat4s, Mat2 {4, 2, 2, 5}));
  CovT2l cltt4(CovT2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltt4, Mat2 {9, 3, 3, 10}));
  CovT2u cutt4(CovT2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutt4, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa4(CovSA2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltsa4, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa4(CovSA2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutsa4, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa4s(CovSA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltsa4s, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa4s(CovSA2u {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutsa4s, Mat2 {4, 2, 2, 5}));
  CovSA2l clsa4d(CovD2 {1, 2});
  EXPECT_TRUE(is_near(clsa4d, Mat2 {1, 0, 0, 2}));
  CovSA2u cusa4d(CovD2 {1, 2});
  EXPECT_TRUE(is_near(cusa4d, Mat2 {1, 0, 0, 2}));
  CovT2l clt4d(CovD2 {1, 2});
  EXPECT_TRUE(is_near(clt4d, Mat2 {1, 0, 0, 2}));
  CovT2u cut4d(CovD2 {1, 2});
  EXPECT_TRUE(is_near(cut4d, Mat2 {1, 0, 0, 2}));
  CovT2u cut4i(CovI2 {i2});
  EXPECT_TRUE(is_near(cut4i, Mat2 {1, 0, 0, 1}));

  // Convert from different square-root covariance type
  CovSA2l clsasa4X(SqCovSA2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(clsasa4X, Mat2 {9, 3, 3, 10}));
  CovSA2u cusasa4X(SqCovSA2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cusasa4X, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat4X(SqCovT2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(clsat4X, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat4X(SqCovT2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cusat4X, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat4sX(SqCovT2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(clsat4sX, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat4sX(SqCovT2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cusat4sX, Mat2 {4, 2, 2, 5}));
  CovT2l cltt4X(SqCovT2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(cltt4X, Mat2 {9, 3, 3, 10}));
  CovT2u cutt4X(SqCovT2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cutt4X, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa4X(SqCovSA2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(cltsa4X, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa4X(SqCovSA2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cutsa4X, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa4sX(SqCovSA2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(cltsa4sX, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa4sX(SqCovSA2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cutsa4sX, Mat2 {4, 2, 2, 5}));
  CovSA2l clsa4dX(SqCovD2 {2, 3});
  EXPECT_TRUE(is_near(clsa4dX, Mat2 {4, 0, 0, 9}));
  CovSA2u cusa4dX(SqCovD2 {2, 3});
  EXPECT_TRUE(is_near(cusa4dX, Mat2 {4, 0, 0, 9}));
  CovT2l clt4dX(SqCovD2 {2, 3});
  EXPECT_TRUE(is_near(clt4dX, Mat2 {4, 0, 0, 9}));
  CovT2u cut4dX(SqCovD2 {2, 3});
  EXPECT_TRUE(is_near(cut4dX, Mat2 {4, 0, 0, 9}));
  CovT2u cut4iX(SqCovI2 {i2});
  EXPECT_TRUE(is_near(cut4iX, Mat2 {1, 0, 0, 1}));

  // Construct from a covariance base
  CovSA2l clsasa5(SA2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsasa5, Mat2 {9, 3, 3, 10}));
  CovSA2u cusasa5(SA2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusasa5, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat5(T2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(clsat5, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat5(T2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cusat5, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat5s(T2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(clsat5s, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat5s(T2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cusat5s, Mat2 {4, 2, 2, 5}));
  CovT2l cltt5(T2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(cltt5, Mat2 {9, 3, 3, 10}));
  CovT2u cutt5(T2l {2, 0, 1, 2});
  EXPECT_TRUE(is_near(cutt5, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa5(SA2u {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltsa5, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa5(SA2l {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutsa5, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa5s(SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(cltsa5s, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa5s(SA2u {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cutsa5s, Mat2 {4, 2, 2, 5}));
  CovSA2l clsa5d(D2 {1, 2});
  EXPECT_TRUE(is_near(clsa5d, Mat2 {1, 0, 0, 2}));
  CovSA2u cusa5d(D2 {1, 2});
  EXPECT_TRUE(is_near(cusa5d, Mat2 {1, 0, 0, 2}));
  CovT2l clt5d(D2 {1, 2});
  EXPECT_TRUE(is_near(clt5d, Mat2 {1, 0, 0, 2}));
  CovT2u cut5d(D2 {1, 2});
  EXPECT_TRUE(is_near(cut5d, Mat2 {1, 0, 0, 2}));
  CovT2l clt5i(covi2);
  EXPECT_TRUE(is_near(clt5i, Mat2 {1, 0, 0, 1}));

  // Construct from a typed matrix
  CovSA2l clsa6(Mat2 {9, 7, 3, 10});
  EXPECT_TRUE(is_near(clsa6, Mat2 {9, 3, 3, 10}));
  CovSA2u cusa6(Mat2 {4, 2, 7, 5});
  EXPECT_TRUE(is_near(cusa6, Mat2 {4, 2, 2, 5}));
  CovT2l clt6(Mat2 {9, 7, 3, 10});
  EXPECT_TRUE(is_near(clt6, Mat2 {9, 3, 3, 10}));
  CovT2u cut6(Mat2 {4, 2, 7, 5});
  EXPECT_TRUE(is_near(cut6, Mat2 {4, 2, 2, 5}));
  CovD2 cd6(TypedMatrix<C, Axis, Eigen::Matrix<double, 2, 1>> {1, 2});
  EXPECT_TRUE(is_near(cd6, Mat2 {1, 0, 0, 2}));

  // Construct from a regular matrix
  CovSA2l clsa7((M2() << 9, 7, 3, 10).finished());
  EXPECT_TRUE(is_near(clsa7, Mat2 {9, 3, 3, 10}));
  CovSA2u cusa7((M2() << 4, 2, 7, 5).finished());
  EXPECT_TRUE(is_near(cusa7, Mat2 {4, 2, 2, 5}));
  CovT2l clt7((M2() << 9, 7, 3, 10).finished());
  EXPECT_TRUE(is_near(clt7, Mat2 {9, 3, 3, 10}));
  CovT2u cut7((M2() << 4, 2, 7, 5).finished());
  EXPECT_TRUE(is_near(cut7, Mat2 {4, 2, 2, 5}));
  CovD2 cd7((Eigen::Matrix<double, 2, 1>() << 1, 2).finished());
  EXPECT_TRUE(is_near(cd7, Mat2 {1, 0, 0, 2}));

  // Construct from a list of coefficients
  CovSA2l clsa8{4, 7, 2, 5};
  EXPECT_TRUE(is_near(clsa8, Mat2 {4, 2, 2, 5}));
  CovSA2u cusa8{9, 3, 7, 10};
  EXPECT_TRUE(is_near(cusa8, Mat2 {9, 3, 3, 10}));
  CovT2l clt8{4, 7, 2, 5};
  EXPECT_TRUE(is_near(clt8, Mat2 {4, 2, 2, 5}));
  CovT2u cut8{9, 3, 7, 10};
  EXPECT_TRUE(is_near(cut8, Mat2 {9, 3, 3, 10}));
  CovD2 cd8({1, 2});
  EXPECT_TRUE(is_near(cd8, Mat2 {1, 0, 0, 2}));

  // Copy assignment
  clsa2 = clsa8;
  EXPECT_TRUE(is_near(clsa2, Mat2 {4, 2, 2, 5}));
  cusa2 = cusa8;
  EXPECT_TRUE(is_near(cusa2, Mat2 {9, 3, 3, 10}));
  clt2 = clt8;
  EXPECT_TRUE(is_near(clt2, Mat2 {4, 2, 2, 5}));
  cut2 = cut8;
  EXPECT_TRUE(is_near(cut2, Mat2 {9, 3, 3, 10}));
  cd2 = cd8;
  EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // Move assignment
  clsa2 = std::move(CovSA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clsa2, Mat2 {9, 3, 3, 10}));
  cusa2 = std::move(CovSA2u {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cusa2, Mat2 {4, 2, 2, 5}));
  clt2 = std::move(CovT2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(clt2, Mat2 {9, 3, 3, 10}));
  cut2 = std::move(CovT2u {4, 2, 2, 5});
  EXPECT_TRUE(is_near(cut2, Mat2 {4, 2, 2, 5}));
  cd2 = std::move(CovD2 {1, 2});
  EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // Assign from different covariance type
  clsasa4 = CovSA2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(clsasa4, Mat2 {4, 2, 2, 5}));
  cusasa4 = CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cusasa4, Mat2 {9, 3, 3, 10}));
  clsat4 = CovT2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(clsat4, Mat2 {4, 2, 2, 5}));
  cusat4 = CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cusat4, Mat2 {9, 3, 3, 10}));
  clsat4s = CovT2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(clsat4s, Mat2 {4, 2, 2, 5}));
  cusat4s = CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cusat4s, Mat2 {9, 3, 3, 10}));
  cltt4 = CovT2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cltt4, Mat2 {4, 2, 2, 5}));
  cutt4 = CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cutt4, Mat2 {9, 3, 3, 10}));
  cltsa4 = CovSA2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cltsa4, Mat2 {4, 2, 2, 5}));
  cutsa4 = CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cutsa4, Mat2 {9, 3, 3, 10}));
  cltsa4s = CovSA2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cltsa4s, Mat2 {4, 2, 2, 5}));
  cutsa4s = CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cutsa4s, Mat2 {9, 3, 3, 10}));
  clsa4d = CovD2 {3, 4};
  EXPECT_TRUE(is_near(clsa4d, Mat2 {3, 0, 0, 4}));
  cusa4d = CovD2 {3, 4};
  EXPECT_TRUE(is_near(cusa4d, Mat2 {3, 0, 0, 4}));
  clt4d = CovD2 {3, 4};
  EXPECT_TRUE(is_near(clt4d, Mat2 {3, 0, 0, 4}));
  cut4d = CovD2 {3, 4};
  EXPECT_TRUE(is_near(cut4d, Mat2 {3, 0, 0, 4}));
  cut4i = CovI2 {i2};
  EXPECT_TRUE(is_near(cut4i, Mat2 {1, 0, 0, 1}));

  // Assign from different square-root covariance type
  clsasa5 = SqCovSA2u {2, 1, 0, 2};
  EXPECT_TRUE(is_near(clsasa5, Mat2 {4, 2, 2, 5}));
  cusasa5 = SqCovSA2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(cusasa5, Mat2 {9, 3, 3, 10}));
  clsat5 = SqCovT2u {2, 1, 0, 2};
  EXPECT_TRUE(is_near(clsat5, Mat2 {4, 2, 2, 5}));
  cusat5 = SqCovT2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(cusat5, Mat2 {9, 3, 3, 10}));
  clsat5s = SqCovT2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clsat5s, Mat2 {4, 2, 2, 5}));
  cusat5s = SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cusat5s, Mat2 {9, 3, 3, 10}));
  cltsa5 = SqCovSA2u {2, 1, 0, 2};
  EXPECT_TRUE(is_near(cltsa5, Mat2 {4, 2, 2, 5}));
  cutsa5 = SqCovSA2l {3, 0, 1, 3};
  EXPECT_TRUE(is_near(cutsa5, Mat2 {9, 3, 3, 10}));
  cltsa5s = SqCovSA2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(cltsa5s, Mat2 {4, 2, 2, 5}));
  cutsa5s = SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cutsa5s, Mat2 {9, 3, 3, 10}));
  clsa5d = SqCovD2 {3, 4};
  EXPECT_TRUE(is_near(clsa5d, Mat2 {9, 0, 0, 16}));
  cusa5d = SqCovD2 {3, 4};
  EXPECT_TRUE(is_near(cusa5d, Mat2 {9, 0, 0, 16}));
  clt5d = SqCovD2 {3, 4};
  EXPECT_TRUE(is_near(clt5d, Mat2 {9, 0, 0, 16}));
  cut5d = SqCovD2 {3, 4};
  EXPECT_TRUE(is_near(cut5d, Mat2 {9, 0, 0, 16}));
  clt5i = SqCovI2 {i2};
  EXPECT_TRUE(is_near(clt5i, Mat2 {1, 0, 0, 1}));

  // Assign from a list of coefficients (via move assignment operator)
  clsa8 = {9, 3, 3, 10};
  EXPECT_TRUE(is_near(clsa8, Mat2 {9, 3, 3, 10}));
  cusa8 = {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cusa8, Mat2 {4, 2, 2, 5}));
  clt8 = {9, 3, 3, 10};
  EXPECT_TRUE(is_near(clt8, Mat2 {9, 3, 3, 10}));
  cut8 = {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cut8, Mat2 {4, 2, 2, 5}));
  cd8 = {3, 4};
  EXPECT_TRUE(is_near(cd8, Mat2 {3, 0, 0, 4}));

  // Increment
  clsasa4 += CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(clsasa4, Mat2 {13, 5, 5, 15}));
  cusasa4 += CovSA2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cusasa4, Mat2 {13, 5, 5, 15}));
  clsat4 += CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(clsat4, Mat2 {13, 5, 5, 15}));
  cusat4 += CovT2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cusat4, Mat2 {13, 5, 5, 15}));
  clsat4s += CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(clsat4s, Mat2 {13, 5, 5, 15}));
  cusat4s += CovT2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cusat4s, Mat2 {13, 5, 5, 15}));
  cltt4 += CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cltt4, Mat2 {13, 5, 5, 15}));
  cutt4 += CovT2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cutt4, Mat2 {13, 5, 5, 15}));
  cltsa4 += CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cltsa4, Mat2 {13, 5, 5, 15}));
  cutsa4 += CovSA2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cutsa4, Mat2 {13, 5, 5, 15}));
  cltsa4s += CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cltsa4s, Mat2 {13, 5, 5, 15}));
  cutsa4s += CovSA2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cutsa4s, Mat2 {13, 5, 5, 15}));
  cd8 += CovD2 {1, 2};
  EXPECT_TRUE(is_near(cd8, Mat2 {4, 0, 0, 6}));

  // Decrement
  clsasa4 -= CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(clsasa4, Mat2 {4, 2, 2, 5}));
  cusasa4 -= CovSA2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cusasa4, Mat2 {9, 3, 3, 10}));
  clsat4 -= CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(clsat4, Mat2 {4, 2, 2, 5}));
  cusat4 -= CovT2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cusat4, Mat2 {9, 3, 3, 10}));
  clsat4s -= CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(clsat4s, Mat2 {4, 2, 2, 5}));
  cusat4s -= CovT2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cusat4s, Mat2 {9, 3, 3, 10}));
  cltt4 -= CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cltt4, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(cutt4, Mat2 {13, 5, 5, 15}));
  cutt4 -= CovT2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cutt4, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(cltsa4, Mat2 {13, 5, 5, 15}));
  cltsa4 -= CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cltsa4, Mat2 {4, 2, 2, 5}));
  cutsa4 -= CovSA2l {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cutsa4, Mat2 {9, 3, 3, 10}));
  cltsa4s -= CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(cltsa4s, Mat2 {4, 2, 2, 5}));
  cutsa4s -= CovSA2u {4, 2, 2, 5};
  EXPECT_TRUE(is_near(cutsa4s, Mat2 {9, 3, 3, 10}));
  cd8 -= CovD2 {1, 2};
  EXPECT_TRUE(is_near(cd8, Mat2 {3, 0, 0, 4}));

  // Scalar multiplication, positive
  clsa2 *= 2;
  EXPECT_TRUE(is_near(clsa2, Mat2 {18, 6, 6, 20}));
  cusa2 *= 2;
  EXPECT_TRUE(is_near(cusa2, Mat2 {8, 4, 4, 10}));
  clt2 *= 2;
  EXPECT_TRUE(is_near(clt2, Mat2 {18, 6, 6, 20}));
  cut2 *= 2;
  EXPECT_TRUE(is_near(cut2, Mat2 {8, 4, 4, 10}));
  cd2 *= 2;
  EXPECT_TRUE(is_near(cd2, Mat2 {2, 0, 0, 4}));

  // Scalar division, negative
  clsa2 /= -2;
  EXPECT_TRUE(is_near(clsa2, Mat2 {-9, -3, -3, -10}));
  cusa2 /= -2;
  EXPECT_TRUE(is_near(cusa2, Mat2 {-4, -2, -2, -5}));
  //clt2 /= -2; // These results would result in a runtime error.
  //EXPECT_TRUE(is_near(clt2, Mat2 {-9, -3, -3, -10}));
  //cut2 /= -2;
  //EXPECT_TRUE(is_near(cut2, Mat2 {-4, -2, -2, -5}));
  cd2 /= -2;
  EXPECT_TRUE(is_near(cd2, Mat2 {-1, 0, 0, -2}));

  // Scalar multiplication, negative
  clsa2 *= -2;
  EXPECT_TRUE(is_near(clsa2, Mat2 {18, 6, 6, 20}));
  cusa2 *= -2;
  EXPECT_TRUE(is_near(cusa2, Mat2 {8, 4, 4, 10}));
  //clt2 *= -2;
  //EXPECT_TRUE(is_near(clt2, Mat2 {18, 6, 6, 20}));
  //cut2 *= -2;
  //EXPECT_TRUE(is_near(cut2, Mat2 {8, 4, 4, 10}));
  cd2 *= -2;
  EXPECT_TRUE(is_near(cd2, Mat2 {2, 0, 0, 4}));

  // Scalar division, positive
  clsa2 /= 2;
  EXPECT_TRUE(is_near(clsa2, Mat2 {9, 3, 3, 10}));
  cusa2 /= 2;
  EXPECT_TRUE(is_near(cusa2, Mat2 {4, 2, 2, 5}));
  clt2 /= 2;
  EXPECT_TRUE(is_near(clt2, Mat2 {9, 3, 3, 10}));
  cut2 /= 2;
  EXPECT_TRUE(is_near(cut2, Mat2 {4, 2, 2, 5}));
  cd2 /= 2;
  EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // .scale()
  clsa2.scale(2);
  EXPECT_TRUE(is_near(clsa2, Mat2 {36, 12, 12, 40}));
  cusa2.scale(2);
  EXPECT_TRUE(is_near(cusa2, Mat2 {16, 8, 8, 20}));
  clt2.scale(2);
  EXPECT_TRUE(is_near(clt2, Mat2 {36, 12, 12, 40}));
  cut2.scale(2);
  EXPECT_TRUE(is_near(cut2, Mat2 {16, 8, 8, 20}));
  cd2.scale(2);
  EXPECT_TRUE(is_near(cd2, Mat2 {4, 0, 0, 8}));

  // .inverse_scale()
  clsa2.inverse_scale(2);
  EXPECT_TRUE(is_near(clsa2, Mat2 {9, 3, 3, 10}));
  cusa2.inverse_scale(2);
  EXPECT_TRUE(is_near(cusa2, Mat2 {4, 2, 2, 5}));
  clt2.inverse_scale(2);
  EXPECT_TRUE(is_near(clt2, Mat2 {9, 3, 3, 10}));
  cut2.inverse_scale(2);
  EXPECT_TRUE(is_near(cut2, Mat2 {4, 2, 2, 5}));
  cd2.inverse_scale(2);
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

  // Zero
  EXPECT_TRUE(is_near(CovSA2l::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(CovSA2u::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(CovT2l::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(CovT2u::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(CovD2::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(CovSA2l::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(CovSA2u::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(CovT2l::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(CovT2u::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(CovD2::zero()), M2::Zero()));

  // Identity
  EXPECT_TRUE(is_near(CovSA2l::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(CovSA2u::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(CovT2l::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(CovT2u::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(CovD2::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(CovSA2l::identity()), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(CovSA2u::identity()), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(CovT2l::identity()), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(CovT2u::identity()), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(CovD2::identity()), M2::Identity()));

  // Subscripts
  EXPECT_NEAR((CovSA2l {9, 3, 3, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((CovSA2l {9, 3, 3, 10})(0, 1), 3, 1e-6);
  EXPECT_NEAR((CovSA2l {9, 3, 3, 10})(1, 0), 3, 1e-6);
  EXPECT_NEAR((CovSA2l {9, 3, 3, 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((CovSA2u {9, 3, 3, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((CovSA2u {9, 3, 3, 10})(0, 1), 3, 1e-6);
  EXPECT_NEAR((CovSA2u {9, 3, 3, 10})(1, 0), 3, 1e-6);
  EXPECT_NEAR((CovSA2u {9, 3, 3, 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((CovT2l {9, 3, 3, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((CovT2l {9, 3, 3, 10})(0, 1), 3, 1e-6);
  EXPECT_NEAR((CovT2l {9, 3, 3, 10})(1, 0), 3, 1e-6);
  EXPECT_NEAR((CovT2l {9, 3, 3, 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((CovT2u {9, 3, 3, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((CovT2u {9, 3, 3, 10})(0, 1), 3, 1e-6);
  EXPECT_NEAR((CovT2u {9, 3, 3, 10})(1, 0), 3, 1e-6);
  EXPECT_NEAR((CovT2u {9, 3, 3, 10})(1, 1), 10, 1e-6);
}


TEST_F(covariance_tests, Covariance_deduction_guides)
{
  EXPECT_TRUE(is_near(Covariance(CovSA2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Covariance(CovSA2l {9, 3, 3, 10}))>::Coefficients, C>);

  EXPECT_TRUE(is_near(Covariance(T2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Covariance(T2l {3, 0, 1, 3}))>::Coefficients, Axes<2>>);

  EXPECT_TRUE(is_near(Covariance(D2 {1, 2}), Mat2 {1, 0, 0, 2}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Covariance(D2 {1, 2}))>::Coefficients, Axes<2>>);

  EXPECT_TRUE(is_near(Covariance(Mat2 {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Covariance(Mat2 {9, 3, 3, 10}))>::Coefficients, C>);

  EXPECT_TRUE(is_near(Covariance((M2() << 9, 3, 3, 10).finished()), Mat2 {9, 3, 3, 10}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Covariance((M2() << 9, 3, 3, 10).finished()))>::Coefficients, Axes<2>>);

  EXPECT_TRUE(is_near(Covariance {9., 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Covariance {9., 3, 3, 10})>::Coefficients, Axes<2>>);
}


TEST_F(covariance_tests, Covariance_make)
{
  // Other covariance:
  EXPECT_TRUE(is_near(make_Covariance(CovSA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_Covariance(CovSA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_Covariance(CovT2l {9, 3, 3, 10}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_Covariance(CovT2u {9, 3, 3, 10}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_Covariance(CovD2 {1, 2}).base_matrix(), Mat2 {1, 0, 0, 2}));

  EXPECT_TRUE(is_near(make_Covariance(SqCovSA2l {3, 0, 1, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_Covariance(SqCovSA2u {3, 1, 0, 3}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_Covariance(SqCovT2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_Covariance(SqCovT2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_Covariance(SqCovD2 {1, 2}).base_matrix(), Mat2 {1, 0, 0, 2}));

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance(CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_Covariance(CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_Covariance(CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance(CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_Covariance(CovD2 {1, 2}).base_matrix())>);

  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_Covariance(adjoint(CovSA2l {9, 3, 3, 10})).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance(adjoint(CovSA2u {9, 3, 3, 10})).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance(adjoint(CovT2l {9, 3, 3, 10})).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_Covariance(adjoint(CovT2u {9, 3, 3, 10})).base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_Covariance(adjoint(CovD2 {1, 2})).base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<CovSA2l>().base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_Covariance<CovSA2u>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_Covariance<CovT2l>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<CovT2u>().base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_Covariance<CovD2>().base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<CovSA2l>().base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_Covariance<CovSA2u>().base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_Covariance<CovD2>().base_matrix())>);

  // Covariance bases:
  EXPECT_TRUE(is_near(make_Covariance<C>(SA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_Covariance<C>(SA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_Covariance<C>(T2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_Covariance<C>(T2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_Covariance<C>(D2 {1, 2}).base_matrix(), Mat2 {1, 0, 0, 2}));

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<C>(SA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_Covariance<C>(SA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_Covariance<C>(T2l {3, 0, 1, 3}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<C>(T2u {3, 1, 0, 3}).base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_Covariance<C>(D2 {1, 2}).base_matrix())>);

  static_assert(is_Eigen_upper_storage_triangle_v<decltype(adjoint(make_Covariance<C>(SA2l {9, 3, 3, 10})).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<C>(adjoint(SA2u {9, 3, 3, 10})).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<C>(adjoint(T2l {3, 0, 1, 3})).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_Covariance<C>(adjoint(T2u {3, 1, 0, 3})).base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_Covariance<C>(adjoint(D2 {1, 2})).base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<C, SA2l>().base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_Covariance<C, SA2u>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_Covariance<C, T2l>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<C, T2u>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<C, D2>().base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance(SA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(make_Covariance<C>(SA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<T2u>().base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_Covariance<C, T2l>().base_matrix())>);
  static_assert(is_diagonal_v<decltype(make_Covariance<C, D2>().base_matrix())>);

  // Regular matrices:
  EXPECT_TRUE(is_near(make_Covariance<C, TriangleType::lower>(Mat2 {9, 3, 3, 10}.base_matrix()).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_Covariance<C, TriangleType::upper>(Mat2 {9, 3, 3, 10}.base_matrix()).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_Covariance<C>(Mat2 {9, 3, 3, 10}.base_matrix()).base_matrix(), Mat2 {9, 3, 3, 10}));

  static_assert(is_lower_triangular_v<decltype(make_Covariance<C, TriangleType::lower>(Mat2 {9, 3, 3, 10}.base_matrix()).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<C, TriangleType::upper>(Mat2 {9, 3, 3, 10}.base_matrix()).base_matrix())>);

  static_assert(is_lower_triangular_v<decltype(make_Covariance<C, TriangleType::lower, M2>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<C, TriangleType::upper, M2>().base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<C>(Mat2 {9, 3, 3, 10}.base_matrix()).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<TriangleType::upper>(Mat2 {9, 3, 3, 10}.base_matrix()).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<C, M2>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<TriangleType::upper, M2>().base_matrix())>);

  // Typed matrices:
  EXPECT_TRUE(is_near(make_Covariance<TriangleType::lower>(Mat2 {9, 3, 3, 10}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_Covariance<TriangleType::upper>(Mat2 {9, 3, 3, 10}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_Covariance(Mat2 {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));

  static_assert(is_lower_triangular_v<decltype(make_Covariance<TriangleType::lower>(Mat2 {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<TriangleType::upper>(Mat2 {9, 3, 3, 10}).base_matrix())>);

  static_assert(is_lower_triangular_v<decltype(make_Covariance<TriangleType::lower, Mat2>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<TriangleType::upper, Mat2>().base_matrix())>);

  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance(Mat2 {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<Mat2>().base_matrix())>);

  // Eigen defaults
  EXPECT_TRUE(is_near(make_Covariance<C, TriangleType::lower>(9, 3, 3, 10).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_Covariance<C, TriangleType::upper>(9, 3, 3, 10).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_Covariance<C>(9, 3, 3, 10).base_matrix(), Mat2 {9, 3, 3, 10}));

  static_assert(is_lower_triangular_v<decltype(make_Covariance<C, TriangleType::lower>(9, 3, 3, 10).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<C, TriangleType::upper>(9, 3, 3, 10).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<C>(9, 3, 3, 10).base_matrix())>);

  static_assert(is_lower_triangular_v<decltype(make_Covariance<TriangleType::lower>(9, 3, 3, 10).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(make_Covariance<C, TriangleType::lower>().base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(make_Covariance<C, TriangleType::upper>().base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(make_Covariance<C>().base_matrix())>);
  static_assert(MatrixTraits<decltype(make_Covariance<C>())>::Coefficients::size == 2);
}


TEST_F(covariance_tests, Covariance_traits)
{
  static_assert(not is_square_root_v<CovSA2l>);
  static_assert(not is_diagonal_v<CovSA2l>);
  static_assert(is_self_adjoint_v<CovSA2l>);
  static_assert(not is_Cholesky_v<CovSA2l>);
  static_assert(not is_triangular_v<CovSA2l>);
  static_assert(not is_lower_triangular_v<CovSA2l>);
  static_assert(not is_upper_triangular_v<CovSA2l>);
  static_assert(not is_identity_v<CovSA2l>);
  static_assert(not is_zero_v<CovSA2l>);

  static_assert(not is_square_root_v<CovT2l>);
  static_assert(not is_diagonal_v<CovT2l>);
  static_assert(is_self_adjoint_v<CovT2l>);
  static_assert(is_Cholesky_v<CovT2l>);
  static_assert(not is_triangular_v<CovT2l>);
  static_assert(not is_lower_triangular_v<CovT2l>);
  static_assert(not is_upper_triangular_v<CovT2l>);
  static_assert(not is_upper_triangular_v<CovT2u>);
  static_assert(not is_identity_v<CovT2l>);
  static_assert(not is_zero_v<CovT2l>);

  static_assert(not is_square_root_v<CovD2>);
  static_assert(is_diagonal_v<CovD2>);
  static_assert(is_self_adjoint_v<CovD2>);
  static_assert(not is_Cholesky_v<CovD2>);
  static_assert(is_triangular_v<CovD2>);
  static_assert(is_lower_triangular_v<CovD2>);
  static_assert(is_upper_triangular_v<CovD2>);
  static_assert(is_upper_triangular_v<CovD2>);
  static_assert(not is_identity_v<CovD2>);
  static_assert(not is_zero_v<CovD2>);

  static_assert(not is_square_root_v<CovI2>);
  static_assert(is_diagonal_v<CovI2>);
  static_assert(is_self_adjoint_v<CovI2>);
  static_assert(not is_Cholesky_v<CovI2>);
  static_assert(is_triangular_v<CovI2>);
  static_assert(is_lower_triangular_v<CovI2>);
  static_assert(is_upper_triangular_v<CovI2>);
  static_assert(is_upper_triangular_v<CovI2>);
  static_assert(is_identity_v<CovI2>);
  static_assert(not is_zero_v<CovI2>);

  static_assert(not is_square_root_v<CovZ2>);
  static_assert(is_diagonal_v<CovZ2>);
  static_assert(is_self_adjoint_v<CovZ2>);
  static_assert(not is_Cholesky_v<CovZ2>);
  static_assert(is_triangular_v<CovZ2>);
  static_assert(is_lower_triangular_v<CovZ2>);
  static_assert(is_upper_triangular_v<CovZ2>);
  static_assert(is_upper_triangular_v<CovZ2>);
  static_assert(not is_identity_v<CovZ2>);
  static_assert(is_zero_v<CovZ2>);

  // MatrixTraits
  EXPECT_TRUE(is_near(MatrixTraits<CovSA2l>::make(SA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<CovSA2u>::make(SA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<CovSA2l>::make(T2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<CovSA2u>::make(T2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<CovSA2l>::zero(), Eigen::Matrix<double, 2, 2>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<CovSA2l>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<CovT2l>::make(SA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<CovT2u>::make(SA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<CovT2l>::make(T2l {3, 0, 1, 3}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<CovT2u>::make(T2u {3, 1, 0, 3}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<CovT2l>::zero(), Eigen::Matrix<double, 2, 2>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<CovT2l>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
}


TEST_F(covariance_tests, Covariance_overloads)
{
  EXPECT_TRUE(is_near(internal::convert_base_matrix(CovSA2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(CovSA2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(CovT2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(CovT2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(CovD2 {1, 2}).base_matrix(), Mean {1., 2}));
  EXPECT_TRUE(is_near(internal::convert_base_matrix(CovI2(i2)), Mat2 {1, 0, 0, 1}));

  EXPECT_TRUE(is_near(base_matrix(CovSA2l {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(base_matrix(CovSA2u {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(base_matrix(CovT2l {9, 3, 3, 10}), Mat2 { 3, 0, 1, 3}));
  EXPECT_TRUE(is_near(base_matrix(CovT2u {9, 3, 3, 10}), Mat2 { 3, 1, 0, 3}));

  // Square root
  EXPECT_TRUE(is_near(square_root(CovSA2l {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square_root(CovSA2u {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(square_root(CovT2l {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square_root(CovT2u {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(square_root(CovD2 {4, 9}), Mat2 {2, 0, 0, 3}));
  EXPECT_TRUE(is_near(square_root(CovI2(i2)), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(square_root(CovZ2()), Mat2 {0, 0, 0, 0}));

  // Semidefinite square root, general case
  EXPECT_TRUE(is_near(square_root(CovSA2l {9, 3, 3, 1}), Mat2 {3, 0, 1, 0}));
  EXPECT_TRUE(is_near(square_root(CovSA2u {4, 2, 2, 1}), Mat2 {2, 1, 0, 0}));
  EXPECT_TRUE(is_near(square_root(CovT2l {9, 3, 3, 1}), Mat2 {3, 0, 1, 0}));
  EXPECT_TRUE(is_near(square_root(CovT2u {4, 2, 2, 1}), Mat2 {2, 1, 0, 0}));

  // Semidefinite square root, constant matrix
  EXPECT_TRUE(is_near(square_root(make_Covariance<C>(EigenSelfAdjointMatrix<M2::ConstantReturnType, TriangleType::lower>(M2::Constant(9)))), Mat2 {3, 0, 3, 0}));
  EXPECT_TRUE(is_near(square_root(make_Covariance<C>(EigenSelfAdjointMatrix<M2::ConstantReturnType, TriangleType::upper>(M2::Constant(4)))), Mat2 {2, 2, 0, 0}));

  EXPECT_TRUE(is_near(to_Cholesky(CovSA2l {9, 3, 3, 10}).base_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_Cholesky(CovSA2u {9, 3, 3, 10}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(from_Cholesky(CovT2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(from_Cholesky(CovT2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_Cholesky(CovD2 {4, 9}).base_matrix(), Mat2 {2, 0, 0, 3}));
  EXPECT_TRUE(is_near(from_Cholesky(CovD2 {4, 9}).base_matrix(), Mat2 {16, 0, 0, 81}));
  EXPECT_TRUE(is_near(to_Cholesky(CovI2(i2)).base_matrix(), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(from_Cholesky(CovI2(i2)).base_matrix(), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_Cholesky(CovZ2()).base_matrix(), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(from_Cholesky(CovZ2()).base_matrix(), Mat2 {0, 0, 0, 0}));

  EXPECT_TRUE(is_near(strict_matrix(CovSA2l {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(strict_matrix(CovSA2u {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(strict_matrix(CovT2l {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(strict_matrix(CovT2l {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));

  static_assert(std::is_same_v<std::decay_t<decltype(strict(CovSA2l {9, 3, 3, 10} * 2))>, CovSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(CovSA2u {9, 3, 3, 10} * 2))>, CovSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(CovT2l {9, 3, 3, 10} * 2))>, CovT2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(CovT2u {9, 3, 3, 10} * 2))>, CovT2u>);

  EXPECT_TRUE(is_near(transpose(CovSA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(transpose(CovSA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(transpose(CovT2l {9, 3, 3, 10}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(transpose(CovT2u {9, 3, 3, 10}).base_matrix(), Mat2 {3, 0, 1, 3}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(CovSA2l {9, 3, 3, 10})))>, CovSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(CovSA2u {9, 3, 3, 10})))>, CovSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(CovT2l {9, 3, 3, 10})))>, CovT2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(CovT2u {9, 3, 3, 10})))>, CovT2l>);

  EXPECT_TRUE(is_near(adjoint(CovSA2l {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(adjoint(CovSA2u {9, 3, 3, 10}).base_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(adjoint(CovT2l {9, 3, 3, 10}).base_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(adjoint(CovT2u {9, 3, 3, 10}).base_matrix(), Mat2 {3, 0, 1, 3}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(CovSA2l {9, 3, 3, 10})))>, CovSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(CovSA2u {9, 3, 3, 10})))>, CovSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(CovT2l {9, 3, 3, 10})))>, CovT2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(CovT2u {9, 3, 3, 10})))>, CovT2l>);

  EXPECT_NEAR(determinant(CovSA2l {9, 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(CovSA2u {9, 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(CovT2l {9, 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(CovT2u {9, 3, 3, 10}), 81, 1e-6);

  EXPECT_NEAR(trace(CovSA2l {9, 3, 3, 10}), 19, 1e-6);
  EXPECT_NEAR(trace(CovSA2u {9, 3, 3, 10}), 19, 1e-6);
  EXPECT_NEAR(trace(CovT2l {9, 3, 3, 10}), 19, 1e-6);
  EXPECT_NEAR(trace(CovT2u {9, 3, 3, 10}), 19, 1e-6);

  auto x1sal = CovSA2l {9., 3, 3, 10};
  auto x1sau = CovSA2u {9., 3, 3, 10};
  auto x1tl = CovT2l {9., 3, 3, 10};
  auto x1tu = CovT2u {9., 3, 3, 10};
  rank_update(x1sal, Mat2 {2, 0, 1, 2}, 4);
  rank_update(x1sau, Mat2 {2, 1, 0, 2}, 4);
  rank_update(x1tl, Mat2 {2, 0, 1, 2}, 4);
  rank_update(x1tu, Mat2 {2, 1, 0, 2}, 4);
  EXPECT_TRUE(is_near(x1sal, Mat2 {25., 11, 11, 30}));
  EXPECT_TRUE(is_near(x1sau, Mat2 {29., 11, 11, 26}));
  EXPECT_TRUE(is_near(x1tl, Mat2 {25., 11, 11, 30}));
  EXPECT_TRUE(is_near(x1tu, Mat2 {29., 11, 11, 26}));
  EXPECT_TRUE(is_near(rank_update(CovSA2l {9., 3, 3, 10}, Mat2 {2, 0, 1, 2}, 4), Mat2 {25., 11, 11, 30}));
  EXPECT_TRUE(is_near(rank_update(CovSA2u {9., 3, 3, 10}, Mat2 {2, 1, 0, 2}, 4), Mat2 {29., 11, 11, 26}));
  EXPECT_TRUE(is_near(rank_update(CovT2l {9., 3, 3, 10}, Mat2 {2, 0, 1, 2}, 4), Mat2 {25., 11, 11, 30}));
  EXPECT_TRUE(is_near(rank_update(CovT2u {9., 3, 3, 10}, Mat2 {2, 1, 0, 2}, 4), Mat2 {29., 11, 11, 26}));

  EXPECT_TRUE(is_near(solve(CovSA2l {9., 3, 3, 10}, Mat2col {15, 23}), Mat2col {1, 2}));
  EXPECT_TRUE(is_near(solve(CovSA2u {9., 3, 3, 10}, Mat2col {15, 23}), Mat2col {1, 2}));
  EXPECT_TRUE(is_near(solve(CovT2l {9., 3, 3, 10}, Mat2col {15, 23}), Mat2col {1, 2}));
  EXPECT_TRUE(is_near(solve(CovT2u {9., 3, 3, 10}, Mat2col {15, 23}), Mat2col {1, 2}));

  EXPECT_TRUE(is_near(reduce_columns(CovSA2l {9, 3, 3, 10}), Mat2col {6., 6.5}));
  EXPECT_TRUE(is_near(reduce_columns(CovSA2u {9, 3, 3, 10}), Mat2col {6., 6.5}));
  EXPECT_TRUE(is_near(reduce_columns(CovT2l {9, 3, 3, 10}), Mat2col {6., 6.5}));
  EXPECT_TRUE(is_near(reduce_columns(CovT2u {9, 3, 3, 10}), Mat2col {6., 6.5}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(CovSA2l {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(CovSA2u {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(CovT2l {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(CovT2u {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));

  EXPECT_TRUE(is_near(square(QR_decomposition(CovSA2l {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(QR_decomposition(CovSA2u {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(QR_decomposition(CovT2l {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(QR_decomposition(CovT2u {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
}


TEST_F(covariance_tests, Covariance_blocks)
{
  using C4 = Concatenate<C, C>;
  using M4 = Eigen::Matrix<double, 4, 4>;
  using Mat4 = TypedMatrix<C4, C4, M4>;
  using CovSA4l = Covariance<C4, EigenSelfAdjointMatrix<M4, TriangleType::lower>>;
  using CovSA4u = Covariance<C4, EigenSelfAdjointMatrix<M4, TriangleType::upper>>;
  using CovT4l = Covariance<C4, EigenTriangularMatrix<M4, TriangleType::lower>>;
  using CovT4u = Covariance<C4, EigenTriangularMatrix<M4, TriangleType::upper>>;
  Mat2 m1 {9, 3, 3, 10}, m2 {4, 2, 2, 5};
  Mat4 n {9, 3, 0, 0,
          3, 10, 0, 0,
          0, 0, 4, 2,
          0, 0, 2, 5};
  EXPECT_TRUE(is_near(concatenate(CovSA2l(m1), CovSA2l(m2)), n));
  EXPECT_TRUE(is_near(concatenate(CovSA2u(m1), CovSA2u(m2)), n));
  EXPECT_TRUE(is_near(concatenate(CovT2l(m1), CovT2l(m2)), n));
  EXPECT_TRUE(is_near(concatenate(CovT2u(m1), CovT2u(m2)), n));

  EXPECT_TRUE(is_near(split(CovSA4l(n)), std::tuple {}));
  EXPECT_TRUE(is_near(split<C, C>(CovSA4l(n)), std::tuple {m1, m2}));
  EXPECT_TRUE(is_near(split<C, C>(CovSA4u(n)), std::tuple {m1, m2}));
  EXPECT_TRUE(is_near(split<C, C>(CovT4l(n)), std::tuple {m1, m2}));
  EXPECT_TRUE(is_near(split<C, C>(CovT4u(n)), std::tuple {m1, m2}));
  EXPECT_TRUE(is_near(split<C, C::Take<1>>(CovSA4l(n)), std::tuple {m1, TypedMatrix<Angle, Angle>{4}}));
  EXPECT_TRUE(is_near(split<C, C::Take<1>>(CovSA4u(n)), std::tuple {m1, TypedMatrix<Angle, Angle>{4}}));
  EXPECT_TRUE(is_near(split<C, C::Take<1>>(CovT4l(n)), std::tuple {m1, TypedMatrix<Angle, Angle>{4}}));
  EXPECT_TRUE(is_near(split<C, C::Take<1>>(CovT4u(n)), std::tuple {m1, TypedMatrix<Angle, Angle>{4}}));

  EXPECT_TRUE(is_near(split_vertical(CovSA4l(n)), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(CovSA4l(n)), std::tuple {concatenate_horizontal(m1, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(CovSA4u(n)), std::tuple {concatenate_horizontal(m1, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(CovT4l(n)), std::tuple {concatenate_horizontal(m1, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(CovT4u(n)), std::tuple {concatenate_horizontal(m1, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2)}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(CovSA4l(n)), std::tuple {concatenate_horizontal(m1, Mat2::zero()), TypedMatrix<Angle, C4>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(CovSA4u(n)), std::tuple {concatenate_horizontal(m1, Mat2::zero()), TypedMatrix<Angle, C4>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(CovT4l(n)), std::tuple {concatenate_horizontal(m1, Mat2::zero()), TypedMatrix<Angle, C4>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(CovT4u(n)), std::tuple {concatenate_horizontal(m1, Mat2::zero()), TypedMatrix<Angle, C4>{0, 0, 4, 2}}));

  EXPECT_TRUE(is_near(split_horizontal(CovSA4l(n)), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(CovSA4l(n)), std::tuple {concatenate_vertical(m1, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(CovSA4u(n)), std::tuple {concatenate_vertical(m1, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(CovT4l(n)), std::tuple {concatenate_vertical(m1, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(CovT4u(n)), std::tuple {concatenate_vertical(m1, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(CovSA4l(n)), std::tuple {concatenate_vertical(m1, Mat2::zero()), TypedMatrix<C4, Angle>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(CovSA4u(n)), std::tuple {concatenate_vertical(m1, Mat2::zero()), TypedMatrix<C4, Angle>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(CovT4l(n)), std::tuple {concatenate_vertical(m1, Mat2::zero()), TypedMatrix<C4, Angle>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(CovT4u(n)), std::tuple {concatenate_vertical(m1, Mat2::zero()), TypedMatrix<C4, Angle>{0, 0, 4, 2}}));

  EXPECT_TRUE(is_near(column(CovSA4l(n), 2), Mean{0., 0, 4, 2}));
  EXPECT_TRUE(is_near(column(CovSA4u(n), 2), Mean{0., 0, 4, 2}));
  EXPECT_TRUE(is_near(column(CovT4l(n), 2), Mean{0., 0, 4, 2}));
  EXPECT_TRUE(is_near(column(CovT4u(n), 2), Mean{0., 0, 4, 2}));

  EXPECT_TRUE(is_near(column<1>(CovSA4l(n)), Mean{3., 10, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(CovSA4l(n)), Mean{3., 10, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(CovT4u(n)), Mean{3., 10, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(CovT4l(n)), Mean{3., 10, 0, 0}));

  EXPECT_TRUE(is_near(apply_columnwise(CovSA2l(m1), [](auto col){ return col * 2; }), Mat2 {18, 6, 6, 20}));
  EXPECT_TRUE(is_near(apply_columnwise(CovSA2u(m1), [](const auto col){ return col * 2; }), Mat2 {18, 6, 6, 20}));
  EXPECT_TRUE(is_near(apply_columnwise(CovT2l(m1), [](auto&& col){ return col * 2; }), Mat2 {18, 6, 6, 20}));
  EXPECT_TRUE(is_near(apply_columnwise(CovT2u(m1), [](const auto& col){ return col * 2; }), Mat2 {18, 6, 6, 20}));

  EXPECT_TRUE(is_near(apply_columnwise(CovSA2l(m1), [](auto col, std::size_t i){ return col * i; }), Mat2 {0, 3, 0, 10}));
  EXPECT_TRUE(is_near(apply_columnwise(CovSA2u(m1), [](const auto col, std::size_t i){ return col * i; }), Mat2 {0, 3, 0, 10}));
  EXPECT_TRUE(is_near(apply_columnwise(CovT2l(m1), [](auto&& col, std::size_t i){ return col * i; }), Mat2 {0, 3, 0, 10}));
  EXPECT_TRUE(is_near(apply_columnwise(CovT2u(m1), [](const auto& col, std::size_t i){ return col * i; }), Mat2 {0, 3, 0, 10}));

  EXPECT_TRUE(is_near(apply_coefficientwise(CovSA2l(m1), [](auto x){ return x + 1; }), Mat2 {10, 4, 4, 11}));
  EXPECT_TRUE(is_near(apply_coefficientwise(CovSA2u(m1), [](const auto x){ return x + 1; }), Mat2 {10, 4, 4, 11}));
  EXPECT_TRUE(is_near(apply_coefficientwise(CovT2l(m1), [](auto&& x){ return x + 1; }), Mat2 {10, 4, 4, 11}));
  EXPECT_TRUE(is_near(apply_coefficientwise(CovT2u(m1), [](const auto& x){ return x + 1; }), Mat2 {10, 4, 4, 11}));

  EXPECT_TRUE(is_near(apply_coefficientwise(CovSA2l(m1), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat2 {9, 4, 4, 12}));
  EXPECT_TRUE(is_near(apply_coefficientwise(CovSA2u(m1), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat2 {9, 4, 4, 12}));
  EXPECT_TRUE(is_near(apply_coefficientwise(CovT2l(m1), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat2 {9, 4, 4, 12}));
  EXPECT_TRUE(is_near(apply_coefficientwise(CovT2u(m1), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat2 {9, 4, 4, 12}));
}


TEST_F(covariance_tests, Covariance_addition)
{
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + CovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + covi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} + covz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + CovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + covi2).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + CovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + covi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + covz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + CovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + covi2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + CovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + covi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + covz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} + CovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} + covi2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} + covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 5, 5, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + CovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + covi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + covz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} + CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} + CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} + CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} + CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} + CovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} + covi2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} + covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovSA2l {4, 2, 2, 5}, Mat2 {13, 2, 2, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovSA2u {4, 2, 2, 5}, Mat2 {13, 2, 2, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovT2l {4, 2, 2, 5}, Mat2 {13, 2, 2, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovT2u {4, 2, 2, 5}, Mat2 {13, 2, 2, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + CovD2 {4, 5}, Mat2 {13, 0, 0, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + covi2, Mat2 {10, 0, 0, 11}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + covz2, Mat2 {9, 0, 0, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovD2 {9, 10} + CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovD2 {9, 10} + CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovD2 {9, 10} + CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovD2 {9, 10} + CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} + CovD2 {4, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} + covi2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} + covz2).base_matrix())>);

  EXPECT_TRUE(is_near(covi2 + CovSA2l {4, 2, 2, 5}, Mat2 {5, 2, 2, 6}));
  EXPECT_TRUE(is_near(covi2 + CovSA2u {4, 2, 2, 5}, Mat2 {5, 2, 2, 6}));
  EXPECT_TRUE(is_near(covi2 + CovT2l {4, 2, 2, 5}, Mat2 {5, 2, 2, 6}));
  EXPECT_TRUE(is_near(covi2 + CovT2u {4, 2, 2, 5}, Mat2 {5, 2, 2, 6}));
  EXPECT_TRUE(is_near(covi2 + CovD2 {4, 5}, Mat2 {5, 0, 0, 6}));
  EXPECT_TRUE(is_near(covi2 + covi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(covi2 + covz2, Mat2 {1, 0, 0, 1}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covi2 + CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covi2 + CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covi2 + CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covi2 + CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 + CovD2 {4, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 + covi2).base_matrix())>);
  static_assert(is_identity_v<decltype((covi2 + covz2).base_matrix())>);

  EXPECT_TRUE(is_near(covz2 + CovSA2l {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covz2 + CovSA2u {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covz2 + CovT2l {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covz2 + CovT2u {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covz2 + CovD2 {4, 5}, Mat2 {4, 0, 0, 5}));
  EXPECT_TRUE(is_near(covz2 + covi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covz2 + covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covz2 + CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covz2 + CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((covz2 + CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((covz2 + CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covz2 + CovD2 {4, 5}).base_matrix())>);
  static_assert(is_identity_v<decltype((covz2 + covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 + covz2).base_matrix())>);
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
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + sqcovi2).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} + sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + SqCovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + sqcovi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} + sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + sqcovi2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} + sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + SqCovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + sqcovi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} + sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} + SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} + sqcovi2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} + sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 3, 4, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 4, 3, 12}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + SqCovD2 {4, 5}, Mat2 {13, 3, 3, 15}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + sqcovi2, Mat2 {10, 3, 3, 11}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} + sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} + SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} + sqcovi2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} + sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovSA2l {2, 0, 1, 2}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovSA2u {2, 1, 0, 2}, Mat2 {11, 1, 0, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovT2l {2, 0, 1, 2}, Mat2 {11, 0, 1, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovT2u {2, 1, 0, 2}, Mat2 {11, 1, 0, 12}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + SqCovD2 {4, 5}, Mat2 {13, 0, 0, 15}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + sqcovi2, Mat2 {10, 0, 0, 11}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} + sqcovz2, Mat2 {9, 0, 0, 10}));
  static_assert(is_diagonal_v<decltype((CovD2 {9, 3, 3, 10} + SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 3, 3, 10} + sqcovi2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 3, 3, 10} + sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(covi2 + SqCovSA2l {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covi2 + SqCovSA2u {2, 1, 0, 2}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(covi2 + SqCovT2l {2, 0, 1, 2}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covi2 + SqCovT2u {2, 1, 0, 2}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(covi2 + SqCovD2 {4, 5}, Mat2 {5, 0, 0, 6}));
  EXPECT_TRUE(is_near(covi2 + sqcovi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(covi2 + sqcovz2, Mat2 {1, 0, 0, 1}));
  static_assert(is_diagonal_v<decltype((covi2 + SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 + sqcovi2).base_matrix())>);
  static_assert(is_identity_v<decltype((covi2 + sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(covz2 + SqCovSA2l {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(covz2 + SqCovSA2u {2, 1, 0, 2}, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(covz2 + SqCovT2l {2, 0, 1, 2}, Mat2 {2, 0, 1, 2}));
  EXPECT_TRUE(is_near(covz2 + SqCovT2u {2, 1, 0, 2}, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(covz2 + SqCovD2 {4, 5}, Mat2 {4, 0, 0, 5}));
  EXPECT_TRUE(is_near(covz2 + sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covz2 + sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((covz2 + SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_identity_v<decltype((covz2 + sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 + sqcovz2).base_matrix())>);

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
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - CovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - covi2).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - CovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - covi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - covz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - CovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - covi2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - CovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - covi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - covz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} - CovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} - covi2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} - covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, 1, 1, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - CovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - covi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - covz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} - CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} - CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} - CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} - CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} - CovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} - covi2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} - covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovSA2l {4, 2, 2, 5}, Mat2 {5, -2, -2, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovSA2u {4, 2, 2, 5}, Mat2 {5, -2, -2, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovT2l {4, 2, 2, 5}, Mat2 {5, -2, -2, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovT2u {4, 2, 2, 5}, Mat2 {5, -2, -2, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - CovD2 {4, 5}, Mat2 {5, 0, 0, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - covi2, Mat2 {8, 0, 0, 9}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - covz2, Mat2 {9, 0, 0, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovD2 {9, 10} - CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovD2 {9, 10} - CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovD2 {9, 10} - CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovD2 {9, 10} - CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} - CovD2 {4, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} - covi2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} - covz2).base_matrix())>);

  EXPECT_TRUE(is_near(covi2 - CovSA2l {4, 2, 2, 5}, Mat2 {-3, -2, -2, -4}));
  EXPECT_TRUE(is_near(covi2 - CovSA2u {4, 2, 2, 5}, Mat2 {-3, -2, -2, -4}));
  EXPECT_TRUE(is_near(covi2 - CovT2l {4, 2, 2, 5}, Mat2 {-3, -2, -2, -4}));
  EXPECT_TRUE(is_near(covi2 - CovT2u {4, 2, 2, 5}, Mat2 {-3, -2, -2, -4}));
  EXPECT_TRUE(is_near(covi2 - CovD2 {4, 5}, Mat2 {-3, 0, 0, -4}));
  EXPECT_TRUE(is_near(covi2 - covi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covi2 - covz2, Mat2 {1, 0, 0, 1}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covi2 - CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covi2 - CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covi2 - CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covi2 - CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 - CovD2 {4, 5}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 - covz2).base_matrix())>);
  static_assert(is_identity_v<decltype((covi2 - covz2).base_matrix())>);

  EXPECT_TRUE(is_near(covz2 - CovSA2l {4, 2, 2, 5}, Mat2 {-4, -2, -2, -5}));
  EXPECT_TRUE(is_near(covz2 - CovSA2u {4, 2, 2, 5}, Mat2 {-4, -2, -2, -5}));
  EXPECT_TRUE(is_near(covz2 - CovT2l {4, 2, 2, 5}, Mat2 {-4, -2, -2, -5}));
  EXPECT_TRUE(is_near(covz2 - CovT2u {4, 2, 2, 5}, Mat2 {-4, -2, -2, -5}));
  EXPECT_TRUE(is_near(covz2 - CovD2 {4, 5}, Mat2 {-4, 0, 0, -5}));
  EXPECT_TRUE(is_near(covz2 - covi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(covz2 - covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covz2 - CovSA2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covz2 - CovSA2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covz2 - CovT2l {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covz2 - CovT2u {4, 2, 2, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covz2 - CovD2 {4, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covz2 - covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 - covz2).base_matrix())>);
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
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - sqcovi2).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} - sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - SqCovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - sqcovi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} - sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - sqcovi2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} - sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - SqCovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - sqcovi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} - sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} - SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovT2l {9, 3, 3, 10} - sqcovi2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} - sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 3, 2, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, 2, 3, 8}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - SqCovD2 {4, 5}, Mat2 {5, 3, 3, 5}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - sqcovi2, Mat2 {8, 3, 3, 9}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} - sqcovz2, Mat2 {9, 3, 3, 10}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} - SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovT2u {9, 3, 3, 10} - sqcovi2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} - sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovSA2l {2, 0, 1, 2}, Mat2 {7, 0, -1, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovSA2u {2, 1, 0, 2}, Mat2 {7, -1, 0, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovT2l {2, 0, 1, 2}, Mat2 {7, 0, -1, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovT2u {2, 1, 0, 2}, Mat2 {7, -1, 0, 8}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - SqCovD2 {4, 5}, Mat2 {5, 0, 0, 5}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - sqcovi2, Mat2 {8, 0, 0, 9}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} - sqcovz2, Mat2 {9, 0, 0, 10}));
  static_assert(is_diagonal_v<decltype((CovD2 {9, 3, 3, 10} - SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 3, 3, 10} - sqcovi2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 3, 3, 10} - sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(covi2 - SqCovSA2l {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(covi2 - SqCovSA2u {2, 1, 0, 2}, Mat2 {-1, -1, 0, -1}));
  EXPECT_TRUE(is_near(covi2 - SqCovT2l {2, 0, 1, 2}, Mat2 {-1, 0, -1, -1}));
  EXPECT_TRUE(is_near(covi2 - SqCovT2u {2, 1, 0, 2}, Mat2 {-1, -1, 0, -1}));
  EXPECT_TRUE(is_near(covi2 - SqCovD2 {4, 5}, Mat2 {-3, 0, 0, -4}));
  EXPECT_TRUE(is_near(covi2 - sqcovi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covi2 - sqcovz2, Mat2 {1, 0, 0, 1}));
  static_assert(is_diagonal_v<decltype((covi2 - SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_zero_v<decltype((covi2 - sqcovi2).base_matrix())>);
  static_assert(is_identity_v<decltype((covi2 - sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(covz2 - SqCovSA2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(covz2 - SqCovSA2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(covz2 - SqCovT2l {2, 0, 1, 2}, Mat2 {-2, 0, -1, -2}));
  EXPECT_TRUE(is_near(covz2 - SqCovT2u {2, 1, 0, 2}, Mat2 {-2, -1, 0, -2}));
  EXPECT_TRUE(is_near(covz2 - SqCovD2 {4, 5}, Mat2 {-4, 0, 0, -5}));
  EXPECT_TRUE(is_near(covz2 - sqcovi2, Mat2 {-1, 0, 0, -1}));
  EXPECT_TRUE(is_near(covz2 - sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((covz2 - SqCovD2 {4, 5}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covz2 - sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 - sqcovz2).base_matrix())>);

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
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovSA2l {9, 3, 3, 10} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * SqCovD2 {3, 3}, Mat2 {27, 9, 9, 30}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * sqcovi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l {9, 3, 3, 10} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovSA2l {9, 3, 3, 10} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovT2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovT2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * CovD2 {9, 10}, Mat2 {81, 30, 27, 100}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * covi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovSA2u {9, 3, 3, 10} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * SqCovD2 {3, 3}, Mat2 {27, 9, 9, 30}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * sqcovi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u {9, 3, 3, 10} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovSA2u {9, 3, 3, 10} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovT2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovT2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * CovD2 {9, 10}, Mat2 {81, 30, 27, 100}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * covi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovT2l {9, 3, 3, 10} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * SqCovD2 {3, 3}, Mat2 {27, 9, 9, 30}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * sqcovi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype((CovT2l {9, 3, 3, 10} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovT2l {9, 3, 3, 10} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovT2l {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovT2u {9, 3, 3, 10}, Mat2 {90, 57, 57, 109}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * CovD2 {9, 10}, Mat2 {81, 30, 27, 100}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * covi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovT2u {9, 3, 3, 10} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {30, 9, 19, 30}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 18, 9, 33}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * SqCovD2 {3, 3}, Mat2 {27, 9, 9, 30}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * sqcovi2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_upper_triangular_v<decltype((CovT2u {9, 3, 3, 10} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovT2u {9, 3, 3, 10} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovSA2l {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovSA2u {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovT2l {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovT2u {9, 3, 3, 10}, Mat2 {81, 27, 30, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * CovD2 {9, 10}, Mat2 {81, 0, 0, 100}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * covi2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} * CovD2 {9, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovD2 {9, 10} * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovSA2l {3, 0, 1, 3}, Mat2 {27, 0, 10, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovSA2u {3, 1, 0, 3}, Mat2 {27, 9, 0, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovT2l {3, 0, 1, 3}, Mat2 {27, 0, 10, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovT2u {3, 1, 0, 3}, Mat2 {27, 9, 0, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * SqCovD2 {3, 3}, Mat2 {27, 0, 0, 30}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * sqcovi2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(CovD2 {9, 10} * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} * SqCovD2 {9, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {9, 10} * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((CovD2 {9, 10} * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(covi2 * CovSA2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covi2 * CovSA2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covi2 * CovT2l {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covi2 * CovT2u {9, 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covi2 * CovD2 {9, 10}, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(covi2 * covi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covi2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covi2 * CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covi2 * CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((covi2 * CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((covi2 * CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 * CovD2 {9, 10}).base_matrix())>);
  static_assert(is_identity_v<decltype((covi2 * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((covi2 * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(covi2 * SqCovSA2l {3, 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covi2 * SqCovSA2u {3, 1, 0, 3}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(covi2 * SqCovT2l {3, 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(covi2 * SqCovT2u {3, 1, 0, 3}, Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(covi2 * SqCovD2 {3, 3}, Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(covi2 * sqcovi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covi2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covi2 * SqCovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covi2 * SqCovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((covi2 * SqCovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((covi2 * SqCovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 * SqCovD2 {9, 10}).base_matrix())>);
  static_assert(is_identity_v<decltype((covi2 * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((covi2 * sqcovz2).base_matrix())>);

  EXPECT_TRUE(is_near(covz2 * CovSA2l {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * CovSA2u {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * CovT2l {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * CovT2u {9, 3, 3, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * CovD2 {9, 10}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * covi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((covz2 * CovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * CovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * CovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * CovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * CovD2 {9, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(covz2 * SqCovSA2l {3, 0, 1, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * SqCovSA2u {3, 1, 0, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * SqCovT2l {3, 0, 1, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * SqCovT2u {3, 1, 0, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * SqCovD2 {3, 3}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * sqcovi2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * sqcovz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((covz2 * SqCovSA2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * SqCovSA2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * SqCovT2l {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * SqCovT2u {9, 3, 3, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * SqCovD2 {9, 10}).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * sqcovi2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * sqcovz2).base_matrix())>);
}


TEST_F(covariance_tests, Covariance_mult_TypedMatrix)
{
  using MatI2 = TypedMatrix<C, C, I2>;
  using MatZ2 = TypedMatrix<C, C, Z2>;
  auto mati2 = MatI2(i2);
  auto matz2 = MatZ2(z2);
  using Cx = Coefficients<Axis, Angle>;
  using MatI2x = TypedMatrix<C, Cx, I2>;
  auto mati2x = MatI2x(i2);

  auto covSA2l = CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(covSA2l * Mat2 {4, 2, 2, 5}, Mat2 {42, 33, 32, 56}));
  EXPECT_TRUE(is_near(covSA2l * mati2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covSA2l * mati2x, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covSA2l * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((covSA2l * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((covSA2l * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covSA2l * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covSA2l * mati2x)>::ColumnCoefficients, Cx>);

  auto covSA2u = CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(covSA2u * Mat2 {4, 2, 2, 5}, Mat2 {42, 33, 32, 56}));
  EXPECT_TRUE(is_near(covSA2u * mati2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covSA2u * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((covSA2u * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((covSA2u * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covSA2u * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covSA2u * mati2x)>::ColumnCoefficients, Cx>);

  auto covT2l = CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(covT2l * Mat2 {4, 2, 2, 5}, Mat2 {42, 33, 32, 56}));
  EXPECT_TRUE(is_near(covT2l * mati2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covT2l * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype((covT2l * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((covT2l * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covT2l * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covT2l * mati2x)>::ColumnCoefficients, Cx>);

  auto covT2u = CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(covT2u * Mat2 {4, 2, 2, 5}, Mat2 {42, 33, 32, 56}));
  EXPECT_TRUE(is_near(covT2u * mati2, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covT2u * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_upper_triangular_v<decltype((covT2u * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((covT2u * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covT2u * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covT2u * mati2x)>::ColumnCoefficients, Cx>);

  auto covD2 = CovD2 {9, 10};
  EXPECT_TRUE(is_near(covD2 * Mat2 {4, 2, 2, 5}, Mat2 {36, 18, 20, 50}));
  EXPECT_TRUE(is_near(covD2 * mati2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(covD2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((covD2 * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((covD2 * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covD2 * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covD2 * mati2x)>::ColumnCoefficients, Cx>);

  EXPECT_TRUE(is_near(covi2 * Mat2 {4, 2, 2, 5}, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(covi2 * mati2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(covi2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_identity_v<decltype((covi2 * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((covi2 * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covi2 * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covi2 * mati2x)>::ColumnCoefficients, Cx>);

  EXPECT_TRUE(is_near(covz2 * Mat2 {4, 2, 2, 5}, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * mati2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(covz2 * matz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((covz2 * mati2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * matz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covz2 * mati2x)>::RowCoefficients, C>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(covz2 * mati2x)>::ColumnCoefficients, Cx>);

  // Scalar multiplication
  Mat2 p1 {4, 2, 2, 5};
  EXPECT_TRUE(is_near(CovSA2l(p1) * 2, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovSA2u(p1) * 2, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovT2l(p1) * 2, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovT2u(p1) * 2, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovD2 {1, 2} * 2, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(covi2 * 2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(covz2 * 2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l(p1) * 2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u(p1) * 2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l(p1) * 2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u(p1) * 2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {1, 2} * 2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 * 2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * 2).base_matrix())>);

  EXPECT_TRUE(is_near(2 * CovSA2l(p1), Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(2 * CovSA2u(p1), Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(2 * CovT2l(p1), Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(2 * CovT2u(p1), Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(2 * CovD2 {1, 2}, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(2 * covi2, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((2 * CovSA2l(p1)).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((2 * CovSA2u(p1)).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((2 * CovT2l(p1)).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((2 * CovT2u(p1)).base_matrix())>);
  static_assert(is_diagonal_v<decltype((2 * CovD2 {1, 2}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((2 * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((2 * covz2).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l(p1) * -2, Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(CovSA2u(p1) * -2, Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(CovD2 {1, 2} * -2, Mat2 {-2, 0, 0, -4}));
  EXPECT_TRUE(is_near(covi2 * -2, Mat2 {-2, 0, 0, -2}));
  EXPECT_TRUE(is_near(covz2 * -2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l(p1) * -2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u(p1) * -2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {1, 2} * -2).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 * -2).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 * -2).base_matrix())>);

  EXPECT_TRUE(is_near(-2 * CovSA2l(p1), Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(-2 * CovSA2u(p1), Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(-2 * CovD2 {1, 2}, Mat2 {-2, 0, 0, -4}));
  EXPECT_TRUE(is_near(-2 * covi2, Mat2 {-2, 0, 0, -2}));
  EXPECT_TRUE(is_near(-2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((-2 * CovSA2l(p1)).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((-2 * CovSA2u(p1)).base_matrix())>);
  static_assert(is_diagonal_v<decltype((-2 * CovD2 {1, 2}).base_matrix())>);
  static_assert(is_diagonal_v<decltype((-2 * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((-2 * covz2).base_matrix())>);

  // Scalar division
  EXPECT_TRUE(is_near(CovSA2l(p1) / 0.5, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovSA2u(p1) / 0.5, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovT2l(p1) / 0.5, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovT2u(p1) / 0.5, Mat2 {8, 4, 4, 10}));
  EXPECT_TRUE(is_near(CovD2 {1, 2} / 0.5, Mat2 {2, 0, 0, 4}));
  EXPECT_TRUE(is_near(covi2 / 0.5, Mat2 {2, 0, 0, 2}));
  EXPECT_TRUE(is_near(covz2 / 0.5, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l(p1) / 0.5).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u(p1) / 0.5).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype((CovT2l(p1) / 0.5).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype((CovT2u(p1) / 0.5).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {1, 2} / 0.5).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 / 0.5).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 / 0.5).base_matrix())>);

  EXPECT_TRUE(is_near(CovSA2l(p1) / -0.5, Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(CovSA2u(p1) / -0.5, Mat2 {-8, -4, -4, -10}));
  EXPECT_TRUE(is_near(CovD2 {1, 2} / -0.5, Mat2 {-2, 0, 0, -4}));
  EXPECT_TRUE(is_near(covi2 / -0.5, Mat2 {-2, 0, 0, -2}));
  EXPECT_TRUE(is_near(covz2 / -0.5, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((CovSA2l(p1) / -0.5).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((CovSA2u(p1) / -0.5).base_matrix())>);
  static_assert(is_diagonal_v<decltype((CovD2 {1, 2} / -0.5).base_matrix())>);
  static_assert(is_diagonal_v<decltype((covi2 / -0.5).base_matrix())>);
  static_assert(is_zero_v<decltype((covz2 / -0.5).base_matrix())>);

  // Scale
  EXPECT_TRUE(is_near(scale(CovSA2l(p1), 2), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(scale(CovSA2u(p1), 2), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(scale(CovT2l(p1), 2), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(scale(CovT2u(p1), 2), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(scale(CovD2 {1, 2}, 2), Mat2 {4, 0, 0, 8}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(CovSA2l(p1), 2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(scale(CovSA2u(p1), 2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(scale(CovT2l(p1), 2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(scale(CovT2u(p1), 2).base_matrix())>);
  static_assert(is_diagonal_v<decltype(scale(CovD2 {1, 2}, 2).base_matrix())>);

  EXPECT_TRUE(is_near(inverse_scale(CovSA2l(p1), 0.5), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(inverse_scale(CovSA2u(p1), 0.5), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(inverse_scale(CovT2l(p1), 0.5), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(inverse_scale(CovT2u(p1), 0.5), Mat2 {16, 8, 8, 20}));
  EXPECT_TRUE(is_near(inverse_scale(CovD2 {4, 8}, 2), Mat2 {1, 0, 0, 2}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(inverse_scale(CovSA2l(p1), 2).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(inverse_scale(CovSA2u(p1), 2).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(inverse_scale(CovT2l(p1), 2).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(inverse_scale(CovT2u(p1), 2).base_matrix())>);
  static_assert(is_diagonal_v<decltype(inverse_scale(CovD2 {1, 2}, 2).base_matrix())>);

  // Rank-deficient case
  using M3 = Eigen::Matrix<double, 3, 3>;
  using Mat3 = TypedMatrix<Coefficients<Angle, Axis, Angle>, Coefficients<Angle, Axis, Angle>, M3>;
  TypedMatrix<Coefficients<Angle, Axis, Angle>, C> a1 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(scale(CovSA2l(p1), a1), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(scale(CovSA2u(p1), a1), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(scale(CovT2l(p1), a1), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(scale(CovT2u(p1), a1), Mat3 {32, 72, 112, 72, 164, 256, 112, 256, 400}));
  EXPECT_TRUE(is_near(scale(CovD2 {1, 2}, a1), Mat3 {9, 19, 29, 19, 41, 63, 29, 63, 97}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(CovSA2l(p1), a1).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(scale(CovSA2u(p1), a1).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(scale(CovT2l(p1), a1).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(scale(CovT2u(p1), a1).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(CovD2 {1, 2}, a1).base_matrix())>);

  // Rank-sufficient case
  using CovSA3l = Covariance<Coefficients<Angle, Axis, Angle>, EigenSelfAdjointMatrix<M3, TriangleType::lower>>;
  using CovSA3u = Covariance<Coefficients<Angle, Axis, Angle>, EigenSelfAdjointMatrix<M3, TriangleType::upper>>;
  using CovT3l = Covariance<Coefficients<Angle, Axis, Angle>, EigenTriangularMatrix<M3, TriangleType::lower>>;
  using CovT3u = Covariance<Coefficients<Angle, Axis, Angle>, EigenTriangularMatrix<M3, TriangleType::upper>>;
  using CovD3 = Covariance<Coefficients<Angle, Axis, Angle>, EigenDiagonal<Eigen::Matrix<double, 3, 1>>>;
  Mat3 q1 {4, 2, 2,
           2, 5, 3,
           2, 3, 6};
  TypedMatrix<C, Coefficients<Angle, Axis, Angle>> b1 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(scale(CovSA3l(q1), b1), Mat2 {134, 317, 317, 761}));
  EXPECT_TRUE(is_near(scale(CovSA3u(q1), b1), Mat2 {134, 317, 317, 761}));
  EXPECT_TRUE(is_near(scale(CovT3l(q1), b1), Mat2 {134, 317, 317, 761}));
  EXPECT_TRUE(is_near(scale(CovT3u(q1), b1), Mat2 {134, 317, 317, 761}));
  EXPECT_TRUE(is_near(scale(CovD3 {4, 5, 6}, b1), Mat2 {78, 174, 174, 405}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(CovSA3l(q1), b1).base_matrix())>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(scale(CovSA3u(q1), b1).base_matrix())>);
  static_assert(is_lower_triangular_v<decltype(scale(CovT3l(q1), b1).base_matrix())>);
  static_assert(is_upper_triangular_v<decltype(scale(CovT3u(q1), b1).base_matrix())>);
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(scale(CovD3 {4, 5, 6}, b1).base_matrix())>);
}


TEST_F(covariance_tests, TypedMatrix_mult_Covariance)
{
  using MatI2 = TypedMatrix<C, C, I2>;
  using MatZ2 = TypedMatrix<C, C, Z2>;
  auto mati2 = MatI2(i2);
  auto matz2 = MatZ2(z2);
  using Cx = Coefficients<Axis, Angle>;
  using MatI2x = TypedMatrix<Cx, C, I2>;
  auto mati2x = MatI2x(i2);

  auto covSA2l = CovSA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covSA2l, Mat2 {42, 32, 33, 56}));
  EXPECT_TRUE(is_near(mati2 * covSA2l, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(mati2x * covSA2l, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(matz2 * covSA2l, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype((mati2 * covSA2l).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * covSA2l).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covSA2l)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covSA2l)>::ColumnCoefficients, C>);

  auto covSA2u = CovSA2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covSA2u, Mat2 {42, 32, 33, 56}));
  EXPECT_TRUE(is_near(mati2 * covSA2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(mati2x * covSA2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(matz2 * covSA2u, Mat2 {0, 0, 0, 0}));
  static_assert(is_Eigen_upper_storage_triangle_v<decltype((mati2 * covSA2u).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * covSA2u).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covSA2u)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covSA2u)>::ColumnCoefficients, C>);

  auto covT2l = CovT2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covT2l, Mat2 {42, 32, 33, 56}));
  EXPECT_TRUE(is_near(mati2 * covT2l, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(matz2 * covT2l, Mat2 {0, 0, 0, 0}));
  static_assert(is_lower_triangular_v<decltype((mati2 * covT2l).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * covT2l).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covT2l)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covT2l)>::ColumnCoefficients, C>);

  auto covT2u = CovT2u {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covT2u, Mat2 {42, 32, 33, 56}));
  EXPECT_TRUE(is_near(mati2 * covT2u, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(matz2 * covT2u, Mat2 {0, 0, 0, 0}));
  static_assert(is_upper_triangular_v<decltype((mati2 * covT2u).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * covT2u).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covT2u)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covT2u)>::ColumnCoefficients, C>);

  auto covD2 = CovD2 {9, 10};
  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covD2, Mat2 {36, 20, 18, 50}));
  EXPECT_TRUE(is_near(mati2 * covD2, Mat2 {9, 0, 0, 10}));
  EXPECT_TRUE(is_near(matz2 * covD2, Mat2 {0, 0, 0, 0}));
  static_assert(is_diagonal_v<decltype((mati2 * covD2).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * covD2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covD2)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covD2)>::ColumnCoefficients, C>);

  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covi2, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(mati2 * covi2, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(matz2 * covi2, Mat2 {0, 0, 0, 0}));
  static_assert(is_identity_v<decltype((mati2 * covi2).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * covi2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covi2)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covi2)>::ColumnCoefficients, C>);

  EXPECT_TRUE(is_near(Mat2 {4, 2, 2, 5} * covz2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(mati2 * covz2, Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(matz2 * covz2, Mat2 {0, 0, 0, 0}));
  static_assert(is_zero_v<decltype((mati2 * covz2).base_matrix())>);
  static_assert(is_zero_v<decltype((matz2 * covz2).base_matrix())>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covz2)>::RowCoefficients, Cx>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(mati2x * covz2)>::ColumnCoefficients, C>);
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
  static_assert(is_zero_v<decltype((-sqcovz2).base_matrix())>);

  EXPECT_TRUE((CovSA2l {9, 3, 3, 10} == CovSA2l {9, 3, 3, 10}));
  EXPECT_TRUE((CovSA2l {9, 3, 3, 10} == CovT2u {9, 3, 3, 10}));
  EXPECT_TRUE((CovSA2u {9, 3, 3, 10} != CovSA2u {9, 2, 3, 10}));
  EXPECT_TRUE((CovSA2l {9, 3, 3, 10} != Covariance<Axes<2>, SA2l> {9, 3, 3, 10}));
}


TEST_F(covariance_tests, Covariance_references_self_adjoint)
{
  using V = Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 4, -6,
        3, -6, -3};
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  EXPECT_EQ(v1(0,1), 4.1);
  v2(0, 2) = 5.2;
  EXPECT_EQ(v1(0,2), 5.2);
  EXPECT_EQ(v1(2,0), 5.2);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>&&>> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<const Eigen::Matrix<double, 3, 3>&>> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}


TEST_F(covariance_tests, Covariance_references_self_adjoint2)
{
  using V = Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 4, -6,
        3, -6, -3};
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  EXPECT_EQ(v1(0,1), 4.1);
  v2(0, 2) = 5.2;
  EXPECT_EQ(v1(0,2), 5.2);
  EXPECT_EQ(v1(2,0), 5.2);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&&> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  Covariance<Coefficients<Axis, Angle, Axis>, const EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}


TEST_F(covariance_tests, Covariance_references_triangular)
{
  using V = Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 20, -18,
        3, -18, 54};
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1 = {1.1, 2.1, 3.1,
        2.1, 20.1, -18.1,
        3.1, -18.1, 54.1};
  EXPECT_NEAR(v2(1,0), 2.1, 1e-6);
  v2 = V {1.2, 2.2, 3.2,
          2.2, 20.2, -18.2,
          3.2, -18.2, 54.2};
  EXPECT_NEAR(v1(2,0), 3.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>> v2_5 = v2;
  EXPECT_NEAR(v2_5(1,0), 2.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&&>> v3 = std::move(v2);
  EXPECT_NEAR(v3(1,0), 2.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<const Eigen::Matrix<double, 3, 3>&>> v4 = v3;
  v3 = V {1.3, 2.3, 3.3,
          2.3, 20.3, -18.3,
          3.3, -18.3, 54.3};
  EXPECT_NEAR(v3(2,2), 54.3, 1e-6);
  EXPECT_NEAR(v4(2,1), -18.3, 1e-6);
  V v5 = v3;
  EXPECT_NEAR(v3(1,1), 20.3, 1e-6);
  v5 = {1.4, 2.4, 3.4,
        2.4, 20.4, -18.4,
        3.4, -18.4, 54.4};
  EXPECT_NEAR(v3(1,1), 20.3, 1e-6);
  EXPECT_NEAR(v5(1,1), 20.4, 1e-6);
  EXPECT_NEAR(v3(2,1), -18.3, 1e-6);
  EXPECT_TRUE(is_near(v3, V {1.3, 2.3, 3.3,
                             2.3, 20.3, -18.3,
                             3.3, -18.3, 54.3}));
}


TEST_F(covariance_tests, Covariance_references_triangular2)
{
  using V = Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 20, -18,
        3, -18, 54};
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1 = {1.1, 2.1, 3.1,
        2.1, 20.1, -18.1,
        3.1, -18.1, 54.1};
  EXPECT_NEAR(v2(1,0), 2.1, 1e-6);
  v2 = V {1.2, 2.2, 3.2,
          2.2, 20.2, -18.2,
          3.2, -18.2, 54.2};
  EXPECT_NEAR(v1(2,0), 3.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&&> v3 = std::move(v2);
  EXPECT_NEAR(v3(1,0), 2.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, const EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&> v4 = v3;
  v3 = V {1.3, 2.3, 3.3,
          2.3, 20.3, -18.3,
          3.3, -18.3, 54.3};
  EXPECT_NEAR(v4(2,1), -18.3, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v5 = {1.4, 2.4, 3.4,
        2.4, 20.4, -18.4,
        3.4, -18.4, 54.4};
  EXPECT_NEAR(v3(1,1), 20.3, 1e-6);
  EXPECT_NEAR(v5(1,1), 20.4, 1e-6);
  EXPECT_NEAR(v3(2,1), -18.3, 1e-6);
  EXPECT_TRUE(is_near(v3, V {1.3, 2.3, 3.3,
                             2.3, 20.3, -18.3,
                             3.3, -18.3, 54.3}));
}

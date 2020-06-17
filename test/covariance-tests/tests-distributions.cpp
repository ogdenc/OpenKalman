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

using M1 = Eigen::Matrix<double, 1, 1>;
using M2 = Eigen::Matrix<double, 2, 2>;
using M2col = Eigen::Matrix<double, 2, 1>;
using M3 = Eigen::Matrix<double, 3, 3>;
using M3col = Eigen::Matrix<double, 3, 1>;
using C2 = Coefficients<Angle, Axis>;
using C3 = Coefficients<Angle, Axis, Axis>;
using Mean2 = Mean<C2, M2col>;
using Mean3 = Mean<C3, M3col>;
using Mat2 = TypedMatrix<C2, C2, M2>;
using Mat3 = TypedMatrix<C3, C3, M3>;
using SA1l = EigenSelfAdjointMatrix<M1, TriangleType::lower>;
using SA2l = EigenSelfAdjointMatrix<M2, TriangleType::lower>;
using SA2u = EigenSelfAdjointMatrix<M2, TriangleType::upper>;
using T1l = EigenTriangularMatrix<M1, TriangleType::lower>;
using T2l = EigenTriangularMatrix<M2, TriangleType::lower>;
using T2u = EigenTriangularMatrix<M2, TriangleType::upper>;
using D2 = EigenDiagonal<M2col>;
using I2 = EigenIdentity<M2>;
using Z2 = EigenZero<M2>;
using CovSA2l = Covariance<C2, SA2l>;
using CovSA2u = Covariance<C2, SA2u>;
using CovT2l = Covariance<C2, T2l>;
using CovT2u = Covariance<C2, T2u>;
using CovD2 = Covariance<C2, D2>;
using CovI2 = Covariance<C2, I2>;
using CovZ2 = Covariance<C2, Z2>;
using DistSA2l = GaussianDistribution<C2, M2col, SA2l>;
using DistSA2u = GaussianDistribution<C2, M2col, SA2u>;
using DistT2l = GaussianDistribution<C2, M2col, T2l>;
using DistT2u = GaussianDistribution<C2, M2col, T2u>;
using DistD2 = GaussianDistribution<C2, M2col, D2>;
using DistI2 = GaussianDistribution<C2, EigenZero<M2col>, I2>;
using DistZ2 = GaussianDistribution<C2, EigenZero<M2col>, Z2>;

inline I2 i2 = M2::Identity();
inline Z2 z2 = EigenZero<M2>();
inline auto covi2 = CovI2(i2);
inline auto covz2 = CovZ2(z2);
inline auto disti2 = DistI2(EigenZero<M2col>(), covi2);
inline auto distz2 = DistZ2(EigenZero<M2col>(), covz2);

TEST_F(covariance_tests, GaussianDistribution_class)
{
  // Default constructor
  DistSA2l distSA2la;
  mean(distSA2la) = Mean2 {1, 2};
  covariance(distSA2la) = SA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(mean(distSA2la), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2la), Mat2 {9, 3, 3, 10}));

  // Copy constructor
  DistSA2l distSA2lb = const_cast<const DistSA2l&>(distSA2la);
  EXPECT_TRUE(is_near(mean(distSA2lb), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2lb), Mat2 {9, 3, 3, 10}));

  // Move constructor
  DistSA2l distSA2lc(std::move(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}}));
  EXPECT_TRUE(is_near(mean(distSA2lc), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2lc), Mat2 {9, 3, 3, 10}));

  // Convert from a related Gaussian distribution
  DistSA2l distSA2ld_1(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ld_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_1), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2ld_2(DistSA2u {Mean2 {1, 2}, CovSA2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ld_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_2), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2ld_3(DistT2l {Mean2 {1, 2}, CovT2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ld_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_3), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2ld_4(DistT2u {Mean2 {1, 2}, CovT2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ld_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_4), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2ld_5(DistD2 {Mean2 {1, 2}, CovD2 {9, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ld_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_5), Mat2 {9, 0, 0, 10}));
  DistSA2l distSA2ld_6(DistI2 {EigenZero<M2col>(), covi2});
  EXPECT_TRUE(is_near(mean(distSA2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_6), Mat2 {1, 0, 0, 1}));
  DistSA2l distSA2ld_7(DistZ2 {EigenZero<M2col>(), covz2});
  EXPECT_TRUE(is_near(mean(distSA2ld_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_7), Mat2 {0, 0, 0, 0}));
  //
  DistSA2u distSA2ud_1(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ud_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_1), Mat2 {9, 3, 3, 10}));
  DistSA2u distSA2ud_2(DistSA2u {Mean2 {1, 2}, CovSA2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ud_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_2), Mat2 {9, 3, 3, 10}));
  DistSA2u distSA2ud_3(DistT2l {Mean2 {1, 2}, CovT2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ud_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_3), Mat2 {9, 3, 3, 10}));
  DistSA2u distSA2ud_4(DistT2u {Mean2 {1, 2}, CovT2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ud_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_4), Mat2 {9, 3, 3, 10}));
  DistSA2u distSA2ud_5(DistD2 {Mean2 {1, 2}, CovD2 {9, 10}});
  EXPECT_TRUE(is_near(mean(distSA2ud_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_5), Mat2 {9, 0, 0, 10}));
  DistSA2u distSA2ud_6(DistI2 {EigenZero<M2col>(), covi2});
  EXPECT_TRUE(is_near(mean(distSA2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_6), Mat2 {1, 0, 0, 1}));
  DistSA2u distSA2ud_7(DistZ2 {EigenZero<M2col>(), covz2});
  EXPECT_TRUE(is_near(mean(distSA2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_7), Mat2 {0, 0, 0, 0}));
  //
  DistT2l distT2ld_1(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distT2ld_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ld_1), Mat2 {9, 3, 3, 10}));
  DistT2l distT2ld_2(DistSA2u {Mean2 {1, 2}, CovSA2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distT2ld_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ld_2), Mat2 {9, 3, 3, 10}));
  DistT2l distT2ld_3(DistT2l {Mean2 {1, 2}, CovT2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distT2ld_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ld_3), Mat2 {9, 3, 3, 10}));
  DistT2l distT2ld_4(DistT2u {Mean2 {1, 2}, CovT2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distT2ld_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ld_4), Mat2 {9, 3, 3, 10}));
  DistT2l distT2ld_5(DistD2 {Mean2 {1, 2}, CovD2 {9, 10}});
  EXPECT_TRUE(is_near(mean(distT2ld_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ld_5), Mat2 {9, 0, 0, 10}));
  DistT2l distT2ld_6(DistI2 {EigenZero<M2col>(), covi2});
  EXPECT_TRUE(is_near(mean(distT2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2ld_6), Mat2 {1, 0, 0, 1}));
  DistT2l distT2ld_7(DistZ2 {EigenZero<M2col>(), covz2});
  EXPECT_TRUE(is_near(mean(distT2ld_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2ld_7), Mat2 {0, 0, 0, 0}));
  //
  DistT2u distT2ud_1(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distT2ud_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ud_1), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ud_2(DistSA2u {Mean2 {1, 2}, CovSA2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distT2ud_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ud_2), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ud_3(DistT2l {Mean2 {1, 2}, CovT2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distT2ud_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ud_3), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ud_4(DistT2u {Mean2 {1, 2}, CovT2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean(distT2ud_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ud_4), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ud_5(DistD2 {Mean2 {1, 2}, CovD2 {9, 10}});
  EXPECT_TRUE(is_near(mean(distT2ud_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ud_5), Mat2 {9, 0, 0, 10}));
  DistT2u distT2ud_6(DistI2 {EigenZero<M2col>(), covi2});
  EXPECT_TRUE(is_near(mean(distT2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2ud_6), Mat2 {1, 0, 0, 1}));
  DistT2u distT2ud_7(DistZ2 {EigenZero<M2col>(), covz2});
  EXPECT_TRUE(is_near(mean(distT2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2ud_7), Mat2 {0, 0, 0, 0}));

  // Construct from different combinations of mean, typed matrix base, covariance, and covariance base.
  DistSA2l distSA2le_1(Mean2 {1, 2}, SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean(distSA2le_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2le_1), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_2((M2col() << 1, 2).finished(), CovSA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean(distSA2le_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2le_2), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_3((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean(distSA2le_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distSA2le_3), Mat2 {9, 3, 3, 10}));
  DistT2l distT2le(Mean2 {1, 2}, T2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(mean(distT2le), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2le), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ue(Mean2 {1, 2}, T2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(mean(distT2ue), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distT2ue), Mat2 {9, 3, 3, 10}));
  DistD2 distD2e(Mean2 {1, 2}, D2 {9, 10});
  EXPECT_TRUE(is_near(mean(distD2e), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance(distD2e), Mat2 {9, 0, 0, 10}));
  DistI2 distI2e(EigenZero<M2col>(), i2);
  EXPECT_TRUE(is_near(mean(distI2e), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distI2e), Mat2 {1, 0, 0, 1}));
  DistZ2 distZ2e(EigenZero<M2col>(), z2);
  EXPECT_TRUE(is_near(mean(distZ2e), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distZ2e), Mat2 {0, 0, 0, 0}));

  // Construct from only covariance or covariance base.
  DistSA2l distSA2lf_1(CovSA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean(distSA2lf_1), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2lf_1), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2lf_2(SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean(distSA2lf_2), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2lf_2), Mat2 {9, 3, 3, 10}));
  DistT2l distT2lf(T2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(mean(distT2lf), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2lf), Mat2 {9, 3, 3, 10}));
  DistT2u distT2uf(T2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(mean(distT2uf), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2uf), Mat2 {9, 3, 3, 10}));
  DistD2 distD2f(D2 {9, 10});
  EXPECT_TRUE(is_near(mean(distD2f), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distD2f), Mat2 {9, 0, 0, 10}));
  DistI2 distI2f(i2);
  EXPECT_TRUE(is_near(mean(distI2f), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distI2f), Mat2 {1, 0, 0, 1}));
  DistZ2 distZ2f(z2);
  EXPECT_TRUE(is_near(mean(distZ2f), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distZ2f), Mat2 {0, 0, 0, 0}));

  // Construct using lists of parameters for mean, covariance, or both.
  DistSA2l distSA2lg_1{{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2lg_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2lg_1), Mat2 {4, 2, 2, 5}));
  DistSA2l distSA2lg_2{Mean2 {3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2lg_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2lg_2), Mat2 {4, 2, 2, 5}));
  DistSA2l distSA2lg_3{{3, 4}, CovSA2l {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2lg_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2lg_3), Mat2 {4, 2, 2, 5}));
  DistSA2l distSA2lg_4{{3, 4}, SA2l {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2lg_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2lg_4), Mat2 {4, 2, 2, 5}));

  // Copy assignment
  distSA2lb = distSA2lg_1;
  EXPECT_TRUE(is_near(mean(distSA2lb), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2lb), Mat2 {4, 2, 2, 5}));

  // Move assignment
  distSA2lc = std::move(DistSA2l {{3, 4}, {4, 2, 2, 5}});
  EXPECT_TRUE(is_near(mean(distSA2lc), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2lc), Mat2 {4, 2, 2, 5}));

  // Assign from different distribution type
  distSA2ld_1 = DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ld_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_1), Mat2 {4, 2, 2, 5}));
  distSA2ld_2 = DistSA2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ld_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_2), Mat2 {4, 2, 2, 5}));
  distSA2ld_3 = DistT2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ld_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_3), Mat2 {4, 2, 2, 5}));
  distSA2ld_4 = DistT2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ld_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_4), Mat2 {4, 2, 2, 5}));
  distSA2ld_5 = DistD2 {{3, 4}, {4, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ld_5), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_5), Mat2 {4, 0, 0, 5}));
  distSA2ld_6 = DistI2 {{}, covi2};
  EXPECT_TRUE(is_near(mean(distSA2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_6), Mat2 {1, 0, 0, 1}));
  distSA2ld_7 = DistZ2 {{}, covz2};
  EXPECT_TRUE(is_near(mean(distSA2ld_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2ld_7), Mat2 {0, 0, 0, 0}));
  //
  distSA2ud_1 = DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ud_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_1), Mat2 {4, 2, 2, 5}));
  distSA2ud_2 = DistSA2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ud_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_2), Mat2 {4, 2, 2, 5}));
  distSA2ud_3 = DistT2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ud_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_3), Mat2 {4, 2, 2, 5}));
  distSA2ud_4 = DistT2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ud_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_4), Mat2 {4, 2, 2, 5}));
  distSA2ud_5 = DistD2 {{3, 4}, {4, 5}};
  EXPECT_TRUE(is_near(mean(distSA2ud_5), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_5), Mat2 {4, 0, 0, 5}));
  distSA2ud_6 = DistI2 {{}, covi2};
  EXPECT_TRUE(is_near(mean(distSA2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_6), Mat2 {1, 0, 0, 1}));
  distSA2ud_7 = DistZ2 {{}, covz2};
  EXPECT_TRUE(is_near(mean(distSA2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distSA2ud_7), Mat2 {0, 0, 0, 0}));
  //
  distT2ld_1 = DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ld_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ld_1), Mat2 {4, 2, 2, 5}));
  distT2ld_2 = DistSA2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ld_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ld_2), Mat2 {4, 2, 2, 5}));
  distT2ld_3 = DistT2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ld_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ld_3), Mat2 {4, 2, 2, 5}));
  distT2ld_4 = DistT2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ld_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ld_4), Mat2 {4, 2, 2, 5}));
  distT2ld_5 = DistD2 {{3, 4}, {4, 5}};
  EXPECT_TRUE(is_near(mean(distT2ld_5), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ld_5), Mat2 {4, 0, 0, 5}));
  distT2ld_6 = DistI2 {{}, covi2};
  EXPECT_TRUE(is_near(mean(distT2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2ld_6), Mat2 {1, 0, 0, 1}));
  distT2ld_7 = DistZ2 {{}, covz2};
  EXPECT_TRUE(is_near(mean(distT2ld_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2ld_7), Mat2 {0, 0, 0, 0}));
  //
  distT2ud_1 = DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ud_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ud_1), Mat2 {4, 2, 2, 5}));
  distT2ud_2 = DistSA2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ud_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ud_2), Mat2 {4, 2, 2, 5}));
  distT2ud_3 = DistT2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ud_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ud_3), Mat2 {4, 2, 2, 5}));
  distT2ud_4 = DistT2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ud_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ud_4), Mat2 {4, 2, 2, 5}));
  distT2ud_5 = DistD2 {{3, 4}, {4, 5}};
  EXPECT_TRUE(is_near(mean(distT2ud_5), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ud_5), Mat2 {4, 0, 0, 5}));
  distT2ud_6 = DistI2 {{}, covi2};
  EXPECT_TRUE(is_near(mean(distT2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2ud_6), Mat2 {1, 0, 0, 1}));
  distT2ud_7 = DistZ2 {{}, covz2};
  EXPECT_TRUE(is_near(mean(distT2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance(distT2ud_7), Mat2 {0, 0, 0, 0}));

  // Assign from a list of coefficients (via move assignment operator)
  distSA2la = {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distSA2la), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distSA2la), Mat2 {4, 2, 2, 5}));

  // Increment
  distT2le += {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2le), Mean2 {4, 6}));
  EXPECT_TRUE(is_near(covariance(distT2le), Mat2 {13, 5, 5, 15}));
  distT2ue += DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean(distT2ue), Mean2 {4, 6}));
  EXPECT_TRUE(is_near(covariance(distT2ue), Mat2 {13, 5, 5, 15}));

  // Decrement
  distT2le -= {{1, 2}, {9, 3, 3, 10}};
  EXPECT_TRUE(is_near(mean(distT2le), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2le), Mat2 {4, 2, 2, 5}));
  distT2ue -= DistSA2l {{1, 2}, {9, 3, 3, 10}};
  EXPECT_TRUE(is_near(mean(distT2ue), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance(distT2ue), Mat2 {4, 2, 2, 5}));

  // Scalar multiplication
  distT2le *= 2;
  EXPECT_TRUE(is_near(mean(distT2le), Mean2 {6-2*M_PI, 8}));
  EXPECT_TRUE(is_near(covariance(distT2le), Mat2 {16, 8, 8, 20}));

  // Scalar division
  distT2le /= 2;
  EXPECT_TRUE(is_near(mean(distT2le), Mean2 {3-M_PI, 4}));
  EXPECT_TRUE(is_near(covariance(distT2le), Mat2 {4, 2, 2, 5}));

  // Zero
  EXPECT_TRUE(is_near(DistSA2l::zero(), distz2));
  static_assert(is_zero_v<decltype(DistSA2l::zero())>);
  EXPECT_TRUE(is_near(DistSA2u::zero(), distz2));
  static_assert(is_zero_v<decltype(DistSA2u::zero())>);
  EXPECT_TRUE(is_near(DistT2l::zero(), distz2));
  static_assert(is_zero_v<decltype(DistT2l::zero())>);
  EXPECT_TRUE(is_near(DistT2u::zero(), distz2));
  static_assert(is_zero_v<decltype(DistT2u::zero())>);
  EXPECT_TRUE(is_near(DistD2::zero(), distz2));
  static_assert(is_zero_v<decltype(DistD2::zero())>);

  // Normal
  EXPECT_TRUE(is_near(DistSA2l::normal(), disti2));
  static_assert(is_zero_v<decltype(mean(DistSA2l::normal()))>);
  static_assert(is_identity_v<decltype(covariance(DistSA2l::normal()))>);
  EXPECT_TRUE(is_near(DistSA2u::normal(), disti2));
  static_assert(is_zero_v<decltype(mean(DistSA2u::normal()))>);
  static_assert(is_identity_v<decltype(covariance(DistSA2u::normal()))>);
  EXPECT_TRUE(is_near(DistT2l::normal(), disti2));
  static_assert(is_zero_v<decltype(mean(DistT2l::normal()))>);
  static_assert(is_identity_v<decltype(covariance(DistT2l::normal()))>);
  EXPECT_TRUE(is_near(DistT2u::normal(), disti2));
  static_assert(is_zero_v<decltype(mean(DistT2u::normal()))>);
  static_assert(is_identity_v<decltype(covariance(DistT2u::normal()))>);
  EXPECT_TRUE(is_near(DistD2::normal(), disti2));
  static_assert(is_zero_v<decltype(mean(DistD2::normal()))>);
  static_assert(is_identity_v<decltype(covariance(DistD2::normal()))>);
}


TEST_F(covariance_tests, GaussianDistribution_class_random)
{
  using V = Mean<C3>;
  M3 d;
  d << 0.9, 0.1, 0.3,
       0.1, 1.4, 0.45,
       0.3, 0.45, 1.1;
  const V true_x {M_PI * 99/100, 10, 5};
  GaussianDistribution dist {true_x, make_Covariance<C3>(d)};
  const V x1 {dist()};
  const V x2 {dist()};
  EXPECT_NE(x1, x2);
  using EV = EuclideanMean<C3>;
  EV mean_x = EV::zero();
  for (int i = 0; i < 1000; i++)
  {
    V x {dist()};
    mean_x = (mean_x * i + to_Euclidean(x)) / (i + 1);
  }
  EXPECT_NE(from_Euclidean(mean_x), true_x);
  EXPECT_TRUE(is_near(from_Euclidean(mean_x) - true_x, V::zero(), MatrixTraits<V>::BaseMatrix::Constant(0.1)));
}


TEST_F(covariance_tests, GaussianDistribution_class_Cholesky_random)
{
  using V = Mean<C3>;
  M3 d;
  d << 0.9, 0.1, 0.3,
       0.1, 1.4, 0.45,
       0.3, 0.45, 1.1;
  const V true_x {M_PI * 99/100, 10, 5};
  GaussianDistribution dist {true_x, make_Covariance<C3, TriangleType::lower>(d)};
  const V x1 {dist()};
  const V x2 {dist()};
  EXPECT_NE(x1, x2);
  using EV = EuclideanMean<C3>;
  EV mean_x = EV::zero();
  for (int i = 0; i < 1000; i++)
  {
    V x {dist()};
    mean_x = (mean_x * i + to_Euclidean(x)) / (i + 1);
  }
  EXPECT_NE(from_Euclidean(mean_x), true_x);
  EXPECT_TRUE(is_near(from_Euclidean(mean_x) - true_x, V::zero(), MatrixTraits<V>::BaseMatrix::Constant(0.1)));
}


TEST_F(covariance_tests, GaussianDistribution_class_statistics)
{
  GaussianDistribution<Axis, M1, T1l> x1 = {M1(2), T1l(M1(3))};
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(1)}), -2.07310637743, 1e-6);
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(0)}), -2.2397730441, 1e-6);
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(1)}, Mean{M1(0)}), -2.07310637743 - 2.2397730441, 1e-6);

  GaussianDistribution<Axes<2>, M2col, T2l> x2 = {{0, 0}, {1, 0, 0.3, 1}};
  EXPECT_NEAR(x2.log_likelihood(Mean {1.5, 0.9}), -3.02698546294, 1e-6);

  GaussianDistribution<Axes<2>, M2col, T2l> x3 = {{0, 0}, {1, 0, 0.8, 1}};
  EXPECT_NEAR(x3.log_likelihood(Mean {2, 0.7}), -4.45205144264, 1e-6);
  EXPECT_NEAR(x3.log_likelihood(Mean {1.5, 0.9}), -2.5770514426433544, 1e-6);
  EXPECT_NEAR(x3.log_likelihood(Mean {2, 0.7}, Mean {1.5, 0.9}), -4.45205144264 - 2.5770514426433544, 1e-6);

  GaussianDistribution<Axis, M1, T1l> x4 = {M1(0), T1l(M1(1))};
  EXPECT_NEAR(x4.entropy(), 2.04709558518, 1e-6);

  GaussianDistribution<Axis, M1, T1l> x5 = {M1(0), T1l(M1(0.5))};
  EXPECT_NEAR(x5.entropy(), 1.04709558518, 1e-6);
}


TEST_F(covariance_tests, Distribution_construction_angle)
{
  Mean<C3> x_mean {M_PI / 3, 10, 5};
  Covariance<C3> c;
  c << 9, 6, 3,
       6, 29, 4.5,
       3, 4.5, 65.25;
  auto c_chol = make_Covariance<C3, TriangleType::lower>(Cholesky_factor(base_matrix(c)));
  GaussianDistribution dist {x_mean, c};
  GaussianDistribution dist_chol {x_mean, c_chol};
  auto x = strict_matrix(x_mean);
  auto m = strict_matrix(c);
  auto m_sqrt = strict_matrix(square_root(c_chol));
  EXPECT_TRUE(is_near(dist, dist_chol));
  EXPECT_TRUE(is_near(mean(dist), x));
  EXPECT_TRUE(is_near(M3 {covariance(dist)}, m));
  EXPECT_TRUE(is_near(M3 {covariance(dist_chol)}, m));
  EXPECT_TRUE(is_near(M3 {square_root(covariance(dist))}, m_sqrt));
  EXPECT_TRUE(is_near(M3 {square_root(covariance(dist_chol))}, m_sqrt));
  EXPECT_TRUE(is_near(M3 {to_Cholesky(covariance(dist))}, m));
  EXPECT_TRUE(is_near(M3 {from_Cholesky(covariance(dist_chol))}, m));
}

TEST_F(covariance_tests, Distribution_scale_angle)
{
  auto a = GaussianDistribution(make_Mean<C2>(20., 30), make_Covariance<C2>(8., 2, 2, 6));
  auto a_chol = GaussianDistribution(make_Mean<C2>(20., 30), make_Covariance<C2, TriangleType::lower>(8., 2, 2, 6));
  Eigen::Matrix<double, 3, 2> f_mat; f_mat << 1, 2, 3, 4, 5, 6;
  auto f_vector = make_Mean<Coefficients<Axis, Angle, Angle>>(f_mat);
  auto f_matrix = make_Matrix<Coefficients<Axis, Angle, Angle>, C2>(f_mat);
  Eigen::Matrix<double, 3, 1> fa_mean; fa_mean << 80, 180, 280;
  Eigen::Matrix<double, 3, 1> fa_mean_wrapped; fa_mean_wrapped << 20-6*M_PI + 60, 180 - M_PI*58, 280 - M_PI*90;
  Eigen::Matrix<double, 3, 3> faf_cov; faf_cov << 40, 92, 144, 92, 216, 340, 144, 340, 536;
  auto a_scaled3 = f_matrix * a;
  static_assert(std::is_same_v<typename decltype(a_scaled3)::Coefficients, typename decltype(f_matrix)::RowCoefficients>);
  EXPECT_TRUE(is_near(mean(a_scaled3), fa_mean_wrapped));
  EXPECT_TRUE(is_near(covariance(a_scaled3), faf_cov));
  auto a_chol_scaled3 = f_matrix * a_chol;
  static_assert(std::is_same_v<typename decltype(a_chol_scaled3)::Coefficients, typename decltype(f_matrix)::RowCoefficients>);
  EXPECT_TRUE(is_near(mean(a_chol_scaled3), fa_mean_wrapped));
  EXPECT_TRUE(is_near(covariance(a_chol_scaled3), faf_cov));
}


TEST_F(covariance_tests, Distribution_concatenate_axis)
{
using C2a = Coefficients<Axis, Axis>;
Mean<C2a> x_mean {20, 30};
Covariance<C2a> x_cov;
x_cov << 9, 3, 3, 8;
GaussianDistribution distx {x_mean, x_cov};
GaussianDistribution distx_sqrt {x_mean, to_Cholesky(x_cov)};
Mean<C2a> y_mean {11, 23};
Covariance<C2a> y_cov;
y_cov << 7, 1, 1, 3;
GaussianDistribution disty {y_mean, y_cov};
GaussianDistribution disty_sqrt {y_mean, to_Cholesky(y_cov)};
Mean<Axes<4>> z_mean {20, 30, 11, 23};
Eigen::Matrix<double, 4, 4> z_cov;
z_cov <<
9, 3, 0, 0,
3, 8, 0, 0,
0, 0, 7, 1,
0, 0, 1, 3;
GaussianDistribution distz {z_mean, Covariance(z_cov)};
GaussianDistribution distz_sqrt {z_mean, to_Cholesky(Covariance(z_cov))};
EXPECT_TRUE(is_near(concatenate(distx, disty), distz));
EXPECT_TRUE(is_near(concatenate(distx_sqrt, disty_sqrt), distz_sqrt));
EXPECT_TRUE(is_near(split<C2a, C2a>(distz), std::tuple(distx, disty)));
EXPECT_TRUE(is_near(split<C2a, C2a>(distz_sqrt), std::tuple(distx_sqrt, disty_sqrt)));
}


TEST_F(covariance_tests, Distribution_construction_axis)
{
Mean<Axes<2>> x_mean {20, 30};
M2 d, d2;
d << 3, 0,
1, 3;
d2 << 9, 3,
3, 10;
GaussianDistribution dist {x_mean, EigenSelfAdjointMatrix(d2)};
EXPECT_TRUE(is_near(mean(dist), x_mean));
EXPECT_TRUE(is_near(covariance(dist), d2));
EXPECT_TRUE(is_near(square_root(covariance(dist)), d));
GaussianDistribution dist_chol {x_mean, EigenTriangularMatrix(d)};
EXPECT_TRUE(is_near(mean(dist_chol), x_mean));
EXPECT_TRUE(is_near(covariance(dist_chol), d2));
EXPECT_TRUE(is_near(square_root(covariance(dist_chol)), d));
}


TEST_F(covariance_tests, Distribution_addition_subtraction_axis)
{
Mean<Axes<2>> x_mean {20, 30};
M2 d;
d << 9, 3,
3, 8;
GaussianDistribution dist1 {x_mean, EigenSelfAdjointMatrix(d)};
Mean<Axes<2>> y_mean {11, 23};
M2 e;
e << 7, 1,
1, 3;
GaussianDistribution dist2 {y_mean, EigenSelfAdjointMatrix(e)};
auto sum1 = dist1 + dist2;
EXPECT_TRUE(is_near(mean(sum1), Mean {31., 53}));
EXPECT_TRUE(is_near(covariance(sum1), Covariance {16., 4, 4, 11}));
auto diff1 = dist1 - dist2;
EXPECT_TRUE(is_near(mean(diff1), Mean {9., 7.}));
EXPECT_TRUE(is_near(covariance(diff1), Covariance {2., 2, 2, 5}));
GaussianDistribution dist1_chol {x_mean, EigenTriangularMatrix(EigenSelfAdjointMatrix(d))};
GaussianDistribution dist2_chol {y_mean, EigenTriangularMatrix(EigenSelfAdjointMatrix(e))};
auto sum2 = dist1_chol + dist2_chol;
EXPECT_TRUE(is_near(mean(sum2), Mean {31., 53}));
EXPECT_TRUE(is_near(covariance(sum2), Covariance {16., 4, 4, 11}));
auto diff2 = dist1_chol - dist2_chol;
EXPECT_TRUE(is_near(mean(diff2), Mean {9., 7.}));
EXPECT_TRUE(is_near(covariance(diff2), Covariance {2., 2, 2, 5}));
auto sum3 = dist1 + dist2_chol;
EXPECT_TRUE(is_near(mean(sum3), Mean {31., 53}));
EXPECT_TRUE(is_near(covariance(sum3), Covariance {16., 4, 4, 11}));
auto diff3 = dist1 - dist2_chol;
EXPECT_TRUE(is_near(mean(diff3), Mean {9., 7.}));
EXPECT_TRUE(is_near(covariance(diff3), Covariance {2., 2, 2, 5}));
auto sum4 = dist1_chol + dist2;
EXPECT_TRUE(is_near(mean(sum4), Mean {31., 53}));
EXPECT_TRUE(is_near(covariance(sum4), Covariance {16., 4, 4, 11}));
auto diff4 = dist1_chol - dist2;
EXPECT_TRUE(is_near(mean(diff4), Mean {9., 7.}));
EXPECT_TRUE(is_near(covariance(diff4), Covariance {2., 2, 2, 5}));
}


TEST_F(covariance_tests, Distribution_scale_axis)
{
auto a = GaussianDistribution(Mean{20., 30}, Covariance{8., 2, 2, 6});
auto a_chol = GaussianDistribution(Mean{20., 30}, to_Cholesky(covariance(a)));
Eigen::Matrix<double, 2, 1> mean_mat {20, 30};
Eigen::Matrix<double, 2, 2> cov_mat; cov_mat << 8, 2, 2, 6;
decltype(a) a_scaled = a * 2;
EXPECT_TRUE(is_near(mean(a_scaled), mean_mat * 2));
EXPECT_TRUE(is_near(covariance(a_scaled), cov_mat * 4));
decltype(a_chol) a_chol_scaled = a_chol * 2;
EXPECT_TRUE(is_near(mean(a_chol_scaled), mean_mat * 2));
EXPECT_TRUE(is_near(covariance(a_chol_scaled), cov_mat * 4));
a_scaled = 2 * a;
EXPECT_TRUE(is_near(mean(a_scaled), 2 * mean_mat));
EXPECT_TRUE(is_near(covariance(a_scaled), 4 * cov_mat));
a_chol_scaled = 2 * a_chol;
EXPECT_TRUE(is_near(mean(a_chol_scaled), 2 * mean_mat));
EXPECT_TRUE(is_near(covariance(a_chol_scaled), 4 * cov_mat));
a_scaled = a / 2;
EXPECT_TRUE(is_near(mean(a_scaled), mean_mat / 2));
EXPECT_TRUE(is_near(covariance(a_scaled), cov_mat / 4));
a_chol_scaled = a_chol / 2;
EXPECT_TRUE(is_near(mean(a_chol_scaled), mean_mat / 2));
EXPECT_TRUE(is_near(covariance(a_chol_scaled), cov_mat / 4));
}


TEST_F(covariance_tests, Distribution_Gaussian_random_axis)
{
using Mat = Mean<Coefficients<Axis, Axis>>;
const Mat true_x {20, 30};
GaussianDistribution dist {true_x, Covariance(9., 3, 3, 10)};
const Mat x1 {dist()};
const Mat x2 {dist()};
EXPECT_NE(x1, x2);
using EMat = EuclideanMean<Coefficients<Axis, Axis>>;
EMat mean_x = EMat::zero();
for (int i = 0; i < 100; i++)
{
Mat x {dist()};
mean_x = (mean_x * i + to_Euclidean(x)) / (i + 1);
}
EXPECT_NE(from_Euclidean(mean_x), true_x);
EXPECT_TRUE(is_near(from_Euclidean(mean_x), true_x, MatrixTraits<Mat>::BaseMatrix::Constant(0.5)));
}


TEST_F(covariance_tests, Distribution_Gaussian_Cholesky_random_axis)
{
using Mat = Mean<Coefficients<Axis, Axis>>;
M2 d;
d << 3, 0,
1, 3;
const Mat true_x {20, 30};
GaussianDistribution dist {true_x, EigenTriangularMatrix(d)};
const Mat x1 {dist()};
const Mat x2 {dist()};
EXPECT_NE(x1, x2);
using EMat = EuclideanMean<Coefficients<Axis, Axis>>;
EMat mean_x = EMat::zero();
for (int i = 0; i < 100; i++)
{
Mat x {dist()};
mean_x = (mean_x * i + to_Euclidean(x)) / (i + 1);
}
EXPECT_NE(from_Euclidean(mean_x), true_x);
EXPECT_TRUE(is_near(from_Euclidean(mean_x), true_x, MatrixTraits<Mat>::BaseMatrix::Constant(0.5)));
}

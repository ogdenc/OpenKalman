/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "matrices.hpp"

using namespace OpenKalman;

using M1 = Eigen::Matrix<double, 1, 1>;
using M2 = Eigen::Matrix<double, 2, 2>;
using M2col = Eigen::Matrix<double, 2, 1>;
using M3 = Eigen::Matrix<double, 3, 3>;
using M3col = Eigen::Matrix<double, 3, 1>;
using M4 = Eigen::Matrix<double, 4, 4>;
using M4col = Eigen::Matrix<double, 4, 1>;
using C2 = Coefficients<Angle, Axis>;
using C3 = Coefficients<Angle, Axis, Axis>;
using C4 = Concatenate<C2, C2>;
using Mean1 = Mean<Angle, M1>;
using Mean2 = Mean<C2, M2col>;
using Mean3 = Mean<C3, M3col>;
using Mean4 = Mean<C4, M4col>;
using Mat2 = Matrix<C2, C2, M2>;
using Mat3 = Matrix<C3, C3, M3>;
using Mat4 = Matrix<C4, C4, M4>;
using SA1l = SelfAdjointMatrix<M1, TriangleType::lower>;
using SA1u = SelfAdjointMatrix<M1, TriangleType::upper>;
using T1l = TriangularMatrix<M1, TriangleType::lower>;
using T1u = TriangularMatrix<M1, TriangleType::upper>;
using SA2l = SelfAdjointMatrix<M2, TriangleType::lower>;
using SA2u = SelfAdjointMatrix<M2, TriangleType::upper>;
using T2l = TriangularMatrix<M2, TriangleType::lower>;
using T2u = TriangularMatrix<M2, TriangleType::upper>;
using D2 = DiagonalMatrix<M2col>;
using I2 = EigenIdentity<M2>;
using Z2 = ZeroMatrix<M2>;
using SA4l = SelfAdjointMatrix<M4, TriangleType::lower>;
using SA4u = SelfAdjointMatrix<M4, TriangleType::upper>;
using T4l = TriangularMatrix<M4, TriangleType::lower>;
using T4u = TriangularMatrix<M4, TriangleType::upper>;
using CovSA2l = Covariance<C2, SA2l>;
using CovSA2u = Covariance<C2, SA2u>;
using CovT2l = Covariance<C2, T2l>;
using CovT2u = Covariance<C2, T2u>;
using CovD2 = Covariance<C2, D2>;
using CovI2 = Covariance<C2, I2>;
using CovZ2 = Covariance<C2, Z2>;
using CovSA4l = Covariance<C4, SA4l>;
using CovSA4u = Covariance<C4, SA4u>;
using CovT4l = Covariance<C4, T4l>;
using CovT4u = Covariance<C4, T4u>;
using DistSA2l = GaussianDistribution<C2, M2col, SA2l>;
using DistSA2u = GaussianDistribution<C2, M2col, SA2u>;
using DistT2l = GaussianDistribution<C2, M2col, T2l>;
using DistT2u = GaussianDistribution<C2, M2col, T2u>;
using DistD2 = GaussianDistribution<C2, M2col, D2>;
using DistI2 = GaussianDistribution<C2, ZeroMatrix<M2col>, I2>;
using DistZ2 = GaussianDistribution<C2, ZeroMatrix<M2col>, Z2>;
using DistSA4l = GaussianDistribution<C4, M4col, SA4l>;
using DistSA4u = GaussianDistribution<C4, M4col, SA4u>;
using DistT4l = GaussianDistribution<C4, M4col, T4l>;
using DistT4u = GaussianDistribution<C4, M4col, T4u>;

inline I2 i2 = M2::Identity();
inline Z2 z2 = ZeroMatrix<M2>();
inline auto covi2 = CovI2(i2);
inline auto covz2 = CovZ2(z2);
inline auto disti2 = DistI2(ZeroMatrix<M2col>(), covi2);
inline auto distz2 = DistZ2(ZeroMatrix<M2col>(), covz2);

TEST_F(matrices, GaussianDistribution_class)
{
  // Default constructor
  DistSA2l distSA2la;
  mean_of(distSA2la) = Mean2 {1, 2};
  covariance_of(distSA2la) = SA2l {9, 3, 3, 10};
  EXPECT_TRUE(is_near(mean_of(distSA2la), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2la), Mat2 {9, 3, 3, 10}));

  // Copy constructor
  DistSA2l distSA2lb = const_cast<const DistSA2l&>(distSA2la);
  EXPECT_TRUE(is_near(mean_of(distSA2lb), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lb), Mat2 {9, 3, 3, 10}));

  // Move constructor
  auto xa = DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}};
  DistSA2l distSA2lc(std::move(xa));
  EXPECT_TRUE(is_near(mean_of(distSA2lc), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lc), Mat2 {9, 3, 3, 10}));

  // Convert from a related Gaussian distribution
  DistSA2l distSA2ld_1(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ld_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_1), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2ld_2(DistSA2u {Mean2 {1, 2}, CovSA2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ld_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_2), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2ld_3(DistT2l {Mean2 {1, 2}, CovT2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ld_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_3), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2ld_4(DistT2u {Mean2 {1, 2}, CovT2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ld_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_4), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2ld_5(DistD2 {Mean2 {1, 2}, CovD2 {9, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ld_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_5), Mat2 {9, 0, 0, 10}));
  DistSA2l distSA2ld_6(DistI2 {ZeroMatrix<M2col>(), covi2});
  EXPECT_TRUE(is_near(mean_of(distSA2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_6), Mat2 {1, 0, 0, 1}));
  DistSA2l distSA2ld_7(DistZ2 {ZeroMatrix<M2col>(), covz2});
  EXPECT_TRUE(is_near(mean_of(distSA2ld_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_7), Mat2 {0, 0, 0, 0}));
  //
  DistSA2u distSA2ud_1(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ud_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_1), Mat2 {9, 3, 3, 10}));
  DistSA2u distSA2ud_2(DistSA2u {Mean2 {1, 2}, CovSA2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ud_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_2), Mat2 {9, 3, 3, 10}));
  DistSA2u distSA2ud_3(DistT2l {Mean2 {1, 2}, CovT2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ud_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_3), Mat2 {9, 3, 3, 10}));
  DistSA2u distSA2ud_4(DistT2u {Mean2 {1, 2}, CovT2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ud_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_4), Mat2 {9, 3, 3, 10}));
  DistSA2u distSA2ud_5(DistD2 {Mean2 {1, 2}, CovD2 {9, 10}});
  EXPECT_TRUE(is_near(mean_of(distSA2ud_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_5), Mat2 {9, 0, 0, 10}));
  DistSA2u distSA2ud_6(DistI2 {ZeroMatrix<M2col>(), covi2});
  EXPECT_TRUE(is_near(mean_of(distSA2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_6), Mat2 {1, 0, 0, 1}));
  DistSA2u distSA2ud_7(DistZ2 {ZeroMatrix<M2col>(), covz2});
  EXPECT_TRUE(is_near(mean_of(distSA2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_7), Mat2 {0, 0, 0, 0}));
  //
  DistT2l distT2ld_1(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ld_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_1), Mat2 {9, 3, 3, 10}));
  DistT2l distT2ld_2(DistSA2u {Mean2 {1, 2}, CovSA2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ld_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_2), Mat2 {9, 3, 3, 10}));
  DistT2l distT2ld_3(DistT2l {Mean2 {1, 2}, CovT2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ld_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_3), Mat2 {9, 3, 3, 10}));
  DistT2l distT2ld_4(DistT2u {Mean2 {1, 2}, CovT2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ld_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_4), Mat2 {9, 3, 3, 10}));
  DistT2l distT2ld_5(DistD2 {Mean2 {1, 2}, CovD2 {9, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ld_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_5), Mat2 {9, 0, 0, 10}));
  DistT2l distT2ld_6(DistI2 {ZeroMatrix<M2col>(), covi2});
  EXPECT_TRUE(is_near(mean_of(distT2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_6), Mat2 {1, 0, 0, 1}));
  DistT2l distT2ld_7(DistZ2 {ZeroMatrix<M2col>(), covz2});
  EXPECT_TRUE(is_near(mean_of(distT2ld_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_7), Mat2 {0, 0, 0, 0}));
  //
  DistT2u distT2ud_1(DistSA2l {Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ud_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_1), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ud_2(DistSA2u {Mean2 {1, 2}, CovSA2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ud_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_2), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ud_3(DistT2l {Mean2 {1, 2}, CovT2l {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ud_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_3), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ud_4(DistT2u {Mean2 {1, 2}, CovT2u {9, 3, 3, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ud_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_4), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ud_5(DistD2 {Mean2 {1, 2}, CovD2 {9, 10}});
  EXPECT_TRUE(is_near(mean_of(distT2ud_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_5), Mat2 {9, 0, 0, 10}));
  DistT2u distT2ud_6(DistI2 {ZeroMatrix<M2col>(), covi2});
  EXPECT_TRUE(is_near(mean_of(distT2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_6), Mat2 {1, 0, 0, 1}));
  DistT2u distT2ud_7(DistZ2 {ZeroMatrix<M2col>(), covz2});
  EXPECT_TRUE(is_near(mean_of(distT2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_7), Mat2 {0, 0, 0, 0}));

  // Construct from different combinations of mean, typed matrix base, covariance, and covariance base.
  DistSA2l distSA2le_1(Mean2 {1, 2}, SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2le_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_1), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_2((M2col() << 1, 2).finished(), CovSA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2le_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_2), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_3((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2le_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_3), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_4(Mean2 {1, 2}, Mat2 {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2le_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_4), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_5(Mean2 {1, 2}, Mat2::identity() * 0.1);
  EXPECT_TRUE(is_near(mean_of(distSA2le_5), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_5), Mat2 {.1, 0, 0, .1}));

  DistT2l distT2le(Mean2 {1, 2}, T2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(mean_of(distT2le), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2le), Mat2 {9, 3, 3, 10}));
  DistT2u distT2ue(Mean2 {1, 2}, T2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(mean_of(distT2ue), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distT2ue), Mat2 {9, 3, 3, 10}));
  DistD2 distD2e(Mean2 {1, 2}, D2 {9, 10});
  EXPECT_TRUE(is_near(mean_of(distD2e), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distD2e), Mat2 {9, 0, 0, 10}));
  DistI2 distI2e(ZeroMatrix<M2col>(), i2);
  EXPECT_TRUE(is_near(mean_of(distI2e), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distI2e), Mat2 {1, 0, 0, 1}));
  DistZ2 distZ2e(ZeroMatrix<M2col>(), z2);
  EXPECT_TRUE(is_near(mean_of(distZ2e), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distZ2e), Mat2 {0, 0, 0, 0}));

  // Construct from only covariance or covariance base.
  DistSA2l distSA2lf_1(CovSA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2lf_1), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lf_1), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2lf_2(SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2lf_2), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lf_2), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2lf_3(Mat2 {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2lf_3), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lf_3), Mat2 {9, 3, 3, 10}));
  DistT2l distT2lf(T2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(mean_of(distT2lf), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2lf), Mat2 {9, 3, 3, 10}));
  DistT2u distT2uf(T2u {3, 1, 0, 3});
  EXPECT_TRUE(is_near(mean_of(distT2uf), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2uf), Mat2 {9, 3, 3, 10}));
  DistD2 distD2f(D2 {9, 10});
  EXPECT_TRUE(is_near(mean_of(distD2f), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distD2f), Mat2 {9, 0, 0, 10}));
  DistI2 distI2f(i2);
  EXPECT_TRUE(is_near(mean_of(distI2f), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distI2f), Mat2 {1, 0, 0, 1}));
  DistZ2 distZ2f(z2);
  EXPECT_TRUE(is_near(mean_of(distZ2f), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distZ2f), Mat2 {0, 0, 0, 0}));

  // Construct using lists of parameters for mean, covariance, or both.
  DistSA2l distSA2lg_1{{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2lg_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lg_1), Mat2 {4, 2, 2, 5}));
  DistSA2l distSA2lg_2{Mean2 {3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2lg_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lg_2), Mat2 {4, 2, 2, 5}));
  DistSA2l distSA2lg_3{{3, 4}, CovSA2l {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2lg_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lg_3), Mat2 {4, 2, 2, 5}));
  DistSA2l distSA2lg_4{{3, 4}, SA2l {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2lg_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lg_4), Mat2 {4, 2, 2, 5}));

  // Copy assignment
  distSA2lb = distSA2lg_1;
  EXPECT_TRUE(is_near(mean_of(distSA2lb), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lb), Mat2 {4, 2, 2, 5}));

  // Move assignment
  auto xb =  DistSA2l {{3, 4}, {4, 2, 2, 5}};
  distSA2lc = std::move(xb);
  EXPECT_TRUE(is_near(mean_of(distSA2lc), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2lc), Mat2 {4, 2, 2, 5}));

  // Assign from different distribution type
  distSA2ld_1 = DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ld_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_1), Mat2 {4, 2, 2, 5}));
  distSA2ld_2 = DistSA2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ld_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_2), Mat2 {4, 2, 2, 5}));
  distSA2ld_3 = DistT2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ld_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_3), Mat2 {4, 2, 2, 5}));
  distSA2ld_4 = DistT2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ld_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_4), Mat2 {4, 2, 2, 5}));
  distSA2ld_5 = DistD2 {{3, 4}, {4, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ld_5), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_5), Mat2 {4, 0, 0, 5}));
  distSA2ld_6 = DistI2 {{}, covi2};
  EXPECT_TRUE(is_near(mean_of(distSA2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_6), Mat2 {1, 0, 0, 1}));
  distSA2ld_7 = DistZ2 {{}, covz2};
  EXPECT_TRUE(is_near(mean_of(distSA2ld_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_7), Mat2 {0, 0, 0, 0}));
  //
  distSA2ud_1 = DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ud_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_1), Mat2 {4, 2, 2, 5}));
  distSA2ud_2 = DistSA2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ud_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_2), Mat2 {4, 2, 2, 5}));
  distSA2ud_3 = DistT2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ud_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_3), Mat2 {4, 2, 2, 5}));
  distSA2ud_4 = DistT2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ud_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_4), Mat2 {4, 2, 2, 5}));
  distSA2ud_5 = DistD2 {{3, 4}, {4, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2ud_5), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_5), Mat2 {4, 0, 0, 5}));
  distSA2ud_6 = DistI2 {{}, covi2};
  EXPECT_TRUE(is_near(mean_of(distSA2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_6), Mat2 {1, 0, 0, 1}));
  distSA2ud_7 = DistZ2 {{}, covz2};
  EXPECT_TRUE(is_near(mean_of(distSA2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_7), Mat2 {0, 0, 0, 0}));
  //
  distT2ld_1 = DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ld_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_1), Mat2 {4, 2, 2, 5}));
  distT2ld_2 = DistSA2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ld_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_2), Mat2 {4, 2, 2, 5}));
  distT2ld_3 = DistT2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ld_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_3), Mat2 {4, 2, 2, 5}));
  distT2ld_4 = DistT2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ld_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_4), Mat2 {4, 2, 2, 5}));
  distT2ld_5 = DistD2 {{3, 4}, {4, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ld_5), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_5), Mat2 {4, 0, 0, 5}));
  distT2ld_6 = DistI2 {{}, covi2};
  EXPECT_TRUE(is_near(mean_of(distT2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_6), Mat2 {1, 0, 0, 1}));
  distT2ld_7 = DistZ2 {{}, covz2};
  EXPECT_TRUE(is_near(mean_of(distT2ld_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_7), Mat2 {0, 0, 0, 0}));
  //
  distT2ud_1 = DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ud_1), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_1), Mat2 {4, 2, 2, 5}));
  distT2ud_2 = DistSA2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ud_2), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_2), Mat2 {4, 2, 2, 5}));
  distT2ud_3 = DistT2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ud_3), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_3), Mat2 {4, 2, 2, 5}));
  distT2ud_4 = DistT2u {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ud_4), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_4), Mat2 {4, 2, 2, 5}));
  distT2ud_5 = DistD2 {{3, 4}, {4, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ud_5), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_5), Mat2 {4, 0, 0, 5}));
  distT2ud_6 = DistI2 {{}, covi2};
  EXPECT_TRUE(is_near(mean_of(distT2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_6), Mat2 {1, 0, 0, 1}));
  distT2ud_7 = DistZ2 {{}, covz2};
  EXPECT_TRUE(is_near(mean_of(distT2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_7), Mat2 {0, 0, 0, 0}));

  // Assign from a list of coefficients (via move assignment operator)
  distSA2la = {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distSA2la), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distSA2la), Mat2 {4, 2, 2, 5}));

  // Increment
  distT2le += {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2le), Mean2 {4, 6}));
  EXPECT_TRUE(is_near(covariance_of(distT2le), Mat2 {13, 5, 5, 15}));
  distT2ue += DistSA2l {{3, 4}, {4, 2, 2, 5}};
  EXPECT_TRUE(is_near(mean_of(distT2ue), Mean2 {4, 6}));
  EXPECT_TRUE(is_near(covariance_of(distT2ue), Mat2 {13, 5, 5, 15}));

  // Decrement
  distT2le -= {{1, 2}, {9, 3, 3, 10}};
  EXPECT_TRUE(is_near(mean_of(distT2le), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2le), Mat2 {4, 2, 2, 5}));
  distT2ue -= DistSA2l {{1, 2}, {9, 3, 3, 10}};
  EXPECT_TRUE(is_near(mean_of(distT2ue), Mean2 {3, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2ue), Mat2 {4, 2, 2, 5}));

  // Scalar multiplication
  distT2le *= 2;
  EXPECT_TRUE(is_near(mean_of(distT2le), Mean2 {6-2*M_PI, 8}));
  EXPECT_TRUE(is_near(covariance_of(distT2le), Mat2 {16, 8, 8, 20}));

  // Scalar division
  distT2le /= 2;
  EXPECT_TRUE(is_near(mean_of(distT2le), Mean2 {3-M_PI, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2le), Mat2 {4, 2, 2, 5}));

  // Zero
  EXPECT_TRUE(is_near(DistSA2l::zero(), distz2));
  EXPECT_TRUE(is_near(DistSA2u::zero(), distz2));
  EXPECT_TRUE(is_near(DistT2l::zero(), distz2));
  EXPECT_TRUE(is_near(DistT2u::zero(), distz2));
  EXPECT_TRUE(is_near(DistD2::zero(), distz2));

  // Normal
  EXPECT_TRUE(is_near(DistSA2l::normal(), disti2));
  EXPECT_TRUE(is_near(DistSA2u::normal(), disti2));
  EXPECT_TRUE(is_near(DistT2l::normal(), disti2));
  EXPECT_TRUE(is_near(DistT2u::normal(), disti2));
  EXPECT_TRUE(is_near(DistD2::normal(), disti2));
}


TEST_F(matrices, GaussianDistribution_class_random)
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
  EXPECT_TRUE(is_near(Mean(from_Euclidean(mean_x) - true_x), V::zero(), MatrixTraits<V>::BaseMatrix::Constant(0.1)));
}


TEST_F(matrices, GaussianDistribution_class_random_axis)
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
  EXPECT_TRUE(is_near(from_Euclidean(mean_x), true_x, MatrixTraits<Mat>::BaseMatrix::Constant(1.0)));
}


TEST_F(matrices, GaussianDistribution_class_Cholesky_random)
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
  EXPECT_TRUE(is_near(Mean(from_Euclidean(mean_x) - true_x), V::zero(), MatrixTraits<V>::BaseMatrix::Constant(0.1)));
}


TEST_F(matrices, GaussianDistribution_class_Cholesky_random_axis)
{
  using Mat = Mean<Coefficients<Axis, Axis>>;
  M2 d;
  d << 3, 0,
  1, 3;
  const Mat true_x {20, 30};
  GaussianDistribution dist {true_x, TriangularMatrix(d)};
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
  EXPECT_TRUE(is_near(from_Euclidean(mean_x), true_x, MatrixTraits<Mat>::BaseMatrix::Constant(1.0)));
}


TEST_F(matrices, GaussianDistribution_class_statistics)
{
  GaussianDistribution<Axis, M1, SA1l> x1 = {M1(2), SA1l(M1(9))};
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(1)}), -2.07310637743, 1e-6);
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(0)}), -2.2397730441, 1e-6);
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(1)}, Mean{M1(0)}), -2.07310637743 - 2.2397730441, 1e-6);

  GaussianDistribution<Axes<2>, M2col, T2l> x2 = {{0, 0}, {1, 0, 0.3, 1}};
  EXPECT_NEAR(x2.log_likelihood(Mean {1.5, 0.9}), -3.02698546294, 1e-6);

  GaussianDistribution<Axes<2>, M2col, T2l> x3 = {{0, 0}, {1, 0, 0.8, 1}};
  EXPECT_NEAR(x3.log_likelihood(Mean {2, 0.7}), -4.45205144264, 1e-6);
  EXPECT_NEAR(x3.log_likelihood(Mean {1.5, 0.9}), -2.5770514426433544, 1e-6);
  EXPECT_NEAR(x3.log_likelihood(Mean {2, 0.7}, Mean {1.5, 0.9}), -4.45205144264 - 2.5770514426433544, 1e-6);

  GaussianDistribution<Axis, M1, T1l> x4 = {M1(0), M1(1)};
  EXPECT_NEAR(x4.entropy(), 2.04709558518, 1e-6);

  GaussianDistribution<Axis, M1, M1> x5 = {M1(0), M1(0.25)};
  EXPECT_NEAR(x5.entropy(), 1.04709558518, 1e-6);
}


TEST_F(matrices, GaussianDistribution_deduction_guides)
{
  EXPECT_TRUE(is_near(GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(Mean2 {1, 2}, Mat2 {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(GaussianDistribution(Mean2 {1, 2}, Mat2 {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(GaussianDistribution(Mean2 {1, 2}, Mat2 {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution((M2col() << 1, 2).finished(), CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(GaussianDistribution((M2col() << 1, 2).finished(), CovSA2l {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(GaussianDistribution((M2col() << 1, 2).finished(), CovSA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution((M2col() << 1, 2).finished(), Mat2 {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(GaussianDistribution((M2col() << 1, 2).finished(), Mat2 {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(GaussianDistribution((M2col() << 1, 2).finished(), Mat2 {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(GaussianDistribution((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}))>::Coefficients, Axes<2>>);
  static_assert(is_self_adjoint_v<decltype(GaussianDistribution((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}))>);
}


TEST_F(matrices, GaussianDistribution_make)
{
  EXPECT_TRUE(is_near(make_GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(Mean2 {1, 2}, (M2() << 9, 3, 3, 10).finished()), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution(Mean2 {1, 2}, (M2() << 9, 3, 3, 10).finished()))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution(Mean2 {1, 2}, (M2() << 9, 3, 3, 10).finished()))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution((M2col() << 1, 2).finished(), CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution((M2col() << 1, 2).finished(), CovSA2l {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution((M2col() << 1, 2).finished(), CovSA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution((M2col() << 1, 2).finished(), Mat2 {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution((M2col() << 1, 2).finished(), Mat2 {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution((M2col() << 1, 2).finished(), Mat2 {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution<C2>((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution<C2>((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution<C2>((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}))>::Coefficients, Axes<2>>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution((M2col() << 1, 2).finished(), SA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution<C2>((M2col() << 1, 2).finished(), (M2() << 9, 3, 3, 10).finished()), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution<C2>((M2col() << 1, 2).finished(), (M2() << 9, 3, 3, 10).finished()))>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution<C2>((M2col() << 1, 2).finished(), (M2() << 9, 3, 3, 10).finished()))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution((M2col() << 1, 2).finished(), (M2() << 9, 3, 3, 10).finished()), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution((M2col() << 1, 2).finished(), (M2() << 9, 3, 3, 10).finished()))>::Coefficients, Axes<2>>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution((M2col() << 1, 2).finished(), (M2() << 9, 3, 3, 10).finished()))>);

  // Defaults
  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution<Mean2, CovSA2l>())>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution<Mean2, CovSA2l>())>);

  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution<Mean2, SA2l>())>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution<Mean2, SA2l>())>);

  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution<M2col, CovSA2l>())>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution<M2col, CovSA2l>())>);

  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution<C2, M2col, SA2l>())>::Coefficients, C2>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution<C2, M2col, SA2l>())>);

  static_assert(is_equivalent_v<typename DistributionTraits<decltype(make_GaussianDistribution<M2col, SA2l>())>::Coefficients, Axes<2>>);
  static_assert(is_self_adjoint_v<decltype(make_GaussianDistribution<M2col, SA2l>())>);
}


TEST_F(matrices, GaussianDistribution_traits)
{
  static_assert(not is_square_root_v<DistSA2l>);
  static_assert(not is_diagonal_v<DistSA2l>);
  static_assert(is_self_adjoint_v<DistSA2l>);
  static_assert(not is_Cholesky_v<DistSA2l>);
  static_assert(not is_triangular_v<DistSA2l>);
  static_assert(not is_lower_triangular_v<DistSA2l>);
  static_assert(not is_upper_triangular_v<DistSA2l>);
  static_assert(not is_zero_v<DistSA2l>);

  static_assert(not is_square_root_v<DistT2l>);
  static_assert(not is_diagonal_v<DistT2l>);
  static_assert(is_self_adjoint_v<DistT2l>);
  static_assert(is_Cholesky_v<DistT2l>);
  static_assert(not is_triangular_v<DistT2l>);
  static_assert(not is_lower_triangular_v<DistT2l>);
  static_assert(not is_upper_triangular_v<DistT2l>);
  static_assert(not is_upper_triangular_v<DistT2l>);
  static_assert(not is_zero_v<DistT2l>);

  static_assert(not is_square_root_v<DistD2>);
  static_assert(is_diagonal_v<DistD2>);
  static_assert(is_self_adjoint_v<DistD2>);
  static_assert(not is_Cholesky_v<DistD2>);
  static_assert(is_triangular_v<DistD2>);
  static_assert(is_lower_triangular_v<DistD2>);
  static_assert(is_upper_triangular_v<DistD2>);
  static_assert(is_upper_triangular_v<DistD2>);
  static_assert(not is_zero_v<DistD2>);

  static_assert(not is_square_root_v<DistI2>);
  static_assert(is_diagonal_v<DistI2>);
  static_assert(is_self_adjoint_v<DistI2>);
  static_assert(not is_Cholesky_v<DistI2>);
  static_assert(is_triangular_v<DistI2>);
  static_assert(is_lower_triangular_v<DistI2>);
  static_assert(is_upper_triangular_v<DistI2>);
  static_assert(is_upper_triangular_v<DistI2>);
  static_assert(not is_zero_v<DistI2>);

  static_assert(not is_square_root_v<DistZ2>);
  static_assert(is_diagonal_v<DistZ2>);
  static_assert(is_self_adjoint_v<DistZ2>);
  static_assert(not is_Cholesky_v<DistZ2>);
  static_assert(is_triangular_v<DistZ2>);
  static_assert(is_lower_triangular_v<DistZ2>);
  static_assert(is_upper_triangular_v<DistZ2>);
  static_assert(is_upper_triangular_v<DistZ2>);
  static_assert(is_zero_v<DistZ2>);

  // DistributionTraits
  EXPECT_TRUE(is_near(DistributionTraits<DistSA2l>::make(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  EXPECT_TRUE(is_near(DistributionTraits<DistSA2l>::make(Mean2 {1, 2}, SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  EXPECT_TRUE(is_near(DistributionTraits<DistSA2l>::zero(), distz2));
  EXPECT_TRUE(is_near(DistributionTraits<DistSA2l>::normal(), disti2));
}


TEST_F(matrices, GaussianDistribution_overloads)
{
  // mean
  EXPECT_TRUE(is_near(mean_of(DistSA2l {{1, 2}, {9, 3, 3, 10}}), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(mean_of(DistSA2u {{1, 2}, {9, 3, 3, 10}}), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(mean_of(DistT2l {{1, 2}, {9, 3, 3, 10}}), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(mean_of(DistT2u {{1, 2}, {9, 3, 3, 10}}), Mean2 {1, 2}));

  // covariance
  EXPECT_TRUE(is_near(covariance_of(DistSA2l {{1, 2}, {9, 3, 3, 10}}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covariance_of(DistSA2u {{1, 2}, {9, 3, 3, 10}}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(base_matrix(covariance_of(DistT2l {{1, 2}, {9, 3, 3, 10}})), Mat2 { 3, 0, 1, 3}));
  EXPECT_TRUE(is_near(base_matrix(covariance_of(DistT2u {{1, 2}, {9, 3, 3, 10}})), Mat2 { 3, 1, 0, 3}));

  EXPECT_TRUE(is_near(base_matrix(covariance_of(to_Cholesky(DistSA2l {{1, 2}, {9, 3, 3, 10}}))), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(base_matrix(covariance_of(to_Cholesky(DistSA2u {{1, 2}, {9, 3, 3, 10}}))), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(base_matrix(covariance_of(from_Cholesky(DistT2l {{1, 2}, {9, 3, 3, 10}}))), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(base_matrix(covariance_of(from_Cholesky(DistT2u {{1, 2}, {9, 3, 3, 10}}))), Mat2 {9, 3, 3, 10}));

  static_assert(std::is_same_v<std::decay_t<decltype(strict(DistSA2l {Mean2 {1, 2} * 2, CovSA2l{9, 3, 3, 10} * 2}))>, DistSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(DistSA2u {Mean2 {1, 2} * 2, CovSA2u{9, 3, 3, 10} * 2}))>, DistSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(DistT2l {Mean2 {1, 2} * 2, CovT2l{9, 3, 3, 10} * 2}))>, DistT2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(DistT2u {Mean2 {1, 2} * 2, CovT2u{9, 3, 3, 10} * 2}))>, DistT2u>);
}


TEST_F(matrices, GaussianDistribution_blocks)
{
  Mean2 a1 {1, 2}, a2 {3, 4};
  Mean4 b {1, 2, 3, 4};
  Mat2 m1 {9, 3, 3, 10}, m2 {4, 2, 2, 5};
  Mat4 n {9, 3, 0, 0,
          3, 10, 0, 0,
          0, 0, 4, 2,
          0, 0, 2, 5};
  EXPECT_TRUE(is_near(concatenate(DistSA2l(a1, m1), DistSA2l(a2, m2)), DistSA4l(b, n)));
  EXPECT_TRUE(is_near(concatenate(DistSA2u(a1, m1), DistSA2u(a2, m2)), DistSA4u(b, n)));
  EXPECT_TRUE(is_near(concatenate(DistT2l(a1, m1), DistT2l(a2, m2)), DistT4l(b, n)));
  EXPECT_TRUE(is_near(concatenate(DistT2u(a1, m1), DistT2u(a2, m2)), DistT4u(b, n)));

  EXPECT_TRUE(is_near(split(DistSA4l(b, n)), std::tuple {}));
  EXPECT_TRUE(is_near(split<C2, C2>(DistSA4l(b, n)), std::tuple {DistSA2l(a1, m1), DistSA2l(a2, m2)}));
  EXPECT_TRUE(is_near(split<C2, C2>(DistSA4u(b, n)), std::tuple {DistSA2u(a1, m1), DistSA2u(a2, m2)}));
  EXPECT_TRUE(is_near(split<C2, C2>(DistT4l(b, n)), std::tuple {DistT2l(a1, m1), DistT2l(a2, m2)}));
  EXPECT_TRUE(is_near(split<C2, C2>(DistT4u(b, n)), std::tuple {DistT2u(a1, m1), DistT2u(a2, m2)}));
  EXPECT_TRUE(is_near(split<C2, C2::Take<1>>(DistSA4l(b, n)), std::tuple {DistSA2l(a1, m1), GaussianDistribution {Mean1 {3}, SA1l {4}}}));
  EXPECT_TRUE(is_near(split<C2, C2::Take<1>>(DistSA4u(b, n)), std::tuple {DistSA2u(a1, m1), GaussianDistribution {Mean1 {3}, SA1u {4}}}));
  EXPECT_TRUE(is_near(split<C2, C2::Take<1>>(DistT4l(b, n)), std::tuple {DistT2l(a1, m1), GaussianDistribution {Mean1 {3}, SA1l {4}}}));
  EXPECT_TRUE(is_near(split<C2, C2::Take<1>>(DistT4u(b, n)), std::tuple {DistT2u(a1, m1), GaussianDistribution {Mean1 {3}, SA1u {4}}}));
}


TEST_F(matrices, GaussianDistribution_addition_subtraction)
{
  Mean<Axes<2>> x_mean {20, 30};
  M2 d;
  d << 9, 3,
  3, 8;
  GaussianDistribution dist1 {x_mean, SelfAdjointMatrix(d)};
  Mean<Axes<2>> y_mean {11, 23};
  M2 e;
  e << 7, 1,
  1, 3;
  GaussianDistribution dist2 {y_mean, SelfAdjointMatrix(e)};
  auto sum1 = dist1 + dist2;
  EXPECT_TRUE(is_near(mean_of(sum1), Mean {31., 53}));
  EXPECT_TRUE(is_near(covariance_of(sum1), Covariance {16., 4, 4, 11}));
  auto diff1 = dist1 - dist2;
  EXPECT_TRUE(is_near(mean_of(diff1), Mean {9., 7.}));
  EXPECT_TRUE(is_near(covariance_of(diff1), Covariance {2., 2, 2, 5}));
  GaussianDistribution dist1_chol {x_mean, TriangularMatrix(SelfAdjointMatrix(d))};
  GaussianDistribution dist2_chol {y_mean, TriangularMatrix(SelfAdjointMatrix(e))};
  auto sum2 = dist1_chol + dist2_chol;
  EXPECT_TRUE(is_near(mean_of(sum2), Mean {31., 53}));
  EXPECT_TRUE(is_near(covariance_of(sum2), Covariance {16., 4, 4, 11}));
  auto diff2 = dist1_chol - dist2_chol;
  EXPECT_TRUE(is_near(mean_of(diff2), Mean {9., 7.}));
  EXPECT_TRUE(is_near(covariance_of(diff2), Covariance {2., 2, 2, 5}));
  auto sum3 = dist1 + dist2_chol;
  EXPECT_TRUE(is_near(mean_of(sum3), Mean {31., 53}));
  EXPECT_TRUE(is_near(covariance_of(sum3), Covariance {16., 4, 4, 11}));
  auto diff3 = dist1 - dist2_chol;
  EXPECT_TRUE(is_near(mean_of(diff3), Mean {9., 7.}));
  EXPECT_TRUE(is_near(covariance_of(diff3), Covariance {2., 2, 2, 5}));
  auto sum4 = dist1_chol + dist2;
  EXPECT_TRUE(is_near(mean_of(sum4), Mean {31., 53}));
  EXPECT_TRUE(is_near(covariance_of(sum4), Covariance {16., 4, 4, 11}));
  auto diff4 = dist1_chol - dist2;
  EXPECT_TRUE(is_near(mean_of(diff4), Mean {9., 7.}));
  EXPECT_TRUE(is_near(covariance_of(diff4), Covariance {2., 2, 2, 5}));
}


TEST_F(matrices, GaussianDistribution_mult_div)
{
  auto a = GaussianDistribution(make_Mean<C2>(2., 30), make_Covariance<C2>(8., 2, 2, 6));
  auto a_chol = GaussianDistribution(make_Mean<C2>(2., 30), make_Covariance<C2, TriangleType::lower>(8., 2, 2, 6));
  auto f_matrix = make_Matrix<C3, C2>(1., 2, 3, 4, 5, 6);
  auto a_scaled3 = f_matrix * a;
  EXPECT_TRUE(is_near(mean_of(a_scaled3), make_Mean<C3>(62., 126, 190)));
  EXPECT_TRUE(is_near(covariance_of(a_scaled3), make_Matrix<C3, C3>(40., 92, 144, 92, 216, 340, 144, 340, 536)));
  static_assert(is_equivalent_v<typename decltype(a_scaled3)::Coefficients, C3>);
  auto a_chol_scaled3 = f_matrix * a_chol;
  EXPECT_TRUE(is_near(mean_of(a_chol_scaled3), make_Mean<C3>(62., 126, 190)));
  EXPECT_TRUE(is_near(covariance_of(a_chol_scaled3), make_Matrix<C3, C3>(40., 92, 144, 92, 216, 340, 144, 340, 536)));
  static_assert(is_equivalent_v<typename decltype(a_chol_scaled3)::Coefficients, C3>);

  Eigen::Matrix<double, 2, 2> cov_mat; cov_mat << 8, 2, 2, 6;
  decltype(a) a_scaled = a * 2;
  EXPECT_TRUE(is_near(mean_of(a_scaled), mean_of(a) * 2));
  EXPECT_TRUE(is_near(covariance_of(a_scaled), covariance_of(a) * 4));
  decltype(a_chol) a_chol_scaled = a_chol * 2;
  EXPECT_TRUE(is_near(mean_of(a_chol_scaled), mean_of(a) * 2));
  EXPECT_TRUE(is_near(covariance_of(a_chol_scaled), covariance_of(a) * 4));
  a_scaled = 2 * a;
  EXPECT_TRUE(is_near(mean_of(a_scaled), 2 * mean_of(a)));
  EXPECT_TRUE(is_near(covariance_of(a_scaled), 4 * covariance_of(a)));
  a_chol_scaled = 2 * a_chol;
  EXPECT_TRUE(is_near(mean_of(a_chol_scaled), 2 * mean_of(a)));
  EXPECT_TRUE(is_near(covariance_of(a_chol_scaled), 4 * covariance_of(a)));
  a_scaled = a / 2;
  EXPECT_TRUE(is_near(mean_of(a_scaled), mean_of(a) / 2));
  EXPECT_TRUE(is_near(covariance_of(a_scaled), covariance_of(a) / 4));
  a_chol_scaled = a_chol / 2;
  EXPECT_TRUE(is_near(mean_of(a_chol_scaled), mean_of(a) / 2));
  EXPECT_TRUE(is_near(covariance_of(a_chol_scaled), covariance_of(a) / 4));
}


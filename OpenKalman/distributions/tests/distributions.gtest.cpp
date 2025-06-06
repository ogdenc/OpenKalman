/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "distributions.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;
using namespace OpenKalman::test;

using numbers::pi;

using M1 = eigen_matrix_t<double, 1, 1>;
using M2 = eigen_matrix_t<double, 2, 2>;
using M2col = eigen_matrix_t<double, 2, 1>;
using M3 = eigen_matrix_t<double, 3, 3>;
using M3col = eigen_matrix_t<double, 3, 1>;
using M4 = eigen_matrix_t<double, 4, 4>;
using M4col = eigen_matrix_t<double, 4, 1>;
using C2 = std::tuple<angle::Radians, Axis>;
using C3 = std::tuple<angle::Radians, Axis, Axis>;
using C4 = Concatenate<C2, C2>;
using Mean1 = Mean<angle::Radians, M1>;
using Mean2 = Mean<C2, M2col>;
using Mean3 = Mean<C3, M3col>;
using Mean4 = Mean<C4, M4col>;
using Mat2 = Matrix<C2, C2, M2>;
using Mat3 = Matrix<C3, C3, M3>;
using Mat4 = Matrix<C4, C4, M4>;
using SA1l = HermitianAdapter<M1, TriangleType::lower>;
using SA1u = HermitianAdapter<M1, TriangleType::upper>;
using T1l = TriangularAdapter<M1, TriangleType::lower>;
using T1u = TriangularAdapter<M1, TriangleType::upper>;
using SA2l = HermitianAdapter<M2, TriangleType::lower>;
using SA2u = HermitianAdapter<M2, TriangleType::upper>;
using T2l = TriangularAdapter<M2, TriangleType::lower>;
using T2u = TriangularAdapter<M2, TriangleType::upper>;
using D2 = DiagonalAdapter<M2col>;
using I2 = Eigen3::IdentityMatrix<M2>;
using Z2 = ZeroAdapter<eigen_matrix_t<double, 2, 2>>;
using SA4l = HermitianAdapter<M4, TriangleType::lower>;
using SA4u = HermitianAdapter<M4, TriangleType::upper>;
using T4l = TriangularAdapter<M4, TriangleType::lower>;
using T4u = TriangularAdapter<M4, TriangleType::upper>;
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
using DistI2 = GaussianDistribution<C2, ZeroAdapter<eigen_matrix_t<double, 2, 1>>, I2>;
using DistZ2 = GaussianDistribution<C2, ZeroAdapter<eigen_matrix_t<double, 2, 1>>, Z2>;
using DistSA4l = GaussianDistribution<C4, M4col, SA4l>;
using DistSA4u = GaussianDistribution<C4, M4col, SA4u>;
using DistT4l = GaussianDistribution<C4, M4col, T4l>;
using DistT4u = GaussianDistribution<C4, M4col, T4u>;

inline I2 i2 = M2::Identity();
inline Z2 z2 = ZeroAdapter<eigen_matrix_t<double, 2, 2>>();
inline auto covi2 = CovI2 {i2};
inline auto covz2 = CovZ2 {z2};
inline auto disti2 = DistI2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covi2};
inline auto distz2 = DistZ2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covz2};

TEST(matrices, GaussianDistribution_class)
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
  DistSA2l distSA2ld_6(DistI2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covi2});
  EXPECT_TRUE(is_near(mean_of(distSA2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ld_6), Mat2 {1, 0, 0, 1}));
  DistSA2l distSA2ld_7(DistZ2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covz2});
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
  DistSA2u distSA2ud_6(DistI2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covi2});
  EXPECT_TRUE(is_near(mean_of(distSA2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distSA2ud_6), Mat2 {1, 0, 0, 1}));
  DistSA2u distSA2ud_7(DistZ2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covz2});
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
  DistT2l distT2ld_6(DistI2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covi2});
  EXPECT_TRUE(is_near(mean_of(distT2ld_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ld_6), Mat2 {1, 0, 0, 1}));
  DistT2l distT2ld_7(DistZ2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covz2});
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
  DistT2u distT2ud_6(DistI2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covi2});
  EXPECT_TRUE(is_near(mean_of(distT2ud_6), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_6), Mat2 {1, 0, 0, 1}));
  DistT2u distT2ud_7(DistZ2 {ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), covz2});
  EXPECT_TRUE(is_near(mean_of(distT2ud_7), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distT2ud_7), Mat2 {0, 0, 0, 0}));

  // Construct from different combinations of mean, typed_matrix_nestable, covariance, and covariance_nestable.
  DistSA2l distSA2le_1(Mean2 {1, 2}, SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2le_1), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_1), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_2(make_dense_object_from<M2col>(1, 2), CovSA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2le_2), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_2), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_3(make_dense_object_from<M2col>(1, 2), SA2l {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2le_3), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_3), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_4(Mean2 {1, 2}, Mat2 {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mean_of(distSA2le_4), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(covariance_of(distSA2le_4), Mat2 {9, 3, 3, 10}));
  DistSA2l distSA2le_5(Mean2 {1, 2}, make_identity_matrix_like<Mat2>() * 0.1);
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
  DistI2 distI2e(ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), i2);
  EXPECT_TRUE(is_near(mean_of(distI2e), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distI2e), Mat2 {1, 0, 0, 1}));
  DistZ2 distZ2e(ZeroAdapter<eigen_matrix_t<double, 2, 1>>(), z2);
  EXPECT_TRUE(is_near(mean_of(distZ2e), Mean2 {0, 0}));
  EXPECT_TRUE(is_near(covariance_of(distZ2e), Mat2 {0, 0, 0, 0}));

  // Construct from only covariance or covariance_nestable.
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
  EXPECT_TRUE(is_near(mean_of(distT2le), Mean2 {6-2*pi, 8}));
  EXPECT_TRUE(is_near(covariance_of(distT2le), Mat2 {16, 8, 8, 20}));

  // Scalar division
  distT2le /= 2;
  EXPECT_TRUE(is_near(mean_of(distT2le), Mean2 {3-pi, 4}));
  EXPECT_TRUE(is_near(covariance_of(distT2le), Mat2 {4, 2, 2, 5}));

  // Zero
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistSA2l>(), distz2));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistSA2u>(), distz2));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistT2l>(), distz2));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistT2u>(), distz2));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistD2>(), distz2));

  // Normal
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistSA2l>(), disti2));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistSA2u>(), disti2));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistT2l>(), disti2));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistT2u>(), disti2));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistD2>(), disti2));
}


TEST(matrices, GaussianDistribution_class_random)
{
  using V = Mean<C3>;
  M3 d;
  d << 0.9, 0.1, 0.3,
       0.1, 1.4, 0.45,
       0.3, 0.45, 1.1;
  const V true_x {pi * 99/100, 10, 5};
  GaussianDistribution dist {true_x, make_covariance<C3>(d)};
  const V x1 {dist()};
  const V x2 {dist()};
  EXPECT_NE(x1, x2);
  using EV = EuclideanMean<C3>;
  EV mean_x = make_zero<EV>();
  for (int i = 0; i < 1000; i++)
  {
    V x {dist()};
    mean_x = (mean_x * i + to_euclidean(x)) / (i + 1);
  }
  EXPECT_NE(from_euclidean(mean_x), true_x);
  EXPECT_TRUE(is_near(Mean(from_euclidean(mean_x) - true_x), make_zero<V>(), nested_object_of_t<V>::Constant(0.2)));
}


TEST(matrices, GaussianDistribution_class_random_axis)
{
  using Mat = Mean<Dimensions<2>>;
  const Mat true_x {20, 30};
  GaussianDistribution dist {true_x, Covariance(9., 3, 3, 10)};
  const Mat x1 {dist()};
  const Mat x2 {dist()};
  EXPECT_NE(x1, x2);
  using EMat = EuclideanMean<Dimensions<2>>;
  EMat mean_x = make_zero<EMat>();
  for (int i = 0; i < 100; i++)
  {
    Mat x {dist()};
    mean_x = (mean_x * i + to_euclidean(x)) / (i + 1);
  }
  EXPECT_NE(from_euclidean(mean_x), true_x);
  EXPECT_TRUE(is_near(from_euclidean(mean_x), true_x, nested_object_of_t<Mat>::Constant(1.0)));
}


TEST(matrices, GaussianDistribution_class_Cholesky_random)
{
  using V = Mean<C3>;
  M3 d;
  d << 0.9, 0.1, 0.3,
       0.1, 1.4, 0.45,
       0.3, 0.45, 1.1;
  const V true_x {pi * 99/100, 10, 5};
  GaussianDistribution dist {true_x, make_covariance<C3, TriangleType::lower>(d)};
  const V x1 {dist()};
  const V x2 {dist()};
  EXPECT_NE(x1, x2);
  using EV = EuclideanMean<C3>;
  EV mean_x = make_zero<EV>();
  for (int i = 0; i < 1000; i++)
  {
    V x {dist()};
    mean_x = (mean_x * i + to_euclidean(x)) / (i + 1);
  }
  EXPECT_NE(from_euclidean(mean_x), true_x);
  EXPECT_TRUE(is_near(Mean(from_euclidean(mean_x) - true_x), make_zero<V>(), nested_object_of_t<V>::Constant(0.2)));
}


TEST(matrices, GaussianDistribution_class_Cholesky_random_axis)
{
  using Mat = Mean<Dimensions<2>>;
  M2 m22;
  m22 << 3, 0, 1, 3;
  const Mat true_x {20, 30};
  GaussianDistribution dist {true_x, TriangularAdapter {m22}};
  const Mat x1 {dist()};
  const Mat x2 {dist()};
  EXPECT_NE(x1, x2);
  using EMat = EuclideanMean<Dimensions<2>>;
  EMat mean_x = make_zero<EMat>();
  for (int i = 0; i < 100; i++)
  {
    Mat x {dist()};
    mean_x = (mean_x * i + to_euclidean(x)) / (i + 1);
  }
  EXPECT_NE(from_euclidean(mean_x), true_x);
  EXPECT_TRUE(is_near(from_euclidean(mean_x), true_x, nested_object_of_t<Mat>::Constant(1.0)));
}


TEST(matrices, GaussianDistribution_class_statistics)
{
  GaussianDistribution<Axis, M1, SA1l> x1 = {M1(2), SA1l(M1(9))};
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(1)}), -2.07310637743, 1e-6);
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(0)}), -2.2397730441, 1e-6);
  EXPECT_NEAR(x1.log_likelihood(Mean{M1(1)}, Mean{M1(0)}), -2.07310637743 - 2.2397730441, 1e-6);

  GaussianDistribution<Dimensions<2>, M2col, T2l> x2 = {{0, 0}, {1, 0, 0.3, 1}};
  EXPECT_NEAR(x2.log_likelihood(Mean {1.5, 0.9}), -3.02698546294, 1e-6);

  GaussianDistribution<Dimensions<2>, M2col, T2l> x3 = {{0, 0}, {1, 0, 0.8, 1}};
  EXPECT_NEAR(x3.log_likelihood(Mean {2, 0.7}), -4.45205144264, 1e-6);
  EXPECT_NEAR(x3.log_likelihood(Mean {1.5, 0.9}), -2.5770514426433544, 1e-6);
  EXPECT_NEAR(x3.log_likelihood(Mean {2, 0.7}, Mean {1.5, 0.9}), -4.45205144264 - 2.5770514426433544, 1e-6);

  GaussianDistribution<Axis, M1, T1l> x4 = {M1(0), M1(1)};
  EXPECT_NEAR(x4.entropy(), 2.04709558518, 1e-6);

  GaussianDistribution<Axis, M1, M1> x5 = {M1(0), M1(0.25)};
  EXPECT_NEAR(x5.entropy(), 1.04709558518, 1e-6);
}


TEST(matrices, GaussianDistribution_deduction_guides)
{
  EXPECT_TRUE(is_near(GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(Mean2 {1, 2}, Mat2 {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(GaussianDistribution(Mean2 {1, 2}, Mat2 {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(GaussianDistribution(Mean2 {1, 2}, Mat2 {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(make_eigen_matrix(1., 2), CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(GaussianDistribution(make_eigen_matrix(1., 2), CovSA2l {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(GaussianDistribution(make_dense_object_from<M2col>(1, 2), CovSA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(make_eigen_matrix(1., 2), Mat2 {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(GaussianDistribution(make_eigen_matrix(1., 2), Mat2 {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(GaussianDistribution(make_dense_object_from<M2col>(1, 2), Mat2 {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(GaussianDistribution(make_eigen_matrix(1., 2), SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(GaussianDistribution(make_eigen_matrix(1., 2), SA2l {9, 3, 3, 10}))>::StaticDescriptor, Dimensions<2>>);
  static_assert(hermitian_matrix<decltype(GaussianDistribution(make_dense_object_from<M2col>(1, 2), SA2l {9, 3, 3, 10}))>);
}


TEST(matrices, GaussianDistribution_make)
{
  EXPECT_TRUE(is_near(make_GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution(DistSA2l {{1, 2}, {9, 3, 3, 10}}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution(Mean2 {1, 2}, CovSA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution(Mean2 {1, 2}, SA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(Mean2 {1, 2}, make_dense_object_from<M2>(9, 3, 3, 10)), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution(Mean2 {1, 2}, make_dense_object_from<M2>(9, 3, 3, 10)))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution(Mean2 {1, 2}, make_dense_object_from<M2>(9, 3, 3, 10)))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(make_eigen_matrix(1., 2), CovSA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution(make_eigen_matrix(1., 2), CovSA2l {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution(make_eigen_matrix(1., 2), CovSA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(make_eigen_matrix(1., 2), Mat2 {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution(make_eigen_matrix(1., 2), Mat2 {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution(make_eigen_matrix(1., 2), Mat2 {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution<C2>(make_eigen_matrix(1., 2), SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution<C2>(make_eigen_matrix(1., 2), SA2l {9, 3, 3, 10}))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution<C2>(make_eigen_matrix(1., 2), SA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(make_eigen_matrix(1., 2), SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution(make_eigen_matrix(1., 2), SA2l {9, 3, 3, 10}))>::StaticDescriptor, Dimensions<2>>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution(make_eigen_matrix(1., 2), SA2l {9, 3, 3, 10}))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution<C2>(make_eigen_matrix(1., 2), make_dense_object_from<M2>(9, 3, 3, 10)), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution<C2>(make_eigen_matrix(1., 2), make_dense_object_from<M2>(9, 3, 3, 10)))>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution<C2>(make_eigen_matrix(1., 2), make_dense_object_from<M2>(9, 3, 3, 10)))>);

  EXPECT_TRUE(is_near(make_GaussianDistribution(make_eigen_matrix(1., 2), make_dense_object_from<M2>(9, 3, 3, 10)), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution(make_eigen_matrix(1., 2), make_dense_object_from<M2>(9, 3, 3, 10)))>::StaticDescriptor, Dimensions<2>>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution(make_eigen_matrix(1., 2), make_dense_object_from<M2>(9, 3, 3, 10)))>);

  // Defaults
  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution<Mean2, CovSA2l>())>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution<Mean2, CovSA2l>())>);

  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution<Mean2, SA2l>())>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution<Mean2, SA2l>())>);

  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution<M2col, CovSA2l>())>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution<M2col, CovSA2l>())>);

  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution<C2, M2col, SA2l>())>::StaticDescriptor, C2>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution<C2, M2col, SA2l>())>);

  static_assert(compares_with<typename DistributionTraits<decltype(make_GaussianDistribution<M2col, SA2l>())>::StaticDescriptor, Dimensions<2>>);
  static_assert(hermitian_matrix<decltype(make_GaussianDistribution<M2col, SA2l>())>);
}


TEST(matrices, GaussianDistribution_traits)
{
  static_assert(not diagonal_matrix<DistSA2l>);
  static_assert(hermitian_matrix<DistSA2l>);
  static_assert(not cholesky_form<DistSA2l>);
  static_assert(not triangular_matrix<DistSA2l>);
  static_assert(not triangular_matrix<DistSA2l, TriangleType::lower>);
  static_assert(not triangular_matrix<DistSA2l, TriangleType::upper>);
  static_assert(not zero<DistSA2l>);

  static_assert(not diagonal_matrix<DistT2l>);
  static_assert(hermitian_matrix<DistT2l>);
  static_assert(cholesky_form<DistT2l>);
  static_assert(not triangular_matrix<DistT2l>);
  static_assert(not triangular_matrix<DistT2l, TriangleType::lower>);
  static_assert(not triangular_matrix<DistT2l, TriangleType::upper>);
  static_assert(not triangular_matrix<DistT2l, TriangleType::upper>);
  static_assert(not zero<DistT2l>);

  static_assert(diagonal_matrix<DistD2>);
  static_assert(hermitian_matrix<DistD2>);
  static_assert(not cholesky_form<DistD2>);
  static_assert(triangular_matrix<DistD2>);
  static_assert(triangular_matrix<DistD2, TriangleType::lower>);
  static_assert(triangular_matrix<DistD2, TriangleType::upper>);
  static_assert(triangular_matrix<DistD2, TriangleType::upper>);
  static_assert(not zero<DistD2>);

  static_assert(diagonal_matrix<DistI2>);
  static_assert(hermitian_matrix<DistI2>);
  static_assert(not cholesky_form<DistI2>);
  static_assert(triangular_matrix<DistI2>);
  static_assert(triangular_matrix<DistI2, TriangleType::lower>);
  static_assert(triangular_matrix<DistI2, TriangleType::upper>);
  static_assert(triangular_matrix<DistI2, TriangleType::upper>);
  static_assert(not zero<DistI2>);

  static_assert(diagonal_matrix<DistZ2>);
  static_assert(hermitian_matrix<DistZ2>);
  static_assert(not cholesky_form<DistZ2>);
  static_assert(triangular_matrix<DistZ2>);
  static_assert(triangular_matrix<DistZ2, TriangleType::lower>);
  static_assert(triangular_matrix<DistZ2, TriangleType::upper>);
  static_assert(triangular_matrix<DistZ2, TriangleType::upper>);
  static_assert(zero<DistZ2>);

  // DistributionTraits
  EXPECT_TRUE(is_near(DistributionTraits<DistSA2l>::template make<C2>(nested_object(Mean2 {1, 2}), nested_object(CovSA2l {9, 3, 3, 10})), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  EXPECT_TRUE(is_near(DistributionTraits<DistSA2l>::template make<C2>(nested_object(Mean2 {1, 2}), SA2l {9, 3, 3, 10}), DistSA2l {{1, 2}, {9, 3, 3, 10}}));
  EXPECT_TRUE(is_near(make_zero_distribution_like<DistSA2l>(), distz2));
  EXPECT_TRUE(is_near(make_normal_distribution_like<DistSA2l>(), disti2));
}


TEST(matrices, GaussianDistribution_overloads)
{
  // mean
  EXPECT_TRUE(is_near(mean_of(DistSA2l {{1, 2}, {9, 3, 3, 10}}), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(mean_of(DistSA2u {{1, 2}, {9, 3, 3, 10}}), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(mean_of(DistT2l {{1, 2}, {9, 3, 3, 10}}), Mean2 {1, 2}));
  EXPECT_TRUE(is_near(mean_of(DistT2u {{1, 2}, {9, 3, 3, 10}}), Mean2 {1, 2}));

  // covariance
  EXPECT_TRUE(is_near(covariance_of(DistSA2l {{1, 2}, {9, 3, 3, 10}}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(covariance_of(DistSA2u {{1, 2}, {9, 3, 3, 10}}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(nested_object(covariance_of(DistT2l {{1, 2}, {9, 3, 3, 10}})), Mat2 { 3, 0, 1, 3}));
  EXPECT_TRUE(is_near(nested_object(covariance_of(DistT2u {{1, 2}, {9, 3, 3, 10}})), Mat2 { 3, 1, 0, 3}));

  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(DistSA2l {Mean2 {1, 2} * 2, CovSA2l{9, 3, 3, 10} * 2}))>, DistSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(DistSA2u {Mean2 {1, 2} * 2, CovSA2u{9, 3, 3, 10} * 2}))>, DistSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(DistT2l {Mean2 {1, 2} * 2, CovT2l{9, 3, 3, 10} * 2}))>, DistT2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(DistT2u {Mean2 {1, 2} * 2, CovT2u{9, 3, 3, 10} * 2}))>, DistT2u>);
}


TEST(matrices, GaussianDistribution_blocks)
{
  Mean2 a1 {1, 2}, a2 {3, 4};
  Mean4 b {1, 2, 3, 4};
  Mat2 ma {9, 3, 3, 10}, mb {4, 2, 2, 5};
  Mat4 n {9, 3, 0, 0,
          3, 10, 0, 0,
          0, 0, 4, 2,
          0, 0, 2, 5};
  EXPECT_TRUE(is_near(concatenate(DistSA2l(a1, ma), DistSA2l(a2, mb)), DistSA4l(b, n)));
  EXPECT_TRUE(is_near(concatenate(DistSA2u(a1, ma), DistSA2u(a2, mb)), DistSA4u(b, n)));
  EXPECT_TRUE(is_near(concatenate(DistT2l(a1, ma), DistT2l(a2, mb)), DistT4l(b, n)));
  EXPECT_TRUE(is_near(concatenate(DistT2u(a1, ma), DistT2u(a2, mb)), DistT4u(b, n)));

  EXPECT_TRUE(is_near(split(DistSA4l(b, n)), std::tuple {}));
  EXPECT_TRUE(is_near(split<C2, C2>(DistSA4l(b, n)), std::tuple {DistSA2l(a1, ma), DistSA2l(a2, mb)}));
  EXPECT_TRUE(is_near(split<C2, C2>(DistSA4u(b, n)), std::tuple {DistSA2u(a1, ma), DistSA2u(a2, mb)}));
  EXPECT_TRUE(is_near(split<C2, C2>(DistT4l(b, n)), std::tuple {DistT2l(a1, ma), DistT2l(a2, mb)}));
  EXPECT_TRUE(is_near(split<C2, C2>(DistT4u(b, n)), std::tuple {DistT2u(a1, ma), DistT2u(a2, mb)}));
  EXPECT_TRUE(is_near(split<C2, angle::Radians>(DistSA4l(b, n)), std::tuple {DistSA2l(a1, ma), GaussianDistribution {Mean1 {3}, SA1l {4}}}));
  EXPECT_TRUE(is_near(split<C2, angle::Radians>(DistSA4u(b, n)), std::tuple {DistSA2u(a1, ma), GaussianDistribution {Mean1 {3}, SA1u {4}}}));
  EXPECT_TRUE(is_near(split<C2, angle::Radians>(DistT4l(b, n)), std::tuple {DistT2l(a1, ma), GaussianDistribution {Mean1 {3}, SA1l {4}}}));
  EXPECT_TRUE(is_near(split<C2, angle::Radians>(DistT4u(b, n)), std::tuple {DistT2u(a1, ma), GaussianDistribution {Mean1 {3}, SA1u {4}}}));
}


TEST(matrices, GaussianDistribution_addition_subtraction)
{
  Mean<Dimensions<2>> x_mean {20, 30};
  M2 d;
  d << 9, 3,
  3, 8;
  GaussianDistribution dist1 {x_mean, HermitianAdapter(d)};
  Mean<Dimensions<2>> y_mean {11, 23};
  M2 e;
  e << 7, 1,
  1, 3;
  GaussianDistribution dist2 {y_mean, HermitianAdapter(e)};
  auto sum1 = dist1 + dist2;
  EXPECT_TRUE(is_near(mean_of(sum1), Mean {31., 53}));
  EXPECT_TRUE(is_near(covariance_of(sum1), Covariance {16., 4, 4, 11}));
  auto diff1 = dist1 - dist2;
  EXPECT_TRUE(is_near(mean_of(diff1), Mean {9., 7.}));
  EXPECT_TRUE(is_near(covariance_of(diff1), Covariance {2., 2, 2, 5}));
  GaussianDistribution dist1_chol {x_mean, cholesky_factor(HermitianAdapter {d})};
  GaussianDistribution dist2_chol {y_mean, cholesky_factor(HermitianAdapter {e})};
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


TEST(matrices, GaussianDistribution_mult_div)
{
  auto a = GaussianDistribution(make_mean<C2>(2., 30), make_covariance<C2>(8., 2, 2, 6));
  auto a_chol = GaussianDistribution(make_mean<C2>(2., 30), make_covariance<C2, TriangleType::lower>(8., 2, 2, 6));
  auto f_matrix = make_vector_space_adapter<C3, C2>(1., 2, 3, 4, 5, 6);
  auto a_scaled3 = f_matrix * a;
  EXPECT_TRUE(is_near(mean_of(a_scaled3), make_mean<C3>(62., 126, 190)));
  EXPECT_TRUE(is_near(covariance_of(a_scaled3), make_vector_space_adapter<C3, C3>(40., 92, 144, 92, 216, 340, 144, 340, 536)));
  static_assert(compares_with<typename DistributionTraits<decltype(a_scaled3)>::StaticDescriptor, C3>);
  auto a_chol_scaled3 = f_matrix * a_chol;
  EXPECT_TRUE(is_near(mean_of(a_chol_scaled3), make_mean<C3>(62., 126, 190)));
  EXPECT_TRUE(is_near(covariance_of(a_chol_scaled3), make_vector_space_adapter<C3, C3>(40., 92, 144, 92, 216, 340, 144, 340, 536)));
  static_assert(compares_with<typename DistributionTraits<decltype(a_chol_scaled3)>::StaticDescriptor, C3>);

  eigen_matrix_t<double, 2, 2> cov_mat; cov_mat << 8, 2, 2, 6;
  decltype(a) a_scaled {a * 2};
  EXPECT_TRUE(is_near(mean_of(a_scaled), mean_of(a) * 2));
  EXPECT_TRUE(is_near(covariance_of(a_scaled), covariance_of(a) * 4));
  decltype(a_chol) a_chol_scaled {a_chol * 2};
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


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
using namespace OpenKalman::test;

using M2 = eigen_matrix_t<double, 2, 2>;
using C = Coefficients<angle::Radians, Axis>;
using Mat2 = Matrix<C, C, M2>;
using Mat2col = Matrix<C, Axis, eigen_matrix_t<double, 2, 1>>;
using SA2l = SelfAdjointMatrix<M2, TriangleType::lower>;
using SA2u = SelfAdjointMatrix<M2, TriangleType::upper>;
using T2l = TriangularMatrix<M2, TriangleType::lower>;
using T2u = TriangularMatrix<M2, TriangleType::upper>;
using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
using D1 = eigen_matrix_t<double, 1, 1>;
using I2 = IdentityMatrix<eigen_matrix_t<double, 2, 2>>;
using Z2 = ZeroMatrix<double, 2, 2>;
using CovSA2l = Covariance<C, SA2l>;
using CovSA2u = Covariance<C, SA2u>;
using CovT2l = Covariance<C, T2l>;
using CovT2u = Covariance<C, T2u>;
using CovD2 = Covariance<C, D2>;
using CovD1 = Covariance<Axis, D1>;
using CovI2 = Covariance<C, I2>;
using CovZ2 = Covariance<C, Z2>;
using SqCovSA2l = SquareRootCovariance<C, SA2l>;
using SqCovSA2u = SquareRootCovariance<C, SA2u>;
using SqCovT2l = SquareRootCovariance<C, T2l>;
using SqCovT2u = SquareRootCovariance<C, T2u>;
using SqCovD2 = SquareRootCovariance<C, D2>;
using SqCovD1 = SquareRootCovariance<Axis, D1>;
using SqCovI2 = SquareRootCovariance<C, I2>;
using SqCovZ2 = SquareRootCovariance<C, Z2>;

inline I2 i2 = M2::Identity();
inline Z2 z2 = ZeroMatrix<double, 2, 2>();
inline CovI2 covi2 {i2};
inline CovZ2 covz2 {z2};
inline SqCovI2 sqcovi2 {i2};
inline SqCovZ2 sqcovz2 {z2};


TEST(covariance_tests, SquareRootCovariance_convert_nested_matrix)
{
  using namespace OpenKalman::internal;
  EXPECT_TRUE(is_near(to_covariance_nestable(SqCovSA2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(SqCovSA2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(SqCovSA2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(SqCovSA2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(SqCovSA2l {3, 0, 1, 3}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(SqCovSA2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(SqCovSA2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(SqCovSA2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(SqCovSA2u {3, 1, 0, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(SqCovSA2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(SqCovT2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(SqCovT2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(SqCovT2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(SqCovT2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(SqCovT2l {3, 0, 1, 3}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(SqCovT2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(SqCovT2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(SqCovT2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(SqCovT2u {3, 1, 0, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(SqCovT2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(SqCovD2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable(SqCovD2 {1, 2}).nested_matrix(), Mean {1., 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(SqCovD2 {2, 3}), Mat2 {2, 0, 0, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(SqCovD2 {2, 3}), Mat2 {2, 0, 0, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(SqCovD2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(SqCovD2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(SqCovD2 {2, 3}), Mat2 {2, 0, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(sqcovi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(sqcovi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(sqcovi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(sqcovi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(sqcovi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(sqcovi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<I2>(sqcovi2), Mat2 {1, 0, 0, 1}));

  EXPECT_TRUE(is_near(to_covariance_nestable(sqcovz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(sqcovz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(sqcovz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(sqcovz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(sqcovz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(sqcovz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<Z2>(sqcovz2), Mat2 {0, 0, 0, 0}));
}


TEST(covariance_tests, SquareRootCovariance_class)
{
  // Default constructor and Eigen3 construction
  SqCovSA2l clsa1;
  clsa1 << 3, 0, 1, 3;
  EXPECT_TRUE(is_near(clsa1, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(const_cast<const SqCovSA2l&>(clsa1), Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusa1;
  cusa1 << 2, 1, 0, 2;
  EXPECT_TRUE(is_near(cusa1, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(const_cast<const SqCovSA2u&>(cusa1), Mat2 {2, 1, 0, 2}));
  SqCovT2l clt1;
  clt1 << 3, 0, 1, 3;
  EXPECT_TRUE(is_near(clt1, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(const_cast<const SqCovT2l&>(clt1), Mat2 {3, 0, 1, 3}));
  SqCovT2u cut1;
  cut1 << 2, 1, 0, 2;
  EXPECT_TRUE(is_near(cut1, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(const_cast<const SqCovT2u&>(cut1), Mat2 {2, 1, 0, 2}));
  SqCovD2 cd1;
  cd1 << 1, 2;
  EXPECT_TRUE(is_near(cd1, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(const_cast<const SqCovD2&>(cd1), Mat2 {1, 0, 0, 2}));
  SqCovI2 ci1(i2);
  EXPECT_TRUE(is_near(ci1, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(const_cast<const SqCovI2&>(ci1), Mat2 {1, 0, 0, 1}));

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
  EXPECT_TRUE(is_near(cd2.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 2}));
  SqCovD2 cd2b = cd1; // Template constructor, not copy constructor.
  EXPECT_TRUE(is_near(cd2b, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(cd2b.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 2}));
  SqCovI2 ci2 = const_cast<const SqCovI2&>(ci1);
  EXPECT_TRUE(is_near(ci2, Mat2 {1, 0, 0, 1}));
  SqCovI2 ci2b = ci1; // Template constructor, not copy constructor.
  EXPECT_TRUE(is_near(ci2b, Mat2 {1, 0, 0, 1}));

  // Move constructor
  auto xa = SqCovSA2l {3, 0, 1, 3};
  SqCovSA2l clsa3(std::move(xa));
  EXPECT_TRUE(is_near(clsa3, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(clsa3.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(clsa3.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(const_cast<const SqCovSA2l&>(clsa3), Mat2 {3, 0, 1, 3}));
  auto xb = SqCovSA2u {2, 1, 0, 2};
  SqCovSA2u cusa3(std::move(xb));
  EXPECT_TRUE(is_near(cusa3, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(cusa3.get_self_adjoint_nested_matrix(), Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(cusa3.get_triangular_nested_matrix(), Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(const_cast<const SqCovSA2u&>(cusa3), Mat2 {2, 1, 0, 2}));
  auto xc = SqCovT2l {3, 0, 1, 3};
  SqCovT2l clt3(std::move(xc));
  EXPECT_TRUE(is_near(clt3, Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(clt3.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(clt3.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(const_cast<const SqCovT2l&>(clt3), Mat2 {3, 0, 1, 3}));
  auto xd = SqCovT2u {2, 1, 0, 2};
  SqCovT2u cut3(std::move(xd));
  EXPECT_TRUE(is_near(cut3, Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(cut3.get_self_adjoint_nested_matrix(), Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(cut3.get_triangular_nested_matrix(), Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(const_cast<const SqCovT2u&>(cut3), Mat2 {2, 1, 0, 2}));
  auto xe = SqCovD2 {1, 2};
  SqCovD2 cd3(std::move(xe));
  EXPECT_TRUE(is_near(cd3, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(cd3.get_self_adjoint_nested_matrix(), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(cd3.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(const_cast<const SqCovD2&>(cd3), Mat2 {1, 0, 0, 2}));
  auto xf = SqCovI2 {i2};
  SqCovI2 ci3(std::move(xf));
  EXPECT_TRUE(is_near(ci3, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(ci3.get_self_adjoint_nested_matrix(), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(ci3.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(const_cast<const SqCovI2&>(ci3), Mat2 {1, 0, 0, 1}));

  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3}.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovT2l {3, 0, 1, 3}.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3}.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovT2u {3, 1, 0, 3}.get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3}.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovSA2l {3, 0, 1, 3}.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3}.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(SqCovSA2u {3, 1, 0, 3}.get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3}.get_self_adjoint_nested_matrix(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(SqCovD2 {3, 3}.get_triangular_nested_matrix(), Mat2 {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(SqCovI2 {i2}.get_self_adjoint_nested_matrix(), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(SqCovI2 {i2}.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 1}));

  // Convert from different square-root covariance type
  SqCovSA2l clsasa4 {adjoint(SqCovSA2u {3, 1, 0, 3})}; EXPECT_TRUE(is_near(clsasa4, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusasa4 {adjoint(SqCovSA2l {2, 0, 1, 2})}; EXPECT_TRUE(is_near(cusasa4, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsat4 {adjoint(SqCovT2u {3, 1, 0, 3})}; EXPECT_TRUE(is_near(clsat4, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusat4 {adjoint(SqCovT2l {2, 0, 1, 2})}; EXPECT_TRUE(is_near(cusat4, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsat4s {SqCovT2l {3, 0, 1, 3}}; EXPECT_TRUE(is_near(clsat4s, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusat4s {SqCovT2u {2, 1, 0, 2}}; EXPECT_TRUE(is_near(cusat4s, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltt4 {adjoint(SqCovT2u {3, 1, 0, 3})}; EXPECT_TRUE(is_near(cltt4, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutt4 {adjoint(SqCovT2l {2, 0, 1, 2})}; EXPECT_TRUE(is_near(cutt4, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltsa4 {adjoint(SqCovSA2u {3, 1, 0, 3})}; EXPECT_TRUE(is_near(cltsa4, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutsa4 {adjoint(SqCovSA2l {2, 0, 1, 2})}; EXPECT_TRUE(is_near(cutsa4, Mat2 {2, 1, 0, 2}));
  SqCovT2l cltsa4s {SqCovSA2l {3, 0, 1, 3}}; EXPECT_TRUE(is_near(cltsa4s, Mat2 {3, 0, 1, 3}));
  SqCovT2u cutsa4s {SqCovSA2u {2, 1, 0, 2}}; EXPECT_TRUE(is_near(cutsa4s, Mat2 {2, 1, 0, 2}));
  SqCovSA2l clsa4d {SqCovD2 {1, 2}}; EXPECT_TRUE(is_near(clsa4d, Mat2 {1, 0, 0, 2}));
  SqCovSA2u cusa4d {SqCovD2 {1, 2}}; EXPECT_TRUE(is_near(cusa4d, Mat2 {1, 0, 0, 2}));
  SqCovT2l clt4d {SqCovD2 {1, 2}}; EXPECT_TRUE(is_near(clt4d, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(clt4d.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 2}));
  SqCovT2u cut4d {SqCovD2 {1, 2}}; EXPECT_TRUE(is_near(cut4d, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(cut4d.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 2}));
  SqCovT2u cut4i {SqCovI2 {i2}}; EXPECT_TRUE(is_near(cut4i, Mat2 {1, 0, 0, 1}));

  /*// Convert from different non-square-root covariance type
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
  SqCovD2 cd4d2X(CovD2 {4, 9});
  EXPECT_TRUE(is_near(cd4d2X, D2 {2, 3}));
  SqCovD1 cd4d1X(CovD1 {4});
  EXPECT_TRUE(is_near(cd4d1X, D1 {2}));
  */

  // Construct from a covariance_nestable
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
  SqCovD2 cd6(Matrix<C, Axis, eigen_matrix_t<double, 2, 1>> {1, 2});
  EXPECT_TRUE(is_near(cd6, Mat2 {1, 0, 0, 2}));

  // Construct from a regular matrix
  SqCovSA2l clsa7(make_native_matrix<M2>(3, 7, 1, 3));
  EXPECT_TRUE(is_near(clsa7, Mat2 {3, 0, 1, 3}));
  SqCovSA2u cusa7(make_native_matrix<M2>(2, 1, 7, 2));
  EXPECT_TRUE(is_near(cusa7, Mat2 {2, 1, 0, 2}));
  SqCovT2l clt7(make_native_matrix<M2>(3, 7, 1, 3));
  EXPECT_TRUE(is_near(clt7, Mat2 {3, 0, 1, 3}));
  SqCovT2u cut7(make_native_matrix<M2>(2, 1, 7, 2));
  EXPECT_TRUE(is_near(cut7, Mat2 {2, 1, 0, 2}));
  SqCovD2 cd7(make_native_matrix<double, 2, 1>(1, 2));
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
  auto ya = SqCovSA2l {3, 0, 1, 3};
  clsa2 = std::move(ya);
  EXPECT_TRUE(is_near(clsa2, Mat2 {3, 0, 1, 3}));
  auto yb = SqCovSA2u {2, 1, 0, 2};
  cusa2 = std::move(yb);
  EXPECT_TRUE(is_near(cusa2, Mat2 {2, 1, 0, 2}));
  auto yc = SqCovT2l {3, 0, 1, 3};
  clt2 = std::move(yc);
  EXPECT_TRUE(is_near(clt2, Mat2 {3, 0, 1, 3}));
  auto yd = SqCovT2u {2, 1, 0, 2};
  cut2 = std::move(yd);
  EXPECT_TRUE(is_near(cut2, Mat2 {2, 1, 0, 2}));
  auto ye = SqCovD2 {1, 2};
  cd2 = std::move(ye);
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
  clsasa4 = adjoint(SqCovSA2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(clsasa4, Mat2 {2, 0, 1, 2}));
  cusasa4 = adjoint(SqCovSA2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(cusasa4, Mat2 {3, 1, 0, 3}));
  clsat4 = adjoint(SqCovT2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(clsat4, Mat2 {2, 0, 1, 2}));
  cusat4 = adjoint(SqCovT2l {3, 0, 1, 3});
  EXPECT_TRUE(is_near(cusat4, Mat2 {3, 1, 0, 3}));
  clsat4s = SqCovT2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clsat4s, Mat2 {2, 0, 1, 2}));
  cusat4s = SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cusat4s, Mat2 {3, 1, 0, 3}));
  cltsa4 = adjoint(SqCovSA2u {2, 1, 0, 2});
  EXPECT_TRUE(is_near(cltsa4, Mat2 {2, 0, 1, 2}));
  cutsa4 = adjoint(SqCovSA2l {3, 0, 1, 3});
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

  /*// Assign from different non-square-root covariance type
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
  cd4d2X = CovD2 {9, 16};
  EXPECT_TRUE(is_near(cd4d2X, D2 {3, 4}));
  cd4d1X = CovD1 {9};
  EXPECT_TRUE(is_near(cd4d1X, D1 {3}));
  */

  // Increment
  clsa8 += {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clsa8.get_self_adjoint_nested_matrix(), Mat2 {25, 10, 10, 29}));
  EXPECT_TRUE(is_near(clsa8, Mat2 {5, 0, 2, 5}));
  cusa8 += SqCovSA2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cusa8.get_self_adjoint_nested_matrix(), Mat2 {25, 10, 10, 29}));
  EXPECT_TRUE(is_near(cusa8, Mat2 {5, 2, 0, 5}));
  clt8 += SqCovT2l {2, 0, 1, 2};
  EXPECT_TRUE(is_near(clt8, Mat2 {5, 0, 2, 5}));
  cut8 += SqCovT2u {3, 1, 0, 3};
  EXPECT_TRUE(is_near(cut8, Mat2 {5, 2, 0, 5}));
  cd8 += SqCovD2 {1, 2};
  EXPECT_TRUE(is_near(cd8, Mat2 {4, 0, 0, 6}));

  // Decrement
  clsa8 -= {2, 0, 1, 2};
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
}

TEST(covariance_tests, SquareRootCovariance_subscripts)
{
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
  EXPECT_NEAR((SqCovD2 {3, 4})(0, 0), 3, 1e-6);
  EXPECT_NEAR((SqCovD2 {3, 4})(1, 1), 4, 1e-6);
  EXPECT_NEAR((SqCovD2 {3, 4})(0), 3, 1e-6);
  EXPECT_NEAR((SqCovD2 {3, 4})(1), 4, 1e-6);

  EXPECT_NEAR(get_element(SqCovSA2l {3, 0, 1, 3}, 0, 0), 3, 1e-6);
  EXPECT_NEAR(get_element(SqCovSA2u {3, 1, 0, 3}, 0, 1), 1, 1e-6);
  EXPECT_NEAR(get_element(SqCovT2l {3, 0, 1, 3}, 1, 0), 1, 1e-6);
  EXPECT_NEAR(get_element(SqCovT2u {3, 1, 0, 3}, 1, 1), 3, 1e-6);
  EXPECT_NEAR(get_element(SqCovD2 {3, 4}, 0, 0), 3, 1e-6);
  EXPECT_NEAR(get_element(SqCovD2 {3, 4}, 1, 1), 4, 1e-6);
  EXPECT_NEAR(get_element(SqCovD2 {3, 4}, 0), 3, 1e-6);
  EXPECT_NEAR(get_element(SqCovD2 {3, 4}, 1), 4, 1e-6);

  auto sa2l = SqCovSA2l {3, 0, 1, 3};
  sa2l(1, 0) = 1.1; EXPECT_NEAR(get_element(sa2l, 1, 0), 1.1, 1e-6);
  set_element(sa2l, 1.2, 1, 0); EXPECT_NEAR(sa2l(1, 0), 1.2, 1e-6);
  auto sa2u = SqCovSA2u {3, 1, 0, 3};
  sa2u(0, 1) = 1.1; EXPECT_NEAR(get_element(sa2u, 0, 1), 1.1, 1e-6);
  set_element(sa2u, 1.2, 0, 1); EXPECT_NEAR(sa2u(0, 1), 1.2, 1e-6);
  auto t2l = SqCovT2l {3, 0, 1, 3};
  t2l(1, 0) = 1.1; EXPECT_NEAR(get_element(t2l, 1, 0), 1.1, 1e-6);
  set_element(t2l, 1.2, 1, 0); EXPECT_NEAR(t2l(1, 0), 1.2, 1e-6);
  auto t2u = SqCovT2u {3, 1, 0, 3};
  t2u(0, 1) = 1.1; EXPECT_NEAR(get_element(t2u, 0, 1), 1.1, 1e-6);
  set_element(t2u, 1.2, 0, 1); EXPECT_NEAR(t2u(0, 1), 1.2, 1e-6);
  auto d2 = SqCovD2 {3, 4};
  d2(1, 1) = 4.1; EXPECT_NEAR(get_element(d2, 1, 1), 4.1, 1e-6);
  set_element(d2, 4.2, 1, 1); EXPECT_NEAR(d2(1, 1), 4.2, 1e-6);
  d2(0) = 3.1; EXPECT_NEAR(get_element(d2, 0), 3.1, 1e-6);
  set_element(d2, 3.2, 0); EXPECT_NEAR(d2(0), 3.2, 1e-6);
}

TEST(covariance_tests, SquareRootCovariance_deduction_guides)
{
  EXPECT_TRUE(is_near(SquareRootCovariance(SqCovSA2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(SquareRootCovariance(SqCovSA2l {3, 0, 1, 3}))>::RowCoefficients, C>);

  EXPECT_TRUE(is_near(SquareRootCovariance(SqCovSA2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(SquareRootCovariance(SqCovSA2l {3, 1, 0, 3}))>::RowCoefficients, C>);

  EXPECT_TRUE(is_near(SquareRootCovariance(T2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(SquareRootCovariance(T2l {3, 0, 1, 3}))>::RowCoefficients, Axes<2>>);

  EXPECT_TRUE(is_near(SquareRootCovariance(T2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(SquareRootCovariance(T2u {3, 1, 0, 3}))>::RowCoefficients, Axes<2>>);

  EXPECT_TRUE(is_near(SquareRootCovariance(D2 {1, 2}), Mat2 {1, 0, 0, 2}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(SquareRootCovariance(D2 {1, 2}))>::RowCoefficients, Axes<2>>);

  EXPECT_TRUE(is_near(SquareRootCovariance(Mat2 {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(SquareRootCovariance(Mat2 {3, 0, 1, 3}))>::RowCoefficients, C>);

  EXPECT_TRUE(is_near(SquareRootCovariance(make_native_matrix<M2>(3, 0, 1, 3)), Mat2 {3, 0, 1, 3}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(SquareRootCovariance(make_native_matrix<M2>(3, 0, 1, 3)))>::RowCoefficients, Axes<2>>);

  EXPECT_TRUE(is_near(SquareRootCovariance {3., 0, 1, 3}, Mat2 {3, 0, 1, 3}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(SquareRootCovariance {3., 0, 1, 3})>::RowCoefficients, Axes<2>>);
}

TEST(covariance_tests, SquareRootCovariance_make)
{
  // Other covariance:
  EXPECT_TRUE(is_near(make_square_root_covariance(SqCovSA2l {3, 0, 1, 3}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_square_root_covariance(SqCovSA2u {3, 1, 0, 3}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_square_root_covariance(SqCovT2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance(SqCovT2u {3, 1, 0, 3}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance(SqCovD2 {1, 2}).get_triangular_nested_matrix(), Mat2 {1, 0, 0, 2}));

  EXPECT_TRUE(is_near(make_square_root_covariance(CovSA2l {9, 3, 3, 10}.square_root()).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_square_root_covariance(CovSA2u {9, 3, 3, 10}.square_root()).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_square_root_covariance(CovT2l {9, 3, 3, 10}.square_root()).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance(CovT2u {9, 3, 3, 10}.square_root()).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance(CovD2 {4, 9}.square_root()).get_triangular_nested_matrix(), Mat2 {2, 0, 0, 3}));

  static_assert(Eigen3::lower_self_adjoint_matrix<decltype(make_square_root_covariance(SqCovSA2l {3, 0, 1, 3}).get_self_adjoint_nested_matrix())>);
  static_assert(Eigen3::upper_self_adjoint_matrix<decltype(make_square_root_covariance(SqCovSA2u {3, 1, 0, 3}).get_self_adjoint_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance(SqCovT2l {3, 0, 1, 3}).get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance(SqCovT2u {3, 1, 0, 3}).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype(make_square_root_covariance(SqCovD2 {1, 2}).get_triangular_nested_matrix())>);

  static_assert(Eigen3::upper_self_adjoint_matrix<decltype(make_square_root_covariance(adjoint(SqCovSA2l {3, 0, 1, 3})).get_self_adjoint_nested_matrix())>);
  static_assert(Eigen3::lower_self_adjoint_matrix<decltype(make_square_root_covariance(adjoint(SqCovSA2u {3, 1, 0, 3})).get_self_adjoint_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance(adjoint(SqCovT2l {3, 0, 1, 3})).get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance(adjoint(SqCovT2u {3, 1, 0, 3})).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype(make_square_root_covariance(adjoint(SqCovD2 {1, 2})).get_triangular_nested_matrix())>);

  static_assert(Eigen3::lower_self_adjoint_matrix<decltype(make_square_root_covariance<SqCovSA2l>().get_self_adjoint_nested_matrix())>);
  static_assert(Eigen3::upper_self_adjoint_matrix<decltype(make_square_root_covariance<SqCovSA2u>().get_self_adjoint_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<SqCovT2l>().get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<SqCovT2u>().get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype(make_square_root_covariance<SqCovD2>().get_triangular_nested_matrix())>);

  static_assert(Eigen3::lower_self_adjoint_matrix<decltype(make_square_root_covariance<SqCovSA2l>().get_self_adjoint_nested_matrix())>);
  static_assert(Eigen3::upper_self_adjoint_matrix<decltype(make_square_root_covariance<SqCovSA2u>().get_self_adjoint_nested_matrix())>);
  static_assert(diagonal_matrix<decltype(make_square_root_covariance<SqCovD2>().get_triangular_nested_matrix())>);

  // SquareRootCovariance bases:
  EXPECT_TRUE(is_near(make_square_root_covariance<C>(SA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_square_root_covariance<C>(SA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_square_root_covariance<C>(T2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance<C>(T2u {3, 1, 0, 3}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance<C>(D2 {1, 2}).get_triangular_nested_matrix(), Mat2 {1, 0, 0, 2}));

  static_assert(Eigen3::lower_self_adjoint_matrix<decltype(make_square_root_covariance<C>(SA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(Eigen3::upper_self_adjoint_matrix<decltype(make_square_root_covariance<C>(SA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C>(T2l {3, 0, 1, 3}).get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<C>(T2u {3, 1, 0, 3}).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype(make_square_root_covariance<C>(D2 {1, 2}).get_triangular_nested_matrix())>);

  static_assert(Eigen3::lower_self_adjoint_matrix<decltype(make_square_root_covariance<C>(adjoint(SA2l {9, 3, 3, 10})).get_self_adjoint_nested_matrix())>);
  static_assert(Eigen3::upper_self_adjoint_matrix<decltype(make_square_root_covariance<C>(adjoint(SA2u {9, 3, 3, 10})).get_self_adjoint_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<C>(adjoint(T2l {3, 0, 1, 3})).get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C>(adjoint(T2u {3, 1, 0, 3})).get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype(make_square_root_covariance<C>(adjoint(D2 {1, 2})).get_triangular_nested_matrix())>);

  static_assert(Eigen3::lower_self_adjoint_matrix<decltype(make_square_root_covariance<C, SA2l>().get_self_adjoint_nested_matrix())>);
  static_assert(Eigen3::upper_self_adjoint_matrix<decltype(make_square_root_covariance<C, SA2u>().get_self_adjoint_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C, T2l>().get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<C, T2u>().get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C, D2>().get_triangular_nested_matrix())>);

  static_assert(Eigen3::lower_self_adjoint_matrix<decltype(make_square_root_covariance(SA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(Eigen3::upper_self_adjoint_matrix<decltype(make_square_root_covariance<C>(SA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<T2u>().get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C, T2l>().get_triangular_nested_matrix())>);
  static_assert(diagonal_matrix<decltype(make_square_root_covariance<C, D2>().get_triangular_nested_matrix())>);

  // Regular matrices:
  EXPECT_TRUE(is_near(make_square_root_covariance<C, TriangleType::lower>(Mat2 {3, 0, 1, 3}.nested_matrix()).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance<C, TriangleType::upper>(Mat2 {3, 1, 0, 3}.nested_matrix()).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance<C>(Mat2 {3, 0, 1, 3}.nested_matrix()).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C, TriangleType::lower>(Mat2 {3, 0, 1, 3}.nested_matrix()).get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<C, TriangleType::upper>(Mat2 {3, 1, 0, 3}.nested_matrix()).get_triangular_nested_matrix())>);

  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C, TriangleType::lower, M2>().get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<C, TriangleType::upper, M2>().get_triangular_nested_matrix())>);

  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C>(Mat2 {3, 0, 1, 3}.nested_matrix()).get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<TriangleType::upper>(Mat2 {3, 1, 0, 3}.nested_matrix()).get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C, M2>().get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<TriangleType::upper, M2>().get_triangular_nested_matrix())>);

  // Typed matrices:
  EXPECT_TRUE(is_near(make_square_root_covariance<TriangleType::lower>(Mat2 {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance<TriangleType::upper>(Mat2 {3, 1, 0, 3}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance(Mat2 {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<TriangleType::lower>(Mat2 {3, 0, 1, 3}).get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<TriangleType::upper>(Mat2 {3, 1, 0, 3}).get_triangular_nested_matrix())>);

  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<TriangleType::lower, Mat2>().get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<TriangleType::upper, Mat2>().get_triangular_nested_matrix())>);

  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance(Mat2 {3, 0, 1, 3}).get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<Mat2>().get_triangular_nested_matrix())>);

  // Eigen defaults
  EXPECT_TRUE(is_near(make_square_root_covariance<C, TriangleType::lower>(3., 0, 1, 3).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance<C, TriangleType::upper>(3., 1, 0, 3).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_square_root_covariance<C>(3., 0, 1, 3).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C, TriangleType::lower>(3., 0, 1, 3).get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<C, TriangleType::upper>(3., 1, 0, 3).get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C>(9., 3, 3, 10).get_triangular_nested_matrix())>);

  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<TriangleType::lower>(3., 0, 1, 3).get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C, TriangleType::lower>().get_triangular_nested_matrix())>);
  static_assert(upper_triangular_matrix<decltype(make_square_root_covariance<C, TriangleType::upper>().get_triangular_nested_matrix())>);
  static_assert(lower_triangular_matrix<decltype(make_square_root_covariance<C>().get_triangular_nested_matrix())>);
  static_assert(MatrixTraits<decltype(make_square_root_covariance<C>())>::RowCoefficients::dimensions == 2);
}

TEST(covariance_tests, SquareRootCovariance_traits)
{
  static_assert(triangular_covariance<SqCovSA2l>);
  static_assert(not self_adjoint_covariance<SqCovSA2l>);
  static_assert(not diagonal_matrix<SqCovSA2l>);
  static_assert(not self_adjoint_matrix<SqCovSA2l>);
  static_assert(not cholesky_form<SqCovSA2l>);
  static_assert(triangular_matrix<SqCovSA2l>);
  static_assert(triangular_matrix<SqCovSA2u>);
  static_assert(lower_triangular_matrix<SqCovSA2l>);
  static_assert(upper_triangular_matrix<SqCovSA2u>);
  static_assert(not identity_matrix<SqCovSA2l>);
  static_assert(not zero_matrix<SqCovSA2l>);

  static_assert(triangular_covariance<SqCovT2l>);
  static_assert(not self_adjoint_covariance<SqCovT2l>);
  static_assert(not diagonal_matrix<SqCovT2l>);
  static_assert(not self_adjoint_matrix<SqCovT2l>);
  static_assert(cholesky_form<SqCovT2l>);
  static_assert(triangular_matrix<SqCovT2l>);
  static_assert(triangular_matrix<SqCovT2u>);
  static_assert(lower_triangular_matrix<SqCovT2l>);
  static_assert(not upper_triangular_matrix<SqCovT2l>);
  static_assert(upper_triangular_matrix<SqCovT2u>);
  static_assert(not identity_matrix<SqCovT2l>);
  static_assert(not zero_matrix<SqCovT2l>);

  static_assert(triangular_covariance<SqCovD2>);
  static_assert(diagonal_matrix<SqCovD2>);
  static_assert(self_adjoint_matrix<SqCovD2>);
  static_assert(not cholesky_form<SqCovD2>);
  static_assert(triangular_matrix<SqCovD2>);
  static_assert(lower_triangular_matrix<SqCovD2>);
  static_assert(upper_triangular_matrix<SqCovD2>);
  static_assert(not identity_matrix<SqCovD2>);
  static_assert(not zero_matrix<SqCovD2>);

  static_assert(triangular_covariance<SqCovI2>);
  static_assert(diagonal_matrix<SqCovI2>);
  static_assert(self_adjoint_matrix<SqCovI2>);
  static_assert(not cholesky_form<SqCovI2>);
  static_assert(triangular_matrix<SqCovI2>);
  static_assert(lower_triangular_matrix<SqCovI2>);
  static_assert(upper_triangular_matrix<SqCovI2>);
  static_assert(identity_matrix<SqCovI2>);
  static_assert(not zero_matrix<SqCovI2>);

  static_assert(triangular_covariance<SqCovZ2>);
  static_assert(diagonal_matrix<SqCovZ2>);
  static_assert(self_adjoint_matrix<SqCovZ2>);
  static_assert(not cholesky_form<SqCovZ2>);
  static_assert(triangular_matrix<SqCovZ2>);
  static_assert(lower_triangular_matrix<SqCovZ2>);
  static_assert(upper_triangular_matrix<SqCovZ2>);
  static_assert(not identity_matrix<SqCovZ2>);
  static_assert(zero_matrix<SqCovZ2>);

  // MatrixTraits
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2l>::make(SA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2u>::make(SA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2l>::make(T2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2u>::make(T2u {3, 1, 0, 3}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2l>::zero(), eigen_matrix_t<double, 2, 2>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovSA2l>::identity(), eigen_matrix_t<double, 2, 2>::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2l>::make(SA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2u>::make(SA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2l>::make(T2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2u>::make(T2u {3, 1, 0, 3}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2l>::zero(), eigen_matrix_t<double, 2, 2>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<SqCovT2l>::identity(), eigen_matrix_t<double, 2, 2>::Identity()));
}

TEST(covariance_tests, SquareRootCovariance_overloads)
{
  SqCovSA2l sqcovsa2l {3, 0, 1, 3};
  SqCovSA2u sqcovsa2u {3, 1, 0, 3};
  SqCovT2l sqcovt2l {3, 0, 1, 3};
  SqCovT2u sqcovt2u {3, 1, 0, 3};

  EXPECT_TRUE(is_near(nested_matrix(SqCovSA2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(nested_matrix(SqCovSA2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(nested_matrix(SqCovT2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(nested_matrix(SqCovT2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(square(SqCovSA2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovSA2l {3, 0, 1, 3}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovSA2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square(sqcovsa2l).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(sqcovsa2l).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  auto squaresqcovsa2l = square(SqCovSA2l {3, 0, 1, 3}); EXPECT_TRUE(is_near(squaresqcovsa2l.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(squaresqcovsa2l.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  EXPECT_TRUE(is_near(square(SqCovSA2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));

  EXPECT_TRUE(is_near(square(SqCovT2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovT2l {3, 0, 1, 3}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovT2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square(sqcovt2l).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(sqcovt2l).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  auto squaresqcovt2l = square(SqCovT2l {3, 0, 1, 3}); EXPECT_TRUE(is_near(squaresqcovt2l.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(squaresqcovt2l.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  EXPECT_TRUE(is_near(square(SqCovT2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square(SqCovD2 {2, 3}), Mat2 {4, 0, 0, 9}));
  EXPECT_TRUE(is_near(square(SqCovI2 {i2}), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(square(SquareRootCovariance<C, ZeroMatrix<double, 2, 2>>()), Mat2 {0, 0, 0, 0}));

  EXPECT_TRUE(is_near(make_native_matrix(SqCovSA2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_native_matrix(SqCovSA2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_native_matrix(SqCovT2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_native_matrix(SqCovT2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));

  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(SqCovSA2l {3, 0, 1, 3} * 2))>, SqCovSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(SqCovSA2u {3, 1, 0, 3} * 2))>, SqCovSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(SqCovT2l {3, 0, 1, 3} * 2))>, SqCovT2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(SqCovT2u {3, 1, 0, 3} * 2))>, SqCovT2u>);

  EXPECT_TRUE(is_near(transpose(SqCovSA2l {3, 0, 1, 3}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(transpose(SqCovSA2u {3, 1, 0, 3}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(transpose(SqCovT2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(transpose(SqCovT2u {3, 1, 0, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  EXPECT_TRUE(is_near(adjoint(SqCovSA2l {3, 0, 1, 3}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(adjoint(SqCovSA2u {3, 1, 0, 3}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(adjoint(SqCovT2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(adjoint(SqCovT2u {3, 1, 0, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  EXPECT_NEAR(determinant(SqCovSA2l {3, 0, 1, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(SqCovSA2u {3, 1, 0, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(SqCovT2l {3, 0, 1, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(SqCovT2u {3, 1, 0, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(square(sqcovsa2l)), 81, 1e-6);
  EXPECT_NEAR(determinant(square(sqcovsa2u)), 81, 1e-6);
  EXPECT_NEAR(determinant(square(sqcovt2l)), 81, 1e-6);
  EXPECT_NEAR(determinant(square(sqcovt2u)), 81, 1e-6);

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

TEST(covariance_tests, SquareRootCovariance_blocks)
{
  using C4 = Concatenate<C, C>;
  using M4 = eigen_matrix_t<double, 4, 4>;
  using Mat4 = Matrix<C4, C4, M4>;
  using SqCovSA4l = SquareRootCovariance<C4, SelfAdjointMatrix<M4, TriangleType::lower>>;
  using SqCovSA4u = SquareRootCovariance<C4, SelfAdjointMatrix<M4, TriangleType::upper>>;
  using SqCovT4l = SquareRootCovariance<C4, TriangularMatrix<M4, TriangleType::lower>>;
  using SqCovT4u = SquareRootCovariance<C4, TriangularMatrix<M4, TriangleType::upper>>;
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

  EXPECT_TRUE(is_near(split_diagonal(SqCovSA4l(nl)), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<C, C>(SqCovSA4l(nl)), std::tuple {m1l, m2l}));
  EXPECT_TRUE(is_near(split_diagonal<C, C>(SqCovSA4u(nu)), std::tuple {m1u, m2u}));
  EXPECT_TRUE(is_near(split_diagonal<C, C>(SqCovT4l(nl)), std::tuple {m1l, m2l}));
  EXPECT_TRUE(is_near(split_diagonal<C, C>(SqCovT4u(nu)), std::tuple {m1u, m2u}));
  EXPECT_TRUE(is_near(split_diagonal<C, C::Take<1>>(SqCovSA4l(nl)), std::tuple {m1l, Matrix<angle::Radians, angle::Radians>{2}}));
  EXPECT_TRUE(is_near(split_diagonal<C, C::Take<1>>(SqCovSA4u(nu)), std::tuple {m1u, Matrix<angle::Radians, angle::Radians>{2}}));
  EXPECT_TRUE(is_near(split_diagonal<C, C::Take<1>>(SqCovT4l(nl)), std::tuple {m1l, Matrix<angle::Radians, angle::Radians>{2}}));
  EXPECT_TRUE(is_near(split_diagonal<C, C::Take<1>>(SqCovT4u(nu)), std::tuple {m1u, Matrix<angle::Radians, angle::Radians>{2}}));

  EXPECT_TRUE(is_near(split_vertical(SqCovSA4l(nl)), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(SqCovSA4l(nl)), std::tuple {concatenate_horizontal(m1l, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2l)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(SqCovSA4u(nu)), std::tuple {concatenate_horizontal(m1u, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2u)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(SqCovT4l(nl)), std::tuple {concatenate_horizontal(m1l, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2l)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(SqCovT4u(nu)), std::tuple {concatenate_horizontal(m1u, Mat2::zero()), concatenate_horizontal(Mat2::zero(), m2u)}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(SqCovSA4l(nl)), std::tuple {concatenate_horizontal(m1l, Mat2::zero()), Matrix<angle::Radians, C4>{0, 0, 2, 0}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(SqCovSA4u(nu)), std::tuple {concatenate_horizontal(m1u, Mat2::zero()), Matrix<angle::Radians, C4>{0, 0, 2, 1}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(SqCovT4l(nl)), std::tuple {concatenate_horizontal(m1l, Mat2::zero()), Matrix<angle::Radians, C4>{0, 0, 2, 0}}));
  EXPECT_TRUE(is_near(split_vertical<C, C::Take<1>>(SqCovT4u(nu)), std::tuple {concatenate_horizontal(m1u, Mat2::zero()), Matrix<angle::Radians, C4>{0, 0, 2, 1}}));

  EXPECT_TRUE(is_near(split_horizontal(SqCovSA4l(nl)), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(SqCovSA4l(nl)), std::tuple {concatenate_vertical(m1l, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2l)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(SqCovSA4u(nu)), std::tuple {concatenate_vertical(m1u, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2u)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(SqCovT4l(nl)), std::tuple {concatenate_vertical(m1l, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2l)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(SqCovT4u(nu)), std::tuple {concatenate_vertical(m1u, Mat2::zero()), concatenate_vertical(Mat2::zero(), m2u)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(SqCovSA4l(nl)), std::tuple {concatenate_vertical(m1l, Mat2::zero()), Matrix<C4, angle::Radians>{0, 0, 2, 1}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(SqCovSA4u(nu)), std::tuple {concatenate_vertical(m1u, Mat2::zero()), Matrix<C4, angle::Radians>{0, 0, 2, 0}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(SqCovT4l(nl)), std::tuple {concatenate_vertical(m1l, Mat2::zero()), Matrix<C4, angle::Radians>{0, 0, 2, 1}}));
  EXPECT_TRUE(is_near(split_horizontal<C, C::Take<1>>(SqCovT4u(nu)), std::tuple {concatenate_vertical(m1u, Mat2::zero()), Matrix<C4, angle::Radians>{0, 0, 2, 0}}));

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


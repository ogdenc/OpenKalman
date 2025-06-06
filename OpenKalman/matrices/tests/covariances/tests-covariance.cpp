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
using D1 = eigen_matrix_t<double, 1, 1>;
using I2 = Eigen3::IdentityMatrix<M2>;
using Z2 = ZeroAdapter<eigen_matrix_t<double, 2, 2>>;
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
inline Z2 z2 = ZeroAdapter<eigen_matrix_t<double, 2, 2>>();
inline auto covi2 = CovI2(i2);
inline auto covz2 = CovZ2(z2);
inline auto sqcovi2 = SqCovI2(i2);
inline auto sqcovz2 = SqCovZ2(z2);


TEST(covariance_tests, Covariance_convert_nested_matrix)
{
  using namespace OpenKalman::internal;
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(SA2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(SA2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(SA2l {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(SA2l {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(SA2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(SA2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(SA2u {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(SA2u {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(T2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(T2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(T2l {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(T2l {3, 0, 1, 3}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(T2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(T2u {3, 1, 0, 3}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(T2u {3, 1, 0, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(T2u {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(D2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(D2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(D2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(D2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(D2 {1, 2}), Mat2 {1, 0, 0, 2}));

  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(i2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(i2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(i2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(i2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(i2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<I2>(i2), Mat2 {1, 0, 0, 1}));

  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(z2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(z2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(z2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(z2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(z2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<Z2>(z2), Mat2 {0, 0, 0, 0}));

  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(make_dense_object_from<M2>(9, 3, 3, 10)), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(make_dense_object_from<M2>(9, 3, 3, 10)), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(make_dense_object_from<M2>(3, 0, 1, 3)), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(make_dense_object_from<M2>(3, 1, 0, 3)), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(make_eigen_matrix(1., 2)), Mat2 {1, 0, 0, 2}));


  EXPECT_TRUE(is_near(to_covariance_nestable(CovSA2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(CovSA2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(CovSA2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(CovSA2l {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(CovSA2l {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(CovSA2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(CovSA2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(CovSA2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(CovSA2u {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(CovSA2u {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(CovT2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(CovT2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(CovT2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(CovT2l {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(CovT2l {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(CovT2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(CovT2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(CovT2u {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(CovT2u {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(CovT2u {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(CovD2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable(CovD2 {1, 2}).nested_object(), Mean {1., 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(CovD2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(CovD2 {1, 2}), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(CovD2 {4, 9}), Mat2 {4, 0, 0, 9}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(CovD2 {4, 9}), Mat2 {4, 0, 0, 9}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(CovD2 {2, 3}), Mat2 {2, 0, 0, 3}));

  EXPECT_TRUE(is_near(to_covariance_nestable(covi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(covi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(covi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(covi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(covi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(covi2), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(to_covariance_nestable<I2>(covi2), Mat2 {1, 0, 0, 1}));

  EXPECT_TRUE(is_near(to_covariance_nestable(covz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(covz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(covz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(covz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(covz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(covz2), Mat2 {0, 0, 0, 0}));
  EXPECT_TRUE(is_near(to_covariance_nestable<Z2>(covz2), Mat2 {0, 0, 0, 0}));

  EXPECT_TRUE(is_near(to_covariance_nestable(Mat2 {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(Mat2 {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(Mat2 {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(Mat2 {3, 0, 1, 3}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(Mat2 {3, 1, 0, 3}), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<D2>(Mean {1., 2}), Mat2 {1, 0, 0, 2}));
}


TEST(covariance_tests, Covariance_class)
{
  // Default constructor and Eigen3 construction
  CovSA2l clsa1;
  clsa1 << 9, 3, 3, 10;
  EXPECT_TRUE(is_near(clsa1, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(const_cast<const CovSA2l&>(clsa1), Mat2 {9, 3, 3, 10}));
  CovSA2u cusa1;
  cusa1 << 4, 2, 2, 5;
  EXPECT_TRUE(is_near(cusa1, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(const_cast<const CovSA2u&>(cusa1), Mat2 {4, 2, 2, 5}));
  CovT2l clt1;
  clt1 << 9, 3, 3, 10;
  EXPECT_TRUE(is_near(clt1, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(const_cast<const CovT2l&>(clt1), Mat2 {9, 3, 3, 10}));
  CovT2u cut1;
  cut1 << 4, 2, 2, 5;
  EXPECT_TRUE(is_near(cut1, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(const_cast<const CovT2u&>(cut1), Mat2 {4, 2, 2, 5}));
  CovD2 cd1;
  cd1 << 1, 2;
  EXPECT_TRUE(is_near(cd1, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(const_cast<const CovD2&>(cd1), Mat2 {1, 0, 0, 2}));
  CovI2 ci1(i2);
  EXPECT_TRUE(is_near(ci1, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(const_cast<const CovI2&>(ci1), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(Covariance<C, M2::ConstantReturnType> {M2::Constant(2)}, Mat2 {2, 2, 2, 2}));

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
  auto xa = CovSA2l {9, 3, 3, 10};
  CovSA2l clsa3(std::move(xa));
  EXPECT_TRUE(is_near(clsa3, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(clsa3.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(clsa3.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(const_cast<const CovSA2l&>(clsa3), Mat2 {9, 3, 3, 10}));
  auto xb = CovSA2u {4, 2, 2, 5};
  CovSA2u cusa3(std::move(xb));
  EXPECT_TRUE(is_near(cusa3, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(cusa3.get_triangular_nested_matrix(), Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(cusa3.get_self_adjoint_nested_matrix(), Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(const_cast<const CovSA2u&>(cusa3), Mat2 {4, 2, 2, 5}));
  auto xc = CovT2l {9, 3, 3, 10};
  CovT2l clt3(std::move(xc));
  EXPECT_TRUE(is_near(clt3, Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(clt3.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(clt3.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(const_cast<const CovT2l&>(clt3), Mat2 {9, 3, 3, 10}));
  auto xd = CovT2u {4, 2, 2, 5};
  CovT2u cut3(std::move(xd));
  EXPECT_TRUE(is_near(cut3, Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(cut3.get_triangular_nested_matrix(), Mat2 {2, 1, 0, 2}));
  EXPECT_TRUE(is_near(cut3.get_self_adjoint_nested_matrix(), Mat2 {4, 2, 2, 5}));
  EXPECT_TRUE(is_near(const_cast<const CovT2u&>(cut3), Mat2 {4, 2, 2, 5}));
  auto xe = CovD2 {1, 2};
  CovD2 cd3(std::move(xe));
  EXPECT_TRUE(is_near(cd3, Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(cd3.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 2.}));
  EXPECT_TRUE(is_near(cd3.get_self_adjoint_nested_matrix(), Mat2 {1, 0, 0, 2}));
  EXPECT_TRUE(is_near(const_cast<const CovD2&>(cd3), Mat2 {1, 0, 0, 2}));
  auto xf = CovI2 {i2};
  CovI2 ci3(std::move(xf));
  EXPECT_TRUE(is_near(ci3, Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(ci3.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(ci3.get_self_adjoint_nested_matrix(), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(const_cast<const CovI2&>(ci3), Mat2 {1, 0, 0, 1}));

  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10}.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2l {9, 3, 3, 10}.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10}.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovSA2u {9, 3, 3, 10}.get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10}.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2l {9, 3, 3, 10}.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10}.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(CovT2u {9, 3, 3, 10}.get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(CovD2 {9, 9}.get_self_adjoint_nested_matrix(), Mat2 {9, 0, 0, 9}));
  EXPECT_TRUE(is_near(CovD2 {9, 9}.get_triangular_nested_matrix(), Mat2 {9, 0, 0, 9}));
  EXPECT_TRUE(is_near(CovI2 {i2}.get_self_adjoint_nested_matrix(), Mat2 {1, 0, 0, 1}));
  EXPECT_TRUE(is_near(CovI2 {i2}.get_triangular_nested_matrix(), Mat2 {1, 0, 0, 1}));

  // Convert from different covariance type
  CovSA2l clsasa4(CovSA2u {9, 3, 3, 10}); EXPECT_TRUE(is_near(clsasa4, Mat2 {9, 3, 3, 10}));
  CovSA2u cusasa4(CovSA2l {4, 2, 2, 5}); EXPECT_TRUE(is_near(cusasa4, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat4(CovT2u {9, 3, 3, 10}); EXPECT_TRUE(is_near(clsat4, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat4(CovT2l {4, 2, 2, 5}); EXPECT_TRUE(is_near(cusat4, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat4s(CovT2l {9, 3, 3, 10}); EXPECT_TRUE(is_near(clsat4s, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat4s(CovT2u {4, 2, 2, 5}); EXPECT_TRUE(is_near(cusat4s, Mat2 {4, 2, 2, 5}));
  CovT2l cltt4(CovT2u {9, 3, 3, 10}); EXPECT_TRUE(is_near(cltt4, Mat2 {9, 3, 3, 10}));
  CovT2u cutt4(CovT2l {4, 2, 2, 5}); EXPECT_TRUE(is_near(cutt4, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa4(CovSA2u {9, 3, 3, 10}); EXPECT_TRUE(is_near(cltsa4, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa4(CovSA2l {4, 2, 2, 5}); EXPECT_TRUE(is_near(cutsa4, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa4s(CovSA2l {9, 3, 3, 10}); EXPECT_TRUE(is_near(cltsa4s, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa4s(CovSA2u {4, 2, 2, 5}); EXPECT_TRUE(is_near(cutsa4s, Mat2 {4, 2, 2, 5}));
  CovSA2l clsa4d(CovD2 {1, 2}); EXPECT_TRUE(is_near(clsa4d, Mat2 {1, 0, 0, 2}));
  CovSA2u cusa4d(CovD2 {1, 2}); EXPECT_TRUE(is_near(cusa4d, Mat2 {1, 0, 0, 2}));
  CovT2l clt4d(CovD2 {1, 2}); EXPECT_TRUE(is_near(clt4d, Mat2 {1, 0, 0, 2}));
  CovT2u cut4d(CovD2 {1, 2}); EXPECT_TRUE(is_near(cut4d, Mat2 {1, 0, 0, 2}));
  CovT2u cut4i(CovI2 {i2}); EXPECT_TRUE(is_near(cut4i, Mat2 {1, 0, 0, 1}));

  /*// Convert from different square-root covariance type
  CovSA2l clsasa4X(SqCovSA2u {3, 1, 0, 3}); EXPECT_TRUE(is_near(clsasa4X, Mat2 {9, 3, 3, 10}));
  CovSA2u cusasa4X(SqCovSA2l {2, 0, 1, 2}); EXPECT_TRUE(is_near(cusasa4X, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat4X(SqCovT2u {3, 1, 0, 3}); EXPECT_TRUE(is_near(clsat4X, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat4X(SqCovT2l {2, 0, 1, 2}); EXPECT_TRUE(is_near(cusat4X, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat4sX(SqCovT2l {3, 0, 1, 3}); EXPECT_TRUE(is_near(clsat4sX, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat4sX(SqCovT2u {2, 1, 0, 2}); EXPECT_TRUE(is_near(cusat4sX, Mat2 {4, 2, 2, 5}));
  CovT2l cltt4X(SqCovT2u {3, 1, 0, 3}); EXPECT_TRUE(is_near(cltt4X, Mat2 {9, 3, 3, 10}));
  CovT2u cutt4X(SqCovT2l {2, 0, 1, 2}); EXPECT_TRUE(is_near(cutt4X, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa4X(SqCovSA2u {3, 1, 0, 3}); EXPECT_TRUE(is_near(cltsa4X, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa4X(SqCovSA2l {2, 0, 1, 2}); EXPECT_TRUE(is_near(cutsa4X, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa4sX(SqCovSA2l {3, 0, 1, 3}); EXPECT_TRUE(is_near(cltsa4sX, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa4sX(SqCovSA2u {2, 1, 0, 2}); EXPECT_TRUE(is_near(cutsa4sX, Mat2 {4, 2, 2, 5}));
  CovSA2l clsa4dX(SqCovD2 {2, 3}); EXPECT_TRUE(is_near(clsa4dX, Mat2 {4, 0, 0, 9}));
  CovSA2u cusa4dX(SqCovD2 {2, 3}); EXPECT_TRUE(is_near(cusa4dX, Mat2 {4, 0, 0, 9}));
  CovT2l clt4dX(SqCovD2 {2, 3}); EXPECT_TRUE(is_near(clt4dX, Mat2 {4, 0, 0, 9}));
  CovT2u cut4dX(SqCovD2 {2, 3}); EXPECT_TRUE(is_near(cut4dX, Mat2 {4, 0, 0, 9}));
  CovT2u cut4iX(SqCovI2 {i2}); EXPECT_TRUE(is_near(cut4iX, Mat2 {1, 0, 0, 1}));
  CovD2 cd4d2X(SqCovD2 {2, 3}); EXPECT_TRUE(is_near(cd4d2X, D2 {4, 9}));
  CovD1 cd4d1X(SqCovD1 {2}); EXPECT_TRUE(is_near(cd4d1X, D1 {4}));
  */

  // Construct from a covariance_nestable
  CovSA2l clsasa5(SA2u {9, 3, 3, 10}); EXPECT_TRUE(is_near(clsasa5, Mat2 {9, 3, 3, 10}));
  CovSA2u cusasa5(SA2l {4, 2, 2, 5}); EXPECT_TRUE(is_near(cusasa5, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat5(T2u {3, 1, 0, 3}); EXPECT_TRUE(is_near(clsat5, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat5(T2l {2, 0, 1, 2}); EXPECT_TRUE(is_near(cusat5, Mat2 {4, 2, 2, 5}));
  CovSA2l clsat5s(T2l {3, 0, 1, 3}); EXPECT_TRUE(is_near(clsat5s, Mat2 {9, 3, 3, 10}));
  CovSA2u cusat5s(T2u {2, 1, 0, 2}); EXPECT_TRUE(is_near(cusat5s, Mat2 {4, 2, 2, 5}));
  CovT2l cltt5(T2u {3, 1, 0, 3}); EXPECT_TRUE(is_near(cltt5, Mat2 {9, 3, 3, 10}));
  CovT2u cutt5(T2l {2, 0, 1, 2}); EXPECT_TRUE(is_near(cutt5, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa5(SA2u {9, 3, 3, 10}); EXPECT_TRUE(is_near(cltsa5, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa5(SA2l {4, 2, 2, 5}); EXPECT_TRUE(is_near(cutsa5, Mat2 {4, 2, 2, 5}));
  CovT2l cltsa5s(SA2l {9, 3, 3, 10}); EXPECT_TRUE(is_near(cltsa5s, Mat2 {9, 3, 3, 10}));
  CovT2u cutsa5s(SA2u {4, 2, 2, 5}); EXPECT_TRUE(is_near(cutsa5s, Mat2 {4, 2, 2, 5}));
  CovSA2l clsa5d(D2 {1, 2}); EXPECT_TRUE(is_near(clsa5d, Mat2 {1, 0, 0, 2}));
  CovSA2u cusa5d(D2 {1, 2}); EXPECT_TRUE(is_near(cusa5d, Mat2 {1, 0, 0, 2}));
  CovT2l clt5d(D2 {1, 2}); EXPECT_TRUE(is_near(clt5d, Mat2 {1, 0, 0, 2}));
  CovT2u cut5d(D2 {1, 2}); EXPECT_TRUE(is_near(cut5d, Mat2 {1, 0, 0, 2}));
  CovT2l clt5i(covi2); EXPECT_TRUE(is_near(clt5i, Mat2 {1, 0, 0, 1}));

  // Construct from a typed matrix
  CovSA2l clsa6(Mat2 {9, 7, 3, 10}); EXPECT_TRUE(is_near(clsa6, Mat2 {9, 3, 3, 10}));
  CovSA2u cusa6(Mat2 {4, 2, 7, 5}); EXPECT_TRUE(is_near(cusa6, Mat2 {4, 2, 2, 5}));
  CovT2l clt6(Mat2 {9, 7, 3, 10}); EXPECT_TRUE(is_near(clt6, Mat2 {9, 3, 3, 10}));
  CovT2u cut6(Mat2 {4, 2, 7, 5}); EXPECT_TRUE(is_near(cut6, Mat2 {4, 2, 2, 5}));
  CovD2 cd6(Matrix<C, Axis, eigen_matrix_t<double, 2, 1>> {1, 2}); EXPECT_TRUE(is_near(cd6, Mat2 {1, 0, 0, 2}));

  // Construct from a regular matrix
  CovSA2l clsa7(make_dense_object_from<M2>(9, 7, 3, 10)); EXPECT_TRUE(is_near(clsa7, Mat2 {9, 3, 3, 10}));
  CovSA2u cusa7(make_dense_object_from<M2>(4, 2, 7, 5)); EXPECT_TRUE(is_near(cusa7, Mat2 {4, 2, 2, 5}));
  CovT2l clt7(make_dense_object_from<M2>(9, 7, 3, 10)); EXPECT_TRUE(is_near(clt7, Mat2 {9, 3, 3, 10}));
  CovT2u cut7(make_dense_object_from<M2>(4, 2, 7, 5)); EXPECT_TRUE(is_near(cut7, Mat2 {4, 2, 2, 5}));
  CovD2 cd7(make_eigen_matrix<double, 2, 1>(1, 2)); EXPECT_TRUE(is_near(cd7, Mat2 {1, 0, 0, 2}));

  // Construct from a list of coefficients
  CovSA2l clsa8{4, 7, 2, 5}; EXPECT_TRUE(is_near(clsa8, Mat2 {4, 2, 2, 5}));
  CovSA2u cusa8{9, 3, 7, 10}; EXPECT_TRUE(is_near(cusa8, Mat2 {9, 3, 3, 10}));
  CovT2l clt8{4, 7, 2, 5}; EXPECT_TRUE(is_near(clt8, Mat2 {4, 2, 2, 5}));
  CovT2u cut8{9, 3, 7, 10}; EXPECT_TRUE(is_near(cut8, Mat2 {9, 3, 3, 10}));
  CovD2 cd8({1, 2}); EXPECT_TRUE(is_near(cd8, Mat2 {1, 0, 0, 2}));

  // Copy assignment
  clsa2 = clsa8; EXPECT_TRUE(is_near(clsa2, Mat2 {4, 2, 2, 5}));
  cusa2 = cusa8; EXPECT_TRUE(is_near(cusa2, Mat2 {9, 3, 3, 10}));
  clt2 = clt8; EXPECT_TRUE(is_near(clt2, Mat2 {4, 2, 2, 5}));
  cut2 = cut8; EXPECT_TRUE(is_near(cut2, Mat2 {9, 3, 3, 10}));
  cd2 = cd8; EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // Move assignment
  auto ya = CovSA2l {9, 3, 3, 10}; clsa2 = std::move(ya); EXPECT_TRUE(is_near(clsa2, Mat2 {9, 3, 3, 10}));
  auto yb = CovSA2u {4, 2, 2, 5}; cusa2 = std::move(yb); EXPECT_TRUE(is_near(cusa2, Mat2 {4, 2, 2, 5}));
  auto yc = CovT2l {9, 3, 3, 10}; clt2 = std::move(yc); EXPECT_TRUE(is_near(clt2, Mat2 {9, 3, 3, 10}));
  auto yd = CovT2u {4, 2, 2, 5}; cut2 = std::move(yd); EXPECT_TRUE(is_near(cut2, Mat2 {4, 2, 2, 5}));
  auto ye = CovD2 {1, 2}; cd2 = std::move(ye); EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // Assign from different covariance type
  clsasa4 = CovSA2u {4, 2, 2, 5}; EXPECT_TRUE(is_near(clsasa4, Mat2 {4, 2, 2, 5}));
  cusasa4 = CovSA2l {9, 3, 3, 10}; EXPECT_TRUE(is_near(cusasa4, Mat2 {9, 3, 3, 10}));
  clsat4 = CovT2u {4, 2, 2, 5}; EXPECT_TRUE(is_near(clsat4, Mat2 {4, 2, 2, 5}));
  cusat4 = CovT2l {9, 3, 3, 10}; EXPECT_TRUE(is_near(cusat4, Mat2 {9, 3, 3, 10}));
  clsat4s = CovT2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(clsat4s, Mat2 {4, 2, 2, 5}));
  cusat4s = CovT2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(cusat4s, Mat2 {9, 3, 3, 10}));
  cltt4 = CovT2u {4, 2, 2, 5}; EXPECT_TRUE(is_near(cltt4, Mat2 {4, 2, 2, 5}));
  cutt4 = CovT2l {9, 3, 3, 10}; EXPECT_TRUE(is_near(cutt4, Mat2 {9, 3, 3, 10}));
  cltsa4 = CovSA2u {4, 2, 2, 5}; EXPECT_TRUE(is_near(cltsa4, Mat2 {4, 2, 2, 5}));
  cutsa4 = CovSA2l {9, 3, 3, 10}; EXPECT_TRUE(is_near(cutsa4, Mat2 {9, 3, 3, 10}));
  cltsa4s = CovSA2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cltsa4s, Mat2 {4, 2, 2, 5}));
  cutsa4s = CovSA2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(cutsa4s, Mat2 {9, 3, 3, 10}));
  clsa4d = CovD2 {3, 4}; EXPECT_TRUE(is_near(clsa4d, Mat2 {3, 0, 0, 4}));
  cusa4d = CovD2 {3, 4}; EXPECT_TRUE(is_near(cusa4d, Mat2 {3, 0, 0, 4}));
  clt4d = CovD2 {3, 4}; EXPECT_TRUE(is_near(clt4d, Mat2 {3, 0, 0, 4}));
  cut4d = CovD2 {3, 4}; EXPECT_TRUE(is_near(cut4d, Mat2 {3, 0, 0, 4}));
  cut4i = CovI2 {i2}; EXPECT_TRUE(is_near(cut4i, Mat2 {1, 0, 0, 1}));

  /*// Assign from different square-root covariance type
  clsasa5 = SqCovSA2u {2, 1, 0, 2}; EXPECT_TRUE(is_near(clsasa5, Mat2 {4, 2, 2, 5}));
  cusasa5 = SqCovSA2l {3, 0, 1, 3}; EXPECT_TRUE(is_near(cusasa5, Mat2 {9, 3, 3, 10}));
  clsat5 = SqCovT2u {2, 1, 0, 2}; EXPECT_TRUE(is_near(clsat5, Mat2 {4, 2, 2, 5}));
  cusat5 = SqCovT2l {3, 0, 1, 3}; EXPECT_TRUE(is_near(cusat5, Mat2 {9, 3, 3, 10}));
  clsat5s = SqCovT2l {2, 0, 1, 2}; EXPECT_TRUE(is_near(clsat5s, Mat2 {4, 2, 2, 5}));
  cusat5s = SqCovT2u {3, 1, 0, 3}; EXPECT_TRUE(is_near(cusat5s, Mat2 {9, 3, 3, 10}));
  cltsa5 = SqCovSA2u {2, 1, 0, 2}; EXPECT_TRUE(is_near(cltsa5, Mat2 {4, 2, 2, 5}));
  cutsa5 = SqCovSA2l {3, 0, 1, 3}; EXPECT_TRUE(is_near(cutsa5, Mat2 {9, 3, 3, 10}));
  cltsa5s = SqCovSA2l {2, 0, 1, 2}; EXPECT_TRUE(is_near(cltsa5s, Mat2 {4, 2, 2, 5}));
  cutsa5s = SqCovSA2u {3, 1, 0, 3}; EXPECT_TRUE(is_near(cutsa5s, Mat2 {9, 3, 3, 10}));
  clsa5d = SqCovD2 {3, 4}; EXPECT_TRUE(is_near(clsa5d, Mat2 {9, 0, 0, 16}));
  cusa5d = SqCovD2 {3, 4}; EXPECT_TRUE(is_near(cusa5d, Mat2 {9, 0, 0, 16}));
  clt5d = SqCovD2 {3, 4}; EXPECT_TRUE(is_near(clt5d, Mat2 {9, 0, 0, 16}));
  cut5d = SqCovD2 {3, 4}; EXPECT_TRUE(is_near(cut5d, Mat2 {9, 0, 0, 16}));
  clt5i = SqCovI2 {i2}; EXPECT_TRUE(is_near(clt5i, Mat2 {1, 0, 0, 1}));
  cd4d2X = SqCovD2 {3, 4}; EXPECT_TRUE(is_near(cd4d2X, D2 {9, 16}));
  cd4d1X = SqCovD1 {3}; EXPECT_TRUE(is_near(cd4d1X, D1 {9}));
  */

  // Assign from a list of coefficients (via move assignment operator)
  clsa8 = {9, 3, 3, 10}; EXPECT_TRUE(is_near(clsa8, Mat2 {9, 3, 3, 10}));
  cusa8 = {4, 2, 2, 5}; EXPECT_TRUE(is_near(cusa8, Mat2 {4, 2, 2, 5}));
  clt8 = {9, 3, 3, 10}; EXPECT_TRUE(is_near(clt8, Mat2 {9, 3, 3, 10}));
  cut8 = {4, 2, 2, 5}; EXPECT_TRUE(is_near(cut8, Mat2 {4, 2, 2, 5}));
  cd8 = {3, 4}; EXPECT_TRUE(is_near(cd8, Mat2 {3, 0, 0, 4}));

  // Increment
  clsasa4 += {9, 3, 3, 10}; clsasa4 -= {9, 3, 3, 10};
  clsasa4 += CovSA2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(clsasa4, Mat2 {13, 5, 5, 15}));
  cusasa4 += CovSA2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cusasa4, Mat2 {13, 5, 5, 15}));
  clsat4 += CovT2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(clsat4, Mat2 {13, 5, 5, 15}));
  cusat4 += CovT2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cusat4, Mat2 {13, 5, 5, 15}));
  clsat4s += CovT2l {9, 3, 3, 10}; EXPECT_TRUE(is_near(clsat4s, Mat2 {13, 5, 5, 15}));
  cusat4s += CovT2u {4, 2, 2, 5}; EXPECT_TRUE(is_near(cusat4s, Mat2 {13, 5, 5, 15}));
  cltt4 += CovT2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(cltt4, Mat2 {13, 5, 5, 15}));
  cutt4 += CovT2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cutt4, Mat2 {13, 5, 5, 15}));
  cltsa4 += CovSA2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(cltsa4, Mat2 {13, 5, 5, 15}));
  cutsa4 += CovSA2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cutsa4, Mat2 {13, 5, 5, 15}));
  cltsa4s += CovSA2l {9, 3, 3, 10}; EXPECT_TRUE(is_near(cltsa4s, Mat2 {13, 5, 5, 15}));
  cutsa4s += CovSA2u {4, 2, 2, 5}; EXPECT_TRUE(is_near(cutsa4s, Mat2 {13, 5, 5, 15}));
  cd8 += CovD2 {1, 2}; EXPECT_TRUE(is_near(cd8, Mat2 {4, 0, 0, 6}));

  // Decrement
  clsasa4 -= CovSA2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(clsasa4, Mat2 {4, 2, 2, 5}));
  cusasa4 -= CovSA2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cusasa4, Mat2 {9, 3, 3, 10}));
  clsat4 -= CovT2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(clsat4, Mat2 {4, 2, 2, 5}));
  cusat4 -= CovT2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cusat4, Mat2 {9, 3, 3, 10}));
  clsat4s -= CovT2l {9, 3, 3, 10}; EXPECT_TRUE(is_near(clsat4s, Mat2 {4, 2, 2, 5}));
  cusat4s -= CovT2u {4, 2, 2, 5}; EXPECT_TRUE(is_near(cusat4s, Mat2 {9, 3, 3, 10}));
  cltt4 -= CovT2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(cltt4, Mat2 {4, 2, 2, 5}));
  cutt4 -= CovT2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cutt4, Mat2 {9, 3, 3, 10}));
  cltsa4 -= CovSA2u {9, 3, 3, 10}; EXPECT_TRUE(is_near(cltsa4, Mat2 {4, 2, 2, 5}));
  cutsa4 -= CovSA2l {4, 2, 2, 5}; EXPECT_TRUE(is_near(cutsa4, Mat2 {9, 3, 3, 10}));
  cltsa4s -= CovSA2l {9, 3, 3, 10}; EXPECT_TRUE(is_near(cltsa4s, Mat2 {4, 2, 2, 5}));
  cutsa4s -= CovSA2u {4, 2, 2, 5}; EXPECT_TRUE(is_near(cutsa4s, Mat2 {9, 3, 3, 10}));
  cd8 -= CovD2 {1, 2}; EXPECT_TRUE(is_near(cd8, Mat2 {3, 0, 0, 4}));

  // Scalar multiplication, positive
  clsa2 *= 2; EXPECT_TRUE(is_near(clsa2, Mat2 {18, 6, 6, 20}));
  cusa2 *= 2; EXPECT_TRUE(is_near(cusa2, Mat2 {8, 4, 4, 10}));
  clt2 *= 2; EXPECT_TRUE(is_near(clt2, Mat2 {18, 6, 6, 20}));
  cut2 *= 2; EXPECT_TRUE(is_near(cut2, Mat2 {8, 4, 4, 10}));
  cd2 *= 2; EXPECT_TRUE(is_near(cd2, Mat2 {2, 0, 0, 4}));

  // Scalar division, negative
  clsa2 /= -2; EXPECT_TRUE(is_near(clsa2, Mat2 {-9, -3, -3, -10}));
  cusa2 /= -2; EXPECT_TRUE(is_near(cusa2, Mat2 {-4, -2, -2, -5}));
  // These results would result in a runtime error.
  //clt2 /= -2; //EXPECT_TRUE(is_near(clt2, Mat2 {-9, -3, -3, -10}));
  //cut2 /= -2; //EXPECT_TRUE(is_near(cut2, Mat2 {-4, -2, -2, -5}));
  cd2 /= -2; EXPECT_TRUE(is_near(cd2, Mat2 {-1, 0, 0, -2}));

  // Scalar multiplication, negative
  clsa2 *= -2; EXPECT_TRUE(is_near(clsa2, Mat2 {18, 6, 6, 20}));
  cusa2 *= -2; EXPECT_TRUE(is_near(cusa2, Mat2 {8, 4, 4, 10}));
  //clt2 *= -2; //EXPECT_TRUE(is_near(clt2, Mat2 {18, 6, 6, 20}));
  //cut2 *= -2; //EXPECT_TRUE(is_near(cut2, Mat2 {8, 4, 4, 10}));
  cd2 *= -2; EXPECT_TRUE(is_near(cd2, Mat2 {2, 0, 0, 4}));

  // Scalar division, positive
  clsa2 /= 2; EXPECT_TRUE(is_near(clsa2, Mat2 {9, 3, 3, 10}));
  cusa2 /= 2; EXPECT_TRUE(is_near(cusa2, Mat2 {4, 2, 2, 5}));
  clt2 /= 2; EXPECT_TRUE(is_near(clt2, Mat2 {9, 3, 3, 10}));
  cut2 /= 2; EXPECT_TRUE(is_near(cut2, Mat2 {4, 2, 2, 5}));
  cd2 /= 2; EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // .scale()
  clsa2.scale(2); EXPECT_TRUE(is_near(clsa2, Mat2 {36, 12, 12, 40}));
  cusa2.scale(2); EXPECT_TRUE(is_near(cusa2, Mat2 {16, 8, 8, 20}));
  clt2.scale(2); EXPECT_TRUE(is_near(clt2, Mat2 {36, 12, 12, 40}));
  cut2.scale(2); EXPECT_TRUE(is_near(cut2, Mat2 {16, 8, 8, 20}));
  cd2.scale(2); EXPECT_TRUE(is_near(cd2, Mat2 {4, 0, 0, 8}));

  // .inverse_scale()
  clsa2.inverse_scale(2); EXPECT_TRUE(is_near(clsa2, Mat2 {9, 3, 3, 10}));
  cusa2.inverse_scale(2); EXPECT_TRUE(is_near(cusa2, Mat2 {4, 2, 2, 5}));
  clt2.inverse_scale(2); EXPECT_TRUE(is_near(clt2, Mat2 {9, 3, 3, 10}));
  cut2.inverse_scale(2); EXPECT_TRUE(is_near(cut2, Mat2 {4, 2, 2, 5}));
  cd2.inverse_scale(2); EXPECT_TRUE(is_near(cd2, Mat2 {1, 0, 0, 2}));

  // Scalar multiplication, zero
  clsa2 *= 0; EXPECT_TRUE(is_near(clsa2, make_zero<Mat2>()));
  cusa2 *= 0; EXPECT_TRUE(is_near(cusa2, make_zero<Mat2>()));
  clt2 *= 0; EXPECT_TRUE(is_near(clt2, make_zero<Mat2>()));
  cut2 *= 0; EXPECT_TRUE(is_near(cut2, make_zero<Mat2>()));
  cd2 *= 0; EXPECT_TRUE(is_near(cd2, make_zero<Mat2>()));

  // Zero
  EXPECT_TRUE(is_near(make_zero<CovSA2l>(), M2::Zero()));
  EXPECT_TRUE(is_near(make_zero<CovSA2u>(), M2::Zero()));
  EXPECT_TRUE(is_near(make_zero<CovT2l>(), M2::Zero()));
  EXPECT_TRUE(is_near(make_zero<CovT2u>(), M2::Zero()));
  EXPECT_TRUE(is_near(make_zero<CovD2>(), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(make_zero<CovSA2l>()), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(make_zero<CovSA2u>()), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(make_zero<CovT2l>()), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(make_zero<CovT2u>()), M2::Zero()));
  EXPECT_TRUE(is_near(square_root(make_zero<CovD2>()), M2::Zero()));

  // Identity
  EXPECT_TRUE(is_near(make_identity_matrix_like<CovSA2l>(), M2::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<CovSA2u>(), M2::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<CovT2l>(), M2::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<CovT2u>(), M2::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<CovD2>(), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(make_identity_matrix_like<CovSA2l>()), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(make_identity_matrix_like<CovSA2u>()), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(make_identity_matrix_like<CovT2l>()), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(make_identity_matrix_like<CovT2u>()), M2::Identity()));
  EXPECT_TRUE(is_near(square_root(make_identity_matrix_like<CovD2>()), M2::Identity()));
}

TEST(covariance_tests, Covariance_subscripts)
{
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
  EXPECT_NEAR((CovD2 {9, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((CovD2 {9, 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((CovD2 {9, 10})(0), 9, 1e-6);
  EXPECT_NEAR((CovD2 {9, 10})(1), 10, 1e-6);

  EXPECT_NEAR(get_component(CovSA2l {9, 3, 3, 10}, 0, 0), 9, 1e-6);
  EXPECT_NEAR(get_component(CovSA2u {9, 3, 3, 10}, 0, 1), 3, 1e-6);
  EXPECT_NEAR(get_component(CovT2l {9, 3, 3, 10}, 1, 0), 3, 1e-6);
  EXPECT_NEAR(get_component(CovT2u {9, 3, 3, 10}, 1, 1), 10, 1e-6);
  EXPECT_NEAR(get_component(CovD2 {9, 10}, 0, 0), 9, 1e-6);
  EXPECT_NEAR(get_component(CovD2 {9, 10}, 1, 1), 10, 1e-6);
  EXPECT_NEAR(get_component(CovD2 {9, 10}, 0), 9, 1e-6);
  EXPECT_NEAR(get_component(CovD2 {9, 10}, 1), 10, 1e-6);

  auto sa2l = CovSA2l {9, 3, 3, 10};
  sa2l(0, 0) = 9.1; EXPECT_NEAR(get_component(sa2l, 0, 0), 9.1, 1e-6);
  set_component(sa2l, 9.2, 0, 0); EXPECT_NEAR(sa2l(0, 0), 9.2, 1e-6);
  auto sa2u = CovSA2u {9, 3, 3, 10};
  sa2u(0, 1) = 3.1; EXPECT_NEAR(get_component(sa2u, 0, 1), 3.1, 1e-6);
  set_component(sa2u, 3.2, 0, 1); EXPECT_NEAR(sa2u(0, 1), 3.2, 1e-6);
  auto t2l = CovT2l {9, 3, 3, 10};
  t2l(1, 0) = 3.1; EXPECT_NEAR(get_component(t2l, 1, 0), 3.1, 1e-6);
  set_component(t2l, 3.2, 1, 0); EXPECT_NEAR(t2l(1, 0), 3.2, 1e-6);
  auto t2u = CovT2u {9, 3, 3, 10};
  t2u(1, 1) = 10.1; EXPECT_NEAR(get_component(t2u, 1, 1), 10.1, 1e-6);
  set_component(t2u, 10.2, 1, 1); EXPECT_NEAR(t2u(1, 1), 10.2, 1e-6);
  auto d2 = CovD2 {9, 10};
  d2(0, 0) = 9.1; EXPECT_NEAR(get_component(d2, 0, 0), 9.1, 1e-6);
  set_component(d2, 9.2, 0, 0); EXPECT_NEAR(d2(0, 0), 9.2, 1e-6);
  d2(1) = 10.1; EXPECT_NEAR(get_component(d2, 1), 10.1, 1e-6);
  set_component(d2, 10.2, 1); EXPECT_NEAR(d2(1), 10.2, 1e-6);
}


TEST(covariance_tests, Covariance_deduction_guides)
{
  EXPECT_TRUE(is_near(Covariance(CovSA2l {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Covariance(CovSA2l {9, 3, 3, 10})), 0>, C>);

  EXPECT_TRUE(is_near(Covariance(T2l {3, 0, 1, 3}), Mat2 {9, 3, 3, 10}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Covariance(T2l {3, 0, 1, 3})), 0>, Dimensions<2>>);

  EXPECT_TRUE(is_near(Covariance(D2 {1, 2}), Mat2 {1, 0, 0, 2}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Covariance(D2 {1, 2})), 0>, Dimensions<2>>);

  EXPECT_TRUE(is_near(Covariance(Mat2 {9, 3, 3, 10}), Mat2 {9, 3, 3, 10}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Covariance(Mat2 {9, 3, 3, 10})), 0>, C>);

  EXPECT_TRUE(is_near(Covariance(make_dense_object_from<M2>(9, 3, 3, 10)), Mat2 {9, 3, 3, 10}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Covariance(make_dense_object_from<M2>(9, 3, 3, 10))), 0>, Dimensions<2>>);

  EXPECT_TRUE(is_near(Covariance {9., 3, 3, 10}, Mat2 {9, 3, 3, 10}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Covariance {9., 3, 3, 10}), 0>, Dimensions<2>>);
}


TEST(covariance_tests, Covariance_make)
{
  // Other covariance:
  EXPECT_TRUE(is_near(make_covariance(CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_covariance(CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_covariance(CovT2l {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_covariance(CovT2u {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_covariance(CovD2 {4, 9}).get_self_adjoint_nested_matrix(), Mat2 {4, 0, 0, 9}));
  EXPECT_TRUE(is_near(make_covariance(CovD2 {4, 9}).get_triangular_nested_matrix(), Mat2 {4, 0, 0, 9}));

  EXPECT_TRUE(is_near(make_covariance(SqCovSA2l {3, 0, 1, 3}.square()).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_covariance(SqCovSA2u {3, 1, 0, 3}.square()).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_covariance(SqCovT2l {3, 0, 1, 3}.square()).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_covariance(SqCovT2u {3, 1, 0, 3}.square()).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_covariance(SqCovD2 {2, 3}.square()).get_self_adjoint_nested_matrix(), Mat2 {4, 0, 0, 9}));
  EXPECT_TRUE(is_near(make_covariance(SqCovD2 {2, 3}.square()).get_triangular_nested_matrix(), Mat2 {4, 0, 0, 9}));

  static_assert(hermitian_adapter<decltype(make_covariance(CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance(CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance(CovT2l {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance(CovT2u {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(make_covariance(CovD2 {1, 2}).get_self_adjoint_nested_matrix())>);

  static_assert(hermitian_adapter<decltype(make_covariance(adjoint(CovSA2l {9, 3, 3, 10})).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance(adjoint(CovSA2u {9, 3, 3, 10})).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance(adjoint(CovT2l {9, 3, 3, 10})).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance(adjoint(CovT2u {9, 3, 3, 10})).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(diagonal_matrix<decltype(make_covariance(adjoint(CovD2 {1, 2})).get_self_adjoint_nested_matrix())>);

  static_assert(hermitian_adapter<decltype(make_covariance<CovSA2l>().get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance<CovSA2u>().get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance<CovT2l>().get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<CovT2u>().get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(make_covariance<CovD2>().get_self_adjoint_nested_matrix())>);

  static_assert(hermitian_adapter<decltype(make_covariance<CovSA2l>().get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance<CovSA2u>().get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(diagonal_matrix<decltype(make_covariance<CovD2>().get_self_adjoint_nested_matrix())>);

  // Covariance bases:
  EXPECT_TRUE(is_near(make_covariance<C>(SA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_covariance<C>(SA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_covariance<C>(T2l {3, 0, 1, 3}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_covariance<C>(T2u {3, 1, 0, 3}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_covariance<C>(D2 {1, 2}).get_self_adjoint_nested_matrix(), Mat2 {1, 0, 0, 2}));

  static_assert(hermitian_adapter<decltype(make_covariance<C>(SA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance<C>(SA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance<C>(T2l {3, 0, 1, 3}).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<C>(T2u {3, 1, 0, 3}).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(make_covariance<C>(D2 {1, 2}).get_self_adjoint_nested_matrix())>);

  static_assert(hermitian_adapter<decltype(adjoint(make_covariance<C>(SA2l {9, 3, 3, 10})).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance<C>(adjoint(SA2u {9, 3, 3, 10})).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance<C>(adjoint(T2l {3, 0, 1, 3})).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance<C>(adjoint(T2u {3, 1, 0, 3})).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(diagonal_matrix<decltype(make_covariance<C>(adjoint(D2 {1, 2})).get_self_adjoint_nested_matrix())>);

  static_assert(hermitian_adapter<decltype(make_covariance<C, SA2l>().get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance<C, SA2u>().get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance<C, T2l>().get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<C, T2u>().get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance<C, D2>().get_triangular_nested_matrix()), TriangleType::upper>);

  static_assert(hermitian_adapter<decltype(make_covariance(SA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance<C>(SA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance<T2u>().get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(make_covariance<C, T2l>().get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(diagonal_matrix<decltype(make_covariance<C, D2>().get_self_adjoint_nested_matrix())>);

  // Regular matrices:
  EXPECT_TRUE(is_near(make_covariance<C, TriangleType::lower>(Mat2 {9, 3, 3, 10}.nested_object()).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_covariance<C, TriangleType::upper>(Mat2 {9, 3, 3, 10}.nested_object()).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_covariance<C>(Mat2 {9, 3, 3, 10}.nested_object()).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));

  static_assert(triangular_matrix<decltype(make_covariance<C, TriangleType::lower>(Mat2 {9, 3, 3, 10}.nested_object()).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<C, TriangleType::upper>(Mat2 {9, 3, 3, 10}.nested_object()).get_triangular_nested_matrix()), TriangleType::upper>);

  static_assert(triangular_matrix<decltype(make_covariance<C, TriangleType::lower, M2>().get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<C, TriangleType::upper, M2>().get_triangular_nested_matrix()), TriangleType::upper>);

  static_assert(hermitian_adapter<decltype(make_covariance<C>(Mat2 {9, 3, 3, 10}.nested_object()).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<TriangleType::upper>(Mat2 {9, 3, 3, 10}.nested_object()).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(hermitian_adapter<decltype(make_covariance<C, M2>().get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<TriangleType::upper, M2>().get_triangular_nested_matrix()), TriangleType::upper>);

  // Typed matrices:
  EXPECT_TRUE(is_near(make_covariance<TriangleType::lower>(Mat2 {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_covariance<TriangleType::upper>(Mat2 {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_covariance(Mat2 {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));

  static_assert(triangular_matrix<decltype(make_covariance<TriangleType::lower>(Mat2 {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<TriangleType::upper>(Mat2 {9, 3, 3, 10}).get_triangular_nested_matrix()), TriangleType::upper>);

  static_assert(triangular_matrix<decltype(make_covariance<TriangleType::lower, Mat2>().get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<TriangleType::upper, Mat2>().get_triangular_nested_matrix()), TriangleType::upper>);

  static_assert(hermitian_adapter<decltype(make_covariance(Mat2 {9, 3, 3, 10}).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_covariance<Mat2>().get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);

  // Eigen defaults
  EXPECT_TRUE(is_near(make_covariance<C, TriangleType::lower>(9., 3, 3, 10).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(make_covariance<C, TriangleType::upper>(9., 3, 3, 10).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(make_covariance<C>(9., 3, 3, 10).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));

  static_assert(triangular_matrix<decltype(make_covariance<C, TriangleType::lower>(9., 3, 3, 10).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<C, TriangleType::upper>(9., 3, 3, 10).get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(hermitian_adapter<decltype(make_covariance<C>(9., 3, 3, 10).get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);

  static_assert(triangular_matrix<decltype(make_covariance<TriangleType::lower>(9., 3, 3, 10).get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<C, TriangleType::lower>().get_triangular_nested_matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(make_covariance<C, TriangleType::upper>().get_triangular_nested_matrix()), TriangleType::upper>);
  static_assert(hermitian_adapter<decltype(make_covariance<C>().get_self_adjoint_nested_matrix()), HermitianAdapterType::lower>);
  static_assert(coordinates::dimension_of_v<vector_space_descriptor_of_t<decltype(make_covariance<C>()), 0>> == 2);
}


TEST(covariance_tests, Covariance_traits)
{
  static_assert(self_adjoint_covariance<CovSA2l>);
  static_assert(not triangular_covariance<CovSA2l>);
  static_assert(not diagonal_matrix<CovSA2l>);
  static_assert(hermitian_matrix<CovSA2l>);
  static_assert(not cholesky_form<CovSA2l>);
  static_assert(not triangular_matrix<CovSA2l>);
  static_assert(not triangular_matrix<CovSA2l, TriangleType::lower>);
  static_assert(not triangular_matrix<CovSA2l, TriangleType::upper>);
  static_assert(not identity_matrix<CovSA2l>);
  static_assert(not zero<CovSA2l>);

  static_assert(self_adjoint_covariance<CovT2l>);
  static_assert(not triangular_covariance<CovT2l>);
  static_assert(not diagonal_matrix<CovT2l>);
  static_assert(hermitian_matrix<CovT2l>);
  static_assert(cholesky_form<CovT2l>);
  static_assert(not triangular_matrix<CovT2l>);
  static_assert(not triangular_matrix<CovT2l, TriangleType::lower>);
  static_assert(not triangular_matrix<CovT2l, TriangleType::upper>);
  static_assert(not triangular_matrix<CovT2u, TriangleType::upper>);
  static_assert(not identity_matrix<CovT2l>);
  static_assert(not zero<CovT2l>);

  static_assert(self_adjoint_covariance<CovD2>);
  static_assert(not triangular_covariance<CovD2>);
  static_assert(diagonal_matrix<CovD2>);
  static_assert(hermitian_matrix<CovD2>);
  static_assert(not cholesky_form<CovD2>);
  static_assert(triangular_matrix<CovD2>);
  static_assert(triangular_matrix<CovD2, TriangleType::lower>);
  static_assert(triangular_matrix<CovD2, TriangleType::upper>);
  static_assert(triangular_matrix<CovD2, TriangleType::upper>);
  static_assert(not identity_matrix<CovD2>);
  static_assert(not zero<CovD2>);

  static_assert(self_adjoint_covariance<CovI2>);
  static_assert(not triangular_covariance<CovI2>);
  static_assert(diagonal_matrix<CovI2>);
  static_assert(hermitian_matrix<CovI2>);
  static_assert(not cholesky_form<CovI2>);
  static_assert(triangular_matrix<CovI2>);
  static_assert(triangular_matrix<CovI2, TriangleType::lower>);
  static_assert(triangular_matrix<CovI2, TriangleType::upper>);
  static_assert(triangular_matrix<CovI2, TriangleType::upper>);
  static_assert(identity_matrix<CovI2>);
  static_assert(not zero<CovI2>);

  static_assert(self_adjoint_covariance<CovZ2>);
  static_assert(not triangular_covariance<CovZ2>);
  static_assert(diagonal_matrix<CovZ2>);
  static_assert(hermitian_matrix<CovZ2>);
  static_assert(not cholesky_form<CovZ2>);
  static_assert(triangular_matrix<CovZ2>);
  static_assert(triangular_matrix<CovZ2, TriangleType::lower>);
  static_assert(triangular_matrix<CovZ2, TriangleType::upper>);
  static_assert(triangular_matrix<CovZ2, TriangleType::upper>);
  static_assert(not identity_matrix<CovZ2>);
  static_assert(zero<CovZ2>);

  EXPECT_TRUE(is_near(make_zero<CovSA2l>(), eigen_matrix_t<double, 2, 2>::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<CovSA2l>(), eigen_matrix_t<double, 2, 2>::Identity()));
  EXPECT_TRUE(is_near(make_zero<CovT2l>(), eigen_matrix_t<double, 2, 2>::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<CovT2l>(), eigen_matrix_t<double, 2, 2>::Identity()));
}


TEST(covariance_tests, Covariance_overloads)
{
  CovSA2l covsa2l {9, 3, 3, 10};
  CovSA2u covsa2u {9, 3, 3, 10};
  CovT2l covt2l {9, 3, 3, 10};
  CovT2u covt2u {9, 3, 3, 10};

  EXPECT_TRUE(is_near(nested_object(CovSA2l {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(nested_object(CovSA2u {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(nested_object(CovT2l {9, 3, 3, 10}), Mat2 { 3, 0, 1, 3}));
  EXPECT_TRUE(is_near(nested_object(CovT2u {9, 3, 3, 10}), Mat2 { 3, 1, 0, 3}));

  // Square root
  EXPECT_TRUE(is_near(square_root(CovSA2l {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square_root(CovSA2l {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square_root(CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(square_root(covsa2l).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square_root(covsa2l).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  auto sqcovsa2l_a = square_root(CovSA2l {9, 3, 3, 10}); EXPECT_TRUE(is_near(sqcovsa2l_a.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovsa2l_a.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));

  EXPECT_TRUE(is_near(square_root(CovSA2u {9, 3, 3, 10}), Mat2 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(square_root(CovT2l {9, 3, 3, 10}), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square_root(CovT2l {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square_root(CovT2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  CovT2l covt2l_a {9, 3, 3, 10}; EXPECT_TRUE(is_near(square_root(covt2l_a).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(square_root(covt2l_a).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  auto sqcovt2l_a = square_root(CovT2l {9, 3, 3, 10}); EXPECT_TRUE(is_near(sqcovt2l_a.get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(sqcovt2l_a.get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));

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
  EXPECT_TRUE(is_near(square_root(make_covariance<C>(HermitianAdapter<M2::ConstantReturnType, TriangleType::lower>(M2::Constant(9)))), Mat2 {3, 0, 3, 0}));
  EXPECT_TRUE(is_near(square_root(make_covariance<C>(HermitianAdapter<M2::ConstantReturnType, TriangleType::upper>(M2::Constant(4)))), Mat2 {2, 2, 0, 0}));

  EXPECT_TRUE(is_near(make_dense_object_from(CovSA2l {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_dense_object_from(CovSA2u {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_dense_object_from(CovT2l {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));
  EXPECT_TRUE(is_near(make_dense_object_from(CovT2l {9, 3, 3, 10}), Mat2 { 9, 3, 3, 10}));

  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(CovSA2l {9, 3, 3, 10} * 2))>, CovSA2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(CovSA2u {9, 3, 3, 10} * 2))>, CovSA2u>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(CovT2l {9, 3, 3, 10} * 2))>, CovT2l>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(CovT2u {9, 3, 3, 10} * 2))>, CovT2u>);

  EXPECT_TRUE(is_near(transpose(CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(transpose(CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(transpose(CovT2l {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(transpose(CovT2u {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  EXPECT_TRUE(is_near(adjoint(CovSA2l {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(adjoint(CovSA2u {9, 3, 3, 10}).get_self_adjoint_nested_matrix(), Mat2 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(adjoint(CovT2l {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 1, 0, 3}));
  EXPECT_TRUE(is_near(adjoint(CovT2u {9, 3, 3, 10}).get_triangular_nested_matrix(), Mat2 {3, 0, 1, 3}));

  EXPECT_NEAR(determinant(CovSA2l {9, 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(CovSA2u {9, 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(CovT2l {9, 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(CovT2u {9, 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(square_root(covsa2l)), 9, 1e-6);
  EXPECT_NEAR(determinant(square_root(covsa2u)), 9, 1e-6);
  EXPECT_NEAR(determinant(square_root(covt2l)), 9, 1e-6);
  EXPECT_NEAR(determinant(square_root(covt2u)), 9, 1e-6);

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

  EXPECT_TRUE(is_near(average_reduce<1>(CovSA2l {9, 3, 3, 10}), Mat2col {6., 6.5}));
  EXPECT_TRUE(is_near(average_reduce<1>(CovSA2u {9, 3, 3, 10}), Mat2col {6., 6.5}));
  EXPECT_TRUE(is_near(average_reduce<1>(CovT2l {9, 3, 3, 10}), Mat2col {6., 6.5}));
  EXPECT_TRUE(is_near(average_reduce<1>(CovT2u {9, 3, 3, 10}), Mat2col {6., 6.5}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(CovSA2l {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(CovSA2u {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(CovT2l {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(LQ_decomposition(CovT2u {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));

  EXPECT_TRUE(is_near(square(QR_decomposition(CovSA2l {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(QR_decomposition(CovSA2u {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(QR_decomposition(CovT2l {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
  EXPECT_TRUE(is_near(square(QR_decomposition(CovT2u {9., 3, 3, 10})), Mat2 {90., 57, 57, 109}));
}


TEST(covariance_tests, Covariance_blocks)
{
  using C4 = static_concatenate_t<C, C>;
  using M4 = eigen_matrix_t<double, 4, 4>;
  using Mat4 = Matrix<C4, C4, M4>;
  using CovSA4l = Covariance<C4, HermitianAdapter<M4, TriangleType::lower>>;
  using CovSA4u = Covariance<C4, HermitianAdapter<M4, TriangleType::upper>>;
  using CovT4l = Covariance<C4, TriangularAdapter<M4, TriangleType::lower>>;
  using CovT4u = Covariance<C4, TriangularAdapter<M4, TriangleType::upper>>;
  Mat2 ma {9, 3, 3, 10}, mb {4, 2, 2, 5};
  Mat4 n {9, 3, 0, 0,
          3, 10, 0, 0,
          0, 0, 4, 2,
          0, 0, 2, 5};
  EXPECT_TRUE(is_near(concatenate(CovSA2l(ma), CovSA2l(mb)), n));
  EXPECT_TRUE(is_near(concatenate(CovSA2u(ma), CovSA2u(mb)), n));
  EXPECT_TRUE(is_near(concatenate(CovT2l(ma), CovT2l(mb)), n));
  EXPECT_TRUE(is_near(concatenate(CovT2u(ma), CovT2u(mb)), n));

  EXPECT_TRUE(is_near(split_diagonal(CovSA4l(n)), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<C, C>(CovSA4l(n)), std::tuple {ma, mb}));
  EXPECT_TRUE(is_near(split_diagonal<C, C>(CovSA4u(n)), std::tuple {ma, mb}));
  EXPECT_TRUE(is_near(split_diagonal<C, C>(CovT4l(n)), std::tuple {ma, mb}));
  EXPECT_TRUE(is_near(split_diagonal<C, C>(CovT4u(n)), std::tuple {ma, mb}));
  EXPECT_TRUE(is_near(split_diagonal<C, angle::Radians>(CovSA4l(n)), std::tuple {ma, Matrix<angle::Radians, angle::Radians>{4}}));
  EXPECT_TRUE(is_near(split_diagonal<C, angle::Radians>(CovSA4u(n)), std::tuple {ma, Matrix<angle::Radians, angle::Radians>{4}}));
  EXPECT_TRUE(is_near(split_diagonal<C, angle::Radians>(CovT4l(n)), std::tuple {ma, Matrix<angle::Radians, angle::Radians>{4}}));
  EXPECT_TRUE(is_near(split_diagonal<C, angle::Radians>(CovT4u(n)), std::tuple {ma, Matrix<angle::Radians, angle::Radians>{4}}));

  EXPECT_TRUE(is_near(split_vertical(CovSA4l(n)), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(CovSA4l(n)), std::tuple {concatenate_horizontal(ma, make_zero<Mat2>()), concatenate_horizontal(make_zero<Mat2>(), mb)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(CovSA4u(n)), std::tuple {concatenate_horizontal(ma, make_zero<Mat2>()), concatenate_horizontal(make_zero<Mat2>(), mb)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(CovT4l(n)), std::tuple {concatenate_horizontal(ma, make_zero<Mat2>()), concatenate_horizontal(make_zero<Mat2>(), mb)}));
  EXPECT_TRUE(is_near(split_vertical<C, C>(CovT4u(n)), std::tuple {concatenate_horizontal(ma, make_zero<Mat2>()), concatenate_horizontal(make_zero<Mat2>(), mb)}));
  EXPECT_TRUE(is_near(split_vertical<C, angle::Radians>(CovSA4l(n)), std::tuple {concatenate_horizontal(ma, make_zero<Mat2>()), Matrix<angle::Radians, C4>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_vertical<C, angle::Radians>(CovSA4u(n)), std::tuple {concatenate_horizontal(ma, make_zero<Mat2>()), Matrix<angle::Radians, C4>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_vertical<C, angle::Radians>(CovT4l(n)), std::tuple {concatenate_horizontal(ma, make_zero<Mat2>()), Matrix<angle::Radians, C4>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_vertical<C, angle::Radians>(CovT4u(n)), std::tuple {concatenate_horizontal(ma, make_zero<Mat2>()), Matrix<angle::Radians, C4>{0, 0, 4, 2}}));

  EXPECT_TRUE(is_near(split_horizontal(CovSA4l(n)), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(CovSA4l(n)), std::tuple {concatenate_vertical(ma, make_zero<Mat2>()), concatenate_vertical(make_zero<Mat2>(), mb)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(CovSA4u(n)), std::tuple {concatenate_vertical(ma, make_zero<Mat2>()), concatenate_vertical(make_zero<Mat2>(), mb)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(CovT4l(n)), std::tuple {concatenate_vertical(ma, make_zero<Mat2>()), concatenate_vertical(make_zero<Mat2>(), mb)}));
  EXPECT_TRUE(is_near(split_horizontal<C, C>(CovT4u(n)), std::tuple {concatenate_vertical(ma, make_zero<Mat2>()), concatenate_vertical(make_zero<Mat2>(), mb)}));
  EXPECT_TRUE(is_near(split_horizontal<C, angle::Radians>(CovSA4l(n)), std::tuple {concatenate_vertical(ma, make_zero<Mat2>()), Matrix<C4, angle::Radians>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_horizontal<C, angle::Radians>(CovSA4u(n)), std::tuple {concatenate_vertical(ma, make_zero<Mat2>()), Matrix<C4, angle::Radians>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_horizontal<C, angle::Radians>(CovT4l(n)), std::tuple {concatenate_vertical(ma, make_zero<Mat2>()), Matrix<C4, angle::Radians>{0, 0, 4, 2}}));
  EXPECT_TRUE(is_near(split_horizontal<C, angle::Radians>(CovT4u(n)), std::tuple {concatenate_vertical(ma, make_zero<Mat2>()), Matrix<C4, angle::Radians>{0, 0, 4, 2}}));

  EXPECT_TRUE(is_near(column(CovSA4l(n), 2), Mean{0., 0, 4, 2}));
  EXPECT_TRUE(is_near(column(CovSA4u(n), 2), Mean{0., 0, 4, 2}));
  EXPECT_TRUE(is_near(column(CovT4l(n), 2), Mean{0., 0, 4, 2}));
  EXPECT_TRUE(is_near(column(CovT4u(n), 2), Mean{0., 0, 4, 2}));

  EXPECT_TRUE(is_near(column<1>(CovSA4l(n)), Mean{3., 10, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(CovSA4l(n)), Mean{3., 10, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(CovT4u(n)), Mean{3., 10, 0, 0}));
  EXPECT_TRUE(is_near(column<1>(CovT4l(n)), Mean{3., 10, 0, 0}));

  EXPECT_TRUE(is_near(apply_columnwise([](auto col){ return col * 2; }, CovSA2l(ma)), Mat2 {18, 6, 6, 20}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto col){ return col * 2; }, CovSA2u(ma)), Mat2 {18, 6, 6, 20}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto&& col){ return col * 2; }, CovT2l(ma)), Mat2 {18, 6, 6, 20}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return col * 2; }, CovT2u(ma)), Mat2 {18, 6, 6, 20}));

  EXPECT_TRUE(is_near(apply_columnwise([](auto col, std::size_t i){ return col * i; }, CovSA2l(ma)), Mat2 {0, 3, 0, 10}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto col, std::size_t i){ return col * i; }, CovSA2u(ma)), Mat2 {0, 3, 0, 10}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto&& col, std::size_t i){ return col * i; }, CovT2l(ma)), Mat2 {0, 3, 0, 10}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return col * i; }, CovT2u(ma)), Mat2 {0, 3, 0, 10}));

  EXPECT_TRUE(is_near(apply_coefficientwise([](auto x){ return x + 1; }, CovSA2l(ma)), Mat2 {10, 4, 4, 11}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto x){ return x + 1; }, CovSA2u(ma)), Mat2 {10, 4, 4, 11}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](auto&& x){ return x + 1; }, CovT2l(ma)), Mat2 {10, 4, 4, 11}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, CovT2u(ma)), Mat2 {10, 4, 4, 11}));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, CovSA2l(ma)), Mat2 {9, 4, 4, 12}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, CovSA2u(ma)), Mat2 {9, 4, 4, 12}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, CovT2l(ma)), Mat2 {9, 4, 4, 12}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, CovT2u(ma)), Mat2 {9, 4, 4, 12}));
}


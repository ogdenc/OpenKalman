/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"

using namespace OpenKalman;

using Mat2 = eigen_matrix_t<double, 2, 2>;
using Mat3 = eigen_matrix_t<double, 3, 3>;
using Axis2 = Coefficients<Axis, Axis>;

TEST_F(eigen3, Eigen_Matrix_class_traits)
{
  static_assert(eigen_native<Mat2>);
  static_assert(not eigen_native<double>);
  static_assert(eigen_matrix<Mat2>);
  static_assert(not eigen_matrix<double>);
  EXPECT_TRUE(is_near(MatrixTraits<eigen_matrix_t<double, 2, 3>>::zero(), make_native_matrix<double, 2, 3>(0, 0, 0, 0, 0, 0)));
  EXPECT_TRUE(is_near(MatrixTraits<Mat3>::identity(), make_native_matrix<Mat3>(1, 0, 0, 0, 1, 0, 0, 0, 1)));
  EXPECT_NEAR((make_native_matrix<double, 2, 2>(1, 2, 3, 4))(0, 0), 1, 1e-6);
  EXPECT_NEAR((make_native_matrix<double, 2, 2>(1, 2, 3, 4))(0, 1), 2, 1e-6);
  auto d1 = make_native_matrix<double, 3, 1>(1, 2, 3);
  EXPECT_NEAR(d1(1), 2, 1e-6);
  d1(0) = 5;
  d1(2, 0) = 7;
  EXPECT_TRUE(is_near(d1, make_native_matrix<double, 3, 1>(5, 2, 7)));

  static_assert(self_adjoint_matrix<Eigen::SelfAdjointView<Mat3, Eigen::Upper>>);
  static_assert(self_adjoint_matrix<Eigen::SelfAdjointView<Mat3, Eigen::Lower>>);
  static_assert(upper_triangular_storage<Eigen::SelfAdjointView<Mat3, Eigen::Upper>>);
  static_assert(lower_triangular_storage<Eigen::SelfAdjointView<Mat3, Eigen::Lower>>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mat3, Eigen::Upper>>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mat3, Eigen::Lower>>);
  static_assert(upper_triangular_matrix<Eigen::TriangularView<Mat3, Eigen::Upper>>);
  static_assert(lower_triangular_matrix<Eigen::TriangularView<Mat3, Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::DiagonalMatrix<double, 3>>);
  static_assert(self_adjoint_matrix<typename Mat3::ConstantReturnType>);
  static_assert(self_adjoint_matrix<typename Mat3::IdentityReturnType>);
  static_assert(upper_triangular_matrix<typename Mat3::IdentityReturnType>);
  static_assert(lower_triangular_matrix<typename Mat3::IdentityReturnType>);
  static_assert(diagonal_matrix<typename Mat3::IdentityReturnType>);

  static_assert(self_contained<const Mat2>);
  static_assert(self_contained<typename Mat3::ConstantReturnType>);
  static_assert(self_contained<typename Mat3::IdentityReturnType>);
  static_assert(self_contained<decltype(2 * MatrixTraits<Mat2>::identity() + MatrixTraits<Mat2>::identity())>);
  static_assert(not self_contained<decltype(2 * MatrixTraits<Mat2>::identity() + Mat2 {1, 2, 3, 4})>);
  static_assert(MatrixTraits<std::remove_const_t<decltype(2 * MatrixTraits<Mat2>::identity() + MatrixTraits<Mat2>::identity())>>::rows == 2);
  static_assert(self_contained<decltype(column<0>(2 * MatrixTraits<Mat2>::identity() + MatrixTraits<Mat2>::identity()))>);
  static_assert(not self_contained<decltype(column<0>(2 * MatrixTraits<Mat2>::identity() + Mat2 {1, 2, 3, 4}))>);
  static_assert(self_contained<const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Mat2>>);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Mat2>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Mat2>>>);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, Mat2>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Mat2>>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const Mat2, const Mat2>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const Mat2,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Mat2>>>);

  static_assert(dynamic_rows<eigen_matrix_t<double, 0, 0>>);
  static_assert(dynamic_columns<eigen_matrix_t<double, 0, 0>>);
  static_assert(dynamic_rows<eigen_matrix_t<double, 0, 1>>);
  static_assert(not dynamic_columns<eigen_matrix_t<double, 0, 1>>);
  static_assert(not dynamic_rows<eigen_matrix_t<double, 1, 0>>);
  static_assert(dynamic_columns<eigen_matrix_t<double, 1, 0>>);
  static_assert(zero_matrix<decltype(MatrixTraits<eigen_matrix_t<double, 0, 0>>::zero(2, 3))>);
  static_assert(identity_matrix<decltype(MatrixTraits<eigen_matrix_t<double, 0, 1>>::identity(3))>);
  EXPECT_EQ(row_count(eigen_matrix_t<double, 0, 0> {2, 3}), 2);
  EXPECT_EQ(column_count(eigen_matrix_t<double, 0, 0> {2, 3}), 3);

  auto rm3 = eigen_matrix_t<double, 0, 1> {3};
  EXPECT_EQ(row_count(rm3), 3);
  static_assert(column_count(rm3) == 1);

  auto cm4 = eigen_matrix_t<double, 1, 0> {4};
  static_assert(row_count(cm4) == 1);
  EXPECT_EQ(column_count(cm4), 4);
}


TEST_F(eigen3, Eigen_Matrix_overloads)
{
  EXPECT_TRUE(is_near(diagonal_of(make_native_matrix<double, 2, 2>(1, 2, 3, 4)), make_native_matrix<double, 2, 1>(1, 4)));
  EXPECT_TRUE(is_near(transpose(make_native_matrix<double, 2, 3>(1, 2, 3, 4, 5, 6)),
    make_native_matrix<double, 3, 2>(1, 4, 2, 5, 3, 6)));
  EXPECT_TRUE(is_near(adjoint(make_native_matrix<double, 2, 3>(1, 2, 3, 4, 5, 6)),
    make_native_matrix<double, 3, 2>(1, 4, 2, 5, 3, 6)));
  EXPECT_NEAR(determinant(make_native_matrix<double, 2, 2>(1, 2, 3, 4)), -2, 1e-6);
  EXPECT_NEAR(determinant(make_native_matrix<double, 1, 1>(2)), 2, 1e-6);
  EXPECT_NEAR(trace(make_native_matrix<double, 2, 2>(1, 2, 3, 4)), 5, 1e-6);
  EXPECT_NEAR(trace(make_native_matrix<double, 1, 1>(3)), 3, 1e-6);
  EXPECT_TRUE(is_near(solve(make_native_matrix<double, 2, 2>(1, 2, 3, 4),
    make_native_matrix<double, 2, 1>(5, 6)),
    make_native_matrix<double, 2, 1>(-4, 4.5)));
  EXPECT_TRUE(is_near(solve(make_native_matrix<double, 1, 1>(2),
    make_native_matrix<double, 1, 1>(6)),
    make_native_matrix<double, 1, 1>(3)));
  EXPECT_TRUE(is_near(reduce_columns(make_native_matrix<double, 2, 3>(1, 2, 3, 4, 5, 6)),
    make_native_matrix<double, 2, 1>(2, 5)));
  EXPECT_TRUE(is_near(LQ_decomposition(make_native_matrix<double, 2, 2>(0.06, 0.08, 0.36, -1.640)),
    make_native_matrix<double, 2, 2>(-0.1, 0, 1.096, -1.272)));
  EXPECT_TRUE(is_near(QR_decomposition(make_native_matrix<double, 2, 2>(0.06, 0.36, 0.08, -1.640)),
    make_native_matrix<double, 2, 2>(-0.1, 1.096, 0, -1.272)));
  //
  using N = std::normal_distribution<double>::param_type;
  using Mat = eigen_matrix_t<double, 3, 1>;
  Mat m;
  m = randomize<Mat>(0.0, 1.0);
  m = randomize<Mat>(0.0, 0.7);
  m = Mat::Zero();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat>(N {1.0, 0.3})) / (i + 1);
  }
  auto offset1 = Mat::Constant(1);
  EXPECT_TRUE(is_near(m, offset1, 0.1));
  EXPECT_FALSE(is_near(m, offset1, 1e-8));

  Mat2 m22;
  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<Mat2>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
  }
  auto offset2 = MatrixTraits<Mat2>::make(1., 1., 2., 2.);
  EXPECT_TRUE(is_near(m22, offset2, 0.1));
  EXPECT_FALSE(is_near(m22, offset2, 1e-8));

  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<Mat2>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})) / (i + 1);
  }
  auto offset3 = MatrixTraits<Mat2>::make(1., 2., 3., 4.);
  EXPECT_TRUE(is_near(m22, offset3, 0.1));
  EXPECT_FALSE(is_near(m22, offset3, 1e-8));
}


TEST_F(eigen3, ConstantMatrix)
{
  ConstantMatrix<double, 3, 2, 2> cm322;
  EXPECT_TRUE(is_near(cm322, eigen_matrix_t<double, 2, 2>::Constant(3)));
  EXPECT_NEAR((ConstantMatrix<double, 3, 2, 2> {}(0, 0)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, 2, 2> {}(0, 1)), 3, 1e-6);
  EXPECT_TRUE(is_near(make_native_matrix(ConstantMatrix<double, 3, 2, 3> {}), eigen_matrix_t<double, 2, 3>::Constant(3)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix<double, 3, 2, 3> {}), eigen_matrix_t<double, 2, 3>::Constant(3)));
  EXPECT_NEAR(determinant(ConstantMatrix<double, 3, 4, 4> {}), 0, 1e-6);
  EXPECT_NEAR(trace(ConstantMatrix<double, 3, 10, 10> {}), 30, 1e-6);
  EXPECT_TRUE(is_near(reduce_columns(ConstantMatrix<double, 3, 2, 3> ()), (eigen_matrix_t<double, 2, 1>::Constant(3))));
  EXPECT_TRUE(is_near(LQ_decomposition(ConstantMatrix<double, 3, 3, 2> ()), LQ_decomposition(make_native_matrix<double, 3, 2>(3, 3, 3, 3, 3, 3)).cwiseAbs()));
  EXPECT_TRUE(is_near(LQ_decomposition(ConstantMatrix<double, 7, 5, 3> ()), LQ_decomposition(make_native_matrix<double, 5, 3>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  EXPECT_TRUE(is_near(QR_decomposition(ConstantMatrix<double, 3, 2, 3> ()), QR_decomposition(make_native_matrix<double, 2, 3>(3, 3, 3, 3, 3, 3)).cwiseAbs()));
  EXPECT_TRUE(is_near(QR_decomposition(ConstantMatrix<double, 7, 3, 5> ()), QR_decomposition(make_native_matrix<double, 3, 5>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  EXPECT_NEAR(get_element(ConstantMatrix<double, 5, 2, 2> {}, 1, 0), 5, 1e-8);
  static_assert(element_gettable<ConstantMatrix<double, 3, 2, 2>, 2>);
  static_assert(not element_settable<ConstantMatrix<double, 3, 2, 2>, 2>);
  EXPECT_TRUE(is_near(column(ConstantMatrix<double, 6, 2, 3> {}, 1), (eigen_matrix_t<double, 2, 1>::Constant(6))));
  EXPECT_TRUE(is_near(column<1>(ConstantMatrix<double, 7, 2, 3> {}), (eigen_matrix_t<double, 2, 1>::Constant(7))));

  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} + ConstantMatrix<double, 5, 2, 2> {}, ConstantMatrix<double, 8, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} - ConstantMatrix<double, 5, 2, 2> {}, ConstantMatrix<double, -2, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {} * ConstantMatrix<double, 5, 3, 2> {}, ConstantMatrix<double, 45, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 4, 3, 4> {} * ConstantMatrix<double, 7, 4, 2> {}, ConstantMatrix<double, 112, 3, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 0, 2, 2> {} * 2.0, ConstantMatrix<double, 0, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} * -2.0, ConstantMatrix<double, -6, 2, 2> {}));
  EXPECT_TRUE(is_near(3.0 * ConstantMatrix<double, 0, 2, 2> {}, ConstantMatrix<double, 0, 2, 2> {}));
  EXPECT_TRUE(is_near(-3.0 * ConstantMatrix<double, 3, 2, 2> {}, ConstantMatrix<double, -9, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 0, 2, 2> {} / 2.0, ConstantMatrix<double, 0, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 8, 2, 2> {} / -2.0, ConstantMatrix<double, -4, 2, 2> {}));
  EXPECT_TRUE(is_near(-ConstantMatrix<double, 7, 2, 2> {}, ConstantMatrix<double, -7, 2, 2> {}));

  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} + eigen_matrix_t<double, 2, 2>::Constant(5), ConstantMatrix<double, 8, 2, 2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 2>::Constant(5) + ConstantMatrix<double, 3, 2, 2> {}, ConstantMatrix<double, 8, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} - eigen_matrix_t<double, 2, 2>::Constant(5), ConstantMatrix<double, -2, 2, 2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 2>::Constant(5) - ConstantMatrix<double, 3, 2, 2> {}, ConstantMatrix<double, 2, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {} * eigen_matrix_t<double, 3, 2>::Constant(5), ConstantMatrix<double, 45, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 4, 3, 4> {} * eigen_matrix_t<double, 4, 2>::Constant(7), ConstantMatrix<double, 112, 3, 2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 3>::Constant(3) * ConstantMatrix<double, 5, 3, 2> {}, ConstantMatrix<double, 45, 2, 2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 3, 4>::Constant(4) * ConstantMatrix<double, 7, 4, 2> {}, ConstantMatrix<double, 112, 3, 2> {}));
}


TEST_F(eigen3, ZeroMatrix)
{
  EXPECT_NEAR((ZeroMatrix<double, 2, 2>()(0, 0)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<double, 2, 2>()(0, 1)), 0, 1e-6);
  EXPECT_TRUE(is_near(make_native_matrix(ZeroMatrix<double, 2, 3>()), eigen_matrix_t<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(make_self_contained(ZeroMatrix<double, 2, 3>()), eigen_matrix_t<double, 2, 3>::Zero()));
  EXPECT_NEAR(determinant(ZeroMatrix<double, 2, 2>()), 0, 1e-6);
  EXPECT_NEAR(trace(ZeroMatrix<double, 2, 2>()), 0, 1e-6);
  EXPECT_TRUE(is_near(reduce_columns(ZeroMatrix<double, 2, 3>()), (eigen_matrix_t<double, 2, 1>::Zero())));
  EXPECT_NEAR(get_element(ZeroMatrix<double, 2, 2>(), 1, 0), 0, 1e-8);
  static_assert(element_gettable<ZeroMatrix<double, 2, 2>, 2>);
  static_assert(not element_settable<ZeroMatrix<double, 2, 2>, 2>);
  EXPECT_TRUE(is_near(column(ZeroMatrix<double, 2, 3>(), 1), (eigen_matrix_t<double, 2, 1>::Zero())));
  EXPECT_TRUE(is_near(column<1>(ZeroMatrix<double, 2, 3>()), (eigen_matrix_t<double, 2, 1>::Zero())));

  ZeroMatrix<double, 0, 0> z00 {2, 2};
  EXPECT_NEAR((z00(0, 0)), 0, 1e-6);
  EXPECT_NEAR((z00(0, 1)), 0, 1e-6);
  EXPECT_NEAR((z00(1, 0)), 0, 1e-6);
  EXPECT_NEAR((z00(1, 1)), 0, 1e-6);
  EXPECT_NEAR((get_element(z00, 0, 0)), 0, 1e-6);
  EXPECT_NEAR((get_element(z00, 0, 1)), 0, 1e-6);
  EXPECT_NEAR((get_element(z00, 1, 0)), 0, 1e-6);
  EXPECT_NEAR((get_element(z00, 1, 1)), 0, 1e-6);
  EXPECT_NEAR((get_element(ZeroMatrix<double, 0, 1> {3}, 0)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<double, 0, 1> {3}(2)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<double, 1, 0> {3}(2)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<double, 0, 2> {3}(2, 1)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<double, 2, 0> {3}(1, 2)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix {eigen_matrix_t<double, 2, 3>{}}(1, 2)), 0, 1e-6);
  EXPECT_EQ(row_count(z00), 2);
  EXPECT_EQ(column_count(z00), 2);
  EXPECT_EQ(row_count(ZeroMatrix<double, 0, 1> {3}), 3);
  EXPECT_EQ((column_count(ZeroMatrix<double, 1, 0> {4})), 4);
  static_assert(dynamic_rows<ZeroMatrix<double, 0, 0>>);
  static_assert(dynamic_columns<ZeroMatrix<double, 0, 0>>);
  static_assert(dynamic_rows<ZeroMatrix<double, 0, 1>>);
  static_assert(not dynamic_columns<ZeroMatrix<double, 0, 1>>);
  static_assert(not dynamic_rows<ZeroMatrix<double, 1, 0>>);
  static_assert(dynamic_columns<ZeroMatrix<double, 1, 0>>);
  static_assert(zero_matrix<decltype(MatrixTraits<ZeroMatrix<double, 0, 0>>::zero(2, 3))>);
  static_assert(identity_matrix<decltype(MatrixTraits<ZeroMatrix<double, 0, 1>>::identity(3))>);

  ZeroMatrix<double, 0, 1> z01 {3};
  static_assert(diagonal_matrix<decltype(to_diagonal(z01))>);
  EXPECT_EQ(row_count(to_diagonal(z01)), 3);
  EXPECT_EQ(column_count(to_diagonal(z01)), 3);
  static_assert(column_vector<decltype(diagonal_of(z00))>);
  EXPECT_EQ(row_count(diagonal_of(z00)), 2);

  ZeroMatrix<double, 0, 0> zc34 {3, 4};
  static_assert(zero_matrix<decltype(adjoint(zc34))>);
  EXPECT_EQ(row_count(adjoint(zc34)), 4);
  EXPECT_EQ(column_count(adjoint(zc34)), 3);
  static_assert(zero_matrix<decltype(transpose(zc34))>);
  EXPECT_EQ(row_count(transpose(zc34)), 4);
  EXPECT_EQ(column_count(transpose(zc34)), 3);
  EXPECT_EQ(determinant(zc34), 0);
  EXPECT_EQ(trace(zc34), 0);

  auto rzc34 = reduce_columns(zc34);
  EXPECT_EQ(row_count(rzc34), 3);
  EXPECT_EQ(column_count(rzc34), 1);
  static_assert(zero_matrix<decltype(rzc34)>);

  auto lqzc34 = LQ_decomposition(zc34);
  EXPECT_EQ(row_count(lqzc34), 3);
  EXPECT_EQ(column_count(lqzc34), 3);
  static_assert(zero_matrix<decltype(lqzc34)>);

  auto qrzc34 = QR_decomposition(zc34);
  EXPECT_EQ(row_count(qrzc34), 4);
  EXPECT_EQ(column_count(qrzc34), 4);
  static_assert(zero_matrix<decltype(qrzc34)>);

  auto czc34 = column(zc34, 1);
  EXPECT_EQ(row_count(czc34), 3);
  static_assert(column_count(czc34) == 1);
  static_assert(zero_matrix<decltype(czc34)>);

  auto czv34 = column<1>(ZeroMatrix<double, 0, 4> {3});
  EXPECT_EQ(row_count(czv34), 3);
  static_assert(column_count(czv34) == 1);
  static_assert(zero_matrix<decltype(czv34)>);

  auto m22y = make_native_matrix<double, 2, 2>(1, 2, 3, 4);
  EXPECT_TRUE(is_near(z00 + m22y, m22y, 1e-6));
  EXPECT_TRUE(is_near(m22y + z00, m22y, 1e-6));
  static_assert(zero_matrix<decltype(z00 + z00)>);
  EXPECT_TRUE(is_near(m22y - z00, m22y, 1e-6));
  EXPECT_TRUE(is_near(z00 - m22y, -m22y, 1e-6));
  EXPECT_TRUE(is_near(z00 - m22y.Identity(), -m22y.Identity(), 1e-6));
  static_assert(diagonal_matrix<decltype(z00 - m22y.Identity())>);
  static_assert(zero_matrix<decltype(z00 - z00)>);
  EXPECT_TRUE(is_near(z00 * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 * m22y, z00, 1e-6));
  EXPECT_TRUE(is_near(m22y * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 * 2, z00, 1e-6));
  EXPECT_TRUE(is_near(2 * z00, z00, 1e-6));
  EXPECT_TRUE(is_near(z00 / 2, z00, 1e-6));
  EXPECT_TRUE(is_near(-z00, z00, 1e-6));
}

TEST_F(eigen3, Matrix_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(make_native_matrix<double, 2, 2>(1, 2, 3, 4),
    make_native_matrix<double, 1, 2>(5, 6)), make_native_matrix<double, 3, 2>(1, 2, 3, 4, 5, 6)));
  EXPECT_TRUE(is_near(concatenate_horizontal(make_native_matrix<double, 2, 2>(1, 2, 3, 4),
    make_native_matrix<double, 2, 1>(5, 6)), make_native_matrix<double, 2, 3>(1, 2, 5, 3, 4, 6)));
  EXPECT_TRUE(is_near(concatenate_diagonal(make_native_matrix<double, 2, 2>(1, 2, 3, 4),
    make_native_matrix<double, 2, 2>(5, 6, 7, 8)), make_native_matrix<double, 4, 4>(1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8)));
  EXPECT_TRUE(is_near(split_vertical(make_native_matrix<double, 2, 2>(1, 0, 0, 2)), std::tuple{}));
  eigen_matrix_t<double, 5, 3> x1;
  x1 <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3,
    4, 0, 0,
    0, 5, 0;
  auto a1 = split_vertical<3, 2>(x1);
  EXPECT_TRUE(is_near(a1, std::tuple{make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3),
               make_native_matrix<double, 2, 3>(
                 4, 0, 0,
                 0, 5, 0)}));
  auto a2 = split_vertical<3, 2>(make_native_matrix<double, 5, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      4, 0, 0,
      0, 5, 0));
  EXPECT_TRUE(is_near(a2,
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3),
               make_native_matrix<double, 2, 3>(
                 4, 0, 0,
                 0, 5, 0)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(make_native_matrix<double, 5, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      4, 0, 0,
      0, 5, 0)),
    std::tuple{make_native_matrix<double, 2, 3>(
      1, 0, 0,
      0, 2, 0),
    make_native_matrix<double, 2, 3>(
                 0, 0, 3,
                 4, 0, 0)}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(MatrixTraits<eigen_matrix_t<double, 5, 3>>::zero()),
    std::tuple {eigen_matrix_t<double, 3, 3>::Zero(), eigen_matrix_t<double, 2, 3>::Zero()}));
  EXPECT_TRUE(is_near(split_horizontal(make_native_matrix<double, 2, 2>(1, 0, 0, 2)), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(make_native_matrix<double, 3, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 4, 0,
      0, 0, 3, 0, 5)),
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3),
               make_native_matrix<double, 3, 2>(
                 0, 0,
                 4, 0,
                 0, 5)}));
  const auto b1 = make_native_matrix<double, 3, 5>(
    1, 0, 0, 0, 0,
    0, 2, 0, 4, 0,
    0, 0, 3, 0, 5);
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(b1),
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3),
               make_native_matrix<double, 3, 2>(
                 0, 0,
                 4, 0,
                 0, 5)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(make_native_matrix<double, 3, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 4, 0,
      0, 0, 3, 0, 5)),
    std::tuple{make_native_matrix<double, 3, 2>(
      1, 0,
      0, 2,
      0, 0),
    make_native_matrix<double, 3, 2>(
                 0, 0,
                 0, 4,
                 3, 0)}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(MatrixTraits<eigen_matrix_t<double, 3, 5>>::zero()),
    std::tuple {eigen_matrix_t<double, 3, 3>::Zero(), eigen_matrix_t<double, 3, 2>::Zero()}));
  EXPECT_TRUE(is_near(split_diagonal(make_native_matrix<double, 2, 2>(1, 0, 0, 2)), std::tuple{}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(make_native_matrix<double, 5, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0,
      0, 0, 0, 4, 0,
      0, 0, 0, 0, 5)),
    std::tuple{make_native_matrix<double, 3, 3>(
        1, 0, 0,
        0, 2, 0,
        0, 0, 3),
    make_native_matrix<double, 2, 2>(
                 4, 0,
                 0, 5)}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(make_native_matrix<double, 5, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0,
      0, 0, 0, 4, 0,
      0, 0, 0, 0, 5)),
    std::tuple{make_native_matrix<double, 2, 2>(
      1, 0,
      0, 2),
    make_native_matrix<double, 2, 2>(
                 3, 0,
                 0, 4)}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(MatrixTraits<eigen_matrix_t<double, 5, 5>>::zero()),
    std::tuple {eigen_matrix_t<double, 3, 3>::Zero(), eigen_matrix_t<double, 2, 2>::Zero()}));

  eigen_matrix_t<double, 2, 2> el; el << 1, 2, 3, 4;
  set_element(el, 5.5, 1, 0);
  EXPECT_NEAR(get_element(el, 1, 0), 5.5, 1e-8);
  static_assert(element_gettable<eigen_matrix_t<double, 3, 2>, 2>);
  static_assert(element_gettable<eigen_matrix_t<double, 3, 1>, 1>);
  static_assert(element_settable<eigen_matrix_t<double, 3, 2>, 2>);
  static_assert(element_settable<eigen_matrix_t<double, 3, 1>, 1>);
  static_assert(not element_gettable<eigen_matrix_t<double, 3, 2>, 1>);
  static_assert(element_gettable<eigen_matrix_t<double, 3, 1>, 2>);
  static_assert(not element_settable<eigen_matrix_t<double, 3, 2>, 1>);
  static_assert(element_settable<eigen_matrix_t<double, 3, 1>, 2>);
  static_assert(not element_settable<const eigen_matrix_t<double, 3, 2>, 2>);
  static_assert(not element_settable<const eigen_matrix_t<double, 3, 1>, 1>);

  EXPECT_TRUE(is_near(column(make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3), 2), make_native_matrix(0., 0, 3)));
  EXPECT_TRUE(is_near(column(MatrixTraits<eigen_matrix_t<double, 3, 3>>::zero(), 2), make_native_matrix(0., 0, 0)));
  EXPECT_TRUE(is_near(column<1>(make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3)), make_native_matrix(0., 2, 0)));
  EXPECT_TRUE(is_near(column<1>(MatrixTraits<eigen_matrix_t<double, 3, 3>>::zero()), make_native_matrix(0., 0, 0)));
  eigen_matrix_t<double, 3, 3> d1;
  d1 <<
  1, 0, 0,
  0, 2, 0,
  0, 0, 3;
  const auto d_reset = d1;
  apply_columnwise(d1, [](auto& col){ col += col.Constant(1); });
  EXPECT_TRUE(is_near(d1,
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_columnwise(make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3), [](const auto& col){ return col + col.Constant(1); }),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_columnwise(MatrixTraits<eigen_matrix_t<double, 3, 3>>::zero(),
    [](const auto& col){ return col + col.Constant(1); }), eigen_matrix_t<double, 3, 3>::Constant(1)));
  d1 = d_reset;
  apply_columnwise(d1, [](auto& col, std::size_t i){ col += col.Constant(i); });
  EXPECT_TRUE(is_near(d1,
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      0, 3, 2,
      0, 1, 5)));
  EXPECT_TRUE(is_near(apply_columnwise(make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3), [](const auto& col, std::size_t i){ return col + col.Constant(i); }),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      0, 3, 2,
      0, 1, 5)));
  EXPECT_TRUE(is_near(apply_columnwise<3>([] { return make_native_matrix(1., 2, 3); }),
    make_native_matrix<double, 3, 3>(
      1, 1, 1,
      2, 2, 2,
      3, 3, 3)));
  EXPECT_TRUE(is_near(apply_columnwise<3>([](std::size_t i) { return make_native_matrix(1. + i, 2 + i, 3 + i); }),
    make_native_matrix<double, 3, 3>(
      1, 2, 3,
      2, 3, 4,
      3, 4, 5)));
  d1 = d_reset;
  apply_coefficientwise(d1, [](auto& x){ x += 1; });
  EXPECT_TRUE(is_near(d1,
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_coefficientwise(make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3), [](const auto& x){ return x + 1; }),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_coefficientwise(MatrixTraits<eigen_matrix_t<double, 3, 3>>::zero(),
    [](const auto& x){ return x + 1; }), eigen_matrix_t<double, 3, 3>::Constant(1)));
  d1 = d_reset;
  apply_coefficientwise(d1, [](auto& x, std::size_t i, std::size_t j){ x += i + j; });
  EXPECT_TRUE(is_near(d1,
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      1, 4, 3,
      2, 3, 7)));
  EXPECT_TRUE(is_near(apply_coefficientwise(make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      1, 4, 3,
      2, 3, 7)));
}

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "typed_matrix_tests.h"

using namespace OpenKalman;

using M12 = Eigen::Matrix<double, 1, 2>;
using M21 = Eigen::Matrix<double, 2, 1>;
using M22 = Eigen::Matrix<double, 2, 2>;
using M23 = Eigen::Matrix<double, 2, 3>;
using M32 = Eigen::Matrix<double, 3, 2>;
using M33 = Eigen::Matrix<double, 3, 3>;
using I22 = EigenIdentity<M22>;
using Z22 = EigenZero<M22>;
using C2 = Coefficients<Axis, Angle>;
using C3 = Coefficients<Axis, Angle, Axis>;
using Mat12 = TypedMatrix<Axis, C2, M12>;
using Mat21 = TypedMatrix<C2, Axis, M21>;
using Mat22 = TypedMatrix<C2, C2, M22>;
using Mat23 = TypedMatrix<C2, C3, M23>;
using Mat32 = TypedMatrix<C3, C2, M32>;
using Mat33 = TypedMatrix<C3, C3, M33>;

inline I22 i22 = M22::Identity();
inline Z22 z22 = Z22();
inline auto covi22 = Covariance<C2, I22>(i22);
inline auto covz22 = Covariance<C2, Z22>(z22);
inline auto sqcovi22 = SquareRootCovariance<C2, I22>(i22);
inline auto sqcovz22 = SquareRootCovariance<C2, Z22>(z22);


TEST_F(typed_matrix_tests, TypedMatrix_class)
{
  // Default constructor and Eigen3 construction
  Mat23 mat23a;
  mat23a << 1, 2, 3, 4, 5, 6;
  EXPECT_TRUE(is_near(mat23a, Mat23 {1, 2, 3, 4, 5, 6}));

  // Copy constructor
  Mat23 mat23b = const_cast<const Mat23&>(mat23a);
  EXPECT_TRUE(is_near(mat23b, Mat23 {1, 2, 3, 4, 5, 6}));

  // Move constructor
  Mat23 mat23c(std::move(Mat23 {6, 5, 4, 3, 2, 1}));
  EXPECT_TRUE(is_near(mat23c, Mat23 {6, 5, 4, 3, 2, 1}));

  // Convert from different covariance types
  TypedMatrix<C2, Axes<3>, M23> mat23_x1(Mean<C2, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x1, Mat23 {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI}));
  TypedMatrix<Axes<2>, Axes<3>, M23> mat23_x2(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x2, Mat23 {1, 2, 3, 4, 5, 6}));
  TypedMatrix<C2, Axes<3>, M23> mat23_x3(EuclideanMean<C2, M33> {
    1, 2, 3,
    0.5, std::sqrt(3)/2, M_SQRT2/2,
    std::sqrt(3)/2, 0.5, M_SQRT2/2});
  EXPECT_TRUE(is_near(mat23_x3, Mat23 {1, 2, 3, M_PI/3, M_PI/6, M_PI/4}));

  // Construct from a regular matrix
  Mat23 mat23d((M23() << 1, 2, 3, 4, 5, 6).finished());
  EXPECT_TRUE(is_near(mat23d, Mat23 {1, 2, 3, 4, 5, 6}));

  // Convert from a compatible covariance
  Mat22 mat22a_1(Covariance<C2, EigenSelfAdjointMatrix<M22, TriangleType::lower>> {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mat22a_1, Mat22 {9, 3, 3, 10}));
  Mat22 mat22a_2(Covariance<C2, EigenSelfAdjointMatrix<M22, TriangleType::upper>> {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mat22a_2, Mat22 {9, 3, 3, 10}));
  Mat22 mat22a_3(Covariance<C2, EigenSelfAdjointMatrix<M22, TriangleType::lower>> {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mat22a_3, Mat22 {9, 3, 3, 10}));
  Mat22 mat22a_4(Covariance<C2, EigenTriangularMatrix<M22, TriangleType::upper>> {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mat22a_4, Mat22 {9, 3, 3, 10}));
  Mat22 mat22a_5(Covariance<C2, EigenDiagonal<M21>> {9, 10});
  EXPECT_TRUE(is_near(mat22a_5, Mat22 {9, 0, 0, 10}));
  Mat22 mat22a_6(covi22);
  EXPECT_TRUE(is_near(mat22a_6, Mat22 {1, 0, 0, 1}));
  Mat22 mat22a_7(covz22);
  EXPECT_TRUE(is_near(mat22a_7, Mat22 {0, 0, 0, 0}));

  // Convert from a compatible square root covariance
  Mat22 mat22b_1(SquareRootCovariance<C2, EigenSelfAdjointMatrix<M22, TriangleType::lower>> {3, 0, 1, 3});
  EXPECT_TRUE(is_near(mat22b_1, Mat22 {3, 0, 1, 3}));
  Mat22 mat22b_2(SquareRootCovariance<C2, EigenSelfAdjointMatrix<M22, TriangleType::upper>> {3, 1, 0, 3});
  EXPECT_TRUE(is_near(mat22b_2, Mat22 {3, 1, 0, 3}));
  Mat22 mat22b_3(SquareRootCovariance<C2, EigenSelfAdjointMatrix<M22, TriangleType::lower>> {3, 0, 1, 3});
  EXPECT_TRUE(is_near(mat22b_3, Mat22 {3, 0, 1, 3}));
  Mat22 mat22b_4(SquareRootCovariance<C2, EigenTriangularMatrix<M22, TriangleType::upper>> {3, 1, 0, 3});
  EXPECT_TRUE(is_near(mat22b_4, Mat22 {3, 1, 0, 3}));
  Mat22 mat22b_5(SquareRootCovariance<C2, EigenDiagonal<M21>> {3, 4});
  EXPECT_TRUE(is_near(mat22b_5, Mat22 {3, 0, 0, 4}));
  Mat22 mat22b_6(sqcovi22);
  EXPECT_TRUE(is_near(mat22b_6, Mat22 {1, 0, 0, 1}));
  Mat22 mat22b_7(sqcovz22);
  EXPECT_TRUE(is_near(mat22b_7, Mat22 {0, 0, 0, 0}));

  // Construct from a list of coefficients
  Mat23 mat23e {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23e, Mat23 {1, 2, 3, 4, 5, 6}));

  // Copy assignment
  mat23c = mat23b;
  EXPECT_TRUE(is_near(mat23c, Mat23 {1, 2, 3, 4, 5, 6}));

  // Move assignment
  mat23c = std::move(Mat23 {3, 4, 5, 6, 7, 8});
  EXPECT_TRUE(is_near(mat23c, Mat23 {3, 4, 5, 6, 7, 8}));

  // assign from different covariance types
  mat23_x1 = Mean<C2, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x1, Mat23 {6, 5, 4, 3, 2, 1}));
  mat23_x2 = EuclideanMean<Axes<2>, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x2, Mat23 {6, 5, 4, 3, 2, 1}));
  mat23_x3 = EuclideanMean<C2, M33> {
    3, 2, 1,
    std::sqrt(3)/2, M_SQRT2/2, 0.5,
    0.5, M_SQRT2/2, std::sqrt(3)/2};
  EXPECT_TRUE(is_near(mat23_x3, Mat23 {3, 2, 1, M_PI/6, M_PI/4, M_PI/3}));

  // assign from a regular matrix
  mat23e = (M23() << 3, 4, 5, 6, 7, 8).finished();

  // Assign from a list of coefficients (via move assignment operator)
  mat23e = {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23e, Mat23 {6, 5, 4, 3, 2, 1}));

  // Increment
  mat23a += Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23a, Mat23 {2, 4, 6, 8, 10, 12}));

  // Increment with a stochastic value
  mat23_x1 = {1, 1, 1, M_PI, M_PI, M_PI};
  GaussianDistribution g = {Mat21 {0, 0}, sqcovi22 * 0.1};
  TypedMatrix<C2, Axes<3>, M23> diff = {1, 1, 1, 1, 1, 1};
  for (int i=0; i<100; i++)
  {
    auto oldmat23_x1 = mat23_x1;
    mat23_x1 += g;
    diff += (mat23_x1 - oldmat23_x1) / 100;
  }
  EXPECT_TRUE(is_near(diff, Mat23 {1, 1, 1, 1, 1, 1}, 0.1));
  EXPECT_FALSE(is_near(diff, Mat23 {1, 1, 1, 1, 1, 1}, 1e-6));

  // Decrement
  mat23a -= Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23a, Mat23 {1, 2, 3, 4, 5, 6}));

  // Decrement with a stochastic value
  mat23_x1 = {1, 1, 1, M_PI, M_PI, M_PI};
  diff = {1, 1, 1, 1, 1, 1};
  for (int i=0; i<100; i++)
  {
    auto oldmat23_x1 = mat23_x1;
    mat23_x1 -= g;
    diff += (mat23_x1 - oldmat23_x1) / 100;
  }
  EXPECT_TRUE(is_near(diff, Mat23 {1, 1, 1, 1, 1, 1}, 0.1));
  EXPECT_FALSE(is_near(diff, Mat23 {1, 1, 1, 1, 1, 1}, 1e-6));

  // Scalar multiplication
  mat23a *= 2;
  EXPECT_TRUE(is_near(mat23a, Mat23 {2, 4, 6, 8, 10, 12}));

  // Scalar division
  mat23a /= 2;
  EXPECT_TRUE(is_near(mat23a, Mat23 {1, 2, 3, 4, 5, 6}));

  // Scalar multiplication, zero
  mat23a *= 0;
  EXPECT_TRUE(is_near(mat23a, M23::Zero()));

  // Zero
  EXPECT_TRUE(is_near(Mat23::zero(), M23::Zero()));

  // Identity
  EXPECT_TRUE(is_near(Mat22::identity(), M22::Identity()));

  // Subscripts
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 0), 1, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 1), 2, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 2), 3, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 0), 4, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 1), 5, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 2), 6, 1e-6);
}


TEST_F(typed_matrix_tests, TypedMatrix_deduction_guides)
{
  auto a = (M23() << 1, 2, 3, 4, 5, 6).finished();
  EXPECT_TRUE(is_near(TypedMatrix(a), a));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(a))>::RowCoefficients, Axes<2>>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(a))>::ColumnCoefficients, Axes<3>>);

  auto b1 = Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(TypedMatrix(b1), a));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(b1))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(b1))>::ColumnCoefficients, C3>);

  auto b2 = Mean<C2, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(TypedMatrix(b2), (M23() << 1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI).finished()));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(b2))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(b2))>::ColumnCoefficients, Axes<3>>);

  auto b3 = EuclideanMean<C2, M33> {1, 2, 3,
                                    0.5, std::sqrt(3)/2, M_SQRT2/2,
                                    std::sqrt(3)/2, 0.5, M_SQRT2/2};
  EXPECT_TRUE(is_near(TypedMatrix(b3), Mat23 {1, 2, 3, M_PI/3, M_PI/6, M_PI/4}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(b3))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(b3))>::ColumnCoefficients, Axes<3>>);

  auto c = Covariance<C2, EigenSelfAdjointMatrix<M22, TriangleType::lower>> {9, 3, 3, 10};
  EXPECT_TRUE(is_near(TypedMatrix(c), Mat22 {9, 3, 3, 10}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(c))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix(c))>::ColumnCoefficients, C2>);
}


TEST_F(typed_matrix_tests, TypedMatrix_make_functions)
{
  auto a = (M23() << 1, 2, 3, 4, 5, 6).finished();
  EXPECT_TRUE(is_near(make_Matrix<C2, C3>(a), a));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix<C2, C3>(a))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix<C2, C3>(a))>::ColumnCoefficients, C3>);
  EXPECT_TRUE(is_near(make_Matrix<C2>(a), a));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix<C2>(a))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix<C2>(a))>::ColumnCoefficients, Axes<3>>);

  auto b = Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(make_Matrix(b), a));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix(b))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix(b))>::ColumnCoefficients, C3>);

  auto c = Covariance<C2, EigenSelfAdjointMatrix<M22, TriangleType::lower>> {9, 3, 3, 10};
  EXPECT_TRUE(is_near(make_Matrix(c), Mat22 {9, 3, 3, 10}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix(c))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix(c))>::ColumnCoefficients, C2>);

  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix<C2, C3, M23>())>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix<C2, C3, M23>())>::ColumnCoefficients, C3>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix<M23>())>::RowCoefficients, Axes<2>>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Matrix<M23>())>::ColumnCoefficients, Axes<3>>);
}


TEST_F(typed_matrix_tests, TypedMatrix_traits)
{
  static_assert(is_typed_matrix_v<Mat23>);
  static_assert(not is_mean_v<Mat23>);
  static_assert(not is_Euclidean_mean_v<Mat23>);
  static_assert(not is_Euclidean_mean_v<TypedMatrix<C2, Axes<2>, I22>>);
  static_assert(not is_Euclidean_transformed_v<Mat23>);
  static_assert(not is_Euclidean_transformed_v<TypedMatrix<C2, Axes<2>, I22>>);
  static_assert(not is_wrapped_v<Mat23>);
  static_assert(not is_wrapped_v<TypedMatrix<C2, Axes<2>, I22>>);
  static_assert(not is_column_vector_v<Mat23>);
  static_assert(is_column_vector_v<TypedMatrix<C2, Axes<2>, I22>>);

  static_assert(not is_identity_v<Mat23>);
  static_assert(is_identity_v<TypedMatrix<C2, C2, I22>>);
  static_assert(not is_identity_v<TypedMatrix<C2, Axes<2>, I22>>);
  static_assert(not is_zero_v<Mat23>);
  static_assert(is_zero_v<TypedMatrix<C2, C2, Z22>>);
  static_assert(is_zero_v<TypedMatrix<C2, C3, EigenZero<M23>>>);

  EXPECT_TRUE(is_near(MatrixTraits<Mat23>::make(
    (Eigen::Matrix<double, 2, 3>() << 1, 2, 3, 4, 5, 6).finished()).base_matrix(), Mat23 {1, 2, 3, 4, 5, 6}));
  EXPECT_TRUE(is_near(MatrixTraits<Mat23>::zero(), Eigen::Matrix<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<Mat22>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
}


TEST_F(typed_matrix_tests, TypedMatrix_overloads)
{
  EXPECT_TRUE(is_near(base_matrix(Mat23 {1, 2, 3, 4, 5, 6}), Mat23 {1, 2, 3, 4, 5, 6}));

  EXPECT_TRUE(is_near(strict_matrix(Mat23 {1, 2, 3, 4, 5, 6}), Mat23 {1, 2, 3, 4, 5, 6}));

  EXPECT_TRUE(is_near(strict(Mat23 {1, 2, 3, 4, 5, 6} * 2), Mat23 {2, 4, 6, 8, 10, 12}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(Mat23 {1, 2, 3, 4, 5, 6} * 2))>, Mat23>);

  EXPECT_TRUE(is_near(to_Euclidean(TypedMatrix<C2, Axes<3>, M23> {1, 2, 3, M_PI*7/3, M_PI*13/6, -M_PI*7/4}),
    EuclideanMean<C2, M33> {1, 2, 3,
                            0.5, std::sqrt(3)/2, M_SQRT2/2,
                            std::sqrt(3)/2, 0.5, M_SQRT2/2}));

  EXPECT_TRUE(is_near(to_diagonal(Mat21 {2, 3}).base_matrix(), Mat22 {2, 0, 0, 3}));
  static_assert(is_diagonal_v<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(is_typed_matrix_v<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(to_diagonal(Mat21 {2, 3}))>::ColumnCoefficients, C2>);

  EXPECT_TRUE(is_near(transpose(Mat23 {1, 2, 3, 4, 5, 6}).base_matrix(), Mat32 {1, 4, 2, 5, 3, 6}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(Mat23 {1, 2, 3, 4, 5, 6})))>, Mat32>);

  EXPECT_TRUE(is_near(adjoint(Mat23 {1, 2, 3, 4, 5, 6}).base_matrix(), Mat32 {1, 4, 2, 5, 3, 6}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(Mat23 {1, 2, 3, 4, 5, 6})))>, Mat32>);

  EXPECT_NEAR(determinant(Mat22 {1, 2, 3, 4}), -2, 1e-6);

  EXPECT_NEAR(trace(Mat22 {1, 2, 3, 4}), 5, 1e-6);

  EXPECT_TRUE(is_near(solve(Mat22 {9., 3, 3, 10}, Mat21 {15, 23}), Mat21 {1, 2}));

  EXPECT_TRUE(is_near(reduce_columns(TypedMatrix<C2, Axes<3>, M23> {1, 2, 3, 4, 5, 6}), Mat21 {2, 5}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(TypedMatrix<C2, Axes<3>, M23> {1, 2, 3, 4, 5, 6})), Mat22 {14, 32, 32, 77}));

  EXPECT_TRUE(is_near(square(QR_decomposition(TypedMatrix<Axes<3>, C2, M32> {1, 4, 2, 5, 3, 6})), Mat22 {14, 32, 32, 77}));

  Mat23 m = Mat23::zero();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat23>(0.0, 0.7)) / (i + 1);
  }
  Mat23 offset = {1, 1, 1, 1, 1, 1};
  EXPECT_TRUE(is_near(m + offset, offset, 0.1));
  EXPECT_FALSE(is_near(m + offset, offset, 1e-6));
}


TEST_F(typed_matrix_tests, TypedMatrix_blocks)
{
  using Mat22x = TypedMatrix<C2, Axes<2>, M22>;
  EXPECT_TRUE(is_near(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}), Mat32 {1, 2, 3, 4, 5, 6}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}))>::RowCoefficients, C3>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}))>::ColumnCoefficients, C2>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}))>::ColumnCoefficients, C3>);

  EXPECT_TRUE(is_near(split_vertical(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C2, Axis>(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {Mat22 {1, 2, 3, 4}, Mat12 {5, 6}}));
  EXPECT_TRUE(is_near(split_horizontal<C2, Axis>(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {Mat22 {1, 2, 4, 5}, Mat21 {3, 6}}));
  EXPECT_TRUE(is_near(split_vertical<Axis, Angle>(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {Mat12 {1, 2}, Mat12 {3, 4}}));
  EXPECT_TRUE(is_near(split_horizontal<Axis, Angle>(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {Mat21 {1, 4}, Mat21 {2, 5}}));

  EXPECT_TRUE(is_near(column(Mat22x {1, 2, 3, 4}, 0), Mean{1., 3}));
  EXPECT_TRUE(is_near(column(Mat22x {1, 2, 3, 4}, 1), Mean{2., 4}));

  EXPECT_TRUE(is_near(column<0>(Mat22 {1, 2, 3, 4}), Mean{1., 3}));
  EXPECT_TRUE(is_near(column<1>(Mat22 {1, 2, 3, 4}), Mean{2., 4}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(column<0>(Mat22 {1, 2, 3, 4}))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(column<1>(Mat22 {1, 2, 3, 4}))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(column<0>(Mat22 {1, 2, 3, 4}))>::ColumnCoefficients, Axis>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(column<1>(Mat22 {1, 2, 3, 4}))>::ColumnCoefficients, Angle>);

  auto m = Mat22x {1, 2, 3, 4};
  EXPECT_TRUE(is_near(apply_columnwise(m, [](auto& col){ col *= 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise(m, [](auto& col, std::size_t i){ col *= i; }), Mat22 {0, 4, 0, 8}));

  EXPECT_TRUE(is_near(apply_columnwise(Mat22x {1, 2, 3, 4}, [](auto col){ return col * 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22x {1, 2, 3, 4}, [](const auto col){ return col * 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22x {1, 2, 3, 4}, [](auto&& col){ return col * 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22x {1, 2, 3, 4}, [](const auto& col){ return col * 2; }), Mat22 {2, 4, 6, 8}));

  EXPECT_TRUE(is_near(apply_columnwise(Mat22x {1, 2, 3, 4}, [](auto col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22x {1, 2, 3, 4}, [](const auto col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22x {1, 2, 3, 4}, [](auto&& col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22x {1, 2, 3, 4}, [](const auto& col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4}));

  EXPECT_TRUE(is_near(apply_columnwise<2>([] { return TypedMatrix<C2, Angle> {1., 2}; }), Mat22 {1, 1, 2, 2}));
  EXPECT_TRUE(is_near(apply_columnwise<2>([](std::size_t i){ return TypedMatrix<C2, Angle> {i + 1., 2*i + 1}; }), Mat22 {1, 2, 1, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<TypedMatrix<C2, Angle>()>()))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<TypedMatrix<C2, Angle>()>()))>::ColumnCoefficients, Coefficients<Angle, Angle>>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<TypedMatrix<C2, Angle>(std::size_t)>()))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<TypedMatrix<C2, Angle>(std::size_t)>()))>::ColumnCoefficients, Coefficients<Angle, Angle>>);

  auto n = Mat22x {1, 2, 3, 4};
  EXPECT_TRUE(is_near(apply_coefficientwise(n, [](auto& x){ x *= 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_coefficientwise(n, [](auto& x, std::size_t i, std::size_t j){ x += i + j; }), Mat22 {2, 5, 7, 10}));

  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4}, [](auto x){ return x + 1; }), Mat22 {2, 3, 4, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4}, [](const auto x){ return x + 1; }), Mat22 {2, 3, 4, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4}, [](auto&& x){ return x + 1; }), Mat22 {2, 3, 4, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4}, [](const auto& x){ return x + 1; }), Mat22 {2, 3, 4, 5}));

  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4}, [](auto x, std::size_t i, std::size_t j){ return x + i + j; }), Mat22 {1, 3, 4, 6}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4}, [](const auto x, std::size_t i, std::size_t j){ return x + i + j; }), Mat22 {1, 3, 4, 6}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4}, [](auto&& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat22 {1, 3, 4, 6}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4}, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat22 {1, 3, 4, 6}));

  EXPECT_TRUE(is_near(apply_coefficientwise<Mat22>([] { return 2; }), Mat22 {2, 2, 2, 2}));
  EXPECT_TRUE(is_near(apply_coefficientwise<Mat22>([](std::size_t i, std::size_t j){ return 1 + i + j; }), Mat22 {1, 2, 2, 3}));
}


TEST_F(typed_matrix_tests, TypedMatrix_arithmetic)
{
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8, 8, 8, 8}));
  EXPECT_TRUE(is_near(TypedMatrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} + Mean<C3, M32> {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8 - 2*M_PI, 8 - 2*M_PI, 8, 8}));
  EXPECT_TRUE(is_near(TypedMatrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8, 8, 8, 8}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, C2>);
  static_assert(is_mean_v<decltype(TypedMatrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} + Mean<C3, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(is_Euclidean_mean_v<decltype(TypedMatrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 0, -2, -4}));
  EXPECT_TRUE(is_near(TypedMatrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - Mean<C3, M32> {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 0, -2, -4}));
  EXPECT_TRUE(is_near(TypedMatrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 0, -2, -4}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, C2>);
  static_assert(is_mean_v<decltype(TypedMatrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - Mean<C3, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(is_Euclidean_mean_v<decltype(TypedMatrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3, 4, 5, 6}));
  static_assert(is_typed_matrix_v<decltype(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(is_typed_matrix_v<decltype(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(is_typed_matrix_v<decltype(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6}, Mat23 {9, 12, 15, 19, 26, 33}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, C3>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 2} * Mean<C2, M23> {1, 2, 3, 3, 2, 1}, Mat23 {7, 6, 5, 9-2*M_PI, 10-4*M_PI, 11-4*M_PI}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, Axes<3>>);
  static_assert(is_mean_v<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(TypedMatrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}, Mat23 {9, 12, 15, 19, 26, 33}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, Axes<3>>);
  static_assert(not is_Euclidean_mean_v<decltype(TypedMatrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>);
  static_assert(not is_mean_v<decltype(TypedMatrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(-Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {-1, -2, -3, -4, -5, -6}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} == Mat22 {1, 2, 3, 4}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} != Mat22 {1, 2, 2, 4}));
  EXPECT_FALSE((Mat22 {1, 2, 3, 4} == TypedMatrix<C2, Axes<2>, M22> {1, 2, 3, 4}));
}


TEST_F(typed_matrix_tests, TypedMatrix_references_axis)
{
  using V = TypedMatrix<C3, C3, M33>;
  V v1 {1., 2, 3,
        2, 4, -6,
        3, 6, -3};
  TypedMatrix<C3, C3, M33&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(0, 1) = 5.2;
  EXPECT_EQ(v1(0,1), 5.2);
  TypedMatrix<C3, C3, M33&&> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  TypedMatrix<C3, C3, const M33&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  TypedMatrix<C3, C3, M33> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}

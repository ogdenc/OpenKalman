/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "matrices.hpp"

using namespace OpenKalman;

using M12 = Eigen::Matrix<double, 1, 2>;
using M21 = Eigen::Matrix<double, 2, 1>;
using M22 = Eigen::Matrix<double, 2, 2>;
using M23 = Eigen::Matrix<double, 2, 3>;
using M32 = Eigen::Matrix<double, 3, 2>;
using M33 = Eigen::Matrix<double, 3, 3>;
using I22 = IdentityMatrix<M22>;
using Z22 = ZeroMatrix<M22>;
using C2 = Coefficients<Axis, angle::Radians>;
using C3 = Coefficients<Axis, angle::Radians, Axis>;
using Mat12 = Matrix<Axis, C2, M12>;
using Mat21 = Matrix<C2, Axis, M21>;
using Mat22 = Matrix<C2, C2, M22>;
using Mat23 = Matrix<C2, C3, M23>;
using Mat32 = Matrix<C3, C2, M32>;
using Mat33 = Matrix<C3, C3, M33>;

inline I22 i22 = M22::Identity();
inline Z22 z22 = Z22();
inline auto covi22 = Covariance<C2, I22>(i22);
inline auto covz22 = Covariance<C2, Z22>(z22);
inline auto sqcovi22 = SquareRootCovariance<C2, I22>(i22);
inline auto sqcovz22 = SquareRootCovariance<C2, Z22>(z22);


TEST_F(matrices, TypedMatrix_class)
{
  // Default constructor and Eigen3 construction
  Mat23 mat23a;
  mat23a << 1, 2, 3, 4, 5, 6;
  EXPECT_TRUE(is_near(mat23a, Mat23 {1, 2, 3, 4, 5, 6}));

  // Copy constructor
  Mat23 mat23b = const_cast<const Mat23&>(mat23a);
  EXPECT_TRUE(is_near(mat23b, Mat23 {1, 2, 3, 4, 5, 6}));

  // Move constructor
  auto xa = Mat23 {6, 5, 4, 3, 2, 1};
  Mat23 mat23c(std::move(xa));
  EXPECT_TRUE(is_near(mat23c, Mat23 {6, 5, 4, 3, 2, 1}));

  // Convert from different covariance types
  Matrix<C2, Axes<3>, M23> mat23_x1(Mean<C2, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x1, Mat23 {1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi}));
  Matrix<Axes<2>, Axes<3>, M23> mat23_x2(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x2, Mat23 {1, 2, 3, 4, 5, 6}));
  Matrix<C2, Axes<3>, M23> mat23_x3(EuclideanMean<C2, M33> {
    1, 2, 3,
    0.5, std::sqrt(3)/2, sqrt2/2,
    std::sqrt(3)/2, 0.5, sqrt2/2});
  EXPECT_TRUE(is_near(mat23_x3, Mat23 {1, 2, 3, pi/3, pi/6, pi/4}));

  // Construct from a regular matrix
  Mat23 mat23d(make_native_matrix<M23>(1, 2, 3, 4, 5, 6));
  EXPECT_TRUE(is_near(mat23d, Mat23 {1, 2, 3, 4, 5, 6}));

  // Convert from a compatible covariance
  Mat22 mat22a_1(Covariance<C2, SelfAdjointMatrix<M22, TriangleType::lower>> {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mat22a_1, Mat22 {9, 3, 3, 10}));
  Mat22 mat22a_2(Covariance<C2, SelfAdjointMatrix<M22, TriangleType::upper>> {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mat22a_2, Mat22 {9, 3, 3, 10}));
  Mat22 mat22a_3(Covariance<C2, SelfAdjointMatrix<M22, TriangleType::lower>> {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mat22a_3, Mat22 {9, 3, 3, 10}));
  Mat22 mat22a_4(Covariance<C2, TriangularMatrix<M22, TriangleType::upper>> {9, 3, 3, 10});
  EXPECT_TRUE(is_near(mat22a_4, Mat22 {9, 3, 3, 10}));
  Mat22 mat22a_5(Covariance<C2, DiagonalMatrix<M21>> {9, 10});
  EXPECT_TRUE(is_near(mat22a_5, Mat22 {9, 0, 0, 10}));
  Mat22 mat22a_6(covi22);
  EXPECT_TRUE(is_near(mat22a_6, Mat22 {1, 0, 0, 1}));
  Mat22 mat22a_7(covz22);
  EXPECT_TRUE(is_near(mat22a_7, Mat22 {0, 0, 0, 0}));

  // Convert from a compatible square root covariance
  Mat22 mat22b_1(SquareRootCovariance<C2, SelfAdjointMatrix<M22, TriangleType::lower>> {3, 0, 1, 3});
  EXPECT_TRUE(is_near(mat22b_1, Mat22 {3, 0, 1, 3}));
  Mat22 mat22b_2(SquareRootCovariance<C2, SelfAdjointMatrix<M22, TriangleType::upper>> {3, 1, 0, 3});
  EXPECT_TRUE(is_near(mat22b_2, Mat22 {3, 1, 0, 3}));
  Mat22 mat22b_3(SquareRootCovariance<C2, SelfAdjointMatrix<M22, TriangleType::lower>> {3, 0, 1, 3});
  EXPECT_TRUE(is_near(mat22b_3, Mat22 {3, 0, 1, 3}));
  Mat22 mat22b_4(SquareRootCovariance<C2, TriangularMatrix<M22, TriangleType::upper>> {3, 1, 0, 3});
  EXPECT_TRUE(is_near(mat22b_4, Mat22 {3, 1, 0, 3}));
  Mat22 mat22b_5(SquareRootCovariance<C2, DiagonalMatrix<M21>> {3, 4});
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
  auto xb = Mat23 {3, 4, 5, 6, 7, 8};
  mat23c = std::move(xb);
  EXPECT_TRUE(is_near(mat23c, Mat23 {3, 4, 5, 6, 7, 8}));

  // assign from different covariance types
  mat23_x1 = Mean<C2, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x1, Mat23 {6, 5, 4, 3, 2, 1}));
  mat23_x2 = EuclideanMean<Axes<2>, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x2, Mat23 {6, 5, 4, 3, 2, 1}));
  mat23_x3 = EuclideanMean<C2, M33> {
    3, 2, 1,
    std::sqrt(3)/2, sqrt2/2, 0.5,
    0.5, sqrt2/2, std::sqrt(3)/2};
  EXPECT_TRUE(is_near(mat23_x3, Mat23 {3, 2, 1, pi/6, pi/4, pi/3}));

  // assign from a regular matrix
  mat23e = make_native_matrix<M23>(3, 4, 5, 6, 7, 8);

  // Assign from a list of coefficients (via move assignment operator)
  mat23e = {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23e, Mat23 {6, 5, 4, 3, 2, 1}));

  // Increment
  mat23a += Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23a, Mat23 {2, 4, 6, 8, 10, 12}));

  // Increment with a stochastic value
  mat23_x1 = {1, 1, 1, pi, pi, pi};
  GaussianDistribution g = {Mat21 {0, 0}, sqcovi22 * 0.1};
  Matrix<C2, Axes<3>, M23> diff = {1, 1, 1, 1, 1, 1};
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
  mat23_x1 = {1, 1, 1, pi, pi, pi};
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
}


TEST_F(matrices, TypedMatrix_subscripts)
{
  static_assert(element_gettable<Mat23, 2>);
  static_assert(not element_gettable<Mat23, 1>);
  static_assert(element_gettable<const Mat23, 2>);
  static_assert(not element_gettable<const Mat23, 1>);
  static_assert(element_gettable<Mat21, 2>);
  static_assert(element_gettable<Mat21, 1>);
  static_assert(element_gettable<const Mat21, 2>);
  static_assert(element_gettable<const Mat21, 1>);
  static_assert(element_gettable<Matrix<C3, C2, M32>, 2>);
  static_assert(not element_gettable<Matrix<C3, C2, M32>, 1>);
  static_assert(element_gettable<Matrix<C2, Axis, M21>, 2>);
  static_assert(element_gettable<Matrix<C2, Axis, M21>, 1>);

  static_assert(element_settable<Mat23, 2>);
  static_assert(not element_settable<Mat23, 1>);
  static_assert(not element_settable<const Mat23, 2>);
  static_assert(not element_settable<const Mat23, 1>);
  static_assert(element_settable<Mat21, 2>);
  static_assert(element_settable<Mat21, 1>);
  static_assert(not element_settable<const Mat21, 2>);
  static_assert(not element_settable<const Mat21, 1>);
  static_assert(element_settable<Matrix<C3, C2, M32>, 2>);
  static_assert(not element_settable<Matrix<C3, C2, M32>, 1>);
  static_assert(not element_settable<Matrix<C3, C2, const M32>, 2>);
  static_assert(not element_settable<Matrix<C3, C2, const M32>, 1>);
  static_assert(element_settable<Matrix<C2, Axis, M21>, 2>);
  static_assert(element_settable<Matrix<C2, Axis, M21>, 1>);
  static_assert(not element_settable<Matrix<C2, Axis, const M21>, 2>);
  static_assert(not element_settable<Matrix<C2, Axis, const M21>, 1>);

  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 0), 1, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 1), 2, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 2), 3, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 0), 4, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 1), 5, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 2), 6, 1e-6);
}


TEST_F(matrices, TypedMatrix_deduction_guides)
{
  auto a = make_native_matrix<M23>(1, 2, 3, 4, 5, 6);
  EXPECT_TRUE(is_near(Matrix(a), a));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(a))>::RowCoefficients, Axes<2>>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(a))>::ColumnCoefficients, Axes<3>>);

  //auto b1 = Mat23 {1, 2, 3, 4, 5, 6};
  //EXPECT_TRUE(is_near(Matrix(b1), a));
  //static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(b1))>::RowCoefficients, C2>);
  //static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(b1))>::ColumnCoefficients, C3>);

  auto b2 = Mean<C2, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(Matrix(b2), (M23() << 1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi).finished()));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(b2))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(b2))>::ColumnCoefficients, Axes<3>>);

  auto b3 = EuclideanMean<C2, M33> {1, 2, 3,
                                    0.5, std::sqrt(3)/2, sqrt2/2,
                                    std::sqrt(3)/2, 0.5, sqrt2/2};
  EXPECT_TRUE(is_near(Matrix(b3), Mat23 {1, 2, 3, pi/3, pi/6, pi/4}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(b3))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(b3))>::ColumnCoefficients, Axes<3>>);

  auto c = Covariance<C2, SelfAdjointMatrix<M22, TriangleType::lower>> {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Matrix(c), Mat22 {9, 3, 3, 10}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(c))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix(c))>::ColumnCoefficients, C2>);
}


TEST_F(matrices, TypedMatrix_make_functions)
{
  auto a = make_native_matrix<M23>(1, 2, 3, 4, 5, 6);
  EXPECT_TRUE(is_near(make_Matrix<C2, C3>(a), a));
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix<C2, C3>(a))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix<C2, C3>(a))>::ColumnCoefficients, C3>);
  EXPECT_TRUE(is_near(make_Matrix<C2>(a), a));
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix<C2>(a))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix<C2>(a))>::ColumnCoefficients, Axes<3>>);

  auto b = Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(make_Matrix(b), a));
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix(b))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix(b))>::ColumnCoefficients, C3>);

  auto c = Covariance<C2, SelfAdjointMatrix<M22, TriangleType::lower>> {9, 3, 3, 10};
  EXPECT_TRUE(is_near(make_Matrix(c), Mat22 {9, 3, 3, 10}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix(c))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix(c))>::ColumnCoefficients, C2>);

  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix<C2, C3, M23>())>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix<C2, C3, M23>())>::ColumnCoefficients, C3>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix<M23>())>::RowCoefficients, Axes<2>>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_Matrix<M23>())>::ColumnCoefficients, Axes<3>>);
}


TEST_F(matrices, TypedMatrix_traits)
{
  static_assert(typed_matrix<Mat23>);
  static_assert(not mean<Mat23>);
  static_assert(not euclidean_mean<Mat23>);
  static_assert(not euclidean_mean<Matrix<C2, Axes<2>, I22>>);
  static_assert(not euclidean_transformed<Mat23>);
  static_assert(not euclidean_transformed<Matrix<C2, Axes<2>, I22>>);
  static_assert(not wrapped_mean<Mat23>);
  static_assert(not wrapped_mean<Matrix<C2, Axes<2>, I22>>);
  static_assert(not column_vector<Mat23>);
  static_assert(column_vector<Matrix<C2, Axes<2>, I22>>);

  static_assert(not identity_matrix<Mat23>);
  static_assert(identity_matrix<Matrix<C2, C2, I22>>);
  static_assert(not identity_matrix<Matrix<C2, Axes<2>, I22>>);
  static_assert(not zero_matrix<Mat23>);
  static_assert(zero_matrix<Matrix<C2, C2, Z22>>);
  static_assert(zero_matrix<Matrix<C2, C3, ZeroMatrix<M23>>>);

  EXPECT_TRUE(is_near(MatrixTraits<Mat23>::make(
    make_native_matrix<double, 2, 3>(1, 2, 3, 4, 5, 6)).base_matrix(), Mat23 {1, 2, 3, 4, 5, 6}));
  EXPECT_TRUE(is_near(MatrixTraits<Mat23>::zero(), Eigen::Matrix<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<Mat22>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
}


TEST_F(matrices, TypedMatrix_overloads)
{
  EXPECT_TRUE(is_near(base_matrix(Mat23 {1, 2, 3, 4, 5, 6}), Mat23 {1, 2, 3, 4, 5, 6}));

  EXPECT_TRUE(is_near(make_native_matrix(Mat23 {1, 2, 3, 4, 5, 6}), Mat23 {1, 2, 3, 4, 5, 6}));

  EXPECT_TRUE(is_near(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6} * 2), Mat23 {2, 4, 6, 8, 10, 12}));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6} * 2))>, Mat23>);

  EXPECT_TRUE(is_near(to_Euclidean(Matrix<C2, Axes<3>, M23> {1, 2, 3, pi*7/3, pi*13/6, -pi*7/4}),
    EuclideanMean<C2, M33> {1, 2, 3,
                            0.5, std::sqrt(3)/2, sqrt2/2,
                            std::sqrt(3)/2, 0.5, sqrt2/2}));

  EXPECT_TRUE(is_near(to_diagonal(Mat21 {2, 3}).base_matrix(), Mat22 {2, 0, 0, 3}));
  static_assert(diagonal_matrix<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(typed_matrix<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(to_diagonal(Mat21 {2, 3}))>::ColumnCoefficients, C2>);

  EXPECT_TRUE(is_near(transpose(Mat23 {1, 2, 3, 4, 5, 6}).base_matrix(), Mat32 {1, 4, 2, 5, 3, 6}));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(transpose(Mat23 {1, 2, 3, 4, 5, 6})))>, Mat32>);

  EXPECT_TRUE(is_near(adjoint(Mat23 {1, 2, 3, 4, 5, 6}).base_matrix(), Mat32 {1, 4, 2, 5, 3, 6}));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(adjoint(Mat23 {1, 2, 3, 4, 5, 6})))>, Mat32>);

  EXPECT_NEAR(determinant(Mat22 {1, 2, 3, 4}), -2, 1e-6);

  EXPECT_NEAR(trace(Mat22 {1, 2, 3, 4}), 5, 1e-6);

  EXPECT_TRUE(is_near(solve(Mat22 {9., 3, 3, 10}, Mat21 {15, 23}), Mat21 {1, 2}));

  EXPECT_TRUE(is_near(reduce_columns(Matrix<C2, Axes<3>, M23> {1, 2, 3, 4, 5, 6}), Mat21 {2, 5}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(Matrix<C2, Axes<3>, M23> {1, 2, 3, 4, 5, 6})), Mat22 {14, 32, 32, 77}));

  EXPECT_TRUE(is_near(square(QR_decomposition(Matrix<Axes<3>, C2, M32> {1, 4, 2, 5, 3, 6})), Mat22 {14, 32, 32, 77}));

  using N = std::normal_distribution<double>::param_type;
  Mat23 m = Mat23::zero();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat23>(N {1.0, 0.1}, 2.0, N {3.0, 0.1}, N {4.0, 0.1}, 5.0, N {6.0, 0.1})) / (i + 1);
  }
  Mat23 offset = {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-8));
}


TEST_F(matrices, TypedMatrix_blocks)
{
  using Mat22x = Matrix<C2, Axes<2>, M22>;
  EXPECT_TRUE(is_near(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}), Mat32 {1, 2, 3, 4, 5, 6}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}))>::RowCoefficients, C3>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}))>::ColumnCoefficients, C2>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}))>::ColumnCoefficients, C3>);

  EXPECT_TRUE(is_near(concatenate_diagonal(Mat12 {1, 2}, Mat21 {3, 4}), Mat33 {1, 2, 0, 0, 0, 3, 0, 0, 4}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(concatenate_diagonal(Mat12 {1, 2}, Mat21 {3, 4}))>::RowCoefficients, Coefficients<Axis, Axis, angle::Radians>>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(concatenate_diagonal(Mat12 {1, 2}, Mat21 {3, 4}))>::ColumnCoefficients, Coefficients<Axis, angle::Radians, Axis>>);

  EXPECT_TRUE(is_near(split_vertical(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C2, Axis>(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {Mat22 {1, 2, 3, 4}, Mat12 {5, 6}}));
  EXPECT_TRUE(is_near(split_horizontal<C2, Axis>(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {Mat22 {1, 2, 4, 5}, Mat21 {3, 6}}));
  EXPECT_TRUE(is_near(split_vertical<Axis, angle::Radians>(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {Mat12 {1, 2}, Mat12 {3, 4}}));
  EXPECT_TRUE(is_near(split_horizontal<Axis, angle::Radians>(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {Mat21 {1, 4}, Mat21 {2, 5}}));

  EXPECT_TRUE(is_near(column(Mat22x {1, 2, 3, 4}, 0), Mean{1., 3}));
  EXPECT_TRUE(is_near(column(Mat22x {1, 2, 3, 4}, 1), Mean{2., 4}));

  EXPECT_TRUE(is_near(column<0>(Mat22 {1, 2, 3, 4}), Mean{1., 3}));
  EXPECT_TRUE(is_near(column<1>(Mat22 {1, 2, 3, 4}), Mean{2., 4}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(column<0>(Mat22 {1, 2, 3, 4}))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(column<1>(Mat22 {1, 2, 3, 4}))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(column<0>(Mat22 {1, 2, 3, 4}))>::ColumnCoefficients, Axis>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(column<1>(Mat22 {1, 2, 3, 4}))>::ColumnCoefficients, angle::Radians>);

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

  EXPECT_TRUE(is_near(apply_columnwise<2>([] { return Matrix<C2, angle::Radians> {1., 2}; }), Mat22 {1, 1, 2, 2}));
  EXPECT_TRUE(is_near(apply_columnwise<2>([](std::size_t i){ return Matrix<C2, angle::Radians> {i + 1., 2*i + 1}; }), Mat22 {1, 2, 1, 3}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Matrix<C2, angle::Radians>()>()))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Matrix<C2, angle::Radians>()>()))>::ColumnCoefficients, Coefficients<angle::Radians, angle::Radians>>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Matrix<C2, angle::Radians>(std::size_t)>()))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Matrix<C2, angle::Radians>(std::size_t)>()))>::ColumnCoefficients, Coefficients<angle::Radians, angle::Radians>>);

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


TEST_F(matrices, TypedMatrix_arithmetic)
{
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8, 8, 8, 8}));
  EXPECT_TRUE(is_near(Matrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} + Mean<C3, M32> {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8, 8 - 2*pi, 8, 8}));
  EXPECT_TRUE(is_near(Matrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8, 8, 8, 8}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, C2>);
  static_assert(typed_matrix<decltype(Matrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} + Mean<C3, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(Matrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 0, -2, -4}));
  EXPECT_TRUE(is_near(Matrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - Mean<C3, M32> {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 2*pi, -2, -4}));
  EXPECT_TRUE(is_near(Matrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 0, -2, -4}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, C2>);
  static_assert(typed_matrix<decltype(Matrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - Mean<C3, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(Matrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3, 4, 5, 6}));
  static_assert(typed_matrix<decltype(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(typed_matrix<decltype(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(typed_matrix<decltype(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6}, Mat23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, C3>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 2} * Mean<C2, M23> {1, 2, 3, 3, 2, 1}, Mat23 {7, 6, 5, 9, 10, 11}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, Axes<3>>);
  static_assert(typed_matrix<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Matrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}, Mat23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Matrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, Axes<3>>);
  static_assert(typed_matrix<decltype(Matrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(Matrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(-Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {-1, -2, -3, -4, -5, -6}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} == Mat22 {1, 2, 3, 4}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} != Mat22 {1, 2, 2, 4}));
  EXPECT_FALSE((Mat22 {1, 2, 3, 4} == Matrix<C2, Axes<2>, M22> {1, 2, 3, 4}));
}


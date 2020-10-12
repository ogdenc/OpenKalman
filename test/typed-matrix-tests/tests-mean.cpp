/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "typed_matrix_tests.h"

#pragma clang diagnostic ignored "-Wpessimizing-move"

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
using Mat12 = Mean<Axis, M12>;
using Mat21 = Mean<C2, M21>;
using Mat22 = Mean<C2, M22>;
using Mat23 = Mean<C2, M23>;
using Mat32 = Mean<C3, M32>;
using Mat33 = Mean<C3, M33>;
using TMat22 = TypedMatrix<C2, Axes<2>, M22>;
using TMat23 = TypedMatrix<C2, Axes<3>, M23>;
using TMat32 = TypedMatrix<C3, Axes<2>, M32>;
using EMat23 = EuclideanMean<C2, M33>;

using SA2l = EigenSelfAdjointMatrix<M22, TriangleType::lower>;
using SA2u = EigenSelfAdjointMatrix<M22, TriangleType::upper>;
using T2l = EigenTriangularMatrix<M22, TriangleType::lower>;
using T2u = EigenTriangularMatrix<M22, TriangleType::upper>;

inline I22 i22 = M22::Identity();
inline Z22 z22 = Z22();
inline auto covi22 = Covariance<C2, I22>(i22);
inline auto covz22 = Covariance<C2, Z22>(z22);
inline auto sqcovi22 = SquareRootCovariance<C2, I22>(i22);
inline auto sqcovz22 = SquareRootCovariance<C2, Z22>(z22);


TEST_F(typed_matrix_tests, Mean_class)
{
  // Default constructor and Eigen3 construction
  Mat23 mat23a;
  mat23a << 1, 2, 3, 4, 5, 6;
  EXPECT_TRUE(is_near(mat23a, TMat23 {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI}));

  // Copy constructor
  Mat23 mat23b = const_cast<const Mat23&>(mat23a);
  EXPECT_TRUE(is_near(mat23b, TMat23 {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI}));

  // Move constructor
  Mat23 mat23c(std::move(Mat23 {6, 5, 4, 3, 2, 1}));
  EXPECT_TRUE(is_near(mat23c, TMat23 {6, 5, 4, 3, 2, 1}));

  // Convert from different covariance types
  Mat23 mat23_x1(TypedMatrix<C2, Axes<3>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x1, Mat23 {1, 2, 3, 4, 5, 6}));
  Mean<Axes<2>, M23> mat23_x2(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x2, Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}));
  Mat23 mat23_x3(EuclideanMean<C2, M33> {
    1, 2, 3,
    0.5, std::sqrt(3)/2, M_SQRT2/2,
    std::sqrt(3)/2, 0.5, M_SQRT2/2});
  EXPECT_TRUE(is_near(mat23_x3, Mat23 {1, 2, 3, M_PI/3, M_PI/6, M_PI/4}));

  // Construct from a typed matrix base
  Mat23 mat23d((M23() << 1, 2, 3, 4, 5, 6).finished());
  EXPECT_TRUE(is_near(mat23d, TMat23 {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI}));

  // Construct from a list of coefficients
  Mat23 mat23e {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23e, TMat23 {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI}));

  // Copy assignment
  mat23c = mat23b;
  EXPECT_TRUE(is_near(mat23c, TMat23 {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI}));

  // Move assignment
  mat23c = std::move(Mat23 {3, 4, 5, 6, 7, 8});
  EXPECT_TRUE(is_near(mat23c, TMat23 {3, 4, 5, 6-2*M_PI, 7-2*M_PI, 8-2*M_PI}));

  // assign from different covariance types
  mat23_x1 = TypedMatrix<C2, Axes<3>, M23> {6, 5, 4, 3, 2, 1};
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
  mat23_x1 += TMat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23_x1, TMat23 {7, 7, 7, 7-2*M_PI, 7-2*M_PI, 7-2*M_PI}));
  mat23a += Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23a, TMat23 {2, 4, 6, 8-2*M_PI, 10-4*M_PI, 12-4*M_PI}));

  // Increment with a stochastic value
  mat23b = {1, 1, 1, M_PI, M_PI, M_PI};
  GaussianDistribution g = {Mat21 {0, 0}, sqcovi22 * 0.1};
  TMat23 diff = {1, 1, 1, 1, 1, 1};
  for (int i=0; i<100; i++)
  {
    Mat23 oldmat23b = mat23b;
    mat23b += g;
    diff += (mat23b - oldmat23b) / 100;
  }
  EXPECT_TRUE(is_near(diff, TMat23 {1, 1, 1, 1, 1, 1}, 0.1));
  EXPECT_FALSE(is_near(diff, TMat23 {1, 1, 1, 1, 1, 1}, 1e-6));

  // Decrement
  mat23_x1 -= TMat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23_x1, TMat23 {6, 5, 4, 3, 2, 1}));
  mat23a -= Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23a, TMat23 {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI}));

  // Decrement with a stochastic value
  mat23b = {1, 1, 1, M_PI, M_PI, M_PI};
  diff = {1, 1, 1, 1, 1, 1};
  for (int i=0; i<100; i++)
  {
    Mat23 oldmat23b = mat23b;
    mat23b -= g;
    diff += (mat23b - oldmat23b) / 100;
  }
  EXPECT_TRUE(is_near(diff, TMat23 {1, 1, 1, 1, 1, 1}, 0.1));
  EXPECT_FALSE(is_near(diff, TMat23 {1, 1, 1, 1, 1, 1}, 1e-6));

  // Scalar multiplication
  mat23a *= 2;
  EXPECT_TRUE(is_near(mat23a, TMat23 {2, 4, 6, 8-2*M_PI, 10-4*M_PI, 12-4*M_PI}));

  // Scalar division
  mat23a /= 2;
  EXPECT_TRUE(is_near(mat23a, TMat23 {1, 2, 3, 4-M_PI, 5-2*M_PI, 6-2*M_PI}));

  // Scalar multiplication, zero
  mat23a *= 0;
  EXPECT_TRUE(is_near(mat23a, M23::Zero()));

  // Zero
  EXPECT_TRUE(is_near(Mat23::zero(), M23::Zero()));

  // Identity
  EXPECT_TRUE(is_near(Mat22::identity(), M22::Identity()));
}


TEST_F(typed_matrix_tests, Mean_subscripts)
{
  static_assert(is_element_gettable_v<Mat23, 2>);
  static_assert(not is_element_gettable_v<Mat23, 1>);
  static_assert(is_element_gettable_v<const Mat23, 2>);
  static_assert(not is_element_gettable_v<const Mat23, 1>);
  static_assert(is_element_gettable_v<Mat21, 2>);
  static_assert(is_element_gettable_v<Mat21, 1>);
  static_assert(is_element_gettable_v<const Mat21, 2>);
  static_assert(is_element_gettable_v<const Mat21, 1>);
  static_assert(is_element_gettable_v<TypedMatrix<C3, C2, M32>, 2>);
  static_assert(not is_element_gettable_v<TypedMatrix<C3, C2, M32>, 1>);
  static_assert(is_element_gettable_v<TypedMatrix<C2, Axis, M21>, 2>);
  static_assert(is_element_gettable_v<TypedMatrix<C2, Axis, M21>, 1>);

  static_assert(is_element_settable_v<Mat23, 2>);
  static_assert(not is_element_settable_v<Mat23, 1>);
  static_assert(not is_element_settable_v<const Mat23, 2>);
  static_assert(not is_element_settable_v<const Mat23, 1>);
  static_assert(is_element_settable_v<Mat21, 2>);
  static_assert(is_element_settable_v<Mat21, 1>);
  static_assert(not is_element_settable_v<const Mat21, 2>);
  static_assert(not is_element_settable_v<const Mat21, 1>);
  static_assert(is_element_settable_v<TypedMatrix<C3, C2, M32>, 2>);
  static_assert(not is_element_settable_v<TypedMatrix<C3, C2, M32>, 1>);
  static_assert(not is_element_settable_v<TypedMatrix<C3, C2, const M32>, 2>);
  static_assert(not is_element_settable_v<TypedMatrix<C3, C2, const M32>, 1>);
  static_assert(is_element_settable_v<TypedMatrix<C2, Axis, M21>, 2>);
  static_assert(is_element_settable_v<TypedMatrix<C2, Axis, M21>, 1>);
  static_assert(not is_element_settable_v<TypedMatrix<C2, Axis, const M21>, 2>);
  static_assert(not is_element_settable_v<TypedMatrix<C2, Axis, const M21>, 1>);

  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 0), 1, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 1), 2, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 2), 3, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 0), 4-2*M_PI, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 1), 5-2*M_PI, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 2), 6-2*M_PI, 1e-6);
}


TEST_F(typed_matrix_tests, Mean_deduction_guides)
{
  auto a = (M23() << 1, 2, 3, 4, 5, 6).finished();
  EXPECT_TRUE(is_near(Mean(a), a));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mean(a))>::RowCoefficients, Axes<2>>);

  auto b1 = Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(Mean(b1), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mean(b1))>::RowCoefficients, C2>);

  auto b2 = TypedMatrix<C2, Axes<3>, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(Mean(b2), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mean(b2))>::RowCoefficients, C2>);

  auto b3 = EuclideanMean<C2, M33> {1, 2, 3, std::sqrt(3)/2, 0.5, std::sqrt(2)/2, 0.5, std::sqrt(3)/2, std::sqrt(2)/2};
  EXPECT_TRUE(is_near(Mean(b3), Mat23 {1, 2, 3, M_PI/6, M_PI/3, M_PI/4}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mean(b3))>::RowCoefficients, C2>);
  static_assert(MatrixTraits<decltype(Mean(b3))>::dimension == 2);
}


TEST_F(typed_matrix_tests, Mean_make_functions)
{
  auto a = (M23() << 1, 2, 3, 4, 5, 6).finished();
  EXPECT_TRUE(is_near(make_Mean<C2>(a), Mat23{a}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Mean<C2>(a))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Mean<C2>(a))>::ColumnCoefficients, Axes<3>>);

  auto b = Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(make_Mean(b), Mat23{a}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Mean(b))>::RowCoefficients, C2>);

  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Mean<C2, M23>())>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(make_Mean<M23>())>::RowCoefficients, Axes<2>>);
}


TEST_F(typed_matrix_tests, Mean_traits)
{
  static_assert(is_typed_matrix_v<Mat23>);
  static_assert(is_mean_v<Mat23>);
  static_assert(not is_Euclidean_mean_v<Mat23>);
  static_assert(not is_Euclidean_transformed_v<Mat23>);
  static_assert(is_wrapped_v<Mat23>);
  static_assert(not is_wrapped_v<Mean<Axes<2>, I22>>);
  static_assert(is_column_vector_v<Mat23>);

  static_assert(not is_identity_v<Mat23>);
  static_assert(is_identity_v<Mean<Axes<2>, I22>>);
  static_assert(not is_identity_v<Mean<Axes<2>, M23>>);
  static_assert(not is_zero_v<Mat23>);
  static_assert(is_zero_v<Mean<C2, Z22>>);
  static_assert(is_zero_v<Mean<C2, EigenZero<M23>>>);

  EXPECT_TRUE(is_near(
    MatrixTraits<Mat23>::make((Eigen::Matrix<double, 2, 3>() << 1, 2, 3, 4, 5, 6).finished()).base_matrix(),
    Mean<Axes<2>, M23> {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6-2*M_PI}));
  EXPECT_TRUE(is_near(MatrixTraits<Mat23>::zero(), Eigen::Matrix<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<Mat22>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
}


TEST_F(typed_matrix_tests, Mean_overloads)
{
  auto w_4 = 4 - 2*M_PI;
  auto w_5 = 5 - 2*M_PI;
  auto w_6 = 6 - 2*M_PI;
  EXPECT_TRUE(is_near(Mean<C2, M23> {1, 2, 3, M_PI*7/3, M_PI*13/6, -M_PI*7/4},
    Mat23 {1, 2, 3, M_PI/3, M_PI/6, M_PI/4}));

  EXPECT_TRUE(is_near(base_matrix(Mat23 {1, 2, 3, 4, 5, 6}), TMat23 {1, 2, 3, w_4, w_5, w_6}));

  EXPECT_TRUE(is_near(strict_matrix(Mat23 {1, 2, 3, 4, 5, 6}), Mat23 {1, 2, 3, w_4, w_5, w_6}));

  EXPECT_TRUE(is_near(strict(Mat23 {1, 2, 3, 4, 5, 6} * 2), Mat23 {2, 4, 6, 8-2*M_PI, 10-4*M_PI, 12-4*M_PI}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(Mat23 {1, 2, 3, 4, 5, 6} * 2))>,
    Mean<C2, decltype(wrap_angles<C2>(std::declval<M23>()))>>);

  EXPECT_TRUE(is_near(to_Euclidean(
    Mean<C2, M23> {1, 2, 3, M_PI*7/3, M_PI*13/6, -M_PI*7/4}),
    EuclideanMean<C2, M33> {1, 2, 3,
                            0.5, std::sqrt(3)/2, M_SQRT2/2,
                            std::sqrt(3)/2, 0.5, M_SQRT2/2}));

  const auto m1 = make_Mean(-2., 5, 3);
  EXPECT_TRUE(is_near(to_Euclidean(m1), m1));

  using A3 = Coefficients<Angle, Axis, Angle>;
  const auto m2 = make_Mean<A3>(M_PI / 6, 5, -M_PI / 3);
  const auto x2 = (Eigen::Matrix<double, 5, 1> {} << std::sqrt(3) / 2, 0.5, 5, 0.5, -std::sqrt(3) / 2).finished();
  EXPECT_TRUE(is_near(to_Euclidean(m2).base_matrix(), x2));

  EXPECT_TRUE(is_near(to_diagonal(Mat21 {2, 3}).base_matrix(), Mat22 {2, 0, 0, 3}));
  static_assert(is_diagonal_v<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(is_typed_matrix_v<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(to_diagonal(Mat21 {2, 3}))>::ColumnCoefficients, C2>);

  EXPECT_TRUE(is_near(to_diagonal(Mat21 {5, 6}).base_matrix(), Mat22 {5, 0, 0, w_6}));
  static_assert(is_diagonal_v<decltype(to_diagonal(Mat21 {5, 6}))>);
  static_assert(is_typed_matrix_v<decltype(to_diagonal(Mat21 {5, 6}))>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(to_diagonal(Mat21 {5, 6}))>::ColumnCoefficients, C2>);

  EXPECT_TRUE(is_near(transpose(Mat23 {1, 2, 3, 4, 5, 6}).base_matrix(), TMat32 {1, w_4, 2, w_5, 3, w_6}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(transpose(Mat23 {1, 2, 3, 4, 5, 6})))>,
    TypedMatrix<Axes<3>, C2, M32>>);

  EXPECT_TRUE(is_near(adjoint(Mat23 {1, 2, 3, 4, 5, 6}).base_matrix(), TMat32 {1, w_4, 2, w_5, 3, w_6}));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(adjoint(Mat23 {1, 2, 3, 4, 5, 6})))>,
    TypedMatrix<Axes<3>, C2, M32>>);

  EXPECT_NEAR(determinant(Mat22 {1, 2, 3, 4}), w_4 - 6, 1e-6);

  EXPECT_NEAR(trace(Mat22 {1, 2, 3, 4}), 1 + w_4, 1e-6);

  EXPECT_TRUE(is_near(solve(Mean<Axes<2>, M22> {9., 3, 3, 10}, Mean<Axes<2>, M21> {15, 23}), Mean<Axes<2>, M21> {1, 2}));

  EXPECT_TRUE(is_near(reduce_columns(Mat23 {1, 2, 3, M_PI/3, M_PI/4, M_PI/6}), Mat21 {2, M_PI/4}));
  EXPECT_TRUE(is_near(reduce_columns(Mat23 {1, 2, 3, 4, 5, 6}), Mat21 {2, w_5}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(Mat23 {1, 2, 3, 4, 5, 6})),
    TMat22 {14, w_4 + 2*w_5 + 3*w_6, w_4 + 2*w_5 + 3*w_6, w_4*w_4 + w_5*w_5 + w_6*w_6}));

  EXPECT_TRUE(is_near(square(QR_decomposition(Mean<Axes<3>, M32> {1, 4, 2, 5, 3, 6})), TMat22 {14, 32, 32, 77}));

  using N = std::normal_distribution<double>::param_type;
  EMat23 m = EMat23::zero();
  for (int i=0; i<100; i++)
  {
    m = (m * i + to_Euclidean(randomize<Mat23>(N {1.0, 0.3}, 2.0))) / (i + 1);
  }
  Mat23 offset = {1, 1, 1, 2, 2, 2};
  EXPECT_TRUE(is_near(from_Euclidean(m), offset, 0.1));
  EXPECT_FALSE(is_near(from_Euclidean(m), offset, 1e-6));

}


TEST_F(typed_matrix_tests, Mean_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}), Mat32 {1, 2, 3, 4, 5, 6}));
  static_assert(is_mean_v<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}))>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}))>::RowCoefficients, C3>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(is_mean_v<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}))>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}))>::RowCoefficients, C2>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5}, TypedMatrix<C2, Angle, M21> {3, 6}),
    TypedMatrix<C2, Coefficients<Axis, Axis, Angle>, M23> {1, 2, 3, 4-2*M_PI, 5-2*M_PI, 6}));
  static_assert(not is_mean_v<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, TypedMatrix<C2, Angle, M21> {3, 6}))>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, TypedMatrix<C2, Angle, M21> {3, 6}))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, TypedMatrix<C2, Angle, M21> {3, 6}))>::ColumnCoefficients, Coefficients<Axis, Axis, Angle>>);

  EXPECT_TRUE(is_near(split_vertical(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C2, Axis>(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {Mat22 {1, 2, 3, 4}, Mat12 {5, 6}}));
  EXPECT_TRUE(is_near(split_horizontal<Axes<2>, Axis>(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {Mat22 {1, 2, 4, 5}, Mat21 {3, 6}}));
  EXPECT_TRUE(is_near(split_vertical<Axis, Angle>(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {Mat12 {1, 2}, Mat12 {3, 4-2*M_PI}}));
  EXPECT_TRUE(is_near(split_horizontal<Axis, Axis>(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {Mat21 {1, 4}, Mat21 {2, 5}}));

  EXPECT_TRUE(is_near(column(Mat22 {1, 2, 3, 4}, 0), Mean{1., 3}));
  EXPECT_TRUE(is_near(column(Mat22 {1, 2, 3, 4}, 1), Mean{2., 4-2*M_PI}));

  EXPECT_TRUE(is_near(column<0>(Mat22 {1, 2, 3, 4}), Mean{1., 3}));
  EXPECT_TRUE(is_near(column<1>(Mat22 {1, 2, 3, 4}), Mean{2., 4-2*M_PI}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(column<0>(Mat22 {1, 2, 3, 4}))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(column<1>(Mat22 {1, 2, 3, 4}))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(column<0>(Mat22 {1, 2, 3, 4}))>::ColumnCoefficients, Axis>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(column<1>(Mat22 {1, 2, 3, 4}))>::ColumnCoefficients, Axis>);

  auto m = Mat22 {1, 2, 3, 4};
  EXPECT_TRUE(is_near(apply_columnwise(m, [](auto& col){ col *= 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise(m, [](auto& col, std::size_t i){ col *= i; }), Mat22 {0, 4, 0, 8}));

  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4}, [](auto col){ return col * 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4}, [](const auto col){ return col * 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4}, [](auto&& col){ return col * 2; }), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4}, [](const auto& col){ return col * 2; }), Mat22 {2, 4, 6, 8}));

  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4}, [](auto col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4}, [](const auto col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4}, [](auto&& col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4}, [](const auto& col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4}));

  EXPECT_TRUE(is_near(apply_columnwise<2>([] { return Mat21 {1., 2}; }), Mat22 {1, 1, 2, 2}));
  EXPECT_TRUE(is_near(apply_columnwise<2>([](std::size_t i){ return Mat21 {i + 1., 2*i + 1}; }), Mat22 {1, 2, 1, 3}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Mat21()>()))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Mat21()>()))>::ColumnCoefficients, Axes<2>>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Mat21(std::size_t)>()))>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Mat21(std::size_t)>()))>::ColumnCoefficients, Axes<2>>);

  auto n = Mat22 {1, 2, 3, 4};
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


TEST_F(typed_matrix_tests, Mean_arithmetic)
{
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6}, TMat32 {8, 8, 8-2*M_PI, 8-2*M_PI, 8, 8}));
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} + TypedMatrix<C3, Axes<2>, M32> {1, 2, 3, 4, 5, 6}, TMat32 {8, 8, 8-2*M_PI, 8-2*M_PI, 8, 8}));
  EXPECT_TRUE(is_near(Mean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, Mean<Axes<3>, M32> {8, 8, 8, 8, 8, 8}));
  static_assert(is_mean_v<decltype(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6})>);
  static_assert(is_typed_matrix_v<decltype(Mat32 {7, 6, 5, 4, 3, 2} + TypedMatrix<C3, Axes<2>, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(is_typed_matrix_v<decltype(Mean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6}, TMat32 {6, 4, 2, 0, -2, -4}));
  EXPECT_TRUE(is_near(TypedMatrix<C3, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - Mean<C3, M32> {1, 2, 3, 4, 5, 6}, TMat32 {6, 4, 2, 2*M_PI, -2, -4}));
  EXPECT_TRUE(is_near(TypedMatrix<Axes<3>, Axes<2>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, TMat32 {6, 4, 2, 0, -2, -4}));
  static_assert(is_typed_matrix_v<decltype(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6})>);
  static_assert(is_typed_matrix_v<decltype(Mat32 {7, 6, 5, 4, 3, 2} - TypedMatrix<C3, Axes<2>, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(is_typed_matrix_v<decltype(Mean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3-M_PI, 4-M_PI, 5, 6}));
  static_assert(is_mean_v<decltype(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(is_mean_v<decltype(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(is_mean_v<decltype(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}, TMat23 {9, 12, 15, 19 - 8*M_PI, 26 - 10*M_PI, 33 - 12*M_PI}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(is_typed_matrix_v<decltype(Mat22 {1, 2, 3, 4} * Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * TypedMatrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6}, TMat23 {9, 12, 15, 19 - 8*M_PI, 26 - 10*M_PI, 33 - 12*M_PI}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * TypedMatrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * TypedMatrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, Axes<3>>);
  static_assert(is_typed_matrix_v<decltype(Mat22 {1, 2, 3, 4} * TypedMatrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 2} * TypedMatrix<Axes<2>, C3, M23> {1, 2, 3, 3, 2, 1}, TypedMatrix<Axes<2>, C3, M23> {7, 6, 5, 9, 10, 11}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * TypedMatrix<Axes<2>, C3, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(Mat22 {1, 2, 3, 4} * TypedMatrix<Axes<2>, C3, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, C3>);
  static_assert(is_typed_matrix_v<decltype(Mat22 {1, 2, 3, 4} * TypedMatrix<Axes<2>, C3, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}, TMat23 {9, 12, 15, 19 - 8*M_PI, 26 - 10*M_PI, 33 - 12*M_PI}));
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, C2>);
  static_assert(is_equivalent_v<typename MatrixTraits<decltype(TypedMatrix<C2, Axes<2>, M22> {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, Axes<3>>);
  static_assert(is_typed_matrix_v<decltype(Mat22 {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(-Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {-1, -2, -3, -4, -5, -6}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} == Mat22 {1, 2, 3, 4}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} != Mat22 {1, 2, 2, 4}));
  EXPECT_FALSE((Mat22 {1, 2, 3, 4} == Mean<Axes<2>, M22> {1, 2, 3, 4}));
}


TEST_F(typed_matrix_tests, Mean_angles_construct_coefficients)
{
  const auto v0 = make_Mean<Coefficients<Angle, Axis>>(Eigen::Matrix<double, 2, 1>(0.5, 2));
  EXPECT_EQ(v0[0], 0.5);
  EXPECT_EQ(v0[1], 2);
  const auto v1 = make_Mean<Coefficients<Axis, Angle>>(6., 7);
  EXPECT_EQ(v1[0], 6);
  EXPECT_EQ(v1[1], 7-2*M_PI);
  const auto v2 = make_Mean<Coefficients<Angle, Axis, Angle>>(7., 8, 9);
  EXPECT_EQ(v2[0], 7-2*M_PI);
  EXPECT_EQ(v2[1], 8);
  EXPECT_EQ(v2[2], 9-2*M_PI);
  Eigen::Matrix<double, 3, 3> m3;
  m3 << 9, 3, 1,
    3, 8 - M_PI*2, 2,
    7, 1, 8;
  auto v3 = make_Mean<double, Coefficients<Axis, Angle, Axis>, 3>();
  v3 << 9, 3, 1,
    3, 8, 2,
    7, 1, 8;
  EXPECT_TRUE(is_near(base_matrix(v3), m3));
  auto v3_1 = make_Mean<Coefficients<Axis, Angle, Axis>>(9., 3, 1, 3, 8, 2, 7, 1, 8);
  EXPECT_TRUE(is_near(v3_1, m3));
  auto v3_2 = v3;
  EXPECT_TRUE(is_near(v3_2, m3));
  static_assert(std::is_same_v<decltype(v3)::BaseMatrix, decltype(m3)>);
  auto v3_3 = make_Mean<double, Coefficients<Axis, Angle, Axis>, 3>();
  v3_3 << v3;
  EXPECT_TRUE(is_near(v3_3, m3));
}


TEST_F(typed_matrix_tests, Mean_angle_concatenate_split)
{
  using C3 = Coefficients<Angle, Axis, Angle>;
  using Var3 = Mean<C3>;
  auto x1 = Var3 {5., 7, 9};
  auto x2 = Var3 {3., 2, 1};
  auto x3 = Mean<Concatenate<C3, C3>> {5., 7, 9, 3, 2, 1};
  EXPECT_TRUE(is_near(concatenate(x1, x2), x3));
  auto x4 = Mean<Concatenate<C3, C3, C3>> {5., 7, 9, 3, 2, 1, 5, 7, 9};
  EXPECT_TRUE(is_near(concatenate(x1, x2, x1), x4));
  auto [x5, x6] = split_vertical<C3, C3>(x3);
  EXPECT_TRUE(is_near(base_matrix(x5), base_matrix(x1)));
  EXPECT_TRUE(is_near(x5, x1));
  EXPECT_TRUE(is_near(x6, x2));
  const Var3 y1 {M_PI / 6, 2, M_PI / 3};
  const Var3 y2 {M_PI / 4, 3, M_PI / 4};
  const Mean<Coefficients<Angle, Axis, Angle, Angle, Axis, Angle>>
    y3 {M_PI / 6, 2, M_PI / 3, M_PI / 4, 3, M_PI / 4};
  EXPECT_TRUE(is_near(concatenate(y1, y2), y3));
  EXPECT_TRUE(is_near(split_vertical<C3, C3>(y3), std::tuple(y1, y2)));
  const Mean<Coefficients<Angle, Axis, Angle, Angle, Axis, Angle, Angle, Axis, Angle>>
    y4 {M_PI / 6, 2, M_PI / 3, M_PI / 4, 3, M_PI / 4, M_PI / 6, 2, M_PI / 3};
  EXPECT_TRUE(is_near(concatenate(y1, y2, y1), y4));
  EXPECT_TRUE(is_near(split_vertical<C3, C3, C3>(y4), std::tuple(y1, y2, y1)));
}


TEST_F(typed_matrix_tests, Mean_angle_Euclidean_conversion)
{
  using Var3 = Mean<Coefficients<Angle, Axis, Angle>>;
  const Var3 x1 {M_PI / 6, 5, -M_PI / 3};
  const Var3 x1p {M_PI / 5.99999999, 5, -M_PI / 2.99999999};
  const auto m1 = (Eigen::Matrix<double, 3, 1> {} << M_PI / 6, 5, -M_PI / 3).finished();
  EXPECT_TRUE(is_near(x1, x1p));
  EXPECT_TRUE(is_near(x1.base_matrix(), x1p.base_matrix()));
  EXPECT_TRUE(is_near(to_Euclidean(x1).base_matrix(), to_Euclidean(x1p).base_matrix()));
  EXPECT_TRUE(is_near(to_Euclidean(x1), to_Euclidean(x1p)));
  EXPECT_NEAR(x1.base_matrix()[0], M_PI / 6, 1e-6);
  EXPECT_NEAR(x1.base_matrix()[1], 5, 1e-6);
  EXPECT_NEAR(x1.base_matrix()[2], -M_PI / 3, 1e-6);
  EXPECT_TRUE(is_near(x1.base_matrix(), m1));
  EXPECT_TRUE(is_near(make_Mean(std::sqrt(3.)/2, 0.5, 5, 0.5, -std::sqrt(3.)/2), to_Euclidean(x1).base_matrix()));
  EXPECT_TRUE(is_near(Var3 {m1}, x1));
}


TEST_F(typed_matrix_tests, Mean_angle_mult_TypedMatrix)
{
  using M = TypedMatrix<Coefficients<Axis, Angle>, Coefficients<Axis, Angle, Axis>>;
  using Vm = Mean<Coefficients<Axis, Angle, Axis>>;
  using Rm = TypedMatrix<Coefficients<Axis, Angle>, Axis>;
  const M m {1, -2, 4,
             -5, 2, -1};
  const Vm vm {2, 1, -1};
  const Rm rm {-4, -7};
  static_assert(std::is_same_v<typename MatrixTraits<decltype(strict(m*vm))>::RowCoefficients, Coefficients<Axis, Angle>>);
  EXPECT_TRUE(is_near(m*vm, rm));
}


TEST_F(typed_matrix_tests, Mean_angle_arithmetic)
{
  using Var3 = Mean<Coefficients<Angle, Axis, Angle, Axis, Axis>>;
  Var3 v1 {1, 2, 3, 4, 5};
  Var3 v2 {2, 4, -6, 8, -10};
  Var3 v3 {3, 6, -3, 12, -5};
  Var3 v4 {-1, -2, 9 - M_PI*2, -4, 15};
  Var3 v5 {3., 6, 9 - M_PI*2, 12, 15};
  EXPECT_TRUE(is_near(v1 + v2, v3));
  EXPECT_TRUE(is_near(v1 - v2, v4));
  EXPECT_TRUE(is_near(v1 * 3., v5));
  EXPECT_TRUE(is_near(3. * v1, v5));
  EXPECT_TRUE(is_near(v5 / 3. + Var3 {0, 0, M_PI*2/3, 0, 0}, v1));
  Var3 v7 = v1;
  v7 *= 3;
  EXPECT_TRUE(is_near(v7, v5));
  v7 /= 3;
  v7 += Var3 {0, 0, M_PI*2/3, 0, 0};
  EXPECT_TRUE(is_near(v7, v1));
  v7 -= v2;
  EXPECT_TRUE(is_near(v7, v4));
  v7 += v2;
  EXPECT_TRUE(is_near(v7, v1));
  auto v6 = make_Mean(v1 + v2);
  static_assert(std::is_same_v<decltype(v6)::BaseMatrix&, decltype(v6.base_matrix())>);
  v7 = v6;
  EXPECT_TRUE(is_near(v7, v6));
  v7 += v6;
  EXPECT_TRUE(is_near(v7, Var3 {6 - M_PI*2, 12, -6 + M_PI*2, 24, -10}));
  v7 -= v6;
  EXPECT_TRUE(is_near(v7, v6));
  using T0 = decltype(v5);
  using T3 = decltype(v5.base_matrix());
  static_assert(std::is_same_v<typename T0::BaseMatrix&, T3>);
  static_assert(std::is_same_v<std::decay_t<decltype((v5 + v2).base_matrix().base_matrix().base_matrix())>, std::decay_t<decltype(v5.base_matrix() + v2.base_matrix())>>);
  static_assert(std::is_same_v<typename MatrixTraits<decltype(strict(v5))>::BaseMatrix, std::remove_reference_t<decltype(strict_matrix(v5))>>);
}


TEST_F(typed_matrix_tests, Mean_angle_arithmetic_Euclidean)
{
  using Var3 = Mean<Coefficients<Angle, Axis, Angle>>;
  Var3 x1 {M_PI / 6, 5, -M_PI / 3};
  Var3 x2 {2 * M_PI, 0, 6 * M_PI};
  auto x1e = to_Euclidean(x1);
  auto x2e = to_Euclidean(x2);
  EXPECT_TRUE(is_near(from_Euclidean(x1e + x2e), Var3 {M_PI / 12, 5, -M_PI / 6}));
  EXPECT_TRUE(is_near(from_Euclidean(x2e + x1e), Var3 {M_PI / 12, 5, -M_PI / 6}));
  EXPECT_TRUE(is_near(from_Euclidean(x1e - x2e), Var3 {M_PI * 7 / 12, 5, -M_PI * 2 / 3}));
  EXPECT_TRUE(is_near(from_Euclidean(x2e - x1e), Var3 {-M_PI * 5 / 12, -5, M_PI / 3}));
  EXPECT_TRUE(is_near(from_Euclidean(-x1e), Var3 {-M_PI * 5 / 6, -5, M_PI * 2 / 3}));
  Var3 x50 {M_PI / 6, 50, -M_PI / 3};
  EXPECT_TRUE(is_near(from_Euclidean(x1e * 10.), x50));
  EXPECT_TRUE(is_near(from_Euclidean(10. * x1e), x50));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x50) / 10), x1));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x50) / 10. + x2e / 10.), Var3 {M_PI / 12, 5, -M_PI / 6}));
  EXPECT_TRUE(is_near(from_Euclidean((to_Euclidean(x1) * 2.0 + to_Euclidean(x1).zero()) / 2.0), x1));
  auto mean_x = Mean(x1);
  mean_x = from_Euclidean((to_Euclidean(mean_x) * 2.0 + to_Euclidean(mean_x).zero()) / 2.0);
  EXPECT_TRUE(is_near(mean_x, x1));
  auto x6 = x1e;
  EXPECT_TRUE(is_near(to_Euclidean(Var3{x6}), x6));
}


TEST_F(typed_matrix_tests, Mean_angle2pi_arithmetic_Euclidean)
{
  using Var3 = Mean<Coefficients<AnglePositiveRadians, Axis, AnglePositiveRadians>>;
  const Var3 x1 {M_PI / 6, 5, M_PI * 5 / 3};
  const Var3 x2 {2 * M_PI, 0, 6 * M_PI};
  const auto x1e = to_Euclidean(x1);
  const auto x2e = to_Euclidean(x2);
  EXPECT_TRUE(is_near(from_Euclidean(x1e + x2e), Var3 {M_PI / 12, 5, M_PI * 11 / 6}));
  EXPECT_TRUE(is_near(from_Euclidean(x2e + x1e), Var3 {M_PI / 12, 5, M_PI * 11 / 6}));
  EXPECT_TRUE(is_near(from_Euclidean(x1e - x2e), Var3 {M_PI * 7 / 12, 5, M_PI * 4 / 3}));
  EXPECT_TRUE(is_near(from_Euclidean(x2e - x1e), Var3 {M_PI * 19 / 12, -5, M_PI / 3}));
  EXPECT_TRUE(is_near(from_Euclidean(-x1e), Var3 {M_PI * 7 / 6, -5, M_PI * 2 / 3}));
  const Var3 x50 {M_PI / 6, 50, M_PI * 5 / 3};
  EXPECT_TRUE(is_near(from_Euclidean(x1e * 10.), x50));
  EXPECT_TRUE(is_near(from_Euclidean(10. * x1e), x50));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x50) / 10.), x1));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x50) / 10. + x2e / 10.), Var3 {M_PI / 12, 5, M_PI * 11 / 6}));
}


TEST_F(typed_matrix_tests, Mean_angle2deg_arithmetic_Euclidean)
{
  using Var3 = Mean<Coefficients<AngleDegrees, Axis, AngleDegrees>>;
  const Var3 x1 {30, 5, 300};
  const Var3 x2 {360, 0, 1080};
  const auto x1e = to_Euclidean(x1);
  const auto x2e = to_Euclidean(x2);
  EXPECT_TRUE(is_near(from_Euclidean(x1e + x2e), Var3 {15, 5, 330}));
  EXPECT_TRUE(is_near(from_Euclidean(x2e + x1e), Var3 {15, 5, 330}));
  EXPECT_TRUE(is_near(from_Euclidean(x1e - x2e), Var3 {105, 5, 240}));
  EXPECT_TRUE(is_near(from_Euclidean(x2e - x1e), Var3 {285, -5, 60}));
  EXPECT_TRUE(is_near(from_Euclidean(-x1e), Var3 {210, -5, 120}));
  const Var3 x50 {30, 50, 300};
  EXPECT_TRUE(is_near(from_Euclidean(x1e * 10.), x50));
  EXPECT_TRUE(is_near(from_Euclidean(10. * x1e), x50));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x50) / 10.), x1));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x50) / 10. + x2e / 10.), Var3 {15, 5, 330}));
}


TEST_F(typed_matrix_tests, Mean_angle_columns)
{
  using Var3 = Mean<Coefficients<Angle, Axis>, Eigen::Matrix<double, 2, 2>>;
  using TVar3 = TypedMatrix<Coefficients<Angle, Axis>, Axes<2>, Eigen::Matrix<double, 2, 2>>;
  Var3 v1 {1, 2, 3, 4};
  Var3 v2 {6, 4, -6, 8};
  Var3 v3 {7 - M_PI*2, 6 - M_PI*2, -3, 12};
  TVar3 v4 {-5 + M_PI*2, -2, 9, -4};
  TVar3 v5 {0.5, 1, 1.5, 2};
  EXPECT_TRUE(is_near(v1 + v2, v3));
  EXPECT_TRUE(is_near(v1 - v2, v4));
  EXPECT_TRUE(is_near(v1 * 0.5, v5));
  EXPECT_TRUE(is_near(0.5 * v1, v5));
  EXPECT_TRUE(is_near(v5 / 0.5, TVar3 {v1}));
}


TEST_F(typed_matrix_tests, Mean_angle_columns_Euclidean)
{
  using Var3 = Mean<Coefficients<Angle, Axis>, Eigen::Matrix<double, 2, 2>>;
  Var3 x1 {M_PI / 6, -M_PI / 3, 5, 2};
  Var3 x2 {2 * M_PI, 6 * M_PI, 0, 0};
  const auto x1e = to_Euclidean(x1);
  const auto x2e = to_Euclidean(x2);
  EXPECT_TRUE(is_near(from_Euclidean(x1e + x2e), Var3 {M_PI / 12, -M_PI / 6, 5, 2}));
  EXPECT_TRUE(is_near(from_Euclidean(x2e + x1e), Var3 {M_PI / 12, -M_PI / 6, 5, 2}));
  EXPECT_TRUE(is_near(from_Euclidean(x1e - x2e), Var3 {M_PI * 7 / 12, -M_PI * 2 / 3, 5, 2}));
  EXPECT_TRUE(is_near(from_Euclidean(x2e - x1e), Var3 {-M_PI * 5 / 12, M_PI / 3, -5, -2}));
  EXPECT_TRUE(is_near(from_Euclidean(-x1e), Var3 {-M_PI * 5 / 6, M_PI * 2 / 3, -5, -2}));
  Var3 x10 {M_PI / 6, -M_PI / 3, 50, 20};
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x1) * 10.), x10));
  EXPECT_TRUE(is_near(from_Euclidean(10. * to_Euclidean(x1)), x10));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x10) / 10.), x1));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(x10) / 10. + x2e / 10.), Var3 {M_PI / 12, -M_PI / 6, 5, 2}));
  Var3 mean_x = x1;
  mean_x = from_Euclidean((to_Euclidean(mean_x) * 2.0 + to_Euclidean(mean_x).zero()) / 2.0);
  EXPECT_TRUE(is_near(mean_x, x1));
  auto x6 = to_Euclidean(x1);
  EXPECT_TRUE(is_near(to_Euclidean(Var3{x6}), x6));
  EXPECT_NEAR(x1(0,0), M_PI / 6, 1e-6);
  EXPECT_NEAR(x1(0,1), -M_PI/3, 1e-6);
  EXPECT_NEAR(x1(1,0), 5, 1e-6);
  EXPECT_NEAR(x1(1,1), 2, 1e-6);
  EXPECT_NEAR(x1e(0,1), 0.5, 1e-6);
  EXPECT_NEAR(x1e(1,0), 0.5, 1e-6);
  EXPECT_NEAR(x1e(2,0), 5, 1e-6);
  EXPECT_NEAR(x1e(2,1), 2, 1e-6);
}


TEST_F(typed_matrix_tests, Wrap_angle)
{
  using R = Mean<Angle>;
  EXPECT_TRUE(is_near(R {-M_PI*.99} - R {M_PI*0.99}, TypedMatrix<Angle> {M_PI*0.02}));
  R x0 {M_PI_4};
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), M_PI_4, 1e-6);
  set_element(x0, 5*M_PI_4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0), -3*M_PI_4, 1e-6);
  set_element(x0, -7*M_PI/6, 0);
  EXPECT_NEAR(get_element(x0, 0), 5*M_PI/6, 1e-6);
}


TEST_F(typed_matrix_tests, Wrap_distance)
{
  using R = Mean<Distance>;
  EXPECT_TRUE(is_near(R {4} - R {5}, TypedMatrix<Axis> {-1}));
  R x0 {-5};
  EXPECT_TRUE(is_near(x0, Eigen::Matrix<double, 1, 1> {5}));
  EXPECT_TRUE(is_near(x0 + R {1.2}, Eigen::Matrix<double, 1, 1> {6.2}));
  EXPECT_TRUE(is_near(from_Euclidean(-to_Euclidean(x0) + to_Euclidean(R {1.2})), Eigen::Matrix<double, 1, 1> {3.8}));
  EXPECT_TRUE(is_near(R {1.1} - 3. * R {1}, -strict_matrix(R {1.9})));
  EXPECT_TRUE(is_near(R {1.2} + R {-3}, R {4.2}));
  EXPECT_NEAR(get_element(x0, 0, 0), 5, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), 5., 1e-6);
  set_element(x0, 4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0, 0), 4., 1e-6);
  set_element(x0, -3, 0, 0);
  EXPECT_NEAR(get_element(x0, 0, 0), 3., 1e-6);
}


TEST_F(typed_matrix_tests, Wrap_inclination)
{
  using R = Mean<InclinationAngle>;
  EXPECT_TRUE(is_near(R {-M_PI/2} - R {M_PI/2}, TypedMatrix<Angle> {-M_PI}));
  EXPECT_TRUE(is_near(R {M_PI * 7 / 12}, R {M_PI * 5 / 12}));
  EXPECT_TRUE(is_near(R {-M_PI * 7 / 12}, R {-M_PI * 5 / 12}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(R {M_PI / 6}) + to_Euclidean(R {M_PI})), R {M_PI / 12}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(R {-M_PI / 6}) + to_Euclidean(R {M_PI})), R {-M_PI / 12}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(R {M_PI * 5 / 6}) + to_Euclidean(R {M_PI / 2})), R {M_PI / 3}));
  R x0 {M_PI_2};
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_2, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), M_PI_2, 1e-6);
  set_element(x0, M_PI_4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_4, 1e-6);
  set_element(x0, 3*M_PI_4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_4, 1e-6);
}


TEST_F(typed_matrix_tests, Wrap_polar)
{
  using P = Mean<Polar<Distance, Angle>>;
  EXPECT_TRUE(is_near(P {4, -M_PI*.99} - P {5, M_PI*0.99}, TypedMatrix<Polar<Distance, Angle>, Axis> {-1, M_PI*0.02}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(P {1., M_PI / 6}) + to_Euclidean(P {-0.5, 0})), P {1.5, M_PI * 7 / 12}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(P {1., M_PI / 6}) + to_Euclidean(P {-1.5, 0})), P {2.5, M_PI * 7 / 12}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(P {1., M_PI * 5 / 6}) + to_Euclidean(P {0, -M_PI * 2 / 3})), P {1., -M_PI * 11 / 12}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(P {1., M_PI * 5 / 6}) + to_Euclidean(P {-1.5, -M_PI * 2 / 3})), P {2.5, M_PI * 7 / 12}));
  P x0 {2, M_PI_4};
  EXPECT_NEAR(get_element(x0, 0, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), M_PI_4, 1e-6);
  set_element(x0, -1.5, 0);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -3*M_PI_4, 1e-6);
  set_element(x0, 7*M_PI/6, 1);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -5*M_PI/6, 1e-6);

  using Q = Mean<Polar<Angle, Distance>>;
  EXPECT_TRUE(is_near(Q {-M_PI*.99, 4} - Q {M_PI*0.99, 5}, TypedMatrix<Polar<Angle, Distance>, Axis> {M_PI*0.02, -1}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(Q {M_PI / 6, 1}) + to_Euclidean(Q {0, -0.5})), Q {M_PI * 7 / 12, 1.5}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(Q {M_PI / 6, 1}) + to_Euclidean(Q {0, -1.5})), Q {M_PI * 7 / 12, 2.5}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(Q {M_PI * 5 / 6, 1}) + to_Euclidean(Q {-M_PI * 2 / 3, 0})), Q {-M_PI * 11 / 12, 1}));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(Q {M_PI * 5 / 6, 1}) + to_Euclidean(Q {-M_PI * 2 / 3, -1.5})), Q {M_PI * 7 / 12, 2.5}));
  Q x1 {M_PI_4, 2};
  EXPECT_NEAR(get_element(x1, 1, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), M_PI_4, 1e-6);
  set_element(x1, -1.5, 1);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -3*M_PI_4, 1e-6);
  set_element(x1, 7*M_PI/6, 0);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -5*M_PI/6, 1e-6);
}


TEST_F(typed_matrix_tests, Wrap_spherical)
{
  using S = Mean<Spherical<Distance, Angle, InclinationAngle>>;
  EXPECT_TRUE(is_near(S {4, -M_PI*.99, -M_PI/2} - S {5, M_PI*0.99, M_PI/2}, TypedMatrix<Spherical<Distance, Angle, InclinationAngle>, Axis> {-1, M_PI*0.02, -M_PI}));
  EXPECT_TRUE(is_near(S {-0.5, M_PI / 3, M_PI / 6}, S {0.5, -M_PI * 2 / 3, -M_PI / 6}));
  EXPECT_TRUE(is_near(S {-0.5, M_PI * 7, M_PI * 7 / 6}, S {0.5, M_PI, M_PI / 6}));
  const auto x1 = S {0.5, std::atan2(3. / 4, 1. + std::sqrt(3.) / 4),
                     std::asin(0.5 / std::hypot(1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  const auto x2 = S {0.5, std::atan2(3. / 4, 1. + std::sqrt(3.) / 4) - M_PI,
                     -std::asin(0.5 / std::hypot(1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  const auto x3 = S {1.0, std::atan2(3. / 4, -1. + std::sqrt(3.) / 4),
                     std::asin(0.5 / std::hypot(-1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(S {1., M_PI / 3, M_PI / 6}) - to_Euclidean(S {-0.5, 0, 0})), x1));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(S {1., M_PI / 3, M_PI / 6}) - to_Euclidean(S {-1.5, 0, 0})), x2));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(S {1., M_PI / 3, M_PI / 6}) + to_Euclidean(S {0, 0, M_PI})), x3));
  S x0 {2, M_PI_4, -M_PI_4};
  EXPECT_NEAR(get_element(x0, 0, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), -M_PI_4, 1e-6);
  set_element(x0, -1.5, 0);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -3*M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), M_PI_4, 1e-6);
  set_element(x0, 7*M_PI/6, 1);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), M_PI_4, 1e-6);
  set_element(x0, 3*M_PI_4, 2);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), M_PI_4, 1e-6);

  using T = Mean<Spherical<Angle, Distance, InclinationAngle>>;
  EXPECT_TRUE(is_near(T {-M_PI*.99, 4, -M_PI/2} - T {M_PI*0.99, 5, M_PI/2}, TypedMatrix<Spherical<Angle, Distance, InclinationAngle>, Axis> {M_PI*0.02, -1, -M_PI}));
  EXPECT_TRUE(is_near(T {M_PI / 3, -0.5, M_PI / 6}, T {-M_PI * 2 / 3, 0.5, -M_PI / 6}));
  EXPECT_TRUE(is_near(T {M_PI * 7, -0.5, M_PI * 7 / 6}, T {M_PI, 0.5, M_PI / 6}));
  const auto y1 = T {std::atan2(3. / 4, 1. + std::sqrt(3.) / 4),
                     0.5, std::asin(0.5 / std::hypot(1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  const auto y2 = T {std::atan2(3. / 4, 1. + std::sqrt(3.) / 4) - M_PI,
                     0.5, -std::asin(0.5 / std::hypot(1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  const auto y3 = T {std::atan2(3. / 4, -1. + std::sqrt(3.) / 4),
                     1.0, std::asin(0.5 / std::hypot(-1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(T {M_PI / 3, 1, M_PI / 6}) - to_Euclidean(T {0, -0.5, 0})), y1));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(T {M_PI / 3, 1, M_PI / 6}) - to_Euclidean(T {0, -1.5, 0})), y2));
  EXPECT_TRUE(is_near(from_Euclidean(to_Euclidean(T {M_PI / 3, 1, M_PI / 6}) + to_Euclidean(T {0, 0, M_PI})), y3));
  T z1 {M_PI_4, 2, -M_PI_4};
  EXPECT_NEAR(get_element(z1, 1, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(z1, 0), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(z1, 2), -M_PI_4, 1e-6);
  set_element(z1, -1.5, 1);
  EXPECT_NEAR(get_element(z1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(z1, 0), -3*M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(z1, 2), M_PI_4, 1e-6);
  set_element(z1, 7*M_PI/6, 0);
  EXPECT_NEAR(get_element(z1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(z1, 0), -5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(z1, 2), M_PI_4, 1e-6);
  set_element(z1, 3*M_PI_4, 2);
  EXPECT_NEAR(get_element(z1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(z1, 0), M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(z1, 2), M_PI_4, 1e-6);

  using U = Mean<Spherical<Angle, InclinationAngle, Distance>>;
  EXPECT_TRUE(is_near(U {-M_PI*.99, -M_PI/2, 4} - U {M_PI*0.99, M_PI/2, 5}, TypedMatrix<Spherical<Angle, InclinationAngle, Distance>, Axis> {M_PI*0.02, -M_PI, -1}));
  U z2 {M_PI_4, -M_PI_4, 2};
  EXPECT_NEAR(get_element(z2, 2, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(z2, 0), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(z2, 1), -M_PI_4, 1e-6);
  set_element(z2, -1.5, 2);
  EXPECT_NEAR(get_element(z2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(z2, 0), -3*M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(z2, 1), M_PI_4, 1e-6);
  set_element(z2, 7*M_PI/6, 0);
  EXPECT_NEAR(get_element(z2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(z2, 0), -5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(z2, 1), M_PI_4, 1e-6);
  set_element(z2, 3*M_PI_4, 1);
  EXPECT_NEAR(get_element(z2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(z2, 0), M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(z2, 1), M_PI_4, 1e-6);
}


TEST_F(typed_matrix_tests, Wrap_angle_polar)
{
  using R = Mean<Coefficients<Angle, Polar<Distance, Angle>>>;
  EXPECT_TRUE(is_near(R {M_PI_4, 1., M_PI/6} + R {-M_PI_2, 0.5, M_PI}, R {-M_PI_4, 1.5, -M_PI*5/6}));
  EXPECT_TRUE(is_near(R {M_PI_4, 1., M_PI/6} + R {M_PI_4, -1.5, -M_PI_2}, R {M_PI_2, 2.5, M_PI*2/3}));
  EXPECT_TRUE(is_near(R {-M_PI_2, 1., M_PI*5/6} + R {M_PI/3, -0.5, M_PI_2}, R {-M_PI/6, 1.5, M_PI/3}));
  R x0 {M_PI_4, 2, M_PI_4};
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), M_PI_4, 1e-6);
  set_element(x0, 5*M_PI_4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0), -3*M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), 2, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), M_PI_4, 1e-6);
  set_element(x0, -7*M_PI/6, 0);
  EXPECT_NEAR(get_element(x0, 0), 5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), 2, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), M_PI_4, 1e-6);
  set_element(x0, -1.5, 1);
  EXPECT_NEAR(get_element(x0, 0), 5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), -3*M_PI_4, 1e-6);
  set_element(x0, 7*M_PI/6, 2);
  EXPECT_NEAR(get_element(x0, 0), 5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), -5*M_PI/6, 1e-6);

  using Q = Mean<Coefficients<Polar<Angle, Distance>, Angle>>;
  EXPECT_TRUE(is_near(Q {M_PI/6, 1, M_PI_4} + Q {M_PI, 0.5, -M_PI_2}, Q {-M_PI*5/6, 1.5, -M_PI_4}));
  EXPECT_TRUE(is_near(Q {M_PI/6, 1, M_PI_4} + Q {-M_PI_2, -1.5, M_PI_4}, Q {M_PI*2/3, 2.5, M_PI_2}));
  EXPECT_TRUE(is_near(Q {M_PI*5/6, 1., -M_PI_2} + Q {M_PI_2, -0.5, M_PI/3}, Q {M_PI/3, 1.5, -M_PI/6}));
  Q x1 {M_PI_4, 2, M_PI_4};
  EXPECT_NEAR(get_element(x1, 2, 0), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), M_PI_4, 1e-6);
  set_element(x1, 5*M_PI_4, 2, 0);
  EXPECT_NEAR(get_element(x1, 2), -3*M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x1, 1), 2, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), M_PI_4, 1e-6);
  set_element(x1, -7*M_PI/6, 2);
  EXPECT_NEAR(get_element(x1, 2), 5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x1, 1), 2, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), M_PI_4, 1e-6);
  set_element(x1, -1.5, 1);
  EXPECT_NEAR(get_element(x1, 2), 5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -3*M_PI_4, 1e-6);
  set_element(x1, 7*M_PI/6, 0);
  EXPECT_NEAR(get_element(x1, 2), 5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -5*M_PI/6, 1e-6);
}

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

using M12 = native_matrix_t<double, 1, 2>;
using M13 = native_matrix_t<double, 1, 3>;
using M21 = native_matrix_t<double, 2, 1>;
using M22 = native_matrix_t<double, 2, 2>;
using M23 = native_matrix_t<double, 2, 3>;
using M31 = native_matrix_t<double, 3, 1>;
using M32 = native_matrix_t<double, 3, 2>;
using M33 = native_matrix_t<double, 3, 3>;
using M42 = native_matrix_t<double, 4, 2>;
using M43 = native_matrix_t<double, 4, 3>;
using I22 = IdentityMatrix<M22>;
using Z22 = ZeroMatrix<M22>;
using C2 = Coefficients<Axis, angle::Radians>;
using C3 = Coefficients<Axis, angle::Radians, Axis>;
using Mat12 = EuclideanMean<Axis, M12>;
using Mat13 = EuclideanMean<Axis, M13>;
using Mat21 = EuclideanMean<C2, M31>;
using Mat22 = EuclideanMean<C2, M32>;
using Mat23 = EuclideanMean<C2, M33>;
using Mat32 = EuclideanMean<C3, M42>;
using Mat33 = EuclideanMean<C3, M43>;
using TM22 = Matrix<Axes<2>, Axes<2>, M22>;
using TM23 = Matrix<Axes<2>, Axes<3>, M23>;
using TM32 = Matrix<Axes<3>, Axes<2>, M32>;
using TM33 = Matrix<Axes<3>, Axes<3>, M33>;
using TM42 = Matrix<Axes<4>, Axes<2>, M42>;

inline I22 i22 = M22::Identity();
inline Z22 z22 = Z22();
inline auto covi22 = Covariance<C2, I22>(i22);
inline auto covz22 = Covariance<C2, Z22>(z22);
inline auto sqcovi22 = SquareRootCovariance<C2, I22>(i22);
inline auto sqcovz22 = SquareRootCovariance<C2, Z22>(z22);


TEST_F(matrices, EuclideanMean_class)
{
  // Default constructor and Eigen3 construction
  Mat23 mat23a;
  mat23a << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  EXPECT_TRUE(is_near(mat23a, TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  // Copy constructor
  Mat23 mat23b = const_cast<const Mat23&>(mat23a);
  EXPECT_TRUE(is_near(mat23b, TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  // Move constructor
  auto xa = Mat23 {9, 8, 7, 6, 5, 4, 3, 2, 1};
  Mat23 mat23c(std::move(xa));
  EXPECT_TRUE(is_near(mat23c, TM33 {9, 8, 7, 6, 5, 4, 3, 2, 1}));

  // Convert from different covariance types
  EuclideanMean<Axes<2>, M23> mat23_x1(Matrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x1, TM23 {1, 2, 3, 4, 5, 6}));
  EuclideanMean<Axes<2>, M23> mat23_x2(Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x2, TM23 {1, 2, 3, 4, 5, 6}));
  Mat23 mat23_x3(Mean<C2, M23> {
    1, 2, 3,
    pi/3, pi/6, pi/4});
  EXPECT_TRUE(is_near(mat23_x3, TM33 {
    1, 2, 3,
    0.5, std::sqrt(3)/2, sqrt2/2,
    std::sqrt(3)/2, 0.5, sqrt2/2}));

  // Construct from a typed_matrix_nestable
  Mat23 mat23d(make_native_matrix<M33>(1, 2, 3, 4, 5, 6, 7, 8, 9));
  EXPECT_TRUE(is_near(mat23d, TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  // Construct from a list of coefficients
  Mat23 mat23e {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(is_near(mat23e, TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  // Copy assignment
  mat23c = mat23b;
  EXPECT_TRUE(is_near(mat23c, TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  // Move assignment
  auto ya = Mat23 {3, 4, 5, 6, 7, 8, 9, 10, 11};
  mat23c = std::move(ya);
  EXPECT_TRUE(is_near(mat23c, Mat23 {3, 4, 5, 6, 7, 8, 9, 10, 11}));

  // assign from different covariance types
  mat23_x1 = Matrix<Axes<2>, Axes<3>, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x1, TM23 {6, 5, 4, 3, 2, 1}));
  mat23_x2 = Mean<Axes<2>, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x2, TM23 {6, 5, 4, 3, 2, 1}));
  mat23_x3 = Mean<C2, M23> {
    3, 2, 1,
    pi/6, pi/4, pi/3};
  EXPECT_TRUE(is_near(mat23_x3, TM33 {
    3, 2, 1,
    std::sqrt(3)/2, sqrt2/2, 0.5,
    0.5, sqrt2/2, std::sqrt(3)/2}));

  // assign from a typed_matrix_nestable
  mat23e = make_native_matrix<M33>(3, 4, 5, 6, 7, 8, 9, 10, 11);

  // Assign from a list of coefficients (via move assignment operator)
  mat23e = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23e, TM33 {9, 8, 7, 6, 5, 4, 3, 2, 1}));

  // Increment
  mat23a += Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(is_near(mat23a, TM33 {2, 4, 6, 8, 10, 12, 14, 16, 18}));

  // Decrement
  mat23a -= Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(is_near(mat23a, Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  // Scalar multiplication
  mat23a *= 2;
  EXPECT_TRUE(is_near(mat23a, TM33 {2, 4, 6, 8, 10, 12, 14, 16, 18}));

  // Scalar division
  mat23a /= 2;
  EXPECT_TRUE(is_near(mat23a, TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  // Scalar multiplication, zero
  mat23a *= 0;
  EXPECT_TRUE(is_near(mat23a, M33::Zero()));

  // Zero
  EXPECT_TRUE(is_near(Mat23::zero(), M33::Zero()));

  // Identity
  EXPECT_TRUE(is_near(EuclideanMean<Axes<2>, M22>::identity(), M22::Identity()));
}


TEST_F(matrices, EuclideanMean_subscripts)
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

  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(0, 0), 1, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(0, 1), 2, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(0, 2), 3, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(1, 0), 4, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(1, 1), 5, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(1, 2), 6, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(2, 0), 7, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(2, 1), 8, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9})(2, 2), 9, 1e-6);
}


TEST_F(matrices, EuclideanMean_deduction_guides)
{
  auto a = make_native_matrix<M23>(1, 2, 3, 4, 5, 6);
  EXPECT_TRUE(is_near(EuclideanMean(a), a));
  static_assert(equivalent_to<typename MatrixTraits<decltype(EuclideanMean(a))>::RowCoefficients, Axes<2>>);

  auto b1 = Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(is_near(EuclideanMean(b1), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(EuclideanMean(b1))>::RowCoefficients, C2>);

  auto b2 = Matrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(EuclideanMean(b2), TM23 {1, 2, 3, 4, 5, 6}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(EuclideanMean(b2))>::RowCoefficients, Axes<2>>);

  auto b3 = Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(EuclideanMean(b3), TM23 {1, 2, 3, 4, 5, 6}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(EuclideanMean(b3))>::RowCoefficients, Axes<2>>);
  static_assert(MatrixTraits<decltype(EuclideanMean(b3))>::dimension == 2);
}


TEST_F(matrices, EuclideanMean_make_functions)
{
  auto a = make_native_matrix<M33>(1, 2, 3, 4, 5, 6, 7, 8, 9);
  EXPECT_TRUE(is_near(make_euclidean_mean<C2>(a), a));
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_euclidean_mean<C2>(a))>::RowCoefficients, C2>);
  EXPECT_TRUE(is_near(make_euclidean_mean<C2>(a), a));
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_euclidean_mean<C2>(a))>::RowCoefficients, C2>);

  auto b = Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(is_near(make_euclidean_mean(b), a));
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_euclidean_mean(b))>::RowCoefficients, C2>);

  static_assert(equivalent_to<typename MatrixTraits<decltype(make_euclidean_mean<C2, M33>())>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(make_euclidean_mean<M23>())>::RowCoefficients, Axes<2>>);
}


TEST_F(matrices, EuclideanMean_traits)
{
  static_assert(typed_matrix<Mat23>);
  static_assert(not mean<Mat23>);
  static_assert(euclidean_mean<Mat23>);
  static_assert(euclidean_transformed<Mat23>);
  static_assert(not euclidean_transformed<EuclideanMean<Axes<2>, I22>>);
  static_assert(not wrapped_mean<Mat23>);
  static_assert(untyped_columns<Mat23>);

  static_assert(not identity_matrix<Mat23>);
  static_assert(identity_matrix<EuclideanMean<Axes<2>, I22>>);
  static_assert(not identity_matrix<EuclideanMean<angle::Radians, I22>>);
  static_assert(not identity_matrix<EuclideanMean<Axes<2>, M23>>);
  static_assert(not zero_matrix<Mat23>);
  static_assert(zero_matrix<EuclideanMean<angle::Radians, Z22>>);
  static_assert(zero_matrix<EuclideanMean<Axes<2>, Z22>>);
  static_assert(zero_matrix<EuclideanMean<C2, ZeroMatrix<M33>>>);

  EXPECT_TRUE(is_near(MatrixTraits<Mat23>::make(
    make_native_matrix<double, 3, 3>(1, 2, 3, 4, 5, 6, 7, 8, 9)).nested_matrix(), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_TRUE(is_near(MatrixTraits<Mat23>::zero(), native_matrix_t<double, 3, 3>::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<EuclideanMean<Axes<2>, I22>>::identity(), native_matrix_t<double, 2, 2>::Identity()));
}


TEST_F(matrices, EuclideanMean_overloads)
{
  EXPECT_TRUE(is_near(nested_matrix(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  EXPECT_TRUE(is_near(make_native_matrix(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  EXPECT_TRUE(is_near(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9} * 2), TM33 {2, 4, 6, 8, 10, 12, 14, 16, 18}));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9} * 2))>, Mat23>);

  EXPECT_TRUE(is_near(from_euclidean(
    EuclideanMean<C2, M33> {1, 2, 3,
                            0.5, std::sqrt(3)/2, sqrt2/2,
                            std::sqrt(3)/2, 0.5, sqrt2/2}),
    Mean<C2, M23> {1, 2, 3,
                   pi/3, pi/6, pi/4}));

  const auto m1 = make_euclidean_mean(-2., 5, 3);
  EXPECT_TRUE(is_near(from_euclidean(m1), m1));

  using A3 = Coefficients<angle::Radians, Axis, angle::Radians>;
  const auto m2 = make_euclidean_mean<A3>(std::sqrt(3) / 2, 0.5, 5, 0.5, -std::sqrt(3) / 2);
  const auto x2 = (native_matrix_t<double, 3, 1> {} << pi / 6, 5, -pi / 3).finished();
  EXPECT_TRUE(is_near(from_euclidean(m2).nested_matrix(), x2));

  EXPECT_TRUE(is_near(to_diagonal(EuclideanMean<Axes<2>, M21> {2, 3}).nested_matrix(), TM22 {2, 0, 0, 3}));
  static_assert(diagonal_matrix<decltype(to_diagonal(EuclideanMean<Axes<2>, M21> {2, 3}))>);
  static_assert(typed_matrix<decltype(to_diagonal(EuclideanMean<Axes<2>, M21> {2, 3}))>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(to_diagonal(EuclideanMean<Axes<2>, M21> {2, 3}))>::ColumnCoefficients, Axes<2>>);

  EXPECT_TRUE(is_near(transpose(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}).nested_matrix(), TM32 {1, 4, 2, 5, 3, 6}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(transpose(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}))>::RowCoefficients, Axes<3>>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(transpose(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}))>::ColumnCoefficients, Axes<2>>);

  EXPECT_TRUE(is_near(adjoint(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}).nested_matrix(), TM32 {1, 4, 2, 5, 3, 6}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(adjoint(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}))>::RowCoefficients, Axes<3>>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(adjoint(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}))>::ColumnCoefficients, Axes<2>>);

  EXPECT_NEAR(determinant(EuclideanMean<Axes<3>, M33> {1, 2, 3, 4, 5, 6, 1, 5, 1}), 24, 1e-6);

  EXPECT_NEAR(trace(EuclideanMean<Axes<3>, M33> {1, 2, 3, 4, 5, 6, 7, 8, 9}), 15, 1e-6);

  EXPECT_TRUE(is_near(solve(EuclideanMean<Axes<2>, M22> {9., 3, 3, 10}, EuclideanMean<Axes<2>, M21> {15, 23}), EuclideanMean<Axes<2>, M21> {1, 2}));

  EXPECT_TRUE(is_near(reduce_columns(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), Mat21 {2, 5, 8}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})), TM22 {14., 32, 32, 77}));

  EXPECT_TRUE(is_near(square(QR_decomposition(EuclideanMean<Axes<3>, M32> {1, 4, 2, 5, 3, 6})), TM22 {14., 32, 32, 77}));

  using N = std::normal_distribution<double>::param_type;
  Mat23 m = Mat23::zero();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat23>(N {1.0, 0.3}, N {2.0, 0.3}, N {3.0, 0.3})) / (i + 1);
  }
  Mat23 offset = {1, 1, 1, 2, 2, 2, 3, 3, 3};
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));
}


TEST_F(matrices, EuclideanMean_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(Mat22 {1, 2, 3, 4, 5, 6}, Mat12 {7, 8}), TM42 {1, 2, 3, 4, 5, 6, 7, 8}));
  static_assert(euclidean_mean<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4, 5, 6}, Mat12 {7, 8}))>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4, 5, 6}, Mat12 {7, 8}))>::RowCoefficients, C3>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5, 7, 8}, Mat21 {3, 6, 9}), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));
  static_assert(euclidean_mean<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5, 7, 8}, Mat21 {3, 6, 9}))>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5, 7, 8}, Mat21 {3, 6, 9}))>::RowCoefficients, C2>);

  EXPECT_TRUE(is_near(split_vertical(Mat32 {1, 2, 3, 4, 5, 6, 7, 8}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C2, Axis>(Mat32 {1, 2, 3, 4, 5, 6, 7, 8}), std::tuple {Mat22 {1, 2, 3, 4, 5, 6}, Mat12 {7, 8}}));
  EXPECT_TRUE(is_near(split_horizontal<Axes<2>, Axis>(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), std::tuple {Mat22 {1, 2, 4, 5, 7, 8}, Mat21 {3, 6, 9}}));
  EXPECT_TRUE(is_near(split_vertical<Axis, angle::Radians>(Mat32 {1, 2, 3, 4, 5, 6, 7, 8}), std::tuple {Mat12 {1, 2},  EuclideanMean<angle::Radians, M22> {3, 4, 5, 6}}));
  EXPECT_TRUE(is_near(split_horizontal<Axis, Axis>(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), std::tuple {Mat21 {1, 4, 7}, Mat21 {2, 5, 8}}));

  EXPECT_TRUE(is_near(column(Mat22 {1, 2, 3, 4, 5, 6}, 0), Mean{1., 3, 5}));
  EXPECT_TRUE(is_near(column(Mat22 {1, 2, 3, 4, 5, 6}, 1), Mean{2., 4, 6}));

  EXPECT_TRUE(is_near(column<0>(Mat22 {1, 2, 3, 4, 5, 6}), Mean{1., 3, 5}));
  EXPECT_TRUE(is_near(column<1>(Mat22 {1, 2, 3, 4, 5, 6}), Mean{2., 4, 6}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(column<0>(Mat22 {1, 2, 3, 4, 5, 6}))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(column<1>(Mat22 {1, 2, 3, 4, 5, 6}))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(column<0>(Mat22 {1, 2, 3, 4, 5, 6}))>::ColumnCoefficients, Axis>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(column<1>(Mat22 {1, 2, 3, 4, 5, 6}))>::ColumnCoefficients, Axis>);

  auto m = Mat22 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(apply_columnwise(m, [](auto& col){ col *= 2; }), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_columnwise(m, [](auto& col, std::size_t i){ col *= i; }), Mat22 {0, 4, 0, 8, 0, 12}));

  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4, 5, 6}, [](auto col){ return col * 2; }), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4, 5, 6}, [](const auto col){ return col * 2; }), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4, 5, 6}, [](auto&& col){ return col * 2; }), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4, 5, 6}, [](const auto& col){ return col * 2; }), Mat22 {2, 4, 6, 8, 10, 12}));

  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4, 5, 6}, [](auto col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4, 0, 6}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4, 5, 6}, [](const auto col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4, 0, 6}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4, 5, 6}, [](auto&& col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4, 0, 6}));
  EXPECT_TRUE(is_near(apply_columnwise(Mat22 {1, 2, 3, 4, 5, 6}, [](const auto& col, std::size_t i){ return col * i; }), Mat22 {0, 2, 0, 4, 0, 6}));

  EXPECT_TRUE(is_near(apply_columnwise<2>([] { return Mat21 {1, 2, 3}; }), Mat22 {1, 1, 2, 2, 3, 3}));
  EXPECT_TRUE(is_near(apply_columnwise<2>([](std::size_t i){ return Mat21 {i + 1., 2*i + 1, 2*i + 2}; }), Mat22 {1, 2, 1, 3, 2, 4}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Mat21()>()))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Mat21()>()))>::ColumnCoefficients, Axes<2>>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Mat21(std::size_t)>()))>::RowCoefficients, C2>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(apply_columnwise<2>(std::declval<Mat21(std::size_t)>()))>::ColumnCoefficients, Axes<2>>);

  auto n = Mat22 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(apply_coefficientwise(n, [](auto& x){ x *= 2; }), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_coefficientwise(n, [](auto& x, std::size_t i, std::size_t j){ x += i + j; }), Mat22 {2, 5, 7, 10, 12, 15}));

  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4, 5, 6}, [](auto x){ return x + 1; }), Mat22 {2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4, 5, 6}, [](const auto x){ return x + 1; }), Mat22 {2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4, 5, 6}, [](auto&& x){ return x + 1; }), Mat22 {2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4, 5, 6}, [](const auto& x){ return x + 1; }), Mat22 {2, 3, 4, 5, 6, 7}));

  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4, 5, 6}, [](auto x, std::size_t i, std::size_t j){ return x + i + j; }), Mat22 {1, 3, 4, 6, 7, 9}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4, 5, 6}, [](const auto x, std::size_t i, std::size_t j){ return x + i + j; }), Mat22 {1, 3, 4, 6, 7, 9}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4, 5, 6}, [](auto&& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat22 {1, 3, 4, 6, 7, 9}));
  EXPECT_TRUE(is_near(apply_coefficientwise(Mat22 {1, 2, 3, 4, 5, 6}, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }), Mat22 {1, 3, 4, 6, 7, 9}));

  EXPECT_TRUE(is_near(apply_coefficientwise<Mat22>([] { return 2; }), Mat22 {2, 2, 2, 2, 2, 2}));
  EXPECT_TRUE(is_near(apply_coefficientwise<Mat22>([](std::size_t i, std::size_t j){ return 1 + i + j; }), Mat22 {1, 2, 2, 3, 3, 4}));
}


TEST_F(matrices, EuclideanMean_arithmetic)
{
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2, 1, 0} + Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, TM42 {8, 8, 8, 8, 8, 8, 8, 8}));
  EXPECT_TRUE(is_near(EuclideanMean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} + Mean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, TM32 {8, 8, 8, 8, 8, 8}));
  EXPECT_TRUE(is_near(EuclideanMean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} + Matrix<Axes<3>, Axes<2>, M32> {1, 2, 3, 4, 5, 6}, TM32 {8, 8, 8, 8, 8, 8}));
  static_assert(euclidean_mean<decltype(Mat32 {7, 6, 5, 4, 3, 2, 1, 0} + Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, Mat32 {8, 8, 8, 8, 8, 8, 8, 8})>);
  static_assert(typed_matrix<decltype(EuclideanMean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} + Mean<Axes<3>, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(EuclideanMean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} + Matrix<Axes<3>, Axes<2>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2, 1, 0} - Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, TM42 {6, 4, 2, 0, -2, -4, -6, -8}));
  EXPECT_TRUE(is_near(EuclideanMean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} - Mean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, TM32 {6, 4, 2, 0, -2, -4}));
  EXPECT_TRUE(is_near(EuclideanMean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}, TM32 {6, 4, 2, 0, -2, -4}));
  static_assert(euclidean_mean<decltype(Mat32 {7, 6, 5, 4, 3, 2, 1, 0} - Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, Mat32 {8, 8, 8, 8, 8, 8, 8, 8})>);
  static_assert(typed_matrix<decltype(EuclideanMean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} - Mean<Axes<3>, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(EuclideanMean<Axes<3>, M32> {7, 6, 5, 4, 3, 2} - Matrix<Axes<3>, Axes<2>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {1, 2, 3, 4, 5, 6, 7, 8} * 2, TM42 {2, 4, 6, 8, 10, 12, 14, 16}));
  EXPECT_TRUE(is_near(2 * Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, TM42 {2, 4, 6, 8, 10, 12, 14, 16}));
  EXPECT_TRUE(is_near(Mat32 {2, 4, 6, 8, 10, 12, 14, 16} / 2, TM42 {1, 2, 3, 4, 5, 6, 7, 8}));
  static_assert(euclidean_mean<decltype(Mat32 {1, 2, 3, 4, 5, 6, 7, 8} * 2, Mat32 {2, 4, 6, 8, 10, 12, 14, 16})>);
  static_assert(euclidean_mean<decltype(2 * Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, Mat32 {2, 4, 6, 8, 10, 12, 14, 16})>);
  static_assert(euclidean_mean<decltype(Mat32 {2, 4, 6, 8, 10, 12, 14, 16} / 2, Mat32 {1, 2, 3, 4, 5, 6, 7, 8})>);

  using Mat12a = EuclideanMean<angle::Radians, M22>;
  EXPECT_TRUE(is_near(Mat12a {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6,}, TM23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat12a {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6,})>::RowCoefficients, angle::Radians>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat12a {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6,})>::ColumnCoefficients, Axes<3>>);
  static_assert(euclidean_mean<decltype(Mat12a {1, 2, 3, 4} * EuclideanMean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat12a {1, 2, 3, 4} * Matrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6}, TM23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat12a {1, 2, 3, 4} * Matrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, angle::Radians>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat12a {1, 2, 3, 4} * Matrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, Axes<3>>);
  static_assert(euclidean_mean<decltype(Mat12a {1, 2, 3, 4} * Matrix<Axes<2>, Axes<3>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat12a {1, 2, 3, 4} * Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6}, TM23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat12a {1, 2, 3, 4} * Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::RowCoefficients, angle::Radians>);
  static_assert(equivalent_to<typename MatrixTraits<decltype(Mat12a {1, 2, 3, 4} * Mean<Axes<2>, M23> {1, 2, 3, 4, 5, 6})>::ColumnCoefficients, Axes<3>>);

  EXPECT_TRUE(is_near(-Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, Mat32 {-1, -2, -3, -4, -5, -6, -7, -8}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4, 5, 6} == Mat22 {1, 2, 3, 4, 5, 6}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4, 5, 6} != Mat22 {1, 2, 2, 4, 5, 6}));
  EXPECT_FALSE((Mat22 {1, 2, 3, 4, 5, 6} == EuclideanMean<Axes<3>, M32> {1, 2, 3, 4, 5, 6}));
}


TEST_F(matrices, Polar_Spherical_toEuclideanExpr)
{
  using P1 = Polar<Distance, angle::Radians>;
  const Mean<P1> m1 {2, pi / 6};
  const auto x1 = make_euclidean_mean<P1>(2, std::sqrt(3) / 2, 0.5);
  EXPECT_TRUE(is_near(to_euclidean(m1), x1));

  using P2 = Polar<angle::Radians, Distance>;
  const Mean<P2> m2 {pi / 6, 2};
  const auto x2 = make_euclidean_mean<P2>(std::sqrt(3) / 2, 0.5, 2);
  EXPECT_TRUE(is_near(to_euclidean(m2), x2));

  using S3 = Spherical<Distance, angle::Radians, inclination::Radians>;
  const Mean<S3> m3 {2, pi / 6, -pi / 3};
  const auto x3 = make_euclidean_mean<S3>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(m3), x3));

  using S4 = Spherical<Distance, inclination::Radians, angle::Radians>;
  const Mean<S4> m4 {2, -pi / 3, pi / 6};
  const auto x4 = make_euclidean_mean<S4>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(m4), x4));

  using S5 = Spherical<angle::Radians, Distance, inclination::Radians>;
  const Mean<S5> m5 {pi / 6, 2, -pi / 3};
  const auto x5 = make_euclidean_mean<S5>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(m5), x5));

  using S6 = Spherical<inclination::Radians, Distance, angle::Radians>;
  const Mean<S6> m6 {-pi / 3, 2, pi / 6};
  const auto x6 = make_euclidean_mean<S6>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(m6), x6));

  using S7 = Spherical<angle::Radians, inclination::Radians, Distance>;
  const Mean<S7> m7 {pi / 6, -pi / 3, 2};
  const auto x7 = make_euclidean_mean<S7>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(m7), x7));

  using S8 = Spherical<inclination::Radians, angle::Radians, Distance>;
  const Mean<S8> m8 {-pi / 3, pi / 6, 2};
  const auto x8 = make_euclidean_mean<S8>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(m8), x8));
}


TEST_F(matrices, Polar_Spherical_fromEuclideanExpr)
{
  using P1 = Polar<Distance, angle::Radians>;
  const Mean<P1> m1 {2, pi / 6};
  const auto x1 = make_euclidean_mean<P1>(2, std::sqrt(3) / 2, 0.5);
  EXPECT_TRUE(is_near(m1, from_euclidean(x1)));

  using P2 = Polar<angle::Radians, Distance>;
  const Mean<P2> m2 {pi / 6, 2};
  const auto x2 = make_euclidean_mean<P2>(std::sqrt(3) / 2, 0.5, 2);
  EXPECT_TRUE(is_near(m2, from_euclidean(x2)));

  using S3 = Spherical<Distance, angle::Radians, inclination::Radians>;
  const Mean<S3> m3 {2, pi / 6, -pi / 3};
  const auto x3 = make_euclidean_mean<S3>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(m3, from_euclidean(x3)));

  using S4 = Spherical<Distance, inclination::Radians, angle::Radians>;
  const Mean<S4> m4 {2, -pi / 3, pi / 6};
  const auto x4 = make_euclidean_mean<S4>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(m4, from_euclidean(x4)));

  using S5 = Spherical<angle::Radians, Distance, inclination::Radians>;
  const Mean<S5> m5 {pi / 6, 2, -pi / 3};
  const auto x5 = make_euclidean_mean<S5>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(m5, from_euclidean(x5)));

  using S6 = Spherical<inclination::Radians, Distance, angle::Radians>;
  const Mean<S6> m6 {-pi / 3, 2, pi / 6};
  const auto x6 = make_euclidean_mean<S6>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(m6, from_euclidean(x6)));

  using S7 = Spherical<angle::Radians, inclination::Radians, Distance>;
  const Mean<S7> m7 {pi / 6, -pi / 3, 2};
  const auto x7 = make_euclidean_mean<S7>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(m7, from_euclidean(x7)));

  using S8 = Spherical<inclination::Radians, angle::Radians, Distance>;
  const Mean<S8> m8 {-pi / 3, pi / 6, 2};
  const auto x8 = make_euclidean_mean<S8>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(m8, from_euclidean(x8)));
}

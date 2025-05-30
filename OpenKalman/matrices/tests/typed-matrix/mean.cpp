/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "typed-matrix.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;
using namespace OpenKalman::test;

using numbers::pi;
using numbers::sqrt2;

using M12 = eigen_matrix_t<double, 1, 2>;
using M21 = eigen_matrix_t<double, 2, 1>;
using M22 = eigen_matrix_t<double, 2, 2>;
using M23 = eigen_matrix_t<double, 2, 3>;
using M32 = eigen_matrix_t<double, 3, 2>;
using M33 = eigen_matrix_t<double, 3, 3>;
using I22 = Eigen3::IdentityMatrix<M22>;
using Z22 = ZeroAdapter<eigen_matrix_t<double, 2, 2>>;
using C2 = std::tuple<Axis, angle::Radians>;
using C3 = std::tuple<Axis, angle::Radians, Axis>;
using Mat12 = Mean<Axis, M12>;
using Mat21 = Mean<C2, M21>;
using Mat22 = Mean<C2, M22>;
using Mat23 = Mean<C2, M23>;
using Mat32 = Mean<C3, M32>;
using Mat33 = Mean<C3, M33>;
using TMat22 = Matrix<C2, Dimensions<2>, M22>;
using TMat23 = Matrix<C2, Dimensions<3>, M23>;
using TMat32 = Matrix<C3, Dimensions<2>, M32>;
using EMat23 = EuclideanMean<C2, M33>;

using SA2l = HermitianAdapter<M22, TriangleType::lower>;
using SA2u = HermitianAdapter<M22, TriangleType::upper>;
using T2l = TriangularAdapter<M22, TriangleType::lower>;
using T2u = TriangularAdapter<M22, TriangleType::upper>;

inline I22 i22 = M22::Identity();
inline Z22 z22 = Z22();
inline auto covi22 = Covariance<C2, I22>(i22);
inline auto covz22 = Covariance<C2, Z22>(z22);
inline auto sqcovi22 = SquareRootCovariance<C2, I22>(i22);
inline auto sqcovz22 = SquareRootCovariance<C2, Z22>(z22);


TEST(matrices, Mean_class)
{
  // Default constructor and Eigen3 construction
  Mat23 mat23a;
  mat23a << 1, 2, 3, 4, 5, 6;
  EXPECT_TRUE(is_near(mat23a, TMat23 {1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi}));

  // Copy constructor
  Mat23 mat23b = const_cast<const Mat23&>(mat23a);
  EXPECT_TRUE(is_near(mat23b, TMat23 {1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi}));

  // Move constructor
  auto xa = Mat23 {6, 5, 4, 3, 2, 1};
  Mat23 mat23c(std::move(xa));
  EXPECT_TRUE(is_near(mat23c, TMat23 {6, 5, 4, 3, 2, 1}));

  // Convert from different covariance types
  Mat23 mat23_x1(Matrix<C2, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x1, Mat23 {1, 2, 3, 4, 5, 6}));
  Mean<Dimensions<2>, M23> mat23_x2(EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x2, Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}));
  Mat23 mat23_x3(EuclideanMean<C2, M33> {
    1, 2, 3,
    0.5, std::sqrt(3)/2, sqrt2/2,
    std::sqrt(3)/2, 0.5, sqrt2/2});
  EXPECT_TRUE(is_near(mat23_x3, Mat23 {1, 2, 3, pi/3, pi/6, pi/4}));

  // Construct from a typed_matrix_nestable
  Mat23 mat23d(make_dense_object_from<M23>(1, 2, 3, 4, 5, 6));
  EXPECT_TRUE(is_near(mat23d, TMat23 {1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi}));

  // Construct from a list of coefficients
  Mat23 mat23e {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23e, TMat23 {1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi}));

  // Copy assignment
  mat23c = mat23b;
  EXPECT_TRUE(is_near(mat23c, TMat23 {1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi}));

  // Move assignment
  auto ya = Mat23 {3, 4, 5, 6, 7, 8};
  mat23c = std::move(ya);
  EXPECT_TRUE(is_near(mat23c, TMat23 {3, 4, 5, 6-2*pi, 7-2*pi, 8-2*pi}));

  // assign from different covariance types
  mat23_x1 = Matrix<C2, Dimensions<3>, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x1, Mat23 {6, 5, 4, 3, 2, 1}));
  mat23_x2 = EuclideanMean<Dimensions<2>, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x2, Mat23 {6, 5, 4, 3, 2, 1}));
  mat23_x3 = EuclideanMean<C2, M33> {
    3, 2, 1,
    std::sqrt(3)/2, sqrt2/2, 0.5,
    0.5, sqrt2/2, std::sqrt(3)/2};
  EXPECT_TRUE(is_near(mat23_x3, Mat23 {3, 2, 1, pi/6, pi/4, pi/3}));

  // assign from a regular matrix
  mat23e = make_dense_object_from<M23>(3, 4, 5, 6, 7, 8);

  // Assign from a list of coefficients (via move assignment operator)
  mat23e = {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23e, Mat23 {6, 5, 4, 3, 2, 1}));

  // Increment
  mat23_x1 += TMat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23_x1, TMat23 {7, 7, 7, 7-2*pi, 7-2*pi, 7-2*pi}));
  mat23a += Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23a, TMat23 {2, 4, 6, 8-2*pi, 10-4*pi, 12-4*pi}));

  // Increment with a stochastic value
  mat23b = {1, 1, 1, pi, pi, pi};
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
  EXPECT_TRUE(is_near(mat23a, TMat23 {1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi}));

  // Decrement with a stochastic value
  mat23b = {1, 1, 1, pi, pi, pi};
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
  EXPECT_TRUE(is_near(mat23a, TMat23 {2, 4, 6, 8-2*pi, 10-4*pi, 12-4*pi}));

  // Scalar division
  mat23a /= 2;
  EXPECT_TRUE(is_near(mat23a, TMat23 {1, 2, 3, 4-pi, 5-2*pi, 6-2*pi}));

  // Scalar multiplication, zero
  mat23a *= 0;
  EXPECT_TRUE(is_near(mat23a, M23::Zero()));

  // Zero
  EXPECT_TRUE(is_near(make_zero<Mat23>(), M23::Zero()));

  // Identity
  EXPECT_TRUE(is_near(make_identity_matrix_like<Mat22>(), M22::Identity()));
}


TEST(matrices, Mean_subscripts)
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

  static_assert(writable_by_component<Mat23&, 2>);
  static_assert(not writable_by_component<Mat23&, 1>);
  static_assert(not writable_by_component<const Mat23&, 2>);
  static_assert(not writable_by_component<const Mat23&, 1>);
  static_assert(writable_by_component<Mat21&, 2>);
  static_assert(writable_by_component<Mat21&, 1>);
  static_assert(not writable_by_component<const Mat21&, 2>);
  static_assert(not writable_by_component<const Mat21&, 1>);
  static_assert(writable_by_component<Matrix<C3, C2, M32>&, 2>);
  static_assert(not writable_by_component<Matrix<C3, C2, M32>&, 1>);
  static_assert(not writable_by_component<Matrix<C3, C2, const M32>&, 2>);
  static_assert(not writable_by_component<Matrix<C3, C2, const M32>&, 1>);
  static_assert(writable_by_component<Matrix<C2, Axis, M21>&, 2>);
  static_assert(writable_by_component<Matrix<C2, Axis, M21>&, 1>);
  static_assert(not writable_by_component<Matrix<C2, Axis, const M21>&, 2>);
  static_assert(not writable_by_component<Matrix<C2, Axis, const M21>&, 1>);

  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 0), 1, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 1), 2, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 2), 3, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 0), 4-2*pi, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 1), 5-2*pi, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 2), 6-2*pi, 1e-6);
}


TEST(matrices, Mean_deduction_guides)
{
  auto a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  EXPECT_TRUE(is_near(Mean(a), a));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mean(a)), 0>, Dimensions<2>>);

  auto b1 = Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(Mean(b1), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mean(b1)), 0>, C2>);

  auto b2 = Matrix<C2, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(Mean(b2), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mean(b2)), 0>, C2>);

  auto b3 = EuclideanMean<C2, M33> {1, 2, 3, std::sqrt(3)/2, 0.5, std::sqrt(2)/2, 0.5, std::sqrt(3)/2, std::sqrt(2)/2};
  EXPECT_TRUE(is_near(Mean(b3), Mat23 {1, 2, 3, pi/6, pi/3, pi/4}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mean(b3)), 0>, C2>);
  static_assert(index_dimension_of_v<decltype(Mean(b3)), 0> == 2);
}


TEST(matrices, Mean_make_functions)
{
  auto a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  EXPECT_TRUE(is_near(make_mean<C2>(a), Mat23{a}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(make_mean<C2>(a)), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(make_mean<C2>(a)), 1>, Dimensions<3>>);

  auto b = Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(make_mean(b), Mat23{a}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(make_mean(b)), 0>, C2>);

  static_assert(compares_with<vector_space_descriptor_of_t<decltype(make_mean<C2, M23>()), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(make_mean<M23>()), 0>, Dimensions<2>>);
}


TEST(matrices, Mean_traits)
{
  static_assert(typed_matrix<Mat23>);
  static_assert(mean<Mat23>);
  static_assert(not euclidean_mean<Mat23>);
  static_assert(not euclidean_transformed<Mat23>);
  static_assert(wrapped_mean<Mat23>);
  static_assert(not wrapped_mean<Mean<Dimensions<2>, I22>>);
  static_assert(untyped_columns<Mat23>);

  static_assert(not identity_matrix<Mat23>);
  static_assert(identity_matrix<Mean<Dimensions<2>, I22>>);
  static_assert(not identity_matrix<Mean<Dimensions<2>, M23>>);
  static_assert(not zero<Mat23>);
  static_assert(zero<Mean<C2, Z22>>);
  static_assert(zero<Mean<C2, ZeroAdapter<eigen_matrix_t<double, 2, 2>>>>);

  EXPECT_TRUE(is_near(make_zero<Mat23>(), eigen_matrix_t<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Mat22>(), eigen_matrix_t<double, 2, 2>::Identity()));
}


TEST(matrices, Mean_overloads)
{
  auto w_4 = 4 - 2*pi;
  auto w_5 = 5 - 2*pi;
  auto w_6 = 6 - 2*pi;
  EXPECT_TRUE(is_near(Mean<C2, M23> {1, 2, 3, pi*7/3, pi*13/6, -pi*7/4},
    Mat23 {1, 2, 3, pi/3, pi/6, pi/4}));

  EXPECT_TRUE(is_near(nested_object(Mat23 {1, 2, 3, 4, 5, 6}), TMat23 {1, 2, 3, w_4, w_5, w_6}));

  EXPECT_TRUE(is_near(make_dense_object_from(Mat23 {1, 2, 3, 4, 5, 6}), Mat23 {1, 2, 3, w_4, w_5, w_6}));

  EXPECT_TRUE(is_near(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6} * 2), Mat23 {2, 4, 6, 8-2*pi, 10-4*pi, 12-4*pi}));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6} * 2))>,
    Mean<C2, std::decay_t<decltype(wrap_angles<C2>(std::declval<M23>()))>>>);

  EXPECT_TRUE(is_near(to_euclidean(
    Mean<C2, M23> {1, 2, 3, pi*7/3, pi*13/6, -pi*7/4}),
    EuclideanMean<C2, M33> {1, 2, 3,
                            0.5, std::sqrt(3)/2, sqrt2/2,
                            std::sqrt(3)/2, 0.5, sqrt2/2}));

  const auto ma = make_mean(-2., 5, 3);
  EXPECT_TRUE(is_near(to_euclidean(ma), ma));

  using A3 = std::tuple<angle::Radians, Axis, angle::Radians>;
  const auto mb = make_mean<A3>(pi / 6, 5, -pi / 3);
  const auto x2 = (eigen_matrix_t<double, 5, 1> {} << std::sqrt(3) / 2, 0.5, 5, 0.5, -std::sqrt(3) / 2).finished();
  EXPECT_TRUE(is_near(to_euclidean(mb).nested_object(), x2));

  EXPECT_TRUE(is_near(to_diagonal(Mat21 {2, 3}).nested_object(), Mat22 {2, 0, 0, 3}));
  static_assert(diagonal_matrix<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(typed_matrix<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(to_diagonal(Mat21 {2, 3})), 1>, C2>);

  EXPECT_TRUE(is_near(to_diagonal(Mat21 {5, 6}).nested_object(), Mat22 {5, 0, 0, w_6}));
  static_assert(diagonal_matrix<decltype(to_diagonal(Mat21 {5, 6}))>);
  static_assert(typed_matrix<decltype(to_diagonal(Mat21 {5, 6}))>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(to_diagonal(Mat21 {5, 6})), 1>, C2>);

  EXPECT_TRUE(is_near(transpose(Mat23 {1, 2, 3, 4, 5, 6}).nested_object(), TMat32 {1, w_4, 2, w_5, 3, w_6}));
  EXPECT_TRUE(is_near(adjoint(Mat23 {1, 2, 3, 4, 5, 6}).nested_object(), TMat32 {1, w_4, 2, w_5, 3, w_6}));

  EXPECT_NEAR(determinant(Mean<Dimensions<2>, M22> {1, 2, 3, 4}), -2, 1e-6);

  EXPECT_NEAR(trace(Mean<Dimensions<2>, M22> {1, 2, 3, 4}), 5, 1e-6);

  EXPECT_TRUE(is_near(solve(Mean<Dimensions<2>, M22> {9., 3, 3, 10}, Mean<Dimensions<2>, M21> {15, 23}), Mean<Dimensions<2>, M21> {1, 2}));

  EXPECT_TRUE(is_near(average_reduce<1>(Mat23 {1, 2, 3, pi/3, pi/4, pi/6}), Mat21 {2, pi/4}));
  EXPECT_TRUE(is_near(average_reduce<1>(Mat23 {1, 2, 3, 4, 5, 6}), Mat21 {2, w_5}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(Mat23 {1, 2, 3, 4, 5, 6})),
    TMat22 {14, w_4 + 2*w_5 + 3*w_6, w_4 + 2*w_5 + 3*w_6, w_4*w_4 + w_5*w_5 + w_6*w_6}));

  EXPECT_TRUE(is_near(square(QR_decomposition(Mean<Dimensions<3>, M32> {1, 4, 2, 5, 3, 6})), TMat22 {14, 32, 32, 77}));

  using N = std::normal_distribution<double>;
  EMat23 m = make_zero<EMat23>();
  for (int i=0; i<100; i++)
  {
    m = (m * i + to_euclidean(randomize<Mat23>(N {1.0, 0.3}, 2.0))) / (i + 1);
  }
  Mat23 offset = {1, 1, 1, 2, 2, 2};
  EXPECT_TRUE(is_near(from_euclidean(m), offset, 0.1));
  EXPECT_FALSE(is_near(from_euclidean(m), offset, 1e-6));

}


TEST(matrices, Mean_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}), Mat32 {1, 2, 3, 4, 5, 6}));
  static_assert(mean<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}))>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6})), 0>, C3>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(mean<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}))>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6})), 0>, C2>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Matrix<C2, angle::Radians, M21> {3, 6}),
    Matrix<C2, std::tuple<Axis, Axis, angle::Radians>, M23> {1, 2, 3, 4-2*pi, 5-2*pi, 6}));
  static_assert(not mean<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Matrix<C2, angle::Radians, M21> {3, 6}))>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Matrix<C2, angle::Radians, M21> {3, 6})), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Matrix<C2, angle::Radians, M21> {3, 6})), 1>, std::tuple<Axis, Axis, angle::Radians>>);

  EXPECT_TRUE(is_near(split_vertical(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C2, Axis>(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {Mat22 {1, 2, 3, 4}, Mat12 {5, 6}}));
  EXPECT_TRUE(is_near(split_horizontal<Dimensions<2>, Axis>(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {Mat22 {1, 2, 4, 5}, Mat21 {3, 6}}));
  EXPECT_TRUE(is_near(split_vertical<Axis, angle::Radians>(Mat32 {1, 2, 3, 4, 5, 6}), std::tuple {Mat12 {1, 2}, Mat12 {3, 4-2*pi}}));
  EXPECT_TRUE(is_near(split_horizontal<Axis, Axis>(Mat23 {1, 2, 3, 4, 5, 6}), std::tuple {Mat21 {1, 4}, Mat21 {2, 5}}));

  EXPECT_TRUE(is_near(column(Mat22 {1, 2, 3, 4}, 0), Mean{1., 3}));
  EXPECT_TRUE(is_near(column(Mat22 {1, 2, 3, 4}, 1), Mean{2., 4-2*pi}));

  EXPECT_TRUE(is_near(column<0>(Mat22 {1, 2, 3, 4}), Mean{1., 3}));
  EXPECT_TRUE(is_near(column<1>(Mat22 {1, 2, 3, 4}), Mean{2., 4-2*pi}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(column<0>(Mat22 {1, 2, 3, 4})), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(column<1>(Mat22 {1, 2, 3, 4})), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(column<0>(Mat22 {1, 2, 3, 4})), 1>, Axis>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(column<1>(Mat22 {1, 2, 3, 4})), 1>, Axis>);

  auto m = Mat22 {1, 2, 3, 4};
  EXPECT_TRUE(is_near(apply_columnwise([](auto& col){ col *= 2; }, m), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto& col, std::size_t i){ col *= i; }, m), Mat22 {0, 4, 0, 8}));

  EXPECT_TRUE(is_near(apply_columnwise([](auto col){ return make_self_contained(col * 2); }, Mat22 {1, 2, 3, 4}), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto col){ return make_self_contained(col * 2); }, Mat22 {1, 2, 3, 4}), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto&& col){ return make_self_contained(col * 2); }, Mat22 {1, 2, 3, 4}), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col * 2); }, Mat22 {1, 2, 3, 4}), Mat22 {2, 4, 6, 8}));

  EXPECT_TRUE(is_near(apply_columnwise([](auto col, std::size_t i){ return make_self_contained(col * i); }, Mat22 {1, 2, 3, 4}), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto col, std::size_t i){ return make_self_contained(col * i); }, Mat22 {1, 2, 3, 4}), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto&& col, std::size_t i){ return make_self_contained(col * i); }, Mat22 {1, 2, 3, 4}), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col * i); }, Mat22 {1, 2, 3, 4}), Mat22 {0, 2, 0, 4}));

  EXPECT_TRUE(is_near(apply_columnwise<2>([] { return Mat21 {1., 2}; }), Mat22 {1, 1, 2, 2}));
  EXPECT_TRUE(is_near(apply_columnwise<2>([](std::size_t i){ return Mat21 {i + 1., 2*i + 1}; }), Mat22 {1, 2, 1, 3}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<Mat21()>())), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<Mat21()>())), 1>, Dimensions<2>>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<Mat21(std::size_t)>())), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<Mat21(std::size_t)>())), 1>, Dimensions<2>>);

  const auto mat22_1234 = Mat22 {1, 2, 3, 4};
  auto n = mat22_1234;

  EXPECT_TRUE(is_near(apply_coefficientwise([](auto& x){ x *= 2; }, n), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, n), Mat22 {2, 5, 7, 10}));

  EXPECT_TRUE(is_near(apply_coefficientwise([](auto x){ return x + 1; }, mat22_1234), Mat22 {2, 3, 4, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto x){ return x + 1; }, mat22_1234), Mat22 {2, 3, 4, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](auto&& x){ return x + 1; }, mat22_1234), Mat22 {2, 3, 4, 5}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, mat22_1234), Mat22 {2, 3, 4, 5}));

  EXPECT_TRUE(is_near(apply_coefficientwise([](auto x, std::size_t i, std::size_t j){ return x + i + j; }, mat22_1234), Mat22 {1, 3, 4, 6}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto x, std::size_t i, std::size_t j){ return x + i + j; }, mat22_1234), Mat22 {1, 3, 4, 6}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](auto&& x, std::size_t i, std::size_t j){ return x + i + j; }, mat22_1234), Mat22 {1, 3, 4, 6}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, mat22_1234), Mat22 {1, 3, 4, 6}));

  EXPECT_TRUE(is_near(apply_coefficientwise<Mat22>([] { return 2; }), Mat22 {2, 2, 2, 2}));
  EXPECT_TRUE(is_near(apply_coefficientwise<Mat22>([](std::size_t i, std::size_t j){ return 1 + i + j; }), Mat22 {1, 2, 2, 3}));
}


TEST(matrices, Mean_arithmetic)
{
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6}, TMat32 {8, 8, 8-2*pi, 8-2*pi, 8, 8}));
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} + Matrix<C3, Dimensions<2>, M32> {1, 2, 3, 4, 5, 6}, TMat32 {8, 8, 8-2*pi, 8-2*pi, 8, 8}));
  EXPECT_TRUE(is_near(Mean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6}, Mean<Dimensions<3>, M32> {8, 8, 8, 8, 8, 8}));
  static_assert(mean<decltype(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(Mat32 {7, 6, 5, 4, 3, 2} + Matrix<C3, Dimensions<2>, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(Mean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6}, TMat32 {6, 4, 2, 0, -2, -4}));
  EXPECT_TRUE(is_near(Matrix<C3, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} - Mean<C3, M32> {1, 2, 3, 4, 5, 6}, TMat32 {6, 4, 2, 2*pi, -2, -4}));
  EXPECT_TRUE(is_near(Matrix<Dimensions<3>, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6}, TMat32 {6, 4, 2, 0, -2, -4}));
  static_assert(typed_matrix<decltype(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(Mat32 {7, 6, 5, 4, 3, 2} - Matrix<C3, Dimensions<2>, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(Mean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3-pi, 4-pi, 5, 6}));
  static_assert(mean<decltype(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(mean<decltype(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(mean<decltype(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}, TMat23 {9, 12, 15, 19 - 8*pi, 26 - 10*pi, 33 - 12*pi}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}), 0>, C2>);
  static_assert(typed_matrix<decltype(Mat22 {1, 2, 3, 4} * Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6}, TMat23 {9, 12, 15, 19 - 8*pi, 26 - 10*pi, 33 - 12*pi}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6}), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6}), 1>, Dimensions<3>>);
  static_assert(typed_matrix<decltype(Mat22 {1, 2, 3, 4} * Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 2} * Matrix<Dimensions<2>, C3, M23> {1, 2, 3, 3, 2, 1}, Matrix<Dimensions<2>, C3, M23> {7, 6, 5, 9, 10, 11}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Matrix<Dimensions<2>, C3, M23> {1, 2, 3, 4, 5, 6}), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Matrix<Dimensions<2>, C3, M23> {1, 2, 3, 4, 5, 6}), 1>, C3>);
  static_assert(typed_matrix<decltype(Mat22 {1, 2, 3, 4} * Matrix<Dimensions<2>, C3, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}, TMat23 {9, 12, 15, 19 - 8*pi, 26 - 10*pi, 33 - 12*pi}));
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Matrix<C2, Dimensions<2>, M22> {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}), 0>, C2>);
  static_assert(compares_with<vector_space_descriptor_of_t<decltype(Matrix<C2, Dimensions<2>, M22> {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}), 1>, Dimensions<3>>);
  static_assert(typed_matrix<decltype(Mat22 {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(-Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {-1, -2, -3, -4, -5, -6}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} == Mat22 {1, 2, 3, 4}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} != Mat22 {1, 2, 2, 4}));
  EXPECT_FALSE((Mat22 {1, 2, 3, 4} == Mean<Dimensions<2>, M22> {1, 2, 3, 4}));
}


TEST(matrices, Mean_angles_construct_coefficients)
{
  const auto v0 = make_mean<std::tuple<angle::Radians, Axis>>(eigen_matrix_t<double, 2, 1>(0.5, 2));
  EXPECT_EQ(v0[0], 0.5);
  EXPECT_EQ(v0[1], 2);
  const auto v1 = make_mean<std::tuple<Axis, angle::Radians>>(6., 7);
  EXPECT_EQ(v1[0], 6);
  EXPECT_EQ(v1[1], 7-2*pi);
  const auto v2 = make_mean<std::tuple<angle::Radians, Axis, angle::Radians>>(7., 8, 9);
  EXPECT_EQ(v2[0], 7-2*pi);
  EXPECT_EQ(v2[1], 8);
  EXPECT_EQ(v2[2], 9-2*pi);
  eigen_matrix_t<double, 3, 3> m33;
  m33 << 9, 3, 1,
    3, 8 - pi*2, 2,
    7, 1, 8;
  auto v3 = make_mean<double, std::tuple<Axis, angle::Radians, Axis>, 3>();
  v3 << 9, 3, 1,
    3, 8, 2,
    7, 1, 8;
  EXPECT_TRUE(is_near(nested_object(v3), m33));
  auto v3_1 = make_mean<std::tuple<Axis, angle::Radians, Axis>>(9., 3, 1, 3, 8, 2, 7, 1, 8);
  EXPECT_TRUE(is_near(v3_1, m33));
  auto v3_2 = v3;
  EXPECT_TRUE(is_near(v3_2, m33));
  static_assert(std::is_same_v<nested_object_of_t<decltype(v3)>, decltype(m33)>);
  auto v3_3 = make_mean<double, std::tuple<Axis, angle::Radians, Axis>, 3>();
  v3_3 << make_dense_object_from(v3);
  EXPECT_TRUE(is_near(v3_3, m33));
}


TEST(matrices, Mean_angle_concatenate_split)
{
  using C3 = std::tuple<angle::Radians, Axis, angle::Radians>;
  using Var3 = Mean<C3>;
  auto x1 = Var3 {5., 7, 9};
  auto x2 = Var3 {3., 2, 1};
  auto x3 = Mean<static_concatenate_t<C3, C3>> {5., 7, 9, 3, 2, 1};
  EXPECT_TRUE(is_near(concatenate(x1, x2), x3));
  auto x4 = Mean<static_concatenate_t<C3, C3, C3>> {5., 7, 9, 3, 2, 1, 5, 7, 9};
  EXPECT_TRUE(is_near(concatenate(x1, x2, x1), x4));
  auto [x5, x6] = split_vertical<C3, C3>(x3);
  EXPECT_TRUE(is_near(nested_object(x5), nested_object(x1)));
  EXPECT_TRUE(is_near(x5, x1));
  EXPECT_TRUE(is_near(x6, x2));
  const Var3 y1 {pi / 6, 2, pi / 3};
  const Var3 y2 {pi / 4, 3, pi / 4};
  const Mean<std::tuple<angle::Radians, Axis, angle::Radians, angle::Radians, Axis, angle::Radians>>
    y3 {pi / 6, 2, pi / 3, pi / 4, 3, pi / 4};
  EXPECT_TRUE(is_near(concatenate(y1, y2), y3));
  EXPECT_TRUE(is_near(split_vertical<C3, C3>(y3), std::tuple(y1, y2)));
  const Mean<std::tuple<angle::Radians, Axis, angle::Radians, angle::Radians, Axis, angle::Radians, angle::Radians, Axis, angle::Radians>>
    y4 {pi / 6, 2, pi / 3, pi / 4, 3, pi / 4, pi / 6, 2, pi / 3};
  EXPECT_TRUE(is_near(concatenate(y1, y2, y1), y4));
  EXPECT_TRUE(is_near(split_vertical<C3, C3, C3>(y4), std::tuple(y1, y2, y1)));
}


TEST(matrices, Mean_angle_Euclidean_conversion)
{
  using Var3 = Mean<std::tuple<angle::Radians, Axis, angle::Radians>>;
  const Var3 x1 {pi / 6, 5, -pi / 3};
  const Var3 x1p {pi / 5.99999999, 5, -pi / 2.99999999};
  const auto ma = (eigen_matrix_t<double, 3, 1> {} << pi / 6, 5, -pi / 3).finished();
  EXPECT_TRUE(is_near(x1, x1p));
  EXPECT_TRUE(is_near(x1.nested_object(), x1p.nested_object()));
  EXPECT_TRUE(is_near(to_euclidean(x1).nested_object(), to_euclidean(x1p).nested_object()));
  EXPECT_TRUE(is_near(to_euclidean(x1), to_euclidean(x1p)));
  EXPECT_NEAR(x1.nested_object()[0], pi / 6, 1e-6);
  EXPECT_NEAR(x1.nested_object()[1], 5, 1e-6);
  EXPECT_NEAR(x1.nested_object()[2], -pi / 3, 1e-6);
  EXPECT_TRUE(is_near(x1.nested_object(), ma));
  EXPECT_TRUE(is_near(make_mean(std::sqrt(3.)/2, 0.5, 5, 0.5, -std::sqrt(3.)/2), to_euclidean(x1).nested_object()));
  EXPECT_TRUE(is_near(Var3 {ma}, x1));
}


TEST(matrices, Mean_angle_mult_TypedMatrix)
{
  using M = Matrix<std::tuple<Axis, angle::Radians>, std::tuple<Axis, angle::Radians, Axis>>;
  using Vm = Mean<std::tuple<Axis, angle::Radians, Axis>>;
  using Rm = Matrix<std::tuple<Axis, angle::Radians>, Axis>;
  const M m {1, -2, 4,
             -5, 2, -1};
  const Vm vm {2, 1, -1};
  const Rm rm {-4, -7};
  static_assert(std::is_same_v<vector_space_descriptor_of_t<decltype(make_self_contained(m*vm)), 0>, std::tuple<Axis, angle::Radians>>);
  EXPECT_TRUE(is_near(m*vm, rm));
}


TEST(matrices, Mean_angle_arithmetic)
{
  using C = std::tuple<angle::Radians, Axis, angle::Radians, Axis, Axis>;
  using Var3 = Mean<C>;
  Var3 v1 {1, 2, 3, 4, 5};
  Var3 v2 {2, 4, -6, 8, -10};
  Var3 v3 {3, 6, -3, 12, -5};
  Var3 v4 {-1, -2, 9 - pi*2, -4, 15};
  Var3 v5 {3., 6, 9 - pi*2, 12, 15};
  EXPECT_TRUE(is_near(v1 + v2, v3));
  EXPECT_TRUE(is_near(v1 - v2, v4));
  EXPECT_TRUE(is_near(v1 * 3., v5));
  EXPECT_TRUE(is_near(3. * v1, v5));
  EXPECT_TRUE(is_near(v5 / 3. + Var3 {0, 0, pi*2/3, 0, 0}, v1));
  Var3 v7 = v1;
  v7 *= 3;
  EXPECT_TRUE(is_near(v7, v5));
  v7 /= 3;
  v7 += Var3 {0, 0, pi*2/3, 0, 0};
  EXPECT_TRUE(is_near(v7, v1));
  v7 -= v2;
  EXPECT_TRUE(is_near(v7, v4));
  v7 += v2;
  EXPECT_TRUE(is_near(v7, v1));
  auto v6 = make_mean(v1 + v2);
  static_assert(std::is_same_v<nested_object_of_t<decltype(v6)>&, decltype(v6.nested_object())>);
  v7 = v6;
  EXPECT_TRUE(is_near(v7, v6));
  v7 += v6;
  EXPECT_TRUE(is_near(v7, Var3 {6 - pi*2, 12, -6 + pi*2, 24, -10}));
  v7 -= v6;
  EXPECT_TRUE(is_near(v7, v6));
  static_assert(std::is_same_v<nested_object_of_t<decltype(v5)>&, decltype(v5.nested_object())>);
  static_assert(std::is_same_v<equivalent_self_contained_t<decltype((v5 + v2).nested_object())>,
    equivalent_self_contained_t<equivalent_self_contained_t<decltype(wrap_angles<C>(v5.nested_object()))>>>);
  static_assert(std::is_same_v<equivalent_self_contained_t<nested_object_of_t<decltype(make_self_contained(v5))>>,
    equivalent_self_contained_t<equivalent_self_contained_t<decltype(wrap_angles<C>(v5.nested_object()))>>>);
}


TEST(matrices, Mean_angle_arithmetic_Euclidean)
{
  using Var3 = Mean<std::tuple<angle::Radians, Axis, angle::Radians>>;
  Var3 x1 {pi / 6, 5, -pi / 3};
  Var3 x2 {2 * pi, 0, 6 * pi};
  auto x1e = to_euclidean(x1);
  auto x2e = to_euclidean(x2);
  EXPECT_TRUE(is_near(from_euclidean(x1e + x2e), Var3 {pi / 12, 5, -pi / 6}));
  EXPECT_TRUE(is_near(from_euclidean(x2e + x1e), Var3 {pi / 12, 5, -pi / 6}));
  EXPECT_TRUE(is_near(from_euclidean(x1e - x2e), Var3 {pi * 7 / 12, 5, -pi * 2 / 3}));
  EXPECT_TRUE(is_near(from_euclidean(x2e - x1e), Var3 {-pi * 5 / 12, -5, pi / 3}));
  EXPECT_TRUE(is_near(from_euclidean(-x1e), Var3 {-pi * 5 / 6, -5, pi * 2 / 3}));
  Var3 x50 {pi / 6, 50, -pi / 3};
  EXPECT_TRUE(is_near(from_euclidean(x1e * 10.), x50));
  EXPECT_TRUE(is_near(from_euclidean(10. * x1e), x50));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x50) / 10), x1));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x50) / 10. + x2e / 10.), Var3 {pi / 12, 5, -pi / 6}));
  EXPECT_TRUE(is_near(from_euclidean((to_euclidean(x1) * 2.0 + make_zero(to_euclidean(x1))) / 2.0), x1));
  auto mean_x = Mean(x1);
  mean_x = from_euclidean((to_euclidean(mean_x) * 2.0 + make_zero(to_euclidean(mean_x))) / 2.0);
  EXPECT_TRUE(is_near(mean_x, x1));
  auto x6 = x1e;
  EXPECT_TRUE(is_near(to_euclidean(Var3{x6}), x6));
}


TEST(matrices, Mean_angle2pi_arithmetic_Euclidean)
{
  using Var3 = Mean<std::tuple<angle::PositiveRadians, Axis, angle::PositiveRadians>>;
  const Var3 x1 {pi / 6, 5, pi * 5 / 3};
  const Var3 x2 {2 * pi, 0, 6 * pi};
  const auto x1e = to_euclidean(x1);
  const auto x2e = to_euclidean(x2);
  EXPECT_TRUE(is_near(from_euclidean(x1e + x2e), Var3 {pi / 12, 5, pi * 11 / 6}));
  EXPECT_TRUE(is_near(from_euclidean(x2e + x1e), Var3 {pi / 12, 5, pi * 11 / 6}));
  EXPECT_TRUE(is_near(from_euclidean(x1e - x2e), Var3 {pi * 7 / 12, 5, pi * 4 / 3}));
  EXPECT_TRUE(is_near(from_euclidean(x2e - x1e), Var3 {pi * 19 / 12, -5, pi / 3}));
  EXPECT_TRUE(is_near(from_euclidean(-x1e), Var3 {pi * 7 / 6, -5, pi * 2 / 3}));
  const Var3 x50 {pi / 6, 50, pi * 5 / 3};
  EXPECT_TRUE(is_near(from_euclidean(x1e * 10.), x50));
  EXPECT_TRUE(is_near(from_euclidean(10. * x1e), x50));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x50) / 10.), x1));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x50) / 10. + x2e / 10.), Var3 {pi / 12, 5, pi * 11 / 6}));
}


TEST(matrices, Mean_angle2deg_arithmetic_Euclidean)
{
  using Var3 = Mean<std::tuple<angle::PositiveDegrees, Axis, angle::PositiveDegrees>>;
  const Var3 x1 {30, 5, 300};
  const Var3 x2 {360, 0, 1080};
  const auto x1e = to_euclidean(x1);
  const auto x2e = to_euclidean(x2);
  EXPECT_TRUE(is_near(from_euclidean(x1e + x2e), Var3 {15, 5, 330}));
  EXPECT_TRUE(is_near(from_euclidean(x2e + x1e), Var3 {15, 5, 330}));
  EXPECT_TRUE(is_near(from_euclidean(x1e - x2e), Var3 {105, 5, 240}));
  EXPECT_TRUE(is_near(from_euclidean(x2e - x1e), Var3 {285, -5, 60}));
  EXPECT_TRUE(is_near(from_euclidean(-x1e), Var3 {210, -5, 120}));
  const Var3 x50 {30, 50, 300};
  EXPECT_TRUE(is_near(from_euclidean(x1e * 10.), x50));
  EXPECT_TRUE(is_near(from_euclidean(10. * x1e), x50));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x50) / 10.), x1));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x50) / 10. + x2e / 10.), Var3 {15, 5, 330}));
}


TEST(matrices, Mean_angle_columns)
{
  using Var3 = Mean<std::tuple<angle::Radians, Axis>, eigen_matrix_t<double, 2, 2>>;
  using TVar3 = Matrix<std::tuple<angle::Radians, Axis>, Dimensions<2>, eigen_matrix_t<double, 2, 2>>;
  Var3 v1 {1, 2, 3, 4};
  Var3 v2 {6, 4, -6, 8};
  Var3 v3 {7 - pi*2, 6 - pi*2, -3, 12};
  TVar3 v4 {-5 + pi*2, -2, 9, -4};
  TVar3 v5 {0.5, 1, 1.5, 2};
  EXPECT_TRUE(is_near(v1 + v2, v3));
  EXPECT_TRUE(is_near(v1 - v2, v4));
  EXPECT_TRUE(is_near(v1 * 0.5, v5));
  EXPECT_TRUE(is_near(0.5 * v1, v5));
  EXPECT_TRUE(is_near(v5 / 0.5, TVar3 {v1}));
}


TEST(matrices, Mean_angle_columns_Euclidean)
{
  using Var3 = Mean<std::tuple<angle::Radians, Axis>, eigen_matrix_t<double, 2, 2>>;
  Var3 x1 {pi / 6, -pi / 3, 5, 2};
  Var3 x2 {2 * pi, 6 * pi, 0, 0};
  const auto x1e = to_euclidean(x1);
  const auto x2e = to_euclidean(x2);
  EXPECT_TRUE(is_near(from_euclidean(x1e + x2e), Var3 {pi / 12, -pi / 6, 5, 2}));
  EXPECT_TRUE(is_near(from_euclidean(x2e + x1e), Var3 {pi / 12, -pi / 6, 5, 2}));
  EXPECT_TRUE(is_near(from_euclidean(x1e - x2e), Var3 {pi * 7 / 12, -pi * 2 / 3, 5, 2}));
  EXPECT_TRUE(is_near(from_euclidean(x2e - x1e), Var3 {-pi * 5 / 12, pi / 3, -5, -2}));
  EXPECT_TRUE(is_near(from_euclidean(-x1e), Var3 {-pi * 5 / 6, pi * 2 / 3, -5, -2}));
  Var3 x10 {pi / 6, -pi / 3, 50, 20};
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x1) * 10.), x10));
  EXPECT_TRUE(is_near(from_euclidean(10. * to_euclidean(x1)), x10));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x10) / 10.), x1));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(x10) / 10. + x2e / 10.), Var3 {pi / 12, -pi / 6, 5, 2}));
  Var3 mean_x = x1;
  mean_x = from_euclidean((to_euclidean(mean_x) * 2.0 + make_zero(to_euclidean(mean_x))) / 2.0);
  EXPECT_TRUE(is_near(mean_x, x1));
  auto x6 = to_euclidean(x1);
  EXPECT_TRUE(is_near(to_euclidean(Var3{x6}), x6));
  EXPECT_NEAR(x1(0,0), pi / 6, 1e-6);
  EXPECT_NEAR(x1(0,1), -pi/3, 1e-6);
  EXPECT_NEAR(x1(1,0), 5, 1e-6);
  EXPECT_NEAR(x1(1,1), 2, 1e-6);
  EXPECT_NEAR(x1e(0,1), 0.5, 1e-6);
  EXPECT_NEAR(x1e(1,0), 0.5, 1e-6);
  EXPECT_NEAR(x1e(2,0), 5, 1e-6);
  EXPECT_NEAR(x1e(2,1), 2, 1e-6);
}


TEST(matrices, Wrap_angle)
{
  using R = Mean<angle::Radians>;
  EXPECT_TRUE(is_near(R {-pi*.99} - R {pi*0.99}, Matrix<angle::Radians> {pi*0.02}));
  R x0 {pi/4};
  EXPECT_NEAR(get_component(x0, 0, 0), pi/4, 1e-6);
  EXPECT_NEAR(get_component(x0, 0), pi/4, 1e-6);
  set_component(x0, 5*pi/4, 0, 0);
  EXPECT_NEAR(get_component(x0, 0), -3*pi/4, 1e-6);
  set_component(x0, -7*pi/6, 0);
  EXPECT_NEAR(get_component(x0, 0), 5*pi/6, 1e-6);
}


TEST(matrices, Wrap_distance)
{
  using R = Mean<Distance>;
  EXPECT_TRUE(is_near(R {4} - R {5}, Matrix<Axis> {-1}));
  R x0 {-5};
  EXPECT_TRUE(is_near(x0, eigen_matrix_t<double, 1, 1> {5}));
  EXPECT_TRUE(is_near(x0 + R {1.2}, eigen_matrix_t<double, 1, 1> {6.2}));
  EXPECT_TRUE(is_near(from_euclidean(-to_euclidean(x0) + to_euclidean(R {1.2})), eigen_matrix_t<double, 1, 1> {3.8}));
  EXPECT_TRUE(is_near(R {1.1} - 3. * R {1}, -make_dense_object_from(R {1.9})));
  EXPECT_TRUE(is_near(R {1.2} + R {-3}, R {4.2}));
  EXPECT_NEAR(get_component(x0, 0, 0), 5, 1e-6);
  EXPECT_NEAR(get_component(x0, 0), 5., 1e-6);
  set_component(x0, 4, 0, 0);
  EXPECT_NEAR(get_component(x0, 0, 0), 4., 1e-6);
  set_component(x0, -3, 0, 0);
  EXPECT_NEAR(get_component(x0, 0, 0), 3., 1e-6);
}


TEST(matrices, Wrap_inclination)
{
  using R = Mean<inclination::Radians>;
  EXPECT_TRUE(is_near(R {-pi/2} - R {pi/2}, Matrix<angle::Radians> {-pi}));
  EXPECT_TRUE(is_near(R {pi * 7 / 12}, R {pi * 5 / 12}));
  EXPECT_TRUE(is_near(R {-pi * 7 / 12}, R {-pi * 5 / 12}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(R {pi / 6}) + to_euclidean(R {pi})), R {pi / 12}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(R {-pi / 6}) + to_euclidean(R {pi})), R {-pi / 12}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(R {pi * 5 / 6}) + to_euclidean(R {pi / 2})), R {pi / 3}));
  R x0 {pi/2};
  EXPECT_NEAR(get_component(x0, 0, 0), pi/2, 1e-6);
  EXPECT_NEAR(get_component(x0, 0), pi/2, 1e-6);
  set_component(x0, pi/4, 0, 0);
  EXPECT_NEAR(get_component(x0, 0, 0), pi/4, 1e-6);
  set_component(x0, 3*pi/4, 0, 0);
  EXPECT_NEAR(get_component(x0, 0, 0), pi/4, 1e-6);
}


TEST(matrices, Wrap_polar)
{
  using P = Mean<Polar<Distance, angle::Radians>>;
  EXPECT_TRUE(is_near(P {4, -pi*.99} - P {5, pi*0.99}, Matrix<Polar<Distance, angle::Radians>, Axis> {-1, pi*0.02}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(P {1., pi / 6}) + to_euclidean(P {-0.5, 0})), P {1.5, pi * 7 / 12}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(P {1., pi / 6}) + to_euclidean(P {-1.5, 0})), P {2.5, pi * 7 / 12}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(P {1., pi * 5 / 6}) + to_euclidean(P {0, -pi * 2 / 3})), P {1., -pi * 11 / 12}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(P {1., pi * 5 / 6}) + to_euclidean(P {-1.5, -pi * 2 / 3})), P {2.5, pi * 7 / 12}));
  P x0 {2, pi/4};
  EXPECT_NEAR(get_component(x0, 0, 0), 2, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), pi/4, 1e-6);
  set_component(x0, -1.5, 0);
  EXPECT_NEAR(get_component(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), -3*pi/4, 1e-6);
  set_component(x0, 7*pi/6, 1);
  EXPECT_NEAR(get_component(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), -5*pi/6, 1e-6);

  using Q = Mean<Polar<angle::Radians, Distance>>;
  EXPECT_TRUE(is_near(Q {-pi*.99, 4} - Q {pi*0.99, 5}, Matrix<Polar<angle::Radians, Distance>, Axis> {pi*0.02, -1}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(Q {pi / 6, 1}) + to_euclidean(Q {0, -0.5})), Q {pi * 7 / 12, 1.5}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(Q {pi / 6, 1}) + to_euclidean(Q {0, -1.5})), Q {pi * 7 / 12, 2.5}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(Q {pi * 5 / 6, 1}) + to_euclidean(Q {-pi * 2 / 3, 0})), Q {-pi * 11 / 12, 1}));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(Q {pi * 5 / 6, 1}) + to_euclidean(Q {-pi * 2 / 3, -1.5})), Q {pi * 7 / 12, 2.5}));
  Q x1 {pi/4, 2};
  EXPECT_NEAR(get_component(x1, 1, 0), 2, 1e-6);
  EXPECT_NEAR(get_component(x1, 0), pi/4, 1e-6);
  set_component(x1, -1.5, 1);
  EXPECT_NEAR(get_component(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x1, 0), -3*pi/4, 1e-6);
  set_component(x1, 7*pi/6, 0);
  EXPECT_NEAR(get_component(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x1, 0), -5*pi/6, 1e-6);
}


TEST(matrices, Wrap_spherical)
{
  using S = Mean<Spherical<Distance, angle::Radians, inclination::Radians>>;
  EXPECT_TRUE(is_near(S {4, -pi*.99, -pi/2} - S {5, pi*0.99, pi/2}, Matrix<Spherical<Distance, angle::Radians, inclination::Radians>, Axis> {-1, pi*0.02, -pi}));
  EXPECT_TRUE(is_near(S {-0.5, pi / 3, pi / 6}, S {0.5, -pi * 2 / 3, -pi / 6}));
  EXPECT_TRUE(is_near(S {-0.5, pi * 7, pi * 7 / 6}, S {0.5, pi, pi / 6}));
  const auto x1 = S {0.5, std::atan2(3. / 4, 1. + std::sqrt(3.) / 4),
                     std::asin(0.5 / std::hypot(1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  const auto x2 = S {0.5, std::atan2(3. / 4, 1. + std::sqrt(3.) / 4) - pi,
                     -std::asin(0.5 / std::hypot(1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  const auto x3 = S {1.0, std::atan2(3. / 4, -1. + std::sqrt(3.) / 4),
                     std::asin(0.5 / std::hypot(-1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(S {1., pi / 3, pi / 6}) - to_euclidean(S {-0.5, 0, 0})), x1));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(S {1., pi / 3, pi / 6}) - to_euclidean(S {-1.5, 0, 0})), x2));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(S {1., pi / 3, pi / 6}) + to_euclidean(S {0, 0, pi})), x3));
  S x0 {2, pi/4, -pi/4};
  EXPECT_NEAR(get_component(x0, 0, 0), 2, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), pi/4, 1e-6);
  EXPECT_NEAR(get_component(x0, 2), -pi/4, 1e-6);
  set_component(x0, -1.5, 0);
  EXPECT_NEAR(get_component(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), -3*pi/4, 1e-6);
  EXPECT_NEAR(get_component(x0, 2), pi/4, 1e-6);
  set_component(x0, 7*pi/6, 1);
  EXPECT_NEAR(get_component(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), -5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(x0, 2), pi/4, 1e-6);
  set_component(x0, 3*pi/4, 2);
  EXPECT_NEAR(get_component(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), pi/6, 1e-6);
  EXPECT_NEAR(get_component(x0, 2), pi/4, 1e-6);

  using T = Mean<Spherical<angle::Radians, Distance, inclination::Radians>>;
  EXPECT_TRUE(is_near(T {-pi*.99, 4, -pi/2} - T {pi*0.99, 5, pi/2}, Matrix<Spherical<angle::Radians, Distance, inclination::Radians>, Axis> {pi*0.02, -1, -pi}));
  EXPECT_TRUE(is_near(T {pi / 3, -0.5, pi / 6}, T {-pi * 2 / 3, 0.5, -pi / 6}));
  EXPECT_TRUE(is_near(T {pi * 7, -0.5, pi * 7 / 6}, T {pi, 0.5, pi / 6}));
  const auto y1 = T {std::atan2(3. / 4, 1. + std::sqrt(3.) / 4),
                     0.5, std::asin(0.5 / std::hypot(1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  const auto y2 = T {std::atan2(3. / 4, 1. + std::sqrt(3.) / 4) - pi,
                     0.5, -std::asin(0.5 / std::hypot(1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  const auto y3 = T {std::atan2(3. / 4, -1. + std::sqrt(3.) / 4),
                     1.0, std::asin(0.5 / std::hypot(-1. + std::sqrt(3.) / 4, 3. / 4, 0.5))};
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(T {pi / 3, 1, pi / 6}) - to_euclidean(T {0, -0.5, 0})), y1));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(T {pi / 3, 1, pi / 6}) - to_euclidean(T {0, -1.5, 0})), y2));
  EXPECT_TRUE(is_near(from_euclidean(to_euclidean(T {pi / 3, 1, pi / 6}) + to_euclidean(T {0, 0, pi})), y3));
  T z1 {pi/4, 2, -pi/4};
  EXPECT_NEAR(get_component(z1, 1, 0), 2, 1e-6);
  EXPECT_NEAR(get_component(z1, 0), pi/4, 1e-6);
  EXPECT_NEAR(get_component(z1, 2), -pi/4, 1e-6);
  set_component(z1, -1.5, 1);
  EXPECT_NEAR(get_component(z1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(z1, 0), -3*pi/4, 1e-6);
  EXPECT_NEAR(get_component(z1, 2), pi/4, 1e-6);
  set_component(z1, 7*pi/6, 0);
  EXPECT_NEAR(get_component(z1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(z1, 0), -5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(z1, 2), pi/4, 1e-6);
  set_component(z1, 3*pi/4, 2);
  EXPECT_NEAR(get_component(z1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(z1, 0), pi/6, 1e-6);
  EXPECT_NEAR(get_component(z1, 2), pi/4, 1e-6);

  using U = Mean<Spherical<angle::Radians, inclination::Radians, Distance>>;
  EXPECT_TRUE(is_near(U {-pi*.99, -pi/2, 4} - U {pi*0.99, pi/2, 5}, Matrix<Spherical<angle::Radians, inclination::Radians, Distance>, Axis> {pi*0.02, -pi, -1}));
  U z2 {pi/4, -pi/4, 2};
  EXPECT_NEAR(get_component(z2, 2, 0), 2, 1e-6);
  EXPECT_NEAR(get_component(z2, 0), pi/4, 1e-6);
  EXPECT_NEAR(get_component(z2, 1), -pi/4, 1e-6);
  set_component(z2, -1.5, 2);
  EXPECT_NEAR(get_component(z2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_component(z2, 0), -3*pi/4, 1e-6);
  EXPECT_NEAR(get_component(z2, 1), pi/4, 1e-6);
  set_component(z2, 7*pi/6, 0);
  EXPECT_NEAR(get_component(z2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_component(z2, 0), -5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(z2, 1), pi/4, 1e-6);
  set_component(z2, 3*pi/4, 1);
  EXPECT_NEAR(get_component(z2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_component(z2, 0), pi/6, 1e-6);
  EXPECT_NEAR(get_component(z2, 1), pi/4, 1e-6);
}


TEST(matrices, Wrap_angle_polar)
{
  using R = Mean<std::tuple<angle::Radians, Polar<Distance, angle::Radians>>>;
  EXPECT_TRUE(is_near(R {pi/4, 1., pi/6} + R {-pi/2, 0.5, pi}, R {-pi/4, 1.5, -pi*5/6}));
  EXPECT_TRUE(is_near(R {pi/4, 1., pi/6} + R {pi/4, -1.5, -pi/2}, R {pi/2, 2.5, pi*2/3}));
  EXPECT_TRUE(is_near(R {-pi/2, 1., pi*5/6} + R {pi/3, -0.5, pi/2}, R {-pi/6, 1.5, pi/3}));
  R x0 {pi/4, 2, pi/4};
  EXPECT_NEAR(get_component(x0, 0, 0), pi/4, 1e-6);
  EXPECT_NEAR(get_component(x0, 0), pi/4, 1e-6);
  set_component(x0, 5*pi/4, 0, 0);
  EXPECT_NEAR(get_component(x0, 0), -3*pi/4, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), 2, 1e-6);
  EXPECT_NEAR(get_component(x0, 2), pi/4, 1e-6);
  set_component(x0, -7*pi/6, 0);
  EXPECT_NEAR(get_component(x0, 0), 5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), 2, 1e-6);
  EXPECT_NEAR(get_component(x0, 2), pi/4, 1e-6);
  set_component(x0, -1.5, 1);
  EXPECT_NEAR(get_component(x0, 0), 5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x0, 2), -3*pi/4, 1e-6);
  set_component(x0, 7*pi/6, 2);
  EXPECT_NEAR(get_component(x0, 0), 5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(x0, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x0, 2), -5*pi/6, 1e-6);

  using Q = Mean<std::tuple<Polar<angle::Radians, Distance>, angle::Radians>>;
  EXPECT_TRUE(is_near(Q {pi/6, 1, pi/4} + Q {pi, 0.5, -pi/2}, Q {-pi*5/6, 1.5, -pi/4}));
  EXPECT_TRUE(is_near(Q {pi/6, 1, pi/4} + Q {-pi/2, -1.5, pi/4}, Q {pi*2/3, 2.5, pi/2}));
  EXPECT_TRUE(is_near(Q {pi*5/6, 1., -pi/2} + Q {pi/2, -0.5, pi/3}, Q {pi/3, 1.5, -pi/6}));
  Q x1 {pi/4, 2, pi/4};
  EXPECT_NEAR(get_component(x1, 2, 0), pi/4, 1e-6);
  EXPECT_NEAR(get_component(x1, 2), pi/4, 1e-6);
  set_component(x1, 5*pi/4, 2, 0);
  EXPECT_NEAR(get_component(x1, 2), -3*pi/4, 1e-6);
  EXPECT_NEAR(get_component(x1, 1), 2, 1e-6);
  EXPECT_NEAR(get_component(x1, 0), pi/4, 1e-6);
  set_component(x1, -7*pi/6, 2);
  EXPECT_NEAR(get_component(x1, 2), 5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(x1, 1), 2, 1e-6);
  EXPECT_NEAR(get_component(x1, 0), pi/4, 1e-6);
  set_component(x1, -1.5, 1);
  EXPECT_NEAR(get_component(x1, 2), 5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x1, 0), -3*pi/4, 1e-6);
  set_component(x1, 7*pi/6, 0);
  EXPECT_NEAR(get_component(x1, 2), 5*pi/6, 1e-6);
  EXPECT_NEAR(get_component(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_component(x1, 0), -5*pi/6, 1e-6);
}

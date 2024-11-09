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
using namespace OpenKalman::test;

using numbers::pi;
using numbers::sqrt2;

using M12 = eigen_matrix_t<double, 1, 2>;
using M13 = eigen_matrix_t<double, 1, 3>;
using M21 = eigen_matrix_t<double, 2, 1>;
using M22 = eigen_matrix_t<double, 2, 2>;
using M23 = eigen_matrix_t<double, 2, 3>;
using M31 = eigen_matrix_t<double, 3, 1>;
using M32 = eigen_matrix_t<double, 3, 2>;
using M33 = eigen_matrix_t<double, 3, 3>;
using M42 = eigen_matrix_t<double, 4, 2>;
using M43 = eigen_matrix_t<double, 4, 3>;
using I22 = Eigen3::IdentityMatrix<M22>;
using Z22 = ZeroAdapter<eigen_matrix_t<double, 2, 2>>;
using C2 = StaticDescriptor<Axis, angle::Radians>;
using C3 = StaticDescriptor<Axis, angle::Radians, Axis>;
using Mat12 = EuclideanMean<Axis, M12>;
using Mat13 = EuclideanMean<Axis, M13>;
using Mat21 = EuclideanMean<C2, M31>;
using Mat22 = EuclideanMean<C2, M32>;
using Mat23 = EuclideanMean<C2, M33>;
using Mat32 = EuclideanMean<C3, M42>;
using Mat33 = EuclideanMean<C3, M43>;
using TM22 = Matrix<Dimensions<2>, Dimensions<2>, M22>;
using TM23 = Matrix<Dimensions<2>, Dimensions<3>, M23>;
using TM32 = Matrix<Dimensions<3>, Dimensions<2>, M32>;
using TM33 = Matrix<Dimensions<3>, Dimensions<3>, M33>;
using TM42 = Matrix<Dimensions<4>, Dimensions<2>, M42>;

inline I22 i22 = M22::Identity();
inline Z22 z22 = Z22();
inline auto covi22 = Covariance<C2, I22>(i22);
inline auto covz22 = Covariance<C2, Z22>(z22);
inline auto sqcovi22 = SquareRootCovariance<C2, I22>(i22);
inline auto sqcovz22 = SquareRootCovariance<C2, Z22>(z22);


TEST(matrices, EuclideanMean_class)
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
  EuclideanMean<Dimensions<2>, M23> mat23_x1(Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x1, TM23 {1, 2, 3, 4, 5, 6}));
  EuclideanMean<Dimensions<2>, M23> mat23_x2(Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6});
  EXPECT_TRUE(is_near(mat23_x2, TM23 {1, 2, 3, 4, 5, 6}));
  Mat23 mat23_x3(Mean<C2, M23> {
    1, 2, 3,
    pi/3, pi/6, pi/4});
  EXPECT_TRUE(is_near(mat23_x3, TM33 {
    1, 2, 3,
    0.5, std::sqrt(3)/2, sqrt2/2,
    std::sqrt(3)/2, 0.5, sqrt2/2}));

  // Construct from a typed_matrix_nestable
  Mat23 mat23d(make_dense_object_from<M33>(1, 2, 3, 4, 5, 6, 7, 8, 9));
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
  mat23_x1 = Matrix<Dimensions<2>, Dimensions<3>, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x1, TM23 {6, 5, 4, 3, 2, 1}));
  mat23_x2 = Mean<Dimensions<2>, M23> {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23_x2, TM23 {6, 5, 4, 3, 2, 1}));
  mat23_x3 = Mean<C2, M23> {
    3, 2, 1,
    pi/6, pi/4, pi/3};
  EXPECT_TRUE(is_near(mat23_x3, TM33 {
    3, 2, 1,
    std::sqrt(3)/2, sqrt2/2, 0.5,
    0.5, sqrt2/2, std::sqrt(3)/2}));

  // assign from a typed_matrix_nestable
  mat23e = make_dense_object_from<M33>(3, 4, 5, 6, 7, 8, 9, 10, 11);

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
  EXPECT_TRUE(is_near(make_zero<Mat23>(), M33::Zero()));

  // Identity
  EXPECT_TRUE(is_near(make_identity_matrix_like<EuclideanMean<Dimensions<2>, M22>>(), M22::Identity()));
}


TEST(matrices, EuclideanMean_subscripts)
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


TEST(matrices, EuclideanMean_deduction_guides)
{
  auto a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  EXPECT_TRUE(is_near(EuclideanMean(a), a));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(EuclideanMean(a)), 0>, Dimensions<2>>);

  auto b1 = Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(is_near(EuclideanMean(b1), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(EuclideanMean(b1)), 0>, C2>);

  auto b2 = Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(EuclideanMean(b2), TM23 {1, 2, 3, 4, 5, 6}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(EuclideanMean(b2)), 0>, Dimensions<2>>);

  auto b3 = Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(EuclideanMean(b3), TM23 {1, 2, 3, 4, 5, 6}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(EuclideanMean(b3)), 0>, Dimensions<2>>);
  static_assert(index_dimension_of_v<decltype(EuclideanMean(b3)), 0> == 2);
}


TEST(matrices, EuclideanMean_make_functions)
{
  auto a = make_dense_object_from<M33>(1, 2, 3, 4, 5, 6, 7, 8, 9);
  EXPECT_TRUE(is_near(make_euclidean_mean<C2>(a), a));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_euclidean_mean<C2>(a)), 0>, C2>);
  EXPECT_TRUE(is_near(make_euclidean_mean<C2>(a), a));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_euclidean_mean<C2>(a)), 0>, C2>);

  auto b = Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_TRUE(is_near(make_euclidean_mean(b), a));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_euclidean_mean(b)), 0>, C2>);

  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_euclidean_mean<C2, M33>()), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_euclidean_mean<M23>()), 0>, Dimensions<2>>);
}


TEST(matrices, EuclideanMean_traits)
{
  static_assert(typed_matrix<Mat23>);
  static_assert(not mean<Mat23>);
  static_assert(euclidean_mean<Mat23>);
  static_assert(euclidean_transformed<Mat23>);
  static_assert(not euclidean_transformed<EuclideanMean<Dimensions<2>, I22>>);
  static_assert(not wrapped_mean<Mat23>);
  static_assert(untyped_columns<Mat23>);

  static_assert(not identity_matrix<Mat23>);
  static_assert(identity_matrix<EuclideanMean<Dimensions<2>, I22>>);
  static_assert(not identity_matrix<EuclideanMean<angle::Radians, I22>>);
  static_assert(not identity_matrix<EuclideanMean<Dimensions<2>, M23>>);
  static_assert(not zero<Mat23>);
  static_assert(zero<EuclideanMean<angle::Radians, Z22>>);
  static_assert(zero<EuclideanMean<Dimensions<2>, Z22>>);
  static_assert(zero<EuclideanMean<C2, ZeroAdapter<eigen_matrix_t<double, 3, 3>>>>);

  EXPECT_TRUE(is_near(make_zero<Mat23>(), eigen_matrix_t<double, 3, 3>::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<EuclideanMean<Dimensions<2>, I22>>(), eigen_matrix_t<double, 2, 2>::Identity()));
}


TEST(matrices, EuclideanMean_overloads)
{
  EXPECT_TRUE(is_near(nested_object(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  EXPECT_TRUE(is_near(make_dense_object_from(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));

  EXPECT_TRUE(is_near(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9} * 2), TM33 {2, 4, 6, 8, 10, 12, 14, 16, 18}));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9} * 2))>, Mat23>);

  EXPECT_TRUE(is_near(from_euclidean(
    EuclideanMean<C2, M33> {1, 2, 3,
                            0.5, std::sqrt(3)/2, sqrt2/2,
                            std::sqrt(3)/2, 0.5, sqrt2/2}),
    Mean<C2, M23> {1, 2, 3,
                   pi/3, pi/6, pi/4}));

  const auto ma = make_euclidean_mean(-2., 5, 3);
  EXPECT_TRUE(is_near(from_euclidean(ma), ma));

  using A3 = StaticDescriptor<angle::Radians, Axis, angle::Radians>;
  const auto mb = make_euclidean_mean<A3>(std::sqrt(3) / 2, 0.5, 5, 0.5, -std::sqrt(3) / 2);
  const auto x2 = (eigen_matrix_t<double, 3, 1> {} << pi / 6, 5, -pi / 3).finished();
  EXPECT_TRUE(is_near(from_euclidean(mb).nested_object(), x2));

  EXPECT_TRUE(is_near(to_diagonal(EuclideanMean<Dimensions<2>, M21> {2, 3}).nested_object(), TM22 {2, 0, 0, 3}));
  static_assert(diagonal_matrix<decltype(to_diagonal(EuclideanMean<Dimensions<2>, M21> {2, 3}))>);
  static_assert(typed_matrix<decltype(to_diagonal(EuclideanMean<Dimensions<2>, M21> {2, 3}))>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(to_diagonal(EuclideanMean<Dimensions<2>, M21> {2, 3})), 1>, Dimensions<2>>);

  EXPECT_TRUE(is_near(transpose(EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}).nested_object(), TM32 {1, 4, 2, 5, 3, 6}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(transpose(EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})), 0>, Dimensions<3>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(transpose(EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})), 1>, Dimensions<2>>);

  EXPECT_TRUE(is_near(adjoint(EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}).nested_object(), TM32 {1, 4, 2, 5, 3, 6}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(adjoint(EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})), 0>, Dimensions<3>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(adjoint(EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})), 1>, Dimensions<2>>);

  EXPECT_NEAR(determinant(EuclideanMean<Dimensions<3>, M33> {1, 2, 3, 4, 5, 6, 1, 5, 1}), 24, 1e-6);

  EXPECT_NEAR(trace(EuclideanMean<Dimensions<3>, M33> {1, 2, 3, 4, 5, 6, 7, 8, 9}), 15, 1e-6);

  EXPECT_TRUE(is_near(solve(EuclideanMean<Dimensions<2>, M22> {9., 3, 3, 10}, EuclideanMean<Dimensions<2>, M21> {15, 23}), EuclideanMean<Dimensions<2>, M21> {1, 2}));

  EXPECT_TRUE(is_near(average_reduce<1>(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), Mat21 {2, 5, 8}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})), TM22 {14., 32, 32, 77}));

  EXPECT_TRUE(is_near(square(QR_decomposition(EuclideanMean<Dimensions<3>, M32> {1, 4, 2, 5, 3, 6})), TM22 {14., 32, 32, 77}));

  using N = std::normal_distribution<double>;
  Mat23 m = make_zero<Mat23>();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat23>(N {1.0, 0.3}, N {2.0, 0.3}, N {3.0, 0.3})) / (i + 1);
  }
  Mat23 offset = {1, 1, 1, 2, 2, 2, 3, 3, 3};
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));
}


TEST(matrices, EuclideanMean_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(Mat22 {1, 2, 3, 4, 5, 6}, Mat12 {7, 8}), TM42 {1, 2, 3, 4, 5, 6, 7, 8}));
  static_assert(euclidean_mean<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4, 5, 6}, Mat12 {7, 8}))>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4, 5, 6}, Mat12 {7, 8})), 0>, C3>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5, 7, 8}, Mat21 {3, 6, 9}), TM33 {1, 2, 3, 4, 5, 6, 7, 8, 9}));
  static_assert(euclidean_mean<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5, 7, 8}, Mat21 {3, 6, 9}))>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5, 7, 8}, Mat21 {3, 6, 9})), 0>, C2>);

  EXPECT_TRUE(is_near(split_vertical(Mat32 {1, 2, 3, 4, 5, 6, 7, 8}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<C2, Axis>(Mat32 {1, 2, 3, 4, 5, 6, 7, 8}), std::tuple {Mat22 {1, 2, 3, 4, 5, 6}, Mat12 {7, 8}}));
  EXPECT_TRUE(is_near(split_horizontal<Dimensions<2>, Axis>(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), std::tuple {Mat22 {1, 2, 4, 5, 7, 8}, Mat21 {3, 6, 9}}));
  EXPECT_TRUE(is_near(split_vertical<Axis, angle::Radians>(Mat32 {1, 2, 3, 4, 5, 6, 7, 8}), std::tuple {Mat12 {1, 2},  EuclideanMean<angle::Radians, M22> {3, 4, 5, 6}}));
  EXPECT_TRUE(is_near(split_horizontal<Axis, Axis>(Mat23 {1, 2, 3, 4, 5, 6, 7, 8, 9}), std::tuple {Mat21 {1, 4, 7}, Mat21 {2, 5, 8}}));

  EXPECT_TRUE(is_near(column(Mat22 {1, 2, 3, 4, 5, 6}, 0), Mean{1., 3, 5}));
  EXPECT_TRUE(is_near(column(Mat22 {1, 2, 3, 4, 5, 6}, 1), Mean{2., 4, 6}));

  EXPECT_TRUE(is_near(column<0>(Mat22 {1, 2, 3, 4, 5, 6}), Mean{1., 3, 5}));
  EXPECT_TRUE(is_near(column<1>(Mat22 {1, 2, 3, 4, 5, 6}), Mean{2., 4, 6}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(column<0>(Mat22 {1, 2, 3, 4, 5, 6})), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(column<1>(Mat22 {1, 2, 3, 4, 5, 6})), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(column<0>(Mat22 {1, 2, 3, 4, 5, 6})), 1>, Axis>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(column<1>(Mat22 {1, 2, 3, 4, 5, 6})), 1>, Axis>);

  auto m = Mat22 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(apply_columnwise([](auto& col){ col *= 2; }, m), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto& col, std::size_t i){ col *= i; }, m), Mat22 {0, 4, 0, 8, 0, 12}));

  EXPECT_TRUE(is_near(apply_columnwise([](auto col){ return make_self_contained(col * 2); }, Mat22 {1, 2, 3, 4, 5, 6}), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto col){ return make_self_contained(col * 2); }, Mat22 {1, 2, 3, 4, 5, 6}), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto&& col){ return make_self_contained(col * 2); }, Mat22 {1, 2, 3, 4, 5, 6}), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col * 2); }, Mat22 {1, 2, 3, 4, 5, 6}), Mat22 {2, 4, 6, 8, 10, 12}));

  EXPECT_TRUE(is_near(apply_columnwise([](auto col, std::size_t i){ return make_self_contained(col * i); }, Mat22 {1, 2, 3, 4, 5, 6}), Mat22 {0, 2, 0, 4, 0, 6}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto col, std::size_t i){ return make_self_contained(col * i); }, Mat22 {1, 2, 3, 4, 5, 6}), Mat22 {0, 2, 0, 4, 0, 6}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto&& col, std::size_t i){ return make_self_contained(col * i); }, Mat22 {1, 2, 3, 4, 5, 6}), Mat22 {0, 2, 0, 4, 0, 6}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col * i); }, Mat22 {1, 2, 3, 4, 5, 6}), Mat22 {0, 2, 0, 4, 0, 6}));

  EXPECT_TRUE(is_near(apply_columnwise<2>([] { return Mat21 {1, 2, 3}; }), Mat22 {1, 1, 2, 2, 3, 3}));
  EXPECT_TRUE(is_near(apply_columnwise<2>([](std::size_t i){ return Mat21 {i + 1., 2*i + 1, 2*i + 2}; }), Mat22 {1, 2, 1, 3, 2, 4}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<Mat21()>())), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<Mat21()>())), 1>, Dimensions<2>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<Mat21(std::size_t)>())), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<Mat21(std::size_t)>())), 1>, Dimensions<2>>);

  const auto mat22_123456 = Mat22 {1, 2, 3, 4, 5, 6};
  auto n = mat22_123456

  EXPECT_TRUE(is_near(apply_coefficientwise([](auto& x){ x *= 2; }, n), Mat22 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](auto& x, std::size_t i, std::size_t j){ x += i + j; }, n), Mat22 {2, 5, 7, 10, 12, 15}));

  EXPECT_TRUE(is_near(apply_coefficientwise([](auto x){ return x + 1; }, mat22_123456), Mat22 {2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto x){ return x + 1; }, mat22_123456), Mat22 {2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](auto&& x){ return x + 1; }, mat22_123456), Mat22 {2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, mat22_123456), Mat22 {2, 3, 4, 5, 6, 7}));

  EXPECT_TRUE(is_near(apply_coefficientwise([](auto x, std::size_t i, std::size_t j){ return x + i + j; }, mat22_123456), Mat22 {1, 3, 4, 6, 7, 9}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto x, std::size_t i, std::size_t j){ return x + i + j; }, mat22_123456), Mat22 {1, 3, 4, 6, 7, 9}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](auto&& x, std::size_t i, std::size_t j){ return x + i + j; }, mat22_123456), Mat22 {1, 3, 4, 6, 7, 9}));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, mat22_123456), Mat22 {1, 3, 4, 6, 7, 9}));

  EXPECT_TRUE(is_near(apply_coefficientwise<Mat22>([] { return 2; }), Mat22 {2, 2, 2, 2, 2, 2}));
  EXPECT_TRUE(is_near(apply_coefficientwise<Mat22>([](std::size_t i, std::size_t j){ return 1 + i + j; }), Mat22 {1, 2, 2, 3, 3, 4}));
}


TEST(matrices, EuclideanMean_arithmetic)
{
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2, 1, 0} + Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, TM42 {8, 8, 8, 8, 8, 8, 8, 8}));
  EXPECT_TRUE(is_near(EuclideanMean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} + Mean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6}, TM32 {8, 8, 8, 8, 8, 8}));
  EXPECT_TRUE(is_near(EuclideanMean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} + Matrix<Dimensions<3>, Dimensions<2>, M32> {1, 2, 3, 4, 5, 6}, TM32 {8, 8, 8, 8, 8, 8}));
  static_assert(euclidean_mean<decltype(Mat32 {7, 6, 5, 4, 3, 2, 1, 0} + Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, Mat32 {8, 8, 8, 8, 8, 8, 8, 8})>);
  static_assert(typed_matrix<decltype(EuclideanMean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} + Mean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(EuclideanMean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} + Matrix<Dimensions<3>, Dimensions<2>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2, 1, 0} - Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, TM42 {6, 4, 2, 0, -2, -4, -6, -8}));
  EXPECT_TRUE(is_near(EuclideanMean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} - Mean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6}, TM32 {6, 4, 2, 0, -2, -4}));
  EXPECT_TRUE(is_near(EuclideanMean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6}, TM32 {6, 4, 2, 0, -2, -4}));
  static_assert(euclidean_mean<decltype(Mat32 {7, 6, 5, 4, 3, 2, 1, 0} - Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, Mat32 {8, 8, 8, 8, 8, 8, 8, 8})>);
  static_assert(typed_matrix<decltype(EuclideanMean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} - Mean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(EuclideanMean<Dimensions<3>, M32> {7, 6, 5, 4, 3, 2} - Matrix<Dimensions<3>, Dimensions<2>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {1, 2, 3, 4, 5, 6, 7, 8} * 2, TM42 {2, 4, 6, 8, 10, 12, 14, 16}));
  EXPECT_TRUE(is_near(2 * Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, TM42 {2, 4, 6, 8, 10, 12, 14, 16}));
  EXPECT_TRUE(is_near(Mat32 {2, 4, 6, 8, 10, 12, 14, 16} / 2, TM42 {1, 2, 3, 4, 5, 6, 7, 8}));
  static_assert(euclidean_mean<decltype(Mat32 {1, 2, 3, 4, 5, 6, 7, 8} * 2, Mat32 {2, 4, 6, 8, 10, 12, 14, 16})>);
  static_assert(euclidean_mean<decltype(2 * Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, Mat32 {2, 4, 6, 8, 10, 12, 14, 16})>);
  static_assert(euclidean_mean<decltype(Mat32 {2, 4, 6, 8, 10, 12, 14, 16} / 2, Mat32 {1, 2, 3, 4, 5, 6, 7, 8})>);

  using Mat12a = EuclideanMean<angle::Radians, M22>;
  EXPECT_TRUE(is_near(Mat12a {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6,}, TM23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat12a {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6,}), 0>, angle::Radians>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat12a {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6,}), 1>, Dimensions<3>>);
  static_assert(euclidean_mean<decltype(Mat12a {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat12a {1, 2, 3, 4} * Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6}, TM23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat12a {1, 2, 3, 4} * Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6}), 0>, angle::Radians>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat12a {1, 2, 3, 4} * Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6}), 1>, Dimensions<3>>);
  static_assert(euclidean_mean<decltype(Mat12a {1, 2, 3, 4} * Matrix<Dimensions<2>, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat12a {1, 2, 3, 4} * Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}, TM23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat12a {1, 2, 3, 4} * Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}), 0>, angle::Radians>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat12a {1, 2, 3, 4} * Mean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}), 1>, Dimensions<3>>);

  EXPECT_TRUE(is_near(-Mat32 {1, 2, 3, 4, 5, 6, 7, 8}, Mat32 {-1, -2, -3, -4, -5, -6, -7, -8}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4, 5, 6} == Mat22 {1, 2, 3, 4, 5, 6}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4, 5, 6} != Mat22 {1, 2, 2, 4, 5, 6}));
  EXPECT_FALSE((Mat22 {1, 2, 3, 4, 5, 6} == EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6}));
}


TEST(matrices, Polar_Spherical_toEuclideanExpr)
{
  using Pa = Polar<Distance, angle::Radians>;
  const Mean<Pa> ma {2, pi / 6};
  const auto xa = make_euclidean_mean<Pa>(2, std::sqrt(3) / 2, 0.5);
  EXPECT_TRUE(is_near(to_euclidean(ma), xa));

  using Pb = Polar<angle::Radians, Distance>;
  const Mean<Pb> mb {pi / 6, 2};
  const auto xb = make_euclidean_mean<P2>(std::sqrt(3) / 2, 0.5, 2);
  EXPECT_TRUE(is_near(to_euclidean(mb), xb));

  using Sc = Spherical<Distance, angle::Radians, inclination::Radians>;
  const Mean<Sc> mc {2, pi / 6, -pi / 3};
  const auto xc = make_euclidean_mean<Sc>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(mc), xc));

  using Sd = Spherical<Distance, inclination::Radians, angle::Radians>;
  const Mean<Sd> md {2, -pi / 3, pi / 6};
  const auto xd = make_euclidean_mean<Sd>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(md), xd));

  using Se = Spherical<angle::Radians, Distance, inclination::Radians>;
  const Mean<Se> me {pi / 6, 2, -pi / 3};
  const auto xe = make_euclidean_mean<Se>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(me), xe));

  using Sf = Spherical<inclination::Radians, Distance, angle::Radians>;
  const Mean<Sf> mf {-pi / 3, 2, pi / 6};
  const auto xf = make_euclidean_mean<Sf>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(mf), xf));

  using Sg = Spherical<angle::Radians, inclination::Radians, Distance>;
  const Mean<Sg> mg {pi / 6, -pi / 3, 2};
  const auto xg = make_euclidean_mean<Sg>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(mg), xg));

  using Sh = Spherical<inclination::Radians, angle::Radians, Distance>;
  const Mean<Sh> mh {-pi / 3, pi / 6, 2};
  const auto xh = make_euclidean_mean<Sh>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(to_euclidean(mh), xh));
}


TEST(matrices, Polar_Spherical_fromEuclideanExpr)
{
  using Pa = Polar<Distance, angle::Radians>;
  const Mean<Pa> ma {2, pi / 6};
  const auto xa = make_euclidean_mean<Pa>(2, std::sqrt(3) / 2, 0.5);
  EXPECT_TRUE(is_near(ma, from_euclidean(xa)));

  using Pb = Polar<angle::Radians, Distance>;
  const Mean<Pb> mb {pi / 6, 2};
  const auto xb = make_euclidean_mean<Pb>(std::sqrt(3) / 2, 0.5, 2);
  EXPECT_TRUE(is_near(mb, from_euclidean(xb)));

  using Sc = Spherical<Distance, angle::Radians, inclination::Radians>;
  const Mean<Sc> mc {2, pi / 6, -pi / 3};
  const auto xc = make_euclidean_mean<Sc>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(mc, from_euclidean(xc)));

  using Sd = Spherical<Distance, inclination::Radians, angle::Radians>;
  const Mean<Sd> md {2, -pi / 3, pi / 6};
  const auto xd = make_euclidean_mean<Sd>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(md, from_euclidean(xd)));

  using Se = Spherical<angle::Radians, Distance, inclination::Radians>;
  const Mean<Se> me {pi / 6, 2, -pi / 3};
  const auto xe = make_euclidean_mean<Se>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(me, from_euclidean(xe)));

  using Sf = Spherical<inclination::Radians, Distance, angle::Radians>;
  const Mean<Sf> mf {-pi / 3, 2, pi / 6};
  const auto xf = make_euclidean_mean<Sf>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(mf, from_euclidean(xf)));

  using Sg = Spherical<angle::Radians, inclination::Radians, Distance>;
  const Mean<Sg> mg {pi / 6, -pi / 3, 2};
  const auto xg = make_euclidean_mean<Sg>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(mg, from_euclidean(xg)));

  using Sh = Spherical<inclination::Radians, angle::Radians, Distance>;
  const Mean<Sh> mh {-pi / 3, pi / 6, 2};
  const auto xh = make_euclidean_mean<Sh>(2, std::sqrt(3) / 4, 0.25, -std::sqrt(3) / 2);
  EXPECT_TRUE(is_near(mh, from_euclidean(xh)));
}

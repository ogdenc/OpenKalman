/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "adapters.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

using numbers::pi;
using numbers::sqrt2;

namespace
{
  using C2 = StaticDescriptor<Axis, angle::Radians>;
  using C3 = StaticDescriptor<Axis, angle::Radians, Axis>;
  using Mat12 = VectorSpaceDescriptor<M12, Axis, C2>;
  using Mat21 = VectorSpaceDescriptor<M21, C2, Axis>;
  using Mat22 = VectorSpaceDescriptor<M22, C2, C2>;
  using Mat23 = VectorSpaceDescriptor<M23, C2, C3>;
  using Mat32 = VectorSpaceDescriptor<M32, C3, C2>;
  using Mat33 = VectorSpaceDescriptor<M33, C3, C3>;

  using SA2l = HermitianAdapter<M22, TriangleType::lower>;
  using SA2u = HermitianAdapter<M22, TriangleType::upper>;
  using T2l = TriangularAdapter<M22, TriangleType::lower>;
  using T2u = TriangularAdapter<M22, TriangleType::upper>;

  inline I22 i22 = M22::Identity();
  inline Z22 z22 = Z22();
}


TEST(adapters, VectorSpaceDescriptor_class)
{
  // Default constructor and Eigen3 construction
  Mat23 mat23a {make_dense_object_from<M23>(1, 2, 3, 4, 5, 6)};
  EXPECT_TRUE(is_near(mat23a, Mat23 {1, 2, 3, 4, 5, 6}));

  // Copy constructor
  Mat23 mat23b = const_cast<const Mat23&>(mat23a);
  EXPECT_TRUE(is_near(mat23b, Mat23 {make_dense_object_from<M23>(1, 2, 3, 4, 5, 6)}));

  // Move constructor
  auto xa = Mat23 {make_dense_object_from<M23>(6, 5, 4, 3, 2, 1)};
  Mat23 mat23c(std::move(xa));
  EXPECT_TRUE(is_near(mat23c, Mat23 {make_dense_object_from<M23>(6, 5, 4, 3, 2, 1)}));

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

  // assign from a regular matrix
  mat23e = make_dense_object_from<M23>(3, 4, 5, 6, 7, 8);

  // Assign from a list of coefficients (via move assignment operator)
  mat23e = {6, 5, 4, 3, 2, 1};
  EXPECT_TRUE(is_near(mat23e, Mat23 {6, 5, 4, 3, 2, 1}));

  // Increment
  mat23a += Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(mat23a, Mat23 {2, 4, 6, 8, 10, 12}));

  // Increment with a stochastic value
  mat23_x1 = {1, 1, 1, pi, pi, pi};
  GaussianDistribution g = {Mat21 {0, 0}, sqcovi22 * 0.1};
  VectorSpaceDescriptor<C2, Dimensions<3>, M23> diff = {1, 1, 1, 1, 1, 1};
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
  EXPECT_TRUE(is_near(make_zero<Mat23>(), M23::Zero()));

  // Identity
  EXPECT_TRUE(is_near(make_identity_matrix_like<Mat22>(), M22::Identity()));
}


TEST(adapters, VectorSpaceDescriptor_subscripts)
{
  static_assert(element_gettable<Mat23, 2>);
  static_assert(not element_gettable<Mat23, 1>);
  static_assert(element_gettable<const Mat23, 2>);
  static_assert(not element_gettable<const Mat23, 1>);
  static_assert(element_gettable<Mat21, 2>);
  static_assert(element_gettable<Mat21, 1>);
  static_assert(element_gettable<const Mat21, 2>);
  static_assert(element_gettable<const Mat21, 1>);
  static_assert(element_gettable<VectorSpaceDescriptor<C3, C2, M32>, 2>);
  static_assert(not element_gettable<VectorSpaceDescriptor<C3, C2, M32>, 1>);
  static_assert(element_gettable<VectorSpaceDescriptor<C2, Axis, M21>, 2>);
  static_assert(element_gettable<VectorSpaceDescriptor<C2, Axis, M21>, 1>);

  static_assert(writable_by_component<Mat23&, 2>);
  static_assert(not writable_by_component<Mat23&, 1>);
  static_assert(not writable_by_component<const Mat23&, 2>);
  static_assert(not writable_by_component<const Mat23&, 1>);
  static_assert(writable_by_component<Mat21&, 2>);
  static_assert(writable_by_component<Mat21&, 1>);
  static_assert(not writable_by_component<const Mat21&, 2>);
  static_assert(not writable_by_component<const Mat21&, 1>);
  static_assert(writable_by_component<VectorSpaceDescriptor<C3, C2, M32>&, 2>);
  static_assert(not writable_by_component<VectorSpaceDescriptor<C3, C2, M32>&, 1>);
  static_assert(not writable_by_component<VectorSpaceDescriptor<C3, C2, const M32>&, 2>);
  static_assert(not writable_by_component<VectorSpaceDescriptor<C3, C2, const M32>&, 1>);
  static_assert(writable_by_component<VectorSpaceDescriptor<C2, Axis, M21>&, 2>);
  static_assert(writable_by_component<VectorSpaceDescriptor<C2, Axis, M21>&, 1>);
  static_assert(not writable_by_component<VectorSpaceDescriptor<C2, Axis, const M21>&, 2>);
  static_assert(not writable_by_component<VectorSpaceDescriptor<C2, Axis, const M21>&, 1>);

  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 0), 1, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 1), 2, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(0, 2), 3, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 0), 4, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 1), 5, 1e-6);
  EXPECT_NEAR((Mat23 {1, 2, 3, 4, 5, 6})(1, 2), 6, 1e-6);
}


TEST(adapters, VectorSpaceDescriptor_deduction_guides)
{
  auto a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  EXPECT_TRUE(is_near(Matrix(a), a));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(a)), 0>, Dimensions<2>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(a)), 1>, Dimensions<3>>);

  //auto b1 = Mat23 {1, 2, 3, 4, 5, 6};
  //EXPECT_TRUE(is_near(Matrix(b1), a));
  //static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(b1)), 0>, C2>);
  //static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(b1)), 1>, C3>);

  auto b2 = Mean<C2, M23> {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(Matrix(b2), (M23() << 1, 2, 3, 4-2*pi, 5-2*pi, 6-2*pi).finished()));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(b2)), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(b2)), 1>, Dimensions<3>>);

  auto b3 = EuclideanMean<C2, M33> {1, 2, 3,
                                    0.5, std::sqrt(3)/2, sqrt2/2,
                                    std::sqrt(3)/2, 0.5, sqrt2/2};
  EXPECT_TRUE(is_near(Matrix(b3), Mat23 {1, 2, 3, pi/3, pi/6, pi/4}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(b3)), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(b3)), 1>, Dimensions<3>>);

  auto c = Covariance<C2, HermitianAdapter<M22, TriangleType::lower>> {9, 3, 3, 10};
  EXPECT_TRUE(is_near(Matrix(c), Mat22 {9, 3, 3, 10}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(c)), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Matrix(c)), 1>, C2>);
}


TEST(adapters, VectorSpaceDescriptor_make_functions)
{
  auto a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  EXPECT_TRUE(is_near(make_vector_space_adapter<C2, C3>(a), a));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter<C2, C3, 0>(a))>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter<C2, C3>(a)), 1>, C3>);
  EXPECT_TRUE(is_near(make_vector_space_adapter<C2>(a), a));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter<C2, 0>(a))>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter<C2>(a)), 1>, Dimensions<3>>);

  auto b = Mat23 {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(make_vector_space_adapter(b), a));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter(b)), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter(b)), 1>, C3>);

  auto c = Covariance<C2, HermitianAdapter<M22, TriangleType::lower>> {9, 3, 3, 10};
  EXPECT_TRUE(is_near(make_vector_space_adapter(c), Mat22 {9, 3, 3, 10}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter(c)), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter(c)), 1>, C2>);

  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter<C2, C3, M23>()), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter<C2, C3, M23>()), 1>, C3>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter<M23>()), 0>, Dimensions<2>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(make_vector_space_adapter<M23>()), 1>, Dimensions<3>>);
}


TEST(adapters, VectorSpaceDescriptor_traits)
{
  static_assert(typed_matrix<Mat23>);
  static_assert(not mean<Mat23>);
  static_assert(not euclidean_mean<Mat23>);
  static_assert(not euclidean_mean<VectorSpaceDescriptor<C2, Dimensions<2>, I22>>);
  static_assert(not euclidean_transformed<Mat23>);
  static_assert(not euclidean_transformed<VectorSpaceDescriptor<C2, Dimensions<2>, I22>>);
  static_assert(not wrapped_mean<Mat23>);
  static_assert(not wrapped_mean<VectorSpaceDescriptor<C2, Dimensions<2>, I22>>);
  static_assert(not untyped_columns<Mat23>);
  static_assert(untyped_columns<VectorSpaceDescriptor<C2, Dimensions<2>, I22>>);

  static_assert(not identity_matrix<Mat23>);
  static_assert(identity_matrix<VectorSpaceDescriptor<C2, C2, I22>>);
  static_assert(not identity_matrix<VectorSpaceDescriptor<C2, Dimensions<2>, I22>>);
  static_assert(not zero<Mat23>);
  static_assert(zero<VectorSpaceDescriptor<C2, C2, Z22>>);
  static_assert(zero<VectorSpaceDescriptor<C2, C3, ZeroAdapter<eigen_matrix_t<double, 2, 3>>>>);

  EXPECT_TRUE(is_near(make_zero<Mat23>(), eigen_matrix_t<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Mat22>(), eigen_matrix_t<double, 2, 2>::Identity()));
}


TEST(adapters, VectorSpaceDescriptor_overloads)
{
  using namespace OpenKalman::internal;
  EXPECT_TRUE(is_near(to_covariance_nestable(Mat22 {9, 3, 3, 10}), Mat22 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2l>(Mat22 {9, 3, 3, 10}), Mat22 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<SA2u>(Mat22 {9, 3, 3, 10}), Mat22 {9, 3, 3, 10}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2l>(Mat22 {3, 0, 1, 3}), Mat22 {3, 0, 1, 3}));
  EXPECT_TRUE(is_near(to_covariance_nestable<T2u>(Mat22 {3, 1, 0, 3}), Mat22 {3, 1, 0, 3}));

  EXPECT_TRUE(is_near(nested_object(Mat23 {1, 2, 3, 4, 5, 6}), Mat23 {1, 2, 3, 4, 5, 6}));

  EXPECT_TRUE(is_near(make_dense_object_from(Mat23 {1, 2, 3, 4, 5, 6}), Mat23 {1, 2, 3, 4, 5, 6}));

  EXPECT_TRUE(is_near(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6} * 2), Mat23 {2, 4, 6, 8, 10, 12}));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Mat23 {1, 2, 3, 4, 5, 6} * 2))>, Mat23>);

  EXPECT_TRUE(is_near(to_euclidean(VectorSpaceDescriptor<C2, Dimensions<3>, M23> {1, 2, 3, pi*7/3, pi*13/6, -pi*7/4}),
    EuclideanMean<C2, M33> {1, 2, 3,
                            0.5, std::sqrt(3)/2, sqrt2/2,
                            std::sqrt(3)/2, 0.5, sqrt2/2}));

  EXPECT_TRUE(is_near(to_diagonal(Mat21 {2, 3}).nested_object(), Mat22 {2, 0, 0, 3}));
  static_assert(diagonal_matrix<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(typed_matrix<decltype(to_diagonal(Mat21 {2, 3}))>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(to_diagonal(Mat21 {2, 3})), 1>, C2>);

  EXPECT_TRUE(is_near(transpose(Mat23 {1, 2, 3, 4, 5, 6}).nested_object(), Mat32 {1, 4, 2, 5, 3, 6}));

  EXPECT_TRUE(is_near(adjoint(Mat23 {1, 2, 3, 4, 5, 6}).nested_object(), Mat32 {1, 4, 2, 5, 3, 6}));

  EXPECT_NEAR(determinant(Mat22 {1, 2, 3, 4}), -2, 1e-6);

  EXPECT_NEAR(trace(Mat22 {1, 2, 3, 4}), 5, 1e-6);

  EXPECT_TRUE(is_near(solve(Mat22 {9., 3, 3, 10}, Mat21 {15, 23}), Mat21 {1, 2}));

  EXPECT_TRUE(is_near(average_reduce<1>(VectorSpaceDescriptor<C2, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6}), Mat21 {2, 5}));

  EXPECT_TRUE(is_near(square(LQ_decomposition(VectorSpaceDescriptor<C2, Dimensions<3>, M23> {1, 2, 3, 4, 5, 6})), Mat22 {14, 32, 32, 77}));

  EXPECT_TRUE(is_near(square(QR_decomposition(VectorSpaceDescriptor<Dimensions<3>, C2, M32> {1, 4, 2, 5, 3, 6})), Mat22 {14, 32, 32, 77}));

  using N = std::normal_distribution<double>;
  Mat23 m = make_zero<Mat23>();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat23>(N {1.0, 0.1}, 2.0, N {3.0, 0.1}, N {4.0, 0.1}, 5.0, N {6.0, 0.1})) / (i + 1);
  }
  Mat23 offset = {1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-8));
}


TEST(adapters, VectorSpaceDescriptor_blocks)
{
  using Mat22x = VectorSpaceDescriptor<C2, Dimensions<2>, M22>;
  EXPECT_TRUE(is_near(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6}), Mat32 {1, 2, 3, 4, 5, 6}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6})), 0>, C3>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(concatenate_vertical(Mat22 {1, 2, 3, 4}, Mat12 {5, 6})), 1>, C2>);

  EXPECT_TRUE(is_near(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6}), Mat23 {1, 2, 3, 4, 5, 6}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6})), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(concatenate_horizontal(Mat22 {1, 2, 4, 5}, Mat21 {3, 6})), 1>, C3>);

  EXPECT_TRUE(is_near(concatenate_diagonal(Mat12 {1, 2}, Mat21 {3, 4}), Mat33 {1, 2, 0, 0, 0, 3, 0, 0, 4}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(concatenate_diagonal(Mat12 {1, 2}, Mat21 {3, 4})), 0>, StaticDescriptor<Axis, Axis, angle::Radians>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(concatenate_diagonal(Mat12 {1, 2}, Mat21 {3, 4})), 1>, StaticDescriptor<Axis, angle::Radians, Axis>>);

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
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(column<0>(Mat22 {1, 2, 3, 4})), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(column<1>(Mat22 {1, 2, 3, 4})), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(column<0>(Mat22 {1, 2, 3, 4})), 1>, Axis>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(column<1>(Mat22 {1, 2, 3, 4})), 1>, angle::Radians>);

  auto m = Mat22x {1, 2, 3, 4};

  EXPECT_TRUE(is_near(apply_columnwise([](auto& col){ col *= 2; }, m), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto& col, std::size_t i){ col *= i; }, m), Mat22 {0, 4, 0, 8}));

  EXPECT_TRUE(is_near(apply_columnwise([](auto col){ return make_self_contained(col * 2); }, Mat22x {1, 2, 3, 4}), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto col){ return make_self_contained(col * 2); }, Mat22x {1, 2, 3, 4}), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto&& col){ return make_self_contained(col * 2); }, Mat22x {1, 2, 3, 4}), Mat22 {2, 4, 6, 8}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col * 2); }, Mat22x {1, 2, 3, 4}), Mat22 {2, 4, 6, 8}));

  EXPECT_TRUE(is_near(apply_columnwise([](auto col, std::size_t i){ return make_self_contained(col * i); }, Mat22x {1, 2, 3, 4}), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto col, std::size_t i){ return make_self_contained(col * i); }, Mat22x {1, 2, 3, 4}), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise([](auto&& col, std::size_t i){ return make_self_contained(col * i); }, Mat22x {1, 2, 3, 4}), Mat22 {0, 2, 0, 4}));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col * i); }, Mat22x {1, 2, 3, 4}), Mat22 {0, 2, 0, 4}));

  EXPECT_TRUE(is_near(apply_columnwise<2>([] { return VectorSpaceDescriptor<C2, angle::Radians> {1., 2}; }), Mat22 {1, 1, 2, 2}));
  EXPECT_TRUE(is_near(apply_columnwise<2>([](std::size_t i){ return VectorSpaceDescriptor<C2, angle::Radians> {i + 1., 2*i + 1}; }), Mat22 {1, 2, 1, 3}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<VectorSpaceDescriptor<C2, angle::Radians>()>())), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<VectorSpaceDescriptor<C2, angle::Radians>()>())), 1>, StaticDescriptor<angle::Radians, angle::Radians>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<VectorSpaceDescriptor<C2, angle::Radians>(std::size_t)>())), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(apply_columnwise<2>(std::declval<VectorSpaceDescriptor<C2, angle::Radians>(std::size_t)>())), 1>, StaticDescriptor<angle::Radians, angle::Radians>>);

  const auto mat22_1234 = Mat22x {1, 2, 3, 4};
  auto n = mat22_1234

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


TEST(adapters, VectorSpaceDescriptor_arithmetic)
{
  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8, 8, 8, 8}));
  EXPECT_TRUE(is_near(VectorSpaceDescriptor<C3, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} + Mean<C3, M32> {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8, 8 - 2*pi, 8, 8}));
  EXPECT_TRUE(is_near(VectorSpaceDescriptor<Dimensions<3>, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6}, Mat32 {8, 8, 8, 8, 8, 8}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat32 {7, 6, 5, 4, 3, 2} + Mat32 {1, 2, 3, 4, 5, 6}), 1>, C2>);
  static_assert(typed_matrix<decltype(VectorSpaceDescriptor<C3, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} + Mean<C3, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(VectorSpaceDescriptor<Dimensions<3>, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} + EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 0, -2, -4}));
  EXPECT_TRUE(is_near(VectorSpaceDescriptor<C3, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} - Mean<C3, M32> {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 2*pi, -2, -4}));
  EXPECT_TRUE(is_near(VectorSpaceDescriptor<Dimensions<3>, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6}, Mat32 {6, 4, 2, 0, -2, -4}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat32 {7, 6, 5, 4, 3, 2} - Mat32 {1, 2, 3, 4, 5, 6}), 1>, C2>);
  static_assert(typed_matrix<decltype(VectorSpaceDescriptor<C3, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} - Mean<C3, M32> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(VectorSpaceDescriptor<Dimensions<3>, Dimensions<2>, M32> {7, 6, 5, 4, 3, 2} - EuclideanMean<Dimensions<3>, M32> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12}));
  EXPECT_TRUE(is_near(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3, 4, 5, 6}));
  static_assert(typed_matrix<decltype(Mat32 {1, 2, 3, 4, 5, 6} * 2, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(typed_matrix<decltype(2 * Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {2, 4, 6, 8, 10, 12})>);
  static_assert(typed_matrix<decltype(Mat32 {2, 4, 6, 8, 10, 12} / 2, Mat32 {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6}, Mat23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6}), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Mat23 {1, 2, 3, 4, 5, 6}), 1>, C3>);

  EXPECT_TRUE(is_near(Mat22 {1, 2, 3, 2} * Mean<C2, M23> {1, 2, 3, 3, 2, 1}, Mat23 {7, 6, 5, 9, 10, 11}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6}), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6}), 1>, Dimensions<3>>);
  static_assert(typed_matrix<decltype(Mat22 {1, 2, 3, 4} * Mean<C2, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(VectorSpaceDescriptor<C2, Dimensions<2>, M22> {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}, Mat23 {9, 12, 15, 19, 26, 33}));
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(VectorSpaceDescriptor<C2, Dimensions<2>, M22> {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}), 0>, C2>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<decltype(VectorSpaceDescriptor<C2, Dimensions<2>, M22> {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6}), 1>, Dimensions<3>>);
  static_assert(typed_matrix<decltype(VectorSpaceDescriptor<C2, Dimensions<2>, M22> {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})>);
  static_assert(typed_matrix<decltype(VectorSpaceDescriptor<C2, Dimensions<2>, M22> {1, 2, 3, 4} * EuclideanMean<Dimensions<2>, M23> {1, 2, 3, 4, 5, 6})>);

  EXPECT_TRUE(is_near(-Mat32 {1, 2, 3, 4, 5, 6}, Mat32 {-1, -2, -3, -4, -5, -6}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} == Mat22 {1, 2, 3, 4}));
  EXPECT_TRUE((Mat22 {1, 2, 3, 4} != Mat22 {1, 2, 2, 4}));
  EXPECT_FALSE((Mat22 {1, 2, 3, 4} == VectorSpaceDescriptor<C2, Dimensions<2>, M22> {1, 2, 3, 4}));
}


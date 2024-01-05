/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, Eigen_Matrix)
{
  static_assert(Eigen3::eigen_matrix_general<M11, true>);
  static_assert(Eigen3::eigen_matrix_general<Mxx, true>);
  static_assert(Eigen3::eigen_general<M11, true>);
  static_assert(Eigen3::eigen_general<Mxx, true>);

  static_assert(index_count_v<M11> == 0);
  static_assert(index_count_v<Mxx> == 2);
  static_assert(index_count_v<M21> == 1);
  static_assert(index_count_v<Mx1> == 1);
  static_assert(index_count_v<M1x> == 2);
  static_assert(index_count_v<M01> == 1);
  static_assert(index_count_v<M00> == 2);
  static_assert(index_count_v<M20> == 2);
  static_assert(index_count_v<M02> == 2);

  static_assert(index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<2>, Dimensions<2>>> == 2);
  static_assert(index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<2>, Dimensions<1>>> == 1);
  static_assert(index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<1>, Dimensions<2>>> == 2);
  static_assert(index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<1>, Dimensions<1>>> == 0);

  // The orders of these empty matrices are considered to be 0 for now:
  static_assert(max_tensor_order_v<M00> == 0);
  static_assert(max_tensor_order_v<M02> == 0);
  static_assert(max_tensor_order_v<M20> == 0);
  static_assert(max_tensor_order_v<M01> == 0);
  static_assert(max_tensor_order_v<M10> == 0);

  static_assert(max_tensor_order_v<M23> == 2);
  static_assert(max_tensor_order_v<M21> == 1);
  static_assert(max_tensor_order_v<M11> == 0);
  static_assert(max_tensor_order_v<M2x> == 2);
  static_assert(max_tensor_order_v<Mx2> == 2);
  static_assert(max_tensor_order_v<M1x> == 1);
  static_assert(max_tensor_order_v<Mx1> == 1);
  static_assert(max_tensor_order_v<Mxx> == 2);

  static_assert(index_dimension_of_v<M11, 0> == 1);
  static_assert(index_dimension_of_v<M21, 0> == 2);
  static_assert(index_dimension_of_v<Mxx, 0> == dynamic_size);
  static_assert(index_dimension_of_v<M11, 1> == 1);
  static_assert(index_dimension_of_v<M21, 1> == 1);
  static_assert(index_dimension_of_v<Mxx, 1> == dynamic_size);
  EXPECT_EQ(get_vector_space_descriptor<0>(M11{}), 1);
  EXPECT_EQ(get_vector_space_descriptor<0>(M21{}), 2);
  EXPECT_EQ((get_vector_space_descriptor<0>(Mxx{2, 1})), 2);
  EXPECT_EQ((get_vector_space_descriptor<1>(M11{})), 1);
  EXPECT_EQ((get_vector_space_descriptor<1>(M21{})), 1);
  EXPECT_EQ((get_vector_space_descriptor<1>(Mxx{2, 1})), 1);

  static_assert(std::is_same_v<typename interface::indexible_object_traits<Mxx>::scalar_type, double>);

  static_assert(dynamic_dimension<eigen_matrix_t<double, dynamic_size, dynamic_size>, 0>);
  static_assert(dynamic_dimension<eigen_matrix_t<double, dynamic_size, dynamic_size>, 1>);
  static_assert(dynamic_dimension<eigen_matrix_t<double, dynamic_size, 1>, 0>);
  static_assert(not dynamic_dimension<eigen_matrix_t<double, dynamic_size, 1>, 1>);
  static_assert(not dynamic_dimension<eigen_matrix_t<double, 1, dynamic_size>, 0>);
  static_assert(dynamic_dimension<eigen_matrix_t<double, 1, dynamic_size>, 1>);

  static_assert(dynamic_index_count_v<M22> == 0);
  static_assert(dynamic_index_count_v<M2x> == 1);
  static_assert(dynamic_index_count_v<Mx2> == 1);
  static_assert(dynamic_index_count_v<Mxx> == 2);

  static_assert(dynamic_index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<2>, Dimensions<2>>> == 0);
  static_assert(dynamic_index_count_v<internal::FixedSizeAdapter<const Mxx, Dimensions<2>, std::size_t>> == 1);
  static_assert(dynamic_index_count_v<internal::FixedSizeAdapter<const Mxx, std::size_t, std::size_t>> == 2);
  static_assert(dynamic_index_count_v<internal::FixedSizeAdapter<const Mxx>> == 0);

  static_assert(std::is_same_v<dense_writable_matrix_t<M33>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<M3x>, M3x>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Mx3>, Mx3>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Mxx>, Mxx>);

  static_assert(writable<M22>);
  static_assert(writable<M2x>);
  static_assert(writable<Mx2>);
  static_assert(writable<Mxx>);
  static_assert(writable<M22&>);
  static_assert(not writable<const M22>);
  static_assert(not writable<const M22&>);
  static_assert(writable<dense_writable_matrix_t<M22>>);

  static_assert(interface::get_component_defined_for<M32, M32, std::array<std::size_t, 2>>);

  static_assert(element_gettable<M32, 2>);
  static_assert(element_gettable<const M32, 2>);
  static_assert(element_gettable<M31, 2>);
  static_assert(element_gettable<M13, 2>);
  static_assert(element_gettable<M3x, 2>);
  static_assert(element_gettable<Mx2, 2>);
  static_assert(element_gettable<Mx1, 2>);
  static_assert(element_gettable<M1x, 2>);
  static_assert(element_gettable<Mxx, 2>);

  static_assert(element_gettable<M32, 2>);
  static_assert(element_gettable<M31, 1>);
  static_assert(element_gettable<M13, 2>);
  static_assert(element_gettable<Mx2, 2>);
  static_assert(element_gettable<Mx1, 1>);
  static_assert(element_gettable<M1x, 2>);
  static_assert(element_gettable<Mxx, 2>);

  static_assert(element_settable<M32, 2>);
  static_assert(element_settable<M32&&, 2>);
  static_assert(not element_settable<const M32&, 2>);
  static_assert(element_settable<M31&, 2>);
  static_assert(element_settable<M13&, 2>);
  static_assert(element_settable<M3x&, 2>);
  static_assert(element_settable<Mx2&, 2>);
  static_assert(element_settable<Mx1&, 2>);
  static_assert(element_settable<M1x&, 2>);
  static_assert(element_settable<Mxx&, 2>);

  static_assert(element_settable<M32&, 2>);
  static_assert(element_settable<M31&, 1>);
  static_assert(element_settable<M13&, 2>);
  static_assert(not element_settable<const M31&, 1>);
  static_assert(element_settable<Mx2&, 2>);
  static_assert(element_settable<Mx1&, 1>);
  static_assert(element_settable<M1x&, 2>);
  static_assert(element_settable<Mxx&, 2>);

  M22 m22; m22 << 1, 2, 3, 4;
  M23 m23; m23 << 1, 2, 3, 4, 5, 6;
  Mx3 mx3_2 {2,3}; mx3_2 << 1, 2, 3, 4, 5, 6;
  M32 m32; m32 << 1, 2, 3, 4, 5, 6;
  CM22 cm22; cm22 << cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1};

  EXPECT_TRUE(is_near(make_dense_object_from<M22>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<M2x>(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(make_dense_object_from<Mx2>(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(make_dense_object_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(index_dimension_of_v<decltype(make_dense_object_from<M2x>(1, 2)), 1> == 1);
  static_assert(index_dimension_of_v<decltype(make_dense_object_from<Mx2>(1, 2)), 0> == 1);

  static_assert(std::is_same_v<vector_space_descriptor_of<M11, 0>::type, Dimensions<1>>);
  static_assert(std::is_same_v<vector_space_descriptor_of<M11, 1>::type, Dimensions<1>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<M11, 0>, Axis>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<M11, 1>, Axis>);
  static_assert(std::is_same_v<vector_space_descriptor_of<M22, 0>::type, Dimensions<2>>);
  static_assert(std::is_same_v<vector_space_descriptor_of<M22, 1>::type, Dimensions<2>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<M22, 0>, TypedIndex<Axis, Axis>>);
  static_assert(equivalent_to<vector_space_descriptor_of_t<M22, 1>, TypedIndex<Axis, Axis>>);

  static_assert(maybe_same_shape_as<M22, M2x, Mx2, Mxx>);
  static_assert(same_shape_as<M22, CM22, M22>);
  EXPECT_TRUE(same_shape(m22, cm22, M2x{m22}, Mx2{m22}, Mxx{m22}));

  static_assert(compatible_with_vector_space_descriptors<M23, std::integral_constant<int, 2>, std::integral_constant<int, 3>>);
  static_assert(not compatible_with_vector_space_descriptors<M23, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_vector_space_descriptors<M23, int, int>);
  static_assert(compatible_with_vector_space_descriptors<M2x, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_vector_space_descriptors<M2x, std::integral_constant<int, 2>, int>);
  static_assert(compatible_with_vector_space_descriptors<M2x, int, int>);
  static_assert(not compatible_with_vector_space_descriptors<M2x, std::integral_constant<int, 3>, int>);
  static_assert(compatible_with_vector_space_descriptors<Mx2, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_vector_space_descriptors<Mx2, std::integral_constant<int, 2>, int>);
  static_assert(compatible_with_vector_space_descriptors<Mx2, int, int>);
  static_assert(not compatible_with_vector_space_descriptors<Mx2, int, std::integral_constant<int, 3>>);
  static_assert(compatible_with_vector_space_descriptors<Mxx, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_vector_space_descriptors<Mxx, int, int>);

  static_assert(empty_object<M00>);
  static_assert(empty_object<M01>);
  static_assert(empty_object<M10>);
  static_assert(empty_object<M0x>);
  static_assert(empty_object<Mx0>);
  static_assert(not empty_object<M11>);
  static_assert(not empty_object<M1x>);
  static_assert(not empty_object<Mx1>);
  static_assert(not empty_object<Mxx>);
  static_assert(empty_object<M1x, Qualification::depends_on_dynamic_shape>);
  static_assert(empty_object<Mx1, Qualification::depends_on_dynamic_shape>);
  static_assert(empty_object<Mxx, Qualification::depends_on_dynamic_shape>);

  static_assert(one_dimensional<M11>);
  static_assert(not one_dimensional<M1x>);
  static_assert(one_dimensional<M1x, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<M00>);
  static_assert(not one_dimensional<M01>);
  static_assert(not one_dimensional<M10>);
  static_assert(not one_dimensional<M00, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Mx0, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<M0x, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<M00, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<M0x, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Mx0, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<M11, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<M22, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<M32, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<M2x, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Mx2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Mxx, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<CM22, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<CM32, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<CM2x, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<CMx2, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<CMxx, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<M00>);
  static_assert(not square_shaped<M0x>);
  static_assert(not square_shaped<Mx0>);
  static_assert(square_shaped<M11>);
  static_assert(square_shaped<M22>);
  static_assert(not square_shaped<M2x>);
  static_assert(not square_shaped<Mx2>);
  static_assert(not square_shaped<Mxx>);
  static_assert(square_shaped<CM22>);
  static_assert(not square_shaped<CM2x>);
  static_assert(not square_shaped<CMx2>);
  static_assert(not square_shaped<CMxx>);

  M11 m11_1{1};
  Eigen::Matrix<double, 0, 0> m00;

  static_assert(is_square_shaped(m00));
  static_assert(is_square_shaped(m22));
  static_assert(*is_square_shaped(m22) == Dimensions<2>{});
  EXPECT_TRUE(is_square_shaped(M2x{m22}));
  EXPECT_TRUE(*is_square_shaped(M2x{m22}) == Dimensions<2>{});
  EXPECT_TRUE(is_square_shaped(Mx2{m22}));
  EXPECT_TRUE(*is_square_shaped(Mx2{m22}) == Dimensions<2>{});
  EXPECT_TRUE(is_square_shaped(Mxx{m22}));
  EXPECT_EQ(*is_square_shaped(Mxx{m22}), 2);
  static_assert(is_square_shaped(m11_1));
  static_assert(*is_square_shaped(m11_1) == Dimensions<1>{});
  EXPECT_TRUE(is_square_shaped(M1x{m11_1}));
  EXPECT_TRUE(*is_square_shaped(M1x{m11_1}) == Dimensions<1>{});
  EXPECT_TRUE(is_square_shaped(Mx1{m11_1}));
  EXPECT_TRUE(*is_square_shaped(Mx1{m11_1}) == Dimensions<1>{});
  EXPECT_TRUE(is_square_shaped(Mxx{m11_1}));
  EXPECT_EQ(*is_square_shaped(Mxx{m11_1}), 1);

  static_assert(is_one_dimensional(m11_1));
  EXPECT_TRUE(is_one_dimensional(M1x{m11_1}));
  EXPECT_TRUE(is_one_dimensional(Mx1{m11_1}));
  EXPECT_TRUE(is_one_dimensional(Mxx{m11_1}));
  static_assert(not is_one_dimensional(m00));

  static_assert(dimension_size_of_index_is<M31, 1, 1>);
  static_assert(dimension_size_of_index_is<Mx1, 1, 1>);
  static_assert(dimension_size_of_index_is<M3x, 1, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(dimension_size_of_index_is<Mxx, 1, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(not dimension_size_of_index_is<M32, 1, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(not dimension_size_of_index_is<Mx2, 1, 1, Qualification::depends_on_dynamic_shape>);

  static_assert(dimension_size_of_index_is<M13, 0, 1>);
  static_assert(dimension_size_of_index_is<M1x, 0, 1>);
  static_assert(dimension_size_of_index_is<Mx3, 0, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(dimension_size_of_index_is<Mxx, 0, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(not dimension_size_of_index_is<M23, 0, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(not dimension_size_of_index_is<M2x, 0, 1, Qualification::depends_on_dynamic_shape>);

  M21 m21 {1, 2};
  M2x m2x_1 {m21};
  Mx1 mx1_2 {m21};
  Mxx mxx_21 {m21};
  M12 m12 {1, 2};
  M1x m1x_2 {m12};
  Mx2 mx2_1 {m12};
  Mxx mxx_12 {m12};

  static_assert(vector<M11>);
  static_assert(is_vector(m11_1));

  static_assert(vector<M21>);
  static_assert(vector<M2x, 0, Qualification::depends_on_dynamic_shape>);
  static_assert(vector<Mx1, 0, Qualification::depends_on_dynamic_shape>);
  static_assert(vector<Mxx, 0, Qualification::depends_on_dynamic_shape>);
  static_assert(not vector<M12, 0, Qualification::depends_on_dynamic_shape>);
  static_assert(not vector<Mx2, 0, Qualification::depends_on_dynamic_shape>);
  static_assert(is_vector(m21));
  EXPECT_TRUE(is_vector(m2x_1));
  static_assert(is_vector(mx1_2));
  EXPECT_TRUE(is_vector(mxx_21));
  EXPECT_FALSE(is_vector(mxx_12));

  static_assert(vector<M12, 1>);
  static_assert(vector<M1x, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(vector<Mx2, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(vector<Mxx, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(not vector<M21, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(not vector<M2x, 1, Qualification::depends_on_dynamic_shape>);
  static_assert(is_vector<1>(m12));
  static_assert(is_vector<1>(m1x_2));
  EXPECT_TRUE(is_vector<1>(mx2_1));
  EXPECT_TRUE(is_vector<1>(mxx_12));
  EXPECT_FALSE(is_vector<1>(mxx_21));

  static_assert(eigen_matrix_general<M22, true>);
  static_assert(eigen_matrix_general<Mxx, true>);
  static_assert(eigen_matrix_general<CM22, true>);
  static_assert(eigen_matrix_general<CMxx, true>);
  static_assert(not eigen_matrix_general<double, true>);

  //static_assert(modifiable<M33, M33>);
  //static_assert(not modifiable<M33, M31>);
  //static_assert(not modifiable<M33, eigen_matrix_t<int, 3, 3>>);
  //static_assert(not modifiable<const M33, M33>);
  //static_assert(modifiable<M33, Eigen3::IdentityMatrix<M33>>);
}


/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;


TEST(eigen3, count_indices)
{
  M23 m23;
  static_assert(decltype(count_indices(m23))::value == 2);
  static_assert(count_indices(m23) == 2);
  M00 m00;
  static_assert(count_indices(m00) == 2);
  M11 m11;
  static_assert(count_indices(m11) == 0);
  Mx1 mx1(0, 1);
  static_assert(count_indices(mx1) == 1);
}


TEST(eigen3, get_vector_space_descriptor)
{
  M23 m23;

  static_assert(dimension_size_of_v<decltype(get_vector_space_descriptor<0>(m23))> == 2);
  static_assert(dimension_size_of_v<decltype(get_vector_space_descriptor<0>(M2x {m23}))> == 2);
  EXPECT_EQ(get_dimension_size_of(get_vector_space_descriptor<0>(Mx3 {m23})), 2);
  EXPECT_EQ(get_dimension_size_of(get_vector_space_descriptor<0>(Mxx {m23})), 2);

  static_assert(dimension_size_of_v<decltype(get_vector_space_descriptor<1>(m23))> == 3);
  EXPECT_EQ(get_dimension_size_of(get_vector_space_descriptor<1>(M2x {m23})), 3);
  static_assert(dimension_size_of_v<decltype(get_vector_space_descriptor<1>(Mx3 {m23}))> == 3);
  EXPECT_EQ(get_dimension_size_of(get_vector_space_descriptor<1>(Mxx {m23})), 3);
}


TEST(eigen3, get_index_dimension_of)
{
  M23 m23;

  static_assert(get_index_dimension_of<0>(m23) == 2);
  EXPECT_EQ(get_index_dimension_of<0>(M2x {m23}), 2);
  EXPECT_EQ(get_index_dimension_of<0>(Mx3 {m23}), 2);
  EXPECT_EQ(get_index_dimension_of<0>(Mxx {m23}), 2);

  static_assert(get_index_dimension_of<1>(m23) == 3);
  EXPECT_EQ(get_index_dimension_of<1>(M2x {m23}), 3);
  EXPECT_EQ(get_index_dimension_of<1>(Mx3 {m23}), 3);
  EXPECT_EQ(get_index_dimension_of<1>(Mxx {m23}), 3);
}


TEST(eigen3, tensor_order)
{
  M23 m23; static_assert(tensor_order(m23) == 2);
  EXPECT_EQ(tensor_order(M2x(2,3)), 2);
  EXPECT_EQ(tensor_order(Mx3(2,3)), 2);
  EXPECT_EQ(tensor_order(Mxx(2,3)), 2);

  M21 m21; static_assert(tensor_order(m21) == 1);
  EXPECT_EQ(tensor_order(M2x(2,1)), 1);
  EXPECT_EQ(tensor_order(Mx1(2,1)), 1);
  EXPECT_EQ(tensor_order(Mxx(2,1)), 1);

  M11 m11; static_assert(tensor_order(m11) == 0);
  EXPECT_EQ(tensor_order(M1x(1,1)), 0);
  EXPECT_EQ(tensor_order(Mx1(1,1)), 0);
  EXPECT_EQ(tensor_order(Mxx(1,1)), 0);
}


TEST(eigen3, all_vector_space_descriptors)
{
  static_assert(dimension_size_of_v<decltype(std::get<0>(all_vector_space_descriptors(std::declval<M23>())))> == 2);
  static_assert(dimension_size_of_v<decltype(std::get<0>(all_vector_space_descriptors<M23>()))> == 2);
  static_assert(dimension_size_of_v<decltype(std::get<0>(all_vector_space_descriptors(std::declval<M2x>())))> == 2);
  static_assert(dimension_size_of_v<decltype(std::get<1>(all_vector_space_descriptors(std::declval<M23>())))> == 3);
  static_assert(dimension_size_of_v<decltype(std::get<1>(all_vector_space_descriptors(std::declval<Mx3>())))> == 3);
  static_assert(std::tuple_size_v<decltype(all_vector_space_descriptors(std::declval<Mxx>()))> == 2);

  EXPECT_EQ(std::get<0>(all_vector_space_descriptors(Mxx(2, 3))), 2);
  EXPECT_EQ(std::get<1>(all_vector_space_descriptors(Mxx(2, 3))), 3);
}


TEST(eigen3, same_shape)
{
  EXPECT_TRUE(same_shape(M23{}, Mxx(2, 3)));
  EXPECT_FALSE(same_shape(M23{}, Mxx(2, 3), Mxx(2, 2)));
}


TEST(eigen3, is_square_shaped)
{
  EXPECT_TRUE(is_square_shaped(M22{}));
  EXPECT_FALSE(is_square_shaped(M23{}));
  EXPECT_TRUE(is_square_shaped(Mxx(3, 3)));
  EXPECT_FALSE(is_square_shaped(Mxx(2, 3)));
}


TEST(eigen3, is_one_dimensional)
{
  EXPECT_TRUE(is_one_dimensional(M11{}));
  EXPECT_FALSE(is_one_dimensional(M12{}));
  EXPECT_TRUE(is_one_dimensional(Mxx(1, 1)));
  EXPECT_FALSE(is_one_dimensional(Mxx(2, 1)));
}


TEST(eigen3, is_vector)
{
  EXPECT_TRUE(is_vector(M11{}));
  EXPECT_TRUE(is_vector(M31{}));
  EXPECT_FALSE(is_vector(M32{}));
  EXPECT_FALSE(is_vector(M13{}));
  EXPECT_TRUE(is_vector(Mxx(1, 1)));
  EXPECT_TRUE(is_vector(Mxx(3, 1)));
  EXPECT_FALSE(is_vector(Mxx(3, 2)));
  EXPECT_FALSE(is_vector(Mxx(1, 3)));

  EXPECT_TRUE(is_vector<1>(M11{}));
  EXPECT_TRUE(is_vector<1>(M13{}));
  EXPECT_FALSE(is_vector<1>(M23{}));
  EXPECT_FALSE(is_vector<1>(M31{}));
  EXPECT_TRUE(is_vector<1>(Mxx(1, 1)));
  EXPECT_TRUE(is_vector<1>(Mxx(1, 3)));
  EXPECT_FALSE(is_vector<1>(Mxx(2, 3)));
  EXPECT_FALSE(is_vector<1>(Mxx(3, 1)));
}


TEST(eigen3, nested_object)
{
  M22 m22_93310; m22_93310 << 9, 3, 3, 10;
  M22 m22_3103; m22_3103 << 3, 1, 0, 3;
  M21 m21 {1, 4};

  EXPECT_TRUE(is_near(nested_object(Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310}), m22_93310));
  EXPECT_TRUE(is_near(nested_object(Eigen::TriangularView<M22, Eigen::Upper> {m22_3103}), m22_3103));
  EXPECT_TRUE(is_near(nested_object(Eigen::DiagonalMatrix<double, 2> {m21}), m21));
  EXPECT_TRUE(is_near(nested_object(Eigen::DiagonalWrapper<M21> {m21}), m21));
}


TEST(eigen3, raw_data)
{
  Eigen::Matrix<double, 2, 2, Eigen::ColMajor> m22_1234c;
  m22_1234c << 1, 2, 3, 4;
  auto* datac = internal::raw_data(m22_1234c);
  static_assert(layout_of_v<decltype(m22_1234c)> == Layout::left);

  EXPECT_EQ(std::get<0>(internal::strides(m22_1234c)), 1);
  EXPECT_EQ(std::get<1>(internal::strides(m22_1234c)), 2);
  EXPECT_EQ(datac[0], 1);
  EXPECT_EQ(datac[1], 3);
  EXPECT_EQ(datac[2], 2);
  EXPECT_EQ(datac[3], 4);

  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> m22_1234r;
  m22_1234r << 1, 2, 3, 4;
  auto* datar = internal::raw_data(m22_1234r);
  static_assert(layout_of_v<decltype(m22_1234r)> == Layout::right);
  EXPECT_EQ(std::get<0>(internal::strides(m22_1234r)), 2);
  EXPECT_EQ(std::get<1>(internal::strides(m22_1234r)), 1);

  EXPECT_EQ(datar[0], 1);
  EXPECT_EQ(datar[1], 2);
  EXPECT_EQ(datar[2], 3);
  EXPECT_EQ(datar[3], 4);
}


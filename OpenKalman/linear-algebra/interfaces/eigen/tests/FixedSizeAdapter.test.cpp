/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;


TEST(eigen3, FixedSizeAdapter)
{
  static_assert(values::dynamic<constant_coefficient<internal::FixedSizeAdapter<const Mxx, std::tuple<Dimensions<1>, Dimensions<1>>>>>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Mxx, std::tuple<Dimensions<1>, std::size_t>>>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Mxx, std::tuple<std::size_t, Dimensions<1>>>>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const M2x, std::tuple<Dimensions<2>, Dimensions<1>>>>);

  static_assert(dimension_size_of_index_is<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>, 0, 2>);
  static_assert(dimension_size_of_index_is<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>, 1, 1>);

  static_assert(not one_dimensional<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>, Applicability::permitted>);
  static_assert(not one_dimensional<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>, Applicability::permitted>);
  static_assert(not one_dimensional<internal::FixedSizeAdapter<const Eigen::Diagonal<Mx2, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>, Applicability::permitted>);
  static_assert(one_dimensional<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, std::tuple<Dimensions<1>, Dimensions<1>>>>);

  static_assert(not square_shaped<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>, Applicability::permitted>);

  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mx2, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>>);
  static_assert(not constant_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>>);
  static_assert(values::dynamic<constant_coefficient<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, std::tuple<Dimensions<1>, Dimensions<1>>>>>);

  static_assert(not constant_diagonal_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<M2x, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>>);
  static_assert(not constant_diagonal_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mx2, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>>);
  static_assert(not constant_diagonal_matrix<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, std::tuple<Dimensions<2>, Dimensions<1>>>>);
  static_assert(values::dynamic<constant_diagonal_coefficient<internal::FixedSizeAdapter<const Eigen::Diagonal<Mxx, 0>, std::tuple<Dimensions<1>, Dimensions<1>>>>>);

  EXPECT_EQ(get_vector_space_descriptor(internal::FixedSizeAdapter<Mxx, std::tuple<Dimensions<2>, Dimensions<3>>>{}, std::integral_constant<std::size_t, 0>{}), 2);
  EXPECT_EQ(get_vector_space_descriptor(internal::FixedSizeAdapter<Mxx, std::tuple<Dimensions<2>, Dimensions<3>>>{}, std::integral_constant<std::size_t, 1>{}), 3);
  EXPECT_EQ(get_vector_space_descriptor(internal::FixedSizeAdapter<Mxx, std::tuple<Dimensions<2>, Dimensions<3>>>{}, std::integral_constant<std::size_t, 2>{}), 1);
}


TEST(eigen3, make_fixed_size_adapter)
{
  auto m22 = make_dense_object_from<M22>(1, 2, 3, 4);
  auto m2x_2 = M2x {m22};
  auto mx2_2 = Mx2 {m22};
  auto mxx_22 = Mxx {m22};

  static_assert(not internal::fixed_size_adapter<decltype(internal::make_fixed_size_adapter<Dimensions<2>, Dimensions<2>>(std::declval<M22>()))>);
  static_assert(internal::fixed_size_adapter<decltype(internal::make_fixed_size_adapter<Dimensions<2>, Dimensions<2>>(std::declval<M2x>()))>);
  static_assert(internal::fixed_size_adapter<decltype(internal::make_fixed_size_adapter<Dimensions<2>, Dimensions<2>>(std::declval<Mx2>()))>);
  static_assert(internal::fixed_size_adapter<decltype(internal::make_fixed_size_adapter<Dimensions<2>, Dimensions<2>>(std::declval<Mxx>()))>);

  static_assert(square_shaped<decltype(internal::make_fixed_size_adapter<Dimensions<2>, Dimensions<2>>(std::declval<M2x>()))>);
  static_assert(square_shaped<decltype(internal::make_fixed_size_adapter<Dimensions<2>, Dimensions<2>>(std::declval<Mx2>()))>);
  static_assert(square_shaped<decltype(internal::make_fixed_size_adapter<Dimensions<2>, Dimensions<2>>(std::declval<Mxx>()))>);

  static_assert(internal::fixed_size_adapter<decltype(internal::make_fixed_size_adapter<Dimensions<2>>(std::declval<M2x>()))>);
  static_assert(index_dimension_of_v<decltype(internal::make_fixed_size_adapter<Dimensions<2>>(std::declval<M2x>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(internal::make_fixed_size_adapter<Dimensions<2>>(std::declval<M2x>())), 1> == 1);
  static_assert(index_dimension_of_v<decltype(internal::make_fixed_size_adapter<Dimensions<2>>(std::declval<Mxx>())), 0> == 2);
  static_assert(index_dimension_of_v<decltype(internal::make_fixed_size_adapter<Dimensions<2>>(std::declval<Mxx>())), 1> == 1);
  static_assert(index_dimension_of_v<decltype(internal::make_fixed_size_adapter(std::declval<Mxx>())), 0> == 1);
  static_assert(index_dimension_of_v<decltype(internal::make_fixed_size_adapter(std::declval<Mxx>())), 1> == 1);
  static_assert(index_dimension_of_v<decltype(internal::make_fixed_size_adapter(std::declval<Mxx>())), 2> == 1);

  static_assert(not internal::fixed_size_adapter<decltype(internal::make_fixed_size_adapter<Dimensions<dynamic_size>>(std::declval<Mx1>()))>);
}
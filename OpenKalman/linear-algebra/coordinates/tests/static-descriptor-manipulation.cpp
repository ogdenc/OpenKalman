/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for coefficient types
 */

#include "basics/tests/tests.hpp"

#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/dynamic_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"

#include "linear-algebra/coordinates/concepts/pattern_tuple.hpp"
#include "linear-algebra/coordinates/concepts/pattern_collection.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern_tuple.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern_collection.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern_tuple.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern_collection.hpp"

#include "linear-algebra/coordinates/concepts/descriptor.hpp" //
#include "linear-algebra/coordinates/concepts/compares_with.hpp" //
#include "linear-algebra/coordinates/traits/dimension_of.hpp" //
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp" //

#include "linear-algebra/coordinates/functions/get_dimension.hpp" //
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp" //
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp" //

#include "linear-algebra/coordinates/functions/to_stat_space.hpp" //
#include "linear-algebra/coordinates/functions/from_stat_space.hpp" //
#include "linear-algebra/coordinates/functions/get_wrapped_component.hpp" //
#include "linear-algebra/coordinates/functions/set_wrapped_component.hpp" //

// descriptors:

#include "linear-algebra/coordinates/descriptors/Dimensions.hpp" //
#include "linear-algebra/coordinates/descriptors/Distance.hpp" //
#include "linear-algebra/coordinates/descriptors/Angle.hpp" //
#include "linear-algebra/coordinates/descriptors/Inclination.hpp" //
#include "linear-algebra/coordinates/descriptors/Polar.hpp" //
#include "linear-algebra/coordinates/descriptors/Spherical.hpp" //

// traits for manipulating static descriptors

#include "linear-algebra/coordinates/traits/internal/uniform_static_vector_space_descriptor_query.hpp" //
#include "linear-algebra/coordinates/concepts/uniform_static_vector_space_descriptor.hpp" //
#include "linear-algebra/coordinates/traits/uniform_static_vector_space_descriptor_component_of.hpp" //
#include "linear-algebra/coordinates/concepts/equivalent_to_uniform_static_vector_space_descriptor_component_of.hpp" //

// functions:

#include "linear-algebra/coordinates/functions/comparison-operators.hpp"

#include "linear-algebra/coordinates/functions/internal/is_uniform_component_of.hpp"
#include "linear-algebra/coordinates/functions/internal/remove_trailing_1D_descriptors.hpp"
#include "linear-algebra/coordinates/functions/internal/best_vector_space_descriptor.hpp"
#include "linear-algebra/coordinates/functions/internal/smallest_vector_space_descriptor.hpp"
#include "linear-algebra/coordinates/functions/internal/largest_vector_space_descriptor.hpp"

#include "linear-algebra/coordinates/functions/get_slice.hpp" //

#include "linear-algebra/coordinates/functions/internal/to_euclidean_vector_space_descriptor_collection.hpp"

using namespace OpenKalman::coordinates;

TEST(coordinates, uniform_static_vector_space_descriptor)
{
  static_assert(not uniform_static_vector_space_descriptor<Dimensions<0>>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<Dimensions<1>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<Dimensions<5>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<std::integral_constant<std::size_t, 5>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<std::size_t>, Axis>);

  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<Axis>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<Distance>, Distance>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<angle::Radians>, angle::Radians>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<inclination::Radians>, inclination::Radians>);
  static_assert(not uniform_static_vector_space_descriptor<Polar<>>);
  static_assert(not uniform_static_vector_space_descriptor<Spherical<>>);

  static_assert(not uniform_static_vector_space_descriptor<std::tuple<>>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<std::tuple<Axis>>, Axis>);
  static_assert(not uniform_static_vector_space_descriptor<std::tuple<Axis, angle::Radians>>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<std::tuple<Axis, Axis>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<std::tuple<angle::Radians, angle::Radians>>, angle::Radians>);
  static_assert(not uniform_static_vector_space_descriptor<std::tuple<Polar<>, Polar<>>>);
}


TEST(coordinates, remove_trailing_1D_descriptors)
{
  using D123 = std::tuple<Dimensions<1>, Dimensions<2>, Dimensions<3>>;
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{})), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}, Dimensions<1>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<1>{}, Dimensions<1>{}})), std::tuple<>>);
}


TEST(coordinates, smallest_vector_space_descriptor_fixed)
{
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<3>{}, Dimensions<4>{}) == Dimensions<3>{});
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<3>{}, angle::Radians{}) == angle::Radians{});
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<1>{}, angle::Radians{}) == Dimensions<1>{});
  static_assert(internal::smallest_vector_space_descriptor<double>(angle::Radians{}, Dimensions<1>{}, angle::Degrees{}) == angle::Radians{});
}


TEST(coordinates, largest_vector_space_descriptor_fixed)
{
  static_assert(internal::largest_vector_space_descriptor<double>(Dimensions<3>{}, Dimensions<4>{}) == Dimensions<4>{});
  static_assert(internal::largest_vector_space_descriptor<double>(Dimensions<3>{}, angle::Radians{}) == Dimensions<3>{});
  static_assert(internal::largest_vector_space_descriptor<double>(Dimensions<2>{}, Spherical<>{}) == Spherical<>{});
  static_assert(internal::largest_vector_space_descriptor<double>(Dimensions<1>{}, angle::Radians{}) == Dimensions<1>{});
  static_assert(internal::largest_vector_space_descriptor<double>(angle::Radians{}, Dimensions<1>{}, angle::Degrees{}) == angle::Radians{});
}


TEST(coordinates, pattern_collection)
{
  static_assert(collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(collection<std::vector<angle::Radians>>);

  static_assert(pattern_tuple<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(pattern_tuple<std::array<Distance, 5>>);
  static_assert(not pattern_tuple<std::tuple<Axis, Distance, double, angle::Radians>>);
  static_assert(not pattern_tuple<std::vector<angle::Radians>>);
  static_assert(not pattern_tuple<std::initializer_list<angle::Radians>>);

  static_assert(euclidean_pattern_tuple<std::tuple<Axis, Dimensions<3>, int, std::integral_constant<std::size_t, 5>>>);
  static_assert(not euclidean_pattern_tuple<std::tuple<Axis, Distance, angle::Radians>>);

  static_assert(pattern_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(pattern_collection<std::array<Distance, 5>>);
  static_assert(pattern_collection<std::vector<angle::Radians>>);
  static_assert(pattern_collection<std::initializer_list<angle::Radians>>);

  static_assert(euclidean_pattern_collection<std::tuple<Axis, Dimensions<3>, int, std::integral_constant<std::size_t, 5>>>);
  static_assert(euclidean_pattern_collection<std::array<Dimensions<4>, 5>>);
  static_assert(euclidean_pattern_collection<std::vector<Axis>>);
  static_assert(euclidean_pattern_collection<std::initializer_list<int>>);
  static_assert(not euclidean_pattern_collection<std::array<Distance, 5>>);
  static_assert(not euclidean_pattern_collection<std::vector<angle::Radians>>);
  static_assert(not euclidean_pattern_collection<std::initializer_list<Distance>>);

  static_assert(fixed_pattern_tuple<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(not fixed_pattern_tuple<std::tuple<Axis, Distance, int, angle::Radians>>);

  static_assert(fixed_pattern_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(fixed_pattern_collection<std::array<Distance, 5>>);
  static_assert(fixed_pattern_collection<std::vector<angle::Radians>>);
  static_assert(fixed_pattern_collection<std::initializer_list<angle::Radians>>);
  static_assert(not fixed_pattern_collection<std::tuple<Axis, int, Distance, angle::Radians>>);
  static_assert(not fixed_pattern_collection<std::array<Dimensions<dynamic_size>, 5>>);
  static_assert(not fixed_pattern_collection<std::vector<int>>);
  static_assert(not fixed_pattern_collection<std::initializer_list<int>>);
}


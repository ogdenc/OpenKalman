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

#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"

#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor_tuple.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor_collection.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor_tuple.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor_collection.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor_tuple.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor_collection.hpp"

#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/static_concatenate.hpp" //

#include "linear-algebra/vector-space-descriptors/concepts/composite_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/atomic_static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_component_count_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_is_euclidean.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/to_euclidean_element.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/from_euclidean_element.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_wrapped_component.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/set_wrapped_component.hpp" //

// internal traits:

#include "linear-algebra/vector-space-descriptors/concepts/internal/prefix_of.hpp" //

// descriptors:

#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/DynamicDescriptor.hpp" //

#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp" //

// traits for manipulating static descriptors

#include "linear-algebra/vector-space-descriptors/traits/replicate_static_vector_space_descriptor.hpp"

#include "linear-algebra/vector-space-descriptors/traits/internal/uniform_static_vector_space_descriptor_query.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/uniform_static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/uniform_static_vector_space_descriptor_component_of.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to_uniform_static_vector_space_descriptor_component_of.hpp" //

// functions:

#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"

#include "linear-algebra/vector-space-descriptors/functions/internal/replicate_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/is_uniform_component_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/remove_trailing_1D_descriptors.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/best_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/smallest_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/largest_vector_space_descriptor.hpp"

#include "linear-algebra/vector-space-descriptors/functions/internal/split_head_tail.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/internal/static_vector_space_descriptor_slice.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_slice.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/internal/to_euclidean_vector_space_descriptor_collection.hpp"

using namespace OpenKalman::descriptor;

TEST(basics, prepend_append)
{
  static_assert(std::is_same_v<StaticDescriptor<angle::Radians, Axis>::Prepend<Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<StaticDescriptor<Axis, angle::Radians>::Prepend<Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<StaticDescriptor<angle::Radians, Axis>::Append<angle::Radians>, StaticDescriptor<angle::Radians, Axis, angle::Radians>>);
  static_assert(!std::is_same_v<StaticDescriptor<Axis, angle::Radians>::Append<angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians>::Append<Polar<Distance, angle::Radians>>, StaticDescriptor<Axis, angle::Radians, Polar<Distance, angle::Radians>>>);
}


TEST(basics, Take)
{
  static_assert(std::is_same_v<StaticDescriptor<>::Take<0>, StaticDescriptor<>>);
  static_assert(std::is_same_v<StaticDescriptor<angle::Radians>::Take<1>, StaticDescriptor<angle::Radians>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(!std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Take<3>, StaticDescriptor<Axis, angle::Radians, Axis, Axis>>);
}


TEST(basics, Drop)
{
  static_assert(std::is_same_v<StaticDescriptor<Axis>::Drop<0>, StaticDescriptor<Axis>>);
  static_assert(std::is_same_v<StaticDescriptor<angle::Radians>::Drop<1>, StaticDescriptor<>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Axis, angle::Radians>::Drop<3>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, angle::Radians, Axis>::Drop<3>, StaticDescriptor<angle::Radians, Axis>>);
}


TEST(basics, Select)
{
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis>::Select<0>, Axis>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis>::Select<1>, angle::Radians>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis>::Select<2>, Axis>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Polar<>, Distance>::Select<3>, Polar<>>);
  static_assert(std::is_same_v<StaticDescriptor<Axis, angle::Radians, Axis, Polar<>, Distance>::Select<4>, Distance>);
  static_assert(std::is_same_v<StaticDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<0>, Dimensions<3>>);
  static_assert(std::is_same_v<StaticDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<1>, Dimensions<2>>);
  static_assert(std::is_same_v<StaticDescriptor<Dimensions<3>, Dimensions<2>, Axis>::Select<2>, Axis>);
}


TEST(basics, replicate_static_vector_space_descriptor)
{
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<angle::Radians, 0>, StaticDescriptor<>>);
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<angle::Radians, 1>, angle::Radians>);
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<angle::Radians, 2>, StaticDescriptor<angle::Radians, angle::Radians>>);
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<StaticDescriptor<angle::Radians, Axis>, 2>, StaticDescriptor<StaticDescriptor<angle::Radians, Axis>, StaticDescriptor<angle::Radians, Axis>>>);
  static_assert(std::is_same_v<replicate_static_vector_space_descriptor_t<Dimensions<3>, 2>, StaticDescriptor<Dimensions<3>, Dimensions<3>>>);
}


TEST(basics, uniform_static_vector_space_descriptor)
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

  static_assert(not uniform_static_vector_space_descriptor<StaticDescriptor<>>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<StaticDescriptor<Axis>>, Axis>);
  static_assert(not uniform_static_vector_space_descriptor<StaticDescriptor<Axis, angle::Radians>>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<StaticDescriptor<Axis, Axis>>, Axis>);
  static_assert(std::is_same_v<uniform_static_vector_space_descriptor_component_of_t<StaticDescriptor<angle::Radians, angle::Radians>>, angle::Radians>);
  static_assert(not uniform_static_vector_space_descriptor<StaticDescriptor<Polar<>, Polar<>>>);
}


TEST(basics, assignment)
{
  static_assert(std::is_assignable_v<Dimensions<10>&, Dimensions<10>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, Dimensions<11>>);
  static_assert(std::is_assignable_v<Polar<>&, Polar<>>);
}


TEST(basics, remove_trailing_1D_descriptors)
{
  using D123 = std::tuple<Dimensions<1>, Dimensions<2>, Dimensions<3>>;
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{})), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<2>{}, Dimensions<3>{}, Dimensions<1>{}, Dimensions<1>{}})), D123>);
  static_assert(std::is_same_v<decltype(internal::remove_trailing_1D_descriptors(std::tuple{Dimensions<1>{}, Dimensions<1>{}, Dimensions<1>{}})), std::tuple<>>);
}


TEST(basics, split_head_tail_fixed)
{
  using namespace internal;

  static_assert(std::is_same_v<split_head_tail_fixed_t<Axis>, std::tuple<Axis, StaticDescriptor<>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<Axis, Distance>>, std::tuple<Axis, Distance>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<Dimensions<2>, Distance, Axis>>, std::tuple<Axis, StaticDescriptor<Axis, Distance, Axis>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<Axis, Polar<Distance, angle::Radians>>>, std::tuple<Axis, Polar<Distance, angle::Radians>>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<Polar<Distance, angle::Radians>, Axis>>, std::tuple<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(std::is_same_v<split_head_tail_fixed_t<StaticDescriptor<StaticDescriptor<Axis, Distance>, Axis>>, std::tuple<Axis, StaticDescriptor<Distance, Axis>>>);
}


TEST(basics, smallest_vector_space_descriptor_fixed)
{
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<3>{}, Dimensions<4>{}) == Dimensions<3>{});
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<3>{}, angle::Radians{}) == angle::Radians{});
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<1>{}, angle::Radians{}) == Dimensions<1>{});
  static_assert(internal::smallest_vector_space_descriptor<double>(angle::Radians{}, Dimensions<1>{}, angle::Degrees{}) == angle::Radians{});
}


TEST(basics, largest_vector_space_descriptor_fixed)
{
  static_assert(internal::largest_vector_space_descriptor<double>(Dimensions<3>{}, Dimensions<4>{}) == Dimensions<4>{});
  static_assert(internal::largest_vector_space_descriptor<double>(Dimensions<3>{}, angle::Radians{}) == Dimensions<3>{});
  static_assert(internal::largest_vector_space_descriptor<double>(Dimensions<2>{}, Spherical<>{}) == Spherical<>{});
  static_assert(internal::largest_vector_space_descriptor<double>(Dimensions<1>{}, angle::Radians{}) == Dimensions<1>{});
  static_assert(internal::largest_vector_space_descriptor<double>(angle::Radians{}, Dimensions<1>{}, angle::Degrees{}) == angle::Radians{});
}


TEST(basics, static_vector_space_descriptor_slice)
{
  using namespace internal;

  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<Dimensions<7>, 0, 7>, Dimensions<7>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<Dimensions<7>, 1, 6>, Dimensions<6>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<Dimensions<7>, 2, 3>, Dimensions<3>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<Dimensions<7>, 2, 0>, Dimensions<0>>);

  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 0, 6>, StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 0, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 5>, StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 4>, StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 2>, StaticDescriptor<Axis, Distance>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 1>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 1, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 4>, StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 3>, StaticDescriptor<Distance, Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 1>, StaticDescriptor<Distance>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 2, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 3>, StaticDescriptor<Polar<Distance, angle::Radians>, Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 2>, StaticDescriptor<Polar<Distance, angle::Radians>>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 3, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 5, 1>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 5, 0>, StaticDescriptor<>>);
  static_assert(equivalent_to<static_vector_space_descriptor_slice_t<StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>, 6, 0>, StaticDescriptor<>>);
}


TEST(basics, get_vector_space_descriptor_slice_fixed)
{
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 7>{}) == Dimensions<7>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 6>{}) == Dimensions<6>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == Dimensions<3>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions<7>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == Dimensions<0>{});

  static_assert(get_vector_space_descriptor_slice<double>(Dimensions<7>{}, 0, 7) == Dimensions<7>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions<7>{}, 1, 6) == Dimensions<6>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions<7>{}, 2, 3) == Dimensions<3>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions<7>{}, 2, 0) == Dimensions<0>{});

  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 6>{}) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 5>{}) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 4>{}) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Axis, Distance>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Axis>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 4>{}) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Distance>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 3>{}) == StaticDescriptor<Polar<Distance, angle::Radians>, Axis>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Polar<Distance, angle::Radians>>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 3>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 1>{}) == StaticDescriptor<Axis>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 5>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});
  static_assert(get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 6>{}, std::integral_constant<std::size_t, 0>{}) == StaticDescriptor<>{});

  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, std::integral_constant<std::size_t, 6>{}) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 0>{}, 6) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, std::integral_constant<std::size_t, 2>{}) == StaticDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, std::integral_constant<std::size_t, 1>{}, 2) == StaticDescriptor<Axis, Distance>{}));

  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 6) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 0, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 5) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 4) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 2) == StaticDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 1, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 4) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 3) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 1) == StaticDescriptor<Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 2, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 3) == StaticDescriptor<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 2) == StaticDescriptor<Polar<Distance, angle::Radians>>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 3, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 5, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}, 6, 0) == StaticDescriptor<>{}));
}


TEST(basics, vector_space_descriptor_collection)
{
  static_assert(internal::collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(internal::collection<std::vector<angle::Radians>>);

  static_assert(vector_space_descriptor_tuple<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(vector_space_descriptor_tuple<std::array<Distance, 5>>);
  static_assert(not vector_space_descriptor_tuple<std::tuple<Axis, Distance, double, angle::Radians>>);
  static_assert(not vector_space_descriptor_tuple<std::vector<angle::Radians>>);
  static_assert(not vector_space_descriptor_tuple<std::initializer_list<angle::Radians>>);

  static_assert(euclidean_vector_space_descriptor_tuple<std::tuple<Axis, Dimensions<3>, int, std::integral_constant<std::size_t, 5>>>);
  static_assert(not euclidean_vector_space_descriptor_tuple<std::tuple<Axis, Distance, angle::Radians>>);

  static_assert(vector_space_descriptor_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(vector_space_descriptor_collection<std::array<Distance, 5>>);
  static_assert(vector_space_descriptor_collection<std::vector<angle::Radians>>);
  static_assert(vector_space_descriptor_collection<std::initializer_list<angle::Radians>>);

  static_assert(euclidean_vector_space_descriptor_collection<std::tuple<Axis, Dimensions<3>, int, std::integral_constant<std::size_t, 5>>>);
  static_assert(euclidean_vector_space_descriptor_collection<std::array<Dimensions<4>, 5>>);
  static_assert(euclidean_vector_space_descriptor_collection<std::vector<Axis>>);
  static_assert(euclidean_vector_space_descriptor_collection<std::initializer_list<int>>);
  static_assert(not euclidean_vector_space_descriptor_collection<std::array<Distance, 5>>);
  static_assert(not euclidean_vector_space_descriptor_collection<std::vector<angle::Radians>>);
  static_assert(not euclidean_vector_space_descriptor_collection<std::initializer_list<Distance>>);

  static_assert(static_vector_space_descriptor_tuple<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(not static_vector_space_descriptor_tuple<std::tuple<Axis, Distance, int, angle::Radians>>);

  static_assert(static_vector_space_descriptor_collection<std::tuple<Axis, Distance, angle::Radians>>);
  static_assert(static_vector_space_descriptor_collection<std::array<Distance, 5>>);
  static_assert(static_vector_space_descriptor_collection<std::vector<angle::Radians>>);
  static_assert(static_vector_space_descriptor_collection<std::initializer_list<angle::Radians>>);
  static_assert(not static_vector_space_descriptor_collection<std::tuple<Axis, int, Distance, angle::Radians>>);
  static_assert(not static_vector_space_descriptor_collection<std::array<Dimensions<dynamic_size>, 5>>);
  static_assert(not static_vector_space_descriptor_collection<std::vector<int>>);
  static_assert(not static_vector_space_descriptor_collection<std::initializer_list<int>>);
}


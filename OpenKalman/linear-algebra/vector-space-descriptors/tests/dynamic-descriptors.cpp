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
 * \brief Tests for \ref dynamic_vector_space_descriptor objects
 */

#include "basics/tests/tests.hpp"
#include "basics/basics.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/composite_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/atomic_static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_component_count_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/index.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/DynamicDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"


using namespace OpenKalman;
using namespace OpenKalman::descriptor;
using numbers::pi;


TEST(descriptors, integral)
{
  static_assert(vector_space_descriptor<unsigned>);
  static_assert(dynamic_vector_space_descriptor<unsigned>);
  static_assert(not static_vector_space_descriptor<unsigned>);
  static_assert(euclidean_vector_space_descriptor<unsigned>);
  static_assert(not composite_vector_space_descriptor<unsigned>);
  static_assert(not atomic_static_vector_space_descriptor<unsigned>);
  static_assert(euclidean_vector_space_descriptor<unsigned>);
  static_assert(dimension_size_of_v<unsigned> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<unsigned> == dynamic_size);
  static_assert(get_dimension_size_of(3u) == 3);
  EXPECT_EQ(get_dimension_size_of(0u), 0);
  EXPECT_EQ(get_dimension_size_of(3u), 3);
  static_assert(get_euclidean_dimension_size_of(3u) == 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(3u), 3);
}


TEST(descriptors, dynamic_Dimensions)
{
  using D = Dimensions<dynamic_size>;

  static_assert(vector_space_descriptor<D>);
  static_assert(dynamic_vector_space_descriptor<D>);
  static_assert(not static_vector_space_descriptor<D>);
  static_assert(euclidean_vector_space_descriptor<D>);
  static_assert(not composite_vector_space_descriptor<D>);
  static_assert(euclidean_vector_space_descriptor<D>);
  static_assert(not atomic_static_vector_space_descriptor<D>);
  static_assert(dimension_size_of_v<D> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<D> == dynamic_size);
  static_assert(get_dimension_size_of(D {0}) == 0);
  static_assert(get_dimension_size_of(D {3}) == 3);
  static_assert(get_dimension_size_of(Dimensions {0}) == 0);
  static_assert(get_dimension_size_of(Dimensions {3}) == 3);
  static_assert(get_euclidean_dimension_size_of(Dimensions{3}) == 3);
  static_assert(get_dimension_size_of(Dimensions<dynamic_size> {Axis {}}) == 1);
  static_assert(get_dimension_size_of(Dimensions<dynamic_size> {Dimensions<3> {}}) == 3);
  static_assert(static_cast<std::size_t>(Dimensions {3}) == 3);
}


TEST(descriptors, DynamicDescriptor_traits)
{
  static_assert(vector_space_descriptor<DynamicDescriptor<float>>);
  static_assert(vector_space_descriptor<DynamicDescriptor<double>>);
  static_assert(dynamic_vector_space_descriptor<DynamicDescriptor<double>>);
  static_assert(dynamic_vector_space_descriptor<DynamicDescriptor<float>>);
  static_assert(dynamic_vector_space_descriptor<DynamicDescriptor<long double>>);
  static_assert(not static_vector_space_descriptor<DynamicDescriptor<double>>);
  static_assert(not euclidean_vector_space_descriptor<DynamicDescriptor<double>>);
  static_assert(composite_vector_space_descriptor<DynamicDescriptor<double>>);
  static_assert(dimension_size_of_v<DynamicDescriptor<double>> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<DynamicDescriptor<float>> == dynamic_size);
}


TEST(descriptors, DynamicDescriptor_construct)
{
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {}), 0);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {Dimensions<0>{}}), 0);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {Axis{}}), 1);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {angle::Degrees{}}), 1);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor<double> {angle::Degrees{}}), 2);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {Dimensions<5>{}}), 5);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {Dimensions{5}}), 5);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {Polar<Distance, angle::Radians>{}}), 2);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {Polar<Distance, angle::PositiveRadians>{}}), 2);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor<double> {Polar<angle::PositiveDegrees, Distance>{}}), 3);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}}), 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor<double> {Spherical<inclination::Radians, Distance, angle::PositiveDegrees>{}}), 4);

  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {Dimensions{5}, Dimensions<1>{}, angle::Degrees{}}), 7);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor<double> {Axis{}, Dimensions{5}, angle::Degrees{}}), 8);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor<double> {StaticDescriptor<Axis, inclination::Radians>{}, angle::Degrees{}, Dimensions{5}}), 8);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor<double> {Dimensions{5}, StaticDescriptor<Axis, inclination::Radians>{}, angle::Degrees{}}), 10);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor<double> {Dimensions{5}, StaticDescriptor<Axis, inclination::Radians>{}, DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);

  DynamicDescriptor<double> d {Dimensions{5}, StaticDescriptor<Axis, inclination::Radians>{}};
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor<double> {d, DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor<double> {std::move(d), DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);

  // Deduction guides:
  static_assert(std::is_same_v<decltype(DynamicDescriptor {std::declval<DynamicDescriptor<long double>>(), Axis{}, angle::Radians{}}), DynamicDescriptor<long double>>);
  static_assert(std::is_same_v<decltype(DynamicDescriptor {std::declval<const DynamicDescriptor<float>>(), Axis{}, angle::Radians{}}), DynamicDescriptor<float>>);
}


TEST(descriptors, DynamicDescriptor_extend)
{
  DynamicDescriptor<double> d;
  EXPECT_EQ(get_dimension_size_of(d), 0); EXPECT_EQ(get_euclidean_dimension_size_of(d), 0); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d), 0);
  d += Axis{};
  EXPECT_EQ(get_dimension_size_of(d), 1); EXPECT_EQ(get_euclidean_dimension_size_of(d), 1); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d), 1);
  (((((d += Dimensions{5}) += Dimensions<5>{}) += angle::Radians{}) += StaticDescriptor<Axis, inclination::Radians>{}) += Polar<angle::Degrees, Distance>{});
  EXPECT_EQ(get_dimension_size_of(d), 16); EXPECT_EQ(get_euclidean_dimension_size_of(d), 19); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d), 11);
}


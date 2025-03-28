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
 * \brief Tests for \ref dynamic_pattern objects
 */

#include "basics/tests/tests.hpp"
#include "basics/basics.hpp"

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
#include "linear-algebra/coordinates/traits/size_of.hpp" //
#include "linear-algebra/coordinates/traits/euclidean_size_of.hpp" //
#include "linear-algebra/coordinates/traits/component_count_of.hpp" //

#include "linear-algebra/coordinates/functions/get_size.hpp" //
#include "linear-algebra/coordinates/functions/get_euclidean_size.hpp" //
#include "linear-algebra/coordinates/functions/get_component_count.hpp" //
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp" //

#include "linear-algebra/coordinates/functions/to_euclidean_element.hpp" //
#include "linear-algebra/coordinates/functions/from_euclidean_element.hpp" //
#include "linear-algebra/coordinates/functions/get_wrapped_component.hpp" //
#include "linear-algebra/coordinates/functions/set_wrapped_component.hpp" //

// descriptors:

#include "linear-algebra/coordinates/descriptors/Dimensions.hpp" //
#include "linear-algebra/coordinates/descriptors/StaticDescriptor.hpp" //
#include "linear-algebra/coordinates/descriptors/DynamicDescriptor.hpp" //

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

using namespace OpenKalman;
using namespace OpenKalman::coordinate;
using numbers::pi;


TEST(coordinates, internal_is_uniform_component_of)
{
  using namespace internal;

  // fixed:
  static_assert(is_uniform_component_of(Axis {}, Axis {}));
  static_assert(is_uniform_component_of(Axis {}, Dimensions<10> {}));
  static_assert(not is_uniform_component_of(Dimensions<2> {}, Dimensions<10> {}));
  static_assert(not is_uniform_component_of(Axis {}, std::tuple<Dimensions<10>, Distance> {}));
  static_assert(is_uniform_component_of(Distance {}, std::tuple<Distance, Distance, Distance, Distance> {}));
  static_assert(is_uniform_component_of(angle::Radians {}, std::tuple<angle::Radians, angle::Radians, angle::Radians, angle::Radians> {}));
  static_assert(not is_uniform_component_of(Polar<> {}, std::array<Polar<>, 4> {}));

  // dynamic:
  static_assert(is_uniform_component_of(1, Dimensions<10> {}));
  static_assert(is_uniform_component_of(Dimensions<1> {}, 10));
  static_assert(is_uniform_component_of(1, 10));
  static_assert(not is_uniform_component_of(2, 10));
  EXPECT_TRUE(is_uniform_component_of(Axis {}, DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE(is_uniform_component_of(Axis {}, DynamicDescriptor<double> {Dimensions<10> {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor<double> {Distance {}}, std::array<Distance, 4> {}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor<double> {Axis {}}, DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor<double> {Axis {}}, DynamicDescriptor<double> {Dimensions<10> {}}));
  EXPECT_FALSE(is_uniform_component_of(DynamicDescriptor<double> {Dimensions<2> {}}, DynamicDescriptor<double> {Dimensions<10> {}}));
  EXPECT_FALSE(is_uniform_component_of(DynamicDescriptor<double> {Axis {}}, DynamicDescriptor<double> {Dimensions<2> {}, Distance {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor<double> {Distance {}}, DynamicDescriptor<double> {Distance {}, Distance {}, Distance {}, Distance {}}));

  auto d1 = DynamicDescriptor<double> {Axis {}};
  auto f2 = DynamicDescriptor<float> {Dimensions<2> {}};
  static_assert(not is_uniform_component_of(d1, f2));
  auto a10 = DynamicDescriptor<double> {Dimensions<10> {}};
  static_assert(not is_uniform_component_of(Dimensions<2> {}, a10));
  auto a2 = DynamicDescriptor<double> {Axis {}, Axis {}};
  EXPECT_TRUE(is_uniform_component_of(d1, a2));
}


TEST(coordinates, smallest_vector_space_descriptor_dynamic)
{
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<3>{}, 4) == 3);
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<4>{}, 3) == 3);
  static_assert(internal::smallest_vector_space_descriptor<double>(3, 4, 5) == 3);

  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}}) == 3));
  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}}) == 3));
  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{angle::Radians{}, Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}, angle::Degrees{}}) == DynamicDescriptor<double>{angle::Radians{}, Dimensions<3>{}}));
}


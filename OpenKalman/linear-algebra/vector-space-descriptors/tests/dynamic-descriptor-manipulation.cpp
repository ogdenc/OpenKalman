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

#include "linear-algebra/vector-space-descriptors/concepts/composite_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/atomic_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/get_size.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_size.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_component_count.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_is_euclidean.hpp" //

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

#include "linear-algebra/vector-space-descriptors/traits/static_concatenate.hpp" //

#include "linear-algebra/vector-space-descriptors/traits/internal/uniform_static_vector_space_descriptor_query.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/uniform_static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/uniform_static_vector_space_descriptor_component_of.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to_uniform_static_vector_space_descriptor_component_of.hpp" //

// functions:

#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"

#include "linear-algebra/vector-space-descriptors/functions/internal/is_uniform_component_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/remove_trailing_1D_descriptors.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/best_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/smallest_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/largest_vector_space_descriptor.hpp"

#include "linear-algebra/vector-space-descriptors/functions/get_slice.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/internal/to_euclidean_vector_space_descriptor_collection.hpp"

using namespace OpenKalman;
using namespace OpenKalman::descriptor;
using numbers::pi;


TEST(descriptors, internal_is_uniform_component_of)
{
  using namespace internal;

  // fixed:
  static_assert(is_uniform_component_of(Axis {}, Axis {}));
  static_assert(is_uniform_component_of(Axis {}, Dimensions<10> {}));
  static_assert(not is_uniform_component_of(Dimensions<2> {}, Dimensions<10> {}));
  static_assert(not is_uniform_component_of(Axis {}, StaticDescriptor<Dimensions<10>, Distance> {}));
  static_assert(is_uniform_component_of(Distance {}, StaticDescriptor<Distance, Distance, Distance, Distance> {}));
  static_assert(is_uniform_component_of(angle::Radians {}, StaticDescriptor<angle::Radians, angle::Radians, angle::Radians, angle::Radians> {}));
  static_assert(not is_uniform_component_of(Polar<> {}, StaticDescriptor<Polar<>, Polar<>, Polar<>, Polar<>> {}));

  // dynamic:
  static_assert(is_uniform_component_of(1, Dimensions<10> {}));
  static_assert(is_uniform_component_of(Dimensions<1> {}, 10));
  static_assert(is_uniform_component_of(1, 10));
  static_assert(not is_uniform_component_of(2, 10));
  EXPECT_TRUE(is_uniform_component_of(Axis {}, DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE(is_uniform_component_of(Axis {}, DynamicDescriptor<double> {Dimensions<10> {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor<double> {Distance {}}, StaticDescriptor<Distance, Distance, Distance, Distance> {}));
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


TEST(descriptors, smallest_vector_space_descriptor_dynamic)
{
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<3>{}, 4) == 3);
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<4>{}, 3) == 3);
  static_assert(internal::smallest_vector_space_descriptor<double>(3, 4, 5) == 3);

  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}}) == 3));
  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}}) == 3));
  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{angle::Radians{}, Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}, angle::Degrees{}}) == DynamicDescriptor<double>{angle::Radians{}, Dimensions<3>{}}));
}


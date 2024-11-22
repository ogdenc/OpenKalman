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
#include "linear-algebra/vector-space-descriptors/concepts/atomic_static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/dimension_difference_of.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_component_count_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_is_euclidean.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/to_euclidean_element.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/from_euclidean_element.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_wrapped_component.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/set_wrapped_component.hpp" //

// internal traits:

#include "linear-algebra/vector-space-descriptors/traits/internal/prefix_base_of.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/internal/prefix_of.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/internal/suffix_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/internal/suffix_base_of.hpp" //

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

#include "linear-algebra/vector-space-descriptors/traits/replicate_static_vector_space_descriptor.hpp" //

#include "linear-algebra/vector-space-descriptors/traits/internal/static_canonical_form.hpp"
#include "linear-algebra/vector-space-descriptors/traits/static_concatenate.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/static_reverse.hpp"

#include "linear-algebra/vector-space-descriptors/traits/internal/uniform_static_vector_space_descriptor_query.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/uniform_static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/uniform_static_vector_space_descriptor_component_of.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to_uniform_static_vector_space_descriptor_component_of.hpp" //

// collection traits

#include "linear-algebra/vector-space-descriptors/traits/internal/vector_space_descriptor_collection_common_type.hpp" //

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

using namespace OpenKalman;
using namespace OpenKalman::descriptor;
using numbers::pi;


TEST(basics, dynamic_comparison)
{
  static_assert(Dimensions{0} == Dimensions{0});
  static_assert(Dimensions{3} == Dimensions{3});
  static_assert(Dimensions{3} == Dimensions{3});
  static_assert(Dimensions{3} <= Dimensions{3});
  static_assert(Dimensions{3} >= Dimensions{3});
  static_assert(Dimensions{3} != Dimensions{4});
  static_assert(Dimensions{3} < Dimensions{4});
  static_assert(Dimensions{3} <= Dimensions{4});
  static_assert(Dimensions{4} > Dimensions{3});
  static_assert(Dimensions{4} >= Dimensions{3});

  static_assert(Dimensions{3} == Dimensions<3>{});
  static_assert(Dimensions{3} <= Dimensions<3>{});
  static_assert(Dimensions{3} >= Dimensions<3>{});
  static_assert(Dimensions{3} != Dimensions<4>{});
  static_assert(Dimensions{3} < Dimensions<4>{});
  static_assert(Dimensions{3} <= Dimensions<4>{});
  static_assert(Dimensions{4} > Dimensions<3>{});
  static_assert(Dimensions{4} >= Dimensions<3>{});

  static_assert(Dimensions{3} == StaticDescriptor<Axis, Axis, Axis>{});
  static_assert(Dimensions{4} == StaticDescriptor<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} <= StaticDescriptor<Axis, Dimensions<2>>{});
  static_assert(Dimensions{3} < StaticDescriptor<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} != StaticDescriptor<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} <= StaticDescriptor<Axis, Axis, Axis, Axis>{});

  static_assert(StaticDescriptor<Axis, Axis, Axis>{} == Dimensions{3});
  static_assert(StaticDescriptor<Axis, Dimensions<2>, Axis>{} == Dimensions{4});
  static_assert(StaticDescriptor<Axis, Dimensions<2>>{} >= Dimensions{3});
  static_assert(StaticDescriptor<Axis, Dimensions<2>, Axis>{} > Dimensions{3});
  static_assert(StaticDescriptor<Axis, Dimensions<2>, Axis>{} != Dimensions{3});
  static_assert(StaticDescriptor<Axis, Axis, Axis, Axis>{} >= Dimensions{3});

  static_assert(Polar<Distance, angle::Radians>{} != Dimensions{5});
  static_assert(not (Polar<Distance, angle::Radians>{} < Dimensions{5}));
  static_assert(Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions{5});
  static_assert(not (Spherical<Distance, inclination::Radians, angle::Radians>{} > Dimensions{5}));

  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<> {}} == DynamicDescriptor<double> {StaticDescriptor<> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<> {}} == DynamicDescriptor<double> {}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} == DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} <= DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} != DynamicDescriptor<double> {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Degrees {}} != DynamicDescriptor<double> {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} != DynamicDescriptor<double> {Polar<> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} == DynamicDescriptor<double> {StaticDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} < DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis> {}} == DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}} > DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis> {}} == DynamicDescriptor<double> {StaticDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}} >= DynamicDescriptor<double> {StaticDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}} == DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}} < DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis, Axis> {}} > DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}} == DynamicDescriptor<double> {Axis {}, angle::Radians {}, Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}} == DynamicDescriptor<double> {Axis {}, angle::Radians {}, StaticDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}} == DynamicDescriptor<double> {Axis {}, angle::Radians {}, StaticDescriptor<StaticDescriptor<Axis>> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5> {}} == DynamicDescriptor<double> {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<StaticDescriptor<Axis>, angle::Radians, StaticDescriptor<Axis>> {}} == DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<4> {}} < DynamicDescriptor<double> {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5> {}} > DynamicDescriptor<double> {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<2>{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}} == DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}, Axis{}} > DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Spherical<Distance, angle::Radians, inclination::Radians> {}} == DynamicDescriptor<double> {Spherical<Distance, angle::Radians, inclination::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, angle::Radians> {}} != DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians> {}} != DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}}));

  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} == Dimensions<1>{}));
  EXPECT_TRUE((Dimensions<1>{} == DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis> {}} == Dimensions<1>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis, Axis> {}} == Dimensions<3>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}} < Dimensions<3>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<4>{}} > StaticDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<4>{}} != StaticDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}, angle::Degrees{}} > Dimensions<3>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}} < StaticDescriptor<Dimensions<3>, angle::Degrees>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}} == StaticDescriptor<Dimensions<3>, angle::Degrees, Dimensions<5>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} == StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} == StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} < StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<4>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} > StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<2>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<3>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} != StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE(not (DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} < StaticDescriptor<Dimensions<4>, angle::Degrees, Dimensions<3>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} == Dimensions{1}));
  EXPECT_TRUE((Dimensions{1} == DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis> {}} == Dimensions{1}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis, Axis> {}} == Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}} < Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions{4}} > StaticDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((Dimensions{4} != DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}}));

  EXPECT_TRUE((StaticDescriptor<> {} == DynamicDescriptor<double> {StaticDescriptor<> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians>{}} == StaticDescriptor<Axis, angle::Radians>{}));
  EXPECT_TRUE((StaticDescriptor<Axis, angle::Radians>{} <= DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians>{}}));
  EXPECT_TRUE((StaticDescriptor<Axis, angle::Radians>{} <= DynamicDescriptor<double> {Axis{}, angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians>{}} >= StaticDescriptor<Axis, angle::Radians>{}));
  EXPECT_TRUE((StaticDescriptor<Axis, angle::Radians>{} < DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis>{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians>{}} <= StaticDescriptor<Axis, angle::Radians, Axis>{}));
  EXPECT_TRUE((angle::Radians{} == DynamicDescriptor<double> {angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {inclination::Radians{}} == inclination::Radians{}));
  EXPECT_TRUE((angle::Radians{} != DynamicDescriptor<double> {inclination::Radians{}}));
  EXPECT_FALSE((angle::Radians{} < inclination::Radians{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_FALSE((Polar<Distance, angle::Radians>{} < DynamicDescriptor<double> {Dimensions<5>{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Spherical<Distance, inclination::Radians, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_FALSE((Spherical<Distance, inclination::Radians, angle::Radians>{} < DynamicDescriptor<double> {Dimensions<5>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Distance> {}} == DynamicDescriptor<float> {StaticDescriptor<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Distance> {}} == DynamicDescriptor<long double> {StaticDescriptor<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Distance> {}} < DynamicDescriptor<float> {StaticDescriptor<Axis, angle::Radians, Distance, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Distance, Axis> {}} > DynamicDescriptor<float> {StaticDescriptor<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Distance> {}} <= DynamicDescriptor<float> {StaticDescriptor<Axis, angle::Radians, Distance, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Distance, Axis> {}} >= DynamicDescriptor<float> {StaticDescriptor<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Distance, Axis> {}} != DynamicDescriptor<float> {StaticDescriptor<Axis, angle::Radians, Distance> {}}));
}


TEST(basics, dynamic_assignment)
{
  static_assert(std::is_assignable_v<std::size_t&, Dimensions<10>>);
  static_assert(std::is_assignable_v<int&, Dimensions<10>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, std::size_t>);
  static_assert(std::is_assignable_v<Dimensions<dynamic_size>&, Dimensions<11>>);
  static_assert(std::is_assignable_v<Dimensions<dynamic_size>&, DynamicDescriptor<double>>);
  static_assert(std::is_assignable_v<DynamicDescriptor<double>&, Dimensions<dynamic_size>>);
  static_assert(std::is_assignable_v<DynamicDescriptor<double>&, Polar<>>);

  static_assert(std::is_assignable_v<Polar<>&, Polar<>>);

  Dimensions<dynamic_size> dim {5};
  EXPECT_EQ(dim, 5);
  dim = 6;
  EXPECT_EQ(dim, 6);
  dim = Dimensions<7>{};
  EXPECT_EQ(dim, 7);
  dim = DynamicDescriptor<double> {Dimensions<3>{}, Dimensions<5>{}};
  EXPECT_EQ(dim, 8);
  EXPECT_ANY_THROW((dim = DynamicDescriptor<double> {Dimensions<3>{}, Polar<>{}}));
  EXPECT_EQ(dim, 8);

  DynamicDescriptor<double> dyn;
  dyn = 5;
  EXPECT_EQ(dyn, 5);
  dyn = Dimensions<6>{};
  EXPECT_EQ(dyn, 6);
  dyn = Polar<>{};
  EXPECT_TRUE(dyn == Polar<>{});
}


TEST(basics, dynamic_arithmetic)
{
  EXPECT_TRUE(Dimensions{3} + Dimensions{4} == Dimensions{7});
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}} + DynamicDescriptor<double> {angle::Degrees{}, Axis{}} == DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}} + StaticDescriptor<angle::Degrees, Axis>{} == DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((StaticDescriptor<Axis, angle::Radians>{} + DynamicDescriptor<double> {angle::Degrees{}, Axis{}} == DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));

  EXPECT_TRUE(Dimensions{7} - Dimensions{7} == Dimensions{0});
  EXPECT_TRUE(Dimensions{7} - Dimensions{4} == Dimensions{3});

  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - StaticDescriptor<>{} == DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - StaticDescriptor<Axis>{} == DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - StaticDescriptor<angle::Degrees, Axis>{} == DynamicDescriptor<double> {Axis{}, angle::Radians{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions{7}} - Dimensions{4} == Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - Dimensions{1} == DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions{7}} - DynamicDescriptor<double> {Dimensions{4}} == Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - DynamicDescriptor<double> {Axis{}} == DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - DynamicDescriptor<double> {angle::Degrees{}, Axis{}} == DynamicDescriptor<double> {Axis{}, angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - DynamicDescriptor<double> {angle::Degrees{}, Axis{}} == StaticDescriptor<Axis, angle::Radians>{}));
}


TEST(basics, internal_replicate_vector_space_descriptor)
{
  using namespace internal;

  // fixed:
  static_assert(std::is_same_v<decltype(replicate_vector_space_descriptor<double>(StaticDescriptor<angle::Radians, Axis> {}, std::integral_constant<std::size_t, 2> {})), StaticDescriptor<StaticDescriptor<angle::Radians, Axis>, StaticDescriptor<angle::Radians, Axis>>>);

  // dynamic:
  auto d1 = replicate_vector_space_descriptor<double>(4, 3);
  EXPECT_EQ(get_dimension_size_of(d1), 12); EXPECT_EQ(get_euclidean_dimension_size_of(d1), 12); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d1), 12);
  auto d2 = replicate_vector_space_descriptor<double>(angle::Radians{}, 4);
  EXPECT_EQ(get_dimension_size_of(d2), 4); EXPECT_EQ(get_euclidean_dimension_size_of(d2), 8); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d2), 4);
  auto d3 = replicate_vector_space_descriptor<double>(Polar<Distance, angle::Radians>{}, 2);
  EXPECT_EQ(get_dimension_size_of(d3), 4); EXPECT_EQ(get_euclidean_dimension_size_of(d3), 6); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d3), 2);
}


TEST(basics, internal_is_uniform_component_of)
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


TEST(basics, smallest_vector_space_descriptor_dynamic)
{
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<3>{}, 4) == 3);
  static_assert(internal::smallest_vector_space_descriptor<double>(Dimensions<4>{}, 3) == 3);
  static_assert(internal::smallest_vector_space_descriptor<double>(3, 4, 5) == 3);

  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}}) == 3));
  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}}) == 3));
  EXPECT_TRUE((internal::smallest_vector_space_descriptor<double>(DynamicDescriptor<double>{angle::Radians{}, Dimensions<3>{}}, DynamicDescriptor<double>{Dimensions<4>{}, angle::Degrees{}}) == DynamicDescriptor<double>{angle::Radians{}, Dimensions<3>{}}));
}


TEST(basics, slice_vector_space_descriptor_dynamic)
{
  using namespace internal;

  EXPECT_TRUE((DynamicDescriptor<double> {}.slice(0, 0) == DynamicDescriptor<double>{}));

  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}.slice(0, 1) == Axis{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}.slice(1, 1) == DynamicDescriptor<double> {angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}.slice(1, 2) == DynamicDescriptor<double> {angle::Radians{}, Distance{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}.slice(0, 3) == DynamicDescriptor<double> {Axis{}, angle::Radians{}, Distance{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 0) == DynamicDescriptor<double>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 1) == angle::Radians{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 2) == DynamicDescriptor<double> {angle::Radians{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 3) == DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}}));
  EXPECT_ANY_THROW((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 4)));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 5) == DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 0) == DynamicDescriptor<double>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 1) == Axis{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 2) == DynamicDescriptor<double> {Axis{}, Distance{}}));
  EXPECT_ANY_THROW((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 3)));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 4) == DynamicDescriptor<double> {Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 0) == DynamicDescriptor<double>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 1) == Distance{}));
  EXPECT_ANY_THROW((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 2)));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 3) == DynamicDescriptor<double> {Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 0) == DynamicDescriptor<double> {}));
  EXPECT_ANY_THROW((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 1)));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 2) == Polar<Distance, angle::Radians>{}));

  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians>{}, Axis{}}.slice(0, 2) == Polar<Distance, angle::Radians>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians>{}, Axis{}}.slice(2, 1) == Axis{}));
}


TEST(basics, get_vector_space_descriptor_slice_dynamic)
{
  using namespace internal;

  static_assert(get_vector_space_descriptor_slice<double>(Dimensions{7}, 0, 7) == Dimensions<7>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions{7}, 1, 6) == Dimensions<6>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions{7}, 2, 3) == Dimensions<3>{});
  static_assert(get_vector_space_descriptor_slice<double>(Dimensions{7}, 2, 0) == Dimensions<0>{});

  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 6) == StaticDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 7)));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, -1, 7)));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 5) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 4) == StaticDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 3)));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 2) == StaticDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 4) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 5)));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 3) == StaticDescriptor<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 2)));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 1) == StaticDescriptor<Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 3) == StaticDescriptor<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 2) == StaticDescriptor<Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 1)));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 5, 1) == StaticDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 5, 0) == StaticDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 6, 0) == StaticDescriptor<>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice<double>(DynamicDescriptor<double> {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 7, -1)));
}


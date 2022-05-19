/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for coefficient types
 */

#include <gtest/gtest.h>
#include "basics/basics.hpp"

using namespace OpenKalman;
using std::numbers::pi;


TEST(index_descriptors, integral)
{
  static_assert(index_descriptor<int>);
  static_assert(dynamic_index_descriptor<int>);
  static_assert(not fixed_index_descriptor<int>);
  static_assert(euclidean_index_descriptor<int>);
  static_assert(not composite_index_descriptor<int>);
  static_assert(not atomic_fixed_index_descriptor<int>);
  static_assert(euclidean_index_descriptor<int>);
  static_assert(not typed_index_descriptor<int>);
  static_assert(dimension_size_of_v<int> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<int> == dynamic_size);
  static_assert(get_dimension_size_of(3) == 3);
  EXPECT_EQ(get_dimension_size_of(3), 3);
  static_assert(get_euclidean_dimension_size_of(3) == 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(3), 3);
}


TEST(index_descriptors, dynamic_Dimensions)
{
  using D = Dimensions<dynamic_size>;

  static_assert(index_descriptor<D>);
  static_assert(dynamic_index_descriptor<D>);
  static_assert(not fixed_index_descriptor<D>);
  static_assert(euclidean_index_descriptor<D>);
  static_assert(not composite_index_descriptor<D>);
  static_assert(euclidean_index_descriptor<D>);
  static_assert(not typed_index_descriptor<D>);
  static_assert(not atomic_fixed_index_descriptor<D>);
  static_assert(dimension_size_of_v<D> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<D> == dynamic_size);
  EXPECT_EQ(get_dimension_size_of(Dimensions{3}), 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(Dimensions{3}), 3);
}


TEST(index_descriptors, DynamicCoefficients_traits)
{
  using D = DynamicTypedIndex<double>;

  static_assert(index_descriptor<D>);
  static_assert(dynamic_index_descriptor<D>);
  static_assert(not fixed_index_descriptor<D>);
  static_assert(not euclidean_index_descriptor<D>);
  static_assert(not typed_index_descriptor<D>);
  static_assert(composite_index_descriptor<D>);
  static_assert(dimension_size_of_v<D> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<D> == dynamic_size);
}


TEST(index_descriptors, DynamicCoefficients_construct)
{
  using D = DynamicTypedIndex<double>;

  EXPECT_EQ(get_dimension_size_of(D {Axis{}}), 1);
  EXPECT_EQ(get_dimension_size_of(D {angle::Degrees{}}), 1);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {angle::Degrees{}}), 2);
  EXPECT_EQ(get_dimension_size_of(D {Dimensions<5>{}}), 5);
  EXPECT_EQ(get_dimension_size_of(D {Dimensions{5}}), 5);
  EXPECT_EQ(get_dimension_size_of(D {Polar<Distance, angle::PositiveRadians>{}}), 2);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Polar<angle::PositiveDegrees, Distance>{}}), 3);
  EXPECT_EQ(get_dimension_size_of(D {Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}}), 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Spherical<inclination::Radians, Distance, angle::PositiveDegrees>{}}), 4);

  EXPECT_EQ(get_dimension_size_of(D {Dimensions{5}, Dimensions<1>{}, angle::Degrees{}}), 7);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Axis{}, Dimensions{5}, angle::Degrees{}}), 8);
  EXPECT_EQ(get_dimension_size_of(D {TypedIndex<Axis, inclination::Radians>{}, angle::Degrees{}, Dimensions{5}}), 8);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Dimensions{5}, TypedIndex<Axis, inclination::Radians>{}, angle::Degrees{}}), 10);
  EXPECT_EQ(get_euclidean_dimension_size_of(D {Dimensions{5}, TypedIndex<Axis, inclination::Radians>{}, D {TypedIndex<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);
}


TEST(index_descriptors, DynamicCoefficients_extend)
{
  using D = DynamicTypedIndex<double>;

  D d;
  EXPECT_EQ(get_dimension_size_of(d), 0); EXPECT_EQ(get_euclidean_dimension_size_of(d), 0); EXPECT_EQ(get_index_descriptor_component_count_of(d), 0);
  d.extend(Axis{});
  EXPECT_EQ(get_dimension_size_of(d), 1); EXPECT_EQ(get_euclidean_dimension_size_of(d), 1); EXPECT_EQ(get_index_descriptor_component_count_of(d), 1);
  d.extend(Dimensions{5}, Dimensions<5>{}, angle::Radians{}, TypedIndex<Axis, inclination::Radians>{}, Polar<angle::Degrees, Distance>{});
  EXPECT_EQ(get_dimension_size_of(d), 16); EXPECT_EQ(get_euclidean_dimension_size_of(d), 19); EXPECT_EQ(get_index_descriptor_component_count_of(d), 15);
}


TEST(index_descriptors, dynamic_comparison)
{
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

  static_assert(Dimensions{3} == TypedIndex<Axis, Axis, Axis>{});
  static_assert(Dimensions{3} <= TypedIndex<Axis, Axis, Axis>{});
  static_assert(Dimensions{3} < TypedIndex<Axis, Axis, Axis, Axis>{});
  static_assert(Dimensions{3} <= TypedIndex<Axis, Axis, Axis, Axis>{});

  EXPECT_TRUE((Polar<Distance, angle::Radians>{} != Dimensions{5}));
  EXPECT_TRUE((Polar<Distance, angle::Radians>{} < Dimensions{5}));
  EXPECT_TRUE((Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions{5}));
  EXPECT_TRUE((Spherical<Distance, inclination::Radians, angle::Radians>{} < Dimensions{5}));

  using D = DynamicTypedIndex<double>;

  EXPECT_TRUE((D {TypedIndex<> {}} == D {TypedIndex<> {}}));
  EXPECT_TRUE((D {TypedIndex<> {}} == D {}));
  EXPECT_TRUE((D {Axis {}} == D {Axis {}}));
  EXPECT_TRUE((D {Axis {}} <= D {Axis {}}));
  EXPECT_TRUE((D {Axis {}} != D {angle::Radians {}}));
  EXPECT_TRUE((D {angle::Degrees {}} != D {angle::Radians {}}));
  EXPECT_TRUE((D {Axis {}} != D {Polar<> {}}));
  EXPECT_TRUE((D {Axis {}} == D {TypedIndex<Axis> {}}));
  EXPECT_TRUE((D {Axis {}} < D {TypedIndex<Axis, Axis> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis> {}} == D {Axis {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, Axis> {}} > D {Axis {}}));
  EXPECT_TRUE((D {TypedIndex<Axis> {}} == D {TypedIndex<Axis> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, Axis> {}} >= D {TypedIndex<Axis> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians, Axis> {}} == D {TypedIndex<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians, Axis> {}} < D {TypedIndex<Axis, Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians, Axis> {}} == D {Axis {}, angle::Radians {}, Axis {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians, Axis> {}} == D {Axis {}, angle::Radians {}, TypedIndex<Axis> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians, Axis> {}} == D {Axis {}, angle::Radians {}, TypedIndex<TypedIndex<Axis>> {}}));
  EXPECT_TRUE((D {TypedIndex<TypedIndex<Axis>, angle::Radians, TypedIndex<Axis>> {}} == D {TypedIndex<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((D {Polar<Distance, angle::Radians> {}} == D {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((D {Polar<Distance, angle::Radians> {}, Axis{}} > D {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((D {Spherical<Distance, angle::Radians, inclination::Radians> {}} == D {Spherical<Distance, angle::Radians, inclination::Radians> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians, angle::Radians> {}} != D {TypedIndex<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians> {}} != D {Polar<Distance, angle::Radians> {}}));

  EXPECT_TRUE((D {Axis {}} == Dimensions<1>{}));
  EXPECT_TRUE((Dimensions<1>{} == D {Axis {}}));
  EXPECT_TRUE((D {TypedIndex<Axis> {}} == Dimensions<1>{}));
  EXPECT_TRUE((D {TypedIndex<Axis, Axis, Axis> {}} == Dimensions<3>{}));
  EXPECT_TRUE((D {TypedIndex<Axis, Axis> {}} < Dimensions<3>{}));
  EXPECT_TRUE((D {Dimensions<4>{}} > TypedIndex<Axis, Axis> {}));
  EXPECT_TRUE((D {Dimensions<4>{}} != TypedIndex<Axis, Axis> {}));
  EXPECT_TRUE((D {angle::Degrees{}} < Dimensions<3>{}));
  EXPECT_TRUE((D {Dimensions<3>{}} > angle::Degrees{}));

  EXPECT_TRUE((D {Axis {}} == Dimensions{1}));
  EXPECT_TRUE((Dimensions{1} == D {Axis {}}));
  EXPECT_TRUE((D {TypedIndex<Axis> {}} == Dimensions{1}));
  EXPECT_TRUE((D {TypedIndex<Axis, Axis, Axis> {}} == Dimensions{3}));
  EXPECT_TRUE((D {TypedIndex<Axis, Axis> {}} < Dimensions{3}));
  EXPECT_TRUE((D {Dimensions{4}} > TypedIndex<Axis, Axis> {}));
  EXPECT_TRUE((Dimensions{4} != D {TypedIndex<Axis, Axis> {}}));
  EXPECT_TRUE((D {angle::Degrees{}} < Dimensions{3}));

  EXPECT_TRUE((TypedIndex<> {} == D {TypedIndex<> {}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians>{}} == TypedIndex<Axis, angle::Radians>{}));
  EXPECT_TRUE((TypedIndex<Axis, angle::Radians>{} <= D {TypedIndex<Axis, angle::Radians>{}}));
  EXPECT_TRUE((TypedIndex<Axis, angle::Radians>{} <= D {Axis{}, angle::Radians{}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians>{}} >= TypedIndex<Axis, angle::Radians>{}));
  EXPECT_TRUE((TypedIndex<Axis, angle::Radians>{} < D {TypedIndex<Axis, angle::Radians, Axis>{}}));
  EXPECT_TRUE((D {TypedIndex<Axis, angle::Radians>{}} <= TypedIndex<Axis, angle::Radians, Axis>{}));
  EXPECT_TRUE((angle::Radians{} == D {angle::Radians{}}));
  EXPECT_TRUE((D {inclination::Radians{}} == inclination::Radians{}));
  EXPECT_TRUE((angle::Radians{} != D {inclination::Radians{}}));
  EXPECT_FALSE((angle::Radians{} < inclination::Radians{}));
  EXPECT_TRUE((D {Polar<Distance, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_TRUE((Polar<Distance, angle::Radians>{} < D {Dimensions<5>{}}));
  EXPECT_TRUE((D {Spherical<Distance, inclination::Radians, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_TRUE((Spherical<Distance, inclination::Radians, angle::Radians>{} < D {Dimensions<5>{}}));
}


TEST(index_descriptors, dynamic_arithmetic)
{
  EXPECT_TRUE(Dimensions{3} + Dimensions{4} == Dimensions{7});
  EXPECT_TRUE(Dimensions{7} - Dimensions{4} == Dimensions{3});
}

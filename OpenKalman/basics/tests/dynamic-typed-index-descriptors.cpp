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
using numbers::pi;


TEST(index_descriptors, integral)
{
  static_assert(index_descriptor<int>);
  static_assert(dynamic_index_descriptor<int>);
  static_assert(not fixed_index_descriptor<int>);
  static_assert(euclidean_index_descriptor<int>);
  static_assert(not composite_index_descriptor<int>);
  static_assert(not atomic_fixed_index_descriptor<int>);
  static_assert(euclidean_index_descriptor<int>);
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
  static_assert(not atomic_fixed_index_descriptor<D>);
  static_assert(dimension_size_of_v<D> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<D> == dynamic_size);
  static_assert(get_dimension_size_of(Dimensions {3}) == 3);
  static_assert(get_euclidean_dimension_size_of(Dimensions{3}) == 3);
  static_assert(get_dimension_size_of(Dimensions<dynamic_size> {Axis {}}) == 1);
  static_assert(get_dimension_size_of(Dimensions<dynamic_size> {Dimensions<3> {}}) == 3);
  static_assert(static_cast<std::size_t>(Dimensions {3}) == 3);
}


TEST(index_descriptors, DynamicTypedIndex_traits)
{
  static_assert(index_descriptor<DynamicTypedIndex<>>);
  static_assert(index_descriptor<DynamicTypedIndex<double>>);
  static_assert(dynamic_index_descriptor<DynamicTypedIndex<>>);
  static_assert(dynamic_index_descriptor<DynamicTypedIndex<float>>);
  static_assert(dynamic_index_descriptor<DynamicTypedIndex<float, long double>>);
  static_assert(not fixed_index_descriptor<DynamicTypedIndex<>>);
  static_assert(not euclidean_index_descriptor<DynamicTypedIndex<>>);
  static_assert(composite_index_descriptor<DynamicTypedIndex<>>);
  static_assert(composite_index_descriptor<DynamicTypedIndex<double, long double>>);
  static_assert(dimension_size_of_v<DynamicTypedIndex<>> == dynamic_size);
  static_assert(dimension_size_of_v<DynamicTypedIndex<double>> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<DynamicTypedIndex<>> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<DynamicTypedIndex<float, long double>> == dynamic_size);
}


TEST(index_descriptors, DynamicTypedIndex_construct)
{
  EXPECT_EQ(get_dimension_size_of(DynamicTypedIndex {Axis{}}), 1);
  EXPECT_EQ(get_dimension_size_of(DynamicTypedIndex {angle::Degrees{}}), 1);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicTypedIndex {angle::Degrees{}}), 2);
  EXPECT_EQ(get_dimension_size_of(DynamicTypedIndex {Dimensions<5>{}}), 5);
  EXPECT_EQ(get_dimension_size_of(DynamicTypedIndex {Dimensions{5}}), 5);
  EXPECT_EQ(get_dimension_size_of(DynamicTypedIndex {Polar<Distance, angle::PositiveRadians>{}}), 2);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicTypedIndex {Polar<angle::PositiveDegrees, Distance>{}}), 3);
  EXPECT_EQ(get_dimension_size_of(DynamicTypedIndex {Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}}), 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicTypedIndex {Spherical<inclination::Radians, Distance, angle::PositiveDegrees>{}}), 4);

  EXPECT_EQ(get_dimension_size_of(DynamicTypedIndex {Dimensions{5}, Dimensions<1>{}, angle::Degrees{}}), 7);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicTypedIndex {Axis{}, Dimensions{5}, angle::Degrees{}}), 8);
  EXPECT_EQ(get_dimension_size_of(DynamicTypedIndex {TypedIndex<Axis, inclination::Radians>{}, angle::Degrees{}, Dimensions{5}}), 8);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicTypedIndex {Dimensions{5}, TypedIndex<Axis, inclination::Radians>{}, angle::Degrees{}}), 10);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicTypedIndex {Dimensions{5}, TypedIndex<Axis, inclination::Radians>{}, DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);

  DynamicTypedIndex d {Dimensions{5}, TypedIndex<Axis, inclination::Radians>{}};
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicTypedIndex {d, DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicTypedIndex {std::move(d), DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);
}


TEST(index_descriptors, DynamicTypedIndex_extend)
{
  DynamicTypedIndex d;
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
  static_assert(Dimensions{4} == TypedIndex<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} <= TypedIndex<Axis, Dimensions<2>>{});
  static_assert(Dimensions{3} < TypedIndex<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} != TypedIndex<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} <= TypedIndex<Axis, Axis, Axis, Axis>{});

  static_assert(TypedIndex<Axis, Axis, Axis>{} == Dimensions{3});
  static_assert(TypedIndex<Axis, Dimensions<2>, Axis>{} == Dimensions{4});
  static_assert(TypedIndex<Axis, Dimensions<2>>{} >= Dimensions{3});
  static_assert(TypedIndex<Axis, Dimensions<2>, Axis>{} > Dimensions{3});
  static_assert(TypedIndex<Axis, Dimensions<2>, Axis>{} != Dimensions{3});
  static_assert(TypedIndex<Axis, Axis, Axis, Axis>{} >= Dimensions{3});

  static_assert(Polar<Distance, angle::Radians>{} != Dimensions{5});
  static_assert(not (Polar<Distance, angle::Radians>{} < Dimensions{5}));
  static_assert(Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions{5});
  static_assert(not (Spherical<Distance, inclination::Radians, angle::Radians>{} > Dimensions{5}));

  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<> {}} == DynamicTypedIndex {TypedIndex<> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<> {}} == DynamicTypedIndex {}));
  EXPECT_TRUE((DynamicTypedIndex {Axis {}} == DynamicTypedIndex {Axis {}}));
  EXPECT_TRUE((DynamicTypedIndex {Axis {}} <= DynamicTypedIndex {Axis {}}));
  EXPECT_TRUE((DynamicTypedIndex {Axis {}} != DynamicTypedIndex {angle::Radians {}}));
  EXPECT_TRUE((DynamicTypedIndex {angle::Degrees {}} != DynamicTypedIndex {angle::Radians {}}));
  EXPECT_TRUE((DynamicTypedIndex {Axis {}} != DynamicTypedIndex {Polar<> {}}));
  EXPECT_TRUE((DynamicTypedIndex {Axis {}} == DynamicTypedIndex {TypedIndex<Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {Axis {}} < DynamicTypedIndex {TypedIndex<Axis, Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis> {}} == DynamicTypedIndex {Axis {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, Axis> {}} > DynamicTypedIndex {Axis {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis> {}} == DynamicTypedIndex {TypedIndex<Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, Axis> {}} >= DynamicTypedIndex {TypedIndex<Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}} == DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}} < DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis, Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis, Axis> {}} > DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}} == DynamicTypedIndex {Axis {}, angle::Radians {}, Axis {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}} == DynamicTypedIndex {Axis {}, angle::Radians {}, TypedIndex<Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}} == DynamicTypedIndex {Axis {}, angle::Radians {}, TypedIndex<TypedIndex<Axis>> {}}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5> {}} == DynamicTypedIndex {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<TypedIndex<Axis>, angle::Radians, TypedIndex<Axis>> {}} == DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<4> {}} < DynamicTypedIndex {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5> {}} > DynamicTypedIndex {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<2>{}}));
  EXPECT_TRUE((DynamicTypedIndex {Polar<Distance, angle::Radians> {}} == DynamicTypedIndex {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicTypedIndex {Polar<Distance, angle::Radians> {}, Axis{}} > DynamicTypedIndex {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians> {}} == DynamicTypedIndex {Spherical<Distance, angle::Radians, inclination::Radians> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians, angle::Radians> {}} != DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians> {}} != DynamicTypedIndex {Polar<Distance, angle::Radians> {}}));

  EXPECT_TRUE((DynamicTypedIndex {Axis {}} == Dimensions<1>{}));
  EXPECT_TRUE((Dimensions<1>{} == DynamicTypedIndex {Axis {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis> {}} == Dimensions<1>{}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, Axis, Axis> {}} == Dimensions<3>{}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, Axis> {}} < Dimensions<3>{}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions<4>{}} > TypedIndex<Axis, Axis> {}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions<4>{}} != TypedIndex<Axis, Axis> {}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions<3>{}, angle::Degrees{}} > Dimensions<3>{}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions<3>{}} < TypedIndex<Dimensions<3>, angle::Degrees>{}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}} == TypedIndex<Dimensions<3>, angle::Degrees, Dimensions<5>>{}));
  EXPECT_TRUE((DynamicTypedIndex {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} == TypedIndex<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicTypedIndex {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} == TypedIndex<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicTypedIndex {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} < TypedIndex<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<4>>{}));
  EXPECT_TRUE((DynamicTypedIndex {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} > TypedIndex<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<2>>{}));
  EXPECT_TRUE((DynamicTypedIndex {Axis{}, Dimensions<3>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} != TypedIndex<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE(not (DynamicTypedIndex {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} < TypedIndex<Dimensions<4>, angle::Degrees, Dimensions<3>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicTypedIndex {Axis {}} == Dimensions{1}));
  EXPECT_TRUE((Dimensions{1} == DynamicTypedIndex {Axis {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis> {}} == Dimensions{1}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, Axis, Axis> {}} == Dimensions{3}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, Axis> {}} < Dimensions{3}));
  EXPECT_TRUE((DynamicTypedIndex {Dimensions{4}} > TypedIndex<Axis, Axis> {}));
  EXPECT_TRUE((Dimensions{4} != DynamicTypedIndex {TypedIndex<Axis, Axis> {}}));

  EXPECT_TRUE((TypedIndex<> {} == DynamicTypedIndex {TypedIndex<> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}} == TypedIndex<Axis, angle::Radians>{}));
  EXPECT_TRUE((TypedIndex<Axis, angle::Radians>{} <= DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}}));
  EXPECT_TRUE((TypedIndex<Axis, angle::Radians>{} <= DynamicTypedIndex {Axis{}, angle::Radians{}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}} >= TypedIndex<Axis, angle::Radians>{}));
  EXPECT_TRUE((TypedIndex<Axis, angle::Radians>{} < DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Axis>{}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians>{}} <= TypedIndex<Axis, angle::Radians, Axis>{}));
  EXPECT_TRUE((angle::Radians{} == DynamicTypedIndex {angle::Radians{}}));
  EXPECT_TRUE((DynamicTypedIndex {inclination::Radians{}} == inclination::Radians{}));
  EXPECT_TRUE((angle::Radians{} != DynamicTypedIndex {inclination::Radians{}}));
  EXPECT_FALSE((angle::Radians{} < inclination::Radians{}));
  EXPECT_TRUE((DynamicTypedIndex {Polar<Distance, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_FALSE((Polar<Distance, angle::Radians>{} < DynamicTypedIndex {Dimensions<5>{}}));
  EXPECT_TRUE((DynamicTypedIndex {Spherical<Distance, inclination::Radians, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_FALSE((Spherical<Distance, inclination::Radians, angle::Radians>{} < DynamicTypedIndex {Dimensions<5>{}}));

  EXPECT_TRUE((DynamicTypedIndex<double> {TypedIndex<Axis, angle::Radians, Distance> {}} == DynamicTypedIndex<float> {TypedIndex<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicTypedIndex {TypedIndex<Axis, angle::Radians, Distance> {}} == DynamicTypedIndex<long double> {TypedIndex<Axis, angle::Radians, Distance> {}}));
}


TEST(index_descriptors, dynamic_arithmetic)
{
  EXPECT_TRUE(Dimensions{3} + Dimensions{4} == Dimensions{7});
  EXPECT_TRUE(Dimensions{7} - Dimensions{4} == Dimensions{3});
  EXPECT_TRUE((DynamicTypedIndex {Axis{}, angle::Radians{}} + DynamicTypedIndex {angle::Degrees{}, Axis{}} == DynamicTypedIndex {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((DynamicTypedIndex {Axis{}, angle::Radians{}} + TypedIndex<angle::Degrees, Axis>{} == DynamicTypedIndex {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((TypedIndex<Axis, angle::Radians>{} + DynamicTypedIndex {angle::Degrees{}, Axis{}} == DynamicTypedIndex {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
}


TEST(index_descriptors, internal_replicate_index_descriptor)
{
  using namespace internal;

  // fixed:
  static_assert(std::is_same_v<decltype(replicate_index_descriptor(TypedIndex<angle::Radians, Axis> {}, std::integral_constant<std::size_t, 2> {})), TypedIndex<TypedIndex<angle::Radians, Axis>, TypedIndex<angle::Radians, Axis>>>);

  // dynamic:
  auto d1 = replicate_index_descriptor(4, 3);
  EXPECT_EQ(get_dimension_size_of(d1), 12); EXPECT_EQ(get_euclidean_dimension_size_of(d1), 12); EXPECT_EQ(get_index_descriptor_component_count_of(d1), 12);
  auto d2 = replicate_index_descriptor(angle::Radians{}, 4);
  EXPECT_EQ(get_dimension_size_of(d2), 4); EXPECT_EQ(get_euclidean_dimension_size_of(d2), 8); EXPECT_EQ(get_index_descriptor_component_count_of(d2), 4);
  auto d3 = replicate_index_descriptor(Polar<Distance, angle::Radians>{}, 2);
  EXPECT_EQ(get_dimension_size_of(d3), 4); EXPECT_EQ(get_euclidean_dimension_size_of(d3), 6); EXPECT_EQ(get_index_descriptor_component_count_of(d3), 2);
}


TEST(index_descriptors, internal_is_uniform_component_of)
{
  using namespace internal;

  // fixed:
  static_assert(is_uniform_component_of(Axis {}, Axis {}));
  static_assert(is_uniform_component_of(Axis {}, Dimensions<10> {}));
  static_assert(not is_uniform_component_of(Dimensions<2> {}, Dimensions<10> {}));
  static_assert(not is_uniform_component_of(Axis {}, TypedIndex<Dimensions<10>, Distance> {}));
  static_assert(is_uniform_component_of(Distance {}, TypedIndex<Distance, Distance, Distance, Distance> {}));
  static_assert(is_uniform_component_of(angle::Radians {}, TypedIndex<angle::Radians, angle::Radians, angle::Radians, angle::Radians> {}));
  static_assert(not is_uniform_component_of(Polar<> {}, TypedIndex<Polar<>, Polar<>, Polar<>, Polar<>> {}));

  // dynamic:
  static_assert(is_uniform_component_of(1, Dimensions<10> {}));
  static_assert(is_uniform_component_of(Dimensions<1> {}, 10));
  static_assert(is_uniform_component_of(1, 10));
  static_assert(not is_uniform_component_of(2, 10));
  EXPECT_TRUE(is_uniform_component_of(Axis {}, DynamicTypedIndex {Axis {}}));
  EXPECT_TRUE(is_uniform_component_of(Axis {}, DynamicTypedIndex {Dimensions<10> {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicTypedIndex {Distance {}}, TypedIndex<Distance, Distance, Distance, Distance> {}));
  EXPECT_TRUE(is_uniform_component_of(DynamicTypedIndex {Axis {}}, DynamicTypedIndex {Axis {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicTypedIndex {Axis {}}, DynamicTypedIndex {Dimensions<10> {}}));
  EXPECT_FALSE(is_uniform_component_of(DynamicTypedIndex {Dimensions<2> {}}, DynamicTypedIndex {Dimensions<10> {}}));
  EXPECT_FALSE(is_uniform_component_of(DynamicTypedIndex {Axis {}}, DynamicTypedIndex {Dimensions<2> {}, Distance {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicTypedIndex {Distance {}}, DynamicTypedIndex {Distance {}, Distance {}, Distance {}, Distance {}}));

  auto d1 = DynamicTypedIndex<double> {Axis {}};
  auto f2 = DynamicTypedIndex<float> {Dimensions<2> {}};
  auto a2 = DynamicTypedIndex {Axis {}, Axis {}};
  auto a10 = DynamicTypedIndex {Dimensions<10> {}};
  static_assert(not is_uniform_component_of(d1, f2));
  static_assert(not is_uniform_component_of(Dimensions<2> {}, a10));
}

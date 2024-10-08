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

#include <gtest/gtest.h>
#include "basics/basics.hpp"

using namespace OpenKalman;
using numbers::pi;


TEST(basics, integral)
{
  static_assert(vector_space_descriptor<int>);
  static_assert(dynamic_vector_space_descriptor<int>);
  static_assert(not fixed_vector_space_descriptor<int>);
  static_assert(euclidean_vector_space_descriptor<int>);
  static_assert(not composite_vector_space_descriptor<int>);
  static_assert(not atomic_fixed_vector_space_descriptor<int>);
  static_assert(euclidean_vector_space_descriptor<int>);
  static_assert(dimension_size_of_v<int> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<int> == dynamic_size);
  static_assert(get_dimension_size_of(3) == 3);
  EXPECT_EQ(get_dimension_size_of(0), 0);
  EXPECT_EQ(get_dimension_size_of(3), 3);
  static_assert(get_euclidean_dimension_size_of(3) == 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(3), 3);
}


TEST(basics, dynamic_Dimensions)
{
  using D = Dimensions<dynamic_size>;

  static_assert(vector_space_descriptor<D>);
  static_assert(dynamic_vector_space_descriptor<D>);
  static_assert(not fixed_vector_space_descriptor<D>);
  static_assert(euclidean_vector_space_descriptor<D>);
  static_assert(not composite_vector_space_descriptor<D>);
  static_assert(euclidean_vector_space_descriptor<D>);
  static_assert(not atomic_fixed_vector_space_descriptor<D>);
  static_assert(dimension_size_of_v<D> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<D> == dynamic_size);
  static_assert(get_dimension_size_of(Dimensions {0}) == 0);
  static_assert(get_dimension_size_of(Dimensions {3}) == 3);
  static_assert(get_euclidean_dimension_size_of(Dimensions{3}) == 3);
  static_assert(get_dimension_size_of(Dimensions<dynamic_size> {Axis {}}) == 1);
  static_assert(get_dimension_size_of(Dimensions<dynamic_size> {Dimensions<3> {}}) == 3);
  static_assert(static_cast<std::size_t>(Dimensions {3}) == 3);
}


TEST(basics, DynamicDescriptor_traits)
{
  static_assert(vector_space_descriptor<DynamicDescriptor<>>);
  static_assert(vector_space_descriptor<DynamicDescriptor<double>>);
  static_assert(dynamic_vector_space_descriptor<DynamicDescriptor<>>);
  static_assert(dynamic_vector_space_descriptor<DynamicDescriptor<float>>);
  static_assert(dynamic_vector_space_descriptor<DynamicDescriptor<float, long double>>);
  static_assert(not fixed_vector_space_descriptor<DynamicDescriptor<>>);
  static_assert(not euclidean_vector_space_descriptor<DynamicDescriptor<>>);
  static_assert(composite_vector_space_descriptor<DynamicDescriptor<>>);
  static_assert(composite_vector_space_descriptor<DynamicDescriptor<double, long double>>);
  static_assert(dimension_size_of_v<DynamicDescriptor<>> == dynamic_size);
  static_assert(dimension_size_of_v<DynamicDescriptor<double>> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<DynamicDescriptor<>> == dynamic_size);
  static_assert(euclidean_dimension_size_of_v<DynamicDescriptor<float, long double>> == dynamic_size);
}


TEST(basics, DynamicDescriptor_construct)
{
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {}), 0);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {Dimensions<0>{}}), 0);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {Axis{}}), 1);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {angle::Degrees{}}), 1);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor {angle::Degrees{}}), 2);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {Dimensions<5>{}}), 5);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {Dimensions{5}}), 5);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {Polar<Distance, angle::Radians>{}}), 2);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {Polar<Distance, angle::PositiveRadians>{}}), 2);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor {Polar<angle::PositiveDegrees, Distance>{}}), 3);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}}), 3);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor {Spherical<inclination::Radians, Distance, angle::PositiveDegrees>{}}), 4);

  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {Dimensions{5}, Dimensions<1>{}, angle::Degrees{}}), 7);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor {Axis{}, Dimensions{5}, angle::Degrees{}}), 8);
  EXPECT_EQ(get_dimension_size_of(DynamicDescriptor {FixedDescriptor<Axis, inclination::Radians>{}, angle::Degrees{}, Dimensions{5}}), 8);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor {Dimensions{5}, FixedDescriptor<Axis, inclination::Radians>{}, angle::Degrees{}}), 10);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor {Dimensions{5}, FixedDescriptor<Axis, inclination::Radians>{}, DynamicDescriptor {FixedDescriptor<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);

  DynamicDescriptor d {Dimensions{5}, FixedDescriptor<Axis, inclination::Radians>{}};
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor {d, DynamicDescriptor {FixedDescriptor<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);
  EXPECT_EQ(get_euclidean_dimension_size_of(DynamicDescriptor {std::move(d), DynamicDescriptor {FixedDescriptor<Axis, angle::Radians>{}}, angle::Degrees{}}), 13);
}


TEST(basics, DynamicDescriptor_extend)
{
  DynamicDescriptor d;
  EXPECT_EQ(get_dimension_size_of(d), 0); EXPECT_EQ(get_euclidean_dimension_size_of(d), 0); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d), 0);
  d.extend(Axis{});
  EXPECT_EQ(get_dimension_size_of(d), 1); EXPECT_EQ(get_euclidean_dimension_size_of(d), 1); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d), 1);
  d.extend(Dimensions{5}, Dimensions<5>{}, angle::Radians{}, FixedDescriptor<Axis, inclination::Radians>{}, Polar<angle::Degrees, Distance>{});
  EXPECT_EQ(get_dimension_size_of(d), 16); EXPECT_EQ(get_euclidean_dimension_size_of(d), 19); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d), 15);
}


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

  static_assert(Dimensions{3} == FixedDescriptor<Axis, Axis, Axis>{});
  static_assert(Dimensions{4} == FixedDescriptor<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} <= FixedDescriptor<Axis, Dimensions<2>>{});
  static_assert(Dimensions{3} < FixedDescriptor<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} != FixedDescriptor<Axis, Dimensions<2>, Axis>{});
  static_assert(Dimensions{3} <= FixedDescriptor<Axis, Axis, Axis, Axis>{});

  static_assert(FixedDescriptor<Axis, Axis, Axis>{} == Dimensions{3});
  static_assert(FixedDescriptor<Axis, Dimensions<2>, Axis>{} == Dimensions{4});
  static_assert(FixedDescriptor<Axis, Dimensions<2>>{} >= Dimensions{3});
  static_assert(FixedDescriptor<Axis, Dimensions<2>, Axis>{} > Dimensions{3});
  static_assert(FixedDescriptor<Axis, Dimensions<2>, Axis>{} != Dimensions{3});
  static_assert(FixedDescriptor<Axis, Axis, Axis, Axis>{} >= Dimensions{3});

  static_assert(Polar<Distance, angle::Radians>{} != Dimensions{5});
  static_assert(not (Polar<Distance, angle::Radians>{} < Dimensions{5}));
  static_assert(Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions{5});
  static_assert(not (Spherical<Distance, inclination::Radians, angle::Radians>{} > Dimensions{5}));

  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<> {}} == DynamicDescriptor {FixedDescriptor<> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<> {}} == DynamicDescriptor {}));
  EXPECT_TRUE((DynamicDescriptor {Axis {}} == DynamicDescriptor {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor {Axis {}} <= DynamicDescriptor {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor {Axis {}} != DynamicDescriptor {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor {angle::Degrees {}} != DynamicDescriptor {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor {Axis {}} != DynamicDescriptor {Polar<> {}}));
  EXPECT_TRUE((DynamicDescriptor {Axis {}} == DynamicDescriptor {FixedDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {Axis {}} < DynamicDescriptor {FixedDescriptor<Axis, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis> {}} == DynamicDescriptor {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, Axis> {}} > DynamicDescriptor {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis> {}} == DynamicDescriptor {FixedDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, Axis> {}} >= DynamicDescriptor {FixedDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}} == DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}} < DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis, Axis> {}} > DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}} == DynamicDescriptor {Axis {}, angle::Radians {}, Axis {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}} == DynamicDescriptor {Axis {}, angle::Radians {}, FixedDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}} == DynamicDescriptor {Axis {}, angle::Radians {}, FixedDescriptor<FixedDescriptor<Axis>> {}}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5> {}} == DynamicDescriptor {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<FixedDescriptor<Axis>, angle::Radians, FixedDescriptor<Axis>> {}} == DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<4> {}} < DynamicDescriptor {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions<3>{}, Dimensions<2>{}, angle::Radians{}, Dimensions<5> {}} > DynamicDescriptor {Dimensions<2>{}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<2>{}}));
  EXPECT_TRUE((DynamicDescriptor {Polar<Distance, angle::Radians> {}} == DynamicDescriptor {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor {Polar<Distance, angle::Radians> {}, Axis{}} > DynamicDescriptor {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor {Spherical<Distance, angle::Radians, inclination::Radians> {}} == DynamicDescriptor {Spherical<Distance, angle::Radians, inclination::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, angle::Radians> {}} != DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians> {}} != DynamicDescriptor {Polar<Distance, angle::Radians> {}}));

  EXPECT_TRUE((DynamicDescriptor {Axis {}} == Dimensions<1>{}));
  EXPECT_TRUE((Dimensions<1>{} == DynamicDescriptor {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis> {}} == Dimensions<1>{}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, Axis, Axis> {}} == Dimensions<3>{}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, Axis> {}} < Dimensions<3>{}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions<4>{}} > FixedDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions<4>{}} != FixedDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions<3>{}, angle::Degrees{}} > Dimensions<3>{}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions<3>{}} < FixedDescriptor<Dimensions<3>, angle::Degrees>{}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}} == FixedDescriptor<Dimensions<3>, angle::Degrees, Dimensions<5>>{}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} == FixedDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} == FixedDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} < FixedDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<4>>{}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} > FixedDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<2>>{}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, Dimensions<3>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} != FixedDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE(not (DynamicDescriptor {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} < FixedDescriptor<Dimensions<4>, angle::Degrees, Dimensions<3>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicDescriptor {Axis {}} == Dimensions{1}));
  EXPECT_TRUE((Dimensions{1} == DynamicDescriptor {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis> {}} == Dimensions{1}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, Axis, Axis> {}} == Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, Axis> {}} < Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor {Dimensions{4}} > FixedDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((Dimensions{4} != DynamicDescriptor {FixedDescriptor<Axis, Axis> {}}));

  EXPECT_TRUE((FixedDescriptor<> {} == DynamicDescriptor {FixedDescriptor<> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians>{}} == FixedDescriptor<Axis, angle::Radians>{}));
  EXPECT_TRUE((FixedDescriptor<Axis, angle::Radians>{} <= DynamicDescriptor {FixedDescriptor<Axis, angle::Radians>{}}));
  EXPECT_TRUE((FixedDescriptor<Axis, angle::Radians>{} <= DynamicDescriptor {Axis{}, angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians>{}} >= FixedDescriptor<Axis, angle::Radians>{}));
  EXPECT_TRUE((FixedDescriptor<Axis, angle::Radians>{} < DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Axis>{}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians>{}} <= FixedDescriptor<Axis, angle::Radians, Axis>{}));
  EXPECT_TRUE((angle::Radians{} == DynamicDescriptor {angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor {inclination::Radians{}} == inclination::Radians{}));
  EXPECT_TRUE((angle::Radians{} != DynamicDescriptor {inclination::Radians{}}));
  EXPECT_FALSE((angle::Radians{} < inclination::Radians{}));
  EXPECT_TRUE((DynamicDescriptor {Polar<Distance, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_FALSE((Polar<Distance, angle::Radians>{} < DynamicDescriptor {Dimensions<5>{}}));
  EXPECT_TRUE((DynamicDescriptor {Spherical<Distance, inclination::Radians, angle::Radians>{}} != Dimensions<5>{}));
  EXPECT_FALSE((Spherical<Distance, inclination::Radians, angle::Radians>{} < DynamicDescriptor {Dimensions<5>{}}));

  EXPECT_TRUE((DynamicDescriptor<double> {FixedDescriptor<Axis, angle::Radians, Distance> {}} == DynamicDescriptor<float> {FixedDescriptor<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicDescriptor {FixedDescriptor<Axis, angle::Radians, Distance> {}} == DynamicDescriptor<long double> {FixedDescriptor<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {FixedDescriptor<Axis, angle::Radians, Distance> {}} < DynamicDescriptor<float> {FixedDescriptor<Axis, angle::Radians, Distance, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {FixedDescriptor<Axis, angle::Radians, Distance, Axis> {}} > DynamicDescriptor<float> {FixedDescriptor<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {FixedDescriptor<Axis, angle::Radians, Distance> {}} <= DynamicDescriptor<float> {FixedDescriptor<Axis, angle::Radians, Distance, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {FixedDescriptor<Axis, angle::Radians, Distance, Axis> {}} >= DynamicDescriptor<float> {FixedDescriptor<Axis, angle::Radians, Distance> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {FixedDescriptor<Axis, angle::Radians, Distance, Axis> {}} != DynamicDescriptor<float> {FixedDescriptor<Axis, angle::Radians, Distance> {}}));
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
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}} + DynamicDescriptor {angle::Degrees{}, Axis{}} == DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}} + FixedDescriptor<angle::Degrees, Axis>{} == DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((FixedDescriptor<Axis, angle::Radians>{} + DynamicDescriptor {angle::Degrees{}, Axis{}} == DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));

  EXPECT_TRUE(Dimensions{7} - Dimensions{7} == Dimensions{0});
  EXPECT_TRUE(Dimensions{7} - Dimensions{4} == Dimensions{3});

  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - FixedDescriptor<>{} == DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - FixedDescriptor<Axis>{} == DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - FixedDescriptor<angle::Degrees, Axis>{} == DynamicDescriptor {Axis{}, angle::Radians{}}));

  EXPECT_TRUE((DynamicDescriptor {Dimensions{7}} - Dimensions{4} == Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - Dimensions{1} == DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}}));

  EXPECT_TRUE((DynamicDescriptor {Dimensions{7}} - DynamicDescriptor {Dimensions{4}} == Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - DynamicDescriptor {Axis{}} == DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - DynamicDescriptor {angle::Degrees{}, Axis{}} == DynamicDescriptor {Axis{}, angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, angle::Degrees{}, Axis{}} - DynamicDescriptor {angle::Degrees{}, Axis{}} == FixedDescriptor<Axis, angle::Radians>{}));
}


TEST(basics, internal_replicate_vector_space_descriptor)
{
  using namespace internal;

  // fixed:
  static_assert(std::is_same_v<decltype(replicate_vector_space_descriptor(FixedDescriptor<angle::Radians, Axis> {}, std::integral_constant<std::size_t, 2> {})), FixedDescriptor<FixedDescriptor<angle::Radians, Axis>, FixedDescriptor<angle::Radians, Axis>>>);

  // dynamic:
  auto d1 = replicate_vector_space_descriptor(4, 3);
  EXPECT_EQ(get_dimension_size_of(d1), 12); EXPECT_EQ(get_euclidean_dimension_size_of(d1), 12); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d1), 12);
  auto d2 = replicate_vector_space_descriptor(angle::Radians{}, 4);
  EXPECT_EQ(get_dimension_size_of(d2), 4); EXPECT_EQ(get_euclidean_dimension_size_of(d2), 8); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d2), 4);
  auto d3 = replicate_vector_space_descriptor(Polar<Distance, angle::Radians>{}, 2);
  EXPECT_EQ(get_dimension_size_of(d3), 4); EXPECT_EQ(get_euclidean_dimension_size_of(d3), 6); EXPECT_EQ(get_vector_space_descriptor_component_count_of(d3), 2);
}


TEST(basics, internal_is_uniform_component_of)
{
  using namespace internal;

  // fixed:
  static_assert(is_uniform_component_of(Axis {}, Axis {}));
  static_assert(is_uniform_component_of(Axis {}, Dimensions<10> {}));
  static_assert(not is_uniform_component_of(Dimensions<2> {}, Dimensions<10> {}));
  static_assert(not is_uniform_component_of(Axis {}, FixedDescriptor<Dimensions<10>, Distance> {}));
  static_assert(is_uniform_component_of(Distance {}, FixedDescriptor<Distance, Distance, Distance, Distance> {}));
  static_assert(is_uniform_component_of(angle::Radians {}, FixedDescriptor<angle::Radians, angle::Radians, angle::Radians, angle::Radians> {}));
  static_assert(not is_uniform_component_of(Polar<> {}, FixedDescriptor<Polar<>, Polar<>, Polar<>, Polar<>> {}));

  // dynamic:
  static_assert(is_uniform_component_of(1, Dimensions<10> {}));
  static_assert(is_uniform_component_of(Dimensions<1> {}, 10));
  static_assert(is_uniform_component_of(1, 10));
  static_assert(not is_uniform_component_of(2, 10));
  EXPECT_TRUE(is_uniform_component_of(Axis {}, DynamicDescriptor {Axis {}}));
  EXPECT_TRUE(is_uniform_component_of(Axis {}, DynamicDescriptor {Dimensions<10> {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor {Distance {}}, FixedDescriptor<Distance, Distance, Distance, Distance> {}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor {Axis {}}, DynamicDescriptor {Axis {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor {Axis {}}, DynamicDescriptor {Dimensions<10> {}}));
  EXPECT_FALSE(is_uniform_component_of(DynamicDescriptor {Dimensions<2> {}}, DynamicDescriptor {Dimensions<10> {}}));
  EXPECT_FALSE(is_uniform_component_of(DynamicDescriptor {Axis {}}, DynamicDescriptor {Dimensions<2> {}, Distance {}}));
  EXPECT_TRUE(is_uniform_component_of(DynamicDescriptor {Distance {}}, DynamicDescriptor {Distance {}, Distance {}, Distance {}, Distance {}}));

  auto d1 = DynamicDescriptor<double> {Axis {}};
  auto f2 = DynamicDescriptor<float> {Dimensions<2> {}};
  auto a2 = DynamicDescriptor {Axis {}, Axis {}};
  auto a10 = DynamicDescriptor {Dimensions<10> {}};
  static_assert(not is_uniform_component_of(d1, f2));
  static_assert(not is_uniform_component_of(Dimensions<2> {}, a10));
}


TEST(basics, smallest_vector_space_descriptor_dynamic)
{
  static_assert(internal::smallest_vector_space_descriptor(Dimensions<3>{}, 4) == 3);
  static_assert(internal::smallest_vector_space_descriptor(Dimensions<4>{}, 3) == 3);
  static_assert(internal::smallest_vector_space_descriptor(3, 4, 5) == 3);

  EXPECT_TRUE((internal::smallest_vector_space_descriptor(DynamicDescriptor{Dimensions<3>{}}, DynamicDescriptor{Dimensions<4>{}}) == 3));
  EXPECT_TRUE((internal::smallest_vector_space_descriptor(DynamicDescriptor{Dimensions<3>{}}, DynamicDescriptor{Dimensions<4>{}}) == 3));
  EXPECT_TRUE((internal::smallest_vector_space_descriptor(DynamicDescriptor{angle::Radians{}, Dimensions<3>{}}, DynamicDescriptor{Dimensions<4>{}, angle::Degrees{}}) == DynamicDescriptor{angle::Radians{}, Dimensions<3>{}}));
}


TEST(basics, slice_vector_space_descriptor_dynamic)
{
  using namespace internal;

  EXPECT_TRUE((DynamicDescriptor {}.slice(0, 0) == DynamicDescriptor{}));

  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, Distance{}}.slice(0, 1) == Axis{}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, Distance{}}.slice(1, 1) == DynamicDescriptor {angle::Radians{}}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, Distance{}}.slice(1, 2) == DynamicDescriptor {angle::Radians{}, Distance{}}));
  EXPECT_TRUE((DynamicDescriptor {Axis{}, angle::Radians{}, Distance{}}.slice(0, 3) == DynamicDescriptor {Axis{}, angle::Radians{}, Distance{}}));

  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 0) == DynamicDescriptor{}));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 1) == angle::Radians{}));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 2) == DynamicDescriptor {angle::Radians{}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 3) == DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}}));
  EXPECT_ANY_THROW((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 4)));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(0, 5) == DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 0) == DynamicDescriptor{}));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 1) == Axis{}));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 2) == DynamicDescriptor {Axis{}, Distance{}}));
  EXPECT_ANY_THROW((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 3)));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(1, 4) == DynamicDescriptor {Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 0) == DynamicDescriptor{}));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 1) == Distance{}));
  EXPECT_ANY_THROW((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 2)));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(2, 3) == DynamicDescriptor {Distance{}, Polar<Distance, angle::Radians>{}}));

  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 0) == DynamicDescriptor {}));
  EXPECT_ANY_THROW((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 1)));
  EXPECT_TRUE((DynamicDescriptor {angle::Radians{}, Axis{}, Distance{}, Polar<Distance, angle::Radians>{}}.slice(3, 2) == Polar<Distance, angle::Radians>{}));

  EXPECT_TRUE((DynamicDescriptor {Polar<Distance, angle::Radians>{}, Axis{}}.slice(0, 2) == Polar<Distance, angle::Radians>{}));
  EXPECT_TRUE((DynamicDescriptor {Polar<Distance, angle::Radians>{}, Axis{}}.slice(2, 1) == Axis{}));
}


TEST(basics, get_vector_space_descriptor_slice_dynamic)
{
  using namespace internal;

  static_assert(get_vector_space_descriptor_slice(Dimensions{7}, 0, 7) == Dimensions<7>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions{7}, 1, 6) == Dimensions<6>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions{7}, 2, 3) == Dimensions<3>{});
  static_assert(get_vector_space_descriptor_slice(Dimensions{7}, 2, 0) == Dimensions<0>{});

  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 6) == FixedDescriptor<Dimensions<2>, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 7)));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, -1, 7)));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 0, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 5) == FixedDescriptor<Axis, Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 4) == FixedDescriptor<Axis, Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 3)));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 2) == FixedDescriptor<Axis, Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 1) == FixedDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 1, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 4) == FixedDescriptor<Distance, Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 5)));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 3) == FixedDescriptor<Distance, Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 2)));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 1) == FixedDescriptor<Distance>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 2, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 3) == FixedDescriptor<Polar<Distance, angle::Radians>, Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 2) == FixedDescriptor<Polar<Distance, angle::Radians>>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 1)));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 3, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 5, 1) == FixedDescriptor<Axis>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 5, 0) == FixedDescriptor<>{}));
  EXPECT_TRUE((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 6, 0) == FixedDescriptor<>{}));
  EXPECT_ANY_THROW((get_vector_space_descriptor_slice(DynamicDescriptor {Dimensions<2>{}, Distance{}, Polar<Distance, angle::Radians>{}, Axis{}}, 7, -1)));
}


/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref vector_space_descriptor equivalence
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/DynamicDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/index.hpp"

using namespace OpenKalman::descriptor;

#include "linear-algebra/vector-space-descriptors/functions/internal/canonical_equivalent.hpp"

TEST(descriptors, canonical_equivalent)
{
  using namespace internal;
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Dimensions<0>{}))>, StaticDescriptor<>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(std::integral_constant<int, 0>{}))>, StaticDescriptor<>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Axis{}))>, Axis>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Dimensions<1>{}))>, Axis>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(std::integral_constant<int, 1>{}))>, Axis>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Dimensions<3>{}))>, Dimensions<3>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(angle::Degrees{}))>, angle::PositiveDegrees>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Angle<angle::Limits<0, 100>>{}))>, std::decay_t<decltype(canonical_equivalent(Angle<angle::Limits<-50, 50>>{}))>>);
#if __cpp_nontype_template_args >= 201911L
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(angle::Radians{}))>, angle::PositiveRadians>);
#endif

  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Polar<Distance, angle::Degrees>{}))>, Polar<Distance, angle::PositiveDegrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Polar<angle::Degrees, Distance>{}))>, Polar<angle::PositiveDegrees, Distance>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Spherical<Distance, angle::Degrees, inclination::Degrees>{}))>, Spherical<Distance, angle::PositiveDegrees, inclination::Degrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Spherical<Distance, inclination::Degrees, angle::Degrees>{}))>, Spherical<Distance, inclination::Degrees, angle::PositiveDegrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Spherical<angle::Degrees, Distance, inclination::Degrees>{}))>, Spherical<angle::PositiveDegrees, Distance, inclination::Degrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Spherical<inclination::Degrees, Distance, angle::Degrees>{}))>, Spherical<inclination::Degrees, Distance, angle::PositiveDegrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Spherical<angle::Degrees, inclination::Degrees, Distance>{}))>, Spherical<angle::PositiveDegrees, inclination::Degrees, Distance>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(Spherical<inclination::Degrees, angle::Degrees, Distance>{}))>, Spherical<inclination::Degrees, angle::PositiveDegrees, Distance>>);

  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<>{}))>, StaticDescriptor<>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<StaticDescriptor<>, StaticDescriptor<>>{}))>, StaticDescriptor<>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Dimensions<0>>{}))>, StaticDescriptor<>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Dimensions<0>, Dimensions<0>>{}))>, StaticDescriptor<>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Axis, Axis, Axis>{}))>, Dimensions<3>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Axis, std::integral_constant<int, 1>, Axis>{}))>, Dimensions<3>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Dimensions<3>>{}))>, Dimensions<3>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Dimensions<3>, Dimensions<2>>{}))>, Dimensions<5>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Axis>{}))>, Axis>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<StaticDescriptor<>, angle::Degrees>{}))>, angle::PositiveDegrees>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<StaticDescriptor<>, angle::Degrees, StaticDescriptor<>>{}))>, angle::PositiveDegrees>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Axis, angle::Degrees>{}))>, StaticDescriptor<Axis, angle::PositiveDegrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<StaticDescriptor<Axis>, angle::Degrees>{}))>, StaticDescriptor<Axis, angle::PositiveDegrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Dimensions<3>, angle::Degrees>{}))>, StaticDescriptor<Dimensions<3>, angle::PositiveDegrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<angle::Degrees, Dimensions<3>>{}))>, StaticDescriptor<angle::PositiveDegrees, Dimensions<3>>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<angle::Degrees, Dimensions<3>, angle::Degrees>{}))>, StaticDescriptor<angle::PositiveDegrees, Dimensions<3>, angle::PositiveDegrees>>);
  static_assert(not std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Axis, angle::Degrees, angle::Degrees, Distance>{}))>, StaticDescriptor<Axis, angle::PositiveDegrees, Axis>>);
  static_assert(not std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Axis, angle::Degrees>{}))>, StaticDescriptor<Polar<Distance, angle::PositiveDegrees>>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<Dimensions<1>, Dimensions<2>, angle::Degrees>{}))>, StaticDescriptor<Dimensions<3>, angle::PositiveDegrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<StaticDescriptor<Dimensions<1>, Dimensions<2>>, angle::Degrees>{}))>, StaticDescriptor<Dimensions<3>, angle::PositiveDegrees>>);
  static_assert(std::is_same_v<std::decay_t<decltype(canonical_equivalent(StaticDescriptor<angle::Degrees, Dimensions<1>, Dimensions<2>>{}))>, StaticDescriptor<angle::PositiveDegrees, Dimensions<3>>>);
}


#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp"

TEST(descriptors, maybe_equivalent_to)
{
  static_assert(maybe_equivalent_to<>);

  static_assert(euclidean_vector_space_descriptor<std::integral_constant<std::size_t, 2>>);
  static_assert(dynamic_vector_space_descriptor<unsigned>);
  static_assert(maybe_equivalent_to<std::integral_constant<std::size_t, 2>, unsigned>);
  static_assert(maybe_equivalent_to<unsigned, std::integral_constant<std::size_t, 2>>);
  static_assert(not maybe_equivalent_to<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>);
  static_assert(maybe_equivalent_to<Axis>);
  static_assert(maybe_equivalent_to<StaticDescriptor<>, unsigned>);
  static_assert(maybe_equivalent_to<unsigned, StaticDescriptor<>>);
  static_assert(dynamic_vector_space_descriptor<Dimensions<>>);
  static_assert(maybe_equivalent_to<Axis, Dimensions<>>);
  static_assert(maybe_equivalent_to<Axis, Dimensions<>, Axis>);
  static_assert(maybe_equivalent_to<Dimensions<>, Axis>);
  static_assert(maybe_equivalent_to<Dimensions<>, Axis, Dimensions<>>);
  static_assert(not maybe_equivalent_to<Axis, Polar<>>);
  static_assert(not maybe_equivalent_to<Polar<>, angle::Radians>);

  static_assert(maybe_equivalent_to<StaticDescriptor<>, Dimensions<0>>);
  static_assert(maybe_equivalent_to<Dimensions<0>, StaticDescriptor<>>);
  static_assert(maybe_equivalent_to<StaticDescriptor<>, Dimensions<0>>);
  static_assert(maybe_equivalent_to<Dimensions<0>, std::size_t>);
  static_assert(maybe_equivalent_to<std::size_t, Dimensions<0>>);
  static_assert(maybe_equivalent_to<Dimensions<1>, std::size_t>);
  static_assert(maybe_equivalent_to<Dimensions<>, Dimensions<10>, unsigned, Dimensions<10>>);
  static_assert(not maybe_equivalent_to<Dimensions<>, Dimensions<10>, int, Dimensions<5>>);
  static_assert(not maybe_equivalent_to<angle::Degrees, Dimensions<1>, std::size_t>);
  static_assert(not maybe_equivalent_to<int, angle::Radians>);
  static_assert(not maybe_equivalent_to<angle::Degrees, int>);
  static_assert(not maybe_equivalent_to<StaticDescriptor<Axis, angle::Radians, angle::Radians>, StaticDescriptor<Axis, angle::Radians, inclination::Radians>>);
  static_assert(not maybe_equivalent_to<StaticDescriptor<Axis, angle::Radians>, Polar<>>);
}


#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp"

TEST(descriptors, equivalent_to)
{
  static_assert(equivalent_to<>);

  static_assert(not equivalent_to<std::integral_constant<std::size_t, 2>, int>);
  static_assert(equivalent_to<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 2>>);
  static_assert(equivalent_to<std::integral_constant<std::size_t, 2>, std::integral_constant<int, 2>>);
  static_assert(not equivalent_to<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>);

  static_assert(equivalent_to<Axis>);
  static_assert(equivalent_to<StaticDescriptor<>, StaticDescriptor<>>);
  static_assert(equivalent_to<Dimensions<0>, StaticDescriptor<>>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<0>>, StaticDescriptor<>>);
  static_assert(equivalent_to<Axis, Axis>);
  static_assert(equivalent_to<Dimensions<1>, Axis>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<1>>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<Axis, Dimensions<1>>);
  static_assert(not equivalent_to<Axis, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, angle::Radians>);
  static_assert(not equivalent_to<Axis, Polar<>>);
  static_assert(equivalent_to<Axis, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis>, Axis>);
  static_assert(equivalent_to<StaticDescriptor<Axis>, StaticDescriptor<Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Dimensions<2>, inclination::Radians, Dimensions<3>>, StaticDescriptor<Axis, Axis, inclination::Radians, Axis, Axis, Axis>>);
  static_assert(equivalent_to<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians, StaticDescriptor<StaticDescriptor<Axis>>>>);
  static_assert(equivalent_to<StaticDescriptor<StaticDescriptor<Axis>, angle::Radians, StaticDescriptor<Axis>>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(equivalent_to<Polar<>, Polar<>>);
  static_assert(equivalent_to<Spherical<>, Spherical<>>);

  static_assert(not equivalent_to<Dimensions<0>, std::size_t>);
  static_assert(not equivalent_to<Dimensions<1>, std::size_t>);
  static_assert(not equivalent_to<Dimensions<>, Dimensions<10>, int, Dimensions<10>>);
  static_assert(not equivalent_to<Dimensions<>, Dimensions<10>, int, Dimensions<5>>);
  static_assert(not equivalent_to<angle::Degrees, Dimensions<1>, std::size_t>);
  static_assert(not equivalent_to<int, angle::Radians>);
  static_assert(not equivalent_to<angle::Degrees, int>);
}


#include "linear-algebra/vector-space-descriptors/concepts/internal/prefix_of.hpp"

TEST(descriptors, prefix_of)
{
  using namespace internal;
  static_assert(prefix_of<StaticDescriptor<>, Axis>);
  static_assert(prefix_of<StaticDescriptor<>, Dimensions<2>>);
  static_assert(prefix_of<StaticDescriptor<>, StaticDescriptor<Axis>>);
  static_assert(prefix_of<StaticDescriptor<>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<StaticDescriptor<Axis>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<StaticDescriptor<Axis>, StaticDescriptor<Dimensions<2>, angle::Radians>>);
  static_assert(prefix_of<Axis, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<Axis, StaticDescriptor<Dimensions<2>, angle::Radians>>);
  static_assert(not prefix_of<angle::Radians, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(not prefix_of<StaticDescriptor<angle::Radians>, StaticDescriptor<Axis, angle::Radians>>);
  static_assert(prefix_of<StaticDescriptor<Axis, angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(prefix_of<StaticDescriptor<Axis, angle::Radians, Axis>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not prefix_of<StaticDescriptor<Axis, angle::Radians, angle::Radians>, StaticDescriptor<Axis, angle::Radians, Axis>>);
  static_assert(not prefix_of<Dimensions<>, Axis>);
}


#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"

TEST(descriptors, fixed_comparison)
{
  static_assert(Dimensions<3>{} == Dimensions<3>{});
  static_assert(Dimensions<3>{} <= Dimensions<3>{});
  static_assert(Dimensions<3>{} >= Dimensions<3>{});
  static_assert((Dimensions<3>{} != Dimensions<4>{}));
  static_assert((Dimensions<3>{} < Dimensions<4>{}));
  static_assert((Dimensions<3>{} <= Dimensions<4>{}));
  static_assert((Dimensions<4>{} > Dimensions<3>{}));
  static_assert((Dimensions<4>{} >= Dimensions<3>{}));

  static_assert((Dimensions<3>{} == StaticDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} <= StaticDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} <= StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<3>{} < StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} >= StaticDescriptor<Axis, Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} >= StaticDescriptor<Axis, Axis, Axis>{}));
  static_assert((Dimensions<4>{} > StaticDescriptor<Axis, Axis, Axis>{}));

  static_assert(StaticDescriptor<> {} == StaticDescriptor<> {});

  static_assert(StaticDescriptor<Axis, angle::Radians>{} == StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis, angle::Radians>{} <= StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{} == StaticDescriptor<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{});
  static_assert(StaticDescriptor<Axis, Dimensions<3>, angle::Radians, Axis, Dimensions<2>>{} < StaticDescriptor<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{});
  static_assert(StaticDescriptor<Axis, angle::Radians>{} < StaticDescriptor<Axis, angle::Radians, Axis>{});
  static_assert(StaticDescriptor<Axis, angle::Radians>{} >= StaticDescriptor<Axis, angle::Radians>{});
  static_assert(StaticDescriptor<Axis, angle::Radians, Axis>{} > StaticDescriptor<Axis, angle::Radians>{});

  static_assert(angle::Radians{} == angle::Radians{});
  static_assert(inclination::Radians{} == inclination::Radians{});
  static_assert(angle::Radians{} != inclination::Radians{});
  static_assert(not (angle::Radians{} < inclination::Radians{}));
  static_assert((Polar<Distance, angle::Radians>{} != Dimensions<5>{}));
  static_assert(not (Polar<Distance, angle::Radians>{} < Dimensions<5>{}));
  static_assert((Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions<5>{}));
  static_assert(not (Spherical<Distance, inclination::Radians, angle::Radians>{} < Dimensions<5>{}));
}


TEST(descriptors, dynamic_comparison)
{
  static_assert(Dimensions{0} == StaticDescriptor<>{});
  static_assert(StaticDescriptor<>{} == Dimensions{0});
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

  static_assert(Polar<>{} != Dimensions{5});
  static_assert(Dimensions{5} != Polar<>{});
  static_assert(not (Polar<>{} < Dimensions{5}));
  static_assert(Spherical<>{} != Dimensions{5});
  static_assert(not (Spherical<>{} > Dimensions{5}));

  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<> {}} == DynamicDescriptor<double> {StaticDescriptor<> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<> {}} == DynamicDescriptor<double> {}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} == DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} <= DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} == DynamicDescriptor<double> {StaticDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} < DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis> {}} == DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}} > DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis> {}} == DynamicDescriptor<double> {StaticDescriptor<Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}} >= DynamicDescriptor<double> {StaticDescriptor<Axis> {}}));

  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<> {}} < DynamicDescriptor<double> {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians {}} > DynamicDescriptor<double> {StaticDescriptor<> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians {}} == DynamicDescriptor<double> {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Degrees {}} == DynamicDescriptor<double> {angle::PositiveDegrees {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Degrees {}} != DynamicDescriptor<double> {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {inclination::Radians {}} != DynamicDescriptor<double> {inclination::Degrees {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} != DynamicDescriptor<double> {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} != DynamicDescriptor<double> {Polar<> {}}));

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
  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}, Axis{}} == DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}, Axis{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}, Axis{}} != DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}, Axis{}} > DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Spherical<Distance, angle::Radians, inclination::Radians> {}} == DynamicDescriptor<double> {Spherical<Distance, angle::Radians, inclination::Radians> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, angle::Radians> {}} != DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, angle::Radians> {}} != DynamicDescriptor<double> {Polar<Distance, angle::Radians> {}}));

  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} == Dimensions<1>{}));
  EXPECT_TRUE((Dimensions<1>{} == DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis> {}} == Dimensions<1>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis, Axis> {}} == Dimensions<3>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}} < Dimensions<3>{}));
  EXPECT_TRUE((StaticDescriptor<Axis, Axis> {} < DynamicDescriptor<double> {Dimensions<4>{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<4>{}} > StaticDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<4>{}} != StaticDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((StaticDescriptor<> {} == DynamicDescriptor<double> {StaticDescriptor<> {}}));

  EXPECT_TRUE((DynamicDescriptor<double> {Axis {}} == Dimensions{1}));
  EXPECT_TRUE((Dimensions{1} == DynamicDescriptor<double> {Axis {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis> {}} == Dimensions{1}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis, Axis> {}} == Dimensions{3}));
  EXPECT_TRUE((DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}} < Dimensions{4}));
  EXPECT_TRUE((StaticDescriptor<Axis, Axis> {} < DynamicDescriptor<double> {Dimensions{4}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions{4}} > StaticDescriptor<Axis, Axis> {}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions{2}} < StaticDescriptor<Axis, Axis, Axis, Axis> {}));
  EXPECT_TRUE((Dimensions{4} != DynamicDescriptor<double> {StaticDescriptor<Axis, Axis> {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Radians {}} == angle::Radians {}));
  EXPECT_TRUE((DynamicDescriptor<double> {angle::Degrees {}} == StaticDescriptor<angle::Degrees> {}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}, angle::Degrees{}} == StaticDescriptor<Dimensions<3>, angle::Degrees>{}));
  EXPECT_TRUE((angle::Radians{} == DynamicDescriptor<double> {angle::Radians {}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}, angle::Degrees{}} > Dimensions<3>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}} < StaticDescriptor<Dimensions<3>, angle::Degrees>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}} == StaticDescriptor<Dimensions<3>, angle::Degrees, Dimensions<5>>{}));
  EXPECT_TRUE((StaticDescriptor<Dimensions<3>, angle::Degrees, Dimensions<5>>{} == DynamicDescriptor<double> {Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} == StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} == StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} < StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<4>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} > StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<2>>{}));
  EXPECT_TRUE((DynamicDescriptor<double> {Axis{}, Dimensions<3>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} != StaticDescriptor<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}));
  EXPECT_FALSE((DynamicDescriptor<double> {Axis{}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}} < StaticDescriptor<Dimensions<4>, angle::Degrees, Dimensions<3>, Dimensions<3>>{}));

  EXPECT_TRUE(DynamicDescriptor<float>{StaticDescriptor<> {}} == DynamicDescriptor<double>{StaticDescriptor<> {}});
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

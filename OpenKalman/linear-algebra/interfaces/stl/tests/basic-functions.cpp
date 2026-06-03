/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "patterns/patterns.hpp"
#include "linear-algebra/functions/copy_from.hpp"
#include "linear-algebra/tests/tests.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/array-object.hpp"

using namespace OpenKalman;

using namespace OpenKalman;
using namespace OpenKalman::test;
using namespace OpenKalman::patterns;

using stdex::numbers::pi;

namespace
{
  using N0 = std::integral_constant<std::size_t, 0>;
  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
  using N4 = std::integral_constant<std::size_t, 3>;

  using F0 = values::fixed_value<double, 0>;
  using F1 = values::fixed_value<double, 1>;
  using F2 = values::fixed_value<double, 2>;
  using F3 = values::fixed_value<double, 3>;
  using F4 = values::fixed_value<double, 4>;
  using F5 = values::fixed_value<double, 5>;
}

#include "linear-algebra/traits/pattern_collection_type_of.hpp"
#include "linear-algebra/functions/attach_patterns.hpp"
#include "linear-algebra/traits/get_index_pattern.hpp"

TEST(stl_interfaces, attach_patterns)
{
  using namespace patterns;
  using M23 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>>;
  using A23 = double[2][3];
  A23 a23 {{1, 2, 3}, {4, 5, 6}};
  M23 m23 = get_mdspan(a23);
  static_assert(stdex::same_as<pattern_collection_type_of_t<std::decay_t<decltype(attach_patterns(m23, std::tuple{Polar{}, Dimensions<3>{}}))>>, std::tuple<Polar<>, Dimensions<3>>>);

  auto p23 = attach_patterns(m23, std::tuple<Polar<>, Spherical<>>{});
  static_assert(stdex::same_as<decltype(get_index_pattern<0>(p23)), Polar<>>);
  static_assert(stdex::same_as<decltype(get_index_pattern<1>(p23)), Spherical<>>);
  static_assert(compares_with_pattern_collection<decltype(p23), std::tuple<Polar<>, Spherical<>>, &stdex::is_eq, applicability::guaranteed>);
  static_assert(pattern_collection_for<std::array<Any<>, 2>, decltype(p23)>);
  static_assert(not pattern_collection_for<std::array<Any<>, 2>, decltype(p23), applicability::guaranteed>);

  auto q23 = attach_patterns(m23, std::array{Any{Polar{}}, Any{Spherical{}}});
  EXPECT_TRUE(compare_pattern_collections(get_pattern_collection(q23), std::tuple{Polar{}, Spherical{}}));
  static_assert(stdex::same_as<decltype(q23.nested_object()), M23&>);
  auto r23 = pattern_adapter {q23, std::tuple{Polar{}, Dimensions<3>{}, Dimensions<1>{}}};
  EXPECT_TRUE(compare_pattern_collections(get_pattern_collection(r23), std::tuple{Polar{}, Dimensions<3>{}, Dimensions<1>{}}));
  static_assert(stdex::same_as<decltype(r23.nested_object()), decltype(q23)&>);
  auto s23 = attach_patterns(r23, std::tuple{Polar{}, Dimensions<3>{}});
  EXPECT_TRUE(compare_pattern_collections(get_pattern_collection(s23), std::tuple{Polar{}, Dimensions<3>{}}));
  static_assert(stdex::same_as<decltype(s23.nested_object()), decltype(q23)&>);

  using M213 = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
  auto a213 = std::array<double, 6>{};
  M213 m213 {a213.data(), 2, 1, 3};
  auto p213 = attach_patterns(m213, std::array{Any{Polar{}}, Any{Distance{}}, Any{Spherical{}}});
  EXPECT_TRUE(compare_pattern_collections(get_pattern_collection(p213), std::tuple{Polar{}, Distance{}, Spherical{}}));
}

#include "linear-algebra/concepts/copyable_from.hpp"
#include "linear-algebra/functions/copy_from.hpp"

TEST(stl_interfaces, copy_from)
{
  using M23 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>>;
  using M23const = stdex::mdspan<const double, stdex::extents<std::size_t, 2, 3>>;
  using M23float = stdex::mdspan<float, stdex::extents<std::size_t, 2, 3>>;
  static_assert(copyable_from<M23&, M23>);
  static_assert(copyable_from<const M23&, M23>);
  static_assert(copyable_from<M23&&, M23>);
  static_assert(copyable_from<const M23&&, M23>);
  static_assert(copyable_from<M23&, M23const>);
  static_assert(copyable_from<M23float&, M23>);
  static_assert(copyable_from<M23&, M23float>);
  static_assert(not copyable_from<M23const&, M23>);
  static_assert(not copyable_from<M23const&, M23const>);

  using A23 = double[2][3];
  A23 a23_1etc {{1, 2, 3}, {4, 5, 6}};
  const M23const m23_1etc = get_mdspan(a23_1etc);
  auto a23_7etc = std::array{7., 8., 9., 10., 11., 12.};
  const M23const m23_7etc {a23_7etc.data()};

  auto a23a = std::array{1., 2., 3., 4., 5., 6.};
  M23 m23a {a23a.data()};
  EXPECT_TRUE(is_near(m23a, m23_1etc));
  EXPECT_FALSE(is_near(m23a, m23_7etc));
  copy_from(m23a, m23_7etc);
  EXPECT_TRUE(is_near(m23a, m23_7etc));
  EXPECT_FALSE(is_near(m23a, m23_1etc));
  copy_from(m23a, m23_1etc);
  EXPECT_FALSE(is_near(m23a, m23_7etc));
  EXPECT_TRUE(is_near(m23a, m23_1etc));

  auto a23_7etc_float = std::array<float, 6>{7, 8, 9, 10, 11, 12};
  M23float m23_7etc_float {a23_7etc_float.data()};
  copy_from(m23a, m23_7etc_float);
  EXPECT_TRUE(is_near(m23a, m23_7etc));

  auto q23a = attach_patterns(m23a, std::array{Any{Polar{}}, Any{Spherical{}}});
  EXPECT_TRUE(is_near(q23a, m23_7etc));
  EXPECT_FALSE(is_near(q23a, m23_1etc));
  copy_from(q23a, m23_1etc);
  EXPECT_TRUE(is_near(q23a, m23_1etc));
  EXPECT_FALSE(is_near(q23a, m23_7etc));
  copy_from(q23a, m23_7etc);
  EXPECT_FALSE(is_near(q23a, m23_1etc));
  EXPECT_TRUE(is_near(q23a, m23_7etc));
}

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Properties of collection in the c++ standard library.
 */

#include <tuple>
#include <array>
#include <vector>
#include <initializer_list>
#include "basics/tests/tests.hpp"
#include "basics/basics.hpp"

using namespace OpenKalman;

namespace C1
{
  struct C1
  {
    mutable int n1 = 4;
  };

  template<std::size_t i>
  constexpr auto& get(const C1& c1) { return c1.n1 += i; }
}

namespace std
{
  template<> struct tuple_size<C1::C1> : std::integral_constant<std::size_t, 7> {};
  template<size_t i> struct tuple_element<i, C1::C1> { using type = int; };
}


struct C2
{
  mutable std::size_t x = 7_uz;

  template<std::size_t i>
  constexpr auto& get() const { return x += i; }
};

namespace std
{
  template<> struct tuple_size<C2> : std::integral_constant<std::size_t, 7> {};
  template<size_t i> struct tuple_element<i, C2> { using type = std::size_t; };
}


namespace C3
{
  struct C3
  {
    mutable int n1 = 4;
  };

  template<std::size_t i>
  constexpr auto& get(const C3& c3) { return c3.n1 += i; }
}

namespace std
{
  template<> struct tuple_size<C3::C3> : std::integral_constant<std::size_t, 7> {};
}


struct C4
{
  mutable std::size_t x = 7_uz;

  template<std::size_t i>
  constexpr auto& get() const { return x += i; }
};

namespace std
{
  template<> struct tuple_size<C4> : std::integral_constant<std::size_t, 7> {};
}


#include "collections/traits/collection_element.hpp"
using namespace OpenKalman::collections;

TEST(collections, collection_element)
{
  static_assert(stdex::same_as<collection_element_t<0, std::tuple<double, int&>>, double>);
  static_assert(stdex::same_as<collection_element_t<1, std::tuple<double, int&>>, int&>);
  static_assert(stdex::same_as<collection_element_t<0, std::array<double, 5>>, double>);
  static_assert(stdex::same_as<collection_element_t<4, std::array<double, 5>>, double>);
  static_assert(stdex::same_as<collection_element_t<0, C1::C1>, int>);
  static_assert(stdex::same_as<collection_element_t<0, C2>, std::size_t>);
  static_assert(stdex::same_as<collection_element_t<0, C3::C3>, int&>);
  static_assert(stdex::same_as<collection_element_t<0, C4>, std::size_t&>);
}


#include "collections/concepts/gettable.hpp"

TEST(collections, gettable)
{
  static_assert(gettable<0, std::tuple<double, int>>);
  static_assert(gettable<0, std::array<double, 5>>);
  static_assert(gettable<0, C1::C1>);
  static_assert(gettable<0, C2>);
}


#include "collections/concepts/uniformly_gettable.hpp"

TEST(collections, uniformly_gettable)
{
  static_assert(uniformly_gettable<std::tuple<double, int>>);
  static_assert(uniformly_gettable<std::array<double, 5>>);
  static_assert(uniformly_gettable<C1::C1>);
  static_assert(uniformly_gettable<C2>);
}


#include "collections/concepts/tuple_like.hpp"

TEST(collections, tuple_like)
{
  static_assert(tuple_like<std::tuple<double, int>>);
  static_assert(tuple_like<std::array<double, 5>>);
  static_assert(tuple_like<C1::C1>);
  static_assert(tuple_like<C2>);
}


#include "collections/concepts/sized.hpp"

TEST(collections, size_random_access_range)
{
  static_assert(sized<int[5]>);
  static_assert(not sized<int*>);
  static_assert(sized<std::tuple<int, double, long double>>);
  static_assert(sized<std::array<double, 5>>);
  static_assert(sized<std::vector<double>>);
  static_assert(sized<std::initializer_list<double>>);
}


#include "collections/concepts/collection.hpp"

TEST(collections, collection)
{
  static_assert(collection<std::tuple<int, double, long double>>);
  static_assert(collection<std::array<double, 5>>);
  static_assert(collection<std::vector<double>>);
  static_assert(collection<std::initializer_list<double>>);
}


#include "collections/concepts/settable.hpp"

TEST(collections, settable)
{
  static_assert(settable<0, std::tuple<double, int>, double>);
  static_assert(settable<1, std::tuple<double, int>, int>);
  static_assert(settable<0, std::array<double, 5>, double>);
  static_assert(settable<0, C1::C1, int>);
  static_assert(settable<0, C2, std::size_t>);
}


#include "collections/concepts/uniformly_settable.hpp"

TEST(collections, uniformly_settable)
{
  static_assert(uniformly_settable<std::tuple<double, int>, int>);
  static_assert(uniformly_settable<std::tuple<double, int>, double>);
  static_assert(uniformly_settable<std::array<double, 5>, double>);
  static_assert(uniformly_settable<C1::C1, int>);
  static_assert(uniformly_settable<C2, std::size_t>);
}


#include "collections/concepts/output_collection.hpp"

TEST(collections, output_collection)
{
  static_assert(output_collection<std::tuple<int, double, long double>, double>);
  static_assert(output_collection<std::tuple<int, double, long double>, long double>);
  static_assert(output_collection<std::array<double, 5>, double>);
  static_assert(output_collection<std::array<double, 5>, int>);
  static_assert(output_collection<std::vector<double>, double>);
  static_assert(not output_collection<std::initializer_list<double>, double>);
}


#include "collections/concepts/invocable_on_collection.hpp"

TEST(collections, invocable_on_collection)
{
  static_assert(invocable_on_collection<std::negate<>, std::tuple<int, double, long double>>);
  static_assert(invocable_on_collection<std::negate<>, std::array<double, 5>>);
  static_assert(invocable_on_collection<std::negate<>, std::vector<double>>);
  static_assert(invocable_on_collection<std::negate<>, std::initializer_list<double>>);
  static_assert(not invocable_on_collection<std::negate<>, std::array<std::monostate, 5>>);
  static_assert(not invocable_on_collection<std::negate<>, std::vector<std::monostate>>);
}


#include "collections/concepts/index.hpp"

TEST(collections, index_collection)
{
  using Tup1 = std::tuple<std::size_t, unsigned, unsigned long>;
  static_assert(collections::index<Tup1>);
  static_assert(not collections::index<std::tuple<std::size_t, unsigned, int>>);

  using Range1 = std::vector<std::size_t>;
  static_assert(collections::index<Range1>);
  using Range2 = std::array<std::size_t, 5>;
  static_assert(collections::index<Range2>);
}


#include "collections/functions/apply.hpp"

TEST(collections, apply)
{
  static_assert(collections::apply(std::equal_to{}, std::tuple{4, 2+2}));
  static_assert(collections::apply(std::equal_to{}, std::array{4, 2+2}));
}


#include "collections/traits/std-extents.hpp"
#include "collections/concepts/viewable_collection.hpp"

TEST(collections, extents)
{
  static_assert(collections::get_element(stdex::extents<std::size_t, 2, 3, 4>{}, std::integral_constant<std::size_t, 0>{}) == 2);
  static_assert(collections::get_element(stdex::extents<std::size_t, 2, 3, 4>{}, 1U) == 3);
  static_assert(collections::get_element(stdex::extents<std::size_t, stdex::dynamic_extent, 3, 4>{2}, std::integral_constant<std::size_t, 0>{}) == 2);
  static_assert(collections::get_element(stdex::extents<std::size_t, 2, stdex::dynamic_extent, 4>{3}, 1U) == 3);
  static_assert(collections::get<0>(stdex::extents<std::size_t, 2, 3, 4>{}) == 2);
  static_assert(collections::get<1>(stdex::extents<std::size_t, 2, stdex::dynamic_extent, 4>{3}) == 3);
  static_assert(uniformly_gettable<stdex::extents<std::size_t, 2, 3, 4>>);
  static_assert(collection<stdex::extents<std::size_t, 2, 3, 4>>);
  static_assert(collection<stdex::extents<std::size_t, 2, 3, stdex::dynamic_extent>>);
  static_assert(viewable_tuple_like<stdex::extents<std::size_t, 2, 3, 4>>);
  static_assert(viewable_collection<stdex::extents<std::size_t, 2, 3, 4>>);
  static_assert(viewable_collection<stdex::extents<std::size_t, 2, stdex::dynamic_extent, 4>>);
}

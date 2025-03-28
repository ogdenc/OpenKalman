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
using namespace OpenKalman::collections;

#include "collections/concepts/tuple_like.hpp"

namespace C1
{
  struct C1
  {
    int n1 = 4;
  };

  template<std::size_t i>
  constexpr auto get(const C1& c1) { return c1.n1 + i; }
}

namespace std
{
  template<> struct tuple_size<C1::C1> : std::integral_constant<std::size_t, 7> {};
  template<size_t i> struct tuple_element<i, C1::C1> { using type = int; };
}


struct C2
{
  template<std::size_t i>
  constexpr auto get() const { return 7_uz + i; }
};

namespace std
{
  template<> struct tuple_size<C2> : std::integral_constant<std::size_t, 7> {};
  template<size_t i> struct tuple_element<i, C2> { using type = std::size_t; };
}


TEST(collections, tuple_like)
{
  static_assert(tuple_like<std::tuple<double, int>>);
  static_assert(tuple_like<std::array<double, 5>>);
  static_assert(tuple_like<C1::C1>);
  static_assert(tuple_like<C2>);
}

#include "collections/concepts/sized_random_access_range.hpp"

TEST(collections, size_random_access_range)
{
  static_assert(sized_random_access_range<int[5]>);
  static_assert(not sized_random_access_range<int*>);
  static_assert(not sized_random_access_range<std::tuple<int, double, long double>>);
  static_assert(sized_random_access_range<std::array<double, 5>>);
  static_assert(sized_random_access_range<std::vector<double>>);
  static_assert(sized_random_access_range<std::initializer_list<double>>);
}


#include "collections/concepts/collection.hpp"

TEST(collections, collections)
{
  static_assert(collection<std::tuple<int, double, long double>>);
  static_assert(collection<std::array<double, 5>>);
  static_assert(collection<std::vector<double>>);
  static_assert(collection<std::initializer_list<double>>);
}


#include "collections/traits/size_of.hpp"

TEST(collections, collections_size_of)
{
  static_assert(size_of_v<std::tuple<int, double, long double>> == 3);
  static_assert(size_of_v<std::array<double, 5>> == 5);
  static_assert(size_of_v<std::vector<double>> == dynamic_size);
  static_assert(size_of_v<std::initializer_list<double>> == dynamic_size);
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

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
 * \brief Tests for scalar types and constexpr math functions.
 */

#include "values/tests/tests.hpp"
#include "values/concepts/fixed.hpp"
#include "values/concepts/dynamic.hpp"
#include "values/traits/value_type_of.hpp"
#include "values/traits/real_type_of.hpp"
#include "values/concepts/integral.hpp"
#include "values/concepts/index.hpp"
#include "values/concepts/size.hpp"

using namespace OpenKalman;

TEST(values, integral)
{
  static_assert(values::index<std::integral_constant<int, 2>>);
  static_assert(values::fixed<std::integral_constant<int, 2>>);
  static_assert(not values::dynamic<std::integral_constant<int, 2>>);
  static_assert(not values::dynamic<std::integral_constant<int, 2>>);
  static_assert(std::is_same_v<values::value_type_of_t<std::integral_constant<int, 2>>, int>);
  static_assert(std::is_same_v<values::value_type_of_t<std::integral_constant<std::size_t, 2>>, std::size_t>);
  static_assert(std::is_same_v<values::value_type_of_t<values::real_type_of_t<std::integral_constant<std::size_t, 2>>>, double>);

  static_assert(values::index<std::size_t>);
  static_assert(not values::fixed<std::size_t>);
  static_assert(values::dynamic<std::size_t>);
  static_assert(std::is_same_v<values::value_type_of_t<std::size_t>, std::size_t>);
  static_assert(std::is_same_v<values::real_type_of_t<int>, double>);
  static_assert(std::is_same_v<values::real_type_of_t<std::size_t>, double>);

  static_assert(not values::index<int>);
  static_assert(not values::index<void>);
  static_assert(values::integral<int>);
  static_assert(not values::fixed<int>);
  static_assert(values::dynamic<int>);

  static_assert(values::size<unsigned>);
  static_assert(values::size<std::size_t>);
  static_assert(values::size<values::unbounded_size_t>);
  static_assert(not values::size<int>);
}

#include "values/concepts/size_compares_with.hpp"

TEST(values, size_compares_with)
{
  static_assert(values::size_compares_with<std::integral_constant<int, 7>, std::integral_constant<int, 7>>);
  static_assert(values::size_compares_with<std::integral_constant<int, 6>, std::integral_constant<int, 7>, &stdex::is_lt>);
  static_assert(values::size_compares_with<std::integral_constant<int, 7>, std::integral_constant<int, 6>, &stdex::is_gt>);

  static_assert(not values::size_compares_with<std::integral_constant<int, 7>, std::size_t, &stdex::is_eq>);
  static_assert(not values::size_compares_with<std::integral_constant<int, 7>, std::size_t, &stdex::is_neq>);
  static_assert(not values::size_compares_with<std::size_t, std::integral_constant<int, 7>, &stdex::is_lt>);
  static_assert(not values::size_compares_with<std::size_t, std::integral_constant<int, 7>, &stdex::is_gteq>);
  static_assert(not values::size_compares_with<std::integral_constant<int, 7>, std::size_t, &stdex::is_gt>);
  static_assert(not values::size_compares_with<std::integral_constant<int, 7>, std::size_t, &stdex::is_lteq>);
  static_assert(values::size_compares_with<std::integral_constant<int, 7>, std::integral_constant<int, 6>, &stdex::is_gteq>);
  static_assert(not values::size_compares_with<std::integral_constant<int, 7>, std::integral_constant<int, 6>, &stdex::is_lt>);
  static_assert(values::size_compares_with<std::integral_constant<int, 6>, std::integral_constant<int, 7>, &stdex::is_lteq>);
  static_assert(not values::size_compares_with<std::integral_constant<int, 6>, std::integral_constant<int, 7>, &stdex::is_gt>);

  static_assert(values::size_compares_with<std::integral_constant<int, 0>, std::size_t, &stdex::is_lteq>);
  static_assert(not values::size_compares_with<std::integral_constant<int, 0>, std::size_t, &stdex::is_gt, applicability::permitted>);
  static_assert(values::size_compares_with<std::size_t, std::integral_constant<int, 0>, &stdex::is_gteq>);
  static_assert(not values::size_compares_with<std::size_t, std::integral_constant<int, 0>, &stdex::is_lt, applicability::permitted>);

  static_assert(values::size_compares_with<values::unbounded_size_t, values::unbounded_size_t>);
  static_assert(not values::size_compares_with<std::integral_constant<int, 7>, values::unbounded_size_t, &stdex::is_eq>);
  static_assert(values::size_compares_with<std::integral_constant<int, 7>, values::unbounded_size_t, &stdex::is_neq>);

  static_assert(not values::size_compares_with<values::unbounded_size_t, std::integral_constant<int, 7>, &stdex::is_eq>);
  static_assert(values::size_compares_with<values::unbounded_size_t, std::integral_constant<int, 7>, &stdex::is_neq>);
  static_assert(not values::size_compares_with<std::size_t, values::unbounded_size_t, &stdex::is_eq>);
  static_assert(values::size_compares_with<std::size_t, values::unbounded_size_t, &stdex::is_neq>);
  static_assert(not values::size_compares_with<values::unbounded_size_t, std::size_t, &stdex::is_eq>);
  static_assert(values::size_compares_with<values::unbounded_size_t, std::size_t, &stdex::is_neq>);
}


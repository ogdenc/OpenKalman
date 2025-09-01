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
  static_assert(values::size<stdcompat::unreachable_sentinel_t>);
  static_assert(not values::size<int>);
}


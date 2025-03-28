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
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/concepts/integral.hpp"
#include "values/concepts/index.hpp"

using namespace OpenKalman;

TEST(values, integral)
{
  static_assert(value::index<std::integral_constant<int, 2>>);
  static_assert(value::fixed<std::integral_constant<int, 2>>);
  static_assert(not value::dynamic<std::integral_constant<int, 2>>);
  static_assert(std::is_same_v<value::number_type_of_t<std::integral_constant<int, 2>>, int>);
  static_assert(std::is_same_v<value::number_type_of_t<std::integral_constant<std::size_t, 2>>, std::size_t>);
  static_assert(std::is_same_v<value::number_type_of_t<value::real_type_of_t<std::integral_constant<std::size_t, 2>>>, double>);

  static_assert(value::index<std::size_t>);
  static_assert(not value::fixed<std::size_t>);
  static_assert(value::dynamic<std::size_t>);
  static_assert(std::is_same_v<value::number_type_of_t<std::size_t>, std::size_t>);
  static_assert(std::is_same_v<value::real_type_of_t<int>, double>);
  static_assert(std::is_same_v<value::real_type_of_t<std::size_t>, double>);

  static_assert(not value::index<int>);
  static_assert(value::integral<int>);
  static_assert(not value::fixed<int>);
  static_assert(value::dynamic<int>);
}


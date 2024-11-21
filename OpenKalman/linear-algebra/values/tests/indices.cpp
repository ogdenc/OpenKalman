/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for scalar types and constexpr math functions.
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/values/concepts/static_index.hpp"
#include "linear-algebra/values/concepts/dynamic_index.hpp"
#include "linear-algebra/values/concepts/index.hpp"

using namespace OpenKalman;



TEST(values, index_values)
{
  static_assert(value::static_index<std::integral_constant<int, 2>>);
  static_assert(not value::static_index<std::size_t>);
  static_assert(value::dynamic_index<int>);
  static_assert(value::dynamic_index<std::size_t>);
  static_assert(not value::dynamic_index<std::integral_constant<int, 2>>);
  static_assert(value::index<std::integral_constant<int, 2>>);
  static_assert(value::index<std::size_t>);
}



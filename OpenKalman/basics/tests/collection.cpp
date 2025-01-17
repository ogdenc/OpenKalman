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

#include <tuple>
#include <array>
#include <vector>
#include <initializer_list>
#include "basics/tests/tests.hpp"


using namespace OpenKalman;

#include "basics/internal/collection.hpp"

TEST(values, collections)
{
  static_assert(not internal::sized_random_access_range<std::tuple<int, double, long double>>);
  static_assert(internal::sized_random_access_range<std::array<double, 5>>);
  static_assert(internal::sized_random_access_range<std::vector<double>>);
  static_assert(internal::sized_random_access_range<std::initializer_list<double>>);

  static_assert(internal::collection<std::tuple<int, double, long double>>);
  static_assert(internal::collection<std::array<double, 5>>);
  static_assert(internal::collection<std::vector<double>>);
  static_assert(internal::collection<std::initializer_list<double>>);
}

#include "basics/internal/static_collection_size.hpp"

TEST(values, static_collection_size)
{
  static_assert(internal::static_collection_size_v<std::tuple<int, double, long double>> == 3);
  static_assert(internal::static_collection_size_v<std::array<double, 5>> == 5);
  static_assert(internal::static_collection_size_v<std::vector<double>> == dynamic_size);
  static_assert(internal::static_collection_size_v<std::initializer_list<double>> == dynamic_size);
}



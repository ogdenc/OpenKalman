/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for utils.hpp
 */

#include <array>
#include <gtest/gtest.h>
#include "basics/utils.hpp"

using namespace OpenKalman;


TEST(basics, tuple_slice)
{
  std::tuple t {1, "c", 5.0, 6.0};
  static_assert(std::is_same_v<decltype(internal::tuple_slice<0, 0>(t)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<0, 1>(t)), std::tuple<int&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 2>(t)), std::tuple<const char*&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 2>(t)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 3>(t)), std::tuple<double&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<3, 4>(t)), std::tuple<double&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(t)), std::tuple<const char*&, double&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 4>(t)), std::tuple<double&, double&>>);

  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(std::tuple {1, "c", 5.0, 6.0})), std::tuple<const char*, double>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 4>(std::tuple {1, "c", 5.0, 6.0})), std::tuple<double, double>>);

  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(std::forward_as_tuple(1, "c", 5.0, 6.0))), std::tuple<const char*, double>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 4>(std::forward_as_tuple(1, "c", 5.0, 6.0))), std::tuple<double, double>>);

  std::array a {1, 2, 3, 4};
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(a)), std::tuple<int&, int&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(std::array {1, 2, 3, 4})), std::tuple<int, int>>);
}


TEST(basics, tuple_replicate)
{
  double d = 7.0;
  static_assert(std::is_same_v<decltype(internal::tuple_replicate<0>(d)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::tuple_replicate<1>(d)), std::tuple<double&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_replicate<4>(d)), std::tuple<double&, double&, double&, double&>>);

  static_assert(std::is_same_v<decltype(internal::tuple_replicate<0>(5.0)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::tuple_replicate<1>(5.0)), std::tuple<double>>);
  static_assert(std::is_same_v<decltype(internal::tuple_replicate<4>(5.0)), std::tuple<double, double, double, double>>);
}


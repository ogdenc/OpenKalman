/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for basic definitions
 */

#include <gtest/gtest.h>
#include "basics/basics.hpp"
#include "tests.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::test;


namespace
{
  using cdouble = std::complex<double>;
  struct C
  {
    static constexpr double value = 1;
  };
}


TEST(basics, global_definitions)
{
#ifdef __cpp_concepts
  static_assert(internal::is_constexpr_n_ary_function<std::plus<void>, C, C>::value);
  static_assert(not internal::is_constexpr_n_ary_function<std::plus<void>, double, double>::value);
#else
  static_assert(internal::is_constexpr_n_ary_function<std::plus<void>, void, C, C>::value);
  static_assert(not internal::is_constexpr_n_ary_function<std::plus<void>, void, double, double>::value);
#endif
}


TEST(basics, tuple_slice)
{
  std::tuple t {1, "c", 5.0, 6.0};
  static_assert(std::tuple_size_v<decltype(internal::tuple_slice<1, 1>(t))> == 0);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<0, 1>(t)), std::tuple<int&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 2>(t)), std::tuple<const char*&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 3>(t)), std::tuple<double&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<3, 4>(t)), std::tuple<double&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(t)), std::tuple<const char*&, double&>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 4>(t)), std::tuple<double&, double&>>);

  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(std::tuple {1, "c", 5.0, 6.0})), std::tuple<const char*&&, double&&>>);

  std::array a {1, 2, 3, 4};
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(a)), std::tuple<int&, int&>>);
}


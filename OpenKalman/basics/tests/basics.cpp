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



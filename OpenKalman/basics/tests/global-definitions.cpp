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
 * \brief Tests for global-definitions.hpp
 */

#include <gtest/gtest.h>
#include <complex>
#include "basics/global-definitions.hpp"

using namespace OpenKalman;


TEST(basics, Qualification)
{

  static_assert(Qualification::unqualified == not Qualification::depends_on_dynamic_shape);
  static_assert((not Qualification::unqualified) == Qualification::depends_on_dynamic_shape);

  static_assert((Qualification::unqualified and Qualification::unqualified) == Qualification::unqualified);
  static_assert((Qualification::unqualified and Qualification::depends_on_dynamic_shape) == Qualification::depends_on_dynamic_shape);
  static_assert((Qualification::depends_on_dynamic_shape and Qualification::unqualified) == Qualification::depends_on_dynamic_shape);
  static_assert((Qualification::depends_on_dynamic_shape and Qualification::depends_on_dynamic_shape) == Qualification::depends_on_dynamic_shape);

  static_assert((Qualification::unqualified or Qualification::unqualified) == Qualification::unqualified);
  static_assert((Qualification::unqualified or Qualification::depends_on_dynamic_shape) == Qualification::unqualified);
  static_assert((Qualification::depends_on_dynamic_shape or Qualification::unqualified) == Qualification::unqualified);
  static_assert((Qualification::depends_on_dynamic_shape or Qualification::depends_on_dynamic_shape) == Qualification::depends_on_dynamic_shape);
}


namespace
{
  using cdouble = std::complex<double>;
  struct C
  {
    static constexpr double value = 1;
  };
}


TEST(basics, constexpr_n_ary_function)
{
  static_assert(internal::constexpr_n_ary_function<std::plus<void>, C, C>);
  static_assert(not internal::constexpr_n_ary_function<std::plus<void>, double, double>);
}


TEST(basics, tuple_like)
{
  static_assert(internal::tuple_like<std::tuple<double, int>>);
  static_assert(internal::tuple_like<std::array<double, 5>>);
}


TEST(basics, remove_rvalue_reference)
{
  static_assert(std::is_same_v<internal::remove_rvalue_reference_t<double&&>, double>);
  static_assert(std::is_same_v<internal::remove_rvalue_reference_t<const double&&>, const double>);
  static_assert(std::is_same_v<internal::remove_rvalue_reference_t<double&>, double&>);
  static_assert(std::is_same_v<internal::remove_rvalue_reference_t<const double>, const double>);
}


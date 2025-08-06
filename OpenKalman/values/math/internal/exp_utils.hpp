/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \brief Utilities for exp and expm1 functions.
 */

#ifndef OPENKALMAN_VALUE_EXP_UTILS_HPP
#define OPENKALMAN_VALUE_EXP_UTILS_HPP

#include "basics/basics.hpp"

namespace OpenKalman::values::internal
{
  // Maclaurin series expansion
  template <typename T>
  constexpr T exp_impl(int i, const T& x, const T& sum, const T& term)
  {
    auto new_sum = sum + term;
    if (sum == new_sum) return sum;
    else return exp_impl(i + 1, x, new_sum, term * x / static_cast<T>(i + 1));
  }


  template <typename Return, typename X>
  constexpr Return integral_exp(const X& x)
  {
    constexpr auto e = stdcompat::numbers::e_v<Return>;
    if (x == X{0}) return Return{1};
    else if (x == X{1}) return e;
    else if (x < X{0}) return Return{1} / integral_exp<Return>(-x);
    else if (x % X{2} == X{1}) return e * integral_exp<Return>(x - X{1}); //< odd
    else { auto ehalf {integral_exp<Return>(x / X{2})}; return ehalf * ehalf; } //< even
  }


} // namespace OpenKalman::values::internal


#endif //OPENKALMAN_VALUE_EXP_UTILS_HPP

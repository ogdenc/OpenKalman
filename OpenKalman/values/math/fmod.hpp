/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \brief Definition for \ref values::fmod.
 */

#ifndef OPENKALMAN_VALUE_FMOD_HPP
#define OPENKALMAN_VALUE_FMOD_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/functions/operation.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isnan.hpp"
#include "values/math/isinf.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A constexpr function for fmod.
   * \details If the compiler offers a constexpr version of std::fmod, it will be called.
   * Otherwise, this function returns x - iquot * y, where iquot is x / y with its fractional part truncated.
   * \note This will cause an exception unless
   * std::numeric_limits<std::intmax_t>::lowest() <= iquot <= std::numeric_limits<std::intmax_t>::max().
   */
#ifdef __cpp_concepts
  template<value X, value Y> requires
    (not complex<number_type_of_t<X>>) and (not complex<number_type_of_t<Y>>) and
    (std::common_with<number_type_of_t<X>, number_type_of_t<Y>>)
  constexpr value auto fmod(const X& x, const Y& y)
#else
  template <typename X, typename Y, std::enable_if_t<value<X> and value<Y> and
    (not complex<number_type_of_t<X>>) and (not complex<number_type_of_t<Y>>), int> = 0>
  constexpr auto fmod(const X& x, const Y& y)
#endif
  {
    if constexpr (fixed<X> or fixed<Y>)
    {
      struct Op { constexpr auto operator()(const number_type_of_t<X>& x_, const number_type_of_t<Y>& y_) const { return values::fmod(x_, y_); } };
      return values::operation(Op{}, x, y);
    }
    else
    {
      using std::fmod;
      using Return = std::decay_t<decltype(fmod(x, y))>;
      struct Op { auto operator()(const X& x, const Y& y) { return fmod(x, y); } };
      if (internal::constexpr_callable<Op>(x, y)) return fmod(x, y);
      if (values::isnan(x) or values::isnan(y) or values::isinf(x) or y == 0) return values::internal::NaN<Return>();
      if (x == 0 or values::isinf(y)) return static_cast<Return>(x);
      auto iquot = static_cast<Return>(x) / static_cast<Return>(y);
      constexpr auto low = static_cast<Return>(std::numeric_limits<std::intmax_t>::lowest());
      constexpr auto high = static_cast<Return>(std::numeric_limits<std::intmax_t>::max());
      if (iquot < low or iquot > high) throw std::out_of_range("y / x is out of range");
      return static_cast<Return>(x - static_cast<std::intmax_t>(iquot) * y);
    }
  }

}


#endif

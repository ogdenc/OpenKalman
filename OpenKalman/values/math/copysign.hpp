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
 * \brief Definition for \ref values::copysign.
 */

#ifndef OPENKALMAN_VALUES_COPYSIGN_HPP
#define OPENKALMAN_VALUES_COPYSIGN_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/value_type_of.hpp"
#include "values/concepts/integral.hpp"
#include "values/functions/operation.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isnan.hpp"
#include "values/math/signbit.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A constexpr function for copysign.
   * \details If the compiler offers a constexpr version of std::copysign, it will be called.
   * Otherwise, this function determines the sign of sgn and potentially negates mag.
   * \note In most pre-c++23 compilations, this function will be inaccurate if the sgn argument is either -NaN or -0.0.
   * This is because prior to the c++23 standard library, there was no way to determine,
   * at compile time, the sign of either ±NaN or ±0.0.
   * \param mag The magnitude of the result.
   * \param sgn A value reflecting the sign of the result.
   */
#ifdef __cpp_concepts
  template<value Mag, value Sgn> requires
    (not complex<value_type_of_t<Mag>>) and (not complex<value_type_of_t<Sgn>>) and
    (std::common_with<value_type_of_t<Mag>, value_type_of_t<Sgn>>)
  constexpr value auto copysign(const Mag& mag, const Sgn& sgn)
#else
  template <typename Mag, typename Sgn, std::enable_if_t<value<Mag> and value<Sgn> and
    (not complex<value_type_of_t<Mag>>) and (not complex<value_type_of_t<Sgn>>), int> = 0>
  constexpr auto copysign(const Mag& mag, const Sgn& sgn)
#endif
  {
    if constexpr (fixed<Mag> or fixed<Sgn>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Mag>& m, const value_type_of_t<Sgn>& s) const { return values::copysign(m, s); } };
      return values::operation(Op{}, mag, sgn);
    }
    else
    {
      using std::copysign;
      using Return = std::decay_t<decltype(copysign(mag, sgn))>;
      struct Op { auto operator()(const Mag& mag, const Sgn& sgn) { return copysign(mag, sgn); } };
      if (internal::constexpr_callable<Op>(mag, sgn)) return copysign(mag, sgn);
      if constexpr (std::is_unsigned_v<Mag> and std::is_unsigned_v<Sgn>) return static_cast<Return>(mag);
      if constexpr (values::integral<Mag>) return values::signbit(sgn) == (mag < 0) ? static_cast<Return>(mag) : -static_cast<Return>(mag);
      if (values::isnan(mag)) return values::signbit(sgn) ? -values::internal::NaN<Return>() : values::internal::NaN<Return>();
      if constexpr (std::numeric_limits<Mag>::is_iec559) if (mag == 0) return values::signbit(sgn) ? static_cast<Return>(-0.0) : static_cast<Return>(0.0);
      return values::signbit(mag) == values::signbit(sgn) ? static_cast<Return>(mag) : -static_cast<Return>(mag);
    }
  }

}


#endif

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
 * \brief Definition for \ref value::copysign.
 */

#ifndef OPENKALMAN_VALUE_COPYSIGN_HPP
#define OPENKALMAN_VALUE_COPYSIGN_HPP

#include "linear-algebra/values/interface/number_traits.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "linear-algebra/values/concepts/integral.hpp"
#include "linear-algebra/values/classes/operation.hpp"
#include "linear-algebra/values/functions/internal/constexpr_callable.hpp"
#include "linear-algebra/values/functions/internal/NaN.hpp"
#include "linear-algebra/values/functions/isnan.hpp"
#include "linear-algebra/values/functions/signbit.hpp"

namespace OpenKalman::value
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
  template<value::value Mag, value::value Sgn> requires
    (not value::complex<value::number_type_of_t<Mag>>) and (not value::complex<value::number_type_of_t<Sgn>>) and
    (std::common_with<value::number_type_of_t<Mag>, value::number_type_of_t<Sgn>>)
  constexpr value::value auto copysign(const Mag& mag, const Sgn& sgn)
#else
  template <typename Mag, typename Sgn, std::enable_if_t<value::value<Mag> and value::value<Sgn> and
    (not value::complex<value::number_type_of_t<Mag>>) and (not value::complex<value::number_type_of_t<Sgn>>), int> = 0>
  constexpr auto copysign(const Mag& mag, const Sgn& sgn)
#endif
  {
    if constexpr (not value::number<Mag> or not value::number<Sgn>)
    {
      struct Op
      {
        constexpr auto operator()(const value::number_type_of_t<Mag>& m, const value::number_type_of_t<Sgn>& s) const
        { return value::copysign(m, s); }
      };
      return value::operation {Op{}, mag, sgn};
    }
    else
    {
      using std::copysign;
      using Return = std::decay_t<decltype(copysign(mag, sgn))>;
      struct Op { constexpr auto operator()(const Mag& mag, const Sgn& sgn) { return copysign(mag, sgn); } };
      if (internal::constexpr_callable<Op>(mag, sgn)) return copysign(mag, sgn);
      if constexpr (std::is_unsigned_v<Mag> and std::is_unsigned_v<Sgn>) return static_cast<Return>(mag);
      if constexpr (value::integral<Mag>) return value::signbit(sgn) == (mag < 0) ? static_cast<Return>(mag) : -static_cast<Return>(mag);
      if (value::isnan(mag)) return value::signbit(sgn) ? -value::internal::NaN<Return>() : value::internal::NaN<Return>();
      if constexpr (std::numeric_limits<Mag>::is_iec559) if (mag == 0) return value::signbit(sgn) ? static_cast<Return>(-0.0) : static_cast<Return>(0.0);
      return value::signbit(mag) == value::signbit(sgn) ? static_cast<Return>(mag) : -static_cast<Return>(mag);
    }
  }

} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_COPYSIGN_HPP

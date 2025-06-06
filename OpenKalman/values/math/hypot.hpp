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
 * \brief Definition for \ref values::hypot.
 */

#ifndef OPENKALMAN_VALUE_HYPOT_HPP
#define OPENKALMAN_VALUE_HYPOT_HPP

#include <algorithm>
#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/classes/operation.hpp"
#include "values/math/real.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/internal/infinity.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"
#include "values/math/signbit.hpp"
#include "values/math/sqrt.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A constexpr alternative to std::hypot.
   * \details This version can take one or more parameters (not limited to 2 or 3 as std::hypot is).
   */
#ifdef __cpp_concepts
  template<values::value...Args> requires (sizeof...(Args) > 0) and
    (... and (not values::complex<Args>))
  constexpr values::value auto hypot(const Args&...args)
#else
  template <typename...Args, std::enable_if_t<(... and values::value<Args>) and (sizeof...(Args) > 0) and
    (... and (not values::complex<Args>)), int> = 0>
  constexpr auto hypot(const Args&...args)
#endif
  {
    if constexpr ((... or (not values::number<Args>)))
    {
      struct Op { constexpr auto operator()(const values::number_type_of_t<Args>&...as) const { return values::hypot(as...); } };
      return values::operation {Op{}, args...};
    }
    else if constexpr (sizeof...(Args) == 1)
    {
      using std::abs;
      struct Op { auto operator()(const Args&...args) { return abs(args...); } };
      if (internal::constexpr_callable<Op>(args...)) return values::real(abs(args...));
      return values::real((..., (values::signbit(args...) ? -args : args)));
    }
    else
    {
      using std::hypot;
      using ArgCommon = std::common_type_t<Args...>;
      using Return = real_type_of_t<ArgCommon>;
      if constexpr (sizeof...(Args) == 2 or sizeof...(Args) == 3)
      {
        struct Op { auto operator()(const Args&...args) { return hypot(args...); } };
        if (internal::constexpr_callable<Op>(args...)) return static_cast<Return>(hypot(args...));
      }
      if ((... or values::isinf(args))) return values::internal::infinity<Return>();
      if ((... or values::isnan(args))) return values::internal::NaN<Return>();
      auto m = static_cast<Return>(std::max({static_cast<ArgCommon>(values::signbit(args) ? -args : args)...}));
      if (m == 0) return m;
      return m * values::sqrt((... + [](const auto& a){ return a * a; }(static_cast<Return>(args)/m)));
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_HYPOT_HPP

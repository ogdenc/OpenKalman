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
 * \brief Definition for \ref values::atan.
 */

#ifndef OPENKALMAN_VALUE_ATAN_HPP
#define OPENKALMAN_VALUE_ATAN_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/traits/real_type_of.hpp"
#include "values/functions/operation.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"
#include "values/math/copysign.hpp"
#include "internal/math_utils.hpp"
#include "internal/atan_utils.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::atan function.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto atan(const Arg& arg)
#else
  template <typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto atan(const Arg& arg)
#endif
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const number_type_of_t<Arg>& a) const { return values::atan(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::atan;
      using Return = std::decay_t<decltype(atan(arg))>;
      struct Op { auto operator()(const Arg& arg) { return atan(arg); } };
      if (values::internal::constexpr_callable<Op>(arg)) return atan(arg);
      else if constexpr (values::complex<Return>)
      {
        using R = real_type_of_t<real_type_of_t<Return>>;
        auto x = values::internal::make_complex_number<R>(arg);
        return internal::make_complex_number<Return>(internal::atan_impl_general(x));
      }
      else
      {
        if (values::isnan(arg)) return values::internal::NaN<Return>();
        if (values::isinf(arg)) return values::copysign(stdcompat::numbers::pi_v<Return> * static_cast<Return>(0.5), arg);
        if (arg == 0) return static_cast<Return>(arg);
        return internal::atan_impl(static_cast<Return>(arg));
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_ATAN_HPP

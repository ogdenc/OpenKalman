/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file \internal
 * \brief Definition for value::internal::constexpr_callable.
 */

#ifndef OPENKALMAN_VALUE_CONSTEXPR_CALLABLE_HPP
#define OPENKALMAN_VALUE_CONSTEXPR_CALLABLE_HPP

#include <type_traits>
#include "linear-algebra/values/concepts/number.hpp"

namespace OpenKalman::value::internal
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Op, typename = void, typename...Args>
    struct constexpr_callable_impl : std::false_type {};

    template<typename Op, typename...Args>
    struct constexpr_callable_impl<Op, std::enable_if_t<
      value::number<decltype(std::declval<Op>()(std::declval<Args>()...))> and
      std::bool_constant<(Op{}(Args{}...), true)>::value>, Args...> : std::true_type {};
  } // namespace detail
#endif


  template<typename Op, typename...Args>
  constexpr bool
  constexpr_callable(const Args&...args)
  {
#ifdef __cpp_concepts
    if constexpr (requires(Op op, const Args&...args) {
      {op(args...)} -> value::number;
      requires std::bool_constant<(Op{}(Args{}...), true)>::value;
    })
#else
    if constexpr (detail::constexpr_callable_impl<Op, void, Args...>::value)
#endif
    {
#ifdef __cpp_lib_is_constant_evaluated
      if (not std::is_constant_evaluated()) return true;
#endif
      return (... or [](const auto& a) {
        if (a != a) return true;
        else if constexpr (std::numeric_limits<Args>::has_infinity)
          return a == std::numeric_limits<Args>::infinity() or a == -std::numeric_limits<Args>::infinity();
        else
          return false;
        }(args));
    }
    else return false;
  }

} // namespace OpenKalman::value::internal


#endif //OPENKALMAN_VALUE_CONSTEXPR_CALLABLE_HPP

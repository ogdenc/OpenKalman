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
 * \file
 * \brief Definition for \values::fixed_value_compares_with.
 */

#ifndef OPENKALMAN_VALUES_FIXED_VALUE_COMPARES_WITH_HPP
#define OPENKALMAN_VALUES_FIXED_VALUE_COMPARES_WITH_HPP

#include "basics/basics.hpp"
#include "fixed.hpp"
#include "values/traits/fixed_value_of.hpp"
#include "values/functions/internal/near.hpp"

namespace OpenKalman::values
{
  namespace detail
  {
    template<auto N, unsigned int epsilon_factor>
    struct compare_three_way_near
    {
      template<typename T>
      constexpr stdex::partial_ordering
      operator() [[nodiscard]] (T&& t) const
      {
        constexpr auto n = static_cast<typename value_type_of<T>::type>(N);
        if constexpr (epsilon_factor == 0U)
        {
          return stdex::compare_three_way{}(std::forward<T>(t), n);
        }
        else
        {
          if (internal::near<epsilon_factor>(t, n)) return stdex::partial_ordering::equivalent;
          return stdex::compare_three_way{}(std::forward<T>(t), n);
        }
      }
    };


#if not defined(__cpp_concepts) or not defined(__cpp_impl_three_way_comparison)
    template<typename T, auto N, auto comp, unsigned int epsilon_factor, typename = void>
    struct fixed_value_compares_with_impl : std::false_type {};

    template<typename T, auto N, auto comp, unsigned int epsilon_factor>
    struct fixed_value_compares_with_impl<T, N, comp, epsilon_factor, std::enable_if_t<
      stdex::invoke(comp, detail::compare_three_way_near<N, epsilon_factor>{}(fixed_value_of<T>::value))>>
      : std::true_type {};
#endif
  }


  /**
   * \brief T has a fixed value that compares with N in a particular way based on parameter comp.
   * \tparam comp A consteval-callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   * \tparam epsilon_factor An epsilon value to account for rounding error.
   * This is multiplied by <code>std::numeric_limits&lt;std::decay_t&lt;T&rt;&rt;::epsilon()</code>, if it exists.
   * If it is zero, the match must be exact.
   */
  template<typename T, auto N, auto comp = &stdex::is_eq, unsigned int epsilon_factor = 0U>
#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  concept fixed_value_compares_with =
    fixed<T> and
    stdex::invoke(comp, detail::compare_three_way_near<N, epsilon_factor>{}(fixed_value_of_v<T>));
#else
  constexpr bool fixed_value_compares_with = detail::fixed_value_compares_with_impl<T, N, comp, epsilon_factor>::value;
#endif

}

#endif

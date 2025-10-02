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
 * \brief Definition for \values::size_compares_with.
 */

#ifndef OPENKALMAN_VALUES_SIZE_COMPARES_WITH_HPP
#define OPENKALMAN_VALUES_SIZE_COMPARES_WITH_HPP

#include "basics/basics.hpp"
#include "values/constants.hpp"
#include "fixed.hpp"
#include "size.hpp"
#include "values/traits/fixed_value_of.hpp"

namespace OpenKalman::values
{
#if not defined(__cpp_concepts) or not defined(__cpp_impl_three_way_comparison)
  namespace detail
  {
    template<typename T, typename U, auto comp, applicability a, typename = void>
    struct size_compares_with_impl1 : std::false_type {};

    template<typename T, typename U, auto comp, applicability a>
    struct size_compares_with_impl1<T, U, comp, a, std::enable_if_t<
      fixed_value_of<T>::value != dynamic_size and fixed_value_of<U>::value != dynamic_size>>
      : std::true_type {};


    template<typename T, typename U, auto comp, applicability a, typename = void>
    struct size_compares_with_impl2 : std::false_type {};

    template<typename T, typename U, auto comp, applicability a>
    struct size_compares_with_impl2<T, U, comp, a, std::enable_if_t<
      fixed_value_of<T>::value == dynamic_size or
      fixed_value_of<U>::value == dynamic_size or
      stdcompat::invoke(comp, stdcompat::compare_three_way{}(fixed_value_of<T>::value, fixed_value_of<U>::value))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T and U are sizes that compare in a particular way based on parameter comp.
   * \tparam comp A consteval-callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
  template<typename T, typename U, auto comp = &stdcompat::is_eq, applicability a = applicability::guaranteed>
#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  concept size_compares_with =
    size<T> and
    size<U> and
    (a != applicability::guaranteed or not index<T> or not index<U> or
      (fixed<T> and fixed<U> and fixed_value_of_v<T> != dynamic_size and fixed_value_of_v<U> != dynamic_size)) and
    (not fixed<T> or fixed_value_of_v<T> == dynamic_size or
      not fixed<U> or fixed_value_of_v<U> == dynamic_size or
      stdcompat::invoke(comp, fixed_value_of_v<T> <=> fixed_value_of_v<U>)) and
    (std::same_as<std::decay_t<T>, stdcompat::unreachable_sentinel_t> ==
      std::same_as<std::decay_t<U>, stdcompat::unreachable_sentinel_t>);
#else
  constexpr bool size_compares_with =
    size<T> and
    size<U> and
    (a != applicability::guaranteed or not index<T> or not index<U> or detail::size_compares_with_impl1<T, U, comp, a>::value) and
    (not fixed<T> or not fixed<U> or detail::size_compares_with_impl2<T, U, comp, a>::value) and
    (stdcompat::same_as<std::decay_t<T>, stdcompat::unreachable_sentinel_t> ==
      stdcompat::same_as<std::decay_t<U>, stdcompat::unreachable_sentinel_t>);
#endif

}

#endif

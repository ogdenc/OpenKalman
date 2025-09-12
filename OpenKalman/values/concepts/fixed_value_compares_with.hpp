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

namespace OpenKalman::values
{
#if not defined(__cpp_concepts) or not defined(__cpp_impl_three_way_comparison)
  namespace detail
  {
    template<typename T, auto N, auto comp, typename = void>
    struct fixed_value_compares_with_impl : std::false_type {};

    template<typename T, auto N, auto comp>
    struct fixed_value_compares_with_impl<T, N, comp, std::enable_if_t<
      (stdcompat::invoke(comp, stdcompat::compare_three_way{}(fixed_value_of<T>::value, static_cast<typename value_type_of<T>::type>(N))))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T has a fixed value that compares with N in a particular way based on parameter comp.
   * \tparam comp A consteval-callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
  template<typename T, auto N, auto comp = &stdcompat::is_eq>
#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  concept fixed_value_compares_with =
    fixed<T> and
    stdcompat::invoke(comp, fixed_value_of_v<T> <=> N);
#else
  constexpr bool fixed_value_compares_with = detail::fixed_value_compares_with_impl<T, N, comp>::value;
#endif

}

#endif

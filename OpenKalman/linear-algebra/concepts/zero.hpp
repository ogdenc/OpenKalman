/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref zero.
 */

#ifndef OPENKALMAN_ZERO_HPP
#define OPENKALMAN_ZERO_HPP

#include "values/values.hpp"
#include "linear-algebra/interfaces/object_traits.hpp"
#include "linear-algebra/traits/element_type_of.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, unsigned int epsilon_factor, typename = void>
    struct is_zero_constant : std::false_type {};

    template<typename T, unsigned int epsilon_factor>
    struct is_zero_constant<T, epsilon_factor, std::enable_if_t<
      values::fixed_value_compares_with<decltype(interface::object_traits<stdex::remove_cvref_t<T>>::
        get_constant(std::declval<T>())), 0, &stdex::is_eq, epsilon_factor>>>
      : std::true_type {};


    template<typename T, unsigned int epsilon_factor, typename = void>
    struct constant_element_is_0_ep : std::false_type {};

    template<typename T, unsigned int epsilon_factor>
    struct constant_element_is_0_ep<T, epsilon_factor, std::enable_if_t<
        values::fixed_value_compares_with<typename element_type_of<T>::type, 0, &stdex::is_eq, epsilon_factor>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is known at compile time to be a constant matrix of value zero.
   * \tparam epsilon_factor An epsilon value to account for rounding error.
   * This is multiplied by <code>std::numeric_limits&lt;std::decay_t&lt;T&rt;&rt;::epsilon()</code>, if it exists.
   * If it is zero, the match must be exact.
   */
  template<typename T, unsigned int epsilon_factor = 0>
#ifdef __cpp_concepts
  concept zero =
    indexible<T> and
    (values::fixed_value_compares_with<decltype(interface::object_traits<stdex::remove_cvref_t<T>>::
      get_constant(std::declval<T>())), 0, &std::is_eq, epsilon_factor> or
    values::fixed_value_compares_with<element_type_of_t<T>, 0, &std::is_eq, epsilon_factor>);
#else
  constexpr bool zero =
    detail::is_zero_constant<T, epsilon_factor>::value or
    detail::constant_element_is_0_ep<T, epsilon_factor>::value;
#endif


}

#endif

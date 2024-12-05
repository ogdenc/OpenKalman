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
 * \file
 * \brief Definition for \ref value::index.
 */

#ifndef OPENKALMAN_VALUE_INDEX_HPP
#define OPENKALMAN_VALUE_INDEX_HPP

#include <type_traits>
#ifdef __cpp_concepts
#include <concepts>
#endif
#include "linear-algebra/values/functions/to_number.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "integral.hpp"

namespace OpenKalman::value
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct fixed_integral_gt_0 : std::false_type {};


    template<typename T>
    struct fixed_integral_gt_0<T, std::enable_if_t<value::fixed<T>>>
      : std::bool_constant<static_cast<long int>(value::to_number(T{})) >= 0> {};


    template<typename T, typename = void>
    struct reduces_to_index : std::false_type {};


    template<typename T>
    struct reduces_to_index<T, std::enable_if_t<value::integral<T>>>
      : std::bool_constant<(std::is_unsigned_v<value::number_type_of_t<T>> or fixed_integral_gt_0<T>::value)> {};
  }
#endif


/**
   * \brief T is an index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept index = value::integral<T> and
    (std::is_unsigned_v<value::number_type_of_t<T>> or (fixed<T> and static_cast<long int>(value::to_number(T{})) >= 0 ));
#else
  template<typename T>
  constexpr bool index = detail::reduces_to_index<T>::value;
#endif

} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_INDEX_HPP

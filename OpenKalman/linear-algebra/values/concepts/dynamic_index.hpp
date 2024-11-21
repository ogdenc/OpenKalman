/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref value::dynamic_index.
 */

#ifndef OPENKALMAN_VALUE_DYNAMIC_INDEX_HPP
#define OPENKALMAN_VALUE_DYNAMIC_INDEX_HPP

#ifdef __cpp_concepts
#include <concepts>
#endif
#include <type_traits>
#include "static_index.hpp"


namespace OpenKalman::value
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_integral_scalar : std::false_type {};

    template<typename T>
    struct is_integral_scalar<T, std::enable_if_t<std::is_integral<typename std::invoke_result<T>::type>::value>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T is a dynamic index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept dynamic_index = (not value::static_index<T>) and
    (std::integral<std::decay_t<T>> or std::integral<std::decay_t<decltype(std::declval<T>()())>>);
#else
  template<typename T>
  constexpr bool dynamic_index = (not value::static_index<T>) and
    (std::is_integral_v<std::decay_t<T>> or detail::is_integral_scalar<T>::value);
#endif

} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_DYNAMIC_INDEX_HPP

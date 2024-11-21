/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref value::static_index.
 */

#ifndef OPENKALMAN_VALUE_STATIC_INDEX_HPP
#define OPENKALMAN_VALUE_STATIC_INDEX_HPP

#ifdef __cpp_concepts
#include <concepts>
#endif
#include <type_traits>
#include "basics/global-definitions.hpp"


namespace OpenKalman::value
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Z, typename = void>
    struct is_static_index_value : std::false_type {};

    template<typename T, typename Z>
    struct is_static_index_value<T, Z, std::enable_if_t<
      std::is_default_constructible_v<std::decay_t<T>> and std::is_convertible_v<T, Z> and
        std::is_convertible_v<decltype(std::decay_t<T>::value), Z> and std::is_convertible_v<decltype(std::decay_t<T>::value), long int>>>
      : std::bool_constant<(static_cast<long int>(std::decay_t<T>::value) >= 0) and
          (not std::is_same_v<decltype(std::decay_t<T>::value), std::size_t> or static_cast<std::size_t>(std::decay_t<T>::value) != dynamic_size)> {};
  }
#endif


  /**
   * \brief T is a static index value.
   * \tparam Z the type to which the index must be convertible.
   */
  template<typename T, typename Z = std::size_t>
#ifdef __cpp_concepts
  concept static_index =
    requires {
      {std::decay_t<T>::value} -> std::convertible_to<Z>;
      {std::decay_t<T>{}} -> std::convertible_to<Z>;
    } and (not std::is_signed_v<decltype(std::decay_t<T>::value)> or static_cast<long int>(std::decay_t<T>::value) >= 0) and
    (not std::same_as<decltype(std::decay_t<T>::value), std::size_t> or std::decay_t<T>::value != dynamic_size);
#else
  constexpr bool static_index = detail::is_static_index_value<T, Z>::value;
#endif


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_STATIC_INDEX_HPP

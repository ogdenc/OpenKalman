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
 * \brief Definition for \ref static_index_value.
 */

#ifndef OPENKALMAN_STATIC_INDEX_VALUE_HPP
#define OPENKALMAN_STATIC_INDEX_VALUE_HPP

#include <type_traits>

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Z, typename = void>
    struct is_static_index_value : std::false_type {};

    template<typename T, typename Z>
    struct is_static_index_value<T, Z, std::enable_if_t<std::is_convertible<decltype(std::decay_t<T>::value), const Z>::value>>
      : std::bool_constant<std::decay_t<T>::value >= 0 and static_cast<Z>(std::decay_t<T>{}) == static_cast<Z>(std::decay_t<T>::value)> {};
  }
#endif


  /**
   * \brief T is a static index value.
   * \tparam Z the type to which the index must be convertible.
   */
  template<typename T, typename Z = std::size_t>
#ifdef __cpp_concepts
  concept static_index_value = (std::decay_t<T>::value >= 0) and
    std::bool_constant<static_cast<Z>(std::decay_t<T>{}) == static_cast<Z>(std::decay_t<T>::value)>::value;
#else
  constexpr bool static_index_value = detail::is_static_index_value<T, Z>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_STATIC_INDEX_VALUE_HPP

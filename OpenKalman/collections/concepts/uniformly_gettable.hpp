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
 * \brief Definition for \ref collections::uniformly_gettable.
 */

#ifndef OPENKALMAN_COLLECTIONS_UNIFORMLY_GETTABLE_HPP
#define OPENKALMAN_COLLECTIONS_UNIFORMLY_GETTABLE_HPP

#include <type_traits>
#include "gettable.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::collections
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, typename = std::make_index_sequence<size_of_v<T>>, typename = void>
    struct uniformly_gettable_sized_impl : std::false_type {};

    template<typename T, std::size_t...i>
    struct uniformly_gettable_sized_impl<T, std::index_sequence<i...>, std::enable_if_t<(... and gettable<i, T>)>> : std::true_type {};


    template<typename T, typename = void>
    struct uniformly_gettable_sized : std::false_type {};

    template<typename T>
    struct uniformly_gettable_sized<T, std::enable_if_t<size_of<T>::value != dynamic_size>>
      : uniformly_gettable_sized_impl<T> {};

  } // namespace detail
#endif


  /**
   * \brief T is \ref gettable for all indices.
   * \details If T is not \ref sized, then it must be gettable for at least index 0 and std::numeric_limits<std::size_t>::max() - 1.
   */
  template<typename T>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept uniformly_gettable =
    ((sized<T> and size_of_v<T> != dynamic_size) or
      (gettable<0_uz, T> and gettable<std::numeric_limits<std::size_t>::max() - 1_uz, T>)) and
    (not sized<T> or size_of_v<T> == dynamic_size or
      []<std::size_t...i>(std::index_sequence<i...>) { return (... and gettable<i, T>); }
        (std::make_index_sequence<size_of_v<T>>{}));
#else
  constexpr bool uniformly_gettable =
    ((sized<T> and size_of_v<T> != dynamic_size) or
      (gettable<0_uz, T> and gettable<std::numeric_limits<std::size_t>::max() - 1_uz, T>)) and
    (not sized<T> or size_of_v<T> == dynamic_size or detail::uniformly_gettable_sized<T>::value);
#endif


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_UNIFORMLY_GETTABLE_HPP

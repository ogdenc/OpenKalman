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

#include "values/values.hpp"
#include "gettable.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::collections
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, typename = std::make_index_sequence<size_of_v<T>>>
    struct uniformly_gettable_sized_impl : std::false_type {};

    template<typename T, std::size_t...i>
    struct uniformly_gettable_sized_impl<T, std::index_sequence<i...>>
      : std::bool_constant<(... and gettable<i, T>)> {};


    template<typename T, typename = void>
    struct uniformly_gettable_sized : std::false_type {};

    template<typename T>
    struct uniformly_gettable_sized<T, std::enable_if_t<size_of<T>::value != dynamic_size>>
      : uniformly_gettable_sized_impl<T> {};
  }
#endif


  /**
   * \brief T is \ref gettable for all indices.
   * \details If T is not \ref sized or has dynamic size, then it must be gettable for at least index 0.
   */
  template<typename T>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept uniformly_gettable =
    ((sized<T> and size_of_v<T> != dynamic_size) or gettable<0_uz, T>) and
    (not sized<T> or size_of_v<T> == dynamic_size or
      []<std::size_t...i>(std::index_sequence<i...>) { return (... and gettable<i, T>); }
        (std::make_index_sequence<size_of_v<T>>{}));
#else
  inline constexpr bool uniformly_gettable =
    ((sized<T> and not values::fixed_number_compares_with<size_of<T>, dynamic_size>) or gettable<0_uz, T>) and
    (not sized<T> or values::fixed_number_compares_with<size_of<T>, dynamic_size> or
      detail::uniformly_gettable_sized<T>::value);
#endif


}

#endif

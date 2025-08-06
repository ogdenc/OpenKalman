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
 * \brief Definition for \ref collections::uniformly_settable.
 */

#ifndef OPENKALMAN_COLLECTIONS_UNIFORMLY_SETTABLE_HPP
#define OPENKALMAN_COLLECTIONS_UNIFORMLY_SETTABLE_HPP

#include "values/values.hpp"
#include "settable.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::collections
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename C, typename T, typename = std::make_index_sequence<size_of_v<C>>, typename = void>
    struct uniformly_settable_sized_impl : std::false_type {};

    template<typename C, typename T, std::size_t...i>
    struct uniformly_settable_sized_impl<C, T, std::index_sequence<i...>, std::enable_if_t<(... and settable<i, C, T>)>>
      : std::true_type {};


    template<typename C, typename T, typename = void>
    struct uniformly_settable_sized : std::false_type {};

    template<typename C, typename T>
    struct uniformly_settable_sized<C, T, std::enable_if_t<size_of<C>::value != dynamic_size>>
      : uniformly_settable_sized_impl<C, T> {};
  }
#endif


  /**
   * \brief C is \ref settable with type C for all indices.
   * \details If C is not \ref sized or has dynamic size, then it must be settable for at least index 0.
   */
  template<typename C, typename T>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept uniformly_settable =
    ((sized<C> and size_of_v<C> != dynamic_size) or settable<0_uz, C, T>) and
    (not sized<C> or size_of_v<C> == dynamic_size or
      []<std::size_t...i>(std::index_sequence<i...>) { return (... and settable<i, C, T>); }
        (std::make_index_sequence<size_of_v<C>>{}));
#else
  constexpr bool uniformly_settable =
    ((sized<C> and not values::fixed_number_compares_with<size_of<C>, dynamic_size>) or settable<0_uz, C, T>) and
    (not sized<C> or size_of_v<C> == dynamic_size or
      detail::uniformly_settable_sized<C, T>::value);
#endif


}

#endif 

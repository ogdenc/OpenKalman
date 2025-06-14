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
 * \brief Definition for \ref collections::settable.
 */

#ifndef OPENKALMAN_COLLECTIONS_SETTABLE_HPP
#define OPENKALMAN_COLLECTIONS_SETTABLE_HPP

#include <tuple>
#include "basics/compatibility/language-features.hpp"
#include "sized.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<std::size_t i, typename C, typename = void>
    struct settable_impl1 : std::false_type {};

    template<std::size_t i, typename C>
    struct settable_impl1<i, C, std::enable_if_t<sized<C>>> : std::bool_constant<(i < size_of_v<C>)> {};

    template<std::size_t i, typename C, typename T, typename = void>
    struct settable_impl2 : std::false_type {};

    template<std::size_t i, typename C, typename T>
    struct settable_impl2<i, C, T, std::void_t<
      decltype(internal::generalized_std_get<i>(std::declval<C&>()) = std::declval<T>())>> : std::true_type {};

  } // namespace detail
#endif


  /**
   * \brief C has an element i that can be set by assigning the result of a get<i>(...) function to an object of type T.
   * \details The get<i>(...) function can be std::get<i>(...), a get<i>(...) member function, or
   * a separately-defined matching get<i>(...) function in T's namespace.
   */
  template<std::size_t i, typename C, typename T>
#ifdef __cpp_concepts
  concept settable =
    (not sized<C> or i < size_of_v<C>) and
    requires(C& c, T t) { internal::generalized_std_get<i>(c) = t; };
#else
  constexpr bool gettable = detail::settable_impl1<i, C, T>::value and detail::settable_impl2<i, C, T>::value;
#endif


} // namespace OpenKalman::collections

#endif 

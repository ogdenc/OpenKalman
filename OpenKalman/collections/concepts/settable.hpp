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

#include "values/values.hpp"
#include "collections/functions/get.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<std::size_t i, typename C, typename T, typename = void>
    struct settable_impl : std::false_type {};

    template<std::size_t i, typename C, typename T>
    struct settable_impl<i, C, T,
      std::void_t<decltype(collections::get<i>(std::declval<C&>()) = std::declval<T>())>> : std::true_type {};
  }
#endif


  /**
   * \brief C has an element i that can be set by assigning the result of a get<i>(...) function to an object of type T.
   * \details The get<i>(...) function can be std::get<i>(...), a get<i>(...) member function, or
   * a separately-defined matching get<i>(...) function in T's namespace.
   */
  template<std::size_t i, typename C, typename T>
#ifdef __cpp_concepts
  concept settable = requires(C& c, T t) { collections::get<i>(c) = t; };
#else
  constexpr bool settable = detail::settable_impl<i, C, T>::value;
#endif


}

#endif 

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
 * \brief Definition for \ref collections::gettable.
 */

#ifndef OPENKALMAN_COLLECTIONS_GETTABLE_HPP
#define OPENKALMAN_COLLECTIONS_GETTABLE_HPP

#include <tuple>
#include "basics/compatibility/language-features.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<std::size_t i, typename T, typename = void>
    struct gettable_impl : std::false_type {};

    template<std::size_t i, typename T>
    struct gettable_impl<i, T, std::void_t<typename std::tuple_element<i, T>::type,
      decltype(internal::generalized_std_get<i>(std::declval<T&>()))>> : std::true_type {};

  } // namespace detail
#endif


  /**
   * \brief T has an element i that is accessible by a get<i>(...) function.
   * \details The get<i>(...) function can be std::get<i>(...), a get<i>(...) member function, or
   * a separately-defined matching get<i>(...) function in T's namespace.
   */
  template<std::size_t i, typename T>
#ifdef __cpp_concepts
  concept gettable =
    requires { typename std::tuple_element<i, std::decay_t<T>>::type;
               internal::generalized_std_get<i>(std::declval<T&>()); };
#else
  constexpr bool gettable = detail::gettable_impl<i, std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_GETTABLE_HPP

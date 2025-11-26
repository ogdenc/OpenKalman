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

#include "collections/functions/get.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<std::size_t i, typename T, typename = void>
    struct gettable_impl : std::false_type {};

    template<std::size_t i, typename T>
    struct gettable_impl<i, T,
      std::void_t<decltype(collections::get<i>(std::declval<T&>()))>> : std::true_type {};
  }
#endif


  /**
   * \brief T has an element i that is accessible by collections::get.
   */
  template<std::size_t i, typename T>
#ifdef __cpp_concepts
  concept gettable = requires(T& t) { collections::get<i>(t); };
#else
  constexpr bool gettable = detail::gettable_impl<i, T>::value;
#endif


}

#endif

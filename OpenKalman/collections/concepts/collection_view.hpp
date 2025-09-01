/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref collections::collection_view.
 */

#ifndef OPENKALMAN_COLLECTIONS_COLLECTION_VIEW_HPP
#define OPENKALMAN_COLLECTIONS_COLLECTION_VIEW_HPP

#include "basics/compatibility/views/view-concepts.hpp"
#include "uniformly_gettable.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_fixed_size : std::false_type {};

    template<typename T>
    struct has_fixed_size<T, std::enable_if_t<sized<T>>> : std::bool_constant<size_of_v<T> != dynamic_size> {};

  }
#endif


  /**
   * \brief A view to a \ref collection which is also a std::ranges:view.
   * \details If T is \ref sized and that size is not dynamic, T must be \ref uniformly_gettable.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept collection_view =
    stdcompat::ranges::view<T> and
    stdcompat::ranges::random_access_range<T> and
    (not sized<T> or size_of_v<T> == dynamic_size or uniformly_gettable<T>);
#else
  constexpr bool collection_view =
    stdcompat::ranges::view<T> and
    stdcompat::ranges::random_access_range<T> and
    (not detail::has_fixed_size<T>::value or uniformly_gettable<T>);
#endif


}

#endif

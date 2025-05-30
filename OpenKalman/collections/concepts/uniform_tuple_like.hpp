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
 * \brief Definition for \ref collections::uniform_tuple_like.
 */

#ifndef OPENKALMAN_COLLECTIONS_UNIFORM_TUPLE_LIKE_HPP
#define OPENKALMAN_COLLECTIONS_UNIFORM_TUPLE_LIKE_HPP

#include "collections/traits/size_of.hpp"
#include "collections/traits/common_tuple_type.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct has_common_tuple_type : std::false_type {};

    template<typename T>
    struct has_common_tuple_type<T, std::void_t<typename common_tuple_type<T>::type>> : std::true_type {};
  }
#endif


  /**
   * \brief A \ref tuple_like object that has a \ref common_collection_type.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept uniform_tuple_like = tuple_like<T> and (size_of_v<T> == 0 or requires { typename common_tuple_type<T>::type; });
#else
  constexpr bool uniform_tuple_like = tuple_like<T> and (size_of_v<T> == 0 or detail::has_common_tuple_type<T>::value);
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_COLLECTIONS_UNIFORM_TUPLE_LIKE_HPP

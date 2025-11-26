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
 * \brief Definition for \ref collections::common_collection_type.
 */

#ifndef OPENKALMAN_COLLECTIONS_COMMON_COLLECTION_TYPE_HPP
#define OPENKALMAN_COLLECTIONS_COMMON_COLLECTION_TYPE_HPP

#include "basics/basics.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/traits/collection_element.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief The common type within a \ref collections::collection "collection", if it exists.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct common_collection_type {};


  /// \overload
#ifdef __cpp_concepts
  template<stdex::ranges::random_access_range T>
  struct common_collection_type<T>
#else
  template<typename T>
  struct common_collection_type<T, std::enable_if_t<stdex::ranges::random_access_range<T>>>
#endif
  {
    using type = stdex::ranges::range_value_t<stdex::remove_cvref_t<T>>;
  };


  namespace detail
  {
    template<typename T, typename = std::make_index_sequence<size_of<T>::value>>
    struct common_tuple_type {};

    template<typename T, std::size_t...i>
    struct common_tuple_type<T, std::index_sequence<i...>>
      : stdex::common_reference<collection_element_t<i, T>...> {};
  }


  /// \overload
#ifdef __cpp_concepts
  template<uniformly_gettable T> requires (not stdex::ranges::random_access_range<T>)
  struct common_collection_type<T>
#else
  template<typename T>
  struct common_collection_type<T,
    std::enable_if_t<uniformly_gettable<T> and not stdex::ranges::random_access_range<T>>>
#endif
    : detail::common_tuple_type<T> {};


  /**
   * \brief Helper template for \ref common_collection_type.
   */
#ifdef __cpp_concepts
  template<collection T>
#else
  template<typename T>
#endif
  using common_collection_type_t = typename common_collection_type<T>::type;

}

#endif

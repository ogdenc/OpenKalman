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

#include "collections/concepts/collection.hpp"
#include "collections/traits/common_tuple_type.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief The common type within a \ref collections::collection "collection", if it exists.
   */
#ifdef __cpp_lib_ranges
  template<collection T>
#else
  template<typename T, typename = void>
#endif
  struct common_collection_type : common_tuple_type<T> {};


#ifdef __cpp_lib_ranges
  template<collection T> requires std::ranges::range<T>
  struct common_collection_type<T>
  {
    using type = std::ranges::range_value_t<std::remove_cvref_t<T>>;
  };
#else
  template<typename T>
  struct common_collection_type<T, std::enable_if_t<ranges::range<T>>>
  {
    using type = ranges::range_value_t<remove_cvref_t<T>>;
  };
#endif


  /**
   * \brief Helper template for \ref common_collection_type.
   */
  template<typename T>
  using common_collection_type_t = typename common_collection_type<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_COLLECTIONS_COMMON_COLLECTION_TYPE_HPP

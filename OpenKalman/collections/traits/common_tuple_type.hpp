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
 * \brief Definition for \ref collections::common_tuple_type.
 */

#ifndef OPENKALMAN_COLLECTIONS_COMMON_TUPLE_TYPE_HPP
#define OPENKALMAN_COLLECTIONS_COMMON_TUPLE_TYPE_HPP

#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename Tup, typename = std::make_index_sequence<std::tuple_size_v<Tup>>>
    struct common_tuple_type_impl {};


    template<typename Tup, std::size_t...ix>
    struct common_tuple_type_impl<Tup, std::index_sequence<ix...>>
#if __cplusplus >= 202002L
      : std::common_reference<std::tuple_element_t<ix, std::decay_t<Tup>>...> {};
#else
      : std::common_type<std::tuple_element_t<ix, std::decay_t<Tup>>...> {};
#endif
  }


  /**
   * \brief The common type within a \ref tuple_like object, if it exists.
   */
#ifdef __cpp_concepts
  template<tuple_like T>
  struct common_tuple_type
#else
  template<typename T, typename = void>
  struct common_tuple_type;

  template<typename T>
  struct common_tuple_type<T, std::enable_if_t<tuple_like<T>>>
#endif
    : detail::common_tuple_type_impl<std::decay_t<T>> {};


  /**
   * \brief Helper template for \ref common_collection_type.
   */
  template<typename T>
  using common_tuple_type_t = typename common_tuple_type<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_COLLECTIONS_COMMON_TUPLE_TYPE_HPP

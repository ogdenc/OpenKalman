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
 * \brief Definition for \ref collections::size_of.
 */

#ifndef OPENKALMAN_COLLECTIONS_SIZE_OF_HPP
#define OPENKALMAN_COLLECTIONS_SIZE_OF_HPP

#include <type_traits>
#include <tuple>
#include "collections/concepts/tuple_like.hpp"
#include "collections/concepts/collection.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief The size of a \ref collection.
   */
#ifdef __cpp_concepts
  template<collection Collection>
#else
  template<typename Collection, typename = void>
#endif
  struct size_of;


#ifdef __cpp_concepts
  template<tuple_like Collection>
  struct size_of<Collection>
#else
  template<typename Collection>
  struct size_of<Collection, std::enable_if_t<tuple_like<Collection>>>
#endif
    : std::tuple_size<std::decay_t<Collection>> {};


#ifdef __cpp_lib_span
  namespace detail
  {
    template<typename Collection>
    struct size_of_impl : std::integral_constant<std::size_t, dynamic_size> {};


    template<typename T, std::size_t Extent>
    struct size_of_impl<std::span<T, Extent>>
      : std::integral_constant<std::size_t, Extent == std::dynamic_extent ? dynamic_size : Extent> {};
  }


  template<sized_random_access_range Collection> requires (not tuple_like<Collection>)
  struct size_of<Collection>
    : detail::size_of_impl<decltype(std::span{std::declval<std::add_lvalue_reference_t<Collection>>()})> {};
#elif defined(__cpp_concepts)
  template<sized_random_access_range Collection> requires (not tuple_like<Collection>)
  struct size_of<Collection>
    : std::integral_constant<std::size_t, dynamic_size> {};
#else
  template<typename Collection>
  struct size_of<Collection, std::enable_if_t<sized_random_access_range<Collection> and (not tuple_like<Collection>)>>
    : std::integral_constant<std::size_t, dynamic_size> {};
#endif


  /**
   * \brief Helper for \ref collections::size_of.
   */
  template<typename Collection>
  constexpr std::size_t size_of_v = size_of<Collection>::value;


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_SIZE_OF_HPP

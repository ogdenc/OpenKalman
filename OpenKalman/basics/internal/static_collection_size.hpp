/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref static_range_size.
 */

#ifndef OPENKALMAN_STATIC_COLLECTION_SIZE_HPP
#define OPENKALMAN_STATIC_COLLECTION_SIZE_HPP

#include <type_traits>
#include <tuple>
#include "basics/global-definitions.hpp"
#include "basics/internal/collection.hpp"


namespace OpenKalman::internal
{
  /**
   * \brief The size of a \ref internal::collection.
   */
#ifdef __cpp_concepts
  template<typename Collection>
#else
  template<typename Collection, typename = void>
#endif
  struct static_collection_size;


#ifdef __cpp_concepts
  template<internal::tuple_like Collection>
  struct static_collection_size<Collection>
#else
  template<typename Collection>
  struct static_collection_size<Collection, std::enable_if_t<internal::tuple_like<Collection>>>
#endif
      : std::tuple_size<std::decay_t<Collection>> {};


#ifdef __cpp_lib_span
  namespace detail
  {
    template<typename Collection>
    struct static_collection_size_impl : std::integral_constant<std::size_t, dynamic_size> {};


    template<typename T, std::size_t Extent>
    struct static_collection_size_impl<std::span<T, Extent>>
      : std::integral_constant<std::size_t, Extent == std::dynamic_extent ? dynamic_size : Extent> {};
  }


  template<internal::collection Collection> requires (not internal::tuple_like<Collection>)
  struct static_collection_size<Collection>
    : detail::static_collection_size_impl<decltype(std::span{std::declval<std::add_lvalue_reference_t<Collection>>()})> {};
#else
  template<typename Collection>
  struct static_collection_size<Collection, std::enable_if_t<
    internal::collection<Collection> and (not internal::tuple_like<Collection>)>>
      : std::integral_constant<std::size_t, dynamic_size> {};
#endif


  /**
   * \brief Helper for \ref static_collection_size.
   */
  template<typename Collection>
  constexpr std::size_t static_collection_size_v = static_collection_size<Collection>::value;


} // namespace OpenKalman::internal

#endif //OPENKALMAN_STATIC_COLLECTION_SIZE_HPP

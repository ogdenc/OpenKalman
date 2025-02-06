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
 * \brief Definition for \ref collection_size_of.
 */

#ifndef OPENKALMAN_COLLECTION_SIZE_OF_HPP
#define OPENKALMAN_COLLECTION_SIZE_OF_HPP

#include <type_traits>
#include <tuple>
#include "tuple_like.hpp"
#include "collection.hpp"

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
  struct collection_size_of;


#ifdef __cpp_concepts
  template<internal::tuple_like Collection>
  struct collection_size_of<Collection>
#else
  template<typename Collection>
  struct collection_size_of<Collection, std::enable_if_t<internal::tuple_like<Collection>>>
#endif
      : std::tuple_size<std::decay_t<Collection>> {};


#ifdef __cpp_lib_span
  namespace detail
  {
    template<typename Collection>
    struct collection_size_of_impl : std::integral_constant<std::size_t, dynamic_size> {};


    template<typename T, std::size_t Extent>
    struct collection_size_of_impl<std::span<T, Extent>>
      : std::integral_constant<std::size_t, Extent == std::dynamic_extent ? dynamic_size : Extent> {};
  }


  template<internal::collection Collection> requires (not internal::tuple_like<Collection>)
  struct collection_size_of<Collection>
    : detail::collection_size_of_impl<decltype(std::span{std::declval<std::add_lvalue_reference_t<Collection>>()})> {};
#elifdef __cpp_concepts
  template<internal::collection Collection> requires (not internal::tuple_like<Collection>)
  struct collection_size_of<Collection>
    : std::integral_constant<std::size_t, dynamic_size> {};
#else
  template<typename Collection>
  struct collection_size_of<Collection, std::enable_if_t<
    internal::collection<Collection> and (not internal::tuple_like<Collection>)>>
    : std::integral_constant<std::size_t, dynamic_size> {};
#endif


  /**
   * \brief Helper for \ref collection_size_of.
   */
  template<typename Collection>
  constexpr std::size_t collection_size_of_v = collection_size_of<Collection>::value;


} // namespace OpenKalman::internal

#endif //OPENKALMAN_COLLECTION_SIZE_OF_HPP

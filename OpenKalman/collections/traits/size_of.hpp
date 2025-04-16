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
#ifdef __cpp_lib_span
#include <span>
#endif
#include "basics/global-definitions.hpp"
#include "collections/concepts/tuple_like.hpp"
#include "collections/concepts/collection.hpp"

namespace OpenKalman::collections
{
  namespace internal
  {
    namespace detail
    {
      template<typename T, typename = void>
      struct has_tuple_size : std::false_type {};

      template<typename T>
      struct has_tuple_size<T, std::void_t<decltype(std::tuple_size<std::decay_t<T>>::value)>> : std::true_type {};
    }


    /**
     * \internal
     * \brief The size of an object, assuming that it is a \ref collection.
     * \details If the object is a collection with dynamic size, the result will be \ref dynamic_size.
     */
#ifdef __cpp_concepts
    template<typename T>
  #else
    template<typename T, typename = void>
  #endif
    struct size_if_collection : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
    template<typename T> requires detail::has_tuple_size<T>::value
    struct size_if_collection<T>
#else
    template<typename T>
    struct size_if_collection<T, std::enable_if_t<detail::has_tuple_size<T>::value>>
#endif
      : std::tuple_size<std::decay_t<T>> {};


#ifdef __cpp_lib_span
    namespace detail
    {
      template<typename T>
      struct collection_size_impl;

      template<typename T, std::size_t Extent>
      struct collection_size_impl<std::span<T, Extent>>
        : std::integral_constant<std::size_t, Extent == std::dynamic_extent ? dynamic_size : Extent> {};
    }


    template<typename T> requires (not detail::has_tuple_size<T>::value) and requires(T& t) { std::span {t}; }
    struct size_if_collection<T>
      : detail::collection_size_impl<decltype(std::span {std::declval<T&>()})> {};
#endif


    template<typename T>
    static constexpr std::size_t
    size_if_collection_v = size_if_collection<T>::value;

  }


  /**
   * \brief The size of a \ref collection.
   */
#ifdef __cpp_concepts
  template<collection T>
  struct size_of : internal::size_if_collection<T> {};
#else
  template<typename T, typename = void>
  struct size_of;

  template<typename T>
  struct size_of<T, std::enable_if_t<collection<T>>> : internal::size_if_collection<T> {};
#endif


  /**
   * \brief Helper for \ref collections::size_of.
   */
#ifdef __cpp_concepts
  template<collection T>
#else
  template<typename T>
#endif
  constexpr std::size_t size_of_v = size_of<T>::value;


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_SIZE_OF_HPP

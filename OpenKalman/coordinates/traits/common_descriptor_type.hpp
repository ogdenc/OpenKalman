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
 * \brief Definition for \ref collections::common_descriptor_type.
 */

#ifndef OPENKALMAN_COLLECTIONS_COMMON_DESCRIPTOR_TYPE_HPP
#define OPENKALMAN_COLLECTIONS_COMMON_DESCRIPTOR_TYPE_HPP

#include "collections/collections.hpp"
#include "coordinates/concepts/pattern.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief The common type within a \ref pattern, if it exists.
   * \details If T is a \ref descriptor, the common type is std::decay_t<T>.
   * If T is a \ref euclidean_pattern the common type is Dimensions<1>.
   * Otherwise, the result is similar to \ref collections::common_collection_type except that zero-dimension descriptors
   * in a \ref descriptor_collection are ignored.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void, typename = void>
#endif
  struct common_descriptor_type {};


  /// \overload
#ifdef __cpp_concepts
  template<descriptor T>
  struct common_descriptor_type<T>
#else
  template<typename T>
  struct common_descriptor_type<T, std::enable_if_t<descriptor<T>>>
#endif
    : std::conditional_t<
        euclidean_pattern<T>,
        stdcompat::type_identity<Dimensions<1>>,
        stdcompat::type_identity<std::decay_t<T>>
      > {};


#ifndef __cpp_concepts
  namespace internal
  {
    template<typename T, typename = void>
    struct has_common_collection_type : std::false_type {};

    template<typename T>
    struct has_common_collection_type<T, std::void_t<typename collections::common_collection_type<T>::type>> : std::true_type {};
  }
#endif


  namespace detail
  {
    template<typename...Ts>
    struct common_descriptor_type_iter : stdcompat::type_identity<std::tuple<>> {};

    template<typename T>
    struct common_descriptor_type_iter<T> : stdcompat::type_identity<T> {};

    template<typename T0, typename T1, typename...Ts>
    struct common_descriptor_type_iter<T0, T1, Ts...>
      : std::conditional_t<
          dimension_of_v<T0> == 0 and dimension_of_v<T1> == 0,
          common_descriptor_type_iter<Ts...>,
          std::conditional_t<
            dimension_of_v<T0> == 0,
            common_descriptor_type_iter<T1, Ts...>,
            std::conditional_t<
              dimension_of_v<T1> == 0,
              common_descriptor_type_iter<T0, Ts...>,
              common_descriptor_type_iter<typename std::conditional_t<
                stdcompat::common_reference_with<T0, T1>,
                stdcompat::common_reference<T0, T1>,
                stdcompat::type_identity<Any<>>
              >::type, Ts...>
            >
          >
        > {};


    template<typename T, typename = std::make_index_sequence<collections::size_of<T>::value>>
    struct common_descriptor_type_expand {};

    template<typename T, std::size_t...i>
    struct common_descriptor_type_expand<T, std::index_sequence<i...>>
      : common_descriptor_type_iter<collections::collection_element_t<i, T>...> {};


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct common_descriptor_type_impl : collections::common_collection_type<T> {};


#ifdef __cpp_concepts
    template<collections::uniformly_gettable T> requires (not stdcompat::ranges::random_access_range<T>)
    struct common_descriptor_type_impl<T>
#else
    template<typename T>
    struct common_descriptor_type_impl<T,
      std::enable_if_t<collections::uniformly_gettable<T> and not stdcompat::ranges::random_access_range<T>>>
#endif
      : common_descriptor_type_expand<T> {};

  }


  /// \overload
#ifdef __cpp_concepts
  template<descriptor_collection T>
  struct common_descriptor_type<T>
#else
  template<typename T>
  struct common_descriptor_type<T, std::enable_if_t<descriptor_collection<T>>>
#endif
    : std::conditional_t<
        euclidean_pattern<T>,
        stdcompat::type_identity<Dimensions<1>>,
        detail::common_descriptor_type_impl<T>
      > {};


  /**
   * \brief Helper template for \ref common_descriptor_type.
   */
  template<typename T>
  using common_descriptor_type_t = typename common_descriptor_type<T>::type;

}

#endif

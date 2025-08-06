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
 * \brief Definition for \ref collections::viewable_tuple_like.
 */

#ifndef OPENKALMAN_COLLECTIONS_UNIFORM_TUPLE_LIKE_HPP
#define OPENKALMAN_COLLECTIONS_UNIFORM_TUPLE_LIKE_HPP

#include "collections/traits/size_of.hpp"
#include "collections/traits/common_tuple_type.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename T>
#ifdef __cpp_concepts
    concept move_constructible_object_or_lvalue_ref =
#else
    inline constexpr bool move_constructible_object_or_lvalue_ref =
#endif
      (stdcompat::move_constructible<T> and std::is_object_v<T>) or
      std::is_lvalue_reference_v<T>;


#ifdef __cpp_concepts
    template<typename T, typename = std::make_index_sequence<size_of_v<T>>>
    struct has_viewable_elements : std::false_type {};

    template<typename T, std::size_t...i> requires
      (... and move_constructible_object_or_lvalue_ref<std::tuple_element_t<i, T>>)
    struct has_viewable_elements<T, std::index_sequence<i...>> : std::true_type {};
#else
    template<typename T, typename = std::make_index_sequence<size_of_v<T>>, typename = void>
    struct has_viewable_elements_impl : std::false_type {};

    template<typename T, std::size_t...i>
    struct has_viewable_elements_impl<T, std::index_sequence<i...>, std::enable_if_t<
      (... and move_constructible_object_or_lvalue_ref<typename std::tuple_element<i, T>::type>)>> : std::true_type {};


    template<typename T, typename = void>
    struct has_viewable_elements : std::false_type {};

    template<typename T>
    struct has_viewable_elements<T, std::enable_if_t<tuple_like<T>>> : has_viewable_elements_impl<T> {};


    template<typename T, typename = void>
    struct has_common_tuple_type : std::false_type {};

    template<typename T>
    struct has_common_tuple_type<T, std::void_t<typename common_tuple_type<T>::type>> : std::true_type {};
#endif
  }


  /**
   * \brief A \ref tuple_like object that has a \ref common_collection_type and can be converted into
   * a \ref collection_view by passing it to \ref collections::views::all.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept viewable_tuple_like =
    tuple_like<T> and
    (size_of_v<T> == 0 or requires { typename common_tuple_type<T>::type; }) and
    (std::is_lvalue_reference_v<T> or detail::has_viewable_elements<std::decay_t<T>>::value);
#else
  inline constexpr bool viewable_tuple_like =
    tuple_like<T> and
    ((sized<T> and values::fixed_number_compares_with<size_of<T>, 0_uz>) or detail::has_common_tuple_type<T>::value) and
    (std::is_lvalue_reference_v<T> or detail::has_viewable_elements<std::decay_t<T>>::value);
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_COLLECTIONS_UNIFORM_TUPLE_LIKE_HPP

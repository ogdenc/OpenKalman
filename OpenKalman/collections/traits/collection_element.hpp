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
 * \brief Definition for \ref collections::collection_element.
 */

#ifndef OPENKALMAN_COLLECTIONS_COLLECTION_ELEMENT_HPP
#define OPENKALMAN_COLLECTIONS_COLLECTION_ELEMENT_HPP

#include "values/values.hpp"
#include "collections/concepts/sized.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/concepts/gettable.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<std::size_t i, typename T, typename = void>
    struct has_tuple_element : std::false_type {};

    template<std::size_t i, typename T>
    struct has_tuple_element<i, T, std::void_t<typename std::tuple_element<i, std::decay_t<T>>::type>>
      : std::true_type {};
  }
#endif


  /**
   * \brief The type of the element at a given index, if it can be determined at compile time.
   * \details This is a generalized version of std::tuple_element;
   */
#ifdef __cpp_concepts
  template<std::size_t i, typename T>
#else
  template<std::size_t i, typename T, typename = void>
#endif
  struct collection_element {};


  /// \overload
#ifdef __cpp_concepts
  template<std::size_t i, sized T> requires
    (size_of_v<T> != dynamic_size) and
    (i < size_of_v<T>) and
    requires { typename std::tuple_element<i, std::decay_t<T>>::type; }
  struct collection_element<i, T>
#else
  template<std::size_t i, typename T>
  struct collection_element<i, T, std::enable_if_t<
    values::fixed_value_compares_with<size_of<T>, dynamic_size, &stdcompat::is_neq> and
    values::fixed_value_compares_with<size_of<T>, i, &stdcompat::is_gt> and
    detail::has_tuple_element<i, T>::value>>
#endif
    : std::tuple_element<i, std::decay_t<T>> {};


/**
 * \overload
 * \details If T has an associated <code>get</code> function, derive the type.
 */
#ifdef __cpp_concepts
  template<std::size_t i, typename T> requires
    (size_of_v<T> != dynamic_size) and
    (i < size_of_v<T>) and
    (not requires { typename std::tuple_element<i, std::decay_t<T>>::type; }) and
    gettable<i, T>
  struct collection_element<i, T>
#else
  template<std::size_t i, typename T>
  struct collection_element<i, T, std::enable_if_t<
    values::fixed_value_compares_with<size_of<T>, dynamic_size, &stdcompat::is_neq> and
    values::fixed_value_compares_with<size_of<T>, i, &stdcompat::is_gt> and
    (not detail::has_tuple_element<i, T>::value) and
    gettable<i, T>>>
#endif
  {
    using type = OpenKalman::internal::remove_rvalue_reference_t<
      decltype(OpenKalman::internal::generalized_std_get<i>(std::declval<stdcompat::remove_cvref_t<T>>()))>;
  };


  /**
   * \overload
   * \details Does not do any bounds checking if the size is dynamic.
   */
#ifdef __cpp_concepts
  template<std::size_t i, stdcompat::ranges::random_access_range T> requires
    (size_of_v<T> == dynamic_size or i < size_of_v<T>) and
    (not requires { typename std::tuple_element<i, std::decay_t<T>>::type; }) and
    (not gettable<i, T>)
  struct collection_element<i, T>
#else
  template<std::size_t i, typename T>
  struct collection_element<i, T, std::enable_if_t<
    (values::fixed_value_compares_with<size_of<T>, dynamic_size> or
      values::fixed_value_compares_with<size_of<T>, i, &stdcompat::is_gt>) and
    stdcompat::ranges::random_access_range<T> and
    not detail::has_tuple_element<i, T>::value and
    not gettable<i, T>>>
#endif
  {
    using type = stdcompat::ranges::range_value_t<stdcompat::remove_cvref_t<T>>;
  };


  /**
   * \brief Helper template for \ref collection_element.
   */
  template<std::size_t i, typename T>
  using collection_element_t = typename collection_element<i, T>::type;


}

#endif

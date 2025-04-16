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
 * \internal
 * \brief Definition for \ref collections::internal::maybe_tuple_element.
 */

#ifndef OPENKALMAN_COLLECTIONS_TUPLE_ELEMENT_BASE_HPP
#define OPENKALMAN_COLLECTIONS_TUPLE_ELEMENT_BASE_HPP

#include <tuple>
#include "values/concepts/fixed.hpp"
#include "values/concepts/index.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::collections::internal
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename I, typename T>
#else
    template<typename I, typename T, typename = void>
#endif
    struct tuple_element_defined : std::false_type {};


#ifdef __cpp_concepts
    template<typename I, typename T> requires requires { typename std::tuple_element<value::fixed_number_of<I>::value, T>::type; }
    struct tuple_element_defined<I, T>
  #else
    template<typename I, typename T>
    struct tuple_element_defined<I, T, std::void_t<typename std::tuple_element<value::fixed_number_of<I>::value, T>::type>>
  #endif
      : std::true_type {};
  }


  /**
   * \internal
   * \brief Base class for std::tuple_element specializations, potentially inheriting from std::tuple_element<i, T>.
   */
#ifdef __cpp_concepts
  template<value::index I, typename T>
#else
  template<typename I, typename T, typename = void>
#endif
  struct maybe_tuple_element {};


#ifdef __cpp_concepts
  template<value::index I, typename T> requires detail::tuple_element_defined<I, T>::value
  struct maybe_tuple_element<I, T>
#else
  template<typename I, typename T>
  struct maybe_tuple_element<I, T, std::enable_if_t<
    value::index<I> and std::is_same_v<T, std::decay_t<T>> and detail::tuple_element_defined<I, T>::value>>
#endif
    : std::tuple_element<value::fixed_number_of_v<I>, T> {};


#ifdef __cpp_concepts
  template<value::index I, typename T> requires detail::tuple_element_defined<I, T>::value
  struct maybe_tuple_element<I, T&>
#else
  template<typename I, typename T>
  struct maybe_tuple_element<I, T&, std::enable_if_t<value::index<I> and detail::tuple_element_defined<I, T>::value>>
#endif
  {
    using type = std::tuple_element_t<value::fixed_number_of_v<I>, T>&;
  };


#ifdef __cpp_concepts
  template<value::index I, typename T> requires detail::tuple_element_defined<I, T>::value
  struct maybe_tuple_element<I, T&&>
#else
  template<typename I, typename T>
  struct maybe_tuple_element<I, T&&, std::enable_if_t<value::index<I> and detail::tuple_element_defined<I, T>::value>>
#endif
  {
    using type = std::tuple_element_t<value::fixed_number_of_v<I>, T>&&;
  };

}

#endif //OPENKALMAN_COLLECTIONS_TUPLE_ELEMENT_BASE_HPP

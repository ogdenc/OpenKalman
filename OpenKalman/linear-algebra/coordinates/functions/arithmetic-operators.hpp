/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Arithmetic operators for \ref coordinates::pattern objects.
 */

#ifndef OPENKALMAN_COORDINATES_ARITHMETIC_OPERATORS_HPP
#define OPENKALMAN_COORDINATES_ARITHMETIC_OPERATORS_HPP

#include "collections/collections.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"

namespace OpenKalman::coordinates
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct tuple_is_nonempty : std::false_type {};

    template<typename T>
    struct tuple_is_nonempty<T, std::enable_if_t<(std::tuple_size<T>::value > 0)>> : std::true_type {};
  }
#endif

  /**
   * \brief Add two sets of \ref coordinates::pattern objects, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<pattern T, pattern U> requires
    (descriptor<T> or collections::viewable_collection<T>) and
    (descriptor<U> or collections::viewable_collection<U>) and
    (not values::value<T> or not values::value<U>) and
    (not collections::tuple_like<T> or not collections::tuple_like<U> or
      std::tuple_size_v<T> > 0 or std::tuple_size_v<U> > 0 or
      collections::collection_view<T> or collections::collection_view<U>)
#else
  template<typename T, typename U, std::enable_if_t<pattern<T> and pattern<U> and
    (descriptor<T> or collections::viewable_collection<T>) and
    (descriptor<U> or collections::viewable_collection<U>) and
    (not values::value<T> or not values::value<U>) and
    (not collections::tuple_like<T> or not collections::tuple_like<U> or
      detail::tuple_is_nonempty<T>::value or detail::tuple_is_nonempty<U>::value or
      collections::collection_view<T> or collections::collection_view<U>), int> = 0>
#endif
  constexpr auto operator+(T&& t, U&& u)
  {
    if constexpr (euclidean_pattern<T> and euclidean_pattern<U>)
      return Dimensions {values::operation(std::plus{}, get_dimension(t), get_dimension(u))};
    else if constexpr (descriptor<T>)
      return coordinates::operator+(std::make_tuple(std::forward<T>(t)), std::forward<U>(u));
    else if constexpr (descriptor<U>)
      return coordinates::operator+(std::forward<T>(t), std::make_tuple(std::forward<U>(u)));
    else
      return collections::views::concat(std::forward<T>(t), std::forward<U>(u));
  }


  /**
   * \brief Replicate a \ref coordinates::pattern some number of times.
   */
#ifdef __cpp_concepts
  template<pattern Arg, values::index N> requires
    (descriptor<Arg> or collections::viewable_collection<Arg>) and (not values::value<Arg>) and
    (not collections::tuple_like<Arg> or std::tuple_size_v<Arg> > 0 or collections::collection_view<Arg>)
#else
  template<typename Arg, typename N, std::enable_if_t<pattern<Arg> and values::index<N> and
    (descriptor<Arg> or collections::viewable_collection<Arg>) and (not values::value<Arg>) and
    (not collections::tuple_like<Arg> or detail::tuple_is_nonempty<Arg>::value or collections::collection_view<Arg>), int> = 0>
#endif
  constexpr auto operator*(Arg&& arg, const N& n)
  {
    if constexpr (euclidean_pattern<Arg>)
      return Dimensions {values::operation(std::multiplies{}, get_dimension(arg), n)};
    else if constexpr (descriptor<Arg>)
      return collections::views::repeat(std::forward<Arg>(arg), n);
    else
      return std::forward<Arg>(arg) | collections::views::all | collections::views::replicate(n);
  }


  /**
   * \overload
   * \brief Replicate a \ref coordinates::pattern some number of times.
   */
#ifdef __cpp_concepts
  template<pattern Arg, values::index N> requires
    (descriptor<Arg> or collections::viewable_collection<Arg>) and (not values::value<Arg>) and
    (not collections::tuple_like<Arg> or std::tuple_size_v<Arg> > 0 or collections::collection_view<Arg>)
#else
  template<typename Arg, typename N, std::enable_if_t<pattern<Arg> and values::index<N> and
    (descriptor<Arg> or collections::viewable_collection<Arg>) and (not values::value<Arg>) and
    (not collections::tuple_like<Arg> or detail::tuple_is_nonempty<Arg>::value or collections::collection_view<Arg>), int> = 0>
#endif
constexpr auto operator*(const N& n, Arg&& arg)
  {
    return operator*(std::forward<Arg>(arg), n);
  }


}


#endif
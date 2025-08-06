/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref compares_with.
 */

#ifndef OPENKALMAN_COORDINATE_COMPARES_WITH_HPP
#define OPENKALMAN_COORDINATE_COMPARES_WITH_HPP

#include "basics/basics.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/dynamic_pattern.hpp"
#include "linear-algebra/coordinates/functions/compare.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
    concept euclidean_status_is_fixed = values::fixed<decltype(coordinates::get_is_euclidean(std::declval<T>()))>;
#else
    template<typename T, typename U, auto comp, Applicability applicability, typename = void>
    struct comparison_invocable : std::false_type {};

    template<typename T, typename U, auto comp, Applicability applicability>
    struct comparison_invocable<T, U, comp, applicability,
      std::void_t<decltype(stdcompat::invoke(comp, compare(std::declval<T>(), std::declval<U>())))>>
      : std::true_type {};


    template<typename T, typename U, auto comp, typename = void>
    struct compares_with_guaranteed : std::false_type {};

    template<typename T, typename U, auto comp>
    struct compares_with_guaranteed<T, U, comp, std::enable_if_t<
      std::bool_constant<stdcompat::invoke(comp, compare(std::decay_t<T>{}, std::decay_t<U>{}))>::value>>
      : std::true_type {};


    template<typename T, typename U, auto comp, Applicability applicability, typename = void>
    struct compares_with_permitted : std::false_type {};

    template<typename T, typename U, auto comp, Applicability applicability>
    struct compares_with_permitted<T, U, comp, applicability, std::enable_if_t<
      (values::fixed<decltype(coordinates::get_is_euclidean(std::declval<T>()))> !=
        values::fixed<decltype(coordinates::get_is_euclidean(std::declval<U>()))>)>>
      : std::true_type {};
#endif
  } // namespace detail


  /**
   * \brief Specifies that a set of \ref coordinates::pattern objects may be equivalent based on what is known at compile time.
   * \details Every \ref coordinate_list in the set must be potentially equivalent to every other \ref coordinate_list in the set.
   * Sets of vector space descriptors are equivalent if they are treated functionally the same.
   * - Any \ref coordinate_list is equivalent to itself.
   * - std::tuple<As...> is equivalent to std::tuple<Bs...>, if each As is equivalent to its respective Bs.
   * - std::tuple<A> is equivalent to A, and vice versa.
   * - Dynamic \ref coordinates::euclidean_pattern objects are equivalent to any other \ref coordinates::euclidean_pattern,
   * \par Examples:
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Axis, Direction&gt;&gt;</code>
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Axis, Direction, angle::Radians&gt;, less_than<>, Applicability::guaranteed&gt;</code>
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Dimensions<>, Direction, angle::Radians&gt;, less_than<>, Applicability::permitted&gt;</code>
   * \tparam comp A callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
  template<typename T, typename U, auto comp = &stdcompat::is_eq, Applicability applicability = Applicability::guaranteed>
#ifdef __cpp_concepts
  concept compares_with = pattern<T> and pattern<U> and
    requires(T t, U u) { {stdcompat::invoke(comp, compare(t, u))} -> std::same_as<bool>; } and
    ((fixed_pattern<T> and fixed_pattern<U> and std::bool_constant<stdcompat::invoke(comp, compare(std::decay_t<T>{}, std::decay_t<U>{}))>::value) or
      (applicability == Applicability::permitted and (dynamic_pattern<T> or dynamic_pattern<U>) and
        (euclidean_pattern<T> == euclidean_pattern<U> or detail::euclidean_status_is_fixed<T> != detail::euclidean_status_is_fixed<U>)));
#else
  constexpr bool compares_with = pattern<T> and pattern<U> and
    detail::comparison_invocable<T, U, comp, applicability>::value and
    ((fixed_pattern<T> and fixed_pattern<U> and detail::compares_with_guaranteed<T, U, comp>::value) or
      (applicability == Applicability::permitted and (dynamic_pattern<T> or dynamic_pattern<U>) and
        (euclidean_pattern<T> == euclidean_pattern<U> or detail::compares_with_permitted<T, U, comp, applicability>::value)));
#endif


} // namespace OpenKalman::coordinates

#endif //OPENKALMAN_COORDINATE_COMPARES_WITH_HPP

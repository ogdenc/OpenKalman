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

#include <type_traits>
#include "basics/global-definitions.hpp"
#include "basics/classes/equal_to.hpp"
#include "values/views/identity.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/dynamic_pattern.hpp"
#include "linear-algebra/coordinates/functions/comparison-operators.hpp"
#include "linear-algebra/coordinates/views/comparison.hpp"

namespace OpenKalman::coordinate
{
  namespace detail
  {
    template<typename C>
    struct comp_adapter
    {
      template<typename T, typename U>
      constexpr bool operator()(const T& t, const U& u) const
      {
        if constexpr (euclidean_pattern<T> and euclidean_pattern<U>)
          return std::decay_t<C>{}(value::to_number(get_size(t)), value::to_number(get_size(u)));
        else if constexpr (collection<T> and collection<U>)
          return std::decay_t<C>{}(comparison_view<T>{t}, u);
        else
          return std::decay_t<C>{}(t, u);
      }
    };


#ifdef __cpp_concepts
    template<typename T>
    concept euclidean_status_is_fixed = value::fixed<decltype(coordinate::get_is_euclidean(std::declval<T>()))>;
#else
    template<typename T, typename U, typename Comparison, Applicability applicability, typename = void>
    struct comparison_invocable : std::false_type {};

    template<typename T, typename U, typename Comparison, Applicability applicability>
    struct comparison_invocable<T, U, Comparison, applicability, std::enable_if_t<
      std::is_convertible<decltype(detail::comp_adapter<Comparison>{}(std::declval<T>(), std::declval<U>())), bool>::value>>
      : std::true_type {};


    template<typename T, typename U, typename Comparison, typename = void>
    struct compares_with_guaranteed : std::false_type {};

    template<typename T, typename U, typename Comparison>
    struct compares_with_guaranteed<T, U, Comparison, std::enable_if_t<
      std::bool_constant<detail::comp_adapter<Comparison>{}(std::decay_t<T>{}, std::decay_t<U>{})>::value>>
      : std::true_type {};


    template<typename T, typename U, typename Comparison, Applicability applicability, typename = void>
    struct compares_with_permitted : std::false_type {};

    template<typename T, typename U, typename Comparison, Applicability applicability>
    struct compares_with_permitted<T, U, Comparison, applicability, std::enable_if_t<
      (value::fixed<decltype(coordinate::get_is_euclidean(std::declval<T>()))> !=
        value::fixed<decltype(coordinate::get_is_euclidean(std::declval<U>()))>)>>
      : std::true_type {};
#endif
  } // namespace detail


  /**
   * \brief Specifies that a set of \ref coordinate::pattern objects may be equivalent based on what is known at compile time.
   * \details Every \ref coordinate_list in the set must be potentially equivalent to every other \ref coordinate_list in the set.
   * Sets of vector space descriptors are equivalent if they are treated functionally the same.
   * - Any \ref coordinate_list is equivalent to itself.
   * - std::tuple<As...> is equivalent to std::tuple<Bs...>, if each As is equivalent to its respective Bs.
   * - std::tuple<A> is equivalent to A, and vice versa.
   * - Dynamic \ref coordinate::euclidean_pattern objects are equivalent to any other \ref coordinate::euclidean_pattern,
   * \par Examples:
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Axis, Direction&gt;&gt;</code>
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Axis, Direction, angle::Radians&gt;, less_than<>, Applicability::guaranteed&gt;</code>
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Dimensions<>, Direction, angle::Radians&gt;, less_than<>, Applicability::permitted&gt;</code>
   */
  template<typename T, typename U, typename Comparison = equal_to<>, Applicability applicability = Applicability::guaranteed>
#ifdef __cpp_concepts
  concept compares_with = pattern<T> and pattern<U> and std::default_initializable<Comparison> and
    std::convertible_to<decltype(detail::comp_adapter<Comparison>{}(std::declval<T>(), std::declval<U>())), bool> and
    ((fixed_pattern<T> and fixed_pattern<U> and std::bool_constant<detail::comp_adapter<Comparison>{}(std::decay_t<T>{}, std::decay_t<U>{})>::value) or
    (applicability == Applicability::permitted and (dynamic_pattern<T> or dynamic_pattern<U>) and
      (euclidean_pattern<T> == euclidean_pattern<U> or detail::euclidean_status_is_fixed<T> != detail::euclidean_status_is_fixed<U>)));
#else
  constexpr bool compares_with = pattern<T> and pattern<U> and std::is_default_constructible_v<Comparison> and
    detail::comparison_invocable<T, U, Comparison, applicability>::value and
      ((fixed_pattern<T> and fixed_pattern<U> and detail::compares_with_guaranteed<T, U, Comparison>::value) or
        (applicability == Applicability::permitted and (dynamic_pattern<T> or dynamic_pattern<U>) and
          (euclidean_pattern<T> == euclidean_pattern<U> or detail::compares_with_permitted<T, U, Comparison, applicability>::value)));
#endif


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_COORDINATE_COMPARES_WITH_HPP

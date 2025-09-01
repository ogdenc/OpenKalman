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
 * \brief Definition for \ref pattern_collection_compares_with.
 */

#ifndef OPENKALMAN_COORDINATE_PATTERN_COLLECTION_COMPARES_WITH_HPP
#define OPENKALMAN_COORDINATE_PATTERN_COLLECTION_COMPARES_WITH_HPP

#include "coordinates/concepts/pattern_collection.hpp"
#include "coordinates/concepts/fixed_pattern_collection.hpp"
#include "coordinates/functions/compare_pattern_collections.hpp"

namespace OpenKalman::coordinates
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename U, auto comp, typename = void>
    struct pc_compares_with_guaranteed : std::false_type {};

    template<typename T, typename U, auto comp>
    struct pc_compares_with_guaranteed<T, U, comp, std::enable_if_t<
      std::bool_constant<compare_pattern_collections<comp>(std::decay_t<T>{}, std::decay_t<U>{})>::value>>
      : std::true_type {};


    template<typename T, typename U, auto comp, typename = void>
    struct pc_compares_with_permitted : std::false_type {};

    template<typename T, typename U, auto comp>
    struct pc_compares_with_permitted<T, U, comp, std::enable_if_t<
      not values::fixed<decltype(compare_pattern_collections<comp>(std::declval<T>(), std::declval<U>()))>>>
      : std::true_type {};
  }
#endif


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
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Axis, Direction, angle::Radians&gt;, less_than<>, applicability::guaranteed&gt;</code>
   * <code>compares_with&lt;std::tuple&lt;Axis, Direction&gt;, std::tuple&lt;Dimensions<>, Direction, angle::Radians&gt;, less_than<>, applicability::permitted&gt;</code>
   * \tparam comp A callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
  template<typename T, typename U, auto comp = &stdcompat::is_eq, applicability a = applicability::guaranteed>
#ifdef __cpp_concepts
  concept pattern_collection_compares_with =
    pattern_collection<T> and pattern_collection<U> and std::is_invocable_r_v<bool, decltype(comp), stdcompat::partial_ordering> and
    ((fixed_pattern_collection<T> and fixed_pattern_collection<U> and
      std::bool_constant<compare_pattern_collections<comp>(std::decay_t<T>{}, std::decay_t<U>{})>::value) or
    (a == applicability::permitted and
      not values::fixed<decltype(compare_pattern_collections<comp>(std::declval<T>(), std::declval<U>()))>));
#else
  constexpr bool pattern_collection_compares_with =
    pattern_collection<T> and pattern_collection<U> and std::is_invocable_r_v<bool, decltype(comp), stdcompat::partial_ordering> and
    ((fixed_pattern_collection<T> and fixed_pattern_collection<U> and detail::pc_compares_with_guaranteed<T, U, comp>::value) or
      (a == applicability::permitted and detail::pc_compares_with_permitted<T, U, comp>::value));
#endif


}

#endif

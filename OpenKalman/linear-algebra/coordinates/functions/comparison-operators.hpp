/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Comparison operators for \ref coordinates::pattern "patterns".
 */

#ifndef OPENKALMAN_COORDINATES_COMPARISON_OPERATORS_HPP
#define OPENKALMAN_COORDINATES_COMPARISON_OPERATORS_HPP

#include "values/values.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_hash_code.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief Comparison operator for library-defined \ref coordinates::descriptor objects
   */
#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<descriptor A, descriptor B> requires (not values::value<A> or not values::value<B>)
  constexpr std::partial_ordering
  operator<=>(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b))
      return stdcompat::compare_three_way{}(values::to_number(get_dimension(a)), values::to_number(get_dimension(b)));
    if (internal::get_descriptor_hash_code(a) == internal::get_descriptor_hash_code(b)) return std::partial_ordering::equivalent;
    return std::partial_ordering::unordered;
  }


  /**
   * Equality operator for library-defined \ref coordinates::pattern objects
   */
  template<descriptor A, descriptor B> requires (not values::value<A> or not values::value<B>)
  constexpr bool
  operator==(const A& a, const B& b)
  {
    return stdcompat::is_eq(a <=> b);
  }
#else
  template<typename A, typename B, std::enable_if_t<
    descriptor<A> and descriptor<B> and not (values::value<A> and values::value<B>), int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b))
      return values::to_number(get_dimension(a)) == values::to_number(get_dimension(b));
    return internal::get_descriptor_hash_code(a) == internal::get_descriptor_hash_code(b);
  }


  template<typename A, typename B, std::enable_if_t<
    descriptor<A> and descriptor<B> and not (values::value<A> and values::value<B>), int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b))
      return values::to_number(get_dimension(a)) < values::to_number(get_dimension(b));
    return false;
  }


  template<typename A, typename B, std::enable_if_t<
    descriptor<A> and descriptor<B> and not (values::value<A> and values::value<B>), int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b))
      return values::to_number(get_dimension(a)) > values::to_number(get_dimension(b));
    return false;
  }


  template<typename A, typename B, std::enable_if_t<
    descriptor<A> and descriptor<B> and not (values::value<A> and values::value<B>), int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b))
      return values::to_number(get_dimension(a)) <= values::to_number(get_dimension(b));
    return internal::get_descriptor_hash_code(a) == internal::get_descriptor_hash_code(b);
  }


  template<typename A, typename B, std::enable_if_t<
    descriptor<A> and descriptor<B> and not (values::value<A> and values::value<B>), int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    if (get_is_euclidean(a) and get_is_euclidean(b))
      return values::to_number(get_dimension(a)) >= values::to_number(get_dimension(b));
    return internal::get_descriptor_hash_code(a) == internal::get_descriptor_hash_code(b);
  }


  template<typename A, typename B, std::enable_if_t<
    descriptor<A> and descriptor<B> and not (values::value<A> and values::value<B>), int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }
#endif


} // namespace OpenKalman::coordinates


#endif

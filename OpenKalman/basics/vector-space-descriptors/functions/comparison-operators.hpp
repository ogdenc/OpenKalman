/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Implementation for \ref VectorSpaceDescriptorComparisonBase.
 */

#ifndef OPENKALMAN_COMPARISON_OPERATORS_HPP
#define OPENKALMAN_COMPARISON_OPERATORS_HPP

#include <type_traits>

#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif

namespace OpenKalman::vector_space_descriptors
{
  /**
   * Comparison operator for library-defined \ref vector_space_descriptor objects
   */
#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr auto operator<=>(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
    {
      if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
        return dimension_size_of_v<A> <=> dimension_size_of_v<B>;
      else if constexpr (equivalent_to<A, B>)
        return std::partial_ordering::equivalent;
      else if constexpr (internal::prefix_of<A, B>)
        return std::partial_ordering::less;
      else if constexpr (internal::prefix_of<B, A>)
        return std::partial_ordering::greater;
      else
        return std::partial_ordering::unordered;
    }
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
    {
      return static_cast<std::size_t>(get_dimension_size_of(a)) <=> static_cast<std::size_t>(get_dimension_size_of(b));
    }
    else
    {
      return std::partial_ordering::unordered;
    }
  }


  /**
   * Equality operator for library-defined \ref vector_space_descriptor objects
   */
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr bool operator==(const A& a, const B& b)
  {
    return std::is_eq(a <=> b);
  }
#else
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return equivalent_to<A, B>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return static_cast<std::size_t>(get_dimension_size_of(a)) == static_cast<std::size_t>(get_dimension_size_of(b));
    else
      return false;
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return internal::prefix_of<A, B>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return static_cast<std::size_t>(get_dimension_size_of(a)) <= static_cast<std::size_t>(get_dimension_size_of(b));
    else
      return false;
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return internal::prefix_of<B, A>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return static_cast<std::size_t>(get_dimension_size_of(a)) >= static_cast<std::size_t>(get_dimension_size_of(b));
    else
      return false;
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    return operator<=(a, b) and not operator==(a, b);
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    return operator>=(a, b) and not operator==(a, b);
  }
#endif


} // namespace OpenKalman::vector_space_descriptors


#endif //OPENKALMAN_COMPARISON_OPERATORS_HPP

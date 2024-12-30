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
 * \brief Comparison operators for \rev vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_COMPARISON_OPERATORS_HPP
#define OPENKALMAN_COMPARISON_OPERATORS_HPP

#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif
#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/are_equivalent.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/is_prefix.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Comparison operator for library-defined \ref vector_space_descriptor objects
   * \details Comparison of dynamic non-euclidean descriptors is defined elsewhere.
   * \todo Streamline this to avoid re-calculating prefix status
   */
#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr auto operator<=>(const A& a, const B& b)
  {
    if (internal::are_equivalent(a, b))
      return std::partial_ordering::equivalent;
    else if (internal::is_prefix(a, b))
      return std::partial_ordering::less;
    else if (internal::is_prefix(b, a))
      return std::partial_ordering::greater;
    else
      return std::partial_ordering::unordered;
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
    return internal::are_equivalent(a, b);
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    return internal::is_prefix(a, b);
  }

  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    return internal::is_prefix(b, a);
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


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_COMPARISON_OPERATORS_HPP

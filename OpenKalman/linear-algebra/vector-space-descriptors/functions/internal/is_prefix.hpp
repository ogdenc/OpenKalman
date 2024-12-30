/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Functions for \ref is_prefix.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_IS_PREFIX_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_IS_PREFIX_HPP

#include <type_traits>
#include <linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_is_euclidean.hpp>

#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/traits-defined.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "canonical_equivalent.hpp"
#include "are_equivalent.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief Whether <code>a</code> is a prefix of <code>b</code>.
   * \details A is a prefix of B if A matches at least the initial portion of B (including if A is equivalent to B).
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr std::convertible_to<bool> auto
#else
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr auto
#endif
  is_prefix(const A& a, const B& b)
  {
    if constexpr (dimension_size_of_v<A> == 0)
    {
      return std::true_type{};
    }
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
    {
      return value::operation{std::less_equal<>{}, get_dimension_size_of(a), get_dimension_size_of(b)};
    }
    else
    {
      using CA = std::decay_t<decltype(canonical_equivalent(a))>;
      using CB = std::decay_t<decltype(canonical_equivalent(b))>;

      if constexpr (static_vector_space_descriptor<CA> and std::is_same_v<CA, CB>)
        return std::true_type{};
      else if constexpr (interface::has_prefix_defined_for<CB, CA>)
        return interface::vector_space_traits<CB>::has_prefix(canonical_equivalent(b), canonical_equivalent(a));
      else if constexpr (euclidean_vector_space_descriptor<B>)
        return value::operation {
          std::logical_and<>{},
          get_vector_space_descriptor_is_euclidean(a),
          value::operation{std::less_equal<>{}, get_dimension_size_of(a), get_dimension_size_of(b)}};
      else
        return are_equivalent(a, b);
    }
  }


} // namespace OpenKalman::descriptor::internal


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_IS_PREFIX_HPP

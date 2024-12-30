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
 * \brief Definition for \ref are_equivalent.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_ARE_EQUIVALENT_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_ARE_EQUIVALENT_HPP

#include <type_traits>
#include "linear-algebra/values/classes/operation.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/traits-defined.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "canonical_equivalent.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief Whether <code>a</code> is equivalent to <code>b</code>.
   * \details A is equivalent to B if A and B are the same in their \ref internal::canonical_equivalent "canonical equivalent" forms.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr std::convertible_to<bool> auto
#else
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B>, int> = 0>
  constexpr auto
#endif
  are_equivalent(const A& a, const B& b)
  {
    if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
    {
      return value::operation {std::equal_to<>{}, get_dimension_size_of(a), get_dimension_size_of(b)};
    }
    else
    {
      using CA = std::decay_t<decltype(canonical_equivalent(a))>;
      using CB = std::decay_t<decltype(canonical_equivalent(b))>;

      if constexpr (static_vector_space_descriptor<CA> and std::is_same_v<CA, CB>)
      {
        return std::true_type{};
      }
      else if constexpr (interface::has_prefix_defined_for<CA, CB>)
      {
        auto ca = canonical_equivalent(a);
        auto cb = canonical_equivalent(b);
        return value::operation {
          std::logical_and<>{},
          interface::vector_space_traits<CA>::has_prefix(ca, cb),
          value::operation{std::equal_to<>{}, get_dimension_size_of(ca), get_dimension_size_of(cb)}};
      }
      else if constexpr (interface::has_prefix_defined_for<CB, CA>)
      {
        auto ca = canonical_equivalent(a);
        auto cb = canonical_equivalent(b);
        return value::operation {
          std::logical_and<>{},
          interface::vector_space_traits<CB>::has_prefix(cb, ca),
          value::operation{std::equal_to<>{}, get_dimension_size_of(ca), get_dimension_size_of(cb)}};
      }
      else
      {
        return std::false_type{};
      }
    }
  }


} // namespace OpenKalman::descriptor::internal


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_ARE_EQUIVALENT_HPP
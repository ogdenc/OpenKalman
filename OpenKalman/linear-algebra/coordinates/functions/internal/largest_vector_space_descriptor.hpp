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
 * \brief Definition for \ref internal::largest_vector_space_descriptor function.
 */

#ifndef OPENKALMAN_LARGEST_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_LARGEST_VECTOR_SPACE_DESCRIPTOR_HPP

#include "linear-algebra/coordinates/concepts/pattern.hpp"

namespace OpenKalman::coordinates::internal
{
  /**
   * \internal
   * \brief Return the largest \ref coordinates::pattern
   * \details If the dimensions are the same, this will return the earlier-listed one.
   * \tparam Scalar The scalar type, if the result is a DynamicDescriptor
   * \tparam V A \ref coordinates::pattern
   * \tparam Vs A set of \ref coordinates::pattern objects
   */
#ifdef __cpp_concepts
  template<values::number Scalar, pattern V, pattern...Vs>
  constexpr coordinates::pattern decltype(auto)
#else
  template<typename Scalar, typename V, typename...Vs, std::enable_if_t<
    values::number<Scalar> and (pattern<V> and ... and pattern<Vs>), int> = 0>
  constexpr decltype(auto)
#endif
  largest_vector_space_descriptor(V&& v, Vs&&...vs)
  {
    if constexpr (sizeof...(Vs) == 0)
    {
      return std::forward<V>(v);
    }
    else
    {
      decltype(auto) tail = largest_vector_space_descriptor<Scalar>(std::forward<Vs>(vs)...);

      if constexpr ((fixed_pattern<V> and fixed_pattern<decltype(tail)>))
      {
        if constexpr (dimension_of_v<V> >= dimension_of_v<decltype(tail)>)
          return std::forward<V>(v);
        else
          return std::forward<decltype(tail)>(tail);
      }
      else if constexpr (euclidean_pattern<V> and euclidean_pattern<decltype(tail)>)
      {
        return coordinates::Dimensions {std::max<std::size_t>(get_dimension(v), get_dimension(tail))};
      }
      else
      {
        if (get_dimension(v) >= get_dimension(tail))
          return coordinates::DynamicDescriptor<Scalar> {std::forward<V>(v)};
        else
          return coordinates::DynamicDescriptor<Scalar> {std::forward<decltype(tail)>(tail)};
      }
    }
  }

} // namespace OpenKalman::coordinates::internal

#endif //OPENKALMAN_LARGEST_VECTOR_SPACE_DESCRIPTOR_HPP

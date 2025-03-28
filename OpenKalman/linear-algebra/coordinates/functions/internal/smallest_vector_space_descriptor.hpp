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
 * \brief Definition for \ref internal::smallest_vector_space_descriptor function.
 */

#ifndef OPENKALMAN_SMALLEST_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_SMALLEST_VECTOR_SPACE_DESCRIPTOR_HPP

namespace OpenKalman::coordinate::internal
{
  /**
   * \internal
   * \brief Return the smallest \ref coordinate::pattern
   * \details If the dimensions are the same, this will return the earlier-listed one.
   * \tparam Scalars The scalar type, if the result is a DynamicDescriptor
   * \tparam V A \ref coordinate::pattern
   * \tparam Vs A set of \ref coordinate::pattern objects
   */
#ifdef __cpp_concepts
  template<value::number Scalar, coordinate::pattern V, coordinate::pattern...Vs>
  constexpr coordinate::pattern decltype(auto)
#else
  template<typename Scalar, typename V, typename...Vs, std::enable_if_t<
    value::number<Scalar> and (coordinate::pattern<V> and ... and coordinate::pattern<Vs>), int> = 0>
  constexpr decltype(auto)
#endif
  smallest_vector_space_descriptor(V&& v, Vs&&...vs)
  {
    if constexpr (sizeof...(Vs) == 0)
    {
      return std::forward<V>(v);
    }
    else
    {
      decltype(auto) tail = smallest_vector_space_descriptor<Scalar>(std::forward<Vs>(vs)...);

      if constexpr ((fixed_pattern<V> and fixed_pattern<decltype(tail)>))
      {
        if constexpr (size_of_v<V> <= size_of_v<decltype(tail)>)
          return std::forward<V>(v);
        else
          return std::forward<decltype(tail)>(tail);
      }
      else if constexpr (euclidean_pattern<V> and euclidean_pattern<decltype(tail)>)
      {
        return coordinate::Dimensions {std::min<std::size_t>(get_size(v), get_size(tail))};
      }
      else
      {
        if (get_size(v) <= get_size(tail))
          return coordinate::DynamicDescriptor<Scalar> {std::forward<V>(v)};
        else
          return coordinate::DynamicDescriptor<Scalar> {std::forward<decltype(tail)>(tail)};
      }
    }
  }

} // namespace OpenKalman::coordinate::internal

#endif //OPENKALMAN_SMALLEST_VECTOR_SPACE_DESCRIPTOR_HPP

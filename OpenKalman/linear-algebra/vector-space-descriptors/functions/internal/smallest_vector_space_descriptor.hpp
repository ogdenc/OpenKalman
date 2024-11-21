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

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Return the smallest \ref vector_space_descriptor
   * \details If the dimensions are the same, this will return the earlier-listed one.
   * \tparam Scalars The scalar type, if the result is a DynamicDescriptor
   * \tparam V A \ref vector_space_descriptor
   * \tparam Vs A set of \ref vector_space_descriptor objects
   */
#ifdef __cpp_concepts
  template<value::number Scalar, vector_space_descriptor V, vector_space_descriptor...Vs>
  constexpr vector_space_descriptor decltype(auto)
#else
  template<typename Scalar, typename V, typename...Vs, std::enable_if_t<
    value::number<Scalar> and (vector_space_descriptor<V> and ... and vector_space_descriptor<Vs>), int> = 0>
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

      if constexpr ((static_vector_space_descriptor<V> and static_vector_space_descriptor<decltype(tail)>))
      {
        if constexpr (dimension_size_of_v<V> <= dimension_size_of_v<decltype(tail)>)
          return std::forward<V>(v);
        else
          return std::forward<decltype(tail)>(tail);
      }
      else if constexpr (euclidean_vector_space_descriptor<V> and euclidean_vector_space_descriptor<decltype(tail)>)
      {
        return descriptors::Dimensions {std::min<std::size_t>(get_dimension_size_of(v), get_dimension_size_of(tail))};
      }
      else
      {
        if (get_dimension_size_of(v) <= get_dimension_size_of(tail))
          return descriptors::DynamicDescriptor<Scalar> {std::forward<V>(v)};
        else
          return descriptors::DynamicDescriptor<Scalar> {std::forward<decltype(tail)>(tail)};
      }
    }
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_SMALLEST_VECTOR_SPACE_DESCRIPTOR_HPP

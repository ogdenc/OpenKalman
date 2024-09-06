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

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Return the largest \ref vector_space_descriptor
   * \details If the dimensions are the same, this will return the earlier-listed one.
   * \tparam Scalars Allowable scalar types, if the result is a DynamicDescriptor
   * \tparam V A \ref vector_space_descriptor
   * \tparam Vs A set of \ref vector_space_descriptor objects
   */
#ifdef __cpp_concepts
  template<scalar_type...Scalars, vector_space_descriptor V, vector_space_descriptor...Vs>
  constexpr vector_space_descriptor decltype(auto)
#else
  template<typename V, typename...Vs, std::enable_if_t<(vector_space_descriptor<V> and ... and vector_space_descriptor<Vs>), int> = 0>
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
      decltype(auto) tail = largest_vector_space_descriptor<Scalars...>(std::forward<Vs>(vs)...);

      if constexpr ((fixed_vector_space_descriptor<V> and fixed_vector_space_descriptor<decltype(tail)>))
      {
        if constexpr (dimension_size_of_v<V> >= dimension_size_of_v<decltype(tail)>)
          return std::forward<V>(v);
        else
          return std::forward<decltype(tail)>(tail);
      }
      else if constexpr (euclidean_vector_space_descriptor<V> and euclidean_vector_space_descriptor<decltype(tail)>)
      {
        return Dimensions {std::max<std::size_t>(get_dimension_size_of(v), get_dimension_size_of(tail))};
      }
      else
      {
        if (get_dimension_size_of(v) >= get_dimension_size_of(tail))
          return DynamicDescriptor<Scalars...> {std::forward<V>(v)};
        else
          return DynamicDescriptor<Scalars...> {std::forward<decltype(tail)>(tail)};
      }
    }
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_LARGEST_VECTOR_SPACE_DESCRIPTOR_HPP

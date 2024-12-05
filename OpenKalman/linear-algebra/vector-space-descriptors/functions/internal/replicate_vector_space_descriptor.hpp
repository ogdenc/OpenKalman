/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition for \ref replicate_vector_space_descriptor.
 */

#ifndef OPENKALMAN_REPLICATE_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_REPLICATE_VECTOR_SPACE_DESCRIPTOR_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Replicate \ref static_vector_space_descriptor T some number of times.
   * \tparam Scalar The scalar type, if the result is a DynamicDescriptor
   */
#ifdef __cpp_concepts
  template<value::number Scalar, vector_space_descriptor T, value::index N>
#else
  template<typename...AllowableScalarTypes, typename T, typename N, std::enable_if_t<
    (value::number<AllowableScalarTypes> and ...) and vector_space_descriptor<T> and value::index<N>, int> = 0>
#endif
  auto replicate_vector_space_descriptor(const T& t, N n)
  {
    if constexpr (euclidean_vector_space_descriptor<T>)
    {
      if constexpr (static_vector_space_descriptor<T> and value::fixed<N>)
        return descriptor::Dimensions<dimension_size_of_v<T> * static_cast<std::size_t>(n)> {};
      else
        return get_dimension_size_of(t) * n;
    }
    else if constexpr (static_vector_space_descriptor<T> and value::fixed<N>)
    {
      return replicate_static_vector_space_descriptor_t<T, static_cast<std::size_t>(n)> {};
    }
    else
    {
      auto ret = descriptor::DynamicDescriptor<Scalar> {t};
      for (std::size_t i = 1; i < static_cast<std::size_t>(n); ++i) ret.extend(t);
      return ret;
    }
  }


} // namespace OpenKalman::internal


#endif //OPENKALMAN_REPLICATE_VECTOR_SPACE_DESCRIPTOR_HPP

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
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
   * \brief Replicate \ref fixed_vector_space_descriptor T some number of times.
   */
#ifdef __cpp_concepts
  template<scalar_type..., fixed_vector_space_descriptor T, static_index_value N>
#else
  template<typename...S, typename T, typename N, std::enable_if_t<(scalar_type<S> and ...) and
    fixed_vector_space_descriptor<T> and static_index_value<N>, int> = 0>
#endif
  auto replicate_vector_space_descriptor(const T& t, N n)
  {
    return replicate_fixed_vector_space_descriptor_t<T, n> {};
  }


  /**
   * \internal
   * \overload
   * \brief Replicate a (potentially) dynamic \ref vector_space_descriptor T some (potentially) dynamic number of times.
   */
#ifdef __cpp_concepts
  template<scalar_type...AllowableScalarTypes, vector_space_descriptor T, index_value N>
  requires (not fixed_vector_space_descriptor<T>) or (not static_index_value<N>)
#else
  template<typename...AllowableScalarTypes, typename T, typename N, std::enable_if_t<
    (scalar_type<AllowableScalarTypes> and ...) and vector_space_descriptor<T> and index_value<N> and
      (not fixed_vector_space_descriptor<T> or not static_index_value<N>), int> = 0>
#endif
  auto replicate_vector_space_descriptor(const T& t, N n)
  {
    auto ret = [](const T& t){
      if constexpr (sizeof...(AllowableScalarTypes) > 0) return DynamicTypedIndex<AllowableScalarTypes...> {t};
      else return DynamicTypedIndex {t};
    }(t);
    for (std::size_t i = 1; i < static_cast<std::size_t>(n); ++i) ret.extend(t);
    return ret;
  }


} // namespace OpenKalman::internal


#endif //OPENKALMAN_REPLICATE_VECTOR_SPACE_DESCRIPTOR_HPP

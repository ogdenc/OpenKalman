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
 * \file
 * \brief Arithmetic operators for \ref vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP

#include <type_traits>


namespace OpenKalman::vector_space_descriptors
{
  /**
   * \brief Add two sets of \ref vector_space_descriptor, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, vector_space_descriptor U> requires (not scalar_constant<T>) or (not scalar_constant<U>)
#else
  template<typename T, typename U, std::enable_if_t<vector_space_descriptor<T> and vector_space_descriptor<U> and
    (not scalar_constant<T> or not scalar_constant<U>), int> = 0>
#endif
  constexpr auto operator+(T&& t, U&& u)
  {
    if constexpr (static_vector_space_descriptor<T> and static_vector_space_descriptor<U>)
    {
      if constexpr (euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U>)
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
      else
        return concatenate_static_vector_space_descriptor_t<T, U> {};
    }
    else if constexpr (euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U>)
    {
      return Dimensions {get_dimension_size_of(t) + get_dimension_size_of(u)};
    }
    else
    {
      return DynamicDescriptor {std::forward<T>(t), std::forward<U>(u)};
    }
  }


  /**
   * \brief Subtract two \ref static_vector_space_descriptor values.
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor T, internal::suffix_of<T> U> requires (not scalar_constant<T>) or (not scalar_constant<U>)
#else
  template<typename T, typename U, std::enable_if_t<static_vector_space_descriptor<T> and internal::suffix_of<U, T> and
    not (scalar_constant<T> and scalar_constant<U>), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u)
  {
    return internal::base_of_t<U, T> {};
  }


  /**
   * \brief Subtract two \ref vector_space_descriptor values in which at least one is \ref dynamic_vector_space_descriptor "dynamic".
   */
#ifdef __cpp_concepts
  template<euclidean_vector_space_descriptor T, euclidean_vector_space_descriptor U> requires
    (dynamic_vector_space_descriptor<T> or dynamic_vector_space_descriptor<U>) and
    (not scalar_constant<T> or not scalar_constant<U>)
#else
  template<typename T, typename U, std::enable_if_t<
    euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U> and
    (dynamic_vector_space_descriptor<T> or dynamic_vector_space_descriptor<U>) and
    (not scalar_constant<T> or not scalar_constant<U>), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u)
  {
    if (get_dimension_size_of(t) < get_dimension_size_of(u))
      throw std::invalid_argument {"Subtraction of dynamic vector_space_descriptor values resulted in negative dimension"};

    return Dimensions{get_dimension_size_of(t) - get_dimension_size_of(u)};
  }


} // namespace OpenKalman::vector_space_descriptors


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP

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
 * \brief Arithmetic operators for \ref vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP

#include <type_traits>
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/static_concatenate.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/DynamicDescriptor.hpp" //


namespace OpenKalman::descriptor
{
  /**
   * \brief Add two sets of \ref vector_space_descriptor objects, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, vector_space_descriptor U> requires (not value::value<T>) or (not value::value<U>)
#else
  template<typename T, typename U, std::enable_if_t<vector_space_descriptor<T> and vector_space_descriptor<U> and
    (not value::value<T> or not value::value<U>), int> = 0>
#endif
  constexpr auto operator+(T&& t, U&& u)
  {
    if constexpr (static_vector_space_descriptor<T> and static_vector_space_descriptor<U>)
    {
      if constexpr (euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U>)
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
      else
        return static_concatenate_t<T, U> {};
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
  template<static_vector_space_descriptor T, static_vector_space_descriptor U> requires
    internal::prefix_of<static_reverse_t<U>, static_reverse_t<T>> and (not value::value<T>) or (not value::value<U>)
#else
  template<typename T, typename U, std::enable_if_t<static_vector_space_descriptor<T> and static_vector_space_descriptor<U> and
    internal::prefix_of<typename static_reverse<U>::type, typename static_reverse<T>::type> and not (value::scalar<T> and value::scalar<U>), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u)
  {
    return static_reverse_t<internal::prefix_base_of_t<static_reverse_t<T>, static_reverse_t<U>>> {};
  }


  /**
   * \brief Subtract two \ref vector_space_descriptor values in which at least one is \ref dynamic_vector_space_descriptor "dynamic".
   */
#ifdef __cpp_concepts
  template<euclidean_vector_space_descriptor T, euclidean_vector_space_descriptor U> requires
    (dynamic_vector_space_descriptor<T> or dynamic_vector_space_descriptor<U>) and
    (not value::value<T> or not value::value<U>)
#else
  template<typename T, typename U, std::enable_if_t<
    euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U> and
    (dynamic_vector_space_descriptor<T> or dynamic_vector_space_descriptor<U>) and
    (not value::value<T> or not value::value<U>), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u)
  {
    if (get_dimension_size_of(t) < get_dimension_size_of(u))
      throw std::invalid_argument {"Subtraction of dynamic vector_space_descriptor values resulted in negative dimension"};

    return Dimensions{get_dimension_size_of(t) - get_dimension_size_of(u)};
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP
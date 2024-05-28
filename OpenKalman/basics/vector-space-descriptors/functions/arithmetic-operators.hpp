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
 * \file
 * \brief Arithmetic operators for \ref vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
    concept vector_space_descriptor_arithmetic_defined =
      interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::operations_defined or interface::dynamic_vector_space_descriptor_traits<std::decay_t<T>>::operations_defined;
#else
    template<typename T, typename = void>
    struct fixed_vector_space_descriptor_arithmetic_defined : std::false_type {};

    template<typename T>
    struct fixed_vector_space_descriptor_arithmetic_defined<T, std::enable_if_t<
      interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::operations_defined>> : std::true_type {};

    template<typename T, typename = void>
    struct dynamic_vector_space_descriptor_arithmetic_defined : std::false_type {};

    template<typename T>
    struct dynamic_vector_space_descriptor_arithmetic_defined<T, std::enable_if_t<
      interface::dynamic_vector_space_descriptor_traits<std::decay_t<T>>::operations_defined>> : std::true_type {};

    template<typename T>
    constexpr bool vector_space_descriptor_arithmetic_defined =
      fixed_vector_space_descriptor_arithmetic_defined<std::decay_t<T>>::value or dynamic_vector_space_descriptor_arithmetic_defined<std::decay_t<T>>::value;
#endif
  } // namespace detail


  /**
   * \brief Add two sets of \ref vector_space_descriptor, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, vector_space_descriptor U> requires
    detail::vector_space_descriptor_arithmetic_defined<T> or detail::vector_space_descriptor_arithmetic_defined<U>
#else
  template<typename T, typename U, std::enable_if_t<vector_space_descriptor<T> and vector_space_descriptor<U> and
    (detail::vector_space_descriptor_arithmetic_defined<T> or detail::vector_space_descriptor_arithmetic_defined<U>), int> = 0>
#endif
  constexpr auto operator+(T&& t, U&& u)
  {
    if constexpr (fixed_vector_space_descriptor<T> and fixed_vector_space_descriptor<U>)
    {
      return concatenate_fixed_vector_space_descriptor_t<T, U> {};
    }
    else if constexpr (euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U>)
    {
      if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
        return Dimensions{get_dimension_size_of(t) + get_dimension_size_of(u)};
      else
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
    }
    else
    {
      return DynamicDescriptor {std::forward<T>(t), std::forward<U>(u)};
    }
  }


  /**
   * \brief Subtract two \ref euclidean_vector_space_descriptor values, whether fixed or dynamic.
   * \warning This does not perform any runtime checks to ensure that the result is non-negative.
   */
#ifdef __cpp_concepts
  template<euclidean_vector_space_descriptor T, euclidean_vector_space_descriptor U> requires (dimension_size_of_v<T> == dynamic_size or
      dimension_size_of_v<U> == dynamic_size or dimension_size_of_v<T> > dimension_size_of_v<U>) and
    (detail::vector_space_descriptor_arithmetic_defined<T> or detail::vector_space_descriptor_arithmetic_defined<U>)
#else
  template<typename T, typename U, std::enable_if_t<euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U> and
    (dimension_size_of<T>::value == dynamic_size or dimension_size_of<U>::value == dynamic_size or
      dimension_size_of<T>::value > dimension_size_of<U>::value) and
      (detail::vector_space_descriptor_arithmetic_defined<T> or detail::vector_space_descriptor_arithmetic_defined<U>), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u) noexcept
  {
    if constexpr (dynamic_vector_space_descriptor<T> or dynamic_vector_space_descriptor<U>)
      return Dimensions{get_dimension_size_of(t) - get_dimension_size_of(u)};
    else
      return Dimensions<dimension_size_of_v<T> - dimension_size_of_v<U>>{};
  }


} // namespace OpenKalman


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP

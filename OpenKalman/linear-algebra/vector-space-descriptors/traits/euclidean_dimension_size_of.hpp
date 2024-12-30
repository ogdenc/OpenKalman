/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref euclidean_dimension_size_of.
 */

#ifndef OPENKALMAN_EUCLIDEAN_DIMENSION_SIZE_OF_HPP
#define OPENKALMAN_EUCLIDEAN_DIMENSION_SIZE_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include <linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp>


namespace OpenKalman::descriptor
{
  /**
   * \brief The dimension size of a set of \ref vector_space_descriptor if it is transformed into Euclidean space.
   * \details The associated static member <code>value</code> is the size of the \ref vector_space_descriptor when transformed
   * to Euclidean space, or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct euclidean_dimension_size_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
  struct euclidean_dimension_size_of<T>
#else
  template<typename T>
  struct euclidean_dimension_size_of<T, std::enable_if_t<static_vector_space_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, value::to_number(get_euclidean_dimension_size_of(T{}))> {};


  /**
   * \brief Helper template for \ref euclidean_dimension_size_of.
   */
  template<typename T>
  constexpr auto euclidean_dimension_size_of_v = euclidean_dimension_size_of<T>::value;


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_EUCLIDEAN_DIMENSION_SIZE_OF_HPP

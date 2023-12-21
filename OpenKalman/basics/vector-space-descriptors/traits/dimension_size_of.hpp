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
 * \brief Definition for \ref dimension_size_of.
 */

#ifndef OPENKALMAN_DIMENSION_SIZE_OF_HPP
#define OPENKALMAN_DIMENSION_SIZE_OF_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * \brief The dimension size of a set of \ref vector_space_descriptor.
   * \details The associated static member <code>value</code> is the size of the \ref vector_space_descriptor,
   * or \ref dynamic_size if not known at compile time.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_size_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    : std::integral_constant<std::size_t, interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::size> {};


  /**
   * \brief Helper template for \ref dimension_size_of.
   */
  template<typename T>
  constexpr auto dimension_size_of_v = dimension_size_of<T>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_DIMENSION_SIZE_OF_HPP

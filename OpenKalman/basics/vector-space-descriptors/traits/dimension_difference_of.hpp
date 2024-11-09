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
 * \brief Definition for \ref dimension_difference_of.
 */

#ifndef OPENKALMAN_DIMENSION_DIFFERENCE_OF_HPP
#define OPENKALMAN_DIMENSION_DIFFERENCE_OF_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * \brief The type of the \ref vector_space_descriptor object when tensors having respective vector_space_descriptor T are subtracted.
   * \details The associated alias <code>type</code> is the difference type.
   * For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Axis,
   * so if <code>T</code> is Distance, the resulting <code>type</code> will be Dimensions<1>.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_difference_of { using type = std::decay_t<T>; };


#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
  struct dimension_difference_of<T>
#else
  template<typename T>
  struct dimension_difference_of<T, std::enable_if_t<static_vector_space_descriptor<T>>>
#endif
  { using type = typename static_vector_space_descriptor_traits<std::decay_t<T>>::difference_type; };


  /**
   * \brief Helper template for \ref dimension_difference_of.
   */
  template<typename T>
  using dimension_difference_of_t = typename dimension_difference_of<std::decay_t<T>>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_DIMENSION_DIFFERENCE_OF_HPP

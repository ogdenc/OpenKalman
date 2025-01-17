/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref scalar_type_of.
 */

#ifndef OPENKALMAN_DESCRIPTORS_SCALAR_TYPE_OF_HPP
#define OPENKALMAN_DESCRIPTORS_SCALAR_TYPE_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief The scalar type, if any, associated with a \ref vector_space_descriptor.
   * \details Defaults to <code>double</code>.
   * This may not have meaning if <code>\ref static_vector_space_descriptor "static_vector_space_descriptor<T>"</code>.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct scalar_type_of { using type = double; };


#ifdef __cpp_concepts
  template<typename T> requires requires { typename interface::vector_space_traits<std::decay_t<T>>::scalar_type; }
  struct scalar_type_of<T>
#else
  template<typename T>
  struct scalar_type_of<T, std::void_t<typename interface::vector_space_traits<std::decay_t<T>>::scalar_type>>
#endif
  {
    using type = typename interface::vector_space_traits<std::decay_t<T>>::scalar_type;
  };


  /**
   * \brief Helper template for \ref scalar_type_of.
   */
  template<typename T>
  using scalar_type_of_t = typename scalar_type_of<T>::type;


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_DESCRIPTORS_SCALAR_TYPE_OF_HPP

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
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_range.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief The scalar type, if any, associated with a \ref coordinate::pattern.
   * \details Defaults to <code>double</code>.
   * This may not have meaning if <code>\ref fixed_pattern "fixed_pattern<T>"</code>.
   */
#ifdef __cpp_concepts
  template<pattern T>
#else
  template<typename T, typename = void, typename = void>
#endif
  struct scalar_type_of { using type = double; };


  /**
   * \brief Helper template for \ref scalar_type_of.
   */
  template<typename T>
  using scalar_type_of_t = typename scalar_type_of<T>::type;


#ifdef __cpp_concepts
  template<descriptor T> requires
    requires { typename interface::coordinate_descriptor_traits<std::decay_t<T>>::scalar_type; }
  struct scalar_type_of<T>
#else
  template<typename T>
  struct scalar_type_of<T, std::enable_if_t<descriptor<T>>,
    std::void_t<typename interface::coordinate_descriptor_traits<std::decay_t<T>>::scalar_type>>
#endif
  {
    using type = typename interface::coordinate_descriptor_traits<std::decay_t<T>>::scalar_type;
  };


#ifdef __cpp_concepts
  template<descriptor_range T> requires
    requires { typename interface::coordinate_set_traits<std::decay_t<T>>::scalar_type; }
  struct scalar_type_of<T>
#else
  template<typename T>
  struct scalar_type_of<T, std::enable_if_t<descriptor_range<T>>,
    std::void_t<typename interface::coordinate_set_traits<std::decay_t<T>>::scalar_type>>
#endif
  {
    using type = typename interface::coordinate_set_traits<std::decay_t<T>>::scalar_type;
  };


#ifdef __cpp_concepts
  template<descriptor_tuple T> requires (not descriptor_range<T>) and
    requires { typename interface::coordinate_set_traits<std::decay_t<T>>::scalar_type; }
  struct scalar_type_of<T>
#else
  template<typename T>
  struct scalar_type_of<T, std::enable_if_t<descriptor_tuple<T>>,
    std::void_t<typename interface::coordinate_set_traits<std::decay_t<T>>::scalar_type>>
#endif
  {
    using type = typename interface::coordinate_set_traits<std::decay_t<T>>::scalar_type;
  };


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_DESCRIPTORS_SCALAR_TYPE_OF_HPP

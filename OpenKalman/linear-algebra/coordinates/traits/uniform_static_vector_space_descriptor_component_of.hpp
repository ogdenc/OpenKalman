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
 * \brief Definition for \ref uniform_static_vector_space_descriptor_component_of.
 */

#ifndef OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_COMPONENT_OF_HPP
#define OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_COMPONENT_OF_HPP

#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/traits/internal/uniform_static_vector_space_descriptor_query.hpp"
#include "linear-algebra/coordinates/concepts/uniform_static_vector_space_descriptor.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief If T is a \ref uniform_static_vector_space_descriptor, <code>type</code> is an alias for the uniform component.
   * \sa uniform_static_vector_space_descriptor_component_of_t
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct uniform_static_vector_space_descriptor_component_of;


#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename>
#endif
  struct uniform_static_vector_space_descriptor_component_of {};


#ifdef __cpp_concepts
  template<uniform_static_vector_space_descriptor T> requires euclidean_pattern<T>
  struct uniform_static_vector_space_descriptor_component_of<T>
#else
  template<typename T>
  struct uniform_static_vector_space_descriptor_component_of<T, std::enable_if_t<uniform_static_vector_space_descriptor<T> and euclidean_pattern<T>>>
#endif
  {
    using type = coordinate::Dimensions<1>;
  };


#ifdef __cpp_concepts
  template<uniform_static_vector_space_descriptor T> requires (not euclidean_pattern<T>)
  struct uniform_static_vector_space_descriptor_component_of<T>
#else
  template<typename T>
  struct uniform_static_vector_space_descriptor_component_of<T, std::enable_if_t<uniform_static_vector_space_descriptor<T> and not euclidean_pattern<T>>>
#endif
  {
    using CT = std::decay_t<decltype(internal::get_component_collection(std::declval<T>()))>;
    using type = typename internal::uniform_static_vector_space_descriptor_query<CT>::uniform_type;
  };


  /**
   * \brief Helper template for \ref uniform_static_vector_space_descriptor_component_of.
   */
#ifdef __cpp_concepts
  template<uniform_static_vector_space_descriptor T>
#else
  template<typename T>
#endif
  using uniform_static_vector_space_descriptor_component_of_t = typename uniform_static_vector_space_descriptor_component_of<T>::type;


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_COMPONENT_OF_HPP

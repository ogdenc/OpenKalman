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

#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/canonical_equivalent.hpp"
#include "linear-algebra/vector-space-descriptors/traits/internal/uniform_static_vector_space_descriptor_query.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/uniform_static_vector_space_descriptor.hpp"

namespace OpenKalman::descriptor
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
  template<uniform_static_vector_space_descriptor T> requires euclidean_vector_space_descriptor<T>
  struct uniform_static_vector_space_descriptor_component_of<T>
#else
  template<typename T>
  struct uniform_static_vector_space_descriptor_component_of<T, std::enable_if_t<uniform_static_vector_space_descriptor<T> and euclidean_vector_space_descriptor<T>>>
#endif
  {
    using type = descriptor::Dimensions<1>;
  };


#ifdef __cpp_concepts
  template<uniform_static_vector_space_descriptor T> requires (not euclidean_vector_space_descriptor<T>)
  struct uniform_static_vector_space_descriptor_component_of<T>
#else
  template<typename T>
  struct uniform_static_vector_space_descriptor_component_of<T, std::enable_if_t<uniform_static_vector_space_descriptor<T> and not euclidean_vector_space_descriptor<T>>>
#endif
  {
    using CT = std::decay_t<decltype(internal::canonical_equivalent(std::declval<T>()))>;
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


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_COMPONENT_OF_HPP

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
 * \brief Definition for \ref uniform_fixed_vector_space_descriptor_component_of.
 */

#ifndef OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_COMPONENT_OF_HPP
#define OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_COMPONENT_OF_HPP


namespace OpenKalman
{
  /**
   * \brief If T is a \ref uniform_fixed_vector_space_descriptor, <code>type</code> is an alias for the uniform component.
   * \sa uniform_fixed_vector_space_descriptor_component_of_t
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct uniform_fixed_vector_space_descriptor_component_of;


#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename>
#endif
  struct uniform_fixed_vector_space_descriptor_component_of {};


#ifdef __cpp_concepts
  template<uniform_fixed_vector_space_descriptor T> requires euclidean_vector_space_descriptor<T>
  struct uniform_fixed_vector_space_descriptor_component_of<T>
#else
  template<typename T>
  struct uniform_fixed_vector_space_descriptor_component_of<T, std::enable_if_t<uniform_fixed_vector_space_descriptor<T> and euclidean_vector_space_descriptor<T>>>
#endif
  {
    using type = Dimensions<1>;
  };


#ifdef __cpp_concepts
  template<uniform_fixed_vector_space_descriptor T> requires (not euclidean_vector_space_descriptor<T>)
  struct uniform_fixed_vector_space_descriptor_component_of<T>
#else
  template<typename T>
  struct uniform_fixed_vector_space_descriptor_component_of<T, std::enable_if_t<uniform_fixed_vector_space_descriptor<T> and not euclidean_vector_space_descriptor<T>>>
#endif
  {
    using type = typename internal::uniform_fixed_vector_space_descriptor_query<canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>>::uniform_type;
  };


  /**
   * \brief Helper template for \ref uniform_fixed_vector_space_descriptor_component_of.
   */
#ifdef __cpp_concepts
  template<uniform_fixed_vector_space_descriptor T>
#else
  template<typename T>
#endif
  using uniform_fixed_vector_space_descriptor_component_of_t = typename uniform_fixed_vector_space_descriptor_component_of<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_COMPONENT_OF_HPP

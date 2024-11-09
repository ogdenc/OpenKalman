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
 * \brief Definition for \ref canonical_static_vector_space_descriptor.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * \brief Concatenate any number of StaticDescriptor<...> types.
   * \details
   * Example:
   * - \code concatenate_static_vector_space_descriptor_t<StaticDescriptor<angle::Radians>,
   * StaticDescriptor<Axis, Distance>> == StaticDescriptor<angle::Radians, Axis, Distance> \endcode.
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs>
#else
  template<typename...Cs>
#endif
  struct concatenate_static_vector_space_descriptor;


  /**
   * \brief Helper template for \ref concatenate_static_vector_space_descriptor.
   */
  template<typename...Cs>
  using concatenate_static_vector_space_descriptor_t = typename concatenate_static_vector_space_descriptor<Cs...>::type;


  /**
   * \brief Reduce a \ref static_vector_space_descriptor into its expanded canonical form.
   * \sa canonical_static_vector_space_descriptor_t
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct canonical_static_vector_space_descriptor;


  /**
   * \brief Helper template for \ref canonical_static_vector_space_descriptor.
   */
  template<typename T>
  using canonical_static_vector_space_descriptor_t = typename canonical_static_vector_space_descriptor<T>::type;


  /**
   * \brief Reverse the order of a \ref vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
#else
  template<typename T>
#endif
  struct reverse_static_vector_space_descriptor;


  /**
   * \brief Helper template for \ref reverse_static_vector_space_descriptor.
   */
  template<typename T>
  using reverse_static_vector_space_descriptor_t = typename reverse_static_vector_space_descriptor<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP

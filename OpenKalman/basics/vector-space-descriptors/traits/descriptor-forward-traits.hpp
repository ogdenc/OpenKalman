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
 * \brief Definition for \ref canonical_fixed_vector_space_descriptor.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * \brief Concatenate any number of FixedDescriptor<...> types.
   * \details
   * Example:
   * - \code concatenate_fixed_vector_space_descriptor_t<FixedDescriptor<angle::Radians>,
   * FixedDescriptor<Axis, Distance>> == FixedDescriptor<angle::Radians, Axis, Distance> \endcode.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor...Cs>
#else
  template<typename...Cs>
#endif
  struct concatenate_fixed_vector_space_descriptor;


  /**
   * \brief Helper template for \ref concatenate_fixed_vector_space_descriptor.
   */
  template<typename...Cs>
  using concatenate_fixed_vector_space_descriptor_t = typename concatenate_fixed_vector_space_descriptor<Cs...>::type;


  /**
   * \brief Reduce a \ref fixed_vector_space_descriptor into its canonical form.
   * \sa canonical_fixed_vector_space_descriptor_t
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct canonical_fixed_vector_space_descriptor;


  /**
   * \brief Helper template for \ref canonical_fixed_vector_space_descriptor.
   */
  template<typename T>
  using canonical_fixed_vector_space_descriptor_t = typename canonical_fixed_vector_space_descriptor<T>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_FORWARD_DECLARATIONS_HPP

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Forward declarations relating to \ref vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_FORWARD_DECLARATIONS_HPP

#include <numeric>
#include "basics/global-definitions.hpp"
#include "linear-algebra/values/values.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief A structure representing the dimensions associated with of a particular index.
   * \details The dimension may or may not be known at compile time. If unknown at compile time, the size is set
   * at the time of construction and cannot be modified thereafter.
   * \tparam size The dimension (or <code>dynamic_size</code>, if not known at compile time)
   */
  template<std::size_t size = dynamic_size>
  struct Dimensions;


  /**
   * \brief A composite \ref static_vector_space_descriptor comprising a sequence of other fixed \ref vector_space_descriptor.
   * \details This is the key to the wrapping functionality of OpenKalman. Each of the static_vector_space_descriptor Cs... matches-up with
   * one or more of the rows or columns of a matrix. The number of coefficients per coefficient depends on the dimension
   * of the coefficient. For example, Axis, Distance, Angle, and Inclination are dimension 1, and each correspond to a
   * single coefficient. Polar is dimension 2 and corresponds to two coefficients (e.g., a distance and an angle).
   * Spherical is dimension 3 and corresponds to three coefficients.
   * Example: <code>StaticDescriptor&lt;Axis, angle::Radians&gt;</code>
   * \tparam Cs Any types within the concept coefficients.
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs>
#else
  template<typename...Cs>
#endif
  struct StaticDescriptor;


  /**
   * \brief A dynamic list of \ref atomic_vector_space_descriptor objects that can be defined or extended at runtime.
   * \details At compile time, the structure is treated if it has zero dimension.
   * \tparam Scalar The scalar type for elements associated with this object.
   */
#ifdef __cpp_concepts
  template<value::number Scalar>
#else
  template<typename Scalar>
#endif
  struct DynamicDescriptor;


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_FORWARD_DECLARATIONS_HPP

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
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

#include <array>
#include <functional>
#include <numeric>
#include "basics/values/values.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"

namespace OpenKalman::descriptors
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
   * \brief A dynamic list of \ref atomic_static_vector_space_descriptor objects that can be defined or extended at runtime.
   * \details At compile time, the structure is treated if it has zero dimension.
   * \tparam Scalar The scalar type for elements associated with this object.
   */
#ifdef __cpp_concepts
  template<value::number Scalar>
#else
  template<typename Scalar>
#endif
  struct DynamicDescriptor;


  namespace internal
  {
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

  } // namespace internal


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


  }


}// namespace OpenKalman::descriptors


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_FORWARD_DECLARATIONS_HPP

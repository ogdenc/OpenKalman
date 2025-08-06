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
 * \brief Definition for \ref uniform_pattern_type.
 */

#ifndef OPENKALMAN_UNIFORM_PATTERN_TYPE_HPP
#define OPENKALMAN_UNIFORM_PATTERN_TYPE_HPP

#include "collections/collections.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/traits/common_descriptor_type.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief If T is a \ref uniform_pattern, <code>type</code> is an alias for the uniform component.
   * \details The uniform component is either \ref coordinates::Dimension "Dimension<1>" if T is a
   * \ref euclidean_pattern or \ref common_descriptor_type_t, which will be a 1D \ref descriptor.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct uniform_pattern_type {};


#ifndef __cpp_concepts
  namespace details
  {
    template<typename T, typename = void>
    struct common_descriptor_dimension_is_1 : std::false_type {};

    template<typename T>
    struct common_descriptor_dimension_is_1<T, std::void_t<typename common_descriptor_type<T>::type>>
      : std::bool_constant<values::fixed_number_compares_with<dimension_of<common_descriptor_type_t<T>>, 1>> {};
  }
#endif


  /// \overload
#ifdef __cpp_concepts
  template<pattern T> requires
    euclidean_pattern<T> or
    values::fixed_number_compares_with<dimension_of<T>, 1> or
    values::fixed_number_compares_with<dimension_of<typename common_descriptor_type<T>::type>, 1>
  struct uniform_pattern_type<T>
#else
  template<typename T>
  struct uniform_pattern_type<T, std::enable_if_t<pattern<T> and
    (euclidean_pattern<T> or
      values::fixed_number_compares_with<dimension_of<T>, 1> or
      details::common_descriptor_dimension_is_1<T>::value)>>
#endif
    : std::conditional<
      euclidean_pattern<T>,
      Dimensions<1>,
      typename std::conditional_t<
        euclidean_pattern<T> or values::fixed_number_compares_with<dimension_of<T>, 1>,
        std::decay<T>,
        common_descriptor_type<T>>::type> {};


  /**
   * \brief Helper template for \ref uniform_pattern_type.
   */
  template<typename T>
  using uniform_pattern_type_t = typename uniform_pattern_type<T>::type;


}

#endif

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref composite_vector_space_descriptor.
 */

#ifndef OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"


namespace OpenKalman::descriptor
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T typename = void>
#endif
    struct is_composite_vector_space_descriptor : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires interface::vector_space_traits<T>::is_composite
    struct is_composite_vector_space_descriptor<T>
#else
    template<typename T>
    struct is_composite_vector_space_descriptor<T, std::enable_if_t<interface::vector_space_traits<T>::is_composite>>
#endif
      : std::true_type {};
  }


  /**
   * \brief T is a \ref composite_vector_space_descriptor.
   * \details A composite \ref vector_space_descriptor object is a container for other \ref vector_space_descriptor, and can either be
   * StaticDescriptor or DynamicDescriptor.
   * \sa StaticDescriptor, DynamicDescriptor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept composite_vector_space_descriptor =
#else
  constexpr bool composite_vector_space_descriptor =
#endif
    vector_space_descriptor<T> and detail::is_composite_vector_space_descriptor<std::decay_t<T>>::value;


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP

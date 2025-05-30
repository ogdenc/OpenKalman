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
 * \brief Definition for \ref uniform_static_vector_space_descriptor.
 */

#ifndef OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "fixed_pattern.hpp"
#include "dynamic_pattern.hpp"
#include "linear-algebra/coordinates/traits/internal/uniform_static_vector_space_descriptor_query.hpp"


namespace OpenKalman::coordinates
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct uniform_static_vector_space_descriptor_impl : std::false_type {};


#ifdef __cpp_concepts
    template<fixed_pattern T>
    struct uniform_static_vector_space_descriptor_impl<T>
#else
    template<typename T>
    struct uniform_static_vector_space_descriptor_impl<T, std::enable_if_t<fixed_pattern<T>>>
#endif
      : internal::uniform_static_vector_space_descriptor_query<std::decay_t<decltype(internal::get_component_collection(std::declval<T>()))>> {};


#ifdef __cpp_concepts
    template<dynamic_pattern T>
    struct uniform_static_vector_space_descriptor_impl<T>
#else
    template<typename T>
    struct uniform_static_vector_space_descriptor_impl<T, std::enable_if_t<dynamic_pattern<T>>>
#endif
      : internal::uniform_static_vector_space_descriptor_query<std::decay_t<T>> {};

  } // namespace detail


  /**
   * \brief T is a \ref coordinates::pattern that can be decomposed into a uniform set of 1D \ref coordinates::pattern.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept uniform_static_vector_space_descriptor =
#else
  constexpr bool uniform_static_vector_space_descriptor =
#endif
    detail::uniform_static_vector_space_descriptor_impl<T>::value;


} // namespace OpenKalman::coordinates

#endif //OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

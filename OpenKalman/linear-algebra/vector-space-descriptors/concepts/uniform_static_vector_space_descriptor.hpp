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
#include "static_vector_space_descriptor.hpp"
#include "dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/canonical_equivalent.hpp"
#include "linear-algebra/vector-space-descriptors/traits/internal/uniform_static_vector_space_descriptor_query.hpp"


namespace OpenKalman::descriptor
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
    template<static_vector_space_descriptor T>
    struct uniform_static_vector_space_descriptor_impl<T>
#else
    template<typename T>
    struct uniform_static_vector_space_descriptor_impl<T, std::enable_if_t<static_vector_space_descriptor<T>>>
#endif
      : internal::uniform_static_vector_space_descriptor_query<std::decay_t<decltype(internal::canonical_equivalent(std::declval<T>()))>> {};


#ifdef __cpp_concepts
    template<dynamic_vector_space_descriptor T>
    struct uniform_static_vector_space_descriptor_impl<T>
#else
    template<typename T>
    struct uniform_static_vector_space_descriptor_impl<T, std::enable_if_t<dynamic_vector_space_descriptor<T>>>
#endif
      : internal::uniform_static_vector_space_descriptor_query<std::decay_t<T>> {};

  } // namespace detail


  /**
   * \brief T is a \ref vector_space_descriptor that can be decomposed into a uniform set of 1D \ref vector_space_descriptor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept uniform_static_vector_space_descriptor =
#else
  constexpr bool uniform_static_vector_space_descriptor =
#endif
    detail::uniform_static_vector_space_descriptor_impl<T>::value;


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

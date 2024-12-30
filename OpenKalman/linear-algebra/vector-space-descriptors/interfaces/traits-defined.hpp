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
 * \brief Concepts for testing whether \ref vector_space_traits are defined for a particular \ref vector_space_descriptor.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TRAITS_DEFINED_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TRAITS_DEFINED_HPP

#include "vector_space_traits.hpp"

namespace OpenKalman::interface
{
#ifdef __cpp_concepts
  template<typename T>
  concept canonical_equivalent_defined_for = requires (T t) { interface::vector_space_traits<std::decay_t<T>>::canonical_equivalent(t); };
#else
  template<typename T, typename = void>
  struct canonical_equivalent_defined_for_impl : std::false_type {};

  template<typename T>
  struct canonical_equivalent_defined_for_impl<T, std::void_t<
    decltype(interface::vector_space_traits<std::decay_t<T>>::canonical_equivalent(std::declval<T>()))>> : std::true_type {};

  template<typename T>
  constexpr bool canonical_equivalent_defined_for = canonical_equivalent_defined_for_impl<T>::value;
#endif


#ifdef __cpp_concepts
  template<typename A, typename B>
  concept has_prefix_defined_for = requires (A a, B b) { interface::vector_space_traits<std::decay_t<A>>::has_prefix(a, b); };
#else
  template<typename A, typename B, typename = void>
  struct has_prefix_defined_for_impl : std::false_type {};

  template<typename A, typename B>
  struct has_prefix_defined_for_impl<A, B, std::void_t<
    decltype(interface::vector_space_traits<std::decay_t<A>>::has_prefix(std::declval<A>(), std::declval<B>()))>> : std::true_type {};

  template<typename A, typename B>
  constexpr bool has_prefix_defined_for = has_prefix_defined_for_impl<A, B>::value;
#endif


} // namespace OpenKalman::interface


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TRAITS_DEFINED_HPP

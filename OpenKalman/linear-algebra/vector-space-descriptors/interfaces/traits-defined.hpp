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
  concept collection_defined_for = requires (T t) { interface::vector_space_traits<std::decay_t<T>>::collection(t); };
#else
  template<typename T, typename = void>
  struct collection_defined_for_impl : std::false_type {};

  template<typename T>
  struct collection_defined_for_impl<T, std::void_t<
    decltype(interface::vector_space_traits<std::decay_t<T>>::collection(std::declval<T>()))>> : std::true_type {};

  template<typename T>
  constexpr bool collection_defined_for = collection_defined_for_impl<T>::value;
#endif


#ifdef __cpp_concepts
  template<typename T>
  concept type_index_defined_for = requires (T t) { interface::vector_space_traits<std::decay_t<T>>::type_index(t); };
#else
  template<typename T, typename = void>
  struct type_index_defined_for_impl : std::false_type {};

  template<typename T>
  struct type_index_defined_for_impl<T, std::void_t<
  decltype(interface::vector_space_traits<std::decay_t<T>>::type_index(std::declval<T>()))>> : std::true_type {};

  template<typename T>
  constexpr bool type_index_defined_for = type_index_defined_for_impl<T>::value;
#endif

} // namespace OpenKalman::interface


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TRAITS_DEFINED_HPP

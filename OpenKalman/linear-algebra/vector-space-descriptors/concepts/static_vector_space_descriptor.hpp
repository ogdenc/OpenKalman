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
 * \brief Definition for \ref static_vector_space_descriptor.
 */

#ifndef OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"

namespace OpenKalman::descriptor
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_static_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_static_vector_space_descriptor<T, std::enable_if_t<
      std::is_default_constructible<T>::value and
      value::index<decltype(interface::vector_space_traits<T>::size(std::declval<T>()))> and
      value::fixed<decltype(interface::vector_space_traits<T>::size(std::declval<T>()))>
      >> : std::true_type {};
  }
#endif


  /**
   * \brief A set of \ref vector_space_descriptor for which the number of dimensions is fixed at compile time.
   * \details This includes any object for which interface::vector_space_traits is defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept static_vector_space_descriptor = std::default_initializable<std::decay_t<T>> and
    requires(const std::decay_t<T>& t) {
      {interface::vector_space_traits<std::decay_t<T>>::size(t)} -> value::index;
      {interface::vector_space_traits<std::decay_t<T>>::size(t)} -> value::fixed;
    };
#else
  constexpr bool static_vector_space_descriptor = detail::is_static_vector_space_descriptor<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_STATIC_VECTOR_SPACE_DESCRIPTOR_HPP

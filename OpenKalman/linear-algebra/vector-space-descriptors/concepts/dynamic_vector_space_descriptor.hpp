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
 * \brief Definition for \ref dynamic_vector_space_descriptor.
 */

#ifndef OPENKALMAN_DYNAMIC_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_DYNAMIC_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"

namespace OpenKalman::descriptor
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_dynamic_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_dynamic_vector_space_descriptor<T, std::enable_if_t<
      value::index<decltype(interface::vector_space_traits<T>::size(std::declval<T>()))> and
      value::dynamic<decltype(interface::vector_space_traits<T>::size(std::declval<T>()))>
      >> : std::true_type {};
  }
#endif


  /**
   * \brief A set of \ref vector_space_descriptor for which the number of dimensions is defined at runtime.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_vector_space_descriptor =
    requires(const std::decay_t<T>& t) {
        {interface::vector_space_traits<std::decay_t<T>>::size(t)} -> value::index;
        {interface::vector_space_traits<std::decay_t<T>>::size(t)} -> value::dynamic;
    };
#else
  constexpr bool dynamic_vector_space_descriptor = detail::is_dynamic_vector_space_descriptor<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_DYNAMIC_VECTOR_SPACE_DESCRIPTOR_HPP

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
 * \brief Definition for \ref fixed_vector_space_descriptor.
 */

#ifndef OPENKALMAN_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_fixed_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_fixed_vector_space_descriptor<T, std::enable_if_t<
      std::is_default_constructible<std::decay_t<T>>::value and
      std::is_convertible<decltype(interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::size), std::size_t>::value and
      std::is_convertible<decltype(interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::euclidean_size), std::size_t>::value and
      std::is_convertible<decltype(interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::component_count), std::size_t>::value and
      std::is_convertible<decltype(interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::always_euclidean), bool>::value and
      std::is_void<std::void_t<typename interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::difference_type>>::value
      >> : std::true_type {};
  }
#endif


  /**
   * \brief A set of \ref vector_space_descriptor for which the number of dimensions is fixed at compile time.
   * \details This includes any object for which interface::fixed_vector_space_descriptor_traits is defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept fixed_vector_space_descriptor = std::default_initializable<std::decay_t<T>> and
    requires {
      {interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::size} -> std::convertible_to<std::size_t>;
      {interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::euclidean_size} -> std::convertible_to<std::size_t>;
      {interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::component_count} -> std::convertible_to<std::size_t>;
      {interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::always_euclidean} -> std::convertible_to<bool>;
      typename interface::fixed_vector_space_descriptor_traits<std::decay_t<T>>::difference_type;
    };
#else
  constexpr bool fixed_vector_space_descriptor = detail::is_fixed_vector_space_descriptor<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

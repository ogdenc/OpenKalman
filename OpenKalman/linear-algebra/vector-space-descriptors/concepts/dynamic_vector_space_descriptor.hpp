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
#include "linear-algebra/vector-space-descriptors/interfaces/dynamic_vector_space_descriptor_traits.hpp"


namespace OpenKalman::descriptor
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_dynamic_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_dynamic_vector_space_descriptor<T, std::enable_if_t<
      std::is_convertible<decltype(std::declval<interface::dynamic_vector_space_descriptor_traits<std::decay_t<T>>>().get_size()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::dynamic_vector_space_descriptor_traits<std::decay_t<T>>>().get_euclidean_size()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::dynamic_vector_space_descriptor_traits<std::decay_t<T>>>().get_component_count()), std::size_t>::value and
      std::is_convertible<decltype(std::declval<interface::dynamic_vector_space_descriptor_traits<std::decay_t<T>>>().is_euclidean()), bool>::value
      >> : std::true_type {};
  }
#endif


  /**
   * \brief A set of \ref vector_space_descriptor for which the number of dimensions is defined at runtime.
   * \details This includes any object for which dynamic_vector_space_descriptor_traits is defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_vector_space_descriptor = (not static_vector_space_descriptor<T>) and
    requires(const interface::dynamic_vector_space_descriptor_traits<std::decay_t<T>>& t) {
      {t.get_size()} -> std::convertible_to<std::size_t>;
      {t.get_euclidean_size()} -> std::convertible_to<std::size_t>;
      {t.get_component_count()} -> std::convertible_to<std::size_t>;
      {t.is_euclidean()} -> std::convertible_to<bool>;
    };
#else
  constexpr bool dynamic_vector_space_descriptor = (not static_vector_space_descriptor<T>) and
    detail::is_dynamic_vector_space_descriptor<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_DYNAMIC_VECTOR_SPACE_DESCRIPTOR_HPP

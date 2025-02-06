/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref euclidean_vector_space_descriptor.
 */

#ifndef OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "linear-algebra/values/concepts/fixed.hpp"
#include "linear-algebra/values/traits/fixed_number_of.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_is_euclidean.hpp"

namespace OpenKalman::descriptor
{
  namespace detail
  {
    template<typename T, typename = void>
    struct euclidean_vector_space_descriptor_impl : std::false_type {};

    template<typename T>
    struct euclidean_vector_space_descriptor_impl<T, std::enable_if_t<value::fixed<T>>>
      : std::bool_constant<value::fixed_number_of_v<T>> {};
  }


  /**
   * \brief A \ref vector_space_descriptor for a normal Euclidean vector.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_vector_space_descriptor = descriptor::vector_space_descriptor<T> and
    (value::fixed_number_of_v<decltype(descriptor::get_is_euclidean(std::declval<T>()))>);
#else
  template<typename T>
  constexpr bool euclidean_vector_space_descriptor = vector_space_descriptor<T> and
    detail::euclidean_vector_space_descriptor_impl<decltype(descriptor::get_is_euclidean(std::declval<T>()))>::value;
#endif

} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP

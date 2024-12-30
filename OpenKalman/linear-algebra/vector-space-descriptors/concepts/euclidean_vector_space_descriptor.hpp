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
 * \brief Definition for \ref euclidean_vector_space_descriptor.
 */

#ifndef OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "linear-algebra/values/traits/fixed_number_of.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "vector_space_descriptor.hpp"

namespace OpenKalman::descriptor
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_euclidean_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_euclidean_vector_space_descriptor<T, std::enable_if_t<
      value::fixed_number_of<decltype(interface::vector_space_traits<std::decay_t<T>>::is_euclidean(std::declval<T>()))>::value>>
      : std::true_type {};
  }
#endif


  /**
   * \brief A \ref vector_space_descriptor for a normal tensor index, which identifies non-modular coordinates in Euclidean space.
   * \details A set of \ref vector_space_descriptor is Euclidean if each element of the tensor is an unconstrained std::arithmetic
   * type. This would occur, for example, if the underlying scalar value is an unconstrained floating or integral value.
   * In most applications, the \ref vector_space_descriptor will be Euclidean.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_vector_space_descriptor = vector_space_descriptor<T> and
    value::fixed_number_of<decltype(interface::vector_space_traits<std::decay_t<T>>::is_euclidean(std::declval<T>()))>::value;
#else
  template<typename T>
  constexpr bool euclidean_vector_space_descriptor = vector_space_descriptor<T> and
    detail::is_euclidean_vector_space_descriptor<std::decay_t<T>>::value;
#endif

} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP

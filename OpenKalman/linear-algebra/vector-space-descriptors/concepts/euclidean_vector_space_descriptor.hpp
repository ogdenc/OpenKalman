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
#include "linear-algebra/vector-space-descriptors/interfaces/static_vector_space_descriptor_traits.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/dynamic_vector_space_descriptor_traits.hpp"
#include "vector_space_descriptor.hpp"


namespace OpenKalman
{

  // ------------------------------------- //
  //   euclidean_vector_space_descriptor   //
  // ------------------------------------- //

  /**
   * \brief A \ref vector_space_descriptor for a normal tensor index, which identifies non-modular coordinates in Euclidean space.
   * \details A set of \ref vector_space_descriptor is Euclidean if each element of the tensor is an unconstrained std::arithmetic
   * type. This would occur, for example, if the underlying scalar value is an unconstrained floating or integral value.
   * In most applications, the \ref vector_space_descriptor will be Euclidean.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_vector_space_descriptor = vector_space_descriptor<T> and
    (interface::static_vector_space_descriptor_traits<std::decay_t<T>>::always_euclidean or
      interface::dynamic_vector_space_descriptor_traits<std::decay_t<T>>::always_euclidean);
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_euclidean_vector_space_descriptor : std::false_type {};

    template<typename T>
    struct is_euclidean_vector_space_descriptor<T, std::enable_if_t<static_vector_space_descriptor<T> and
      static_vector_space_descriptor_traits<T>::always_euclidean>>
      : std::true_type {};

    template<typename T>
    struct is_euclidean_vector_space_descriptor<T, std::enable_if_t<dynamic_vector_space_descriptor<T> and
      dynamic_vector_space_descriptor_traits<std::decay_t<T>>::always_euclidean>>
      : std::true_type {};
  }

  template<typename T>
  constexpr bool euclidean_vector_space_descriptor = detail::is_euclidean_vector_space_descriptor<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_EUCLIDEAN_VECTOR_SPACE_DESCRIPTOR_HPP

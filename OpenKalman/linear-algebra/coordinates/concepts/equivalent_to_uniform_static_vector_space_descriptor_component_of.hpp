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
 * \brief Definition for \ref equivalent_to_uniform_static_vector_space_descriptor_component_of.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP

#include <type_traits>
#include "linear-algebra/coordinates/concepts/compares_with.hpp"
#include "linear-algebra/coordinates/traits/uniform_static_vector_space_descriptor_component_of.hpp"

namespace OpenKalman
{

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename C, typename = void>
    struct equivalent_to_uniform_static_vector_space_descriptor_component_of_impl : std::false_type {};

    template<typename T, typename C>
    struct equivalent_to_uniform_static_vector_space_descriptor_component_of_impl<T, C, std::enable_if_t<
      compares_with<T, typename uniform_static_vector_space_descriptor_component_of<C>::type>>> : std::true_type {};
  }
#endif


  /**
   * \brief T is equivalent to the uniform dimension type of C.
   * \tparam T A 1D \ref coordinates::descriptor
   * \tparam C a \ref uniform_static_vector_space_descriptor
   */
  template<typename T, typename C>
#ifdef __cpp_concepts
  concept equivalent_to_uniform_static_vector_space_descriptor_component_of = coordinates::compares_with<T, uniform_static_vector_space_descriptor_component_of_t<C>>;
#else
  constexpr bool equivalent_to_uniform_static_vector_space_descriptor_component_of = detail::equivalent_to_uniform_static_vector_space_descriptor_component_of_impl<T, C>::value;
#endif

} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP

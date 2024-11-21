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
 * \internal
 * \brief Definition for \ref suffix_of.
 */

#ifndef OPENKALMAN_SUFFIX_OF_HPP
#define OPENKALMAN_SUFFIX_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"


namespace OpenKalman::internal
{
  #ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct suffix_of_impl : std::false_type {};

    template<typename T, typename U>
    struct suffix_of_impl<T, U, std::enable_if_t<static_vector_space_descriptor<T> and static_vector_space_descriptor<U>>>
      : std::bool_constant<internal::prefix_of<reverse_static_vector_space_descriptor_t<T>, reverse_static_vector_space_descriptor_t<U>>> {};
  }
  #endif


  /**
   * \brief T is a suffix of U, where T and U are sets of \ref static_vector_space_descriptor types.
   * \details If T is a suffix of U, then U is equivalent_to concatenating the remaining part of U with T.
   * Cs is a suffix of StaticDescriptor<C, Cs...> for any \ref static_vector_space_descriptor Cs.
   * T is a suffix of U if equivalent_to<T, U>.
   * StaticDescriptor<> is a suffix of any \ref static_vector_space_descriptor.
   * \par Example:
   * <code>suffix_of&lt;StaticDescriptor&lt;angle::Radians, Axis&gt;, StaticDescriptor&lt;Axis&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept suffix_of = static_vector_space_descriptor<T> and static_vector_space_descriptor<U> and
    prefix_of<reverse_static_vector_space_descriptor_t<T>, reverse_static_vector_space_descriptor_t<U>>;
#else
  constexpr bool suffix_of = detail::suffix_of_impl<T, U>::value;
#endif


} // namespace OpenKalman::internal

#endif //OPENKALMAN_SUFFIX_OF_HPP

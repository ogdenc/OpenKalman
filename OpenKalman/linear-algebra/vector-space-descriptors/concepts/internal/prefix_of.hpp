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
 * \internal
 * \brief Definition for \ref prefix_of.
 */

#ifndef OPENKALMAN_PREFIX_OF_HPP
#define OPENKALMAN_PREFIX_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/internal/prefix_base_of.hpp"


namespace OpenKalman::descriptor::internal
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct prefix_of_impl : std::false_type {};

    template<typename T, typename U>
    struct prefix_of_impl<T, U, std::void_t<typename prefix_base_of<T, U>::type>> : std::true_type {};
  } // namespace detail
#endif


  /**
   * \brief T is a prefix of U, where T and U are sets of \ref static_vector_space_descriptor types.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of StaticDescriptor<C, Cs...> for any \ref static_vector_space_descriptor Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * StaticDescriptor<> is a prefix of any \ref static_vector_space_descriptor.
   * \par Example:
   * <code>prefix_of&lt;StaticDescriptor&lt;Axis&gt;, StaticDescriptor&lt;Axis, angle::Radians&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of = static_vector_space_descriptor<T> and static_vector_space_descriptor<U> and
    requires { typename prefix_base_of_t<T, U>; };
#else
  constexpr bool prefix_of =
    static_vector_space_descriptor<T> and static_vector_space_descriptor<U> and detail::prefix_of_impl<T, U>::value;
#endif


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_PREFIX_OF_HPP

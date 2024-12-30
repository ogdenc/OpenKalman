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
 * \brief Definition for \ref maybe_equivalent_to.
 */

#ifndef OPENKALMAN_DESCRIPTORS_MAYBE_EQUIVALENT_TO_HPP
#define OPENKALMAN_DESCRIPTORS_MAYBE_EQUIVALENT_TO_HPP

#include <type_traits>
#include "linear-algebra/values/traits/fixed_number_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/are_equivalent.hpp"

namespace OpenKalman::descriptor
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename U>
    struct is_maybe_equivalent_to_impl : std::false_type {};


    template<typename T, typename U>
    struct is_maybe_equivalent_to_impl<T, U, std::enable_if_t<internal::are_equivalent(T{}, Ts{})>>
      : std::false_type {};//value::fixed_number_of<decltype(internal::are_equivalent(std::declval<T>(), std::declval<U>()))> {};
#endif


#ifdef __cpp_concepts
    template<typename...Ts>
#else
    template<typename = void, typename...Ts>
#endif
    struct is_maybe_equivalent_to : std::false_type {};


    template<>
    struct is_maybe_equivalent_to<> : std::true_type {};


#ifdef __cpp_concepts
    template<typename T, typename...Ts> requires (... and (
        descriptor::dynamic_vector_space_descriptor<T> or descriptor::dynamic_vector_space_descriptor<Ts> or
        internal::are_equivalent(T{}, Ts{})))
    struct is_maybe_equivalent_to<T, Ts...> : is_maybe_equivalent_to<Ts...> {};
#else
    template<typename T, typename...Ts>
    struct is_maybe_equivalent_to<std::enable_if_t<(... and
        (descriptor::dynamic_vector_space_descriptor<T> or descriptor::dynamic_vector_space_descriptor<Ts> or
        is_maybe_equivalent_to_impl<T, Ts>::value)), T, Ts...>
      : is_maybe_equivalent_to<void, Ts...> {};
#endif
  }


  /**
   * \brief Specifies that a set of \ref vector_space_descriptor objects may be equivalent based on what is known at compile time.
   * \details Every descriptor in the set must be potentially equivalent to every other descriptor in the set.
   * Sets of vector space descriptors are equivalent if they are treated functionally the same.
   * - Any descriptor or group of descriptor is equivalent to itself.
   * - StaticDescriptor<As...> is equivalent to StaticDescriptor<Bs...>, if each As is equivalent to its respective Bs.
   * - StaticDescriptor<A> is equivalent to A, and vice versa.
   * - Dynamic \ref euclidean_vector_space_descriptor objects are equivalent to any other \ref euclidean_vector_space_descriptor,
   * \par Example:
   * <code>maybe_equivalent_to&lt;Axis, StaticDescriptor&lt;Axis&gt;&gt;</code>
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept maybe_equivalent_to =
    (... and descriptor::vector_space_descriptor<Ts>) and detail::is_maybe_equivalent_to<Ts...>::value;
#else
  constexpr bool maybe_equivalent_to =
    (... and descriptor::vector_space_descriptor<Ts>) and detail::is_maybe_equivalent_to<void, Ts...>::value;
#endif


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_DESCRIPTORS_MAYBE_EQUIVALENT_TO_HPP

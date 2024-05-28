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
 * \brief Definition for \ref maybe_equivalent_to.
 */

#ifndef OPENKALMAN_MAYBE_EQUIVALENT_TO_HPP
#define OPENKALMAN_MAYBE_EQUIVALENT_TO_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_maybe_equivalent_to_impl : std::true_type {};


#ifdef __cpp_concepts
    template<fixed_vector_space_descriptor T, fixed_vector_space_descriptor U>
    struct is_maybe_equivalent_to_impl<T, U>
#else
    template<typename T, typename U>
    struct is_maybe_equivalent_to_impl<T, U, std::enable_if_t<fixed_vector_space_descriptor<T> and fixed_vector_space_descriptor<U>>>
#endif
      : std::bool_constant<std::is_same_v<canonical_fixed_vector_space_descriptor_t<T>, canonical_fixed_vector_space_descriptor_t<U>>> {};


    template<typename...Ts>
    struct is_maybe_equivalent_to : std::true_type {};

    template<typename T, typename...Ts>
    struct is_maybe_equivalent_to<T, Ts...> : std::bool_constant<
      dynamic_vector_space_descriptor<T> ?
      is_maybe_equivalent_to<Ts...>::value :
      (... and is_maybe_equivalent_to_impl<T, Ts>::value)> {};
  }


  /**
   * \brief Specifies that a set of \ref vector_space_descriptor objects may be equivalent based on what is known at compile time.
   * \details Sets of coefficients are equivalent if they are treated functionally the same.
   * - Any coefficient or group of coefficients is equivalent to itself.
   * - FixedDescriptor<As...> is equivalent to FixedDescriptor<Bs...>, if each As is equivalent to its respective Bs.
   * - FixedDescriptor<A> is equivalent to A, and vice versa.
   * \par Example:
   * <code>equivalent_to&lt;Axis, FixedDescriptor&lt;Axis&gt;&gt;</code>
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept maybe_equivalent_to =
#else
  constexpr bool maybe_equivalent_to =
#endif
    detail::is_maybe_equivalent_to<Ts...>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_MAYBE_EQUIVALENT_TO_HPP

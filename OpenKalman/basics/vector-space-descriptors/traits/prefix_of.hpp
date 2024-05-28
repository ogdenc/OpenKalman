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
 * \brief Definition for \ref prefix_of.
 */

#ifndef OPENKALMAN_PREFIX_OF_HPP
#define OPENKALMAN_PREFIX_OF_HPP

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
    struct is_prefix_of : std::false_type {};


#ifdef __cpp_concepts
    template<typename C1, typename C2> requires equivalent_to<C1, C2>
    struct is_prefix_of<C1, C2>
#else
    template<typename C1, typename C2>
    struct is_prefix_of<C1, C2, std::enable_if_t<equivalent_to<C1, C2>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typename C>
    struct is_prefix_of<FixedDescriptor<>, C>
#else
    template<typename C>
    struct is_prefix_of<FixedDescriptor<>, C, std::enable_if_t<not equivalent_to<FixedDescriptor<>, C>>>
#endif
      : std::true_type {};


    template<typename C1, typename...Cs>
    struct is_prefix_of<C1, FixedDescriptor<C1, Cs...>> : std::true_type {};


#ifdef __cpp_concepts
    template<typename C, typename...C1, typename...C2>
    struct is_prefix_of<FixedDescriptor<C, C1...>, FixedDescriptor<C, C2...>>
#else
    template<typename C, typename...C1, typename...C2>
    struct is_prefix_of<FixedDescriptor<C, C1...>, FixedDescriptor<C, C2...>, std::enable_if_t<
      (not equivalent_to<FixedDescriptor<C, C1...>, FixedDescriptor<C, C2...>>)>>
#endif
      : std::bool_constant<is_prefix_of<FixedDescriptor<C1...>, FixedDescriptor<C2...>>::value> {};

  } // namespace detail


  /**
   * \brief T is a prefix of U, where T and U are sets of coefficients.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of FixedDescriptor<C, Cs...> for any typed \ref vector_space_descriptor Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * FixedDescriptor<> is a prefix of any set of coefficients.
   * \par Example:
   * <code>prefix_of&lt;FixedDescriptor&lt;Axis&gt;, FixedDescriptor&lt;Axis, angle::Radians&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of =
#else
  constexpr bool prefix_of =
#endif
    fixed_vector_space_descriptor<T> and fixed_vector_space_descriptor<U> and detail::is_prefix_of<
      canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>, canonical_fixed_vector_space_descriptor_t<std::decay_t<U>>>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_PREFIX_OF_HPP

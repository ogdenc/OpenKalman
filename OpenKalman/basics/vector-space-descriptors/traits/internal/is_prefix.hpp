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
 * \brief Definition for \ref is_prefix.
 */

#ifndef OPENKALMAN_IS_PREFIX_HPP
#define OPENKALMAN_IS_PREFIX_HPP

#include <type_traits>


namespace OpenKalman::internal
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_prefix_impl : std::false_type {};


#ifdef __cpp_concepts
    template<typename C, equivalent_to<C> D>
    struct is_prefix_impl<C, D>
#else
    template<typename C, typename D>
    struct is_prefix_impl<C, D, std::enable_if_t<equivalent_to<C, D>>>
#endif
      : std::true_type { using base = FixedDescriptor<>; };


    template<typename C>
#ifdef __cpp_concepts
    struct is_prefix_impl<FixedDescriptor<>, C>
#else
    struct is_prefix_impl<FixedDescriptor<>, C, std::enable_if_t<not equivalent_to<FixedDescriptor<>, C>>>
#endif
      : std::true_type { using base = C; };


#ifdef __cpp_concepts
    template<typename B, equivalent_to<B> C, typename...Cs>
    struct is_prefix_impl<B, FixedDescriptor<C, Cs...>>
#else
    template<typename B, typename C, typename...Cs>
    struct is_prefix_impl<B, FixedDescriptor<C, Cs...>, std::enable_if_t<equivalent_to<B, C> and
      not equivalent_to<B, FixedDescriptor<C, Cs...>> and not std::is_same_v<B, FixedDescriptor<>>>>
#endif
      : std::true_type { using base = FixedDescriptor<Cs...>; };


#ifdef __cpp_concepts
    template<typename C, typename...Cs, equivalent_to<C> D, typename...Ds> requires
      is_prefix_impl<FixedDescriptor<Cs...>, FixedDescriptor<Ds...>>::value
    struct is_prefix_impl<FixedDescriptor<C, Cs...>, FixedDescriptor<D, Ds...>>
#else
    template<typename C, typename...Cs, typename D, typename...Ds>
    struct is_prefix_impl<FixedDescriptor<C, Cs...>, FixedDescriptor<D, Ds...>, std::enable_if_t<
      equivalent_to<C, D> and is_prefix_impl<FixedDescriptor<Cs...>, FixedDescriptor<Ds...>>::value and
      not equivalent_to<FixedDescriptor<C, Cs...>, FixedDescriptor<D, Ds...>> and
      not equivalent_to<FixedDescriptor<C, Cs...>, D>>>
#endif
      : is_prefix_impl<FixedDescriptor<Cs...>, FixedDescriptor<Ds...>> {};

  } // namespace detail


  /**
   * \internal
   * \brief Determine whether T is a prefix of U, and if so, determine the base.
   * \details Whether T is a prefix of U is indicated by the bool <code>is_prefix::value<>.
   * If T is a prefix of U, <code>typename is_prefix::base<> is an alias for the base.
   */
  template<typename T, typename U>
  struct is_prefix : detail::is_prefix_impl<
    canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>,
    canonical_fixed_vector_space_descriptor_t<std::decay_t<U>>>
  {};


} // namespace OpenKalman::internal

#endif //OPENKALMAN_IS_PREFIX_HPP

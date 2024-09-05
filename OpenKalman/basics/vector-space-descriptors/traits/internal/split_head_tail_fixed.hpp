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
 * \internal
 * \file
 * \brief Definition for \ref split_head_tail_fixed trait.
 */

#ifndef OPENKALMAN_SPLIT_HEAD_TAIL_FIXED_HPP
#define OPENKALMAN_SPLIT_HEAD_TAIL_FIXED_HPP

#include <type_traits>


namespace OpenKalman::internal
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct split_head_tail_fixed_impl {};


#ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C>
    struct split_head_tail_fixed_impl<C>
#else
    template<typename C>
    struct split_head_tail_fixed_impl<C, std::enable_if_t<atomic_fixed_vector_space_descriptor<C>>>
#endif
    {
      using type = std::tuple<C, FixedDescriptor<>>;
    };


    template<typename C>
    struct split_head_tail_fixed_impl<FixedDescriptor<C>>
    {
      using type = std::tuple<C, FixedDescriptor<>>;
    };


    template<typename C0, typename C1>
    struct split_head_tail_fixed_impl<FixedDescriptor<C0, C1>>
    {
      using type = std::tuple<C0, C1>;
    };


    template<typename C0, typename C1, typename...Cs>
    struct split_head_tail_fixed_impl<FixedDescriptor<C0, C1, Cs...>>
    {
      using type = std::tuple<C0, FixedDescriptor<C1, Cs...>>;
    };

  } // namespace detail


  /**
   * \brief Split a \ref fixed_vector_space_descriptor into head and tail components.
   * \detail T is first converted to its \ref canonical_fixed_vector_space_descriptor "canonical" form.
   * Member alias <code>head</code> will reflect the head type and
   * member alias <code>tail</code> will reflect the tail type.
   * The above aliases are undefined for an empty vector space descriptor.
   * \tparam T A \ref fixed_vector_space_descriptor
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct split_head_tail_fixed;


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct split_head_tail_fixed
#else
  template<typename T>
  struct split_head_tail_fixed<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    : detail::split_head_tail_fixed_impl<canonical_fixed_vector_space_descriptor_t<T>> {};


  /**
   * \brief Helper template for \ref split_head_tail_fixed.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
#else
  template<typename T>
#endif
  using split_head_tail_fixed_t = typename split_head_tail_fixed<T>::type;


} // namespace OpenKalman::internal


#endif //OPENKALMAN_SPLIT_HEAD_TAIL_FIXED_HPP

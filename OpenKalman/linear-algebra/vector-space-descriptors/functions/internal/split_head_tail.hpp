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
 * \brief Definition for \ref split_head_tail trait.
 */

#ifndef OPENKALMAN_SPLIT_HEAD_TAIL_HPP
#define OPENKALMAN_SPLIT_HEAD_TAIL_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/atomic_static_vector_space_descriptor.hpp"


namespace OpenKalman::internal
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct split_head_tail_impl {};


#ifdef __cpp_concepts
    template<atomic_static_vector_space_descriptor C>
    struct split_head_tail_impl<C>
#else
    template<typename C>
    struct split_head_tail_impl<C, std::enable_if_t<atomic_static_vector_space_descriptor<C>>>
#endif
    {
      using type = std::tuple<C, descriptors::StaticDescriptor<>>;
    };


    template<typename C>
    struct split_head_tail_impl<descriptors::StaticDescriptor<C>>
    {
      using type = std::tuple<C, descriptors::StaticDescriptor<>>;
    };


    template<typename C0, typename C1>
    struct split_head_tail_impl<descriptors::StaticDescriptor<C0, C1>>
    {
      using type = std::tuple<C0, C1>;
    };


    template<typename C0, typename C1, typename...Cs>
    struct split_head_tail_impl<descriptors::StaticDescriptor<C0, C1, Cs...>>
    {
      using type = std::tuple<C0, descriptors::StaticDescriptor<C1, Cs...>>;
    };

  } // namespace detail


  /**
   * \brief Split a \ref static_vector_space_descriptor into head and tail components.
   * \detail T is first converted to its \ref internal::canonical_static_vector_space_descriptor "canonical" form.
   * Member alias <code>head</code> will reflect the head type and
   * member alias <code>tail</code> will reflect the tail type.
   * The above aliases are undefined for an empty vector space descriptor.
   * \tparam T A \ref static_vector_space_descriptor
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
#else
  template<typename T, typename = void>
#endif
  struct split_head_tail;


#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
  struct split_head_tail
#else
  template<typename T>
  struct split_head_tail<T, std::enable_if_t<static_vector_space_descriptor<T>>>
#endif
    : detail::split_head_tail_impl<canonical_static_vector_space_descriptor_t<T>> {};


  /**
   * \brief Helper template for \ref split_head_tail.
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
#else
  template<typename T>
#endif
  using split_head_tail_t = typename split_head_tail<T>::type;


} // namespace OpenKalman::internal


#endif //OPENKALMAN_SPLIT_HEAD_TAIL_HPP

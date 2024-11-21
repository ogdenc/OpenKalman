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
 * \brief Definition for \ref prefix_base_of.
 */

#ifndef OPENKALMAN_PREFIX_BASE_OF_HPP
#define OPENKALMAN_PREFIX_BASE_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp"


namespace OpenKalman::internal
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct prefix_base_of_impl {};


#ifdef __cpp_concepts
    template<typename C, equivalent_to<C> D>
    struct prefix_base_of_impl<C, D>
#else
    template<typename C, typename D>
    struct prefix_base_of_impl<C, D, std::enable_if_t<equivalent_to<C, D>>>
#endif
    { using type = descriptors::StaticDescriptor<>; };


    template<typename D>
#ifdef __cpp_concepts
    struct prefix_base_of_impl<descriptors::StaticDescriptor<>, D>
#else
    struct prefix_base_of_impl<StaticDescriptor<>, D, std::enable_if_t<not equivalent_to<StaticDescriptor<>, D>>>
#endif
    { using type = D; };


#ifdef __cpp_concepts
    template<typename C, equivalent_to<C> D, typename...Ds>
    struct prefix_base_of_impl<C, descriptors::StaticDescriptor<D, Ds...>>
#else
    template<typename C, typename D, typename...Ds>
    struct prefix_base_of_impl<C, StaticDescriptor<D, Ds...>, std::enable_if_t<
      equivalent_to<C, D> and
      (not equivalent_to<C, StaticDescriptor<D, Ds...>>) and
      (not std::is_same_v<C, StaticDescriptor<>>)>>
#endif
    { using type = descriptors::StaticDescriptor<Ds...>; };


#ifdef __cpp_concepts
    template<typename C, typename...Cs, equivalent_to<C> D, typename...Ds>
    struct prefix_base_of_impl<descriptors::StaticDescriptor<C, Cs...>, descriptors::StaticDescriptor<D, Ds...>>
#else
    template<typename C, typename...Cs, typename D, typename...Ds>
    struct prefix_base_of_impl<StaticDescriptor<C, Cs...>, StaticDescriptor<D, Ds...>, std::enable_if_t<
      equivalent_to<C, D> and
      (not equivalent_to<StaticDescriptor<C, Cs...>, StaticDescriptor<D, Ds...>>) and
      (not equivalent_to<StaticDescriptor<C, Cs...>, D>)>>
#endif
      : prefix_base_of_impl<descriptors::StaticDescriptor<Cs...>, descriptors::StaticDescriptor<Ds...>> {};

  } // namespace detail


  /**
   * \internal
   * \brief If T is a \ref internal::prefix_of "prefix of" U, return the non-overlapping base part.
   * \details If T is a prefix of U, <code>type</code> will be an alias for the base.
   */
#ifdef __cpp_concepts
  template<typename T, typename U>
#else
  template<typename T, typename U, typename = void>
#endif
  struct prefix_base_of {};


#ifdef __cpp_concepts
  template<static_vector_space_descriptor T, static_vector_space_descriptor U>
  struct prefix_base_of<T, U>
#else
  template<typename T, typename U>
  struct is_prefix<T, U, std::enable_if_t<static_vector_space_descriptor<T> and static_vector_space_descriptor<U>>>
#endif
  : detail::prefix_base_of_impl<
      canonical_static_vector_space_descriptor_t<std::decay_t<T>>,
      canonical_static_vector_space_descriptor_t<std::decay_t<U>>> {};


  /**
   * \brief Helper template for \ref base_of.
   */
  template<typename T, typename U>
  using prefix_base_of_t = typename prefix_base_of<T, U>::type;


} // namespace OpenKalman::internal

#endif //OPENKALMAN_PREFIX_BASE_OF_HPP

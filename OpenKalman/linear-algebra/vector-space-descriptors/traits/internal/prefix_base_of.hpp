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
 * \brief Definition for \ref descriptor::internal::prefix_base_of.
 */

#ifndef OPENKALMAN_DESCRIPTORS_PREFIX_BASE_OF_HPP
#define OPENKALMAN_DESCRIPTORS_PREFIX_BASE_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/traits/internal/static_canonical_form.hpp"

namespace OpenKalman::descriptor::internal
{
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
  struct prefix_base_of;


  /**
   * \brief Helper template for \ref base_of.
   */
  template<typename T, typename U>
  using prefix_base_of_t = typename prefix_base_of<T, U>::type;


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct prefix_base_of_impl {};


    template<typename T>
    struct prefix_base_of_impl<T, T>
    {
      using type = StaticDescriptor<>;
    };


    template<typename T>
    struct prefix_base_of_impl<StaticDescriptor<>, T>
    {
      using type = T;
    };


    template<typename T, typename...Ts>
    struct prefix_base_of_impl<T, StaticDescriptor<T, Ts...>>
    {
      using type = StaticDescriptor<Ts...>;
    };


#ifdef __cpp_concepts
    template<typename T, typename...Cs, typename...Ds> requires (not (... and std::same_as<Cs, Ds>))
    struct prefix_base_of_impl<StaticDescriptor<T, Cs...>, StaticDescriptor<T, Ds...>>
#else
    template<typename T, typename...Cs, typename...Ds>
    struct prefix_base_of_impl<StaticDescriptor<T, Cs...>, StaticDescriptor<T, Ds...>,
      std::enable_if_t<not std::is_same_v<StaticDescriptor<T, Cs...>, StaticDescriptor<T, Ds...>>>>
#endif
      : prefix_base_of_impl<StaticDescriptor<Cs...>, StaticDescriptor<Ds...>> {};

  } // namespace detail


#ifdef __cpp_concepts
  template<static_vector_space_descriptor T, static_vector_space_descriptor U>
  struct prefix_base_of<T, U>
#else
  template<typename T, typename U>
  struct is_prefix<T, U, std::enable_if_t<static_vector_space_descriptor<T> and static_vector_space_descriptor<U>>>
#endif
  : detail::prefix_base_of_impl<
      static_canonical_form_t<std::decay_t<T>>,
      static_canonical_form_t<std::decay_t<U>>> {};


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_DESCRIPTORS_PREFIX_BASE_OF_HPP

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
 * \brief Definition for \ref has_uniform_dimension_type.
 */

#ifndef OPENKALMAN_HAS_UNIFORM_DIMENSION_TYPE_HPP
#define OPENKALMAN_HAS_UNIFORM_DIMENSION_TYPE_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename C>
#else
    template<typename C, typename = void>
#endif
    struct uniform_fixed_dimension_impl : std::false_type {};


#ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C> requires (dimension_size_of_v<C> == 1)
    struct uniform_fixed_dimension_impl<C>
#else
    template<typename C>
    struct uniform_fixed_dimension_impl<C, std::enable_if_t<atomic_fixed_vector_space_descriptor<C> and (dimension_size_of_v<C> == 1)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<typename C> requires (dimension_size_of_v<C> == 1)
    struct uniform_fixed_dimension_impl<TypedIndex<C>>
#else
    template<typename C>
    struct uniform_fixed_dimension_impl<TypedIndex<C>, std::enable_if_t<dimension_size_of_v<C> == 1>>
#endif
      : uniform_fixed_dimension_impl<C> {};


#ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C, fixed_vector_space_descriptor...Cs> requires (dimension_size_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and std::same_as<C, typename uniform_fixed_dimension_impl<TypedIndex<Cs...>>::uniform_type>
    struct uniform_fixed_dimension_impl<TypedIndex<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct uniform_fixed_dimension_impl<TypedIndex<C, Cs...>, std::enable_if_t<
      atomic_fixed_vector_space_descriptor<C> and (... and fixed_vector_space_descriptor<Cs>) and (dimension_size_of_v<C> == 1) and
        (sizeof...(Cs) > 0) and std::is_same<C, typename uniform_fixed_dimension_impl<TypedIndex<Cs...>>::uniform_type>::value>>
#endif
      : std::true_type { using uniform_type = C; };


#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct has_uniform_dimension_type_impl : std::false_type {};

    template<typename T>
    struct has_uniform_dimension_type_impl<T, std::void_t<typename canonical_fixed_vector_space_descriptor<T>::type>>
      : detail::uniform_fixed_dimension_impl<canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>> {};
#endif

  } // namespace detail


  /**
   * \brief T is a \ref vector_space_descriptor comprising a uniform set of 1D \ref atomic_fixed_vector_space_descriptor types.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_uniform_dimension_type = euclidean_vector_space_descriptor<T> or
    detail::uniform_fixed_dimension_impl<canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>>::value;
#else
  constexpr bool has_uniform_dimension_type = euclidean_vector_space_descriptor<T> or detail::has_uniform_dimension_type_impl<T>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_HAS_UNIFORM_DIMENSION_TYPE_HPP

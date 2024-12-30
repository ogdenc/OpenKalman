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
 * \brief Definition for \ref vector_space_descriptor_tuple.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TUPLE_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TUPLE_HPP

#include <type_traits>
#include <tuple>
#include <utility>
#include "basics/global-definitions.hpp"
#include "vector_space_descriptor.hpp"

namespace OpenKalman::descriptor
{
#if not defined(__cpp_concepts) or not defined(__cpp_lib_remove_cvref) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, std::size_t...Ix>
    constexpr bool is_descriptor_tuple_impl(std::index_sequence<Ix...>)
    {
      return (... and vector_space_descriptor<std::tuple_element_t<Ix, T>>);
    }


    template<typename T, typename = void>
    struct is_descriptor_tuple : std::false_type {};

 
    template<typename T>
    struct is_descriptor_tuple<T, std::enable_if_t<internal::tuple_like<T>>> 
      : std::bool_constant<is_descriptor_tuple_impl<T>(std::make_index_sequence<std::tuple_size_v<T>>{})> {};

  } // namespace detail
#endif

	
  /**
   * \brief An object describing a tuple-like collection of /ref vector_space_descriptor objects.
   */
  template<typename T>
#if defined(__cpp_concepts) and defined(__cpp_lib_remove_cvref) and __cpp_generic_lambdas >= 201707L
  concept vector_space_descriptor_tuple =
    internal::tuple_like<T> and
    []<std::size_t...Ix>(std::index_sequence<Ix...>)
      { return (... and vector_space_descriptor<std::tuple_element_t<Ix, std::remove_cvref_t<T>>>); }
      (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{});
#else
  constexpr bool vector_space_descriptor_tuple =
    detail::is_descriptor_tuple<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TUPLE_HPP

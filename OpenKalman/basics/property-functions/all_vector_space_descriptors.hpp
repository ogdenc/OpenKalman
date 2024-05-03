/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref all_vector_space_descriptors function.
 */

#ifndef OPENKALMAN_ALL_VECTOR_SPACE_DESCRIPTORS_HPP
#define OPENKALMAN_ALL_VECTOR_SPACE_DESCRIPTORS_HPP


namespace OpenKalman
{

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto all_vector_space_descriptors_impl(const T& t, std::index_sequence<I...>)
    {
      return std::tuple {get_vector_space_descriptor<I>(t)...};
    }
  }


  /**
   * \brief Return a tuple of \ref vector_space_descriptor items defining the dimensions of T.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T> requires interface::get_vector_space_descriptor_defined_for<T> and
    requires(const T& t) { {count_indices(t)} -> static_index_value; }
#else
  template<typename T, std::enable_if_t<interface::count_indices_defined_for<T> and
    interface::get_vector_space_descriptor_defined_for<T> and static_index_value<decltype(count_indices(std::declval<const T&>()))>, int> = 0>
#endif
  constexpr decltype(auto) all_vector_space_descriptors(const T& t)
  {
    constexpr std::make_index_sequence<std::decay_t<decltype(count_indices(t))>::value> seq;
    return detail::all_vector_space_descriptors_impl(t, seq);
  }


  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool no_dynamic_dimensions_impl(std::index_sequence<I...>)
    {
      return (... and fixed_vector_space_descriptor<decltype(get_vector_space_descriptor<I>(std::declval<T>()))>);
    }


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct no_dynamic_dimensions : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires static_index_value<decltype(count_indices(std::declval<T>()))>
    struct no_dynamic_dimensions<T>
#else
    template<typename T>
    struct no_dynamic_dimensions<T, std::enable_if_t<static_index_value<decltype(count_indices(std::declval<T>()))>>>
#endif
      : std::bool_constant<no_dynamic_dimensions_impl<T>(std::make_index_sequence<std::decay_t<decltype(count_indices(std::declval<T>()))>::value>{})> {};


    template<typename T, std::size_t...I>
    constexpr auto all_vector_space_descriptors_impl(std::index_sequence<I...>)
    {
      return std::tuple {std::decay_t<decltype(get_vector_space_descriptor<I>(std::declval<T>()))>{}...};
    }
  }


  /**
   * \overload
   * \brief Return a tuple of \ref vector_space_descriptor defining the dimensions of T.
   * \details This overload is only enabled if all dimensions of T are known at compile time.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T> requires interface::get_vector_space_descriptor_defined_for<T> and
    requires(const T& t) { {count_indices(t)} -> static_index_value; } and detail::no_dynamic_dimensions<T>::value
#else
  template<typename T, std::enable_if_t<interface::count_indices_defined_for<T> and
    interface::get_vector_space_descriptor_defined_for<T> and static_index_value<decltype(count_indices(std::declval<const T&>()))> and
    detail::no_dynamic_dimensions<T>::value, int> = 0>
#endif
  constexpr auto all_vector_space_descriptors()
  {
    constexpr std::make_index_sequence<std::decay_t<decltype(count_indices(std::declval<T>()))>::value> seq;
    return detail::all_vector_space_descriptors_impl<T>(seq);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ALL_VECTOR_SPACE_DESCRIPTORS_HPP

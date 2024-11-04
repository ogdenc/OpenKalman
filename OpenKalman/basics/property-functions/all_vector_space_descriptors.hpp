/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
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
  } // namespace detail


  /**
   * \brief Return a collection of \ref vector_space_descriptor objects associated with T.
     \details This will be a \ref vector_space_descriptor_collection in the form of a std::tuple or a std::vector.
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T> requires interface::get_vector_space_descriptor_defined_for<T> 
  constexpr vector_space_descriptor_collection decltype(auto) 
#else
  template<typename T, std::enable_if_t<
    interface::count_indices_defined_for<T> and interface::get_vector_space_descriptor_defined_for<T>, int> = 0>
  constexpr decltype(auto) 
#endif
  all_vector_space_descriptors(const T& t)
  {
#ifdef __cpp_concepts
    if constexpr (requires(const T& t) { {count_indices(t)} -> static_index_value; })
#else
    if constexpr (static_index_value<decltype(count_indices(std::declval<const T&>()))>)
#endif
    {
      constexpr std::make_index_sequence<std::decay_t<decltype(count_indices(t))>::value> seq;
      return detail::all_vector_space_descriptors_impl(t, seq);
    }
    else 
    {
      return internal::VectorSpaceDescriptorRange<T> {t};
    }
  }


  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto all_vector_space_descriptors_impl(std::index_sequence<I...>)
    {
      return std::tuple {std::decay_t<decltype(get_vector_space_descriptor<I>(std::declval<T>()))>{}...};
    }

  } // namespace detail


  /**
   * \overload
   * \brief Return a collection of \ref fixed_vector_space_descriptor objects associated with T.
   * \details This overload is only enabled if all vector space descriptors are static.
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T> requires interface::get_vector_space_descriptor_defined_for<T> and
    requires(const T& t) { {count_indices(t)} -> static_index_value; } and (not has_dynamic_dimensions<T>) 
  constexpr vector_space_descriptor_tuple auto 
#else
  template<typename T, std::enable_if_t<interface::count_indices_defined_for<T> and
    interface::get_vector_space_descriptor_defined_for<T> and static_index_value<decltype(count_indices(std::declval<const T&>()))> and
    (not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto 
#endif
  all_vector_space_descriptors()
  {
    constexpr std::make_index_sequence<std::decay_t<decltype(count_indices(std::declval<T>()))>::value> seq;
    return detail::all_vector_space_descriptors_impl<T>(seq);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ALL_VECTOR_SPACE_DESCRIPTORS_HPP

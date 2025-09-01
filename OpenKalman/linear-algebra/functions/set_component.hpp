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
 * \brief Definition for \ref set_component function.
 */

#ifndef OPENKALMAN_SET_COMPONENT_HPP
#define OPENKALMAN_SET_COMPONENT_HPP


namespace OpenKalman
{
  // \todo Add functions that return stl ranges.

  namespace detail
  {
    template<typename Arg, typename Indices>
    constexpr Arg&&
    set_component_impl(Arg&& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      using Trait = interface::library_interface<std::decay_t<Arg>>;
      Trait::set_component(arg, s, internal::truncate_indices(indices, count_indices(arg)));
      return std::forward<Arg>(arg);
    }
  }


  /**
   * \overload
   * \brief Set a component of an object at a particular set of indices.
   * \tparam Arg The object to be accessed.
   * \tparam Indices An input range object containing the indices.
   * \return The modified Arg
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, index_collection_for<Arg> Indices> requires writable_by_component<Arg, Indices>
#else
  template<typename Arg, typename Indices, std::enable_if_t<
    indexible<Arg> and index_collection_for<Indices, Arg> and writable_by_component<Arg, Indices>, int> = 0>
#endif
  inline Arg&&
  set_component(Arg&& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
  {
    return detail::set_component_impl(std::forward<Arg>(arg), s, indices);
  }


  /**
   * \brief Set a component of an object at an initializer list of indices.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, values::index Ix> requires writable_by_component<Arg, const std::initializer_list<Ix>&>
#else
  template<typename Arg, typename Ix, std::enable_if_t<
    indexible<Arg> and values::index<Ix> and writable_by_component<Arg, const std::initializer_list<Ix>&>, int> = 0>
#endif
  inline Arg&&
  set_component(Arg&& arg, const scalar_type_of_t<Arg>& s, const std::initializer_list<Ix>& indices)
  {
    return detail::set_component_impl(std::forward<Arg>(arg), s, indices);
  }


  /**
   * \overload
   * \brief Set a component of an object using a fixed number of indices.
   * \details The number of indices must be at least <code>index_count_v&lt;Arg&gt;</code>. If the indices are
   * integral constants, the function performs compile-time bounds checking to the extent possible.
   */
#ifdef __cpp_lib_concepts
  template<typename Arg, values::index...I> requires writable_by_component<Arg, std::array<std::size_t, sizeof...(I)>> and
    (index_count_v<Arg> == dynamic_size or sizeof...(I) >= index_count_v<Arg>) and
    internal::static_indices_within_bounds<Arg, I...>::value
#else
  template<typename Arg, typename...I, std::enable_if_t<
    (values::index<I> and ...) and writable_by_component<Arg, std::array<std::size_t, sizeof...(I)>> and
    (index_count<Arg>::value == dynamic_size or sizeof...(I) >= index_count<Arg>::value) and
    internal::static_indices_within_bounds<Arg, I...>::value, int> = 0>
#endif
  inline Arg&&
  set_component(Arg&& arg, const scalar_type_of_t<Arg>& s, I&&...i)
  {
    if constexpr (sizeof...(I) == 0)
      return detail::set_component_impl(std::forward<Arg>(arg), s, std::array<std::size_t, 0> {});
    else
      return detail::set_component_impl(std::forward<Arg>(arg), s, 
        std::array {static_cast<std::common_type_t<I...>>(std::forward<I>(i))...});
  }


}

#endif

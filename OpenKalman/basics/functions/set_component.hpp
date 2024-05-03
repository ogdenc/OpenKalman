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
 * \brief Definition for \ref set_component function.
 */

#ifndef OPENKALMAN_SET_COMPONENT_HPP
#define OPENKALMAN_SET_COMPONENT_HPP

#ifdef __cpp_lib_ranges
#include<ranges>
//#else
#include<algorithm>
#endif

namespace OpenKalman
{
  // \todo Add functions that return stl-compatible iterators.

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
  } // namespace detail


  /**
   * \brief Set a component of an object at a particular set of indices.
   * \tparam Arg The object to be accessed.
   * \tparam Indices An input range object containing the indices.
   * \return The modified Arg
   */
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
  template<indexible Arg, std::ranges::input_range Indices> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and index_value<std::ranges::range_value_t<Indices>> and
    (static_range_size_v<Indices> == dynamic_size or index_count_v<Arg> == dynamic_size or static_range_size_v<Indices> >= index_count_v<Arg>) and
    (not empty_object<Arg>) and
    interface::set_component_defined_for<Arg, std::add_lvalue_reference_t<Arg>, const scalar_type_of_t<Arg>&, const Indices&>
#else
  template<typename Arg, typename Indices, std::enable_if_t<indexible<Arg> and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and index_value<decltype(*std::declval<Indices>().begin())> and
    (static_range_size<Indices>::value == dynamic_size or index_count<Arg>::value == dynamic_size or static_range_size<Indices>::value >= index_count<Arg>::value) and
    (not empty_object<Arg>) and
    interface::set_component_defined_for<Arg, typename std::add_lvalue_reference<Arg>::type, const typename scalar_type_of<Arg>::type&, const Indices&>, int> = 0>
#endif
  inline Arg&&
  set_component(Arg&& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
  {
    return detail::set_component_impl(std::forward<Arg>(arg), s, indices);
  }


  /**
   * \overload
   * \brief Set a component of an object using an initializer list.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, index_value Indices> requires
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not empty_object<Arg>) and
    interface::set_component_defined_for<Arg,
      std::add_lvalue_reference_t<Arg>, const scalar_type_of_t<Arg>&, const std::initializer_list<Indices>&>
#else
  template<typename Arg, typename Indices, std::enable_if_t<indexible<Arg> and index_value<Indices> and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not empty_object<Arg>) and
    interface::set_component_defined_for<Arg,
      typename std::add_lvalue_reference<Arg>::type, const typename scalar_type_of<Arg>::type&, const std::initializer_list<Indices>&>, int> = 0>
#endif
  inline Arg&&
  set_component(Arg&& arg, const scalar_type_of_t<Arg>& s, const std::initializer_list<Indices>& indices)
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
  template<indexible Arg, index_value...I> requires (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (not empty_object<Arg>) and detail::static_indices_within_bounds<Arg, I...>::value and
    interface::set_component_defined_for<Arg,
      std::add_lvalue_reference_t<Arg>, const scalar_type_of_t<Arg>&, const std::array<std::size_t, sizeof...(I)>&>
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and (index_value<I> and ...) and
    (not std::is_const_v<std::remove_reference_t<Arg>>) and (not empty_object<Arg>) and
    detail::static_indices_within_bounds<Arg, I...>::value and
    interface::set_component_defined_for<Arg, typename std::add_lvalue_reference<Arg>::type,
      const typename scalar_type_of<Arg>::type&, const std::array<std::size_t, sizeof...(I)>&>, int> = 0>
#endif
  inline Arg&&
  set_component(Arg&& arg, const scalar_type_of_t<Arg>& s, I&&...i)
  {
    const auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
    return detail::set_component_impl(std::forward<Arg>(arg), s, indices);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SET_COMPONENT_HPP

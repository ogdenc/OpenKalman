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
 * \brief Definition for \ref get_component function.
 */

#ifndef OPENKALMAN_GET_COMPONENT_HPP
#define OPENKALMAN_GET_COMPONENT_HPP

#include "linear-algebra/values/concepts/index.hpp"
#include "linear-algebra/concepts/index_range_for.hpp"
#include "linear-algebra/interfaces/default/library_interface.hpp"
#include "linear-algebra/concepts/empty_object.hpp"

namespace OpenKalman
{
  // \todo Add functions that return stl-compatible iterators.

  namespace detail
  {
    template<typename Arg, typename Indices>
    constexpr decltype(auto)
    get_component_impl(Arg&& arg, const Indices& indices)
    {
      using Trait = interface::library_interface<std::decay_t<Arg>>;
      return Trait::get_component(std::forward<Arg>(arg), internal::truncate_indices(indices, count_indices(arg)));
    }
  } // namespace detail


  /**
   * \brief Get a component of an object at a particular set of indices.
   * \tparam Arg The object to be accessed.
   * \tparam Indices A sized input range containing the indices.
   * \return a \ref value::scalar
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, index_range_for<Arg> Indices> requires (not empty_object<Arg>)
  constexpr value::scalar decltype(auto)
#else
  template<typename Arg, typename Indices, std::enable_if_t<
    index_range_for<Indices, Arg> and (not empty_object<Arg>), int> = 0>
  constexpr decltype(auto)
#endif
  get_component(Arg&& arg, const Indices& indices)
  {
    return detail::get_component_impl(std::forward<Arg>(arg), indices);
  }


  /**
   * \overload 
   * \brief Get a component of an object at an initializer list of indices.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, value::index Ix> requires (not empty_object<Arg>)
  constexpr value::scalar decltype(auto)
#else
  template<typename Arg, typename Ix, std::enable_if_t<value::index<Ix> and (not empty_object<Arg>), int> = 0>
  constexpr decltype(auto)
#endif
  get_component(Arg&& arg, const std::initializer_list<Ix>& indices)
  {
    return detail::get_component_impl(std::forward<Arg>(arg), indices);
  }


  namespace internal
  {
    namespace detail
    {
      template<typename Arg, typename...V, std::size_t...Ix>
      constexpr bool static_indices_within_bounds_impl(std::index_sequence<Ix...>)
      {
        return ([]{
          if constexpr (value::fixed<V>)
            return (std::decay_t<V>::value >= 0 and std::decay_t<V>::value < index_dimension_of_v<Arg, Ix>);
          else 
            return true;
        }() and ...);
      }
    } // namespace detail


    template<typename Arg, typename...I>
    struct static_indices_within_bounds
      : std::bool_constant<(detail::static_indices_within_bounds_impl<Arg, I...>(std::index_sequence_for<I...>{}))> {};

  } // namespace internal


  /**
   * \overload
   * \brief Get a component of an object using a fixed number of indices.
   * \details The number of indices must be at least <code>index_count_v&lt;Arg&gt;</code>. If the indices are
   * integral constants, the function performs compile-time bounds checking to the extent possible.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, value::index...I> requires
    (index_count_v<Arg> == dynamic_size or sizeof...(I) >= index_count_v<Arg>) and
    (not empty_object<Arg>) and internal::static_indices_within_bounds<Arg, I...>::value
  constexpr value::scalar decltype(auto)
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and (... and value::index<I>) and
    (index_count<Arg>::value == dynamic_size or sizeof...(I) >= index_count<Arg>::value) and
    (not empty_object<Arg>) and internal::static_indices_within_bounds<Arg, I...>::value, int> = 0>
  constexpr decltype(auto)
#endif
  get_component(Arg&& arg, I&&...i)
  {
    if constexpr (sizeof...(I) == 0)
      return detail::get_component_impl(std::forward<Arg>(arg), std::array<std::size_t, 0> {});
    else
      return detail::get_component_impl(std::forward<Arg>(arg), 
        std::array {static_cast<std::common_type_t<I...>>(std::forward<I>(i))...});
  }


} // namespace OpenKalman

#endif //OPENKALMAN_GET_COMPONENT_HPP

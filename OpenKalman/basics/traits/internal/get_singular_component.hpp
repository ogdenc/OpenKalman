/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref internal::singular_component.
 */

#ifndef OPENKALMAN_GET_SINGULAR_COMPONENT_HPP
#define OPENKALMAN_GET_SINGULAR_COMPONENT_HPP


namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename Arg, typename Indices>
    constexpr decltype(auto)
    get_singular_component_impl(Arg &&arg, const Indices &indices)
    {
      using Trait = interface::library_interface<std::decay_t<Arg>>;
      return ScalarConstant<scalar_type_of_t<Arg>> {
        Trait::get_component(std::forward<Arg>(arg), internal::truncate_indices(indices, count_indices(arg)))
      };
    }


    template<typename Arg, std::size_t...Ix>
    constexpr decltype(auto)
    get_singular_component_impl(Arg &&arg, std::index_sequence<Ix...>)
    {
      auto indices = std::array<std::size_t, sizeof...(Ix)>{static_cast<decltype(Ix)>(0)...};
      return detail::get_singular_component_impl(std::forward<Arg>(arg), indices);
    }
  } // namespace detail


  /**
   * \internal
   * \brief Get the initial component of Arg, which will be the singular component in a one-dimensional object.
   * \tparam arg A tensor (vector, matrix, etc.)
   */
  template<typename Arg>
  constexpr decltype(auto)
  get_singular_component(Arg&& arg)
  {
    if constexpr (index_count_v<Arg> == dynamic_size)
      return detail::get_singular_component_impl(std::forward<Arg>(arg), std::vector<std::size_t>{count_indices(arg), 0});
    else
      return detail::get_singular_component_impl(std::forward<Arg>(arg), std::make_index_sequence<index_count_v<Arg>>{});
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_GET_SINGULAR_COMPONENT_HPP

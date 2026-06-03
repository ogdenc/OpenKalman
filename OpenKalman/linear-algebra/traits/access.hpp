/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref access function.
 */

#ifndef OPENKALMAN_ACCESS_HPP
#define OPENKALMAN_ACCESS_HPP

#include "collections/collections.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/concepts/index_collection_for.hpp"
#include "linear-algebra/concepts/empty_object.hpp"
#include "linear-algebra/concepts/constant_object.hpp"

namespace OpenKalman
{
  namespace detail
  {
#ifdef __cpp_contracts
    template<typename Mds, typename Indices>
    constexpr bool
    check_access_bounds(const Mds& mds, const Indices& indices)
    {
      return true;
    }
#endif

    template<typename Indices, std::size_t...ix>
    constexpr decltype(auto)
    make_index_array(const Indices& indices, std::index_sequence<ix...>)
    {
      return std::array<std::size_t, sizeof...(ix)> {collections::get<ix>(indices)...};
    }

  }


  /**
   * \brief Access a component of an \ref indexible object at a given set of indices.
   * \details This performs no bounds checking
   * \tparam Arg The \ref indexible object to be accessed.
   * \tparam Indices A sized input range containing the indices.
   * \return a reference to a \ref values::value
   */
#ifdef __cpp_concepts
  template<indexible Arg, index_collection_for<Arg> Indices> requires
    (not empty_object<Arg>) and
    (not values::size_compares_with<collections::size_of<Indices>, index_count<Arg>, &stdex::is_lt>)
  constexpr values::value decltype(auto)
#else
  template<typename Arg, typename Indices, std::enable_if_t<
    index_collection_for<Indices, Arg> and (not empty_object<Arg>) and
    values::size_compares_with<collections::size_of<Indices>, index_count<Arg>, &stdex::is_gteq>, int> = 0>
  constexpr decltype(auto)
#endif
  access(Arg&& arg, const Indices& indices)
#ifdef __cpp_contracts
  pre(detail::check_access_bounds(get_mdspan(std::forward<Arg>(arg)), indices))
#endif
  {
    decltype(auto) mds = get_mdspan(std::forward<Arg>(arg));
    using Mds = decltype(mds);
    constexpr std::size_t rank = std::decay_t<Mds>::rank();
    return std::forward<Mds>(mds)[detail::make_index_array(indices, std::make_index_sequence<rank>{})];
  }


  /**
   * \overload
   * \brief Access a component based on a pack of indices.
   */
#ifdef __cpp_concepts
  template<indexible Arg, values::index...I> requires
    (not empty_object<Arg>) and
    values::size_compares_with<std::integral_constant<std::size_t, sizeof...(I)>, index_count<Arg>, &stdex::is_gteq>
  constexpr values::value decltype(auto)
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and (... and values::index<I>) and
    (not empty_object<Arg>) and
    values::size_compares_with<std::integral_constant<std::size_t, sizeof...(I)>, index_count<Arg>, &stdex::is_gteq>, int> = 0>
  constexpr decltype(auto)
#endif
  access(Arg&& arg, I...i)
  {
    return access(std::forward<Arg>(arg), std::array<std::size_t, sizeof...(I)>{std::move(i)...});
  }


}

#endif

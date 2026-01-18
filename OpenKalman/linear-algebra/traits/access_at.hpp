/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref access_at function.
 */

#ifndef OPENKALMAN_ACCESS_AT_HPP
#define OPENKALMAN_ACCESS_AT_HPP

#include "collections/collections.hpp"
#include "linear-algebra/concepts/index_collection_for.hpp"
#include "linear-algebra/concepts/empty_object.hpp"
#include "linear-algebra/traits/index_dimension_of.hpp"

namespace OpenKalman
{
  /**
   * \brief Access a component of an \ref indexible object at a given set of indices, with bounds checking.
   * \details This is the same as \ref access, except with added bounds checking
   * \tparam Arg The object to be accessed.
   * \tparam Indices A sized input range containing the indices.
   * \return a \ref values::scalar (value or reference)
   * \sa access
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, index_collection_for<Arg> Indices> requires
    (not empty_object<Arg>) and
    (not values::size_compares_with<collections::size_of<Indices>, index_count<Arg>, &stdex::is_lt>)
  constexpr values::value decltype(auto)
#else
  template<typename Arg, typename Indices, std::enable_if_t<
    index_collection_for<Indices, Arg> and (not empty_object<Arg>) and
    (not values::size_compares_with<collections::size_of<Indices>, index_count<Arg>, &stdex::is_lt>), int> = 0>
  constexpr decltype(auto)
#endif
  access_at(Arg&& arg, const Indices& indices)
  {
    bool in_bounds = collections::compare_indices<&stdex::is_lt>(indices, patterns::views::dimensions(get_pattern_collection(arg)));
    if (not in_bounds) throw std::out_of_range {"One or more indices out of range."};
    return access(std::forward<Arg>(arg), indices);
  }


  /**
   * \overload
   * \brief Access a component based on a pack of indices.
   */
#ifdef __cpp_lib_concepts
  template<indexible Arg, values::index...I> requires
    (not empty_object<Arg>) and
    (not values::size_compares_with<std::integral_constant<std::size_t, sizeof...(I)>, index_count<Arg>, &stdex::is_lt>)
  constexpr values::value decltype(auto)
#else
  template<typename Arg, typename...I, std::enable_if_t<indexible<Arg> and (... and values::index<I>) and
    (not empty_object<Arg>) and
    (not values::size_compares_with<std::integral_constant<std::size_t, sizeof...(I)>, index_count<Arg>, &stdex::is_lt>), int> = 0>
  constexpr decltype(auto)
#endif
  access_at(Arg&& arg, I...i)
  {
    return access_at(std::forward<Arg>(arg), std::array<std::size_t, sizeof...(I)>{std::move(i)...});
  }


}

#endif

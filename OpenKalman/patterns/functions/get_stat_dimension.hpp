/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns::get_stat_dimension.
 */

#ifndef OPENKALMAN_GET_EUCLIDEAN_DIMENSION_HPP
#define OPENKALMAN_GET_EUCLIDEAN_DIMENSION_HPP

#include <functional>
#include "basics/basics.hpp"
#include "collections/concepts/uniformly_gettable.hpp"
#include "patterns/concepts/sized_pattern.hpp"
#include "patterns/concepts/descriptor.hpp"
#include "patterns/functions/internal/get_descriptor_stat_dimension.hpp"

namespace OpenKalman::patterns
{
  namespace detail
  {
#ifndef __cpp_lib_ranges
    template<typename T, typename = void>
    struct range_value_has_fixed_stat_dimension : std::false_type {};

    template<typename T>
    struct range_value_has_fixed_stat_dimension<T, std::enable_if_t<stdex::ranges::range<T>>>
      : std::bool_constant<values::fixed_value_compares_with<decltype(internal::get_descriptor_stat_dimension(
          std::declval<stdex::ranges::range_value_t<T>>())), 0>> {};
#endif


    template<std::size_t i = 0, typename T>
    static constexpr auto get_stat_dimension_fixed(const T& t)
    {
      if constexpr (i < collections::size_of_v<T>)
      {
        return values::operation(
          std::plus{},
          internal::get_descriptor_stat_dimension(collections::get<i>(t)),
          get_stat_dimension_fixed<i + 1>(t));
      }
      else return std::integral_constant<std::size_t, 0_uz>{};
    }
  }


  /**
   * \brief Get the vector dimension of \ref patterns::pattern Arg when transformed into statistical space.
   * \details This is the dimension of a vector corresponding to Arg that has been transformed to Euclidan space for directional statistics.
   */
#ifdef __cpp_concepts
  template<sized_pattern Arg> requires descriptor<Arg> or collections::sized<Arg> or
    values::fixed_value_compares_with<decltype(internal::get_descriptor_stat_dimension(std::declval<stdex::ranges::range_value_t<Arg>>())), 0>
  constexpr values::index auto
#else
  template<typename Arg, std::enable_if_t<descriptor<Arg> or
    (descriptor_collection<Arg> and (collections::sized<Arg> or detail::range_value_has_fixed_stat_dimension<Arg>::value)), int> = 0>
  constexpr auto
#endif
  get_stat_dimension(const Arg& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return internal::get_descriptor_stat_dimension(arg);
    }
    else if constexpr (values::fixed_value_compares_with<collections::size_of<Arg>, 0>)
    {
      return std::integral_constant<std::size_t, 0_uz>{};
    }
    else if constexpr (values::fixed_value_compares_with<collections::size_of<Arg>, stdex::dynamic_extent, &stdex::is_neq>)
    {
      return detail::get_stat_dimension_fixed(arg);
    }
    else // dynamic size
    {
      using C = decltype(internal::get_descriptor_stat_dimension(std::declval<stdex::ranges::range_value_t<Arg>>()));
      if constexpr (values::fixed_value_compares_with<C, 0_uz>)
        return std::integral_constant<std::size_t, 0_uz>{};
      else if constexpr (values::fixed<C>)
        return values::fixed_value_of_v<C> * collections::get_size(arg);
      else
#ifdef __cpp_lib_ranges_fold
        return std::ranges::fold_left(collections::views::all(arg), 0_uz,
          [](const auto& a, const auto& b) { return a + internal::get_descriptor_stat_dimension(b); });
#else
      {
        std::size_t ret = 0_uz;
        for (const auto& c : collections::views::all(arg)) ret += internal::get_descriptor_stat_dimension(c);
        return ret;
      }
#endif
    }
  }


}

#endif

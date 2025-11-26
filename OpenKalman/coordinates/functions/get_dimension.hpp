/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref coordinates::get_dimension.
 */

#ifndef OPENKALMAN_COORDINATES_GET_DIMENSION_HPP
#define OPENKALMAN_COORDINATES_GET_DIMENSION_HPP

#include "collections/collections.hpp"
#include "coordinates/concepts/pattern.hpp"
#include "coordinates/concepts/descriptor.hpp"
#include "coordinates/functions/internal/get_descriptor_dimension.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<std::size_t i = 0, typename T>
    static constexpr auto get_dimension_fixed(const T& t)
    {
      if constexpr (i < collections::size_of_v<T>)
      {
        return values::operation(
          std::plus{},
          internal::get_descriptor_dimension(collections::get<i>(t)),
          get_dimension_fixed<i + 1>(t));
      }
      else return std::integral_constant<std::size_t, 0_uz>{};
    }
  }


  /**
   * \brief Get the vector dimension of \ref coordinates::pattern Arg
   */
#ifdef __cpp_concepts
  template<pattern Arg> requires descriptor<Arg> or collections::sized<Arg>
  constexpr values::index auto
#else
  template<typename Arg, std::enable_if_t<pattern<Arg> and
    (descriptor<Arg> or collections::sized<Arg>), int> = 0>
  constexpr auto
#endif
  get_dimension(const Arg& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return internal::get_descriptor_dimension(arg);
    }
    else if constexpr (values::fixed_value_compares_with<collections::size_of<Arg>, 0>)
    {
      return std::integral_constant<std::size_t, 0_uz>{};
    }
    else if constexpr (values::fixed_value_compares_with<collections::size_of<Arg>, stdex::dynamic_extent, &stdex::is_neq>)
    {
      return detail::get_dimension_fixed(arg);
    }
    else
    {
      using C = decltype(internal::get_descriptor_dimension(std::declval<collections::common_collection_type_t<Arg>>()));
      if constexpr (values::fixed_value_compares_with<C, 0_uz>)
      {
        return std::integral_constant<std::size_t, 0_uz>{};
      }
      else if constexpr (values::fixed<C>)
      {
        return values::fixed_value_of_v<C> * collections::get_size(arg);
      }
      else
      {
#ifdef __cpp_lib_ranges_fold
        return std::ranges::fold_left(collections::views::all(arg), 0_uz,
          [](const auto& a, const auto& b) { return a + internal::get_descriptor_dimension(b); });
#else
        std::size_t ret = 0_uz;
        for (const auto& c : collections::views::all(arg)) { ret += internal::get_descriptor_dimension(c); }
        return ret;
#endif
      }
    }
  }


}

#endif

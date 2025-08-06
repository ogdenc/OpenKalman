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

#include <functional>
#include "basics/basics.hpp"
#include "values/functions/operation.hpp"
#include "collections/concepts/tuple_like.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_dimension.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<std::size_t i = 0, typename Tup>
    static constexpr auto get_dimension_tuple(const Tup& tup)
    {
      if constexpr (i < std::tuple_size_v<Tup>)
      {
        return values::operation(
          std::plus{},
          internal::get_descriptor_dimension(OpenKalman::internal::generalized_std_get<i>(tup)),
          get_dimension_tuple<i + 1>(tup));
      }
      else return std::integral_constant<std::size_t, 0_uz>{};
    }
  } // namespace detail


  /**
   * \brief Get the vector dimension of \ref coordinates::pattern Arg
   */
#ifdef __cpp_concepts
  template<pattern Arg> requires descriptor<Arg> or collections::sized<Arg>
  constexpr values::index auto
#else
  template<typename Arg, std::enable_if_t<pattern<Arg> and (descriptor<Arg> or collections::sized<Arg>), int> = 0>
  constexpr auto
#endif
  get_dimension(const Arg& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return internal::get_descriptor_dimension(arg);
    }
    else if constexpr (collections::size_of_v<Arg> == 0)
    {
      return std::integral_constant<std::size_t, 0_uz>{};
    }
    else if constexpr (collections::tuple_like<Arg>)
    {
      return detail::get_dimension_tuple(arg);
    }
    else
    {
#ifdef __cpp_lib_ranges_fold
      return std::ranges::fold_left(collections::views::all(arg), 0_uz,
        [](const auto& a, const auto& b) { return a + internal::get_descriptor_dimension(b); });
#else
      std::size_t ret = 0_uz;
      for (const auto& c : collections::views::all(arg)) ret += internal::get_descriptor_dimension(c);
      return ret;
#endif
    }
  }


}


#endif

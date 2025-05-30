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
 * \brief Definition for \ref coordinates::get_stat_dimension.
 */

#ifndef OPENKALMAN_GET_EUCLIDEAN_DIMENSION_HPP
#define OPENKALMAN_GET_EUCLIDEAN_DIMENSION_HPP

#include <functional>
#include "basics/global-definitions.hpp"
#include "collections/concepts/tuple_like.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_stat_dimension.hpp"

namespace OpenKalman::coordinates
{
  namespace detail
  {
    template<std::size_t i = 0, typename Tup>
    static constexpr auto get_euclidean_dimension_tuple(const Tup& tup)
    {
      if constexpr (i < std::tuple_size_v<Tup>)
      {
        return values::operation {std::plus{}, internal::get_descriptor_stat_dimension(OpenKalman::internal::generalized_std_get<i>(tup)), get_euclidean_dimension_tuple<i + 1>(tup)};
      }
      else return std::integral_constant<std::size_t, 0_uz>{};
    }
  } // namespace detail


  /**
   * \brief Get the vector dimension of \ref coordinates::pattern Arg when transformed into statistical space.
   * \details This is the dimension of a vector corresponding to Arg that has been transformed to Euclidan space for directional statistics.
   */
#ifdef __cpp_concepts
  template<pattern Arg>
  constexpr values::index auto
#else
  template<typename Arg, std::enable_if_t<pattern<Arg>, int> = 0>
  constexpr auto
#endif
  get_stat_dimension(const Arg& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return internal::get_descriptor_stat_dimension(arg);
    }
    else if constexpr (collections::tuple_like<Arg>)
    {
      return detail::get_euclidean_dimension_tuple(arg);
    }
    else
    {
      std::size_t ret = 0_uz;
      for (auto& c : arg) ret += internal::get_descriptor_stat_dimension(c);
      return ret;
    }
  }


} // namespace OpenKalman::coordinates


#endif //OPENKALMAN_GET_EUCLIDEAN_DIMENSION_HPP

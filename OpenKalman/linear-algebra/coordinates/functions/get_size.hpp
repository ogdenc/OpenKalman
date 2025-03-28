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
 * \brief Definition for \ref coordinate::get_size.
 */

#ifndef OPENKALMAN_COORDINATE_GET_SIZE_HPP
#define OPENKALMAN_COORDINATE_GET_SIZE_HPP

#include <functional>
#include "basics/global-definitions.hpp"
#include "collections/concepts/tuple_like.hpp"
#include "values/classes/operation.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/internal/get_descriptor_size.hpp"

namespace OpenKalman::coordinate
{
  namespace detail
  {
    template<std::size_t i = 0, typename Tup>
    static constexpr auto get_size_tuple(const Tup& tup)
    {
      if constexpr (i < std::tuple_size_v<Tup>)
      {
        return value::operation {std::plus{}, internal::get_descriptor_size(collections::get<i>(tup)), get_size_tuple<i + 1>(tup)};
      }
      else return std::integral_constant<std::size_t, 0_uz>{};
    }
  } // namespace detail


  /**
   * \brief Get the size of \ref coordinate::pattern Arg
   */
#ifdef __cpp_concepts
  template<pattern Arg>
  constexpr value::index auto
#else
  template<typename Arg, std::enable_if_t<pattern<Arg>, int> = 0>
  constexpr auto
#endif
  get_size(const Arg& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return internal::get_descriptor_size(arg);
    }
    else if constexpr (tuple_like<Arg>)
    {
      return detail::get_size_tuple(arg);
    }
    else
    {
      std::size_t ret = 0_uz;
      for (auto& c : arg) ret += internal::get_descriptor_size(c);
      return ret;
    }
  }


} // namespace OpenKalman::coordinate


#endif //OPENKALMAN_COORDINATE_GET_SIZE_HPP

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
 * \brief Definition for \ref get_is_euclidean.
 */

#ifndef OPENKALMAN_GET_IS_EUCLIDEAN_HPP
#define OPENKALMAN_GET_IS_EUCLIDEAN_HPP

#include "collections/concepts/tuple_like.hpp"
#include "values/concepts/index.hpp"
#include "values/classes/operation.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"

namespace OpenKalman::coordinate
{
  namespace detail
  {
    template<typename Arg>
    constexpr auto get_descriptor_is_euclidean(const Arg& arg)
    {
      if constexpr (interface::coordinate_descriptor_traits<Arg>::is_specialized)
      {
        return interface::coordinate_descriptor_traits<Arg>::is_euclidean(arg);
      }
      else
      {
        static_assert(value::index<Arg>);
        return std::true_type{};
      }
    }


    template<std::size_t i = 0, typename Tup>
    static constexpr auto get_is_euclidean_tuple(const Tup& tup)
    {
      if constexpr (i < std::tuple_size_v<Tup>)
      {
        return value::operation {std::logical_and{}, get_descriptor_is_euclidean(collections::get<i>(tup)), get_is_euclidean_tuple<i + 1>(tup)};
      }
      else return std::true_type {};
    }
  } // namespace detail


  /**
   * \brief Determine, whether \ref coordinate::pattern Arg is euclidean.
   */
#ifdef __cpp_concepts
  template<pattern Arg>
#else
  template<typename Arg, std::enable_if_t<pattern<Arg>, int> = 0>
#endif
  constexpr auto
  get_is_euclidean(const Arg& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return detail::get_descriptor_is_euclidean(arg);
    }
    else if constexpr (tuple_like<Arg>)
    {
      return detail::get_is_euclidean_tuple(arg);
    }
    else
    {
      for (auto& c : arg) if (not detail::get_descriptor_is_euclidean(c)) return false;
      return true;
    }
  }


} // namespace OpenKalman::coordinate


#endif //OPENKALMAN_GET_IS_EUCLIDEAN_HPP

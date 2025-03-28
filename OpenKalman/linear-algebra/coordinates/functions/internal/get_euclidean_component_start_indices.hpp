/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref coordinate::internal::get_euclidean_component_start_indices.
 */

#ifndef OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_COMPONENT_START_INDICES_HPP
#define OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_COMPONENT_START_INDICES_HPP

#include <type_traits>
#include <functional>
#include "values/classes/operation.hpp"
#include "collections/concepts/index.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_tuple.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "get_descriptor_euclidean_size.hpp"

namespace OpenKalman::coordinate::internal
{
  namespace detail
  {
    template<std::size_t i = 0, typename Tup, typename CurrLoc = std::integral_constant<std::size_t, 0>, typename...Locs>
    static constexpr auto euclidean_component_start_indices_tuple(const Tup& tup, CurrLoc currloc = {}, Locs...locs)
    {
      if constexpr (i < std::tuple_size_v<Tup>)
      {
        auto next_loc = value::operation {std::plus{}, currloc, get_descriptor_euclidean_size(std::get<i>(tup))};
        return euclidean_component_start_indices_tuple<i + 1>(tup, std::move(next_loc), std::move(locs)..., std::move(currloc));
      }
      else return std::tuple {std::move(locs)...};
    }
  } // namespace detail


  /**
   * \brief A \ref collections::index mapping a component of T to an \ref value::index "index" within a vector.
   * \details The size must be the same as <code>component_collection(t)</code>.
   * Each component of the resulting collection must map to the corresponding starting index within a vector
   * transformed to Euclidean space for directional statistics.
   * \returns A \ref collection of \ref value::index "index" values
   */
#ifdef __cpp_lib_constexpr_vector
  template<pattern Arg>
  constexpr collections::index auto
#else
  template<typename Arg, std::enable_if_t<pattern<Arg>, int> = 0>
  auto
#endif
  get_euclidean_component_start_indices(Arg&& arg)
  {
    if constexpr (descriptor<Arg>)
    {
      return std::array {std::integral_constant<std::size_t, 0>{}};
    }
    else if constexpr (descriptor_tuple<Arg>)
    {
      return detail::euclidean_component_start_indices_tuple(arg);
    }
    else
    {
      std::size_t i = 0, loc = 0;
      std::vector<std::size_t> indices(get_collection_size(arg));
      for (auto& c : arg)
      {
        indices[i++] = loc;
        loc += get_descriptor_euclidean_size(c);
      }
      return indices;
    }
  }


} // namespace OpenKalman::coordinate::internal


#endif //OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_COMPONENT_START_INDICES_HPP

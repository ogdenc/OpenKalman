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
 * \brief Definition for \ref coordinates::internal::get_euclidean_index_table.
 */

#ifndef OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_INDEX_TABLE_HPP
#define OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_INDEX_TABLE_HPP

#include <vector>
#include "collections/views/internal/repeat_tuple_view.hpp"
#include "collections/concepts/index.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "get_descriptor_stat_dimension.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"

namespace OpenKalman::coordinates::internal
{
  namespace detail
  {
    template<std::size_t c = 0, std::size_t local = 0, typename Tup, std::size_t...is>
    static constexpr auto
    euclidean_index_table_tuple(const Tup& tup, std::index_sequence<is...> seq = std::index_sequence<>{})
    {
      if constexpr (c < std::tuple_size_v<Tup>)
      {
        if constexpr (local < stat_dimension_of_v<std::tuple_element_t<c, Tup>>)
          return euclidean_index_table_tuple<c, local + 1>(tup, std::index_sequence<is..., c>{});
        else
          return euclidean_index_table_tuple<c + 1, 0>(tup, seq);
      }
      else return std::tuple {std::integral_constant<std::size_t, is>{}...};
    }
  } // namespace detail


  /**
   * \brief A \ref collection mapping each index of an \ref indexible vector, transformed to statistical space)
   * to a corresponding \ref values::index "index" within component_collection(t).
   * transformed to Euclidean space for directional statistics.
   * \returns A \ref collection of \ref values::index "index" values
   */
#ifdef __cpp_lib_constexpr_vector
  template<pattern Arg>
  constexpr collections::index auto
#else
  template<typename Arg, std::enable_if_t<pattern<Arg>, int> = 0>
  auto
#endif
  get_euclidean_index_table(Arg&& arg)
  {
    if constexpr (stat_dimension_of_v<Arg> != dynamic_size)
    {
      if constexpr (descriptor<Arg>)
      {
        constexpr std::size_t N = values::to_number(get_euclidean_descriptor_size(arg));
        return OpenKalman::collections::internal::repeat_tuple_view<N, std::integral_constant<std::size_t, 0_uz>>(std::integral_constant<std::size_t, 0_uz>{});
      }
      else
      {
        return detail::euclidean_index_table_tuple(arg);
      }
    }
    else
    {
      std::vector<std::size_t> table;
      std::size_t c = 0;
      for (auto& comp : arg)
      {
        for (std::size_t local = 0; local < get_descriptor_stat_dimension(comp); ++local) table.emplace_back(c);
        ++c;
      }
      return table;
    }
  }


} // namespace OpenKalman::coordinates::internal


#endif //OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_INDEX_TABLE_HPP

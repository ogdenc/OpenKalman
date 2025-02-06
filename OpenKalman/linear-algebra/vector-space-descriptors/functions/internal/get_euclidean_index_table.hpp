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
 * \brief Definition for \ref descriptor::internal::get_euclidean_index_table.
 */

#ifndef OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_INDEX_TABLE_HPP
#define OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_INDEX_TABLE_HPP

#include <type_traits>
#include "linear-algebra/values/concepts/index_collection.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/composite_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \brief A \ref internal::collection "collection" mapping each index of an \ref indexible vector, transformed to statistical space)
   * to a corresponding \ref value::index "index" within component_collection(t).
   * transformed to Euclidean space for directional statistics.
   * \returns A \ref internal::collection "collection" of \ref value::index "index" values
   */
#ifdef __cpp_concepts
  template<descriptor::vector_space_descriptor Arg>
  constexpr value::index_collection auto
#else
  template<typename Arg, std::enable_if_t<descriptor::vector_space_descriptor<Arg>, int> = 0>
  constexpr auto
#endif
  get_euclidean_index_table(Arg&& arg)
  {
    if constexpr (descriptor::composite_vector_space_descriptor<Arg>)
    {
      auto ret = interface::coordinate_set_traits<std::decay_t<Arg>>::euclidean_index_table(std::forward<Arg>(arg));
      if constexpr (descriptor::static_vector_space_descriptor<Arg>)
        static_assert(std::tuple_size_v<std::decay_t<decltype(ret)>> == descriptor::vector_space_component_count_v<Arg>);
      return ret;
    }
    else
    {
      return std::array {std::integral_constant<std::size_t, 0_uz>{}};
    }
  }


} // namespace OpenKalman::descriptor::internal


#endif //OPENKALMAN_DESCRIPTORS_GET_EUCLIDEAN_INDEX_TABLE_HPP

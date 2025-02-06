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
 * \brief Definition for \ref descriptor::internal::get_component_start_indices.
 */

#ifndef OPENKALMAN_DESCRIPTORS_GET_COMPONENT_INDEX_HPP
#define OPENKALMAN_DESCRIPTORS_GET_COMPONENT_INDEX_HPP

#include "linear-algebra/values/concepts/index_collection.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/atomic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include <linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp>

namespace OpenKalman::descriptor::internal
{
  /**
   * \brief A \ref value::index_collection mapping a component of T to an \ref value::index "index" within a vector.
   * \details The size must be the same as <code>component_collection(t)</code>.
   * Each component of the resulting collection must map to the corresponding starting index within a vector.
   * \returns A \ref internal::collection "collection" of \ref value::index "index" values
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor Arg>
  constexpr value::index_collection auto
#else
  template<typename Arg, std::enable_if_t<vector_space_descriptor<Arg>, int> = 0>
  constexpr auto
#endif
  get_component_start_indices(Arg&& arg)
  {
    if constexpr (atomic_vector_space_descriptor<Arg>)
    {
      return std::array {std::integral_constant<std::size_t, 0>{}};
    }
    else
    {
      auto ret = interface::coordinate_set_traits<std::decay_t<Arg>>::component_start_indices(std::forward<Arg>(arg));
      if constexpr (descriptor::static_vector_space_descriptor<Arg>)
        static_assert(std::tuple_size_v<std::decay_t<decltype(ret)>> == descriptor::vector_space_component_count_v<Arg>);
      return ret;
    }
  }


} // namespace OpenKalman::descriptor::internal


#endif //OPENKALMAN_DESCRIPTORS_GET_COMPONENT_INDEX_HPP

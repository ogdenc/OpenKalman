/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref descriptor::internal::get_component_collection.
 */

#ifndef OPENKALMAN_DESCRIPTORS_GET_COMPONENT_COLLECTION_HPP
#define OPENKALMAN_DESCRIPTORS_GET_COMPONENT_COLLECTION_HPP

#include "basics/internal/tuple_like.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/coordinate_set_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/composite_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor_collection.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \brief Get the \ref descriptor::vector_space_descriptor_collection associated with \ref vector_space_descriptor T
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor Arg>
  constexpr vector_space_descriptor_collection auto
#else
  template<typename Arg, std::enable_if_t<vector_space_descriptor<Arg>, int> = 0>
  constexpr auto
#endif
  get_component_collection(Arg&& arg)
  {
    if constexpr (composite_vector_space_descriptor<Arg>)
    {
      auto ret = interface::coordinate_set_traits<std::decay_t<Arg>>::component_collection(std::forward<Arg>(arg));
      static_assert(not static_vector_space_descriptor<Arg> or OpenKalman::internal::tuple_like<decltype(ret)>);
      return ret;
    }
    else
    {
      return std::array<std::decay_t<Arg>, 1> {std::forward<Arg>(arg)};
    }
  }


} // namespace OpenKalman::descriptor::internal


#endif //OPENKALMAN_DESCRIPTORS_GET_COMPONENT_COLLECTION_HPP

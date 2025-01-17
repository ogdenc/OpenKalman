/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref canonical_equivalent.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_CANONICAL_EQUIVALENT_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_CANONICAL_EQUIVALENT_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_collection_of.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief Whether <code>a</code> is equivalent to <code>b</code>.
   * \details A is equivalent to B if A and B are the same in their \ref internal::canonical_equivalent "canonical equivalent" forms.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor Arg>
  constexpr descriptor::vector_space_descriptor decltype(auto)
#else
  template<typename Arg, std::enable_if_t<vector_space_descriptor<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  canonical_equivalent(Arg&& arg)
  {
    if constexpr (static_vector_space_descriptor<Arg>)
    {
      if constexpr (vector_space_component_count_v<Arg> == 1)
      {
        auto ret {std::get<0>(interface::vector_space_traits<std::decay_t<Arg>>::collection(std::forward<Arg>(arg)))};
        return ret;
      }
      else return std::apply([](const auto&...a) {
          return descriptor::StaticDescriptor<std::decay_t<decltype(a)>...>{};
        }, descriptor::get_collection_of(arg));
    }
    else return std::forward<Arg>(arg);
  }


} // namespace OpenKalman::descriptor::internal


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_CANONICAL_EQUIVALENT_HPP

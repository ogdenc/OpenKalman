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
 * \brief Definition for \ref get_collection_of.
 */

#ifndef OPENKALMAN_DESCRIPTORS_GET_COLLECTION_OF_HPP
#define OPENKALMAN_DESCRIPTORS_GET_COLLECTION_OF_HPP

#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor_collection.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Get the \ref descriptor::vector_space_descriptor_collection that comprises \ref vector_space_descriptor T
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor Arg>
  constexpr descriptor::vector_space_descriptor_collection auto
#else
  template<typename Arg, std::enable_if_t<vector_space_descriptor<Arg>, int> = 0>
  constexpr auto
#endif
  get_collection_of(Arg&& arg)
  {
    auto ret = interface::vector_space_traits<std::decay_t<Arg>>::collection(std::forward<Arg>(arg));
    static_assert(not static_vector_space_descriptor<Arg> or internal::tuple_like<decltype(ret)>);
    return ret;
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_DESCRIPTORS_GET_COLLECTION_OF_HPP

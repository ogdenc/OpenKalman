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
 * \brief Definition for \ref get_type_index.
 */

#ifndef OPENKALMAN_GET_TYPE_INDEX_HPP
#define OPENKALMAN_GET_TYPE_INDEX_HPP

#include <typeindex>
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/traits-defined.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/atomic_vector_space_descriptor.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Obtain the std::type_index for an \ref atomic_vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<descriptor::atomic_vector_space_descriptor Arg>
#else
  template<typename Arg, std::enable_if_t<descriptor::atomic_vector_space_descriptor<Arg>, int> = 0>
#endif
  constexpr auto
  get_type_index(const Arg& arg)
  {
    if constexpr (interface::type_index_defined_for<Arg>)
    {
      return interface::vector_space_traits<std::decay_t<Arg>>::type_index(arg);
    }
    else if constexpr (descriptor::atomic_vector_space_descriptor<Arg>)
    {
      return std::type_index {typeid(Arg)};
    }
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_GET_TYPE_INDEX_HPP

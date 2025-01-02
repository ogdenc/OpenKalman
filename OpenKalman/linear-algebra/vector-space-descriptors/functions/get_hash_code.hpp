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
 * \brief Definition for \ref get_vector_space_descriptor_is_euclidean.
 */

#ifndef OPENKALMAN_GET_HASH_CODE_HPP
#define OPENKALMAN_GET_HASH_CODE_HPP

#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/traits-defined.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/canonical_equivalent.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Obtain a unique hash code for \ref vector_space_descriptor T.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor Arg>
#else
  template<typename Arg, std::enable_if_t<vector_space_descriptor<T>, int> = 0>
#endif
  constexpr auto
  get_hash_code(const Arg& arg)
  {
    if constexpr (interface::hash_code_defined_for<Arg>)
    {
      return interface::vector_space_traits<std::decay_t<Arg>>::hash_code(arg);
    }
    else
    {
      return typeid(internal::canonical_equivalent(arg)).hash_code();
    }
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_GET_HASH_CODE_HPP

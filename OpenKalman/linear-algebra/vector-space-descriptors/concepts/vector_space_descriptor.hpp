/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref vector_space_descriptor.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_HPP

#include "atomic_vector_space_descriptor.hpp"
#include "composite_vector_space_descriptor.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief An object describing the set of coordinates associated with a tensor index.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept vector_space_descriptor =
#else
  constexpr bool vector_space_descriptor =
#endif
    atomic_vector_space_descriptor<T> or composite_vector_space_descriptor<T>;


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_HPP

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref atomic_static_vector_space_descriptor.
 */

#ifndef OPENKALMAN_ATOMIC_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_ATOMIC_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include "static_vector_space_descriptor.hpp"
#include "composite_vector_space_descriptor.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief T is an atomic (non-separable or non-composite) group of fixed \ref vector_space_descriptor.
   * \details These \ref vector_space_descriptor objects are suitable for incorporation in \ref composite_vector_space_descriptor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept atomic_static_vector_space_descriptor =
#else
  constexpr bool atomic_static_vector_space_descriptor =
#endif
    static_vector_space_descriptor<T> and (not composite_vector_space_descriptor<T>);


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_ATOMIC_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

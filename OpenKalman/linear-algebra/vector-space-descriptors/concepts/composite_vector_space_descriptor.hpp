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
 * \brief Definition for \ref composite_vector_space_descriptor.
 */

#ifndef OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP

#include "linear-algebra/vector-space-descriptors/interfaces/coordinate_set_traits.hpp"
#include "atomic_vector_space_descriptor.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief T is a \ref composite_vector_space_descriptor.
   * \details A container that may include multiple \ref atomic_vector_space_descriptor objects.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept composite_vector_space_descriptor =
#else
  constexpr bool composite_vector_space_descriptor =
#endif
    interface::coordinate_set_traits<std::decay_t<T>>::is_specialized and (not atomic_vector_space_descriptor<T>);


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP

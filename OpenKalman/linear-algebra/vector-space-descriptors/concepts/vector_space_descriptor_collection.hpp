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
 * \brief Definition for \ref vector_space_descriptor_collection.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP

#include "basics/internal/collection.hpp"
#include "vector_space_descriptor_tuple.hpp"
#include "vector_space_descriptor_range.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief An object describing a collection of /ref vector_space_descriptor objects.
   * \details This will be a \ref vector_space_descriptor_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept vector_space_descriptor_collection =
#else
  constexpr bool vector_space_descriptor_collection =
#endif
    internal::collection<T> and (vector_space_descriptor_tuple<T> or vector_space_descriptor_range<T>);


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP

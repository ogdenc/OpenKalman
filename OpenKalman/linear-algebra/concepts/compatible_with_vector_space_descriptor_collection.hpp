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
 * \brief Definition for \ref compatible_with_vector_space_descriptor_collection.
 */

#ifndef OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
#define OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/concepts/indexible.hpp"

namespace OpenKalman
{
  /**
   * \brief \ref indexible T is compatible with \ref pattern_collection D.
   * \tparam T An \ref indexible object
   * \tparam D A \ref pattern_collection
   */
  template<typename T, typename D, auto comp = &stdcompat::is_eq, applicability a = applicability::permitted>
#ifdef __cpp_concepts
  concept compatible_with_vector_space_descriptor_collection =
#else
  constexpr bool compatible_with_vector_space_descriptor_collection =
#endif
    indexible<T> and pattern_collection<D> and
      coordinates::pattern_collection_compares_with<decltype(get_pattern_collection(std::declval<T>())), D, comp, a>;


}

#endif

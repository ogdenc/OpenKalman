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
 * \brief Definition for \ref compares_with_pattern_collection.
 */

#ifndef OPENKALMAN_COMPARES_WITH_PATTERN_COLLECTION_HPP
#define OPENKALMAN_COMPARES_WITH_PATTERN_COLLECTION_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/concepts/indexible.hpp"

namespace OpenKalman
{
  /**
   * \brief Compares the associated pattern collection of \ref indexible T with \ref pattern_collection D.
   * \tparam T An \ref indexible object
   * \tparam D A \ref pattern_collection
   */
  template<typename T, typename D, auto comp = &stdex::is_eq, applicability a = applicability::permitted>
#ifdef __cpp_concepts
  concept compares_with_pattern_collection =
#else
  constexpr bool compares_with_pattern_collection =
#endif
    indexible<T> and
    coordinates::pattern_collection<D> and
    coordinates::pattern_collection_compares_with<decltype(get_pattern_collection(std::declval<T>())), D, comp, a>;

}

#endif

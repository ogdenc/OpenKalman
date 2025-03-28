/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref vector_space_descriptors_match_with.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_MATCH_WITH_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_MATCH_WITH_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that a set of indexible objects have equivalent vector space descriptors for each index.
   * \tparam Ts A set of \ref indexible objects
   * \sa vector_space_descriptors_may_match_with
   * \sa vector_space_descriptors_match
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept vector_space_descriptors_match_with =
#else
  constexpr bool vector_space_descriptors_match_with =
#endif
    vector_space_descriptors_may_match_with<Ts...> and ((not has_dynamic_dimensions<Ts>) and ...);


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_MATCH_WITH_HPP

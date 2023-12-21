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
 * \brief Definition for \ref has_untyped_index.
 */

#ifndef OPENKALMAN_HAS_UNTYPED_INDEX_HPP
#define OPENKALMAN_HAS_UNTYPED_INDEX_HPP


namespace OpenKalman
{
  /**
   * \brief Specifies that T has an untyped index N.
   * \details Index N of T is Euclidean and non-modular (e.g., Axis, TypedIndex<Axis, Axis>, etc.).
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N>
  concept has_untyped_index =
#else
  template<typename T, std::size_t N>
  constexpr bool has_untyped_index =
#endif
    euclidean_vector_space_descriptor<vector_space_descriptor_of_t<T, N>>;


} // namespace OpenKalman

#endif //OPENKALMAN_HAS_UNTYPED_INDEX_HPP

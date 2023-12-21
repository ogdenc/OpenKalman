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
 * \brief Definition for \ref indexible.
 */

#ifndef OPENKALMAN_INDEXIBLE_HPP
#define OPENKALMAN_INDEXIBLE_HPP


namespace OpenKalman
{
  /**
   * \brief T is a generalized tensor type.
   * \details T can be a tensor over a vector space, but can also be an analogous algebraic structure over a
   * tensor product of modules over division rings (e.g., an vector-like structure that contains angles).
   * \internal \sa interface::indexible_object_traits::count_indices
   */
  template<typename T>
#ifdef __cpp_concepts
  concept indexible =
#else
  constexpr bool indexible =
#endif
    interface::count_indices_defined_for<T>;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEXIBLE_HPP

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for best_vector_space_descriptor function.
 */

#ifndef OPENKALMAN_BEST_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_BEST_VECTOR_SPACE_DESCRIPTOR_HPP

namespace OpenKalman::internal
{
  /**
   * \brief Given one or more /ref vector_space_descriptor objects, return the "best" one (i.e., the one that is static).
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor D, vector_space_descriptor...Ds> requires maybe_equivalent_to<D, Ds...>
#else
  template<typename D, typename...Ds, std::enable_if_t<maybe_equivalent_to<D, Ds...>, int> = 0>
#endif
  constexpr decltype(auto) best_vector_space_descriptor(D&& d, Ds&&...ds)
  {
    if constexpr (sizeof...(Ds) == 0) return std::forward<D>(d);
    else if constexpr (fixed_vector_space_descriptor<D>) return std::forward<D>(d);
    else return best_vector_space_descriptor(std::forward<Ds>(ds)...);
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_BEST_VECTOR_SPACE_DESCRIPTOR_HPP

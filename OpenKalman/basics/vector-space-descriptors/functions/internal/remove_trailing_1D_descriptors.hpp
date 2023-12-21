/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition for \ref remove_trailing_1D_descriptors.
 */

#ifndef OPENKALMAN_REMOVE_TRAILING_1D_DESCRIPTORS_HPP
#define OPENKALMAN_REMOVE_TRAILING_1D_DESCRIPTORS_HPP

#include <type_traits>


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Remove any trailing, one-dimensional \ref vector_space_descriptor objects.
   */
#ifdef __cpp_concepts
  template<tuple_like DTup> requires tuple_like<std::decay_t<DTup>>
#else
  template<typename DTup, std::enable_if_t<tuple_like<std::decay_t<DTup>>, int> = 0>
#endif
  constexpr auto remove_trailing_1D_descriptors(DTup&& d_tup)
  {
    constexpr auto N = std::tuple_size_v<DTup>;
    if constexpr (N == 0)
      return std::forward<DTup>(d_tup);
    else if constexpr (equivalent_to<std::tuple_element_t<N - 1, std::decay_t<DTup>>, Dimensions<1>>)
      return remove_trailing_1D_descriptors(tuple_slice<0, N - 1>(std::forward<DTup>(d_tup)));
    else
      return std::forward<DTup>(d_tup);
  }


} // namespace OpenKalman::internal


#endif //OPENKALMAN_REMOVE_TRAILING_1D_DESCRIPTORS_HPP

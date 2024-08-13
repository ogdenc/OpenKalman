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
   * \return A tuple containing the resulting, potentially shortened, list of vector space descriptors
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor...Ds>
#else
  template<typename...Ds, std::enable_if_t<(... and vector_space_descriptor<Ds>), int> = 0>
#endif
  constexpr auto
  remove_trailing_1D_descriptors(Ds&&...ds)
  {
    constexpr auto N = sizeof...(Ds);
    if constexpr (N == 0)
    {
      return std::tuple {};
    }
    else if constexpr (equivalent_to<std::tuple_element_t<N - 1, std::tuple<Ds...>>, Dimensions<1>>)
    {
      return std::apply(
        [](auto&&...ds2){ return remove_trailing_1D_descriptors(std::forward<decltype(ds2)>(ds2)...); },
        tuple_slice<0, N - 1>(std::forward_as_tuple(std::forward<Ds>(ds)...)));
    }
    else
    {
      return std::tuple {std::forward<Ds>(ds)...};
    }
  }


} // namespace OpenKalman::internal


#endif //OPENKALMAN_REMOVE_TRAILING_1D_DESCRIPTORS_HPP

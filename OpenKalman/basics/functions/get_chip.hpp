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
 * \brief Definition of \ref get_chip function.
 */

#ifndef OPENKALMAN_GET_CHIP_HPP
#define OPENKALMAN_GET_CHIP_HPP

namespace OpenKalman
{
  /**
   * \brief Extract a sub-array having rank less than the rank of the input object.
   * \details A chip is a special type of "thin" slice of width 1 in one or more dimensions, and otherwise no
   * reduction in extents. For example, the result could be a row vector, a column vector, a matrix (e.g., if the
   * input object is a rank-3 or higher tensor), etc.
   * \tparam indices The index or indices of the dimension(s) to be collapsed to a single dimension.
   * For example, if the input object is a matrix, a value of {0} will result in a row vector, a value of {1} will
   * result in a column vector, and a value of {0, 1} will result in a one-dimensional vector.
   * If the input object is a rank-3 tensor, a value of {1, 2} will result in a row vector.
   * If no indices are listed, the argument will be returned unchanged.
   * \param ixs The index value corresponding to each of the <code>indices</code>, in the same order. The values
   * may be positive \ref std::integral types or a positive \ref std::integral_constant.
   * \return A sub-array of the argument
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, index_value...Ixs> requires (sizeof...(indices) == sizeof...(Ixs))
#else
  template<std::size_t...indices, typename Arg, typename...Ixs, std::enable_if_t<
    indexible<Arg> and (index_value<Ixs> and ...) and (sizeof...(indices) == sizeof...(Ixs)), int> = 0>
#endif
  constexpr decltype(auto)
  get_chip(Arg&& arg, Ixs...ixs)
  {
    (... , []{
      if constexpr (static_index_value<Ixs> and not dynamic_dimension<Arg, indices>)
        static_assert(std::decay_t<Ixs>::value < index_dimension_of_v<Arg, indices>, "get_chip: indices must be in range");
    }());

    if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
    else return get_block<indices...>(
      std::forward<Arg>(arg),
      std::tuple{ixs...}, // begin points
      std::tuple{(std::integral_constant<decltype(indices), 1> {})...}); // block sizes (always 1 in each collapsed dimension)
  }


} // namespace OpenKalman

#endif //OPENKALMAN_GET_CHIP_HPP

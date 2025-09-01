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
 * \brief Definition of \ref set_chip function.
 */

#ifndef OPENKALMAN_SET_CHIP_HPP
#define OPENKALMAN_SET_CHIP_HPP

namespace OpenKalman
{

  namespace detail
  {
    template<std::size_t>
    constexpr auto chip_index_match() { return std::integral_constant<std::size_t, 0> {}; }

    template<std::size_t ai, std::size_t index, std::size_t...indices, typename I, typename...Is>
    constexpr auto chip_index_match(I i, Is...is)
    {
      if constexpr (ai == index) return i;
      else return chip_index_match<ai, indices...>(is...);
    }

    template<std::size_t...indices, typename Arg, typename Chip, std::size_t...all_indices, typename...Is>
    constexpr Arg& set_chip_impl(Arg&& arg, Chip&& chip, std::index_sequence<all_indices...>, Is...is)
    {
      return set_slice(std::forward<Arg>(arg), std::forward<Chip>(chip), chip_index_match<all_indices, indices...>(is...)...);
    }
  }


  /**
   * \brief Set a sub-array having rank less than the rank of the input object.
   * \details A chip is a special type of "thin" slice of width 1 in one or more dimensions, and otherwise no
   * reduction in extents. For example, the result could be a row vector, a column vector, a matrix (e.g., if the
   * input object is a rank-3 or higher tensor), etc.
   * \tparam indices The index or indices of the dimension(s) that have been collapsed to a single dimension.
   * For example, if the input object is a matrix, a value of {0} will result in a row vector and a value of {1} will
   * result in a column vector. If the input object is a rank-3 tensor, a value of {0, 1} will result in a matrix.
   * \param arg The indexible object in which the chip is to be set.
   * \param chip The chip to be set. It must be a chip, meaning that the dimension is 1 for each of <code>indices</code>.
   * \param is The index value(s) corresponding to <code>indices</code>, in the same order. The values
   * may be positive \ref std::integral types or a positive \ref std::integral_constant.
   * \return arg as modified
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, writable Arg, indexible Chip, values::index...Ixs> requires
    (sizeof...(indices) == sizeof...(Ixs))
#else
  template<std::size_t...indices, typename Arg, typename Chip, typename...Ixs, std::enable_if_t<
    writable<Arg> and indexible<Chip> and (values::index<Ixs> and ...) and (sizeof...(indices) == sizeof...(Ixs)), int> = 0>
#endif
  constexpr Arg&&
  set_chip(Arg&& arg, Chip&& chip, Ixs...ixs)
  {
    (... , []{
      if constexpr (values::fixed<Ixs> and not dynamic_dimension<Arg, indices>)
        static_assert(std::decay_t<Ixs>::value < index_dimension_of_v<Arg, indices>, "set_chip: indices must be in range");
    }());

    static_assert((... and dimension_size_of_index_is<Chip, indices, 1, applicability::permitted>),
      "Argument chip to set_chip must be 1D in all the specified indices.");

    return detail::set_chip_impl<indices...>(std::forward<Arg>(arg), std::forward<Chip>(chip),
      std::make_index_sequence<index_count_v<Arg>> {}, ixs...);
  }

}

#endif

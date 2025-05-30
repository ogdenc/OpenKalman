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
 * \brief Definition of internal::check_block_limits function.
 */

#ifndef OPENKALMAN_CHECK_BLOCK_LIMITS_HPP
#define OPENKALMAN_CHECK_BLOCK_LIMITS_HPP

namespace OpenKalman::internal
{

  namespace detail
  {
    template<std::size_t limit_ix, std::size_t index, typename Arg, typename...Limits>
    constexpr void check_block_limit(const Arg& arg, const Limits&...)
    {
      if constexpr (((limit_ix >= std::tuple_size_v<Limits>) or ...)) return;
      else if constexpr (((values::fixed<std::tuple_element_t<limit_ix, Limits>>) and ...) and not dynamic_dimension<Arg, index>)
      {
        constexpr std::size_t block_limit = (static_cast<std::size_t>(std::decay_t<std::tuple_element_t<limit_ix, Limits>>::value) + ... + 0_uz);
        static_assert(block_limit <= index_dimension_of_v<Arg, index>, "Block limits must be in range");
      }
      /*else // Not necessary: the matrix/tensor library should check runtime limits.
      {
        auto lim = (std::get<limit_ix>(limits) + ... + 0);
        auto max = get_index_dimension_of<index>(arg);
        if (lim < 0 or lim > max) throw std::out_of_range {"Block function limits are out of range for index " + std::to_string(index)};
      }*/
    }

  } // namespace detail


  /**
   * \internal
   * \brief Check the limits of a block function
   */
  template<std::size_t...limit_ix, std::size_t...indices, typename Arg, typename...Limits>
  constexpr void check_block_limits(
    std::index_sequence<limit_ix...>, std::index_sequence<indices...>, const Arg& arg, const Limits&...limits)
  {
    (detail::check_block_limit<limit_ix, indices>(arg, limits...), ...);
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_CHECK_BLOCK_LIMITS_HPP

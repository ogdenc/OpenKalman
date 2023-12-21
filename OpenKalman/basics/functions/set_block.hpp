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
 * \brief Definition of \ref set_block function.
 */

#ifndef OPENKALMAN_SET_BLOCK_HPP
#define OPENKALMAN_SET_BLOCK_HPP

namespace OpenKalman
{
  /**
   * \brief Extract a block from a matrix or tensor.
   * \param arg The indexible object in which the block is to be set.
   * \param block The block to be set.
   * \param begin A tuple specifying, for each index of Arg in order, the beginning \ref index_value.
   * \param size A tuple specifying, for each index of Arg in order, the dimensions of the extracted block.
   * \return arg as modified
   */
#ifdef __cpp_concepts
  template<writable Arg, indexible Block, index_value...Begin> requires (sizeof...(Begin) >= index_count_v<Arg>)
#else
  template<typename Arg, typename Block, typename...Begin, std::enable_if_t<writable<Arg> and indexible<Block> and
    (index_value<Begin> and ...) and (sizeof...(Begin) >= index_count<Arg>::value), int> = 0>
#endif
  constexpr Arg&&
  set_block(Arg&& arg, Block&& block, const Begin&...begin)
  {
    std::index_sequence_for<Begin...> begin_seq;
    internal::check_block_limits(begin_seq, begin_seq, arg, std::tuple{begin...});
    internal::check_block_limits(begin_seq, begin_seq, arg, std::tuple{begin...},
      std::apply([](auto&&...a){
        return std::tuple{[](auto&& a){
          if constexpr (fixed_vector_space_descriptor<decltype(a)>)
            return std::integral_constant<std::size_t, dimension_size_of_v<decltype(a)>> {};
          else
            return get_dimension_size_of(std::forward<decltype(a)>(a));
        }(std::forward<decltype(a)>(a))...};
      }, all_vector_space_descriptors(block)));

    interface::library_interface<std::decay_t<Arg>>::set_block(arg, std::forward<Block>(block), begin...);
    return std::forward<Arg>(arg);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SET_BLOCK_HPP

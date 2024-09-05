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
 * \brief Definition of \ref set_slice function.
 */

#ifndef OPENKALMAN_SET_SLICE_HPP
#define OPENKALMAN_SET_SLICE_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename Arg, typename Block, typename BeginTup, typename...J>
    static void assign_slice_elements(Arg& arg, Block&& block, const BeginTup& begin_tup, std::index_sequence<>, J...j)
    {
      set_component(arg, get_component(std::forward<Block>(block), std::get<0>(j)...), std::get<1>(j)...);
    }


    template<typename Arg, typename Block, typename BeginTup, std::size_t I, std::size_t...Is, typename...J>
    static void assign_slice_elements(Arg& arg, Block&& block, const BeginTup& begin_tup, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(block); i++)
        assign_slice_elements(arg, std::forward<Block>(block), begin_tup, std::index_sequence<Is...>{}, j...,
          std::tuple{i, std::get<I>(begin_tup) + i});
    }
  } // namespace detail


  /**
   * \brief Assign an object to a particular slice of a matrix or tensor.
   * \param arg The \ref writable object in which the slice is to be assigned.
   * \param block The block to be set.
   * \param begin A tuple specifying, for each index of Arg in order, the beginning \ref index_value.
   * \param size A tuple specifying, for each index of Arg in order, the dimensions of the extracted block.
   * \return A reference to arg as modified.
   * \todo Add a static check that Block has the correct vector space descriptors
   */
#ifdef __cpp_concepts
  template<writable Arg, indexible Block, index_value...Begin> requires (sizeof...(Begin) >= index_count_v<Arg>)
#else
  template<typename Arg, typename Block, typename...Begin, std::enable_if_t<writable<Arg> and indexible<Block> and
    (index_value<Begin> and ...) and (sizeof...(Begin) >= index_count<Arg>::value), int> = 0>
#endif
  constexpr Arg&&
  set_slice(Arg&& arg, Block&& block, const Begin&...begin)
  {
    std::index_sequence_for<Begin...> begin_seq;
    internal::check_block_limits(begin_seq, begin_seq, arg, std::tuple{begin...});
    internal::check_block_limits(begin_seq, begin_seq, arg, std::tuple{begin...},
      std::apply([](auto&&...a) -> decltype(auto) {
        return std::forward_as_tuple(get_dimension_size_of(std::forward<decltype(a)>(a))...);
      }, all_vector_space_descriptors(block)));

    if constexpr (interface::set_slice_defined_for<Arg, Arg&, Block&&, const Begin&...>)
    {
      interface::library_interface<std::decay_t<Arg>>::set_slice(arg, std::forward<Block>(block), begin...);
    }
    else if constexpr (interface::set_slice_defined_for<Arg, Arg&, decltype(to_native_matrix<Arg>(std::declval<Block&&>())), const Begin&...>)
    {
      interface::library_interface<std::decay_t<Arg>>::set_slice(arg, to_native_matrix<Arg>(std::forward<Block>(block)), begin...);;
    }
    // \todo If arg is directly_accessible and the library interface is not defined, set the block based on the raw data.
    else
    {
      std::make_index_sequence<sizeof...(Begin)> seq;
      detail::assign_slice_elements(arg, std::forward<Block>(block), std::forward_as_tuple(begin...), seq);
    }
    return std::forward<Arg>(arg);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SET_SLICE_HPP

/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Functions that tile multiple objects into a larger object.
 */

#ifndef OPENKALMAN_TILE_HPP
#define OPENKALMAN_TILE_HPP

namespace OpenKalman
{
  // ====== //
  //  tile  //
  // ====== //

  namespace detail
  {
    template<std::size_t direction, typename...Index, std::size_t...dims, typename Arg>
    constexpr void tile_impl(std::tuple<Index...>& current_position, std::tuple<Index...>& current_block_size,
      std::index_sequence<dims...>, Arg& arg) {}


    template<std::size_t direction, typename...Index, std::size_t...dims, typename Arg, typename Block, typename...Blocks>
    constexpr void tile_impl(std::tuple<Index...>& current_position, std::tuple<Index...>& current_block_size,
      std::index_sequence<dims...> seq, Arg& arg, Block&& block, Blocks&&...blocks)
    {

      if constexpr (direction == 0) ((std::get<dims>(current_block_size) = get_index_dimension_of<dims>(block)),...);

      auto& cur_pos = std::get<direction>(current_position);
      auto dim_direction = get_index_dimension_of<direction>(arg);

      if constexpr (direction == sizeof...(dims) - 1)
      {
        set_block(arg, std::forward<Block>(block), std::get<dims>(current_position)...);

        if (cur_pos > 0 and ((dims != direction and get_index_dimension_of<dims>(block) != std::get<dims>(current_block_size)) or ...))
          throw std::length_error {"Block argument to tile function is not the right tile size."};

        cur_pos += get_index_dimension_of<direction>(block);

        if (cur_pos < dim_direction)
        {
          tile_impl<direction>(current_position, current_block_size, seq, arg, std::forward<Blocks>(blocks)...);
        }
        else if (cur_pos == dim_direction)
        {
          constexpr std::size_t new_direction = direction - 1;
          cur_pos = 0;
          std::get<new_direction>(current_position) += get_index_dimension_of<new_direction>(block);
          tile_impl<new_direction>(current_position, current_block_size, seq, arg, std::forward<Blocks>(blocks)...);
        }
        else // cur_pos > dim_direction
        {
          throw std::length_error {"Block argument to tile function is too large in dimension " + std::to_string(direction) + "."};
        }
      }
      else
      {
        static_assert((direction < sizeof...(dims) - 1));
        std::get<direction>(current_block_size) = get_index_dimension_of<direction>(block);

        if (cur_pos < dim_direction)
        {
          tile_impl<direction + 1>(current_position, current_block_size, seq, arg, std::forward<Block>(block), std::forward<Blocks>(blocks)...);
        }
        else
        {
          if constexpr (direction > 0)
          {
            if (cur_pos > dim_direction) throw std::length_error {
              "Block argument to tile function is too large in dimension " + std::to_string(direction) + "."};
            constexpr std::size_t new_direction = direction - 1;
            cur_pos = 0;
            std::get<new_direction>(current_position) += get_index_dimension_of<new_direction>(block);
            tile_impl<new_direction>(current_position, current_block_size, seq, arg, std::forward<Block>(block), std::forward<Blocks>(blocks)...);
          }
          else
          {
            throw std::length_error {"Tile function has too many blocks to fit within specified index descriptors."};
          }
        }
      }
    }
  } // namespace detail


  /**
   * \brief Create a matrix or tensor by tiling individual blocks.
   * \tparam Ds A set of index descriptors for the resulting matrix or tensor.
   * \tparam Block The first block
   * \tparam Blocks Subsequent blocks
   */
#ifdef __cpp_concepts
  template<index_descriptor...Ds, indexible Block, indexible...Blocks>
  requires (sizeof...(Ds) >= std::max({max_indices_of_v<Block>, max_indices_of_v<Blocks>...}))
#else
  template<typename...Ds, typename Block, typename...Blocks, std::enable_if_t<
    (index_descriptor<Ds> and ...) and (indexible<Block> and ... and indexible<Blocks>) and
    (sizeof...(Ds) >= std::max({max_indices_of<Block>::value, max_indices_of<Blocks>::value...})), int> = 0>
#endif
  constexpr decltype(auto) tile(const std::tuple<Ds...>& ds_tuple, Block&& block, Blocks&&...blocks)
  {
    if constexpr (sizeof...(Blocks) == 0)
    {
      return std::forward<Block>(block);
    }
    else
    {
      auto m = std::apply(
        [](auto&&...d) { return make_default_dense_writable_matrix_like<Block>(std::forward<decltype(d)>(d)...); },
        ds_tuple);

      auto current_position = std::tuple{(index_descriptor<Ds> ? std::size_t(0) : std::size_t(-1))...};
      decltype(current_position) current_block_size;

      detail::tile_impl<0>(current_position, current_block_size, std::index_sequence_for<Ds...> {}, m, block, blocks...);
      return m;
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TILE_HPP

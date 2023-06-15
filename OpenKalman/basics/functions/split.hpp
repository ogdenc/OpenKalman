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
 * \brief Functions that split objects into smaller parts.
 */

#ifndef OPENKALMAN_SPLIT_HPP
#define OPENKALMAN_SPLIT_HPP

namespace OpenKalman
{
  using namespace interface;

  // ======= //
  //  split  //
  // ======= //

  namespace detail
  {
    template<std::size_t index, typename Arg, typename...Ds>
    constexpr void check_split_descriptors(Arg&& arg, Ds&&...ds)
    {
      if constexpr ((dynamic_index_descriptor<Ds> or ... or dynamic_dimension<Arg, index>))
      {
        if (not ((get_dimension_size_of(ds) + ... + std::size_t{0}) <= get_index_descriptor<index>(arg)))
          throw std::logic_error {"When concatenated, the index descriptors provided to split function are not a "
            "prefix of the argument's index descriptor along at least index " + std::to_string(index)};
      }
      else
      {
        static_assert(prefix_of<TypedIndex<std::decay_t<Ds>...>, index_descriptor_of_t<Arg, index>>,
          "Concatenated index descriptors provided to split function must be a prefix of the argument's index descriptor");
      }
    }


    template<std::size_t index>
    constexpr auto split_dummy(std::size_t x) { return x; };

    template<std::size_t...indices, typename Arg, typename Blocks_tup>
    auto split_symmetric(Arg&& arg, std::size_t begin, Blocks_tup&& blocks_tup)
    {
      return std::forward<Blocks_tup>(blocks_tup);
    }

    template<std::size_t...indices, typename Arg, typename Blocks_tup, typename D, typename...Ds>
    auto split_symmetric(Arg&& arg, std::size_t begin, Blocks_tup&& blocks_tup, D&& d, Ds&&...ds)
    {
      auto block_size = get_dimension_size_of(d);
      auto begin_tup = std::tuple{split_dummy<indices>(begin)...};
      auto size_tup = std::tuple{split_dummy<indices>(block_size)...};
      auto block = get_block<indices...>(std::forward<Arg>(arg), begin_tup, size_tup);
      auto new_blocks_tup = std::tuple_cat(blocks_tup, std::tuple {std::move(block)});
      return split_symmetric<indices...>(std::forward<Arg>(arg), begin + block_size, std::move(new_blocks_tup), std::forward<Ds>(ds)...);
    }

  } // namespace detail


  /**
   * \brief Split a matrix or tensor into sub-parts, where the split is the same for every index.
   * \details This is an inverse of the \ref OpenKalman::concatenate "concatenate" operation.
   * In other words, for all <code>std::size_t i..., j...</code> and <code>indexible a...</code>, and given
   * the function <code>template<std::size_t...i> auto f(auto a) { return get_index_descriptor<i>(a)...}; }</code>
   * <code>((split<i...>(concatenate<i...>(a...), get_index_descriptor<j>(a)...) == std::tuple{a...}) and ...)</code>.
   * \tparam indices The indices along which to make the split. E.g., 0 means to split along rows,
   * 1 means to split along columns, {0, 1} means to split diagonally.
   * \tparam Arg The matrix or tensor to be split.
   * \tparam Ds A set of index descriptors (the same for for each of indices)
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, index_descriptor...Ds> requires (sizeof...(indices) > 0)
#else
  template<std::size_t...indices, typename Arg, typename...Ds, std::enable_if_t<indexible<Arg> and
    (index_descriptor<Ds> and ...) and (sizeof...(indices) > 0), int> = 0>
#endif
  inline auto
  split(Arg&& arg, Ds&&...ds)
  {
    (detail::check_split_descriptors<indices>(arg, ds...),...);
    return detail::split_symmetric<indices...>(arg, 0, std::tuple{}, std::forward<Ds>(ds)...);
  }


  namespace detail
  {
    template<std::size_t index, std::size_t index_ix, typename Arg, typename...Ds_tups>
    constexpr void check_split_descriptors_tup_impl(Arg&& arg, Ds_tups&&...ds_tups)
    {
      check_split_descriptors<index>(arg, std::get<index_ix>(ds_tups)...);
    }

    template<std::size_t...indices, typename Arg, typename...Ds_tups, std::size_t...indices_ix>
    constexpr void check_split_descriptors_tup(Arg&& arg, std::index_sequence<indices_ix...>, Ds_tups&&...ds_tups)
    {
      (check_split_descriptors_tup_impl<indices, indices_ix>(arg, ds_tups...),...);
    }


    template<std::size_t...indices, typename Arg, typename Begin_tup, typename Blocks_tup, std::size_t...indices_ix>
    auto split_impl(Arg&& arg, Begin_tup begin_tup, Blocks_tup&& blocks_tup, std::index_sequence<indices_ix...>)
    {
      return std::forward<Blocks_tup>(blocks_tup);
    }

    template<std::size_t...indices, typename Arg, typename Begin_tup, typename Blocks_tup, std::size_t...indices_ix,
      typename Ds_tup, typename...Ds_tups>
    auto split_impl(Arg&& arg, Begin_tup begin_tup, Blocks_tup&& blocks_tup, std::index_sequence<indices_ix...> seq,
      Ds_tup&& ds_tup, Ds_tups&&...ds_tups)
    {
      auto size_tup = std::tuple{get_dimension_size_of(std::get<indices_ix>(ds_tup))...};
      auto block = get_block<indices...>(std::forward<Arg>(arg), begin_tup, size_tup);
      auto new_blocks_tup = std::tuple_cat(blocks_tup, std::tuple {std::move(block)});
      auto new_begin_tup = std::tuple{std::get<indices_ix>(begin_tup) + std::get<indices_ix>(size_tup)...};
      return split_impl<indices...>(std::forward<Arg>(arg), new_begin_tup, std::move(new_blocks_tup), seq, std::forward<Ds_tups>(ds_tups)...);
    }

  } // namespace detail


  /**
   * \overload
   * \brief Split a matrix or tensor into sub-parts of a size defined independently for each index.
   * \details This is an inverse of the \ref OpenKalman::concatenate "concatenate" operation.
   * In other words, for all <code>std::size_t i...</code> and <code>indexible a...</code>, and given
   * the function <code>template<std::size_t...i> auto f(auto a) { return std::tuple{get_index_descriptor<i>(a)...}; }</code>
   * <code>split<i...>(concatenate<i...>(a...), f<i...>(a)...) == std::tuple{a...}</code>.
   * \tparam indices The indices along which to make the split. E.g., 0 means to split along rows,
   * 1 means to split along columns, {0, 1} means to split diagonally.
   * \tparam Arg The matrix or tensor to be split.
   * \tparam Ds_tups A set of tuples of index descriptors, each tuple having <code>sizeof...(indices)</code> elements
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, typename...Ds_tups> requires
    (sizeof...(indices) > 0) and (sizeof...(Ds_tups) > 0) and
    ((sizeof...(indices) == std::tuple_size<Ds_tups>::value) and ...)
#else
  template<std::size_t...indices, typename Arg, typename...Ds_tups, std::enable_if_t<indexible<Arg> and
    (sizeof...(indices) > 0) and (sizeof...(Ds_tups) > 0) and
    ((sizeof...(indices) == std::tuple_size<Ds_tups>::value) and ...), int> = 0>
#endif
  inline auto
  split(Arg&& arg, const Ds_tups&...ds_tups)
  {
    std::make_index_sequence<sizeof...(indices)> seq;
    detail::check_split_descriptors_tup<indices...>(arg, seq, ds_tups...);

    if constexpr (sizeof...(indices) == 1)
    {
      return detail::split_symmetric<indices...>(arg, 0, std::tuple{}, std::get<0>(ds_tups)...);
    }
    else
    {
      auto init_begin = std::tuple {detail::split_dummy<indices>(0)...};
      return detail::split_impl<indices...>(arg, init_begin, std::tuple{}, seq, ds_tups...);
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SPLIT_HPP

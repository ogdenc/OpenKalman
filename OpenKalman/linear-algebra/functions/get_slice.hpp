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
 * \brief Definition of \ref get_slice function.
 */

#ifndef OPENKALMAN_GET_SLICE_HPP
#define OPENKALMAN_GET_SLICE_HPP

namespace OpenKalman
{

  namespace detail
  {
    template<bool is_offset, std::size_t arg_ix, typename Arg>
    constexpr auto get_block_limits(const Arg& arg)
    {
      if constexpr (is_offset) return std::integral_constant<std::size_t, 0>{};
      else if constexpr (dynamic_dimension<Arg, arg_ix>) return get_index_dimension_of<arg_ix>(arg);
      else return std::integral_constant<std::size_t, index_dimension_of_v<Arg, arg_ix>>{};
    }


    template<bool is_offset, std::size_t arg_ix, std::size_t index, std::size_t...indices, typename Arg, typename Limit, typename...Limits>
    constexpr auto get_block_limits(const Arg& arg, const Limit& limit, const Limits&...limits)
    {
      if constexpr (arg_ix == index)
      {
        static_assert(((index != indices) and ...), "Duplicate index parameters are not allowed in block function.");
        return limit;
      }
      else
      {
        return get_block_limits<is_offset, arg_ix, indices...>(arg, limits...);
      }
    }


    template<bool is_offset, std::size_t...indices, typename Arg, typename Limit_tup, std::size_t...arg_ix, std::size_t...limits_ix>
    constexpr auto expand_block_limits(std::index_sequence<arg_ix...>, std::index_sequence<limits_ix...>, const Arg& arg, const Limit_tup& limit_tup)
    {
      return std::tuple {get_block_limits<is_offset, arg_ix, indices...>(arg, std::get<limits_ix>(limit_tup)...)...};
    }


    template<typename Arg, typename...Offset, typename...Extent, std::size_t...Ix>
    constexpr auto get_slice_impl(Arg&& arg, const std::tuple<Offset...>& offsets, const std::tuple<Extent...>& extents, std::index_sequence<Ix...> seq)
    {
      auto slice_descriptors = std::tuple {
        get_slice<scalar_type_of_t<Arg>>(get_vector_space_descriptor<Ix>(std::forward<Arg>(arg)), std::get<Ix>(offsets), std::get<Ix>(extents))...};

      if constexpr (constant_matrix<Arg>)
      {
        return make_constant<Arg>(constant_coefficient{arg}, std::move(slice_descriptors));
      }
      else
      {
        return make_vector_space_adapter(
          interface::library_interface<std::decay_t<Arg>>::get_slice(std::forward<Arg>(arg), offsets, extents),
          std::move(slice_descriptors));
      }
      // \todo If arg is directly_accessible and the library interface is not defined, extract the block from the raw data.
    }

  } // namespace detail


  /**
   * \brief Extract a slice from a matrix or tensor.
   * \details If indices are specified, only those indices will be subsetted. Otherwise, the Offset and Extent parameters
   * are taken in index order. Any omitting trailing indices (for which there are no Offset or Extent parameters) are included whole.
   * \tparam indices The index or indices of the particular dimensions to be specified, in any order (optional).
   * If this is omitted, the Offset and Extent parameters proceed in index order.
   * \param arg The indexible object from which a slice is to be taken.
   * \param offsets A tuple corresponding to each of indices, each element specifying the offsetning \ref value::index.
   * If indices are not specified, the tuple proceeds in normal index order.
   * \param extents A tuple corresponding to each of indices, each element specifying the dimensions of the extracted block.
   * If indices are not specified, the tuple proceeds in normal index order.
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, value::index...Offset, value::index...Extent> requires
    (sizeof...(Offset) == sizeof...(Extent)) and internal::has_uniform_static_vector_space_descriptors<Arg, indices...> and
    (sizeof...(indices) == 0 or sizeof...(indices) == sizeof...(Offset))
  constexpr indexible decltype(auto)
#else
  template<std::size_t...indices, typename Arg, typename...Offset, typename...Extent, std::enable_if_t<
    indexible<Arg> and (value::index<Offset> and ...) and (value::index<Extent> and ...) and
    (sizeof...(Offset) == sizeof...(Extent)) and internal::has_uniform_static_vector_space_descriptors<Arg, indices...> and
    (sizeof...(indices) == 0 or sizeof...(indices) == sizeof...(Offset)), int> = 0>
  constexpr decltype(auto)
#endif
  get_slice(Arg&& arg, const std::tuple<Offset...>& offsets, const std::tuple<Extent...>& extents)
  {
    if constexpr (sizeof...(Offset) == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      std::index_sequence_for<Offset...> offset_seq;
      std::conditional_t<sizeof...(indices) == 0, decltype(offset_seq), std::index_sequence<indices...>> indices_seq;
      internal::check_block_limits(offset_seq, indices_seq, arg, offsets);
      internal::check_block_limits(offset_seq, indices_seq, arg, offsets, extents);

      if constexpr (sizeof...(indices) == 0)
      {
        return detail::get_slice_impl(std::forward<Arg>(arg), offsets, extents, offset_seq);
      }
      else
      {
        auto arg_ix_seq = std::make_index_sequence<index_count_v<Arg>>{};
        return detail::get_slice_impl(std::forward<Arg>(arg),
          detail::expand_block_limits<true, indices...>(arg_ix_seq, offset_seq, arg, offsets),
          detail::expand_block_limits<false, indices...>(arg_ix_seq, offset_seq, arg, extents),
		  offset_seq);
      }
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_GET_SLICE_HPP

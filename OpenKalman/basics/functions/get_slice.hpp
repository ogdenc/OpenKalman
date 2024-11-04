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
    template<bool is_begin, std::size_t arg_ix, typename Arg>
    constexpr auto get_block_limits(const Arg& arg)
    {
      if constexpr (is_begin) return std::integral_constant<std::size_t, 0>{};
      else if constexpr (dynamic_dimension<Arg, arg_ix>) return get_index_dimension_of<arg_ix>(arg);
      else return std::integral_constant<std::size_t, index_dimension_of_v<Arg, arg_ix>>{};
    }


    template<bool is_begin, std::size_t arg_ix, std::size_t index, std::size_t...indices, typename Arg, typename Limit, typename...Limits>
    constexpr auto get_block_limits(const Arg& arg, const Limit& limit, const Limits&...limits)
    {
      if constexpr (arg_ix == index)
      {
        static_assert(((index != indices) and ...), "Duplicate index parameters are not allowed in block function.");
        return limit;
      }
      else
      {
        return get_block_limits<is_begin, arg_ix, indices...>(arg, limits...);
      }
    }


    template<bool is_begin, std::size_t...indices, typename Arg, typename Limit_tup, std::size_t...arg_ix, std::size_t...limits_ix>
    constexpr auto expand_block_limits(std::index_sequence<arg_ix...>, std::index_sequence<limits_ix...>, const Arg& arg, const Limit_tup& limit_tup)
    {
      return std::tuple {get_block_limits<is_begin, arg_ix, indices...>(arg, std::get<limits_ix>(limit_tup)...)...};
    }


    template<typename Arg, typename...Begin, typename...Size, std::size_t...Ix>
    constexpr auto get_slice_descriptors(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size, std::index_sequence<Ix...>)
    {
      return std::tuple{get_vector_space_descriptor_slice(get_vector_space_descriptor<Ix>(std::forward<Arg>(arg)), std::get<Ix>(begin), std::get<Ix>(size))...};
    }


    template<typename Arg, typename...Begin, typename...Size, std::size_t...Ix>
    constexpr auto get_slice_impl(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size, std::index_sequence<Ix...> seq)
    {
      if constexpr (zero<Arg>)
      {
        return std::apply([](auto&&...ds){
          return make_zero<Arg>(std::forward<decltype(ds)>(ds)...);
          }, get_slice_descriptors(std::forward<Arg>(arg), begin, size, seq));
      }
      else if constexpr (constant_matrix<Arg>)
      {
        return std::apply(
          [](const auto& c, auto&&...ds){
            return make_constant<Arg>(c, std::forward<decltype(ds)>(ds)...);
            }, std::tuple_cat(std::tuple{constant_coefficient{arg}}, get_slice_descriptors(std::forward<Arg>(arg), begin, size, seq)));
      }
      else
      {
        return std::apply(
          [](auto&& a, auto&&...ds){
            return make_vector_space_adapter(std::forward<decltype(a)>(a), std::forward<decltype(ds)>(ds)...);
            }, std::tuple_cat(
              std::forward_as_tuple(interface::library_interface<std::decay_t<Arg>>::get_slice(std::forward<Arg>(arg), begin, size)),
              get_slice_descriptors(std::forward<Arg>(arg), begin, size, seq)));
      }
      // \todo If arg is directly_accessible and the library interface is not defined, extract the block from the raw data.
    }

  } // namespace detail


  /**
   * \overload
   * \brief Extract a slice from a matrix or tensor.
   * \details If indices are specified, only those indices will be subsetted. Otherwise, the Begin and Size parameters
   * are taken in index order. Any omitting trailing indices (for which there are no Begin or Size parameters) are included whole.
   * \tparam indices The index or indices of the particular dimensions to be specified, in any order (optional).
   * If this is omitted, the Begin and Size parameters proceed in index order.
   * \param arg The indexible object from which a slice is to be taken.
   * \param begin A tuple corresponding to each of indices, each element specifying the beginning \ref index_value.
   * If indices are not specified, the tuple proceeds in normal index order.
   * \param size A tuple corresponding to each of indices, each element specifying the dimensions of the extracted block.
   * If indices are not specified, the tuple proceeds in normal index order.
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, index_value...Begin, index_value...Size> requires
    (sizeof...(Begin) == sizeof...(Size)) and internal::has_uniform_fixed_vector_space_descriptors<Arg, indices...> and
    (sizeof...(indices) == 0 or sizeof...(indices) == sizeof...(Begin))
  constexpr indexible decltype(auto)
#else
  template<std::size_t...indices, typename Arg, typename...Begin, typename...Size, std::enable_if_t<
    indexible<Arg> and (index_value<Begin> and ...) and (index_value<Size> and ...) and
    (sizeof...(Begin) == sizeof...(Size)) and internal::has_uniform_fixed_vector_space_descriptors<Arg, indices...> and
    (sizeof...(indices) == 0 or sizeof...(indices) == sizeof...(Begin)), int> = 0>
  constexpr decltype(auto)
#endif
  get_slice(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size)
  {
    if constexpr (sizeof...(Begin) == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      std::index_sequence_for<Begin...> begin_seq;
      std::conditional_t<sizeof...(indices) == 0, decltype(begin_seq), std::index_sequence<indices...>> indices_seq;
      internal::check_block_limits(begin_seq, indices_seq, arg, begin);
      internal::check_block_limits(begin_seq, indices_seq, arg, begin, size);

      if constexpr (sizeof...(indices) == 0)
      {
        return detail::get_slice_impl(std::forward<Arg>(arg), begin, size, begin_seq);
      }
      else
      {
        auto arg_ix_seq = std::make_index_sequence<index_count_v<Arg>>{};
        return detail::get_slice_impl(std::forward<Arg>(arg),
          detail::expand_block_limits<true, indices...>(arg_ix_seq, begin_seq, arg, begin),
          detail::expand_block_limits<false, indices...>(arg_ix_seq, begin_seq, arg, size),
		  begin_seq);
      }
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_GET_SLICE_HPP

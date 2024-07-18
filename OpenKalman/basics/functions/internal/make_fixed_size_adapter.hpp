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
 * \brief Definition for make_fixed_size_adapter function.
 */

#ifndef OPENKALMAN_MAKE_FIXED_SIZE_ADAPTER_HPP
#define OPENKALMAN_MAKE_FIXED_SIZE_ADAPTER_HPP

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename Arg, typename DTup, std::size_t...Ix>
    constexpr decltype(auto)
    make_fixed_size_adapter_impl(Arg&& arg, DTup&& d_tup, std::index_sequence<Ix...> seq)
    {
      if constexpr (fixed_size_adapter<Arg>)
      {
        return make_fixed_size_adapter_impl(nested_object(std::forward<Arg>(arg)),
          std::forward_as_tuple(best_vector_space_descriptor(get_vector_space_descriptor(arg, Ix), std::get<Ix>(d_tup))...), seq);
      }
      else if constexpr ((... or (dynamic_dimension<Arg, Ix> and not dynamic_dimension<DTup, Ix>)) or sizeof...(Ix) < index_count_v<Arg>)
      {
        return FixedSizeAdapter {std::forward<Arg>(arg),
          best_vector_space_descriptor(get_vector_space_descriptor(arg, Ix), std::get<Ix>(d_tup))...};
      }
      else
      {
        return std::forward<Arg>(arg);
      }
    }
  } // namespace detail


  /**
   * \brief Make the best possible \ref FixedSizeAdapter, if applicable, based on a set of vector space descriptors.
   * \tparam Ds Vector space descriptors reflecting the dimensions of the new object
   * \return (1) A fixed size adapter or (2) a reference to the argument unchanged.
   */
#ifdef __cpp_concepts
  template<indexible Arg, vector_space_descriptor...Ds> requires
    compatible_with_vector_space_descriptors<Arg, Ds...> and (index_count_v<Arg> != dynamic_size)
#else
  template<typename Arg, typename...Ds, std::enable_if_t<(... and vector_space_descriptor<Ds>) and
    compatible_with_vector_space_descriptors<Arg, Ds...> and (index_count<Arg>::value != dynamic_size), int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_size_adapter(Arg&& arg, Ds&&...ds)
  {
    auto d_tup {remove_trailing_1D_descriptors(std::forward<Ds>(ds)...)};
    std::make_index_sequence<std::tuple_size_v<decltype(d_tup)>> seq {};
    return detail::make_fixed_size_adapter_impl(std::forward<Arg>(arg), std::move(d_tup), seq);
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_MAKE_FIXED_SIZE_ADAPTER_HPP

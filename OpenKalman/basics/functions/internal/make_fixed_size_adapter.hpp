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
    template<typename...Ts>
    using best_desc = std::decay_t<decltype(best_vector_space_descriptor(std::declval<Ts>()...))>;


    template<typename DTup, typename Arg, std::size_t...Ix>
    constexpr decltype(auto)
    make_fixed_size_adapter_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      if constexpr (sizeof...(Ix) == 0)
      {
        if constexpr (index_count_v<Arg> > 0) return FixedSizeAdapter {std::forward<Arg>(arg)};
        else return std::forward<Arg>(arg);
      }
      else
      {
        using F = FixedSizeAdapter<Arg, best_desc<vector_space_descriptor_of_t<Arg, Ix>, std::tuple_element_t<Ix, DTup>>...>;
        if constexpr ((... or (dynamic_dimension<Arg, Ix> and not dynamic_dimension<F, Ix>)) or sizeof...(Ix) < index_count_v<Arg>)
          return F {std::forward<Arg>(arg)};
        else
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
  template<vector_space_descriptor...Ds, compatible_with_vector_space_descriptors<Ds...> Arg> requires
    (index_count_v<Arg> != dynamic_size)
#else
  template<typename...Ds, typename Arg, std::enable_if_t<(... and vector_space_descriptor<Ds>) and
    compatible_with_vector_space_descriptors<Arg, Ds...> and (index_count<Arg>::value != dynamic_size), int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_size_adapter(Arg&& arg)
  {
    using DTup = decltype(remove_trailing_1D_descriptors(std::declval<std::tuple<Ds...>>()));
    return detail::make_fixed_size_adapter_impl<DTup>(std::forward<Arg>(arg), std::make_index_sequence<std::tuple_size_v<DTup>>{});
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_MAKE_FIXED_SIZE_ADAPTER_HPP

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
    template<typename Arg, std::size_t N, std::size_t...offset>
    constexpr bool an_extended_dim_is_dynamic_impl(std::index_sequence<offset...>)
    {
      return ((... or (dynamic_dimension<Arg, N + offset>)));
    }

    template<typename Arg, std::size_t N>
    constexpr bool an_extended_dim_is_dynamic()
    {
      if constexpr (index_count_v<Arg> != dynamic_size and index_count_v<Arg> > N)
        return an_extended_dim_is_dynamic_impl<Arg, N>(std::make_index_sequence<index_count_v<Arg> - N>{});
      else
        return false;
    }


    template<typename DTup, typename Arg, std::size_t...Ix>
    constexpr decltype(auto)
    make_fixed_size_adapter_impl(Arg&& arg, std::index_sequence<Ix...> seq)
    {
      if constexpr ((... or (dynamic_vector_space_descriptor<std::tuple_element_t<Ix, DTup>> != dynamic_dimension<Arg, Ix>)) or
        an_extended_dim_is_dynamic<Arg, sizeof...(Ix)>())
      {
        return FixedSizeAdapter<Arg,
          decltype(best_vector_space_descriptor(get_vector_space_descriptor(std::declval<Arg>(), Ix),
            std::get<Ix>(std::declval<DTup>())))...> {std::forward<Arg>(arg)};
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
  template<vector_space_descriptor...Ds, compatible_with_vector_space_descriptors<Ds...> Arg> requires
    internal::not_more_fixed_than<Arg, Ds...>
#else
  template<typename...Ds, typename Arg, std::enable_if_t<(... and vector_space_descriptor<Ds>) and
    compatible_with_vector_space_descriptors<Arg, Ds...> and internal::not_more_fixed_than<Arg, Ds...>, int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_size_adapter(Arg&& arg)
  {
    if constexpr (fixed_size_adapter<Arg>)
    {
      return make_fixed_size_adapter<Ds...>(nested_object(std::forward<Arg>(arg)));
    }
    else
    {
      using DTup = std::decay_t<decltype(remove_trailing_1D_descriptors(std::declval<Ds>()...))>;
      std::make_index_sequence<std::tuple_size_v<DTup>> seq {};
      return detail::make_fixed_size_adapter_impl<DTup>(std::forward<Arg>(arg), seq);
    }
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_MAKE_FIXED_SIZE_ADAPTER_HPP

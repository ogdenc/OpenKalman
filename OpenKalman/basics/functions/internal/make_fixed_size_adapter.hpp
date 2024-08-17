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
    template<typename DTup, typename Arg, std::size_t...Ix>
    constexpr decltype(auto)
    make_fixed_size_adapter_impl(Arg&& arg, std::index_sequence<Ix...> seq)
    {
      return FixedSizeAdapter<Arg,
        decltype(best_vector_space_descriptor(get_vector_space_descriptor(std::declval<Arg>(), Ix),
          std::get<Ix>(std::declval<DTup>())))...> {std::forward<Arg>(arg)};
    }
  } // namespace detail


  /**
   * \brief Make the best possible \ref FixedSizeAdapter, if applicable, based on a set of vector space descriptors.
   * \details The function will only return an adapter if at least one of the specified vector space descriptors is
   * more fixed than the corresponding descriptor of the argument.
   * \tparam Ds Vector space descriptors reflecting the dimensions of the new object
   * \return (1) A fixed size adapter if some dimension becomes more fixed, or (2) a reference to the argument unchanged.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor...Ds, compatible_with_vector_space_descriptors<Ds...> Arg>
#else
  template<typename...Ds, typename Arg, std::enable_if_t<(... and vector_space_descriptor<Ds>) and
    compatible_with_vector_space_descriptors<Arg, Ds...>, int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_size_adapter(Arg&& arg)
  {
    if constexpr (fixed_size_adapter<Arg>)
    {
      return make_fixed_size_adapter<Ds...>(nested_object(std::forward<Arg>(arg)));
    }
    else if constexpr (internal::less_fixed_than<Arg, Ds...>)
    {
      using DTup = std::decay_t<decltype(remove_trailing_1D_descriptors(std::declval<Ds>()...))>;
      std::make_index_sequence<std::tuple_size_v<DTup>> seq {};
      return detail::make_fixed_size_adapter_impl<DTup>(std::forward<Arg>(arg), seq);
    }
    else
    {
      return std::forward<Arg>(arg);
    }
  }


} // namespace OpenKalman::internal

#endif //OPENKALMAN_MAKE_FIXED_SIZE_ADAPTER_HPP

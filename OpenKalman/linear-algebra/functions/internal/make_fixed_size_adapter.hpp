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
        std::tuple<decltype(most_fixed_pattern(get_pattern_collection(std::declval<Arg>(), Ix),
          std::get<Ix>(std::declval<DTup>())))...>> {std::forward<Arg>(arg)};
    }
  }


  /**
   * \brief Make the best possible \ref FixedSizeAdapter, if applicable, based on a set of vector space descriptors.
   * \details The function will only return an adapter if at least one of the specified vector space descriptors is
   * more fixed than the corresponding descriptor of the argument.
   * \tparam Descriptors A \rev pattern_collection reflecting the new object
   * \return (1) A fixed size adapter if some dimension becomes more fixed, or (2) a reference to the argument unchanged.
   */
#ifdef __cpp_concepts
  template<pattern_collection Descriptors, compatible_with_vector_space_descriptor_collection<Descriptors> Arg>
#else
  template<typename Descriptors, typename Arg, std::enable_if_t<pattern_collection<Descriptors> and
    compatible_with_vector_space_descriptor_collection<Arg, Descriptors>, int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_size_adapter(Arg&& arg)
  {
    if constexpr (fixed_size_adapter<Arg>)
    {
      return make_fixed_size_adapter<Descriptors>(nested_object(std::forward<Arg>(arg)));
    }
    else if constexpr (internal::less_fixed_than<Arg, Descriptors>)
    {
      using DTup = std::decay_t<decltype(patterns::internal::strip_1D_tail(std::declval<Descriptors>()))>;
      std::make_index_sequence<collections::size_of_v<DTup>> seq {};
      return detail::make_fixed_size_adapter_impl<DTup>(std::forward<Arg>(arg), seq);
    }
    else
    {
      return std::forward<Arg>(arg);
    }
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<patterns::pattern...Ds, compatible_with_vector_space_descriptor_collection<std::tuple<Ds...>> Arg>
#else
  template<typename...Ds, typename Arg, std::enable_if_t<(... and patterns::pattern<Ds>) and
    compatible_with_vector_space_descriptor_collection<Arg, std::tuple<Ds...>>, int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_size_adapter(Arg&& arg)
  {
    return make_fixed_size_adapter<std::tuple<Ds...>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<indexible Arg, pattern_collection Descriptors>
    requires compatible_with_vector_space_descriptor_collection<Arg, Descriptors>
#else
  template<typename Arg, typename Descriptors, std::enable_if_t<indexible<Arg> and pattern_collection<Descriptors> and
    compatible_with_vector_space_descriptor_collection<Arg, Descriptors>, int> = 0>
#endif
  constexpr decltype(auto)
  make_fixed_size_adapter(Arg&& arg, Descriptors&&)
  {
    return make_fixed_size_adapter<Descriptors>(std::forward<Arg>(arg));
  }


}

#endif
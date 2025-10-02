/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref vector_space_descriptors_match function.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_MATCH_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_MATCH_HPP

#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"

namespace OpenKalman
{

  namespace detail
  {
    template<std::size_t...Is>
    constexpr bool vector_space_descriptors_match_impl(std::index_sequence<Is...>) { return true; }

    template<std::size_t...Is, typename T, typename...Ts>
    constexpr bool vector_space_descriptors_match_impl(std::index_sequence<Is...>, const T& t, const Ts&...ts)
    {
      return ([](auto I_const, const T& t, const Ts&...ts){
        constexpr std::size_t I = decltype(I_const)::value;
        return ((get_pattern_collection<I>(t) == get_pattern_collection<I>(ts)) and ...);
      }(std::integral_constant<std::size_t, Is>{}, t, ts...) and ...);
    }


    constexpr bool vector_space_descriptors_match_dyn_impl() { return true; }

    template<typename T, typename...Ts>
    constexpr bool vector_space_descriptors_match_dyn_impl(const T& t, const Ts&...ts)
    {
      auto count = std::max({static_cast<std::size_t>(count_indices(t)), static_cast<std::size_t>(count_indices(ts))...});
      for (std::size_t i = 0; i < count; ++i)
        if (((get_pattern_collection(t, i) != get_pattern_collection(ts, i)) or ...)) return false;
      return true;
    }
  }


  /**
   * \brief Return true if every set of \ref coordinates::pattern of a set of objects match.
   * \tparam Ts A set of tensors or matrices
   * \sa vector_space_descriptors_match_with
   * \sa vector_space_descriptors_may_match_with
   */
#ifdef __cpp_concepts
  template<indexible...Ts>
#else
  template<typename...Ts, std::enable_if_t<(indexible<Ts> and ...), int> = 0>
#endif
  constexpr bool vector_space_descriptors_match(const Ts&...ts)
  {
    if constexpr ((... and values::fixed<decltype(count_indices(ts))>))
    {
      constexpr std::make_index_sequence<std::max({std::decay_t<decltype(count_indices(ts))>::value...})> seq;
      return detail::vector_space_descriptors_match_impl(seq, ts...);
    }
    else
    {
      return detail::vector_space_descriptors_match_dyn_impl(ts...);
    }
  }




}

#endif

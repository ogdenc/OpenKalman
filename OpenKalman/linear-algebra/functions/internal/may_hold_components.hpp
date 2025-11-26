/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref may_hold_components function.
 */

#ifndef OPENKALMAN_MAY_HOLD_COMPONENTS_HPP
#define OPENKALMAN_MAY_HOLD_COMPONENTS_HPP


namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename T, std::size_t N, std::size_t...I>
    constexpr bool may_hold_components_impl(std::index_sequence<I...>)
    {
      constexpr auto dims = ((dynamic_dimension<T, I> ? 1 : index_dimension_of_v<T, I>) * ... * 1);
      if constexpr (N == 0) return dims == 0;
      else if constexpr (dims == 0) return false;
      else return N % dims == 0;
    }
  }


  template<typename T, typename...Components>
#ifdef __cpp_concepts
  concept may_hold_components = indexible<T> and (std::convertible_to<Components, const scalar_type_of_t<T>> and ...) and
#else
  constexpr bool may_hold_components = indexible<T> and (stdex::convertible_to<Components, const scalar_type_of_t<T>> and ...) and
#endif
    detail::may_hold_components_impl<T, sizeof...(Components)>(std::make_index_sequence<index_count_v<T>> {});


}

#endif

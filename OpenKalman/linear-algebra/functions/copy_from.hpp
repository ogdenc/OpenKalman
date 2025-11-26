/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref copy_from function.
 */

#ifndef OPENKALMAN_COPY_FROM_HPP
#define OPENKALMAN_COPY_FROM_HPP

#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/get_index_extent.hpp"
#include "linear-algebra/traits/access.hpp"
#include "linear-algebra/concepts/copyable_from.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename LHS, typename RHS, typename...J>
    static void copy_tensor_elements(LHS&& lhs, RHS&& rhs, std::index_sequence<>, J...j)
    {
      access(lhs, j...) = access(std::forward<RHS>(rhs), j...);
    }


    template<typename LHS, typename RHS, std::size_t I, std::size_t...Is, typename...J>
    static void copy_tensor_elements(LHS&& lhs, RHS&& rhs, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_extent<I>(rhs); i++)
        copy_tensor_elements(lhs, std::forward<RHS>(rhs), std::index_sequence<Is...> {}, j..., i);
    }
  }


  /**
   * \brief Copy elements from one object to another.
   * \param dest The destination object.
   * \param source The source object.
   * \return A reference to the destination object as modified
   * \details By default, this simply loops through each element of the associated mdspans.
   * \internal To provide a custom solution, define \ref interface::library_interface::copy.
   */
#ifdef __cpp_concepts
  template<indexible Dest, indexible Source> requires copyable_from<Dest, Source>
#else
  template<typename Source, typename Dest, std::enable_if_t<copyable_from<Dest, Source>, int> = 0>
#endif
  constexpr decltype(auto)
  copy_from(Dest&& dest, Source&& source)
  {
    if constexpr (interface::copy_from_defined_for<Dest&, Source&&>)
    {
      interface::library_interface<Dest>::copy(dest, std::forward<Source>(source));
    }
    else if constexpr (interface::copy_from_defined_for<Dest&, decltype(get_mdspan(std::declval<Source&&>()))>)
    {
      interface::library_interface<Dest>::copy(dest, get_mdspan(std::forward<Source>(source)));
    }
    else
    {
      detail::copy_tensor_elements(get_mdspan(dest), get_mdspan(std::forward<Source>(source)), std::make_index_sequence<index_count_v<Dest>>{});
    }
    return std::forward<Dest>(dest);
  }


}

#endif

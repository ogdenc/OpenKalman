/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref assign function.
 */

#ifndef OPENKALMAN_ASSIGN_HPP
#define OPENKALMAN_ASSIGN_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/concepts/writable.hpp"
#include "linear-algebra/interfaces/library-interfaces-defined.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/functions/get_component.hpp"
#include "linear-algebra/functions/set_component.hpp"
#include "linear-algebra/functions/to_native_matrix.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename M, typename Arg, typename...J>
    static void copy_tensor_elements(M& m, Arg&& arg, std::index_sequence<>, J...j)
    {
      set_component(m, get_component(std::forward<Arg>(arg), j...), j...);
    }


    template<typename M, typename Arg, std::size_t I, std::size_t...Is, typename...J>
    static void copy_tensor_elements(M& m, Arg&& arg, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_extent<I>(arg); i++)
        copy_tensor_elements(m, std::forward<Arg>(arg), std::index_sequence<Is...> {}, j..., i);
    }
  }


  /**
   * \brief Assign a writable object from an indexible object.
   * \tparam LHS The writable object to be assigned.
   * \tparam RHS The indexible object from which to assign.
   * \return the assigned object as modified
   */
#ifdef __cpp_concepts
  template<writable LHS, vector_space_descriptors_may_match_with<LHS> RHS> requires (not std::is_const_v<LHS>)
#else
  template<typename LHS, typename RHS, std::enable_if_t<
    writable<LHS> and
    vector_space_descriptors_may_match_with<RHS, LHS> and
    (not std::is_const_v<LHS>), int> = 0>
#endif
  constexpr LHS&
  assign(LHS& lhs, RHS&& rhs)
  {
    if constexpr (interface::assign_defined_for<LHS&, RHS&&>)
    {
      interface::library_interface<LHS>::assign(lhs, std::forward<RHS>(rhs));
    }
    else if constexpr (interface::assign_defined_for<LHS&, decltype(to_native_matrix<LHS>(std::declval<RHS&&>()))>)
    {
      interface::library_interface<LHS>::assign(lhs, to_native_matrix<LHS>(std::forward<RHS>(rhs)));
    }
    else if constexpr (stdcompat::assignable_from<LHS&, RHS&&>)
    {
      lhs = std::forward<RHS>(rhs);
    }
    else if constexpr (stdcompat::assignable_from<LHS&, decltype(to_native_matrix<LHS>(std::declval<RHS&&>()))>)
    {
      lhs = to_native_matrix<LHS>(std::forward<RHS>(rhs));
    }
    // \todo include the case where A is \ref directly_accessible
    else
    {
      detail::copy_tensor_elements(lhs, std::forward<RHS>(rhs), std::make_index_sequence<index_count_v<LHS>>{});
    }
    return std::forward<LHS>(lhs);
  }


}

#endif

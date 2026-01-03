/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref object_traits for std::mdspan.
 */

#ifndef OPENKALMAN_INTERFACES_MDSPAN_LIBRARY_HPP
#define OPENKALMAN_INTERFACES_MDSPAN_LIBRARY_HPP

#include "basics/basics.hpp"
#include "mdspan-object.hpp"
#include "linear-algebra/interfaces/library_interface.hpp"
#include "to_diagonal_mdspan_policies.hpp"
#include "transpose_mdspan_policies.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief Library interface to an std::mdspan.
   */
  template<typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
  struct library_interface<stdex::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>>
  {
    static constexpr auto
    to_diagonal = [](auto&& m) -> decltype(auto)
    {
      if constexpr (Extents::rank() == 0)
      {
        return std::forward<decltype(m)>(m);
      }
      else
      {
        auto ext = patterns::to_extents(patterns::to_diagonal_pattern_collection(m.extents()));
        using extents_type = std::decay_t<decltype(ext)>;
        using nested_layout = typename std::decay_t<decltype(m)>::layout_type;
        using nested_accessor = typename std::decay_t<decltype(m)>::accessor_type;
        using mapping_type = typename layout_to_diagonal<nested_layout>::template mapping<extents_type>;
        using accessor_type = to_diagonal_accessor<nested_accessor>;
        auto nested_m = m.mapping();
        auto acc = accessor_type {m.accessor(), nested_m.required_span_size()};
        auto map = mapping_type {std::move(nested_m), std::move(ext)};
        return stdex::mdspan(m.data_handle(), std::move(map), std::move(acc));
      }
    };


    static constexpr auto
    conjugate = [](auto&& m) -> decltype(auto)
    {
      return stdex::linalg::conjugated(std::forward<decltype(m)>(m));
    };


    template<std::size_t indexa, std::size_t indexb>
    static constexpr auto
    transpose = [](auto&& m) -> decltype(auto)
    {
      if constexpr (Extents::rank() == 2 and indexa == 0 and indexb == 1)
      {
        return stdex::linalg::transposed(std::forward<decltype(m)>(m));
      }
      else
      {
        using nested_layout = typename std::decay_t<decltype(m)>::layout_type;
        return stdex::mdspan(
          m.data_handle(),
          layout_transpose<nested_layout, indexa, indexb>::mapping(m.mapping()),
          m.accessor());
      }
    };


    template<std::size_t indexa, std::size_t indexb>
    static constexpr auto
    adjoint = [](auto&& m) -> decltype(auto)
    {
      if constexpr (Extents::rank() == 2 and indexa == 0 and indexb == 1)
        return stdex::linalg::conjugate_transposed(std::forward<decltype(m)>(m));
      else
        return conjugate(transpose<indexa, indexb>(std::forward<decltype(m)>(m)));
    };

  };

}


#endif

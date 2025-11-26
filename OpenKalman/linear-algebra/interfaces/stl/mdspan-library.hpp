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

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief Library interface to an std::mdspan.
   */
  template<typename T, typename IndexType, std::size_t...Extents, typename LayoutPolicy, typename AccessorPolicy>
  struct library_interface<stdex::mdspan<T, stdex::extents<IndexType, Extents...>, LayoutPolicy, AccessorPolicy>>
  {
  private:

    using M = stdex::mdspan<T, stdex::extents<IndexType, Extents...>, LayoutPolicy, AccessorPolicy>;

    using extents = stdex::extents<IndexType, Extents...>;

    static constexpr std::size_t rank = extents::rank();
    static constexpr std::size_t rank_dynamic = extents::rank_dynamic();

  public:

    static constexpr auto
    conjugate = [](auto&& m)
    {
      return stdex::linalg::conjugated(std::forward<decltype(m)>(m));
    };


    static constexpr auto
#ifdef __cpp_concepts
    transpose = [](auto&& m) -> decltype(auto) requires (rank <= 2)
#else
    transpose = [](auto&& m, std::enable_if_t<std::decay_t<decltype(m)>::rank() <= 2, int> = 0) -> decltype(auto)
#endif
    {
      if constexpr (rank == 2)
        return stdex::linalg::transposed(std::forward<decltype(m)>(m));
      else if constexpr (rank == 1 and rank_dynamic == 0)
        return stdex::extents<IndexType, 1_uz, Extents...> {};
      else if constexpr (rank == 1)
        return stdex::extents<IndexType, 1_uz, Extents...> {m.extent(0)};
      else // if constexpr (rank == 0)
        return std::forward<decltype(m)>(m);
    };


    static constexpr auto
#ifdef __cpp_concepts
    adjoint = [](auto&& m) -> decltype(auto) requires (rank <= 2)
#else
    adjoint = [](auto&& m, std::enable_if_t<std::decay_t<decltype(m)>::rank() <= 2, int> = 0) -> decltype(auto)
#endif
    {
      if constexpr (rank == 2)
        return stdex::linalg::conjugate_transposed(std::forward<decltype(m)>(m));
      else
        return conjugate(transpose(std::forward<decltype(m)>(m)));
    };

  };

}


#endif

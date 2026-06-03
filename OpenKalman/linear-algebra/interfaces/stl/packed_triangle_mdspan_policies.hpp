/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief mdspan policies for packed triangular matrices
 */

#ifndef OPENKALMAN_PACKED_TRIANGLE_MDSPAN_POLICIES_HPP
#define OPENKALMAN_PACKED_TRIANGLE_MDSPAN_POLICIES_HPP

#include <iostream>
#include "patterns/patterns.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief A layout policy that returns a 1D index to elements of a generalized BLAS-packed triangle.
   * \details The resulting array can have any number of ranks, and need not be square.
   * If there are more than two ranks, every 2D slice of the array will be triangular.
   * If the result is not square, extra zero-elements or explicit elements will be added along the larger dimension.
   */
  template<typename Triangle, typename StorageOrder>
  struct layout_packed_triangle
  {
    template<class Extents>
    struct mapping
    {
      using extents_type = Extents;
      using index_type = typename extents_type::index_type;
      using size_type = typename extents_type::size_type;
      using rank_type = typename extents_type::rank_type;
      using layout_type = layout_packed_triangle;

    private:

      static constexpr bool large =
        (std::is_same_v<Triangle, stdex::linalg::lower_triangle_t> and
          std::is_same_v<StorageOrder, stdex::linalg::column_major_t>) or
        (std::is_same_v<Triangle, stdex::linalg::upper_triangle_t> and
          std::is_same_v<StorageOrder, stdex::linalg::row_major_t>);

      constexpr auto
      min_dim()
      {
        if constexpr (extents_type::rank() < 2) return 1;
        else return std::min(extents_.extent(0), extents_.extent(1));
      }

      constexpr auto
      max_dim()
      {
        if constexpr (extents_type::rank() < 2) return extents_.extent(0);
        else return std::max(extents_.extent(0), extents_.extent(1));
      }

      template<rank_type r>
      constexpr index_type
      calc_offset()
      {
        if constexpr (r < extents_type::rank())
        return calc_offset<r + 1>();
      }

    public:

      constexpr explicit
      mapping(const extents_type& e) : extents_(e) {}

      constexpr const extents_type&
      extents() const noexcept { return extents_; }

#ifdef __cpp_concepts
      constexpr index_type
      operator() (std::convertible_to<index_type> auto i0) const requires (extents_type::rank() == 1)
#else
      template<typename I0, std::enable_if_t<
        std::is_convertible_v<I0, index_type> and (extents_type::rank() == 1), int> = 0>
      constexpr index_type
      operator() (I0 i0) const
#endif
      {
        if (i0 > i1) return (*this)(i1, i0);
        if constexpr (large) return i1 + extents_.extent(1) * i0 - i0 * (i0 + 1) / 2;
        else
        {
          auto e0 = extents_.extent(0);
          if (i1 > e0) return (3 + i0) * i0 / 2 + e0 * (i1 - e0) + i0;
          return i0 + i1 * (i1 + 1) / 2;
        }
      }

#ifdef __cpp_concepts
      constexpr index_type
      operator() (
        std::convertible_to<index_type> auto i0,
        std::convertible_to<index_type> auto i1,
        std::convertible_to<index_type> auto...is) const requires (sizeof...(is) + 2 == extents_type::rank())
#else
      template<typename I0, typename I1, typename...Is, std::enable_if_t<
        std::is_convertible_v<I0, index_type> and
        std::is_convertible_v<I1, index_type> and
        (... and std::is_convertible_v<Is, index_type>) and
        (sizeof...(Is) + 2 == extents_type::rank()), int> = 0>
      constexpr index_type
      operator() (I0 i0, I1 i1, Is...is) const
#endif
      {
        if (i0 > i1) return (*this)(i1, i0);
        if constexpr (large) return i1 + extents_.extent(1) * i0 - i0 * (i0 + 1) / 2;
        else
        {
          auto e0 = extents_.extent(0);
          if (i1 > e0) return (3 + i0) * i0 / 2 + e0 * (i1 - e0) + i0;
          return i0 + i1 * (i1 + 1) / 2;
        }
      }

      constexpr index_type
      required_span_size() const noexcept
      {
        constexpr std::size_t min = min_dim(), max = max_dim();
        if constexpr (large) return min * (min + 1) / 2 + min * (max - min);
        else return min * (min + 1) / 2;
      }

      static constexpr bool is_always_unique() noexcept
      {
        if constexpr (extents_type::rank() == 0) return true;
        else if constexpr (extents_type::rank() == 1) return large;
        else return
          (extents_type::static_extent(0) != stdex::dynamic_extent and extents_type::static_extent(0) < 2) or
          (extents_type::static_extent(1) != stdex::dynamic_extent and extents_type::static_extent(1) < 2);
      }

      static constexpr bool
      is_always_exhaustive() noexcept { return true; }

      static constexpr bool
      is_always_strided() noexcept { return is_always_unique(); }

      constexpr bool
      is_unique() const noexcept
      {
        if constexpr (extents_type::rank() == 0) return true;
        else if constexpr (extents_type::rank() == 1) return large;
        else return extents_.extent(0) < 2 or extents_.extent(1) < 2;
      }

      constexpr bool
      is_exhaustive() const noexcept { return true; }

      constexpr bool
      is_strided() const noexcept { return is_unique(); }

#ifdef __cpp_concepts
      constexpr index_type
      stride(std::size_t r) const noexcept requires (extents_type::rank() > 0)
#else
      template<bool Enable = true, std::enable_if_t<Enable and (extents_type::rank() > 0), int> = 0>
      constexpr index_type
      stride(std::size_t r) const noexcept
#endif
      {
        if constexpr (std::is_same_v<StorageOrder, stdex::linalg::column_major_t>)
        {
          index_type s = 1;
          for (rank_type i = 0; i < r; ++i) s *= extents_.extent(i);
          return s;
        }
        else
        {
          static_assert(std::is_same_v<StorageOrder, stdex::linalg::row_major_t>);
          index_type s = 1;
          for (rank_type i = r + 1; i < extents_type::rank(); ++i) s *= extents_.extent(i);
          return s;
        }
      }

      template<class OtherExtents>
      friend constexpr bool
      operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
      {
        return patterns::compare_pattern_collections(lhs.extents(), rhs.extents());
      }

    private:

      extents_type extents_;

    };
  };


}

#endif

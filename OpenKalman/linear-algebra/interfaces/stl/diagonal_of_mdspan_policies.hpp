/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief mdspan policies for \ref diagonal_of operation
 */

#ifndef OPENKALMAN_DIAGONAL_OF_MDSPAN_POLICIES_HPP
#define OPENKALMAN_DIAGONAL_OF_MDSPAN_POLICIES_HPP

#include <iostream>
#include "patterns/patterns.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief A layout policy that returns a 1D index to a diagonal matrix with the nested object being the main diagonal.
   */
  template<typename NestedLayout, typename NestedExtents>
  struct layout_diagonal_of
  {
    template<class Extents>
    struct mapping
    {
      using extents_type = Extents;
      using index_type = typename extents_type::index_type;
      using size_type = typename extents_type::size_type;
      using rank_type = typename extents_type::rank_type;
      using layout_type = layout_diagonal_of;

    private:

      using nested_extents_type = NestedExtents;
      using nested_mapping_type = typename NestedLayout::template mapping<nested_extents_type>;

      template<typename...Is>
      constexpr index_type
      access_with_padded_indices(Is...is) const
      {
        if constexpr (sizeof...(Is) < nested_extents_type::rank())
          return access_with_padded_indices(is..., 0_uz);
        else
          return nested_mapping_(is...);
      }

    public:

      constexpr explicit
      mapping(const nested_mapping_type& map, const extents_type& e)
        : nested_mapping_(map), extents_(e) {}

      constexpr const extents_type&
      extents() const noexcept { return extents_; }

#ifdef __cpp_concepts
      constexpr index_type
      operator() () const requires (extents_type::rank() == 0)
#else
      template<bool Enable = true, std::enable_if_t<Enable and (extents_type::rank() == 0), int> = 0>
      constexpr index_type
      operator() () const
#endif
      {
        return access_with_padded_indices();
      }

#ifdef __cpp_concepts
      constexpr index_type
      operator() (
        std::convertible_to<index_type> auto i0,
        std::convertible_to<index_type> auto...is) const requires (1 + sizeof...(is) == extents_type::rank())
#else
      template<typename IndexType0, typename...IndexTypes, std::enable_if_t<
        std::is_convertible_v<IndexType0, index_type> and
        (... and std::is_convertible_v<IndexTypes, index_type>) and
        (1 + sizeof...(IndexTypes) == extents_type::rank()), int> = 0>
      constexpr index_type
      operator() (IndexType0 i0, IndexTypes...is) const
#endif
      {
        return access_with_padded_indices(i0, i0, is...);
      }

      constexpr index_type
      required_span_size() const noexcept { return nested_mapping_.required_span_size(); }

      static constexpr bool is_always_unique() noexcept { return nested_mapping_type::is_always_unique(); }

      static constexpr bool
      is_always_exhaustive() noexcept
      {
        if constexpr (not nested_mapping_type::is_always_exhaustive()) return false;
        else if constexpr (nested_extents_type::rank() == 0) return true;
        else if constexpr (nested_extents_type::rank() == 1) return nested_extents_type::static_extent(0) == 1;
        else return nested_extents_type::static_extent(0) == 1 and nested_extents_type::static_extent(1) == 1;
      }

      static constexpr bool
      is_always_strided() noexcept { return nested_mapping_type::is_always_strided(); }

      constexpr bool
      is_unique() const { return nested_mapping_type::is_unique(); }

      constexpr bool
      is_exhaustive() const
      {
        if (not nested_mapping_type::is_exhaustive()) return false;
        if constexpr (nested_extents_type::rank() == 0) return true;
        else if constexpr (nested_extents_type::rank() == 1) return nested_mapping_.extents().extent(0) == 1;
        else return nested_mapping_.extents().extent(0) == 1 and nested_mapping_.extents().extent(1) == 1;
      }

      constexpr bool
      is_strided() const { return nested_mapping_type::is_strided(); }

      constexpr index_type
      stride(std::size_t r) const
      {
        if (r == 0) return nested_mapping_.stride(0) + nested_mapping_.stride(1);
        return nested_mapping_.stride(r - 1);
      }

      template<class OtherExtents>
      friend constexpr bool
      operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
      {
        return patterns::compare_pattern_collections(lhs.extents(), rhs.extents());
      }

    private:

      nested_mapping_type nested_mapping_;
      extents_type extents_;

    };
  };


}

#endif

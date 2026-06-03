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
 * \brief mdspan layout for \ref to_diagonal operation
 */

#ifndef OPENKALMAN_LAYOUT_TO_DIAGONAL_HPP
#define OPENKALMAN_LAYOUT_TO_DIAGONAL_HPP

#include "patterns/patterns.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief A layout policy that returns a 1D index to accessor of the nested object, if on the diagonal,
   * or the nested required_span_size() otherwise.
   */
  template<typename NestedLayout, typename NestedExtents>
  struct layout_to_diagonal
  {
    template<class Extents>
    struct mapping
    {
      using extents_type = Extents;
      using index_type = typename extents_type::index_type;
      using size_type = typename extents_type::size_type;
      using rank_type = typename extents_type::rank_type;
      using layout_type = layout_to_diagonal;

    private:

      using nested_extents_type = NestedExtents;
      using nested_mapping_type = typename NestedLayout:: template mapping<nested_extents_type>;

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
      operator() (std::convertible_to<index_type> auto i0) const requires (extents_type::rank() == 1)
#else
      template<typename IndexType0, std::enable_if_t<
        std::is_convertible_v<IndexType0, index_type> and
        (extents_type::rank() == 1), int> = 0>
      constexpr index_type
      operator() (IndexType0 i0) const
#endif
      {
        if (i0 == 0) return access_with_padded_indices();
        return nested_mapping_.required_span_size();
      }

#ifdef __cpp_concepts
      constexpr index_type
      operator() (
        std::convertible_to<index_type> auto i0,
        std::convertible_to<index_type> auto i1,
        std::convertible_to<index_type> auto...is) const requires (2 + sizeof...(is) == extents_type::rank())
#else
      template<typename IndexType0, typename IndexType1, typename...IndexTypes, std::enable_if_t<
        std::is_convertible_v<IndexType0, index_type> and
        std::is_convertible_v<IndexType1, index_type> and
        (... and std::is_convertible_v<IndexTypes, index_type>) and
        (2 + sizeof...(IndexTypes) == extents_type::rank()), int> = 0>
      constexpr index_type
      operator() (IndexType0 i0, IndexType1 i1, IndexTypes...is) const
#endif
      {
        if (i0 == i1) return access_with_padded_indices(i1, is...);
        return nested_mapping_.required_span_size();
      }

      constexpr index_type
      required_span_size() const noexcept
      {
        auto s = nested_mapping_.required_span_size();
        return s <= 1 ? s : s + 1;
      }

      static constexpr bool
      is_always_unique() noexcept
      {
        if constexpr (not nested_mapping_type::is_always_unique()) return false;
        else if constexpr (extents_type::rank() == 0) return true;
        else if constexpr (extents_type::rank() == 1) return extents_type::static_extent(0) == 1;
        else return extents_type::static_extent(0) == 1 and extents_type::static_extent(1) == 1;
      }

      static constexpr bool
      is_always_exhaustive() noexcept { return nested_mapping_type::is_always_exhaustive(); }

      static constexpr bool
      is_always_strided() noexcept { return false; }

      constexpr bool
      is_unique() const
      {
        if (not nested_mapping_.is_unique()) return false;
        if constexpr (extents_type::rank() == 0) return true;
        else if constexpr (extents_type::rank() == 1) return extents_.extent(0) <= 2;
        else return extents_.extent(0) <= 2 and extents_.extent(1) <= 2 and
          (extents_.extent(0) <= 1 or extents_.extent(1) <= 1);
      }

      constexpr bool
      is_exhaustive() const { return nested_mapping_.is_exhaustive(); }

      constexpr bool
      is_strided() const { return false; }

      constexpr index_type
      stride(std::size_t r) const
      {
        assert(false);
        return 0;
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

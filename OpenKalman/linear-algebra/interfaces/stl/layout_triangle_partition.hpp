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
 * \brief mdspan layout that partitions an object into lower triangle, diagonal, and upper triangle parts
 */

#ifndef OPENKALMAN_LAYOUT_TRIANGLE_PARTITION_HPP
#define OPENKALMAN_LAYOUT_TRIANGLE_PARTITION_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/enumerations.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief A layout policy that splits an object into three partitions associated with the triangular parts.
   * Each of the three partitions has size required_span_size().
   * \tparam NestedLayout A nested layout
   * \tparam tri Indicates how the triangles are partitioned:
   * - triangle_type::diagonal: the diagonal is in the first partition and the remainder is in the third partition.
   * The second partition is empty.
   * - triangle_type::lower: the lower off-diagonal triangle is in the first partition, the diagonal is in
   * the second partition, and the upper off-diagonal triangle is in the third partition, transposed.
   * - triangle_type::upper: the upper off-diagonal triangle is in the first partition, the diagonal is in
   * the second partition, and the lower off-diagonal triangle is in the third partition, transposed.
 */
  template<typename NestedLayout, triangle_type tri>
#ifdef __cpp_concepts
    requires (tri != triangle_type::none)
#endif
  struct layout_triangle_partition
  {
    template<class Extents>
    struct mapping
    {
      using extents_type = Extents;
      using index_type = typename extents_type::index_type;
      using size_type = typename extents_type::size_type;
      using rank_type = typename extents_type::rank_type;
      using layout_type = layout_triangle_partition;

    private:

      using nested_mapping_type = typename NestedLayout::template mapping<extents_type>;

    public:

      constexpr explicit
      mapping(const nested_mapping_type& m) : nested_mapping_(m) {}


      constexpr const extents_type&
      extents() const noexcept { return nested_mapping_.extents(); }


#ifdef __cpp_concepts
      constexpr index_type
      operator() () const requires (extents_type::rank() == 0)
#else
      template<bool Enable = true, std::enable_if_t<Enable and (extents_type::rank() == 0), int> = 0>
      constexpr index_type
      operator() () const
#endif
      {
        return nested_mapping_();
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
        if (i0 == 0) return nested_mapping_(i0);
        if constexpr (tri == triangle_type::lower)
          return nested_mapping_(i0) + nested_mapping_.required_span_size();
        else
          return nested_mapping_(i0) + 2 * nested_mapping_.required_span_size();
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
        if (i0 == i1) return nested_mapping_(i0, i0, is...); // first partition
        if constexpr (tri == triangle_type::lower)
        {
          if (i0 > i1) return nested_mapping_(i0, i1, is...) + nested_mapping_.required_span_size(); // second partition
          return nested_mapping_(i1, i0, is...) + 2 * nested_mapping_.required_span_size(); // third partition
        }
        else if constexpr (tri == triangle_type::upper)
        {
          if (i0 < i1) return nested_mapping_(i0, i1, is...) + nested_mapping_.required_span_size(); // second partition
          return nested_mapping_(i1, i0, is...) + 2 * nested_mapping_.required_span_size(); // third partition
        }
        else // if constexpr (tri == triangle_type::diagonal)
        {
          return nested_mapping_(i0, i1, is...) + 2 * nested_mapping_.required_span_size(); // third partition
        }
      }


      constexpr index_type
      required_span_size() const noexcept
      {
        return 3 * nested_mapping_.required_span_size();
      }


      static constexpr bool
      is_always_unique() noexcept
      {
        return nested_mapping_type::is_always_unique();
      }


      static constexpr bool
      is_always_exhaustive() noexcept
      {
        return nested_mapping_type::is_always_exhaustive() and
          extents_type::static_extent(0) <= 1 and extents_type::static_extent(1) <= 1;
      }


      static constexpr bool
      is_always_strided() noexcept
      {
        return nested_mapping_type::is_always_strided();
      }


      constexpr bool
      is_unique() const
      {
        return nested_mapping_.is_unique();
      }


      constexpr bool
      is_exhaustive() const
      {
        return nested_mapping_.is_exhaustive() and
          nested_mapping_.extent(0) <= 1 and nested_mapping_.extent(1) <= 1;
      }


      constexpr bool
      is_strided() const
      {
        return nested_mapping_.is_strided();
      }


      constexpr index_type
      stride(std::size_t r) const
      {
        return nested_mapping_.stride(r);
      }


      template<class OtherExtents>
      friend constexpr bool
      operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
      {
        return patterns::compare_pattern_collections(lhs.extents(), rhs.extents());
      }

    private:

      nested_mapping_type nested_mapping_;

    };
  };


}

#endif

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
 * \internal
 * \brief mdspan policies for taking a generalized transpose
 */

#ifndef OPENKALMAN_TRANSPOSE_MDSPAN_POLICIES_HPP
#define OPENKALMAN_TRANSPOSE_MDSPAN_POLICIES_HPP

#include "basics/basics.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief A layout policy that returns index 0 for every set of indices.
   */
  template<typename NestedLayout, std::size_t indexa, std::size_t indexb>
  struct layout_transpose
  {
    template<class E>
    struct mapping {
      using extents_type = E;
      using index_type = typename extents_type::index_type;
      using size_type = typename extents_type::size_type;
      using rank_type = typename extents_type::rank_type;
      using layout_type = layout_transpose;

    private:

      template<typename = std::make_index_sequence<extents_type::rank()>>
      struct transposed_extents {};

      template<std::size_t...i>
      struct transposed_extents<std::index_sequence<i...>>
      {
        using type = stdex::extents<index_type,
          extents_type::static_extent(i == indexa ? indexb : i == indexb ? indexa : i)...>;
      };

      using transposed_extents_t = typename transposed_extents<extents_type>::type;

      template<std::size_t...i>
      constexpr transposed_extents_t
      transpose_extents(const stdex::extents<index_type, i...>& e)
      {
        return {e.extent(i == indexa ? indexb : i == indexb ? indexa : i)...};
      }

      using nested_mapping_type = typename NestedLayout::template mapping<transposed_extents_t>;

      template<typename IndexTuple, index_type...i>
      index_type
      transpose_indices(IndexTuple index_tuple, std::index_sequence<i...>) const
      {
        return nested_mapping_(std::get<i == indexa ? indexb : i == indexb ? indexa : i>(std::move(index_tuple))...);
      }

    public:

      constexpr explicit
      mapping(const nested_mapping_type& map) : nested_mapping_(map), extents_(transpose_extents(map.extents())) {}

      constexpr const extents_type&
      extents() const noexcept { return extents_; }

#ifdef __cpp_concepts
      template<std::convertible_to<index_type>...IndexTypes> requires (sizeof...(IndexTypes) == extents_type::rank())
#else
      template<typename...IndexTypes, std::enable_if_t<
        (... and std::is_convertible_v<IndexTypes, index_type>) and
        (sizeof...(IndexTypes) == extents_type::rank()), int> = 0>
#endif
      index_type
      operator() (IndexTypes...i) const
      {
        return transpose_indices(std::forward_as_tuple(std::move(i)...), std::index_sequence_for<IndexTypes...>{});
      }

      constexpr index_type
      required_span_size() const noexcept(noexcept(nested_mapping_.required_span_size()))
      {
        return nested_mapping_.required_span_size();
      }

      const nested_mapping_type&
      nested_mapping() const { return nested_mapping_; }

      static constexpr bool
      is_always_unique() noexcept { return nested_mapping_type::is_always_unique(); }

      static constexpr bool
      is_always_exhaustive() noexcept { return nested_mapping_type::is_always_contiguous(); }

      static constexpr bool
      is_always_strided() noexcept { return nested_mapping_type::is_always_strided(); }

      constexpr bool
      is_unique() const { return nested_mapping_.is_unique(); }

      constexpr bool
      is_exhaustive() const { return nested_mapping_.is_exhaustive(); }

      constexpr bool
      is_strided() const { return nested_mapping_.is_strided(); }

      constexpr index_type
      stride(size_t r) const
      {
        assert(this->is_strided());
        assert(r < extents_type::rank());
        return nested_mapping_.stride(r == indexa ? indexb : r == indexb ? indexa : r);
      }

      template<class OtherExtents>
      friend constexpr bool
      operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
      {
        return lhs.nested_mapping_ == rhs.nested_mapping_;
      }

    private:

      nested_mapping_type nested_mapping_;
      extents_type extents_;

    };
  };


}

#endif

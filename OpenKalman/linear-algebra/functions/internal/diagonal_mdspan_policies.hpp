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
 * \brief mdspan policies for \ref diagonal_matrix
 */

#ifndef OPENKALMAN_DIAGONAL_MDSPAN_POLICIES_HPP
#define OPENKALMAN_DIAGONAL_MDSPAN_POLICIES_HPP

#include "basics/basics.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief A layout policy that returns the index of a diagonal element or a dummy index representing the 0 elements.
   */
  struct layout_diagoanl
  {
    template<class Extents>
    struct mapping
    {
      using extents_type = Extents;
      using index_type = typename extents_type::index_type;
      using size_type = typename extents_type::size_type;
      using rank_type = typename extents_type::rank_type;
      using layout_type = layout_diagoanl;

      constexpr explicit
      mapping(const extents_type& e) : extents_(e) {}

      constexpr const extents_type&
      extents() const noexcept { return extents_; }

      constexpr index_type
#ifdef __cpp_concepts
      operator() (std::convertible_to<index_type> auto...i) const
#else
      template<typename...IndexTypes, std::enable_if_t<
        (... and std::is_convertible_v<IndexTypes, index_type>), int> = 0>
      operator() (IndexTypes...i) const
#endif
      {
        return 0;
      }

      constexpr index_type
      required_span_size() const noexcept { return 0; }

      static constexpr bool is_always_unique() noexcept { return false; }
      static constexpr bool is_always_exhaustive() noexcept { return false; }
      static constexpr bool is_always_strided() noexcept { return false; }

      constexpr bool is_unique() const { return false; }
      constexpr bool is_exhaustive() const { return false; }
      constexpr bool is_strided() const { return false; }

      constexpr index_type
      stride(std::size_t) const { return 0; }

      template<class OtherExtents>
      friend constexpr bool
      operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
      {
        return lhs.extents() == rhs.extents();
      }

    private:

      extents_type extents_;

    };
  };


  /**
   * \internal
   * \brief An accessor that returns a diagonal element or 0.
   * \details The accessor returns 0 if the index is one greater than the maximum index of the diagonal.
   */
  template<typename ElementType>
  struct accessor_diagonal {
    using offset_policy = accessor_diagonal;
    using element_type = ElementType;
    using reference = const element_type&;
    using data_handle_type = const element_type*;

    constexpr accessor_diagonal() noexcept = default;

#ifdef __cpp_concepts
    template<stdex::convertible_to<element_type> OtherElementType>
#else
    template<typename OtherElementType, std::enable_if_t<stdex::convertible_to<OtherElementType, element_type>, int> = 0>
#endif
    constexpr accessor_diagonal(const accessor_diagonal<OtherElementType>& other) noexcept
      : element_ {other.element_} {}

#ifdef __cpp_concepts
    template<stdex::convertible_to<element_type> OtherElementType>
#else
    template<typename OtherElementType, std::enable_if_t<stdex::convertible_to<OtherElementType, element_type>, int> = 0>
#endif
    constexpr accessor_diagonal(accessor_diagonal<OtherElementType>&& other) noexcept
      : element_ {std::move(other).element_} {}

    constexpr accessor_diagonal(ElementType e) : element_ {std::move(e)} {}

    constexpr reference
    access(data_handle_type p, std::size_t i) const noexcept { return element_; }

    constexpr data_handle_type
    offset(data_handle_type p, std::size_t i) const noexcept { return &element_; }

    constexpr data_handle_type
    data_handle() const noexcept { return &element_; }

  private:

    nested_mapping_type nested_mapping_;
    const ElementType element_;

    template<typename OtherElementType>
    friend struct accessor_diagonal;

  };


}

#endif

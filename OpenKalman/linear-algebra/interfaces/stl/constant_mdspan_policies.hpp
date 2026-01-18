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
 * \brief mdspan policies for \ref constant_object
 */

#ifndef OPENKALMAN_CONSTANT_MDSPAN_POLICIES_HPP
#define OPENKALMAN_CONSTANT_MDSPAN_POLICIES_HPP

#include "basics/basics.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief A layout policy that returns index 0 for every set of indices.
   */
  struct layout_constant
  {
    template<class Extents>
    struct mapping
    {
      using extents_type = Extents;
      using index_type = typename extents_type::index_type;
      using size_type = typename extents_type::size_type;
      using rank_type = typename extents_type::rank_type;
      using layout_type = layout_constant;

      constexpr explicit
      mapping(const extents_type& e) : extents_(e) {}

      constexpr const extents_type&
      extents() const noexcept { return extents_; }

#ifdef __cpp_concepts
      constexpr index_type
      operator() (std::convertible_to<index_type> auto...i) const
#else
      template<typename...IndexTypes, std::enable_if_t<
        (... and std::is_convertible_v<IndexTypes, index_type>), int> = 0>
      constexpr index_type
      operator() (IndexTypes...i) const
#endif
      {
        return 0;
      }

      constexpr index_type
      required_span_size() const noexcept
      {
        for(unsigned r = 0; r < extents_type::rank(); r++) if (extents().extent(r) == 0) return 0;
        return 1;
      }

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
   * \brief An accessor that returns a constant value for every index.
   */
  template<typename ElementType>
  struct constant_accessor {
    using element_type = ElementType;
    using reference = const element_type;
    using data_handle_type = element_type;
    using offset_policy = constant_accessor;

    constexpr constant_accessor() noexcept = default;

#ifdef __cpp_concepts
    template<stdex::convertible_to<element_type> OtherElementType> requires
      (not std::is_same_v<element_type, OtherElementType>)
#else
    template<typename OtherElementType, std::enable_if_t<
      stdex::convertible_to<OtherElementType, element_type> and
      (not std::is_same_v<element_type, OtherElementType>), int> = 0>
#endif
    constexpr constant_accessor(const constant_accessor<OtherElementType>& other) noexcept {}

    constexpr reference
    access(data_handle_type p, std::size_t) const noexcept { return std::move(p); }

    constexpr data_handle_type
    offset(data_handle_type p, std::size_t) const noexcept { return std::move(p); }

  };


}

#endif

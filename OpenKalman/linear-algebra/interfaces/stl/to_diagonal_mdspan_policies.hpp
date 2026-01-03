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

#ifndef OPENKALMAN_TO_DIAGONAL_MDSPAN_POLICIES_HPP
#define OPENKALMAN_TO_DIAGONAL_MDSPAN_POLICIES_HPP

#include "patterns/patterns.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief A layout policy that returns the index of a diagonal element or std::dynamic_extent representing the 0 elements.
   */
  template<typename NestedLayout>
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

      constexpr std::size_t
      diag_extent_0() noexcept
      {
        if constexpr (extents_type::rank() == 0) return 0;
        else if constexpr (extents_type::rank() == 1) return std::min(extents_.extent(0), 1);
        else return std::min(extents_type::static_extent(0), extents_.extent(1));
      }

      using nested_mapping_type = typename NestedLayout::template mapping<std::decay_t<decltype(
        patterns::to_extents(patterns::pattern_collection_of_diagonal(std::declval<extents_type>())))>>;

    public:

      constexpr explicit
      mapping(const nested_mapping_type& map, const extents_type& e)
        : nested_mapping_(map),
          extents_(e) {}

      constexpr const extents_type&
      extents() const noexcept { return extents_; }

      constexpr index_type
#ifdef __cpp_concepts
      operator() () const requires (extents_type::rank() == 0)
#else
      template<bool Enable = true, std::enable_if_t<Enable and (extents_type::rank() == 0), int> = 0>
      operator() () const
#endif
      {
        return nested_mapping_(0);
      }

      constexpr index_type
#ifdef __cpp_concepts
      operator() (std::convertible_to<index_type> auto i0) const requires (extents_type::rank() == 1)
#else
      template<typename IndexType0, std::enable_if_t<
        std::is_convertible_v<IndexType0, index_type> and
        (extents_type::rank() == 1), int> = 0>
      operator() (IndexType0 i0) const
#endif
      {
        if (i0 == 0) return nested_mapping_(0);
        else return nested_mapping_.required_span_size();
      }

      constexpr index_type
#ifdef __cpp_concepts
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
      operator() (IndexType0 i0, IndexType i1, IndexTypes...is) const
#endif
      {
        if (i0 == i1) return nested_mapping_(i1, is...);
        else return nested_mapping_.required_span_size();
      }

      constexpr index_type
      required_span_size() const noexcept { return nested_mapping_.required_span_size() + 1; }

      static constexpr bool is_always_unique() noexcept
      {
        if constexpr (extents_type::rank() == 0) return true;
        else if constexpr (extents_type::rank() == 1) return extents_type::static_extent(0) == 1;
        else return extents_type::static_extent(0) == 1 and extents_type::static_extent(1) == 1;
      }

      static constexpr bool is_always_exhaustive() noexcept { return nested_mapping_type::is_always_exhaustive(); }

      static constexpr bool is_always_strided() noexcept { return false; }

      constexpr bool is_unique() const
      {
        if constexpr (extents_type::rank() == 0) return true;
        else if constexpr (extents_type::rank() == 1) return extents_.extent(0) == 1;
        else return extents_.extent(0) == 1 and extents_.extent(1) == 1;
      }

      constexpr bool is_exhaustive() const { return nested_mapping_type::is_exhaustive(); }

      constexpr bool is_strided() const { return false; }

      constexpr index_type
      stride(std::size_t r) const
      {
        assert(false);
        assert(r < extents_type::rank());
        if (r < 2 and extents_.extent(r) != diag_extent_0()) return 0;
        else return nested_mapping_.stride(r - 1);
      }

      template<class OtherExtents>
      friend constexpr bool
      operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
      {
        return lhs.extents() == rhs.extents();
      }

    private:

      nested_mapping_type nested_mapping_;
      extents_type extents_;

    };
  };


  /**
   * \internal
   * \brief An accessor that returns a diagonal element or 0.
   * \details The accessor returns 0 if the index is a special index designated for as the off-diagonal index.
   */
  template<typename NestedAccessor>
  struct to_diagonal_accessor {
  private:

    using nested_element_type = typename NestedAccessor::element_type;

  public:

    using element_type = std::add_const_t<values::value_type_of_t<nested_element_type>>;

    using reference = std::add_lvalue_reference_t<element_type>;
    static_assert(stdex::convertible_to<typename NestedAccessor::reference, reference>);

    using data_handle_type = std::add_pointer_t<element_type>;
    static_assert(stdex::convertible_to<values::value_type_of_t<typename NestedAccessor::data_handle_type>, data_handle_type>);

    using offset_policy = to_diagonal_accessor<typename NestedAccessor::offset_policy>;

    /**
     * \brief
     * \param acc The nested accessor
     * \param off_diagonal_index The index of the off-diagonal element (which should be the maximum of the accessible range + 1).
     */
    constexpr
    to_diagonal_accessor(NestedAccessor acc, std::size_t off_diagonal_index)
      : nested_accessor_(std::move(acc)), off_diagonal_index_(off_diagonal_index) {}

    to_diagonal_accessor() = delete;

#ifdef __cpp_concepts
    template<stdex::convertible_to<NestedAccessor> OtherNestedAccessor> requires
      (not std::is_same_v<NestedAccessor, OtherNestedAccessor>)
#else
    template<typename OtherNestedAccessor, std::enable_if_t<
      stdex::convertible_to<OtherNestedAccessor, NestedAccessor> and
      (not std::is_same_v<NestedAccessor, OtherNestedAccessor>), int> = 0>
#endif
    constexpr to_diagonal_accessor(const to_diagonal_accessor<OtherNestedAccessor>& other) noexcept
      : nested_accessor_ {other.element_}, off_diagonal_index_(other.off_diagonal_index_) {}

#ifdef __cpp_concepts
    template<stdex::convertible_to<NestedAccessor> OtherNestedAccessor> requires
      (not std::is_same_v<NestedAccessor, OtherNestedAccessor>)
#else
    template<typename OtherNestedAccessor, std::enable_if_t<
      stdex::convertible_to<OtherNestedAccessor, NestedAccessor> and
      (not std::is_same_v<NestedAccessor, OtherNestedAccessor>), int> = 0>
#endif
    constexpr to_diagonal_accessor(to_diagonal_accessor<OtherNestedAccessor>&& other) noexcept
      : nested_accessor_ {std::move(other).element_}, off_diagonal_index_(std::move(other).off_diagonal_index_) {}

  private:

    static constexpr data_handle_type
    get_zero()
    {
      static element_type z(0);
      return &z;
    }

  public:

    constexpr reference
    access(data_handle_type p, std::size_t i) const noexcept
    {
      if (i == off_diagonal_index_) return *get_zero();
      return nested_accessor_.access(p, i);
    }

    constexpr data_handle_type
    offset(data_handle_type p, std::size_t i) const noexcept
    {
      if (i == off_diagonal_index_) return get_zero();
      return values::to_value_type(nested_accessor_.offset(p, i));
    }

    const NestedAccessor&
    nested_accessor() const noexcept { return nested_accessor_; }

  private:

    NestedAccessor nested_accessor_;

    std::size_t off_diagonal_index_;

  };


}

#endif
